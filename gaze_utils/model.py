import torch
import torch.nn as nn
import math
from .resnet import resnet18


class GazeLSTM(nn.Module):
    def __init__(self):
        super(GazeLSTM, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(2*self.img_feature_dim, 3)


    def forward(self, input):

        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))

        base_out = base_out.view(input.size(0),7,self.img_feature_dim)

        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        output = self.last_layer(lstm_out).view(-1,3)


        angular_output = output[:,:2]
        angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])

        var = math.pi*nn.Sigmoid()(output[:,2:3])
        var = var.view(-1,1).expand(var.size(0),2)

        return angular_output,var
class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1-self.q1

    def forward(self, output_o,target_o,var_o):
        q_10 = target_o-(output_o-var_o)
        q_90 = target_o-(output_o+var_o)

        loss_10 = torch.max(self.q1*q_10, (self.q1-1)*q_10)
        loss_90 = torch.max(self.q9*q_90, (self.q9-1)*q_90)


        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90


model = GazeLSTM()
model = torch.nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()
    gaze_device = torch.device('cuda')
else:
    import sys
    print('Gaze model using CPU!', file=sys.stderr)
    gaze_device = torch.device('cpu')

checkpoint = torch.load('gaze360_model.pth.tar', map_location=gaze_device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()