import gaze_utils.realsense as rs
from gaze_utils.headpose import infer
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import sys

# Giving the linear transform some ad-hoc nonlinearity
# (projecting to sqrt(x) before linear) drops the loss from 30 mm to 18 mm
def transform(w, b, x):
    if type(x) != torch.Tensor:
        x = torch.Tensor(x)

    # Need dim >= 2
    if len(x.size()) == 1:
        x = x[None, :]

    x = x.T

    sign = torch.sign(x)
    x = torch.abs(x)

    x = torch.stack((
        torch.sqrt(x[0]),
        torch.sqrt(x[1]),
        x[2]
    ))

    proj = torch.mm(w, x) + b

    square = torch.stack((
                proj[0] ** 2,
                proj[1] ** 2,
                proj[2] ** 1
            )) * sign

    if square.T.size()[0] == 1:
        return square.T[0]
    return square.T

# def transform(w, b, x):
#     return (torch.mm(w, x) + b)

# Fitting
def best_transform(src, truth, step_size=0.3):
    ''' fit src, a (M, N) list
        to truth, a (M, N) list
        where M is the number of samples and N is the size of
        a single data vector'''
    src = torch.Tensor(src)
    truth = torch.Tensor(truth)

    product = torch.tensor([[1, 0, 0], 
                            [0, 1, 0],
                            [0, 0, 1]], 
                        dtype=torch.float32, requires_grad=True)
    bias = torch.tensor([[0], [0], [0]], 
                        dtype=torch.float32, requires_grad=True)

    for i in range(50):
        loss = torch.mean((truth - transform(product, bias, src)) ** 2)
        loss.backward()
        if i == 0:
            starting_loss = loss.item()

        # Gradient Descent
        product.data -= product.grad.data * step_size
        bias.data -= bias.grad.data * step_size

        # Reset
        bias.grad.data.zero_()
        product.grad.data.zero_()

    # Avg error per axis
    # print(f'x: {torch.mean(delta[:, 0]):.9f} x: {torch.mean(delta[:, 1]):.9f} x: {torch.mean(delta[:, 2]):.9f}')

    print(f'Starting loss {starting_loss:.4f} Final Loss: {loss.item():.4f}')

    return product.detach(), bias.detach()

# import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, insize, hiddensize, outsize):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(insize, hiddensize),
#             nn.Tanh(),
#             nn.Linear(hiddensize, outsize)
#         )

#     def forward(self, x):
#         return self.layers(x).reshape(x.size())

# def best_transform(src, truth, lr=0.0001):
#     mlp = MLP(3, 15, 3)
#     loss_function = nn.MSELoss()
#     optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

#     starting_loss = loss_function(src, truth)

#     for i in range(20):
#         optimizer.zero_grad()
#         outputs = mlp(src)
#         loss = loss_function(outputs, truth)
#         loss.backward()
#         optimizer.step()
#         print('loss:', i, loss.item())

#     return mlp

if __name__ == '__main__':
    GREEN = '\033[92m'
    RESET = '\033[0m'

    print(f'{GREEN}This calibrates the head pose estimation for gaze tracking using depth data fom a realsense series camera')
    print(f'May have to modify the RSCapture class if you wish to not read from bag file, and skip_frame variable')
    print(f'Move your face around in (x y z) space for the range of poses you want to calibrate for')
    print(f'You will see an annotated RGB stream with bboxes, the estimated (blue) head pose, and the true (green) head pose')
    print(f'When done, press Ctrl + C once and this will save params to gaze_cam_calib.yaml in the working directory {RESET}')

    pred_poses = []
    true_poses = []

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Realsense Camera
    v_reader = rs.RSCapture('calibration.bag')
    # v_reader = rs.RSCapture()

    w = None
    b = None
    try:
        skip_frame = 1
        for i, frame in enumerate(v_reader):
            if i % skip_frame != 0:
                continue

            ax.clear()
            ax.set_xlim(-0.5, 0.5) # x
            ax.set_ylim(0, 1) # z
            ax.set_zlim(-0.5, 0.5) # y

            ax.set_title('People pose and gazes (Pred is Blue, Truth is Green)')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('y')

            # Get images
            color_image, depth_image, depth_frame, color_frame = frame
            # Init Canvas
            drawable = cv2.cvtColor(color_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            
            # Predict
            poses, bboxes = infer(color_image)

            for pose, box in zip(poses, bboxes):
                # Integerify Box
                left, top, right, bottom = box
                left = int(left)
                top = int(top)
                right = int(right)
                bottom = int(bottom)
                box = (left, top, right, bottom)

                pose = pose[3:6]

                # Ground Truth
                true_pose = v_reader.pose_of_bb(box, depth_frame)
                # print("pred: ", pose, '\n', "true: ", true_pose)

                # Store
                # pred_poses.append(np.asarray(pose) / np.array([10., 40., 40.]))

                if np.isnan(np.sum(pose)) or np.isnan(np.sum(true_pose)):
                    continue
                
                pred_poses.append(pose.tolist())
                true_poses.append(true_pose)

                # Do fitting
                if len(pred_poses) > 5:
                    w, b = best_transform(pred_poses, true_poses)
                    with torch.no_grad():
                        pred_poses_fit = transform(w, b, pred_poses).tolist()

                    # mlp = best_transform(torch.Tensor(pred_poses), torch.Tensor(true_poses))
                    # with torch.no_grad():
                    #     pred_poses_fit = mlp(torch.Tensor(pred_poses)).tolist()

                else:
                    pred_poses_fit = pred_poses

                # Plot
                pred_coords = list(zip(*pred_poses_fit))
                true_coords = list(zip(*true_poses))
                

                ax.scatter(pred_coords[0], pred_coords[2], pred_coords[1], c='blue')
                ax.scatter(true_coords[0], true_coords[2], true_coords[1], c='green')
                plt.pause(0.001)

                # Draw face bbox
                drawable = cv2.rectangle(drawable, (box[0], box[1]), (box[2], box[3]), (100, 200, 100), thickness=3)
                drawable = cv2.putText(drawable, 
                                    f'Pred: {pred_poses_fit[-1][0]:.{2}f} {pred_poses_fit[-1][1]:.{2}f} {pred_poses_fit[-1][2]:.{2}f}', 
                                    (box[0], box[3]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (255, 0, 0), 
                                    2, cv2.LINE_AA)
                drawable = cv2.putText(drawable, 
                                    f'True: {true_pose[0]:.{2}f} {true_pose[1]:.{2}f} {true_pose[2]:.{2}f}', 
                                    (box[0], box[3] + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 
                                    2, cv2.LINE_AA)

            cv2.imshow("viz", drawable)
            cv2.waitKey(1)

    except KeyboardInterrupt or StopIteration:
        print("Ctrl + C pressed or stream over, saving data to gaze_cam_calib.yaml")
        import yaml
        with open('gaze_cam_calib.yaml', "w") as file:
            try:
                yaml.dump((w.tolist(), b.tolist()), file)
            except yaml.YAMLError as exc:
                print(exc)