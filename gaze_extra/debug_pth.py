import cv2
import matplotlib.pyplot as plt
import torch

path = 'gaze_extra/outputs/flashlight/image_000_left.pth'
path = 'gaze_extra/outputs/flashlight/image_000_right.pth'

img = torch.load(path)

print(img.dtype, img.shape)

plt.imshow(img)
plt.show()