import math
from matplotlib import pyplot as plt
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from gaze_utils.constants import d415_intrinsics

import numpy as np
import torch
from gaze_utils.realsense import RSCapture
from mover import moveit, init, deg_to_rad, front, depth2pc, cam_tool_offset, transform_point, view_pc, transform
import cv2

import rtde_control
import rtde_receive
import robotiq_gripper

def tobgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

ip_address='192.168.1.123'

rtde_c = rtde_control.RTDEControlInterface(ip_address)
rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

gripper = robotiq_gripper.RobotiqGripper()
gripper.connect(ip_address, 63352)
gripper.activate(auto_calibrate=False)

# Open gripper
gripper.move(gripper.get_open_position(), 64, 1)

# Go to front position
rtde_c.moveJ(deg_to_rad(front), 1.5, 0.9, asynchronous=False)

wrist_cam = RSCapture(serial_number='044122070299', use_meters=False, preset='Default')

rgb, depth, _, _ = wrist_cam.get_frames(rotate=True)
pose = rtde_c.getForwardKinematics(rtde_r.getActualQ(), tcp_offset=cam_tool_offset)
print('pose', pose)
from scipy.spatial.transform import Rotation as R
rot = R.from_rotvec(pose[3:6])
orient = rot.as_euler('xyz', degrees=False)
print('orient', [math.degrees(r) for r in orient])

np.savez('rgb.npz', rgb, depth)

dat = np.load('rgb.npz')
rgb, depth = dat[dat.files[0]], dat[dat.files[1]]
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

print('depth', depth.shape)

hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

blue = (hsv[..., 0] > 60) & (hsv[..., 0] < 95) & (hsv[..., 1] > 100) & (hsv[..., 2] > 120) & (hsv[..., 2] < 230)
blue = blue.astype(bool)
blue[0:blue.shape[0]//2, :] = False
# blue[:, 0:200] = False
# blue[:, 400:] = False


nd = depth.copy()
nd[~blue] = 0

pc, col = depth2pc(nd, d415_intrinsics)
xyz = pc.mean(axis=0)

xyz = transform_point(xyz, pose).squeeze()

print('xyz', xyz)

vpc, vcol = transform(depth, rgb/256, pose, k)
indices = np.random.choice(vpc.shape[0], size=50000, replace=False)
# view_pc(vpc[indices], vcol[indices])

nh = hsv.copy()
nh[~blue] = 0

rgb[~blue] = 0

fig, ax = plt.subplots(1, 4)
ax[0].imshow(hsv)
ax[1].imshow(rgb)
ax[2].imshow(nh)
ax[3].imshow(depth)

fig.set_size_inches(23, 4)
plt.show()

# xyz = [-0.66663341, -0.32239173,  0.20136154] # Left
# xyz = [-0.71831174, -0.16698045,  0.20479876] # Right Just 16cm?

xyz[2] += 0.18
rtde_c.moveL(list(xyz) + [0, 3.1415, 0], 0.3, 0.3, asynchronous=False)



# plt.imshow(hsv)

# plt.show()