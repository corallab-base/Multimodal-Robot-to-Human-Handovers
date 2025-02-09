import inspect
import subprocess
import time
import cv2
import matplotlib
import torch
import numpy as np
import math
import os
from PIL import Image
from remote_cam import get_image_depth
from gaze_utils import realsense as rs

from goto import goto
import rtde_control
import rtde_receive
import robotiq_gripper
# from .rmp_utils import execute_RMP_path, setup

from matplotlib import pyplot as plt
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from gaze_utils.constants import cam_tool_offset, camera_tweaks, front

def deg_to_rad(l):
    return list(map(math.radians, l))


def get_cam_pose():
    '''
    Get the current pose of the camera lens in UR5 base coordinates
    '''
    # Note: the following statement would be identical to rtde_r.getActualTCPPose()
    # if not for our custom-defined tcp_offset
    fk = rtde_c.getForwardKinematics(rtde_r.getActualQ(), tcp_offset=cam_tool_offset)
    return fk

def capture_wrist_cam():
    global wrist_cam
    col, depth, _, _ = wrist_cam.get_frames(rotate=False, viz=False)
    return col, depth

def init(real_life):

    global rtde_c, rtde_r, gripper, wrist_cam

    ip_address='192.168.1.125'

    if real_life:
        # Delete outputs
        try: os.remove('capture_pointcloud/front_data')
        except OSError: pass
        try: os.remove('capture_pointcloud/right_data')
        except OSError: pass
        try: os.remove('capture_pointcloud/left_data')
        except OSError: pass

    try: os.remove('capture_pointcloud/merged_pc.npz')
    except OSError: pass

    if real_life:
        # wrist_cam = rs.RSCapture(serial_number='044122070299', use_meters=False, preset='Default')

        # Setup for real robot
        rtde_c = rtde_control.RTDEControlInterface(ip_address)
        rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)
        
        gripper = robotiq_gripper.RobotiqGripper()
        gripper.connect(ip_address, 63352)
        gripper.activate(auto_calibrate=False)

        # Open gripper
        gripper.move(gripper.get_open_position(), 64, 1)

        # Go to front position
        rtde_c.moveJ(deg_to_rad(front), 1.5, 0.9, asynchronous=False)
        front_cam_pose = get_cam_pose()
        front_cam_pose = None

        front_rgb, front_depth = get_image_depth()
        front_rgb, front_depth = np.rot90(front_rgb, 2), np.rot90(front_depth, 2)
        torch.save(front_rgb, 'capture_pointcloud/front_rgb')
        Image.fromarray(front_rgb[..., ::-1]).save('capture_pointcloud/front_rgb.png')
        torch.save((front_cam_pose, front_rgb, front_depth), 'capture_pointcloud/front_data')

        return front_cam_pose, front_rgb, front_depth