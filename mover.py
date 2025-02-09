# Imports
import os
import math
import time

import cv2
import torch
import numpy as np
from PIL import Image
from reset import compute_iou
from remote_cam import get_image, get_depth
import matplotlib
from matplotlib.widgets import Button
from matplotlib import pyplot as plt
from gaze_utils.pc_utils import depth2pc
from gaze_utils.sshlib import get_cograsp, get_glip
from scipy.spatial.transform import Rotation as R
from gaze_utils.constants import cam_tool_offset, camera_tweaks, front, d415_intrinsics

from gaze_utils.pc_utils import view_pc_o3d

import rtde_control
import rtde_receive
import robotiq_gripper

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


def rv2rpy(rx,ry,rz):
  theta = math.sqrt(rx*rx + ry*ry + rz*rz)
  kx = rx/theta
  ky = ry/theta
  kz = rz/theta
  cth = math.cos(theta)
  sth = math.sin(theta)
  vth = 1-math.cos(theta)
  
  r11 = kx*kx*vth + cth
  r12 = kx*ky*vth - kz*sth
  r13 = kx*kz*vth + ky*sth
  r21 = kx*ky*vth + kz*sth
  r22 = ky*ky*vth + cth
  r23 = ky*kz*vth - kx*sth
  r31 = kx*kz*vth - ky*sth
  r32 = ky*kz*vth + kx*sth
  r33 = kz*kz*vth + cth
  
  beta = math.atan2(-r31,math.sqrt(r11*r11+r21*r21))
  
  if beta > math.radians(89.99):
    beta = math.radians(89.99)
    alpha = 0
    gamma = math.atan2(r12,r22)
  elif beta < -math.radians(89.99):
    beta = -math.radians(89.99)
    alpha = 0
    gamma = -math.atan2(r12,r22)
  else:
    cb = math.cos(beta)
    alpha = math.atan2(r21/cb,r11/cb)
    gamma = math.atan2(r32/cb,r33/cb)
  
  return gamma, beta, alpha  


def transform(depth, rgb, pose, K):
  '''
  Convert the given depth (in meters) and rgb [0, 256) to a pointcloud, then transform by the given UR5
  pose [x, y, z, rotvec_, rotvec_Y, rotvec_X], as defined in base coordinates. Such a pose is returned
  by functions like getForwardKinematics() or getActualTCPPose from the UR5 api.

  Provide a K, which is the camera intrinsics for converting the depth into pointcloud. The default is for a D435.
  Such a value can be obtained with rs-enumerate -c, and is unique to the camera model and resolution

  Note: Anything farther than 1 meter in the depth image will be filtered out
  '''
  depth[depth > 1] = 0

  # Intrinsics for realsense D435 at 1280x720 (use rs-enumerate-devices -c)
  '''
      [fx  0 cx]
  K = [ 0 fy cy]
      [ 0  0  1]

  PPX:        	649.206298828125
  PPY:        	358.177185058594
  Fx:         	910.571960449219
  Fy:         	911.160827636719
  '''
  
  pc, rgb = depth2pc(depth, K, rgb=rgb)

  # Convert from rotation vector to RPY
  rot = R.from_rotvec(pose[3:6])
  roll, pitch, yaw = rot.as_euler('xyz')
  roll, pitch, yaw = camera_tweaks(roll, pitch, yaw)
  rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

  # Apply rotation to pc
  transformed_pc = rot.apply(pc)

  # Apply translation
  tx, ty, tz = pose[0:3]
  transformed_pc += np.array([tx, ty, tz])
    
  return transformed_pc, rgb

def transform_point(pc, pose):
  '''
  Transform the given [x, y, z] array by the given UR5
  pose [x, y, z, rotvec_, rotvec_Y, rotvec_X], as defined in base coordinates

  Returns array of shape [1, 3]
  '''
  # Convert from rotation vector to RPY
  roll, pitch, yaw = rv2rpy(*pose[3:6])
  roll, pitch, yaw = camera_tweaks(roll, pitch, yaw)
  rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

  # Apply rotation to pc
  transformed_pc = rot.apply(pc)

  # Apply translation
  tx, ty, tz = pose[0:3]
  transformed_pc += np.array([tx, ty, tz]) 

  # print('rpy', math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
  return transformed_pc

def transform_rot(rot_intrin, pose):
  '''
  Transform the given scipy Rotation by the given UR5
  pose [x, y, z, rotvec_, rotvec_Y, rotvec_X], as defined in base coordinates

  Returns rotvec
  '''
  # Convert from rotation vector to RPY
  roll, pitch, yaw = rv2rpy(*pose[3:6])
  roll, pitch, yaw = camera_tweaks(roll, pitch, yaw)
  rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

  # Apply rotation to pc
  transformed_rot = rot * rot_intrin    
  return transformed_rot

rtde_c, rtde_r, gripper, wrist_cam = None, None, None, None


def deg_to_rad(l):
    return list(map(math.radians, l))

def wait_for_file(filename):
    while True:
        file_exists = os.path.exists(filename)
        if file_exists: break
        time.sleep(0.2)

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

def filter_depth_boundaries(depth_image, threshold=0.1):
    """
    Compute the average of the "farther" pixels at boundaries in a depth image where the depth changes significantly,
    using vectorized operations for efficiency.

    Parameters:
    - depth_image (numpy.ndarray): The input depth image.
    - threshold (float): The depth change threshold to identify significant changes.

    Returns:
    - float: The average depth of the "farther" pixels at significant depth boundaries.
    """
    # Calculate the differences between adjacent pixels horizontally and vertically
    diff_h = np.abs(np.diff(depth_image, axis=1))
    diff_v = np.abs(np.diff(depth_image, axis=0))

    # Create masks where the depth change exceeds the threshold
    mask_h = diff_h > threshold
    mask_v = diff_v > threshold

    # For horizontal differences
    # Identify where mask_h is True and compare depth values horizontally
    farther_pixels_h = np.maximum(depth_image[:, :-1][mask_h], depth_image[:, 1:][mask_h])

    # For vertical differences
    # Identify where mask_v is True and compare depth values vertically
    farther_pixels_v = np.maximum(depth_image[:-1, :][mask_v], depth_image[1:, :][mask_v])

    # Combine the farther pixels from both directions
    farther_pixels = np.concatenate((farther_pixels_h, farther_pixels_v))

    # Calculate the average of the "farther" pixels
    if farther_pixels.size > 0:
        average_depth = np.mean(farther_pixels)
    else:
        average_depth = 0  # Handle case with no significant boundaries

    cop = depth_image.copy()
    cop[cop > average_depth + 0.01] = 0
    return cop

def raw_to_pc(front_depth, front_bgr, front_cam_pose, save, filter=True):
    '''
    Convert a RGBD to pointcloud and remove the tabletop points
    '''
    front_depth[front_depth > 1] = 0

    if filter:
      front_depth = filter_depth_boundaries(front_depth, 0.07)

    pc, col_bgr = depth2pc(front_depth, d415_intrinsics, rgb=front_bgr)

    tpc = transform_point(pc, front_cam_pose)
    colhsv = matplotlib.colors.rgb_to_hsv(col_bgr[..., ::-1] / 256)
    valid_ind = (tpc[..., 2] > 0.06) | \
                      ((colhsv[..., 2] > 100/255) & (tpc[..., 2] > 0.04))

    # view_pc_o3d(tpc, col=col_bgr[..., ::-1], show=False, title='Grabbable region')
    
    tpc = tpc[valid_ind]
    col_bgr = col_bgr[valid_ind]
    colhsv = colhsv[valid_ind]

    
    torch.save({'xyz': tpc, 'col':col_bgr[..., ::-1]}, save)

    return tpc, col_bgr[..., ::-1]

def grabbable_region(rgb, depth, partname, objname, largen_part):
    '''
    Return grabbable region from image
    '''
    whitelist_obj = np.zeros(shape=depth.shape, dtype=bool)
    whitelist_part = np.zeros(shape=depth.shape, dtype=bool)

    bbs_obj, ann_obj, score = get_glip(objname, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    # Get heatmap
    wait_for_file('heatmap.pth')
    heatmap = torch.load('heatmap.pth')

    # Merge object regions and heatmap
    intersection_scores = []
    for bb in bbs_obj:
        x1, y1, x2, y2 = bb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        area = ((y2 - y1) * (x2 - x1))
        intersection_scores.append(heatmap[y1:y2, x1:x2].sum() * 1/area)

    # Get best object regions
    bbs_obj = bbs_obj[np.argmax(intersection_scores)]

    # Get object region
    x1, y1, x2, y2 = bbs_obj
    x1, y1, x2, y2 = int(x1) - 10, int(y1) - 20, int(x2) + 10, int(y2) + 20
    whitelist_obj[y1:y2, x1:x2] = True
    
    # Get part region
    if partname is not None:
        bbs_part, ann_part, scores = get_glip(partname, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)) # rgb[y1:y2, x1:x2]
        for bb in bbs_part:
            x1, y1, x2, y2 = bb

            if compute_iou(bb, bbs_obj) > 0.8:
              continue

            if largen_part:
              x1, y1, x2, y2 = int(x1) - 10, int(y1) - 10, int(x2) + 10, int(y2) + 10
            else:
              x1, y1, x2, y2 = int(x1) + 10, int(y1) + 10, int(x2) - 10, int(y2) - 10

            whitelist_part[y1:y2, x1:x2] = True
    else:
        whitelist_part = True

    whitelist_part = whitelist_obj & whitelist_part
    return whitelist_obj, whitelist_part


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

        img, dep = get_image(), get_depth()
        front_rgb = np.rot90(img, 2)
        torch.save(front_rgb, 'capture_pointcloud/front_rgb')
        Image.fromarray(front_rgb[..., ::-1]).save('capture_pointcloud/front_rgb.png')
        front_depth = np.rot90(dep, 2)
        torch.save((front_cam_pose, front_rgb, front_depth), 'capture_pointcloud/front_data')

        return front_cam_pose, front_rgb, front_depth

def moveit(objname, partname, target_holder, save_pc=None):
    '''
    Moves the UR5 (with wrist mounted depth camera) to different poses in order to construct a more complete pointcloud
    than one perspective would provide
    '''
    global rtde_c, rtde_r, gripper

    try: os.remove('arm_target.txt') 
    except FileNotFoundError: pass
    try: os.remove('robot_grasp_viz.png')
    except FileNotFoundError: pass
    
    wait_for_file('capture_pointcloud/front_data')
    front_cam_pose, front_bgr, front_depth = torch.load('capture_pointcloud/front_data')

    # clip depth
    front_depth[front_depth > 1] = 0
    whitelist_obj, whitelist_part = grabbable_region(front_bgr, front_depth, partname, objname, 
                                                     largen_part = target_holder == 'robot')

    '''
    Three cases:

    1 Just object specified: select farthest robot grasp from hand candidates
    2 Object and part, with robot grasp part: do the grasps in OBJECT A PART
    3 Object and part, with robot grasp part: do the grasps in OBJECT A ~PART
    '''

    if partname is None:
      grab_mask = whitelist_obj.copy()
    else:
      if target_holder == 'robot':
        grab_mask = whitelist_obj.copy()
        grab_mask[~(whitelist_obj & whitelist_part)] = 0.0
      elif target_holder == 'human':
        # If the part BB fails, then it will be the same size as object box. This is a problem because it makes the
        # rest of the region small
        grab_mask = whitelist_obj.copy()
        grab_mask[~(whitelist_obj & ~whitelist_part)] = 0.0

    depth_obj = front_depth.copy()
    depth_obj[~grab_mask] = 0.0

    # Python imshow 
    f, axarr = plt.subplots(1, 2) 
    f.set_size_inches(13, 5)
    plt.title(f'Detected Object {objname} and Grabable Part {partname}')

    button_ax = plt.axes([0.4, 0.05, 0.3, 0.1]) 
    button = Button(button_ax, 'Continue', color='gainsboro', hovercolor='white')
    button.on_clicked(lambda event: plt.close())

    bgr = front_bgr[..., ::-1]
    a = bgr.copy()
    b = bgr.copy()
    a[~whitelist_obj] = a[~whitelist_obj] / 2
    b[~grab_mask] = b[~grab_mask] / 2
    axarr[0].imshow(a)
    if partname is not None:
      axarr[1].imshow(b)
    plt.show()

    if True:
      dest = f'dataset/scenes/raw_pc_{objname}.pth'
      xyz, col = raw_to_pc(depth_obj, front_bgr, front_cam_pose, dest)

      avoid_hands = partname is None
      best_grasp_mat = get_cograsp(xyz, col, avoid_hands)

    # Get best grasp
    
    # pc_full, pred_grasps_cam, scores, contact_pts, pc_colors, hand_pcs, hand_cols = \
    #           get_grasp(front_bgr, front_depth, 
    #                     mask=grab_mask, 
    #                     avoid_hands=avoid_hands # Case 1, avoid hands
    #                     )

    # best_grasp, best_grasp_rot = best_grasp(pc_full, pred_grasps_cam, scores, contact_pts, pc_colors, hand_pcs, hand_cols, avoid_hands)
    # best_grasp_rot = transform_rot(best_grasp_rot, front_cam_pose)
    # flip = Rotation.from_euler('Z', 180, degrees=True)
    # # best_grasp_rot = best_grasp_rot * flip
    # tool_vec = best_grasp_rot.apply([0, 0, 0.06])
    # # print('tool vector direction', tool_vec)
    # best_grasp = transform_point(best_grasp, front_cam_pose).squeeze()

    best_grasp_rot = R.from_matrix(best_grasp_mat[0:3, 0:3])
    best_grasp = np.array([best_grasp_mat[0:3, 3]])
    tool_vec = best_grasp_rot.apply([0, 0, 0.06])

    best_grasp = list(best_grasp.reshape(-1))

    def nicelist(li):
      return ', '.join([f'{l.item():.3f}' for l in li])
    
    print('Best robot grasp pose', nicelist(best_grasp) + ', ' + nicelist(best_grasp_rot.as_rotvec()))

    # Go to staging
    height = 0.5
    stage = list(best_grasp - 2 * tool_vec) + list(best_grasp_rot.as_rotvec())
    final = list(best_grasp + -height * tool_vec) + list(best_grasp_rot.as_rotvec())

    step = 0
    while final[2] < 0.1:
      step -= 0.1
      final = list(best_grasp + (-height + step) * tool_vec) + list(best_grasp_rot.as_rotvec())
      print('Gripper intersecting table, reduced to', -height + step, 'at', (best_grasp + (-height + step))[2])

    with open('arm_target.txt', 'w') as file:
      file.write(str(stage) + '\n' + str(final))
      print('arm_target.txt saved')

    # goto(rtde_c, rtde_r, gripper, grasp=True, handover=True)

    # print('Execute Path')
    # result = subprocess.run('/home/corallab/anaconda3/envs/handover/bin/python goto.py', shell=True, capture_output=True, text=True)
    # print(result)

    plt.show()
    

if __name__ == '__main__':
  from mover import moveit, init as initArm
  initArm(True)
  moveit(objname='bowl', partname=None, target_holder='robot', save_pc=True)