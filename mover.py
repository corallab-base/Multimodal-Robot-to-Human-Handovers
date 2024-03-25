import time
from matplotlib import pyplot as plt
import torch
import numpy as np
import math
import os
from PIL import Image

from gaze_utils import realsense as rs

from gaze_utils.pc_utils import view_pc, depth2pc
from gaze_utils.sshlib import get_glip, get_grasp

import rtde_control
import rtde_receive
import robotiq_gripper
# from .rmp_utils import execute_RMP_path, setup


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

def transform(depth, rgb, pose, K = np.array([[910.571960449219, 0, 649.206298828125], [0, 911.160827636719, 358.177185058594], [0, 0, 1]])):
  '''
  Convert the given depth (in meters) and rgb [0, 256) to a pointcloud, then transform by the given UR5
  pose [x, y, z, rotvec_, rotvec_Y, rotvec_X], as defined in base coordinates. Such a pose is returned
  by functions like getForwardKinematics() or getActualTCPPose from the UR5 api.

  Provide a K, which is the camera intrinsics for converting the depth into pointcloud. The default is for a D435.
  Such a value can be obtained with rs-enumerate -c, and is unique to the camera model and resolution

  Note: Anything farther than 1 meter in the depth image will be filtered out
  '''

  from gaze_heatmap_to_mask import depth2pc
  from scipy.spatial.transform import Rotation as R

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

  rgb[:, :] = rgb[:, [2, 1, 0]]

  # Convert from rotation vector to RPY
  roll, pitch, yaw = rv2rpy(*pose[3:6])
  rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

  # Camera tilts back a little, so add 1 deg of intrinsic pitch
  roll_instrin, pitch_instrin, yaw_instrin = rot.as_euler('YXZ', degrees=False)
  pitch_instrin += math.radians(1)
  rot = R.from_euler('YXZ', [roll_instrin, pitch_instrin, yaw_instrin], degrees=False)

  # Apply rotation to pc
  transformed_pc = rot.apply(pc)

  # Apply translation
  tx, ty, tz = pose[0:3]
  transformed_pc += np.array([tx, ty, tz]) 

  # print('rpy', math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
  return transformed_pc, rgb

def transform_point(pc, pose):
  '''
  Transform the given [x, y, z] array by the given UR5
  pose [x, y, z, rotvec_, rotvec_Y, rotvec_X], as defined in base coordinates

  Returns array of shape [1, 3]
  '''

  from scipy.spatial.transform import Rotation as R

  # Convert from rotation vector to RPY
  roll, pitch, yaw = rv2rpy(*pose[3:6])
  rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

  # Camera tilts back a little, so add 1 deg of intrinsic pitch
  roll_instrin, pitch_instrin, yaw_instrin = rot.as_euler('YXZ', degrees=False)
  pitch_instrin += math.radians(1)
  rot = R.from_euler('YXZ', [roll_instrin, pitch_instrin, yaw_instrin], degrees=False)

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

  from scipy.spatial.transform import Rotation as R

  # Convert from rotation vector to RPY
  roll, pitch, yaw = rv2rpy(*pose[3:6])
  rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

  # Camera tilts back a little, so add 1 deg of intrinsic pitch
  # Also roll 180deg instrinsically
  roll_instrin, pitch_instrin, yaw_instrin = rot.as_euler('YXZ', degrees=False)
  # roll_instrin += math.radians(-90)
  # yaw_instrin += math.radians(180)
  rot = R.from_euler('YXZ', [roll_instrin, pitch_instrin, yaw_instrin], degrees=False)

  # Apply rotation to pc
  transformed_rot = rot * rot_intrin

  # print('rpy', math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
  return transformed_rot

rtde_c, rtde_r, gripper, wrist_cam = None, None, None, None

# In camera coordinates: [(-)left-to-right(+), (-)up-to-down(+), (-)back-to-front(+), rotvec ]
cam_tool_offset = [-0.01, -0.075, 0.05, 0, 0, 0]
right = [60.1, -167.45, -62.0, 49.6, 149.6, 0.1]
left = [125.9, -158.6, -91.7, 69.97, 8.71, 0.35]
front = [164.88, -46.17, -124.62, -53.24, 101.27, -10.25]

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

def grabbable_region(rgb, depth, partname, objname):
    '''
    Return grabbable region from image
    '''
    import cv2

    whitelist_obj = np.zeros(shape=depth.shape, dtype=bool)
    whitelist_part = np.zeros(shape=depth.shape, dtype=bool)

    bbs_obj, ann_obj = get_glip(objname, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    # Get heatmap
    wait_for_file('heatmap.pth')
    heatmap = torch.load('heatmap.pth')
    print('loaded heatmap', heatmap.dtype, heatmap.shape)

    # Merge object regions and heatmap
    intersection_scores = []
    for bb in bbs_obj:
        x1, y1, x2, y2 = bb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        intersection_scores.append(heatmap[y1:y2, x1:x2] / ((y2 - y1) * x2 - x1))

    # Get best obejct regions
    bbs_obj = [bbs_obj[np.argmax(intersection_scores)]]

    # Get object region
    for bb in bbs_obj:
        x1, y1, x2, y2 = bb
        x1, y1, x2, y2 = int(x1) - 10, int(y1) - 40, int(x2) + 10, int(y2) + 40
        whitelist_obj[y1:y2, x1:x2] = True
    
    # Get part region
    if partname is not None:
        bbs_part, ann_part = get_glip(partname, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        for bb in bbs_part:
            x1, y1, x2, y2 = bb
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            whitelist_part[y1:y2, x1:x2] = True
    else:
        whitelist_part = True

    whitelist_part = whitelist_obj & whitelist_part

    from matplotlib import pyplot as plt

    f, axarr = plt.subplots(1, 2) 
    f.set_size_inches(13, 4)
    plt.title("Detected Object and Part")

    bgr = rgb[..., ::-1]

    axarr[0].imshow(bgr * np.dstack((whitelist_obj, whitelist_obj, whitelist_obj)))
    if partname is not None:
        axarr[1].imshow(bgr * np.dstack((whitelist_part, whitelist_part, whitelist_part)))
      
    plt.pause(0.0001)

    return whitelist_obj, whitelist_obj & whitelist_part 

def init(real_life):

    global rtde_c, rtde_r, gripper, wrist_cam

    ip_address='192.168.1.123'

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
        wrist_cam = rs.RSCapture(serial_number='044122070299', use_meters=False, preset='Default')

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
        front_rgb, front_depth = capture_wrist_cam()
        torch.save((front_cam_pose, front_rgb, front_depth), 'capture_pointcloud/front_data')
        Image.fromarray(front_rgb).save('capture_pointcloud/front.png')

def moveit(real_life, objname, partname, target_holder, ):
    '''
    Moves the UR5 (with wrist mounted depth camera) to different poses in order to construct a more complete pointcloud
    than one perspective would provide
    '''
    
    global rtde_c, rtde_r, gripper
          
    front_cam_pose, front_rgb, front_depth = torch.load('capture_pointcloud/front_data')

    whitelist_obj, whitelist_part = grabbable_region(front_rgb, front_depth, partname, objname)
    
    depth_obj = front_depth.copy()
    depth_obj[~whitelist_obj] = 0.0


    # Resolve what is grabbale and what to avoid
    if partname is None:
      grab_mask = depth_obj
    else:
      if target_holder == 'robot':
        depth_part = front_depth.copy()
        depth_part[~(whitelist_obj & whitelist_part)] = 0.0
        grab_mask = depth_part
      elif target_holder == 'human':
        depth_notpart = front_depth.copy()
        depth_notpart[~(whitelist_obj & ~whitelist_part)] = 0.0
        grab_mask = depth_notpart

    # K = np.array([[910.571960449219, 0, 649.206298828125], [0, 911.160827636719, 358.177185058594]])
    # pc, col = depth2pc(grab_mask, K, rgb=front_rgb)
    # view_pc(pc, col=col/256, show=False, title='Grabbable region')
    # plt.show(block=False)

    '''
    Three cases:

    1 Just object specified: select farthest robot grasp from hand candidates
    2 Object and part, with robot grasp part: do the grasps in OBJECT A PART
    3 Object and part, with robot grasp part: do the grasps in OBJECT A ~PART
    '''

    # Get best grasp
    best_grasp, best_grasp_rot = get_grasp(front_rgb, front_depth, 
                                            mask=grab_mask, 
                                            avoid_hands=partname is None # Case 1, avoid hands
                                            )
    

    # pc_world, col_world = transform(depth_obj, front_rgb, front_cam_pose)
    # pcs_world_obj += [pc_world]
    # cols_world_obj += [col_world]

    # if partname is not None:
    #   pc_world, col_world = transform(depth_part, front_rgb, front_cam_pose)
    #   pcs_world_part += [pc_world]
    #   cols_world_part += [col_world]

    best_grasp_rot = transform_rot(best_grasp_rot, front_cam_pose)
    tool_vec = best_grasp_rot.apply([0, 0, 0.06])
    # print('tool vector direction', tool_vec)

    best_grasp = transform_point(best_grasp, front_cam_pose).squeeze()
    # best_grasp[2] = max(best_grasp[2], 0.195) # Any height below 0.20 intersects table

    print('Best robot grasp pose', list(best_grasp) + list(best_grasp_rot.as_rotvec()))
    
    def goto():
      # Setup for Sim
      gym, sim, env, viewer, camera_handle, arm, dof_states = setup()
      # I believe execute_RMP_path takes in rads, but double check.
      path = execute_RMP_path(gym, sim, env, viewer, arm, [], dof_states, front)[2]

      index = 0
      while index < len(path):
        joint_config = path[index]
        rtde_c.moveJ(joint_config, 1.5, .9, asychronous=False)
        index += 10
      rtde_c.moveJ(path[-1], 1.5, .9, asychronous=False)


    if real_life:
      
        # Go to staging
        rtde_c.moveL(list(best_grasp - 2 * tool_vec + np.array([0, 0, -0.02])) + list(best_grasp_rot.as_rotvec()), 0.3, 0.3) # [3.14, 0, 0]

        # Go to final
        rtde_c.moveL(list(best_grasp + 0.5 * tool_vec + np.array([-0, 0, 0])) + list(best_grasp_rot.as_rotvec()), 0.3, 0.3) # [3.14, 0, 0]

        # Close gripper
        # final_pos, status = gripper.move_and_wait_for_pos(gripper.get_closed_position(), 100, 1)

        # if status == robotiq_gripper.RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT:
        #     print("Gripper grasped an object")
        #     final_pos, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), 100, 1)

        # else:
        #     final_pos, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), 100, 1)
        #     print('Gripper felt nothing')

        # Move back out to front
        rtde_c.moveJ(deg_to_rad(front), 1.5, 0.9, asynchronous=False)
    
    plt.show()

    # Consolidate
    # pcs_combined = np.concatenate(pcs_world_obj, axis=0) # [N, 3]
    # cols_combined = np.concatenate(cols_world_obj, axis=0) # [N, 3] in range [0, 1)

    # np.savez('capture_pointcloud/pc_obj.npz', pcs_combined=pcs_combined, cols_combined=cols_combined / 256)

    # pcs_combined = np.concatenate(pcs_world_part, axis=0) # [N, 3]
    # cols_combined = np.concatenate(cols_world_part, axis=0) # [N, 3] in range [0, 1)

    # np.savez('capture_pointcloud/pc_part.npz', pcs_combined=pcs_combined, cols_combined=cols_combined / 256)

    # Reset for next capture
    # rtde_c.moveJ(deg_to_rad(front), 1.5, 0.9, asynchronous=False)

