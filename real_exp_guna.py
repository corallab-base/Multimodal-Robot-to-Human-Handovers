# General Imports
import os
from util import *
import numpy as np
from time import sleep
from constants import *
from gym_config import *
from kinematics import *
from isaacgym import gymapi
import matplotlib.pyplot as plt
from main import execute_RMP_path
from world_builder import create_table, create_target_box
import CoGrasp.CoGrasp.contact_graspnet_util as cgu

# Needs to be imported after IsaacGym
import torch

# Real World Experiment Imports
import rtde_control
import rtde_receive
import robotiq_gripper
from rtde_control import RTDEControlInterface as RTDEControl


# Util function to display a point cloud
def view_pc(pc, title=''):
    # Setting up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud data
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)

    # Set the axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.title(title)

    # Show the plot
    plt.show()

# Gets dof into specific range
def get_valid_dof(dof, lower=0, upper=pi):
    for index in range (len(dof)):
        while (dof[index] < lower):
            dof[index] += pi
        while (dof[index] > upper):
            dof[index] -= pi
        if (dof[index] == 0.0):
            dof[index] = .00001
    return dof


# Powers off the robot
def power_off_pose(rtde_c):
    print('Reset to initial upstraight pose')
    print(initial_dof)
    initial_dof2 = get_valid_dof(initial_dof)
    print(initial_dof2)
    rtde_c.moveJ([initial_dof2])


# Setting up isaac gym env
def setup():
    # Configuring the environment
    gym, sim, viewer, env, camera_handle = configure_env()

    # Loading the robot arm asset
    ur5e = load_asset(gym=gym, sim=sim, asset_name="urdf/ur5e_mimic_real.urdf")
    
    # Creating an arm actor with default state
    arm = create_arm_actor(gym, env, ur5e, "arm")

    # Correctly initialize default positions, limits, and speeds
    num_dofs = gym.get_asset_dof_count(ur5e)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    setAllDOF(gym, env, arm, dof_states, initial_dof)

    # Creating the table
    # table_handle, table_pose, collision_mesh, static_pts = create_table(gym, env, sim)

    return gym, sim, env, viewer, camera_handle, arm, dof_states
    

# Gets the RMP path from point A to B
def get_rmp_path(gym, sim, env, arm, viewer, dof_states, start, end):
    # Creating random obstacles to test path planning ability
    static_obj_pts = np.array([])

    # Getting the start and end points for path planning
    setAllDOF(gym, env, arm, dof_states, start)

    # Getting the rmp path
    goal = get_final_position(end)
    ret = execute_RMP_path(gym, sim, env, viewer, arm, static_obj_pts, dof_states, goal)
    return ret[2]


# Gets the environmental info. from Gaze and returns grasp for target object
def get_initial_grasp(gym, sim, env, camera_handle, viewer, input_dict, arm, dof_states, viz=False):

    # Reading the input data
    rgb_image = np.float32(input_dict['RGB'])
    depth_image = np.float32(input_dict['depth'])
    seg_image = np.array(input_dict['mask'])
    pc = np.array(input_dict['pc'])

    # Focusing in on the object
    rgb_image_mask = np.stack([seg_image, seg_image, seg_image], axis=-1) * rgb_image
    rgb_image_mask = crop(rgb_image_mask)

    mask = crop_slice(seg_image)
    depth_image_mask = depth_image[mask]

    seg_image = np.ones(shape=(depth_image_mask.shape[0], depth_image_mask.shape[1]))
    print("Depth/RGB Shapes:", depth_image_mask.shape, rgb_image_mask.shape)

    # Saving input for cograsp
    graspnet_input = {'depth': depth_image_mask, 'rgb': rgb_image_mask, 'segmap': seg_image, 'pc': pc}
    input_path = "inputs/cograsp_input.npy"
    input_path = os.path.abspath(input_path)
    np.save(input_path, graspnet_input)
    print(f"Input Data Saved at:", input_path)

    # Generate Grasps
    # cgu.generate_grasps(input_path)
    # print("Done generating grasps!\n")

    # Read the predictions
    # results = np.load('CoGrasp/CoGrasp/results/predictions_cograsp_input.npz', allow_pickle = True)

    # Printing out information to debug
    ob_id = 1
    # print("Predicted Grasps:", results['pred_grasps_cam'][()][ob_id])
    # print("Contact Pts:", results['contact_pts'][()][ob_id])
    # print("Gripper Openings:", results['gripper_openings'][()][ob_id])
    # print("Scores:", results['scores'][()][ob_id])

    # Viewing the point cloud
    # pc = results['pc'][()][ob_id]
    if viz: view_pc(pc, "Untransformed PC")

    # Viewing the pc after transformation
    camera_translation, camera_rotation = (1.20, 0, .85), R.from_quat([0, 0, 1, 0])
    print("Point cloud mean (raw):", pc[:, 0].mean(), pc[:, 1].mean(), pc[:, 2].mean())
    pc = transform_grasps(camera_translation, camera_rotation, pc)
    if viz:
        view_pc(pc, 'Transformed PC')

    print("Point cloud mean (transformed):", pc[:, 0].mean(), pc[:, 1].mean(), pc[:, 2].mean())
    transformed_pred_pts = np.array([[pc[:, 0].mean(), pc[:, 1].mean(), pc[:, 2].mean()]])
    
    # Viewing transformed grasps
    # pred_grasps_cam = results['pred_grasps_cam'][()][ob_id]
    # transformed_pred_pts = transform_grasps(camera_translation, camera_rotation, pred_grasps_cam[:, 0:3, 3])
    if viz:
        view_pc(transformed_pred_pts)

    # Finding rotation to fix that of camera
    rot3 = camera_rotation.as_quat().tolist()
    rot2 = R.from_euler('Y', pi / 2).as_quat().tolist()
    rot1 = R.from_euler('Z', -pi / 2).as_quat().tolist()
    concat_rot = rotation_concat(rot3, rotation_concat(rot2, rot1))
    concat_rot = R.from_quat(concat_rot)

    # Getting correct pos and rot
    index = 0
    grasp_pts = transformed_pred_pts[index]

    x, y, z = grasp_pts[0], grasp_pts[1], grasp_pts[2]
    # x, y, z = x + robot_offset[0] + .30, y + robot_offset[1] - .1, z + .276
    x, y, z = x + robot_offset[0] + .30, y + robot_offset[1] - .1, z + .22

    sol = None
    for _ in range(100):
        print('Looking for grasp solution')
        sol = ik_solver.get_IK([x, y, z], [.707, 0, -.707, 0])
        if sol is not None:
            print("Found grasp solution!", sol)
            break

    # if (sol != None):
    #     # Step the physics
    #     setAllDOF(gym, env, arm, dof_states, sol)
    #     gym.simulate(sim)
    #     gym.fetch_results(sim, True)

    #     # Updating the viewer
    #     gym.step_graphics(sim)
    #     gym.draw_viewer(viewer, sim, True)

    #     # render the camera sensors
    #     gym.render_all_camera_sensors(sim)

    #     # Synchronizes the physics simulation with the rendering rate.
    #     if (real_time_rendering):
    #         gym.sync_frame_time(sim)

        
    #     sleep(3)

    print(sol)
    print(np.degrees(sol))

    # Finding viable solution to grasp
    # x, y, z, sol = 0, 0, 0, None
    # for index in np.argsort(results['scores'][()][1]):
    #     # Getting correct pos and rot
    #     grasp_pts = transformed_pred_pts[index]
    #     rotation = pred_grasps_cam[index, 0:3, 0:3]
    #     rotation = R.from_matrix(concat_rot.apply(rotation)).as_quat()

    #     x, y, z = grasp_pts[0] - .03, grasp_pts[1], grasp_pts[2] + .17
    #     sol = ik_solver.get_IK([x, y, z], rotation)

    #     if (sol != None):
    #         # Step the physics
    #         setAllDOF(gym, env, arm, dof_states, sol)
    #         gym.simulate(sim)
    #         gym.fetch_results(sim, True)

    #         # Updating the viewer
    #         gym.step_graphics(sim)
    #         gym.draw_viewer(viewer, sim, True)

    #         # render the camera sensors
    #         gym.render_all_camera_sensors(sim)

    #         # Synchronizes the physics simulation with the rendering rate.
    #         if (real_time_rendering):
    #             gym.sync_frame_time(sim)

    #         print("Current solution:", sol)
    #         sleep(3)

    # return sol
    return sol, x, y, z


def get_real_rotation(rx, ry, rz):

    vector = np.array([rx, ry, rz])
    mag = np.linalg.norm(vector)
    vector /= mag
    rot1 = R.from_rotvec(mag * vector)
    rot2 = R.from_euler("YX", [-math.pi/2, math.pi/2])

    rot2 = rot1 * rot2
    t1, t2, t3, t4 = rot2.as_quat()
    rot3 = R.from_quat([-t1, -t2, t3, t4])
    #print (rot3.apply([-1, 0, 0]))
    #print (rot3.apply([0, -1, 0]))
    #print (rot3.apply([0, 0, 1]))
    rot4 = R.from_euler('Z', math.pi)
    rot4 = rot3*rot4
    #print (rot4.apply([1, 0, 0]))
    #print (rot4.apply([0, 1, 0]))
    #print (rot4.apply([0, 0, 1])
    #print(rot4.as_quat())
    return rot4.as_quat()

def get_real_location(x, y, z, rotation):
    offset = R.from_quat(rotation).apply([-0.1, 0.0, 0.0])
    print ("offset is: ")
    print (offset)
    return [- x + offset[0], - y + offset[1], z + offset[2]]


def main(input_dict, ip_address='192.168.1.123', viz=False):
    # Setup for real robot
    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ip_address, 63352)

    rtde_c.moveJ(initial_dof, 0.5, 0.5)
    gripper.activate(auto_calibrate=True)


    # power_off_pose(rtde_c)

    # initial grasp
    # path = np.load("cur_path.npy")
    # path = path.tolist()
    # print("Type:", type(path), type(path[0]))

    # opt_path = []
    # for p in path:
    #     if p.all() < 2 and p.all() > 0:
    #         opt_path.append(p)

    # # print(opt_path)
    # rtde_c.moveJ(path[0:2])
    # gripper.activate()


    # Setup for Sim
    gym, sim, env, viewer, camera_handle, arm, dof_states = setup()

    # # Getting env info
    grasp_dof, x, y, z = get_initial_grasp(gym, sim, env, camera_handle, viewer, input_dict, arm, dof_states)
    path = execute_RMP_path(gym, sim, env, viewer, arm, [], dof_states, grasp_dof)[2]

    # velocity, acceleration, blend = .3, .1, .05
    # fixed_path = []
    # for joint_config in path:
    #     new_jc = list(joint_config.copy())
    #     new_jc.append(velocity)
    #     new_jc.append(acceleration)
    #     new_jc.append(blend)
    #     fixed_path.append(new_jc)
    # fixed_path[0][-1] = 0
    # print("Path:", type(path), path)
    # print("New path:", fixed_path)

    # rtde_c.moveL(fixed_path)
    # exit(0)
    index = 0
    while index < len(path):
        joint_config = path[index]
        rtde_c.moveJ(joint_config, .05, .05)
        index += 10
    rtde_c.moveJ(path[-1], 0.5, 0.9)
    exit()

    # z -= .03
    
    # sol = None
    # for _ in range(100):
    #     print('Looking for grasp solution')
    #     sol = ik_solver.get_IK([x, y, z], [.707, 0, -.707, 0])
    #     if sol is not None:
    #         print("Found grasp solution!", sol)
    #         break

    # print("moving to solution!")
    # rtde_c.moveJ(sol, 0.5, 0.9)
    
    gripper.move_and_wait_for_pos(gripper.get_closed_position(), 64, 1)

    handoff = ik_solver.get_IK([0, 0.5, 0.5], [.707, 0, -.707, 0])

    if handoff is None:
        print("ERROR: Could not find handoff solution!")
    rtde_c.moveJ(initial_dof, 0.5, 0.9)
    
    if viz:
        setAllDOF(gym, env, arm, dof_states, grasp_dof)
        while not gym.query_viewer_has_closed(viewer):
            actual_joints = rtde_r.getActualQ()

            setAllDOF(gym, env, arm, dof_states, actual_joints)

            # Step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Updating the viewer
            gym.step_graphics(sim)
            
            gym.draw_viewer(viewer, sim, True)

            # render the camera sensors
            gym.render_all_camera_sensors(sim)

            # Synchronizes the physics simulation with the rendering rate.
            if (real_time_rendering):
                gym.sync_frame_time(sim)

    # initial grasp
    
    # end_goal = [-0.38029122485476474, -0.15830819885596056, 0.586445018182197, -1.920210385330609, -1.6476219275051471, 0.5311758787505217]
    # print(np.degrees(end_goal))
    # path = get_rmp_path(gym, sim, env, arm, viewer, dof_states, initial_dof, grasp_dof)
    #x, y, z = .881 + robot_offset[0], -.124 + robot_offset[1], .202 + robot_offset[2]
    # transform_grasps(camera_position, camera_rotation, np.array([x, y,]))

    # console_trans = [-0.893, -0.074, 0.252]
    # console_rot = [2.311, 2.161, -0.067]
    # true_rot = get_real_rotation(console_rot[0], console_rot[1], console_rot[2])
    # true_trans = get_real_location(console_trans[0], console_trans[1], console_trans[2], true_rot)


    # print(true_rot, true_trans)
    # exit(1)

    # print(x, y, z)
    # x, y, z = z, x, y
    # true_rot = [true_rot[3], true_rot[0], true_rot[1], true_rot[2]]
    # true_trans = [true_trans[0] + robot_offset[0], true_trans[1] + robot_offset[1], true_trans[2] + robot_offset[2]]
    # print(true_trans, true_rot)
    # end_goal = list(ik_solver.get_IK(true_trans, true_rot))
    # print(end_goal, np.degrees(end_goal))
    # path = [initial_dof, end_goal]
    # rtde_c.movePath(path)
    
    # rtde_c.servoC(end_goal, 0.25, 1.2, 0.1)


if __name__ == '__main__':
    input_path = "/home/corallab/gaze/shared.pth"
    main(torch.load(input_path))
