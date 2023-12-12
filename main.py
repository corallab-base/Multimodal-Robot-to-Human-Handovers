# Imports
import os
import tensorflow.compat.v1 as tf
import pickle
from time import sleep
from util import *
import numpy as np
from rmp import RMP
from constants import *
from gym_config import *
from rfm import TargetRMP
from isaacgym import gymapi
from robot_config import rac
import matplotlib.pyplot as plt
from ompl_planner import path_planner
from timeit import default_timer as timer
from scipy.spatial.transform import Rotation as R
import CoGrasp.CoGrasp.contact_graspnet_util as cgu
from kinematics import get_final_position, ik_solver
from world_builder import create_table, create_target_box, create_obstacle


# Importing PoinTr from its root directory
os.chdir("CoGrasp/PoinTr")
# from CoGrasp.PoinTr import shape_completion
os.chdir("../../")


# Main function
def main(args):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf.keras.backend.set_session(tf.Session(config=config))

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

    # Creates a point cloud of the valid task space (resembles semi-sphere)
    if (graph_task_space):
        graph_valid_task_space(3)

    # Creating the table
    # table_handle, table_pose, collision_mesh, static_pts = create_table(gym, env, sim)

    # Intializing the target object
    target_handle, target_pose, target_collision_object = create_target_box(gym, env, sim, object_name)

    # Target location
    x, y, z = target_pose.p.x, target_pose.p.y, target_pose.p.z
    print("Real Location:", x, y, z)

    static_pts = []
    # Creating random obstacles to test path planning ability
    # static_pts = np.squeeze(static_pts)
    static_envs_objects = np.array([])
    if (want_obtacles):
        o1 = create_obstacle(gym, env, sim, 0, -.3, .25, .25, .65)
        o2 = create_obstacle(gym, env, sim, 0, .5, .25, .25, .95)
        o3 = create_obstacle(gym, env, sim, .4, .25, .25, .25, .95)
        o4 = create_obstacle(gym, env, sim, 0, .37, .25, .01, 1.2)
        o4 = create_obstacle(gym, env, sim, .05, .37, .01, .25, 1)
        o5 = create_obstacle(gym, env, sim, .1, .5, .01, .25, 1.2)

        static_envs_objects = np.concatenate((static_envs_objects, o1[2], o2[2], o3[2], o4[2], o5[2]))
        static_pts = np.concatenate((static_pts, o1[3], o2[3], o3[3], o4[3], o5[3]))

    # Testing specified item
    paths = []
    if (testing == "rmp"):
        # Runs the RMP based path planner
        paths_tmp, avg_dof_speed = test_RMP(gym, sim, env, viewer, arm, dof_states, static_pts, static_envs_objects, num_iters)
        paths = paths_tmp
    elif (testing == "ompl"):
        # Using the rmp based speeds when executing OMPL paths
        avg_dof_speed = [0.023190913962963043, 0.025254645658154846, 0.03518762417421687, 0.03676363879448167, 0.041092060280995484, 1.2353503925588205e-14]
        avg_dof_speed[5] = np.average(np.array(avg_dof_speed))

        # Runs the OMPL based path planner
        paths_tmp, avg_dof_speed = test_OMPL(gym, sim, env, viewer, arm, dof_states, plane, static_envs_objects, avg_dof_speed, num_iters)
        paths = paths_tmp
    elif (testing == "none"):
        # Holds position (prevents gravity) as not testing anything
        do_nothing(gym, sim, env, viewer, arm, dof_states, camera_handle, target_handle, static_pts)
    

    # Saves the executed paths of the rmp or ompl planner
    if (testing != "none" and save_paths):
        with open('outputs/' + testing + '_paths.pkl', 'wb') as file:
            pickle.dump(paths, file)


# Fights gravity and holds the current arm position
def do_nothing(gym, sim, env, viewer, arm, dof_states, camera_handle, target_handle, static_pts):
    # For CoGrasp
    t = 0

    # Runnning through the simulation
    while not gym.query_viewer_has_closed(viewer):
        # Preventing gravity
        setAllDOF(gym, env, arm, dof_states, dof_states['pos'])

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Updating the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # render the camera sensors
        gym.render_all_camera_sensors(sim)

        # Saving the current image
        if t == 25:
            # Set up
            object = "apple"
            ob_id = object_segment_dict[object]

            # Writing images to disk
            gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, "inputs/rgb.png")
            gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION, 'inputs/seg.png')
            gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_DEPTH, "inputs/depth.png")

            # Getting different images of environment
            rgb_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
            rgb_image = rgb_image.reshape(rgb_image.shape[0], -1, 4)
            rgb_image = rgba_to_rgb(rgb_image)

            depth_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
            segmented_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)

            print(f'Unique segments: {np.unique(segmented_image)}')

            graspnet_input = {'depth': -depth_image, 'rgb': rgb_image, 'segmap': segmented_image}
            input_path = "inputs/cograsp_input.npy"
            dir = os.path.dirname(__file__)
            input_path = os.path.join(dir, input_path)
            np.save(input_path, graspnet_input)
            print(f"Input Data Saved at:", input_path)

            import torch
            input_path2 = "/home/corallab/gaze/shared.pth"
            input_dict = torch.load(input_path2)

            # Reading the input data
            rgb_image = np.float32(input_dict['RGB'])
            depth_image = np.float32(input_dict['depth'])
            seg_image = np.array(input_dict['mask'])
            pc = np.array(input_dict['pc'])

            # print(input_dict)

            ll =pc.shape[0]
            print(pc)
            print(pc[:,0])
            center = [np.mean(pc[:,0]),  np.mean(pc[:,1]), np.mean(pc[:, 2])]
            print("center:", center)

            # Generate Grasps
            # cgu.generate_grasps(input_path)
            print("Done generating grasps!\n")

            # Read the predictions
            results = np.load('CoGrasp/CoGrasp/results/predictions_cograsp_input.npz', allow_pickle = True)
            keys = [k for k in results.files]

            # Printing out information to debug
            print(keys)
            # print("Predicted Grasps:", results['pred_grasps_cam'][()][ob_id])
            # print("Contact Pts:", results['contact_pts'][()][ob_id])
            # print("Gripper Openings:", results['gripper_openings'][()][ob_id])
            # print("Scores:", results['scores'][()][ob_id])
            # print("PC:", results['pc'][()])
            
            # Checking if transformation is correct
            transform = gym.get_camera_transform(sim, env, camera_handle)
            camera_position, camera_rotation = transform.p, transform.r
            camera_position = [camera_position.x, camera_position.y, camera_position.z]
            camera_rotation = R.from_quat([camera_rotation.x, camera_rotation.y, camera_rotation.z, camera_rotation.w])

            print(np.array(center).T)
            transformed_center = transform_grasps(camera_position, camera_rotation, np.array([np.array(center)]))
            print("transformed center:", transformed_center)


            # rotation = pred_grasps_cam[index, 0:3, 0:3]
            # rot3 = camera_rotation.as_quat().tolist()
            # rot2 = R.from_euler('Y', pi / 2).as_quat().tolist()
            # rot1 = R.from_euler('Z', -pi / 2).as_quat().tolist()
            # concat_rot = rotation_concat(rot3, rotation_concat(rot2, rot1))
            # concat_rot = R.from_quat(concat_rot)
            # qauternion = R.from_matrix(concat_rot.apply(rotation)).as_quat()

            x, y, z = transformed_center[0][0], transformed_center[0][1], transformed_center[0][2]
            
            # x, y, z = x + .06, y, z + .35
            sol = ik_solver.get_IK([.3, 0, .85], [.707, 0.0, 0.0, .707])
            print(sol)
            print(x, y, z)

            if (sol == None):
                print("no sol found!")
                t += 1
                continue
    
            
            # Building 3D figure of object in correct coord. system
            # transformed_pc_pts = transform_grasps(camera_position, camera_rotation, results['pc'][()][ob_id])
            # view_pc(transformed_pc_pts)

            # pred_grasps_cam = results['pred_grasps_cam'][()][ob_id]
            # transformed_pred_grasps = transform_grasps(camera_position, camera_rotation, pred_grasps_cam[:, 0:3, 3])

            # x, y, z, sol = 0, 0, 0, None
            # for index in np.argsort(results['scores'][()][ob_id]):
            #     grasp = transformed_pred_grasps[index]

            #     # Fixing rotation
            #     rotation = pred_grasps_cam[index, 0:3, 0:3]
            #     rot3 = camera_rotation.as_quat().tolist()
            #     rot2 = R.from_euler('Y', pi / 2).as_quat().tolist()
            #     rot1 = R.from_euler('Z', -pi / 2).as_quat().tolist()
            #     concat_rot = rotation_concat(rot3, rotation_concat(rot2, rot1))
            #     concat_rot = R.from_quat(concat_rot)
            #     qauternion = R.from_matrix(concat_rot.apply(rotation)).as_quat()

            #     x, y, z = grasp[0], grasp[1], grasp[2]
            #     # x, y, z = grasp[0] + .06, grasp[1], grasp[2] + .25
            #     sol = ik_solver.get_IK([x, y, z], qauternion)

            #     if (sol != None):
            #         break

            # if (sol == None):
            #     print("No solution found!")
            #     exit()

            ret = execute_RMP_path(gym, sim, env, viewer, arm, static_pts, dof_states, sol)
            path = ret[2]

            np.save("cur_path.npy", path)

            # Setting current position to solution
            print("Final end effector position:", get_final_position(dof_states['pos']))

            # Get object point cloud from simulation
            # depth_image_segmented = np.where(segmented_image==object_segment_dict[object], depth_image, 0)
            # obj_pc, _ = depth2pc(-depth_image_segmented, K)

            # import CoGrasp.GraspTTA.hand_prediction as hp
            # obj_pc, hand_pc = hp.predict_hand_grasp(obj_pc, 'cuda')
            # print(hand_pc)

            # obj_pc_world = pc_from_camera_to_world(obj_pc, camera_transform)
            # obj_pc_center = pc_from_camera_to_object(obj_pc, camera_transform, obj_pc_world.mean(axis=0))

            # os.chdir("CoGrasp/PoinTr")
            # obj_pc_prediction = shape_completion.complete_pc(obj_pc_center)
            # os.chdir("../../")
            # print(f'Completed Object PointCloud: {obj_pc_prediction}')

            # object = "apple"
            # print(results[keys[0]][object_segment_dict[object]])
            # pred_grasps_cam = results[keys[0]][()][object_segment_dict[object]]
            # scores = results[keys[1]][()][object_segment_dict[object]]
            # print(pred_grasps_cam, scores)
            # transformed_pred_grasps = transform_grasps(pred_grasps_cam, camera_transform, object_translations[object][i])

        t += 1


        # Synchronizes the physics simulation with the rendering rate.
        if (real_time_rendering):
            gym.sync_frame_time(sim)


# Tests the RMP methodology
def test_RMP(gym, sim, env, viewer, arm, dof_states, static_obj_pts, static_envs_objects, iterations=250):
    total_time = 0.0
    total_dist = 0.0
    total_config_dist = 6 * [0.0]
    
    paths = []
    
    for iter in range (iterations):
        # Getting the start and end points for path planning
        start, end = generate_start_end(static_envs_objects)
        setAllDOF(gym, env, arm, dof_states, start)
        
        start_time = timer()
        ret = execute_RMP_path(gym, sim, env, viewer, arm, static_obj_pts, dof_states, end)
        
        if (ret != None):
            dist = ret[0]
            config_dist = ret[1]
            paths.append(ret[2])

            end_time = timer()

            total_dist += dist
            for i in range (6):
                total_config_dist[i] += config_dist[i]
            total_time += end_time - start_time
        else:
            iter -= 1

    return paths, print_info("RMP", iterations, total_dist, total_config_dist, total_time,)


# Executes the RMP based path planning
def execute_RMP_path(gym, sim, env, viewer, arm, static_points, dof_states, goal_dof, goal_dist_threshold=.01):
    # Testing/tracking
    total_dist = 0.0
    total_config_dist = 6 *[0.0]

    # Getting the final pos
    goal_dof = np.append(goal_dof, 6 * [0.0], axis=0)
    goal_pos = get_final_position(goal_dof)

    # Building the target RMP
    rmp = RMP(TargetRMP([goal_pos[0], goal_pos[1], goal_pos[2]]))

    # Getting the ground pts
    # ground_pts = get_ground_pts()
    
    # Combining with existing static objects
    # static_points = np.extent(static_points, ground_pts)
    static_points = np.array([])

    # Plotting the static points
    # view_pc(static_points)

    # Setting up for velocity and acceleration management
    task_vel = 3 * [0.0]

    # Building the path to return
    path = []

    # Runnning through the simulation
    while not gym.query_viewer_has_closed(viewer):
        path.append(dof_states['pos'][0:6].copy())

        self_collision_points = get_robot_points(dof_states)
        acceleration = rmp.pull(dof_states['pos'], task_vel, static_points, self_collision_points)
        cur_pos = get_final_position(dof_states['pos'])

        # Manually stepping through physics
        for dof_index in range (0, 6):
            dof_states['vel'][dof_index] += acceleration[dof_index] * dt
            dof_states['pos'][dof_index] += dof_states['vel'][dof_index]
            total_config_dist[dof_index] += abs(dof_states['vel'][dof_index])

        # Updating to current transformation
        gym.set_actor_dof_states(env, arm, dof_states, gymapi.STATE_POS)

        dt_pos = get_final_position(dof_states['pos'])
        dist = distance(cur_pos, dt_pos)

        # Updating the distance travelled
        total_dist += dist

        # Checking if reached end destination
        if (distance(cur_pos, goal_pos) < goal_dist_threshold):
            return total_dist, total_config_dist, path

        # Updating the velocity
        task_vel = (dt_pos - cur_pos) / dt

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


# Tests the pre-existing OMPL methodology
def test_OMPL(gym, sim, env, viewer, arm, dof_states, plane_mesh, static_envs_objects, avg_dof_speed, iterations=250):
    # Runs the traditional path planning setup using OMPL
    total_time = 0.0
    total_dist = 0.0
    total_config_dist = 6 * [0.0]
    paths = []

    for iter in range (iterations):
        # Getting the start and end points for path planning
        start, end = generate_start_end(static_envs_objects)
        setAllDOF(gym, env, arm, dof_states, start)
        
        # Starting timer
        start_time = timer()

        # Getting a path
        planner = path_planner(static_envs_objects)
        path = planner.plan(start, end)

        if (path != None):
            paths.append(path)
            dist, config_dist = execute_traditional_path(gym, sim, env, viewer, arm, dof_states, path, avg_dof_speed)
            
            # Updating total time
            end_time = timer()
            total_time += end_time - start_time

            # Updating total distance
            total_dist += dist
            for i in range (6):
                total_config_dist[i] += config_dist[i]
        else:
            iter -= 1

    return paths, print_info("OMPL", iterations, total_dist, total_config_dist, total_time)


# Executes the traditional path generated by OMPL
def execute_traditional_path(gym, sim, env, viewer, arm, dof_states, path, dof_speed = 6 * [.01]):
    # Keeps track of current path goal
    path_index = 1
    total_dis = 0.0
    total_config_dist = 6 * [0.0]

    # Runnning through the simulation
    while not gym.query_viewer_has_closed(viewer):
        # Current path goal to reach
        cur_path_goal = path[path_index]

        # Updating the dof to get to one critical point after another
        reached = True
        cur_pos = get_final_position(dof_states['pos'])
        for dof_index in range (0, 6):
            diff = dof_states['pos'][dof_index] - cur_path_goal[dof_index]
            if (abs(diff) < 1.5 * dof_speed[dof_index]):
                continue
            elif (diff < 0):
                dof_states['pos'][dof_index] += dof_speed[dof_index]
                total_config_dist[dof_index] += dof_speed[dof_index]
                reached = False
            else:
                dof_states['pos'][dof_index] -= dof_speed[dof_index]
                total_config_dist[dof_index] += dof_speed[dof_index]
                reached = False

        # Making changes reflect
        gym.set_actor_dof_states(env, arm, dof_states, gymapi.STATE_POS)

        end_pos = get_final_position(dof_states['pos'])
        total_dis += distance(cur_pos, end_pos)

        # Checking if reached the cur_path_goal
        if (reached):
            if (path_index < len(path) - 1):
                path_index += 1
            else:
                return total_dis, total_config_dist

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Updating the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Synchronizes the physics simulation with the rendering rate.
        if (real_time_rendering):
            gym.sync_frame_time(sim)



def print_info(str, iterations, total_dist, total_config_dist, total_time):
    print("Based on", iterations, "iterations, for", str, ":")
    print("Total task space distance:", total_dist)
    print("Average task space distance:", total_dist / float(iterations))
    print("Total config space distance:", total_config_dist)

    avg_config_space_dist = [x / float(iterations) for x in total_config_dist]
    print("Average config space distance:", avg_config_space_dist)

    avg_time = total_time / float(iterations)
    print("Average execution time:", avg_time)

    avg_dof_speeds = [((x / avg_time) * dt) for x in avg_config_space_dist]
    print("Average dof speed:", avg_dof_speeds)

    return avg_dof_speeds