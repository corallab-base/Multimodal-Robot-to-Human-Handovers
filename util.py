# Imports
import numpy as np
from numpy import pi
import pyvista as pv
from isaacgym import gymapi
from robot_config import rac
import matplotlib.pyplot as plt
from constants import robot_offset
from kinematics import get_all_final_positions, ik_solver



# Normalizes (scales to 1) input vector. "c" defines how close behavior is
# to regular normalization. 0 or inf --> regular normalization. 1 --> soft normalization
def soft_normalization(input, c):
     norm = np.linalg.norm(input)
     magnitude = norm + c * np.log(1 + np.exp(-2 * c * input))
     return input / magnitude


# Gets the distance between 2 vectors
def distance(vec1, vec2):
     return np.linalg.norm(vec2 - vec1)


# Processes the rgb image
def rgba_to_rgb(rgba):
    return np.delete(rgba, -1, axis=-1)


# Helper function for loops with floats
def range_with_floats(start, stop, step):
    while stop > start:
        yield start
        start += step


# Generates valid start and end points
def generate_start_end(static_envs_objects):
    while(True):
        start, end = 12 * [0.0], 12 * [0.0]
        start[0], end[0] = np.random.rand() * pi, np.random.rand() * pi
        for index in range (1, 6):
            start[index], end[index] = np.random.uniform(high=pi, low=-pi), np.random.uniform(high=pi, low=-pi)
    
        if (not rac.has_collisions(start, static_envs_objects) and not rac.has_collisions(end, static_envs_objects)):
            return start, end
        

# Builds and visualizes a valid task space graph
def graph_valid_task_space(x_lim, y_lim, z_lim, detail=.1):
    # Going through all points in cube and storing valid points
    valid_points = []
    x = -x_lim
    while (x < x_lim):
        y = -y_lim
        while (y < y_lim):
            z = -z_lim
            while (z < z_lim):
                dof = ik_solver.get_IK([x, y, z])
                if (dof != None):
                    dof = np.resize(np.array(dof), (12))
                    if (not rac.has_collisions(dof, [])):
                        valid_points.append([x, y, z])
                z += detail
            y += detail
        x += detail

    # Plotting the valid points
    pv.plot(np.array(valid_points),
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False)


# Computes all the points on the robot in current transformation (need to be
# avoided to prevent-collisions)
def get_robot_points(dof_states, detail=.1, margin_of_error = .2):
    robot_points = []
    all_final_positions = get_all_final_positions(dof_states['pos'])
    for index in range (len(all_final_positions) - 1):
        cur = np.array(all_final_positions[index])
        next = np.array(all_final_positions[index + 1])
        
        # Tracing a line between joints
        while (np.linalg.norm(cur - next) > margin_of_error):
            robot_points.append(np.reshape(cur, newshape=(3, 1)))
            for index2 in range (len(cur)):
                if (cur[index2] < next[index2]):
                    cur[index2] += detail
                else:
                    cur[index2] -= detail

    return np.array(robot_points)


"""
Convert depth and intrinsics to point cloud and optionally point cloud color
:param depth: hxw depth map in m
:param K: 3x3 Camera Matrix with intrinsics
:returns: (Nx3 point cloud, point cloud color)
"""
def depth2pc(depth, K, rgb=None):

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]

    pc = np.vstack((world_x, world_y, world_z)).T
    return pc, rgb



# Converts point cloud from camera frame to object frame
def pc_from_camera_to_object(object_pc, camera_transform, object_translation):
    camera_transform_array = np.array([camera_transform.p.x, camera_transform.p.y, camera_transform.p.z])
    pc_world = np.copy(object_pc)
    pc_world[:, [0,1,2]] = object_pc[:,[2,0,1]]
    pc_world[:, 0] *= -1
    pc_world[:, 2] *= -1
    pc_world = pc_world + camera_transform_array
    pc_center = pc_world - object_translation
    return pc_center


# Converts point cloud from camera frame to world frame
def pc_from_camera_to_world(object_pc, camera_transform):
    camera_transform_array = np.array([camera_transform.p.x, camera_transform.p.y, camera_transform.p.z])
    pc_world = np.copy(object_pc)
    pc_world[:, [0,1,2]] = object_pc[:,[2,0,1]]
    pc_world[:, 0] *= -1
    pc_world[:, 2] *= -1
    pc_world = pc_world + camera_transform_array
    return pc_world


# Transforms grasps from camera frame to object frame
def transform_grasps(camera_translation, camera_rotation, pred_grasps_cam):
    # Fixing axis
    pred_grasps_cam = np.array(pred_grasps_cam)
    pred_grasps_cam[:, [0, 1, 2]] = pred_grasps_cam[:, [2, 0, 1]]
    pred_grasps_cam[:, 1] *= -1
    pred_grasps_cam[:, 2] *= -1

    # Getting camera transformation
    # camera_translation = [camera_translation.x, camera_translation.y, camera_translation.z]

    # camera_rotation = [camera_rotation.x, camera_rotation.y, camera_rotation.z, camera_rotation.w]
    # camera_rotation = R.from_quat(camera_rotation)

    # Applying transformation
    rotated_grasps = camera_rotation.apply(pred_grasps_cam)
    transformed_grasps = rotated_grasps + camera_translation

    return np.array(transformed_grasps)


# Multiplies 2 quatenioins
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.w, quaternion0.x, quaternion0.y, quaternion0.z
    w1, x1, y1, z1 = quaternion1.w, quaternion1.x, quaternion1.y, quaternion1.z
    return gymapi.Quat(x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)


# Applies (concatenates) 2 rotations
def rotation_concat(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]
    return [x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0]


# Util function to display a point cloud
def view_pc(pc):
    # Setting up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud data
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)

    # Set the axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Show the plot
    plt.show()


# Getting points to avoid ground collision
def get_ground_pts():
    r_squared, outside = 1.25**2, 1.5
    ground_pts = []
    for row in range_with_floats (robot_offset[0] - outside, robot_offset[0] + outside, outside / 5):
        for col in range_with_floats (robot_offset[1] - outside, robot_offset[1] + outside, outside / 5):
            if (((row - robot_offset[0])**2) + ((col - robot_offset[1])**2) <= r_squared):
                # Only adding circle points to reduce unnecessary RMPs
                ground_pts.append(np.reshape(np.array([row, col, robot_offset[2]]), newshape=(3, 1)))

    return np.array(ground_pts)


# Cropping the 0 edges of a matrix
def crop(image):
    true_points = np.argwhere(image)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    return image[top_left[0]:bottom_right[0]+1, 
              top_left[1]:bottom_right[1]+1]


def crop_slice(image):
    true_points = np.argwhere(image)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    return np.s_[top_left[0]:bottom_right[0]+1, 
              top_left[1]:bottom_right[1]+1]