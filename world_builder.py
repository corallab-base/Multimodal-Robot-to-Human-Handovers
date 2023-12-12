# Imports
import fcl
import numpy as np
from math import pi
from constants import *
from isaacgym import gymapi
from gym_config import load_asset
from robot_config import STL_Reader


# Positioning other objects according to the table
object_poses = {'banana': gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.05, table_dims.z),
                'mug': gymapi.Vec3(table_pose.p.x + 0.1, table_pose.p.y + 0.3, table_dims.z),
                'orange': gymapi.Vec3(table_pose.p.x, table_pose.p.y - 0.15, table_dims.z),
                'bowl': gymapi.Vec3(table_pose.p.x - 0.2, table_pose.p.y - 0.25, table_dims.z),
                'apple': gymapi.Vec3(table_pose.p.x - 0.1, table_pose.p.y + 0.25, table_dims.z)}


# Creates a target box for the arm to move
def create_target_box(gym, env, sim, x, y, z, name="013_apple"):
    # Create the box asset fro YCB urdf file
    box_asset = load_asset(gym, sim, name + '.urdf', "./assets/urdf/ycb/" + name, fix_base_link=True)

    # Getting position on the table
    target_pose = gymapi.Transform()
    # target_pose.p = gymapi.Vec3(np.random.uniform(.3, .45), table_pose.p.y + np.random.uniform(-.3, .3), table_dims.z)
    # target_pose.p = gymapi.Vec3(1, -.6, table_dims.z)
    target_pose.p = gymapi.Vec3(x, y, z)
    target_pose.p.z += .1
    target_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-pi, pi))

    # Creating the target
    target_handle = gym.create_actor(env, box_asset, target_pose, "target", 0, 0, segmentationId=object_segment_dict[name[4:]])

    # Setting a color
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, target_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # Creating a collision mesh for tagret box
    target_stl = STL_Reader("./assets/urdf/ycb/" + name + "/nontextured.stl")
    target_collision_object = target_stl.get_fcl_collision_object(target_pose.p.x, target_pose.p.y, target_pose.p.z, target_pose.r.w, target_pose.r.x, target_pose.r.y, target_pose.r.z)

    return target_handle, target_pose, target_collision_object


# Create random obstacle
def create_random_obstacle(gym, env, sim, length, width, height):
    x = np.random.uniform(-.5, .5)
    y = np.random.uniform(-.75, .75)
    return create_obstacle(gym, env, sim, x, y, length, width, height)


# Create obstacle in given location
def create_obstacle(gym, env, sim, x, y, length, width, height):
    # Create table asset
    obs_dims = gymapi.Vec3(length, width, height)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    obs_asset = gym.create_box(sim, obs_dims.x, obs_dims.y, obs_dims.z, asset_options)

    # Positioning the obstacle
    z = .05
    obs_pose = gymapi.Transform()
    obs_pose.p = gymapi.Vec3(x, y, z)

    # Adding the obstacle to the environment
    obs_handle = gym.create_actor(env, obs_asset, obs_pose, "obs", 0, 0, 100)

    # creating a collision mesh for the object
    collision_mesh = create_collision_object(x, y, z, length, width, height)

    # getting points to avoid for rmp
    static_pts = get_ext_pts(x, y, z, length, width, height)

    return obs_handle, obs_pose, collision_mesh, static_pts


# Create obstacle in given location
def create_table(gym, env, sim):
    # Create table asset
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    rigid_shape_properties = gymapi.RigidShapeProperties()
    rigid_shape_properties.friction = 2.0
    gym.set_asset_rigid_shape_properties(table_asset, [rigid_shape_properties])

    # Adding the obstacle to the environment
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0, segmentationId=object_segment_dict['table'])
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # creating a collision mesh for the object
    collision_mesh = create_collision_object(table_pose.p.x, table_pose.p.y, table_pose.p.z, table_dims.x, table_dims.y, table_dims.z)

    # getting points to avoid for rmp
    static_pts = get_ext_pts(table_pose.p.x, table_pose.p.y, table_pose.p.z, table_dims.x, table_dims.y, table_dims.z)

    return table_handle, table_pose, collision_mesh, static_pts


# Create a collision mesh using fcl
def create_collision_object(x, y, z, length, width, height):
    obstacle_geometry = fcl.Box(length, width, height)
    obstacle_translation = np.array([x, y, z])
    obstacle_transformation = fcl.Transform(obstacle_translation)
    obstacle = fcl.CollisionObject(obstacle_geometry, obstacle_transformation)
    return obstacle


# Getting exterior set of points for a box
def get_ext_pts(x, y, z, length, width, height):
    pts = []
    detail = .5
    move_x = x - length
    while (move_x < x + length):
        move_y = y - width
        while (move_y < y + width):
            move_z = z - height
            while (move_z < z + height):
                pts.append([move_x, move_y, move_z])
                move_z += detail
            move_y += detail
        move_x += detail

    for index in range (0, len(pts)):
        pts[index] = np.reshape(np.array(pts[index]), newshape=(3, 1))
    return np.array(pts)