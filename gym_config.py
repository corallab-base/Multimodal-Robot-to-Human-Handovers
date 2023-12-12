# Imports
from math import pi, sqrt
from isaacgym import gymapi, gymutil
from constants import dt, robot_rot, object_segment_dict, robot_offset, camera_position, camera_target


# Configures the environment and returns the gym, simulator, viewer, and environment
def configure_env():
    # Initializing the gym
    gym = gymapi.acquire_gym()

    # Parsing the arguments
    args = gymutil.parse_arguments(
        description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create"},
            {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
            {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

    # Configure the simulator
    sim_params = gymapi.SimParams()
    sim_params.dt = dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0)
    sim_params.substeps = 3
    sim_params.physx.solver_type = 1

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim ***")
        quit()

    # Configuring a ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Setting up the env grid
    num_envs = 1
    spacing = 3
    num_per_row = int(sqrt(num_envs))
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Creating an environment
    env =  gym.create_env(sim, env_lower, env_upper, num_per_row)

    # Configuring the camera
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1280
    camera_properties.height = 720
    camera_properties.horizontal_fov = 70.25
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    # Getting the camera handle
    camera_handle = gym.create_camera_sensor(env, camera_properties)
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)

    # Configuring the viewer
    viewer = gym.create_viewer(sim, camera_properties)
    gym.viewer_camera_look_at(viewer, env, camera_position, camera_target)

    return gym, sim, viewer, env, camera_handle


# Loads the asset given its name, gym, and simulator
def load_asset(gym, sim, asset_name, asset_root = "./assets", fix_base_link=True):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fix_base_link
    asset_options.disable_gravity = False
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset = gym.load_asset(sim, asset_root, asset_name, asset_options)
    return asset


# Creates an arm actor in default position ideal for viewing
def create_arm_actor(gym, env, ur5e, name):
    # Creating an initial pose for the arm
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(robot_offset[0], robot_offset[1], robot_offset[2])
    pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*pi)

    # Adding an arm actor to the environment
    arm = gym.create_actor(env, ur5e, pose, name, 0, 0, segmentationId = object_segment_dict["ur5e"])
    return arm


# Updates a specific dof by a given value
def updateDOF(gym, env, arm, dof_states, current_dof, value):
    dof_states['pos'][current_dof] += value
    dof_states['vel'][current_dof] = (value / dt)
    gym.set_actor_dof_states(env, arm, dof_states, gymapi.STATE_POS)


# Sets a specific dof to a given value
def setDOF(gym, env, arm, dof_states, current_dof, value):
    dof_states['pos'][current_dof] = value
    dof_states['vel'][current_dof] = 0
    gym.set_actor_dof_states(env, arm, dof_states, gymapi.STATE_POS)


# Sets all dof joints to specified new values
def setAllDOF(gym, env, arm, dof_states, new_dof):
    for dof in range (6):
        dof_states['pos'][dof] = new_dof[dof]
        dof_states['vel'][dof] = 0
    gym.set_actor_dof_states(env, arm, dof_states, gymapi.STATE_POS)


# Updates all dof joints by given values
def updateAllDOF(gym, env, arm, dof_states, updates):
    for dof in range (6):
        dof_states['pos'][dof] += updates[dof]
        dof_states['vel'][dof] = (updates[dof] / dt)
    gym.set_actor_dof_states(env, arm, dof_states, gymapi.STATE_POS)


# Given a asset, this prints its DOF properties
def print_props(gym, asset):
    # num of dof for a specified asset
    num_dofs = gym.get_asset_dof_count(asset)

    # get array of DOF names
    dof_names = gym.get_asset_dof_names(asset)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(asset)

    # get the limit-related slices of the DOF properties array
    has_limits = dof_props['hasLimits']
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']

    for i in range(num_dofs):
        print("DOF %d" % i)
        print("\tName:\t\t%s" % dof_names[i])
        print(" \tLimited?\t%r" % has_limits[i])
        if has_limits[i]:
            print("\tLower\t\t%f" % lower_limits[i])
            print("\tUpper\t\t%f" % upper_limits[i])