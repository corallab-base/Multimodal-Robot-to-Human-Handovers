# Executes the RMP based path planning
import numpy as np

real_time_rendering = False

from kinematics import get_final_position, get_jacobian
    
from rfm import DynamicObjectRMP
from isaacgym import gymapi, gymutil
from math import pi, sqrt

from constants import dt, robot_rot, object_segment_dict, robot_offset, camera_position, camera_target, initial_dof

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
    
# Sets all dof joints to specified new values
def setAllDOF(gym, env, arm, dof_states, new_dof):
    for dof in range (6):
        dof_states['pos'][dof] = new_dof[dof]
        dof_states['vel'][dof] = 0
    gym.set_actor_dof_states(env, arm, dof_states, gymapi.STATE_POS)



# Defining an RMP class
class RMP:
    # Constructor
    def __init__(self, target_rmp):
        self.target_rmp = target_rmp


    # Evaluates the value of a function at a given point
    def eval_function(self, pos, vel, static_points, dynamic_points):
        # Reshaping to numpy standards
        pos = np.reshape(pos, newshape=(3, 1))
        vel = np.reshape(vel, newshape=(3, 1))

        # Evaluating all the static RMPs
        rmp_evals = (1 + len(static_points) + len(dynamic_points)) * [0]
        rmp_evals[0] = self.target_rmp.evaluate(pos, vel)

        # Evaluating all the avoid points' RMP
        index = 1
        dynamic_rmp = DynamicObjectRMP()
        while index <= len(static_points):
            rmp_evals[index] = dynamic_rmp.evaluate(pos, vel, static_points[index - 1])
            index += 1

        while index <= len(static_points) + len(dynamic_points):
            rmp_evals[index] = dynamic_rmp.evaluate(pos, vel, dynamic_points[index - len(static_points) - 1])
            index += 1

        # Calculating the rm and rf sums
        sum_of_riemannian_metrics = np.zeros(shape=(3, 3))
        sum_of_riemannian_metrics_func = np.zeros(shape=(3,1))
        for index in range (0, len(rmp_evals)):
            sum_of_riemannian_metrics += rmp_evals[index][0]
            # print("Starting:")
            # print(rmp_evals[index][0], ", ", rmp_evals[index][1])
            # print(np.matmul(rmp_evals[index][0], rmp_evals[index][1]))
            sum_of_riemannian_metrics_func += np.matmul(rmp_evals[index][0], rmp_evals[index][1])
        
        # Multiplying sum of metrics pseudoinverse with sum of evaluated metrics and functions
        sum_of_riemannian_metrics_func = np.matmul(np.linalg.pinv(sum_of_riemannian_metrics), sum_of_riemannian_metrics_func)

        return sum_of_riemannian_metrics_func
    

    # Pull function for config --> task space
    def pull(self, pos, vel, static_points=[], dynamic_points=[]):
        epsilon = .005
        jacobian_mat = get_jacobian(pos, epsilon)
        
        jacobian_mat_pinv = np.linalg.pinv(jacobian_mat)
        func_res = self.eval_function(get_final_position(pos), vel, static_points, dynamic_points)

        func_result = np.matmul(jacobian_mat_pinv, func_res)
        return func_result

from constants import s, w, alpha, beta, c, dt
from util import soft_normalization, get_robot_points, distance

# Class that acts as the riemannian metric policy for end effector reaching target RMP
class TargetRMP:
    # Constructor
    def __init__(self, target_pos):
        self.target_pos = np.reshape(target_pos, newshape=(3, 1))


    # Evaluates both the metric and function
    def evaluate(self, end_effector_pos, end_effector_vel):
        return self.evaluate_metric(), self.evaluate_func(end_effector_pos, end_effector_vel)


    # Evaluates the riemannian metric at a given position and velocity
    def evaluate_metric(self):
        return np.eye(3)


    # Evaluates the function at a given position and velocity
    def evaluate_func(self, end_effector_pos, end_effector_vel):
        difference = self.target_pos - end_effector_pos
        return (alpha * soft_normalization(difference, c)) - (beta * end_effector_vel)


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
        # gym.step_graphics(sim)
        # gym.draw_viewer(viewer, sim, True)

        # render the camera sensors
        # gym.render_all_camera_sensors(sim)

        # Synchronizes the physics simulation with the rendering rate.
        if (real_time_rendering):
            gym.sync_frame_time(sim)

