# Imports
import fcl
import numpy as np
from math import sqrt, pi
from isaacgym import gymapi 


# Control variables
testing_options = ["none", "rmp", "ompl"]
testing = testing_options[0]
save_paths = True
num_iters = 20

want_obtacles = False
graph_task_space = False

object_name = "013_apple"


# Global Variables
blend_radius = .001
dt = 1.0 / 60.0
real_time_rendering = True


# Camera settings
camera_position = gymapi.Vec3(1.35, 0.0, 1.04)
# camera_position = gymapi.Vec3(1.65, 0.1, 1.04)
camera_target = gymapi.Vec3(0.0, 0.0, 1.04)


# Hyperparameters (need to tune)
# "c" determines normalization behavior from regular to soft
# "alpha" determines force proportional to error size
# "beta" Determines dampening proportional to velocity
c = .1
alpha = 1
beta = 1
w = 1
s = 1


# Camera placement configuration for CoGrasp
K = np.array([[911.445649104, 0, 641.169], [0, 891.51236121, 352.77], [0, 0, 1]])


# Positioning the obstacle
table_dims = gymapi.Vec3(0.92, 0.92, 0.75)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.46, 0.0, 0.5 * table_dims.z)


# Determines initial transformation of the robot arm
robot_offset = [-0.4125, 0, 0.7]
# robot_offset = [-0.4125, 0, 0.92]
# robot_offset = [-0.25, 0, 0.7]
# robot_rot = [0.0, -sqrt(2) / 2, -sqrt(2) / 2, 0]
robot_rot = [sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2]
initial_dof = [0.0, -pi / 2, pi / 2, -pi / 2, -pi / 2, 0.0]



# Creates a plane mesh using fcl
normal = np.array([0, 0, 1]).astype('double')
plane = fcl.Plane(normal, 0.0)
transform = fcl.Transform(robot_rot, [0, 0, 0])
plane = fcl.CollisionObject(plane, transform)


# For image segmentation
object_segment_dict = {'table': 1, 'ur5e': 2, 'banana': 3, 'foam_brick': 4, 'mug': 5, 'chips_can': 6, 'cracker_box': 7, 'potted_meat_can': 8, 'master_chef_can': 9, 'sugar_box': 10, 'mustard_bottle': 11,
'tomato_soup_can': 12, 'tuna_fish_can': 13, 'pudding_box': 14, 'strawberry': 15, 'gelatin_box': 16, 'lemon': 17, 'apple': 18, 'peach': 19, 'orange': 20, 'pear': 21, 'plum': 22, 'pitcher_base': 23, 'bleach_cleanser': 24,
'bowl': 25, 'sponge': 26, 'fork': 27, 'spoon': 28, 'knife': 29, 'power_drill': 30, 'wood_block': 31, 'scissors': 32, 'padlock': 33, 'large_marker': 34, 'small_marker': 35, 'phillips_screwdriver': 36, 'flat_screwdriver': 37,
'hammer': 38, 'medium_clamp': 39, 'large_clamp': 40, 'extra_large_clamp': 41, 'mini_soccer_ball': 42, 'softball': 43, 'baseball': 44, 'tennis_ball': 45, 'racquetball': 46, 'golf_ball': 47, 'chain': 48, 'dice': 49, 'a_marbles': 50,
'd_marbles': 51, 'a_cups': 52, 'b_cups': 53, 'c_cups': 54, 'd_cups': 55, 'e_cups': 56, 'f_cups': 57, 'h_cups': 58, 'g_cups': 59, 'i_cups': 60, 'j_cups': 61, 'a_colored_wood_blocks': 62, 'nine_hole_peg_test': 63, 'a_toy_airplane': 64,
'b_toy_airplane': 65, 'c_toy_airplane': 66, 'd_toy_airplane': 67,'e_toy_airplane': 68, 'f_toy_airplane': 69, 'h_toy_airplane': 70, 'i_toy_airplane': 71, 'j_toy_airplane': 72, 'k_toy_airplane': 73, 'a_lego_duplo': 74,
'b_lego_duplo': 75, 'c_lego_duplo': 76, 'd_lego_duplo': 77, 'e_lego_duplo': 78, 'f_lego_duplo': 79, 'g_lego_duplo': 80, 'h_lego_duplo': 81, 'i_lego_duplo': 82, 'j_lego_duplo': 83, 'k_lego_duplo': 84, 'l_lego_duplo': 85, 'm_lego_duplo': 86,
'timer': 87, 'rubiks_cube': 88}

object_rotations = {'table': [], 'ur5e': [], 'banana': [], 'foam_brick': [], 'mug': [], 'chips_can': [], 'cracker_box': [], 'potted_meat_can': [], 'master_chef_can': [], 'sugar_box': [], 'mustard_bottle': [],
'tomato_soup_can': [], 'tuna_fish_can': [], 'pudding_box': [], 'strawberry': [], 'gelatin_box': [], 'lemon': [], 'apple': [], 'peach': [], 'orange': [], 'pear': [], 'plum': [], 'pitcher_base': [], 'bleach_cleanser': [],
'bowl': [], 'sponge': [], 'fork': [], 'spoon': [], 'knife': [], 'power_drill': [], 'wood_block': [], 'scissors': [], 'padlock': [], 'large_marker': [], 'small_marker': [], 'phillips_screwdriver': [], 'flat_screwdriver': [],
'hammer': [], 'medium_clamp': [], 'large_clamp': [], 'extra_large_clamp': [], 'mini_soccer_ball': [], 'softball': [], 'baseball': [], 'tennis_ball': [], 'racquetball': [], 'golf_ball': [], 'chain': [], 'dice': [], 'a_marbles': [],
'd_marbles': [], 'a_cups': [], 'b_cups': [], 'c_cups': [], 'd_cups': [], 'e_cups': [], 'f_cups': [], 'h_cups': [],'g_cups': [], 'i_cups': [], 'j_cups': [], 'a_colored_wood_blocks': [], 'nine_hole_peg_test': [], 'a_toy_airplane': [],
'b_toy_airplane': [], 'c_toy_airplane': [], 'd_toy_airplane': [], 'e_toy_airplane': [], 'f_toy_airplane': [], 'h_toy_airplane': [], 'i_toy_airplane': [], 'j_toy_airplane': [], 'k_toy_airplane': [], 'a_lego_duplo': [],
'b_lego_duplo': [], 'c_lego_duplo': [], 'd_lego_duplo': [], 'e_lego_duplo': [], 'f_lego_duplo': [], 'g_lego_duplo': [], 'h_lego_duplo': [], 'i_lego_duplo': [], 'j_lego_duplo': [], 'k_lego_duplo': [], 'l_lego_duplo': [], 'm_lego_duplo': [],
'timer': [], 'rubiks_cube': []}

object_translations = {'table': [], 'ur5e': [], 'banana': [], 'foam_brick': [], 'mug': [], 'chips_can': [], 'cracker_box': [], 'potted_meat_can': [], 'master_chef_can': [], 'sugar_box': [], 'mustard_bottle': [],
'tomato_soup_can': [], 'tuna_fish_can': [], 'pudding_box': [], 'strawberry': [], 'gelatin_box': [], 'lemon': [], 'apple': [], 'peach': [], 'orange': [], 'pear': [], 'plum': [], 'pitcher_base': [], 'bleach_cleanser': [],
'bowl': [], 'sponge': [], 'fork': [], 'spoon': [], 'knife': [], 'power_drill': [], 'wood_block': [], 'scissors': [], 'padlock': [], 'large_marker': [], 'small_marker': [], 'phillips_screwdriver': [], 'flat_screwdriver': [],
'hammer': [], 'medium_clamp': [], 'large_clamp': [], 'extra_large_clamp': [], 'mini_soccer_ball': [], 'softball': [], 'baseball': [], 'tennis_ball': [], 'racquetball': [], 'golf_ball': [], 'chain': [], 'dice': [], 'a_marbles': [],
'd_marbles': [], 'a_cups': [], 'b_cups': [], 'c_cups': [], 'd_cups': [], 'e_cups': [], 'f_cups': [], 'h_cups': [], 'g_cups': [], 'i_cups': [], 'j_cups': [], 'a_colored_wood_blocks': [], 'nine_hole_peg_test': [], 'a_toy_airplane': [],
'b_toy_airplane': [], 'c_toy_airplane': [], 'd_toy_airplane': [], 'e_toy_airplane': [], 'f_toy_airplane': [], 'h_toy_airplane': [], 'i_toy_airplane': [], 'j_toy_airplane': [], 'k_toy_airplane': [], 'a_lego_duplo': [],
'b_lego_duplo': [], 'c_lego_duplo': [], 'd_lego_duplo': [], 'e_lego_duplo': [], 'f_lego_duplo': [], 'g_lego_duplo': [], 'h_lego_duplo': [], 'i_lego_duplo': [], 'j_lego_duplo': [], 'k_lego_duplo': [], 'l_lego_duplo': [], 'm_lego_duplo': [],
'timer': [], 'rubiks_cube': []}