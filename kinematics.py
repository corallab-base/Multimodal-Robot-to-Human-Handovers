# Imports
import math
import numpy as np
from trac_ik_python.trac_ik import IK
from constants import robot_rot, robot_offset
from scipy.spatial.transform import Rotation as R



# Another way to interface with fk (d(position)/d(angles))
def get_jacobian(angles, detail=.01):
    jacobian = np.zeros(shape=(3, 6))
    for dof_index in range (0, 6):
        angles1 = np.copy(angles)
        angles1[dof_index] += detail
        pos1 = get_final_position(angles1)
        angles1[dof_index] -= 2 * detail
        pos2 = get_final_position(angles1)

        diff = []
        for item1, item2 in zip(pos1, pos2):
            diff.append(item1 - item2)

        for i in range (0, 3):
            jacobian[i, dof_index] = diff[i] / (2 * detail)
            
    return jacobian


# Forward kinematics algorithm wrapper to get only the final pose
def get_final_pose(angles):
     # Getting final transformation
    all_poses = fk(angles)
    return all_poses[7]


# Forward kinematics algorithm wrapper to get only the final position
def get_final_position(angles):
    # Getting final transformation
    final_pose = get_final_pose(angles)

    # Getting translation
    return np.array(final_pose[0])
    

# forward kinematics algorithm wrapper to get only the position
def get_all_final_positions(angles):
    # getting all transformation
    all_poses = fk(angles)
    
    # finding x, y, and z of all joints
    positions = []
    for transformation in all_poses:
        # getting x, y, z
        positions.append(transformation[0])

    return positions


# Helper function for forward kinematics
def rotation_concat(quaternion1, quaternion0):
	x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
	x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]

	return [x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
			-x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
			x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
			-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0]


# Given a set of angles, calculates all the translations and rotations
def fk(dofs):
    #link1 pose
    trans1, rot1 = [0, 0, 0], [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]

    #link2 pose
    rot2_initial = [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot2_new = R.from_euler('z', dofs[0]).as_quat().tolist()
    rot2_final = rotation_concat(rot2_new, rot2_initial)
    trans2, rot2 = [0, 0, 0.1625], rot2_final

    #link3 pose
    rot3_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
    rot3_vector = R.from_quat(rot2_new).apply([0, 1, 0])
    rot3_final = rotation_concat(rot2_new, rot3_initial)
    rot3_new = R.from_rotvec(dofs[1]*rot3_vector).as_quat().tolist()
    rot3_final = rotation_concat(rot3_new, rot3_final)
    trans3, rot3 = [0, 0, 0.1625], rot3_final

    #link4 pose
    rot4_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
    rot4_vector = rot3_vector
    rot4_final = rot3_final
    rot4_offset = R.from_quat(rot3_final).apply([0, 0, 0.425])
    rot4_new = R.from_rotvec(dofs[2]*rot4_vector).as_quat().tolist()
    rot4_final = rotation_concat(rot4_new, rot4_final)
    trans4, rot4 = trans3 + rot4_offset, rot4_final

    #link5 pose
    rot5_offset = R.from_quat(rot4_final).apply([0, -0.1333, 0.3915])
    rot5_initial = [0, -0, 1, 0]
    rot5_vector = rot4_vector
    rot5_final = rotation_concat(rot2_new, rot5_initial)
    rot5_final = rotation_concat(rot3_new, rot5_final)
    rot5_final = rotation_concat(rot4_new, rot5_final)
    rot5_new = R.from_rotvec(dofs[3]*rot5_vector).as_quat().tolist()
    rot5_final = rotation_concat(rot5_new, rot5_final)
    trans5, rot5 = trans4 + rot5_offset, rot5_final

    #link6 pose
    rot6_offset = R.from_quat(rot5_final).apply([0, 0, 0])
    rot6_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot6_final = rotation_concat(rot2_new, rot6_initial)
    rot6_final = rotation_concat(rot3_new, rot6_final)
    rot6_final = rotation_concat(rot4_new, rot6_final)
    rot6_final = rotation_concat(rot5_new, rot6_final)
    rot6_vector = [0, 0, -1]
    rot6_vector = R.from_quat(rot2_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot3_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot4_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot5_new).apply(rot6_vector)
    rot6_new = R.from_rotvec(dofs[4]*rot6_vector).as_quat().tolist()
    rot6_final = rotation_concat(rot6_new, rot6_final)
    trans6, rot6 = trans5 + rot6_offset, rot6_final

    #link7 pose
    rot7_offset = R.from_quat(rot6_final).apply([0, -0.0996, 0])
    rot7_initial = [math.sqrt(2)/2, math.sqrt(2)/2, 0, 0]
    rot7_final = rotation_concat(rot2_new, rot7_initial)
    rot7_final = rotation_concat(rot3_new, rot7_final)
    rot7_final = rotation_concat(rot4_new, rot7_final)
    rot7_final = rotation_concat(rot5_new, rot7_final)
    rot7_final = rotation_concat(rot6_new, rot7_final)
    rot7_vector = [0, 1, 0]
    rot7_vector = R.from_quat(rot2_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot3_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot4_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot5_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot6_new).apply(rot7_vector)
    rot7_new = R.from_rotvec(dofs[5]*rot7_vector).as_quat().tolist()
    rot7_final = rotation_concat(rot7_new, rot7_final)
    trans7, rot7 = trans6 + rot7_offset, rot7_final

    # robotiq_85_base_link_coarse pose
    rot8_offset = R.from_quat(rot7_final).apply([0.094, 0, 0.0])
    rot8_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot8_final = rotation_concat(rot2_new, rot8_initial)
    rot8_final = rotation_concat(rot3_new, rot8_final)
    rot8_final = rotation_concat(rot4_new, rot8_final)
    rot8_final = rotation_concat(rot5_new, rot8_final)
    rot8_final = rotation_concat(rot6_new, rot8_final)
    rot8_final = rotation_concat(rot7_new, rot8_final)
    rot8_new = rot7_new
    rot8_final = rotation_concat(rot8_new, rot8_final)
    trans8, rot8 = trans7 + rot8_offset, rot8_final

    # left inner knuckle pose
    rot9_offset = R.from_quat(rot8_final).apply([0.0127000000001501, 0, 0.0693074999999639])
    rot9_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot9_final = rotation_concat(rot2_new, rot9_initial)
    rot9_final = rotation_concat(rot3_new, rot9_final)
    rot9_final = rotation_concat(rot4_new, rot9_final)
    rot9_final = rotation_concat(rot5_new, rot9_final)
    rot9_final = rotation_concat(rot6_new, rot9_final)
    rot9_final = rotation_concat(rot7_new, rot9_final)
    rot9_final = rotation_concat(rot8_new, rot9_final)
    rot9_vector = [0, 0, -1]
    rot9_vector = R.from_quat(rot2_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot3_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot4_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot5_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot6_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot7_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot8_new).apply(rot9_vector)
    rot9_new = R.from_rotvec(dofs[6]*rot9_vector).as_quat().tolist()
    rot9_final = rotation_concat(rot9_new, rot9_final)
    trans9, rot9 = trans8 + rot9_offset, rot9_final

    # left inner finger pose
    rot10_offset = R.from_quat(rot9_final).apply([0.034585310861294, 0, 0.0454970193817975])
    rot10_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot10_final = rotation_concat(rot2_new, rot10_initial)
    rot10_final = rotation_concat(rot3_new, rot10_final)
    rot10_final = rotation_concat(rot4_new, rot10_final)
    rot10_final = rotation_concat(rot5_new, rot10_final)
    rot10_final = rotation_concat(rot6_new, rot10_final)
    rot10_final = rotation_concat(rot7_new, rot10_final)
    rot10_final = rotation_concat(rot8_new, rot10_final)
    rot10_final = rotation_concat(rot9_new, rot10_final)
    rot10_vector = [0, 0, -1]
    rot10_vector = R.from_quat(rot2_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot3_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot4_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot5_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot6_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot7_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot8_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot9_new).apply(rot10_vector)
    rot10_new = R.from_rotvec(dofs[7]*rot10_vector).as_quat().tolist()
    rot10_final = rotation_concat(rot10_new, rot10_final)
    trans10, rot10 = trans9 + rot10_offset, rot10_final

    # left outer knuckle pose
    rot11_offset = R.from_quat(rot8_final).apply([0.0306011444260539, 0, 0.0627920162695395])
    rot11_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot11_final = rotation_concat(rot2_new, rot11_initial)
    rot11_final = rotation_concat(rot3_new, rot11_final)
    rot11_final = rotation_concat(rot4_new, rot11_final)
    rot11_final = rotation_concat(rot5_new, rot11_final)
    rot11_final = rotation_concat(rot6_new, rot11_final)
    rot11_final = rotation_concat(rot7_new, rot11_final)
    rot11_final = rotation_concat(rot8_new, rot11_final)
    rot11_vector = [0, 0, -1]
    rot11_vector = R.from_quat(rot2_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot3_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot4_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot5_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot6_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot7_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot8_new).apply(rot11_vector)
    rot11_new = R.from_rotvec(dofs[8]*rot11_vector).as_quat().tolist()
    rot11_final = rotation_concat(rot11_new, rot11_final)
    trans11, rot11 = trans8 + rot11_offset, rot11_final

    # right inner knuckle pose
    rot12_offset = R.from_quat(rot8_final).apply([-0.0126999999998499, 0, 0.0693075000000361])
    rot12_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot12_final = rotation_concat(rot2_new, rot12_initial)
    rot12_final = rotation_concat(rot3_new, rot12_final)
    rot12_final = rotation_concat(rot4_new, rot12_final)
    rot12_final = rotation_concat(rot5_new, rot12_final)
    rot12_final = rotation_concat(rot6_new, rot12_final)
    rot12_final = rotation_concat(rot7_new, rot12_final)
    rot12_final = rotation_concat(rot8_new, rot12_final)
    rot12_vector = [0, 0, -1]
    rot12_vector = R.from_quat(rot2_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot3_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot4_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot5_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot6_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot7_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot8_new).apply(rot12_vector)
    rot12_new = R.from_rotvec(dofs[9]*rot12_vector).as_quat().tolist()
    rot12_final = rotation_concat(rot12_new, rot12_final)
    trans12, rot12 = trans8 + rot12_offset, rot12_final

    # right inner finger pose
    rot13_offset = R.from_quat(rot12_final).apply([0.0341060475457406, 0, 0.0458573878541688])
    rot13_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot13_final = rotation_concat(rot2_new, rot13_initial)
    rot13_final = rotation_concat(rot3_new, rot13_final)
    rot13_final = rotation_concat(rot4_new, rot13_final)
    rot13_final = rotation_concat(rot5_new, rot13_final)
    rot13_final = rotation_concat(rot6_new, rot13_final)
    rot13_final = rotation_concat(rot7_new, rot13_final)
    rot13_final = rotation_concat(rot8_new, rot13_final)
    rot13_final = rotation_concat(rot12_new, rot13_final)
    rot13_vector = [0, 0, -1]
    rot13_vector = R.from_quat(rot2_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot3_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot4_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot5_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot6_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot7_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot8_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot12_new).apply(rot13_vector)
    rot13_new = R.from_rotvec(dofs[10]*rot13_vector).as_quat().tolist()
    rot13_final = rotation_concat(rot13_new, rot13_final)
    trans13, rot13 = trans12 + rot13_offset, rot13_final

    # right outer knuckle pose
    rot14_offset = R.from_quat(rot8_final).apply([-0.0306011444258893, 0, 0.0627920162695395])
    rot14_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot14_final = rotation_concat(rot2_new, rot14_initial)
    rot14_final = rotation_concat(rot3_new, rot14_final)
    rot14_final = rotation_concat(rot4_new, rot14_final)
    rot14_final = rotation_concat(rot5_new, rot14_final)
    rot14_final = rotation_concat(rot6_new, rot14_final)
    rot14_final = rotation_concat(rot7_new, rot14_final)
    rot14_final = rotation_concat(rot8_new, rot14_final)
    rot14_vector = [0, 0, -1]
    rot14_vector = R.from_quat(rot2_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot3_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot4_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot5_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot6_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot7_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot8_new).apply(rot14_vector)
    rot14_new = R.from_rotvec(dofs[11]*rot14_vector).as_quat().tolist()
    rot14_final = rotation_concat(rot14_new, rot14_final)
    trans14, rot14 = trans8 + rot14_offset, rot14_final

    # left outer finger pose
    rot15_offset = R.from_quat(rot11_final).apply([0.0316910442266543, 0, -0.00193396375724605])
    rot15_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot15_final = rotation_concat(rot2_new, rot15_initial)
    rot15_final = rotation_concat(rot3_new, rot15_final)
    rot15_final = rotation_concat(rot4_new, rot15_final)
    rot15_final = rotation_concat(rot5_new, rot15_final)
    rot15_final = rotation_concat(rot6_new, rot15_final)
    rot15_final = rotation_concat(rot7_new, rot15_final)
    rot15_final = rotation_concat(rot8_new, rot15_final)
    rot15_final = rotation_concat(rot11_new, rot15_final)
    rot15_new = rot11_new
    rot15_final = rotation_concat(rot15_new, rot15_final)
    trans15, rot15 = trans11 + rot15_offset, rot15_final

    # right outer finger pose
    rot16_offset = R.from_quat(rot14_final).apply([0.0317095909367246, 0, -0.0016013564954687])
    rot16_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot16_final = rotation_concat(rot2_new, rot16_initial)
    rot16_final = rotation_concat(rot3_new, rot16_final)
    rot16_final = rotation_concat(rot4_new, rot16_final)
    rot16_final = rotation_concat(rot5_new, rot16_final)
    rot16_final = rotation_concat(rot6_new, rot16_final)
    rot16_final = rotation_concat(rot7_new, rot16_final)
    rot16_final = rotation_concat(rot8_new, rot16_final)
    rot16_final = rotation_concat(rot14_new, rot16_final)
    rot16_new = rot14_new
    rot16_final = rotation_concat(rot16_new, rot16_final)
    trans16, rot16 = trans14 + rot16_offset, rot16_final

    transformations = [[trans1, rot1],
            [trans2, rot2],
            [trans3, rot3],
            [trans4, rot4],
            [trans5, rot5],
            [trans6, rot6],
            [trans7, rot7],
            [trans8, rot8],
            [trans9, rot9],
            [trans10, rot10],
            [trans11, rot11],
            [trans12, rot12],
            [trans13, rot13],
            [trans14, rot14],
            [trans15, rot15],
            [trans16, rot16]]
    
    transformations_with_offset = []
    for (translation, rotation) in transformations:
        translation_with_offset = [val + offset for val, offset in zip(translation, robot_offset)]
        transformations_with_offset.append(np.array([translation_with_offset, rotation]))

    return np.array(transformations_with_offset)

    ans = np.array([[trans1, rot1],
            [trans2, rot2],
            [trans3, rot3],
            [trans4, rot4],
            [trans5, rot5],
            [trans6, rot6],
            [trans7, rot7],
            [trans8, rot8],
            [trans9, rot9],
            [trans10, rot10],
            [trans11, rot11],
            [trans12, rot12],
            [trans13, rot13],
            [trans14, rot14],
            [trans15, rot15],
            [trans16, rot16]])
    
    return ans
    

# Inverse kinematics solver class
class ik():
    def __init__(self):
        # Getting the URDF file
        urdf_str = ''.join(open('./assets/urdf/ur5e_mimic_real.urdf', 'r').readlines())
    
        # Building the ik solver by specifying start and end links
        #Hanwen: the link on the second argement is wrong previously
        self.ik_solver = IK('base_link', 'wrist_3_link', urdf_string=urdf_str)

        # setting upper and lower bounds
        lower_bound, upper_bound = self.ik_solver.get_joint_limits()
        self.ik_solver.set_joint_limits(lower_bound, upper_bound)

        # Saving the seed angle
        self.seed_angle = [0.0] * 6


    def get_IK(self, pos, quat):
        # Getting quat information
        q_x, q_y, q_z, q_w = quat[1], quat[2], quat[3], quat[0]

        # subtracting the robot offset
        pos = [pos_x - off_x for pos_x, off_x in zip(pos, robot_offset)]

        # transforming the points into robot coordinate system
        pos = np.array(pos)
        rotation = R.from_quat(robot_rot).as_matrix()
        pos = np.matmul(np.linalg.inv(rotation), pos)
        #Hanwen: the rotation in the global coordinate need to be transformed as well just like the pos
        q_x, q_y, q_z, q_w = rotation_concat([-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2], quat)

        # getting the solutions
        # for k in range(1000):
        solutions = self.ik_solver.get_ik(self.seed_angle, pos[0], pos[1], pos[2], q_x, q_y, q_z, q_w,
                                          0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
            # print([round(x,2) for x in np.degrees(solutions)])
            # print(pos)
        return solutions
    

# Creating a global IK solver for memory efficiency
ik_solver = ik()