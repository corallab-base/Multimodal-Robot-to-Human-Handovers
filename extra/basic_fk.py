# Imports
import numpy as np
from constants import DH_matrix


# Given a set of joint angles, this forward kinematics algorithm produces poses of all joints
def fk(angles):
    # initializing arrays to store current transformation and final transformation
    poses = []
    cur_transformation = np.zeros(shape=(4, 4), dtype=np.float32)
    final_transformation = np.eye(4, dtype=np.float32)

    # part of dh calculation method
    cur_transformation[3, 3] = 1.0

    # calculating each transformation one by one
    for transformation in range (0, 6):
        # getting the dh parameters
        theta = angles[transformation]
        dh_parameters = DH_matrix[transformation, :]

        # speeds up calculations by preventing redundant calculations4
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_alpha = np.sin(dh_parameters[1])
        cos_alpha = np.cos(dh_parameters[1])

        # calculations
        cur_transformation[0, 0] = cos_theta
        cur_transformation[0, 1] = -1 * sin_theta * cos_alpha
        cur_transformation[0, 2] = sin_theta * sin_alpha
        cur_transformation[0, 3] = cos_theta * dh_parameters[0]

        cur_transformation[1, 0] = sin_theta
        cur_transformation[1, 1] = cos_theta * cos_alpha
        cur_transformation[1, 2] = -1 * cos_theta * sin_alpha
        cur_transformation[1, 3] = sin_theta * dh_parameters[0]

        cur_transformation[2, 1] = sin_alpha
        cur_transformation[2, 2] = cos_alpha
        cur_transformation[2, 3] = dh_parameters[2]

        # final_transformation acts as a walking transformation collector
        final_transformation = np.matmul(final_transformation, cur_transformation)

        # saving position of joint
        poses.append(final_transformation)

    return np.array(poses)