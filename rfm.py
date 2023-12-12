# Imports
import numpy as np
from constants import s, w, alpha, beta, c
from util import soft_normalization


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


# Class that acts as the riemannian metric for end effector avoiding other points
class DynamicObjectRMP:
    # Constructor
    def __init__(self):
        self.dynamic = True


    # Evaluates both the metric and function
    def evaluate(self, end_effector_pos, end_effector_vel, dynamic_point_pos):
        acc = self.evaluate_func(end_effector_pos, end_effector_vel, dynamic_point_pos)
        return self.evaluate_metric(end_effector_pos, dynamic_point_pos, acc), acc


    # Evaluates the riemannian metric at a given position and velocity
    def evaluate_metric(self, end_effector_pos, dynamic_point_pos, acc):
        diff = np.linalg.norm(end_effector_pos - dynamic_point_pos)
        result = (w * diff) * np.matmul((s * acc), (s * np.transpose(acc)))
        return np.array(result)


    # Evaluates the function at a given position and velocity
    def evaluate_func(self, end_effector_pos, end_effector_vel, dynamic_point_pos):
        diff = end_effector_pos - dynamic_point_pos
        dist = np.linalg.norm(diff)
        unit_difference = diff / dist

        result = (alpha * unit_difference)
        result -= (beta * np.matmul(np.matmul(unit_difference, np.transpose(unit_difference)), end_effector_vel))
        result *= dist

        return result