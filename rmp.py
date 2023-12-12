# Imports
import numpy as np
from kinematics import get_final_position, get_jacobian
from rfm import DynamicObjectRMP


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