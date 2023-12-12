# Importing the OMPL collision management system
import sys
sys.path.append('/home/corallab/Downloads/ompl/py-bindings')
import ompl.base as ob
import ompl.geometric as og
from robot_config import rac



# OMPL Path Planner
class path_planner():
    # Interfacing sub-class for OMPL path planning
    class ur5e_valid(ob.StateValidityChecker):
        def __init__(self, si, static_env_models):
            super().__init__(si)
            self.static_env_models_ = static_env_models

        def isValid(self, dof_state):
            return not rac.has_collisions(dof_state, self.static_env_models_)
        

    # Initializes dof space
    def __init__(self, static_env_models):
        self.space_ = ob.RealVectorStateSpace(0)
        self.space_.addDimension(-3.14, 3.14)
        self.space_.addDimension(-3.14, 3.14)
        self.space_.addDimension(-3.14, 3.14)
        self.space_.addDimension(-3.14, 3.14)
        self.space_.addDimension(-3.14, 3.14)
        self.space_.addDimension(-3.14, 3.14)

        self.si_ = ob.SpaceInformation(self.space_)
        self.static_env_models_ = static_env_models


    # Plans from start to end points (returns only critical points)
    def plan(self, dof_start, dof_result):
        self.start_ = ob.State(self.space_)
        self.start_[0] = dof_start[0] 
        self.start_[1] = dof_start[1] 
        self.start_[2] = dof_start[2] 
        self.start_[3] = dof_start[3] 
        self.start_[4] = dof_start[4] 
        self.start_[5] = dof_start[5] 

        self.goal_ = ob.State(self.space_)
        self.goal_[0] = dof_result[0]
        self.goal_[1] = dof_result[1]
        self.goal_[2] = dof_result[2]
        self.goal_[3] = dof_result[3]
        self.goal_[4] = dof_result[4]
        self.goal_[5] = dof_result[5]

        validityChecker = self.ur5e_valid(self.si_, self.static_env_models_,)
        self.si_.setStateValidityChecker(validityChecker)
        self.si_.setStateValidityCheckingResolution(0.001)
        self.si_.setup()


        pdef = ob.ProblemDefinition(self.si_)
        pdef.setStartAndGoalStates(self.start_, self.goal_)
        pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(self.si_))

        optimizingPlanner = og.RRTConnect(self.si_)
        optimizingPlanner.setProblemDefinition(pdef)

        optimizingPlanner.setRange(1000000)
        optimizingPlanner.setup()
        
        temp_res = optimizingPlanner.solve(1)

        if temp_res.asString() == 'Exact solution':
            path = pdef.getSolutionPath()
            path_simp = og.PathSimplifier(self.si_)
            res = path_simp.reduceVertices(path)

            path_list = []
            for t in range(path.getStateCount()):
                state = path.getState(t)
                path_list.append([state[0], state[1], state[2], state[3], state[4], state[5]])

            return path_list
        else:
            return None