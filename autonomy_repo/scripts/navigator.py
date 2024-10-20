#!/usr/bin/env python3

import scipy
from scipy.interpolate import splev
import typing as T
import numpy 
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo
        self.statespace_hi = statespace_hi
        self.occupancy = occupancy           # occupancy grid (a StochOccupancyGrid2D object)
        self.resolution = resolution         # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if x==self.x_init or x==self.x_goal:
            return True
        for dim in range(len(x)):
            if x[dim] < self.statespace_lo[dim]:
                return False
            if x[dim] > self.statespace_hi[dim]:
                return False
        if not self.occupancy.is_free(np.array(x)):
            return False
        return True
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance
        """
        return np.linalg.norm(np.array(x1)-np.array(x2))

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES
        """
        neighbors = []
        for dx1 in [-self.resolution, 0, self.resolution]:
            for dx2 in [-self.resolution, 0, self.resolution]:
                if dx1==0 and dx2==0:
                    # don't include itself
                    continue
                new_x = (x[0]+dx1,x[1]+dx2)
                if self.is_free(new_x):
                    neighbors.append(self.snap_to_grid(new_x))
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found
        """
        while len(self.open_set)>0:
            current = self.find_best_est_cost_through()
            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(current)
            self.closed_set.add(current)
            for n in self.get_neighbors(current):
                if n in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[current] + self.distance(current,n)
                if n not in self.open_set:
                    self.open_set.add(n)
                elif tentative_cost_to_arrive >= self.cost_to_arrive[n]:
                    continue
                self.came_from[n] = current
                self.cost_to_arrive[n] = tentative_cost_to_arrive
                self.est_cost_through[n] = self.cost_to_arrive[n] + self.distance(n,self.x_goal)
        return False

class TurtleBotNavigator(BaseNavigator):
    """ Heading controller
    """
    def __init__(self):
        super().__init__()

        # heading direction param
        self.declare_parameter("kp", 2.0)

        # trajectory params closed loop
        self.V_PREV_THRES = 0.0001
        self.declare_parameter("kpx", 2.0)
        self.declare_parameter("kpy", 2.0)
        self.declare_parameter("kdx", 2.0)
        self.declare_parameter("kdy", 2.0)

        self.reset()

    def reset(self):
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

    @property
    def kpx(self) -> float:
        return self.get_parameter("kpx").value
   
    @property
    def kpy(self) -> float:
        return self.get_parameter("kpy").value

    @property
    def kdx(self) -> float:
        return self.get_parameter("kdx").value
    
    @property
    def kdy(self) -> float:
        return self.get_parameter("kdy").value

    def compute_heading_control(self, 
            currstate: TurtleBotState, 
            goalstate: TurtleBotState) -> TurtleBotControl:
        """ compute control with goal 
        """
        kp = self.kp
        herr = wrap_angle(goalstate.theta - currstate.theta)
        ang_velocity = herr * kp
        retCtrl = TurtleBotControl()
        retCtrl.omega = ang_velocity
        return retCtrl

    def get_desired_state(self, 
            t: float, 
            plan : TrajectoryPlan):
        """ uses splev to get desired derivates in x, y direction
        """
        xd_d = float(splev(t, plan.path_x_spline, der=1))
        yd_d = float(splev(t, plan.path_y_spline, der=1))
        x_d = float(splev(t, plan.path_x_spline, der=0))
        y_d = float(splev(t, plan.path_y_spline, der=0))
        xdd_d = float(splev(t, plan.path_x_spline, der=2))
        ydd_d = float(splev(t, plan.path_y_spline, der=2))
        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d


    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        """ compute control target like in hw2, p2_trajectory_tracking
            Use the following hints as a guide:
        """
        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = \
                self.get_desired_state(t, plan)
        x, y, th = state.x, state.y, state.theta
        
        V_PREV_THRES = self.V_PREV_THRES

        if self.V_prev < V_PREV_THRES:
            self.V_prev = V_PREV_THRES
        ########## Code starts here ##########
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - self.V_prev * np.cos(th))
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - self.V_prev * np.sin(th))
        
        J = [[np.cos(th), -self.V_prev * np.sin(th)], [np.sin(th), self.V_prev * np.cos(th)]]

        print(J) 
        [a, om] = np.linalg.solve(J, [u1, u2])
        V = self.V_prev + a * dt

        control = TurtleBotControl()
        control.v = V
        control.omega = om
        ########## Code ends here ##########

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        return control 


    def compute_smooth_plan(self, path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
        path = np.asarray(path)

        # Compute and set the following variables:
        #   1. ts:
        #      Compute an array of time stamps for each planned waypoint assuming some constant
        #      velocity between waypoints.
        #   2. path_x_spline, path_y_spline:
        #      Fit cubic splines to the x and y coordinates of the path separately
        #      with respect to the computed time stamp array.

        ts_n = np.shape(path)[0]
        ts = np.zeros(ts_n)
        for i in range(ts_n-1):
            ts[i+1] = np.linalg.norm(path[i+1] - path[i]) / v_desired
            ts[i+1] = ts[i+1] + ts[i]
        path_x_spline = scipy.interpolate.splrep(ts, path[: ,0], k=3, s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts, path[: ,1], k=3, s=spline_alpha)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )

    
    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        astar = AStar(
            (state.x, state.y), 
            (state.x + horizon, state.y + horizon), 
            (state.x, state.y), 
            (goal.x, goal.y),
            occupancy, 
            resolution
        )

        status = astar.solve()
        if not status or len(astar.path) < 4:
            print('path finding failed..')
            return None

        self.reset()
        return self.compute_smooth_plan(astar.path)


if __name__ == '__main__':
    rclpy.init()
    tbnavigator = TurtleBotNavigator()
    rclpy.spin(tbnavigator)
    rclpy.shutdown()
