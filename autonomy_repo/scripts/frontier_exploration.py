#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import scipy
from scipy.signal import convolve2d

from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, Bool

from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import inspect
import time

# Dontt resume detection until STOP_DETECTION_DELAY
# number of seconds elapsed from last detected
STOP_DETECTION_DELAY = 5.0

def explore(occupancy, curr_pos):
    """ returns potential states to explore
    Args:
        occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states.
        See class in first section of notebook.

    Returns:
        frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore.
        Shape is (N, 2), where N is the number of possible states to explore.

    """

    window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
    t_win_entries = window_size ** 2
    frontier_states = []

    kernel = np.ones((window_size, window_size))

    ########################### Code starts here ###########################
    unk_result = convolve2d(occupancy.probs < 0., kernel, mode='same', boundary='fill')
    occ_result = convolve2d(occupancy.probs > 0., kernel, mode='same', boundary='fill')
    free_result = convolve2d(occupancy.probs == 0., kernel, mode='same', boundary='fill')
    UNK_THRESH = 0.2 * t_win_entries
    FREE_THRESH = 0.3 * t_win_entries
    for (i, j),  nocc in np.ndenumerate(occ_result):
        nunk = unk_result[i, j]
        nfree = free_result[i,j]
        if nocc == 0 and nunk >= UNK_THRESH and nfree >= FREE_THRESH:
            frontier_states.append([i, j])
    ########################### Code ends here ###########################
    if not frontier_states:
        return None, None 

    frontier_idx =  np.array(frontier_states)
    npfrontier = occupancy.grid2state(frontier_idx)
    bcast = npfrontier - curr_pos
    norms = np.linalg.norm(bcast, axis=1, keepdims=True)
    chosen_idx = np.argmin(norms)
    return npfrontier, npfrontier[chosen_idx]


def timenow(node): return node.get_clock().now().nanoseconds/1e9

class FrontierExplorer(Node):
    """ Frontier explorer node
    """

    def __init__(self):
        super().__init__('frontier_explorer')

        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            "/map", 
            self.map_callback, 
            10
        )

        self.nav_sub = self.create_subscription(
            Bool, 
            "/nav_success", 
            self.nav_callback, 
            10
        )
        
        self.state_sub = self.create_subscription(
            TurtleBotState, 
            "/state", 
            self.state_callback, 
            10
        )

        self.state_sub = self.create_subscription(
            Bool, 
            "/detector_bool", 
            self.stop_detection_callback, 
            10
        )

        self.goal_pub = self.create_publisher(
            TurtleBotState,
            "/cmd_nav",
            10
        )

        self.declare_parameter("active", True)
        self.curr_state = None # current state
        self.occupancy = None #occupancy grid
        self._in_navigation = False
        self._stop_detected = False 
        self._last_stop_detection_time = None

    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

    @active.setter
    def active(self, new_val):
        self.set_parameters(
            [rclpy.Parameter("active", value=new_val)]
        )

    def stop_detection_callback(self, msg) -> None:
        """ Callback invoked when stop sign is detected
        """
        stop_sign_detected = msg.data
        logger = self.get_logger() 
        if stop_sign_detected:
            if not self._last_stop_detection_time:
                logger.info(f'Stop sign detected..pausing')
                self.active = False
                self._last_stop_detection_time = timenow(self)
            else:
                duration = timenow(self) - self._last_stop_detection_time
                if duration > STOP_DETECTION_DELAY:
                    logger.info(f'Stop sign detected..UNPAUSING')
                    self.active = True
                    self._last_stop_detection_time = None
        else:
            if self._last_stop_detection_time:
                duration = timenow(self) - self._last_stop_detection_time
                logger.info(f'Stop undetected for {duration=}')
                if duration < STOP_DETECTION_DELAY:
                    return
            self._last_stop_detection_time = None
            self.active = True #call setter
            

    def state_callback(self, msg) -> None:
        #self.get_logger().info(f'state callback {msg}')
        self.curr_state = np.array([msg.x, msg.y])
       
    def _publish_goal(self, chosen_state): 
        self.get_logger().info(f'Publishing new goal {chosen_state} {inspect.stack()[1][3]}')
        gpos = TurtleBotState()
        gpos.x = chosen_state[0]
        gpos.y = chosen_state[1]
        self.goal_pub.publish(gpos) 
        
    def nav_callback(self, msg: Bool) -> None:
        """ publisher Callback received from nav_success 
        """
        logger = self.get_logger() 
        occupancy, curr_state = self.occupancy, self.curr_state
        resolution = occupancy.resolution

        if not self.active:
            logger.info('Stopped exploring..In stop sign detection delay')
            return

        cs = np.array([
            occupancy.size_xy[1]-(curr_state[1]/resolution), 
            curr_state[0]/resolution
        ])
        state_xy, chosen_state = explore(occupancy, 
                occupancy.grid2state(cs)) 

        if state_xy is None:
            logger.info(f'Finished exploring..')
            return
        
        self._publish_goal(chosen_state) 


    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )
       
        logger = self.get_logger() 
        if self.curr_state is None:
            logger.info('Did not obtain current state yet')
            return 

        if self._in_navigation:
            return 

        self._in_navigation = True
        logger.info(f'Starting map exploration from {self.curr_state}')
        occupancy, curr_state = self.occupancy, self.curr_state
        resolution = occupancy.resolution

        cs = np.array([
            occupancy.size_xy[1]-(curr_state[1]/resolution), 
            curr_state[0]/resolution
        ])
        state_xy, chosen_state = explore(occupancy, 
                occupancy.grid2state(cs)) 

        if state_xy is None:
            logger.info(f'Finished exploring..')
            return

        self._publish_goal(chosen_state)
        

if __name__ == '__main__':
    rclpy.init()
    fExplorer = FrontierExplorer()
    rclpy.spin(fExplorer)
    rclpy.shutdown()
