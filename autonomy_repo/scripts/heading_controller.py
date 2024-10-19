#!/usr/bin/env python3

import numpy 
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    """ Heading controller
    """
    def __init__(self):
        super().__init__()
        self.declare_parameter("kp", 2.0)

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value


    def compute_control_with_goal(self, 
            currstate: TurtleBotState, 
            goalstate: TurtleBotState) -> TurtleBotControl:
        """ compute control with goal 
        """
        herr = wrap_angle(goalstate.theta - currstate.theta)
        ang_velocity = herr * self.kp
        retCtrl = TurtleBotControl()
        retCtrl.omega = ang_velocity
        return retCtrl


if __name__ == '__main__':
    rclpy.init()
    hcontroller = HeadingController()
    rclpy.spin(hcontroller)
    rclpy.shutdown()

