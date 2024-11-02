#!/usr/bin/env python3

import numpy 
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from asl_tb3_lib.control import BaseController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class PerceptionController(BaseController):
    """ Heading controller
    """
    def __init__(self):
        super().__init__('perception_controller')
        self.declare_parameter("active", True)
        self.clock_start = False

    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

    @active.setter
    def active(self, new_active):
        self.set_parameters(
            [self.get_parameter("active").set_value(new_active)]
        )

    def compute_control(self) -> TurtleBotControl:
        """ compute control with goal 
        """
        rctrl = TurtleBotControl()
        is_active = self.active
        logger = self.get_logger()

        #self.get_logger().info(f'Called compute_control {is_active=}')
        if is_active:
            rctrl.v = 0.
            rctrl.omega = 0.5
        else:
            rctrl.v = 0.
            rctrl.omega = 0.
            if not self.clock_start:
                self.tstart = self.get_clock().now().nanoseconds / 1e9
                self.clock_start = True

            tnow = self.get_clock().now().nanoseconds / 1e9
            duration = tnow - self.tstart
            logger.info(f'False path invoked {duration=}{self.active=}')
            if duration > 5:
                self.set_parameters(
                    [rclpy.Parameter("active", value=True)]
                )
                self.clock_start = False
                logger.info(f'Reset active to {self.active=}')

        return rctrl


if __name__ == '__main__':
    rclpy.init()
    p_controller = PerceptionController()
    rclpy.spin(p_controller)
    rclpy.shutdown()

