#!/usr/bin/env python3

import numpy 
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from asl_tb3_lib.control import BaseController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

ANG_VELOCITY = 0.5
PAUSE_TIMEOUT = 5.0 # how long to pause 
STOP_DETECTION_DELAY = 5.0 # dont resume detection until these number of seconds elapsed from last detected

def timenow(node): return node.get_clock().now().nanoseconds/1e9

class PerceptionController(BaseController):
    """ Heading controller
    """
    def __init__(self):
        super().__init__('perception_controller')
        self.declare_parameter("active", True)
        self.clock_start = False
        self._stop_detected = False 
        self._last_stop_detection_time = None

        self.detector_subscription = self.create_subscription(
            Bool,          # Message type
            'detector_bool',       # Topic name
            self.stop_cb,  # Callback function
            10               # QoS profile (queue size)
        )

    def stop_cb(self, msg: Bool) -> None:
        """  callback received on detector_bool channel
        """
        stop_sign_detected = msg.data
        logger = self.get_logger() 
        if stop_sign_detected:
            # see if we are in detection delay: active should be True 
            # and 
            # stop sign detected stop robot set active to False
            logger.info(f'Stop sign is detected..Yahoo')
            if not self._last_stop_detection_time:
                # first time stop encounter stop the bot
                logger.info('Set Active=FALSE')
                self.active = False
                #self.set_parameters(
                #    [rclpy.Parameter("active", value=False)]
                #)
            else:
                duration = timenow(self) - self._last_stop_detection_time
                if duration > STOP_DETECTION_DELAY:
                    # hopefully stop sign is not visible any more 
                    # reactivate
                    self.active = True
                    #self.set_parameters(
                    #    [rclpy.Parameter("active", value=True)]
                    #)
                    self._last_stop_detection_time = None
                    logger.info(f'Reset active to {self.active=}')
        else:
            #logger.info('Stop not detected..')
            if self._last_stop_detection_time:
                duration = timenow(self) - self._last_stop_detection_time
                logger.info(f'Stop undetected for {duration=}')
                if duration < STOP_DETECTION_DELAY:
                    return
            # stop sign is not detected, we are out of delay loop, make sure 
            # active is set

            self._last_stop_detection_time = None
            self.active = True #call setter
            
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

    @active.setter
    def active(self, new_val):
        self.set_parameters(
            [rclpy.Parameter("active", value=new_val)]
        )

    def compute_control(self) -> TurtleBotControl:
        """ compute control with goal 
        """
        rctrl = TurtleBotControl()
        is_active = self.active
        logger = self.get_logger()

        if is_active:
            rctrl.v = 0.
            rctrl.omega = 0.5
        else:
            logger.debug(f'False path invoked {self.active=}')
            rctrl.v = 0.
            rctrl.omega = 0.
            #
            # record the time when robot was last stopped after getting
            # active=False, set by STOP detection sign from /detector_bool
            # channel
            # 
            if not self._last_stop_detection_time:
                self._last_stop_detection_time = timenow(self)


        return rctrl


if __name__ == '__main__':
    rclpy.init()
    p_controller = PerceptionController()
    rclpy.spin(p_controller)
    rclpy.shutdown()

