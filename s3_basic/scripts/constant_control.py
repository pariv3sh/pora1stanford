#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Int64, Bool
from geometry_msgs.msg import Twist


class Heartbeat(Node):
    def __init__(self) -> None:
        # initialize base class (must happen before everything else)
        super().__init__("twist_ctrl")
                
        # a heartbeat counter
        self.velocity = 1.0
        self.ang_velocity = 1.0

        # create publisher with: self.create_publisher(<msg type>, <topic>, <qos>)
        self.hb_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # create a timer with: self.create_timer(<second>, <callback>)
        self.hb_timer = self.create_timer(1.0, self.twister_callback)

        # create subscription with: self.create_subscription(<msg type>, <topic>, <callback>, <qos>)
        self.kill_sub = self.create_subscription(Bool, 
                                "/kill", 
                                self.kill_callback, 
                                10)

    def twister_callback(self) -> None:
        """
        Heartbeat callback triggered by the timer
        """
        print('sending constant control ...')
        # construct heartbeat message
        msg = Twist()
        msg.linear.x = self.velocity
        # stop the angular velocity: should make the robot walk in straight line
        #msg.angular.z = self.ang_velocity

        # publish heartbeat counter
        self.hb_pub.publish(msg)


    def kill_callback(self, msg: Bool) -> None:
        """
        Sensor health callback triggered by subscription
        """
        print(f'Got kill message {msg.data}') 
        if msg.data:
            self.hb_timer.cancel()
            zeromsg = Twist()  
            zeromsg.linear.x = 0.0
            zeromsg.angular.z = 0.0
            self.hb_pub.publish(zeromsg)

        else:
            self.get_logger().info("Ignoring empty")


if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Heartbeat()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context
