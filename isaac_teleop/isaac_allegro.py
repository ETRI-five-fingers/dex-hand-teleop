#!/usr/bin/env python3
import platform

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String
import numpy as np
import time


class IsaacAllegro(Node):
    def __init__(self):

        super().__init__("test_ros2bridge")

        # Create the publisher. This publisher will publish a JointState message to the /joint_command topic.
        self.publisher_ = self.create_publisher(JointState, "joint_command", 10)
        self.subs_joint_state = self.create_subscription(JointState, 'joint_states', self.subs_callback, 10)

        # Create a JointState message
        self.joint_state = JointState()

        self.joint_state.name = [
            "joint_0",
            "joint_12",
            "joint_4",
            "joint_8",
            "joint_1",
            "joint_13",
            "joint_5",
            "joint_9",
            "joint_2",
            "joint_14",
            "joint_6",
            "joint_10",
            "joint_3",
            "joint_15",
            "joint_7",
            "joint_11"
        ]

        num_joints = len(self.joint_state.name)

        # make sure kit's editor is playing for receiving messages
        self.joint_state.position = np.array([0.0] * num_joints, dtype=np.float64).tolist()
        self.default_joints = [0.0, 0.263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # limiting the movements to a smaller range (this is not the range of the robot, just the range of the movement
        self.max_joints = np.array(self.default_joints) + 1.5
        self.min_joints = np.array(self.default_joints) - 1.5

        # position control the robot to wiggle around each joint
        self.time_start = time.time()

        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()

        joint_position = (
            np.sin(time.time() - self.time_start) * (self.max_joints - self.min_joints) * 0.5 + self.default_joints
        )
        self.joint_state.position = joint_position.tolist()

        # Publish the message to the topic
        self.publisher_.publish(self.joint_state)

    def subs_callback(self, msg):
        for n, p, v, e in zip(msg.name, msg.position, msg.velocity, msg.effort):
            print("{}: pos: {}, vel: {}, effort: {}".format(n, p, v, e))
        # self.get_logger().info('I heard: "%s"' % msg)


def main(args=None):
    rclpy.init(args=args)

    ros2_publisher = IsaacAllegro()

    rclpy.spin(ros2_publisher)

    # Destroy the node explicitly
    ros2_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
