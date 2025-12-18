---
title: Chapter 8 - Practical ROS 2 Examples for Physical AI
description: Hands-on examples of ROS 2 implementations for Physical AI and humanoid robotics
sidebar_position: 8
---

# Chapter 8: Practical ROS 2 Examples for Physical AI

## Learning Objectives

After completing this chapter, you should be able to:
- Implement practical ROS 2 nodes for Physical AI applications
- Create message passing systems for robot control
- Design distributed systems for humanoid robot control

## Introduction

This chapter provides practical examples of ROS 2 implementations specifically tailored for Physical AI and humanoid robotics. These examples demonstrate real-world applications of the concepts learned in previous chapters.

## Example 1: Joint State Publisher

A fundamental node that publishes joint states from a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Timer to publish joint states at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_joint_states)

        # Initialize joint names and positions
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]

        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

        self.get_logger().info('Joint State Publisher initialized')

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        # Publish the message
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Joint State Publisher')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example 2: Balance Controller

A node that maintains the balance of a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Subscribe to IMU data for balance feedback
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Publish joint commands to maintain balance
        self.command_pub = self.create_publisher(
            JointState, 'joint_commands', 10)

        # PID controller parameters
        self.kp = 10.0  # Proportional gain
        self.ki = 0.1   # Integral gain
        self.kd = 0.5   # Derivative gain

        # Error accumulation for integral term
        self.integral_error = 0.0
        self.previous_error = 0.0

        # Robot state
        self.current_orientation = None
        self.current_joint_positions = None

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.balance_control_loop)

        self.get_logger().info('Balance Controller initialized')

    def imu_callback(self, msg):
        """Process IMU data to determine robot orientation"""
        self.current_orientation = msg.orientation

    def joint_callback(self, msg):
        """Process current joint positions"""
        self.current_joint_positions = list(msg.position)

    def balance_control_loop(self):
        """Main balance control loop"""
        if self.current_orientation is not None:
            # Calculate error from desired upright position
            # For simplicity, we'll use the z component of orientation quaternion
            # as a measure of how upright the robot is
            desired_orientation = 0.707  # Approximately upright (cos(45°))
            current_orientation_z = self.current_orientation.z

            error = desired_orientation - current_orientation_z

            # Calculate PID terms
            self.integral_error += error * 0.01  # dt = 0.01s
            derivative_error = (error - self.previous_error) / 0.01

            # Calculate control output
            control_output = (self.kp * error +
                            self.ki * self.integral_error +
                            self.kd * derivative_error)

            # Generate balance correction commands
            balance_commands = self.generate_balance_commands(control_output)

            # Publish commands
            if balance_commands is not None:
                self.command_pub.publish(balance_commands)

            self.previous_error = error

    def generate_balance_commands(self, control_output):
        """Generate joint commands to correct balance"""
        if self.current_joint_positions is None:
            return None

        # Create joint state command
        cmd = JointState()
        cmd.name = ['left_ankle', 'right_ankle', 'left_hip', 'right_hip']
        cmd.position = [0.0] * len(cmd.name)

        # Apply balance correction to ankle joints
        # This is a simplified approach - real implementations would be more complex
        cmd.position[0] = control_output * 0.1  # Left ankle correction
        cmd.position[1] = control_output * 0.1  # Right ankle correction
        cmd.position[2] = -control_output * 0.05  # Left hip correction
        cmd.position[3] = -control_output * 0.05  # Right hip correction

        return cmd

def main(args=None):
    rclpy.init(args=args)
    node = BalanceController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Balance Controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example 3: Perception Node

A node that processes sensor data to perceive the environment:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Subscribe to various sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)

        # Publish perception results
        self.obstacle_pub = self.create_publisher(
            PointStamped, 'obstacle_detected', 10)
        self.perception_status_pub = self.create_publisher(
            String, 'perception_status', 10)

        # Parameters
        self.min_obstacle_distance = 0.5  # meters
        self.obstacle_threshold = 0.3     # meters

        self.get_logger().info('Perception Node initialized')

    def laser_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        # Convert ranges to numpy array for easier processing
        ranges = np.array(msg.ranges)

        # Filter out invalid readings (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            # Find minimum distance
            min_distance = np.min(valid_ranges)

            if min_distance < self.min_obstacle_distance:
                # Obstacle detected - publish location
                obstacle_msg = PointStamped()
                obstacle_msg.header = msg.header

                # Find the angle of the closest obstacle
                closest_idx = np.argmin(ranges)
                closest_angle = msg.angle_min + closest_idx * msg.angle_increment

                # Convert polar to Cartesian coordinates
                obstacle_msg.point.x = min_distance * np.cos(closest_angle)
                obstacle_msg.point.y = min_distance * np.sin(closest_angle)
                obstacle_msg.point.z = 0.0

                self.obstacle_pub.publish(obstacle_msg)

                # Log the detection
                self.get_logger().info(
                    f'Obstacle detected at distance: {min_distance:.2f}m, '
                    f'angle: {closest_angle:.2f}rad')

    def camera_callback(self, msg):
        """Process camera image for object detection"""
        # In a real implementation, this would use computer vision libraries
        # like OpenCV to process the image

        # For this example, we'll just check if the image is valid
        if msg.width > 0 and msg.height > 0:
            status_msg = String()
            status_msg.data = f'Camera: {msg.width}x{msg.height} image received'
            self.perception_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Perception Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example 4: Walking Pattern Generator

A node that generates walking patterns for humanoid robots:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import math

class WalkingPatternGenerator(Node):
    def __init__(self):
        super().__init__('walking_pattern_generator')

        # Subscribe to velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # Publish joint trajectories for walking
        self.trajectory_pub = self.create_publisher(
            Float64MultiArray, 'walking_trajectory', 10)

        # Walking parameters
        self.step_frequency = 1.0  # Hz
        self.step_length = 0.2     # meters
        self.step_height = 0.05    # meters
        self.step_phase = 0.0      # current phase of walking cycle

        # Walking state
        self.is_walking = False
        self.desired_velocity = 0.0
        self.desired_angular = 0.0

        # Timer for walking pattern generation
        self.pattern_timer = self.create_timer(0.01, self.generate_walking_pattern)

        self.get_logger().info('Walking Pattern Generator initialized')

    def cmd_vel_callback(self, msg):
        """Process velocity commands"""
        self.desired_velocity = msg.linear.x
        self.desired_angular = msg.angular.z

        # Start walking if velocity is non-zero
        self.is_walking = abs(self.desired_velocity) > 0.01

    def generate_walking_pattern(self):
        """Generate walking pattern based on desired velocity"""
        if not self.is_walking:
            # Publish neutral position when not walking
            neutral_msg = Float64MultiArray()
            neutral_msg.data = [0.0] * 12  # 12 joints for example
            self.trajectory_pub.publish(neutral_msg)
            return

        # Update walking phase
        self.step_phase += 2 * math.pi * self.step_frequency * 0.01

        # Generate walking pattern
        trajectory_msg = Float64MultiArray()

        # Simplified walking pattern - in reality this would be much more complex
        # with proper inverse kinematics and gait planning
        joints = []

        # Hip joints - create walking motion
        left_hip = math.sin(self.step_phase) * 0.1
        right_hip = math.sin(self.step_phase + math.pi) * 0.1

        # Knee joints - coordinate with hip for walking
        left_knee = math.cos(self.step_phase) * 0.05
        right_knee = math.cos(self.step_phase + math.pi) * 0.05

        # Ankle joints - maintain balance
        left_ankle = -math.sin(self.step_phase) * 0.05
        right_ankle = -math.sin(self.step_phase + math.pi) * 0.05

        # Add some angular motion for turning
        if abs(self.desired_angular) > 0.01:
            # Adjust for turning
            left_hip += self.desired_angular * 0.1
            right_hip -= self.desired_angular * 0.1

        # Add to joint array (simplified - real robot would have more joints)
        joints.extend([left_hip, left_knee, left_ankle,
                      right_hip, right_knee, right_ankle])

        # Add arm movements for balance
        joints.extend([math.sin(self.step_phase + math.pi/2) * 0.05,  # left arm
                      math.cos(self.step_phase + math.pi/2) * 0.05,   # left elbow
                      math.sin(self.step_phase + math.pi/2) * 0.05,   # right arm
                      math.cos(self.step_phase + math.pi/2) * 0.05])  # right elbow

        trajectory_msg.data = joints
        self.trajectory_pub.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    node = WalkingPatternGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Walking Pattern Generator')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example 5: Multi-Node System Integration

A complete example showing how multiple nodes work together:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import threading
import time

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointState, 'joint_commands', 10)
        self.status_pub = self.create_publisher(
            String, 'robot_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)

        # Robot state
        self.current_joints = None
        self.imu_data = None
        self.desired_behavior = 'idle'  # idle, walk, balance, etc.

        # Behavior timer
        self.behavior_timer = self.create_timer(0.05, self.execute_behavior)

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint state"""
        self.current_joints = msg

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = msg

    def execute_behavior(self):
        """Execute current behavior based on robot state"""
        if self.desired_behavior == 'idle':
            self.execute_idle_behavior()
        elif self.desired_behavior == 'balance':
            self.execute_balance_behavior()
        elif self.desired_behavior == 'walk':
            self.execute_walk_behavior()

    def execute_idle_behavior(self):
        """Execute idle behavior - maintain neutral position"""
        if self.current_joints is not None:
            cmd = JointState()
            cmd.name = self.current_joints.name
            cmd.position = [0.0] * len(self.current_joints.name)
            self.joint_cmd_pub.publish(cmd)

    def execute_balance_behavior(self):
        """Execute balance behavior using IMU feedback"""
        if self.imu_data is not None and self.current_joints is not None:
            # Simple balance control based on IMU orientation
            orientation = self.imu_data.orientation
            # Implement balance control logic here
            cmd = JointState()
            cmd.name = self.current_joints.name
            cmd.position = [0.0] * len(self.current_joints.name)

            # Adjust ankle positions based on orientation
            # Simplified for example
            cmd.position[0] = orientation.z * 0.1  # Left ankle
            cmd.position[1] = orientation.z * 0.1  # Right ankle

            self.joint_cmd_pub.publish(cmd)

    def execute_walk_behavior(self):
        """Execute walking behavior"""
        # This would integrate with the walking pattern generator
        # For this example, we'll just send a simple walking pattern
        if self.current_joints is not None:
            cmd = JointState()
            cmd.name = self.current_joints.name
            cmd.position = [0.0] * len(self.current_joints.name)

            # Simple walking pattern
            time_factor = self.get_clock().now().nanoseconds / 1e9
            cmd.position[0] = math.sin(time_factor) * 0.1  # Left ankle
            cmd.position[1] = math.sin(time_factor + math.pi) * 0.1  # Right ankle

            self.joint_cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Humanoid Controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files for Multi-Node Systems

Create a launch file to start multiple nodes together:

```python
# launch/humanoid_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='humanoid_control',
            executable='joint_state_publisher',
            name='joint_state_publisher'
        ),
        Node(
            package='humanoid_control',
            executable='balance_controller',
            name='balance_controller'
        ),
        Node(
            package='humanoid_control',
            executable='perception_node',
            name='perception_node'
        ),
        Node(
            package='humanoid_control',
            executable='walking_pattern_generator',
            name='walking_pattern_generator'
        ),
        Node(
            package='humanoid_control',
            executable='humanoid_controller',
            name='humanoid_controller'
        ),
    ])
```

## Summary

These practical examples demonstrate how to implement various components of a humanoid robot control system using ROS 2. Each example builds on the concepts learned in previous chapters and shows real-world applications of Physical AI principles. Understanding these patterns is crucial for developing complex humanoid robot systems.