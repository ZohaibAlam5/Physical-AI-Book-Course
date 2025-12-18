---
title: Chapter 12 - Module 1 Capstone Project
description: Capstone project integrating all concepts from Module 1 on Physical AI Foundations
sidebar_position: 12
---

# Chapter 12: Module 1 Capstone Project - Building Your First Physical AI System

## Learning Objectives

After completing this chapter, you should be able to:
- Integrate all concepts learned in Module 1 into a cohesive Physical AI system
- Implement a complete humanoid robot control system using ROS 2
- Apply best practices for safety, reliability, and maintainability
- Demonstrate proficiency in Physical AI foundations

## Introduction

The Module 1 capstone project brings together all the concepts covered in the Physical AI Foundations module. You will build a complete humanoid robot simulation system that demonstrates:

- ROS 2 distributed architecture with nodes, topics, and services
- Robot modeling using URDF with proper kinematic chains
- Python-based AI agents that interact with the robot
- Simulation integration with Gazebo
- Safety monitoring and control systems

## Project Overview

### Project Goals
1. Create a functional humanoid robot simulation in Gazebo
2. Implement a distributed ROS 2 system for robot control
3. Develop AI agents that can interact with the robot
4. Integrate safety systems to ensure safe operation
5. Validate the system through testing and simulation

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agent      │    │ Safety Monitor   │    │  Joint Control  │
│   Node          │    │   Node           │    │   Node          │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          │ /joint_commands      │ /safety_ok           │ /joint_states
          │                     ─┼─                     │
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    ROS 2 Middleware      │
                    │    (DDS Implementation)   │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Gazebo Simulator      │
                    │    (Physics Engine)       │
                    └───────────────────────────┘
```

## Implementation Steps

### Step 1: Robot Model Creation

First, create a complete URDF model for your humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="capstone_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_mass" value="10.0" />
  <xacro:property name="limb_mass" value="2.0" />
  <xacro:property name="head_mass" value="3.0" />

  <!-- Materials -->
  <material name="red">
    <color rgba="0.8 0.2 0.2 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.2 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.2 0.8 0.2 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <mass value="0.001"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${torso_mass}"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="10.0" velocity="2.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${head_mass}"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="20.0" velocity="2.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="15.0" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm -->
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.2 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="20.0" velocity="2.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${limb_mass}"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="15.0" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.05 -0.1 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/3}" effort="30.0" velocity="1.5"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="30.0" velocity="1.5"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_ankle" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Right Leg -->
  <joint name="right_hip" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.05 -0.1 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/3}" effort="30.0" velocity="1.5"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="30.0" velocity="1.5"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_ankle" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/capstone_robot</robotNamespace>
    </plugin>
  </gazebo>

  <!-- IMU sensor in torso -->
  <gazebo reference="torso">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
    </sensor>
  </gazebo>

  <!-- Joint transmissions -->
  <transmission name="left_shoulder_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder_pitch">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_shoulder_pitch_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_shoulder_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_shoulder_pitch">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_shoulder_pitch_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Add transmissions for other joints similarly -->

</robot>
```

### Step 2: Main Control Node

Create the main control node that orchestrates the system:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
import numpy as np
import math
from threading import Lock

class CapstoneController(Node):
    """
    Main controller for the capstone humanoid robot project.

    This node integrates all the concepts learned in Module 1:
    - Distributed system architecture
    - Robot modeling and control
    - AI agent interaction
    - Safety monitoring
    - Simulation integration
    """

    def __init__(self):
        super().__init__('capstone_controller')

        # Internal state management
        self.state_lock = Lock()
        self.current_joint_states = None
        self.imu_data = None
        self.safety_ok = True
        self.operational_mode = 'standby'  # standby, active, emergency

        # Declare parameters
        self.declare_parameter('control_frequency', 100)
        self.declare_parameter('safety_timeout', 1.0)
        self.declare_parameter('balance_threshold', 0.2)

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointState, 'joint_commands', 10)
        self.status_pub = self.create_publisher(
            Bool, 'system_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)
        self.safety_sub = self.create_subscription(
            Bool, 'safety_ok', self.safety_callback, 10)

        # Services
        self.activate_srv = self.create_service(
            Trigger, 'activate_system', self.activate_system_callback)
        self.standby_srv = self.create_service(
            Trigger, 'standby_system', self.standby_system_callback)
        self.emergency_srv = self.create_service(
            Trigger, 'emergency_stop', self.emergency_stop_callback)

        # Timers
        control_freq = self.get_parameter('control_frequency').value
        self.control_timer = self.create_timer(
            1.0/control_freq, self.control_loop)

        self.status_timer = self.create_timer(0.1, self.publish_status)

        # AI agent integration
        self.ai_agent = SimpleAIAgent(self)

        self.get_logger().info('Capstone Controller initialized')
        self.get_logger().info('System ready for activation')

    def joint_state_callback(self, msg):
        """Update current joint state"""
        with self.state_lock:
            self.current_joint_states = msg

    def imu_callback(self, msg):
        """Update IMU data for balance control"""
        with self.state_lock:
            self.imu_data = msg

    def safety_callback(self, msg):
        """Update safety status"""
        with self.state_lock:
            self.safety_ok = msg.data

    def control_loop(self):
        """Main control loop running at specified frequency"""
        with self.state_lock:
            if not self.safety_ok:
                self.get_logger().warn('Safety violation - stopping control')
                self.publish_stop_command()
                return

            if self.operational_mode == 'active':
                # Execute AI agent behavior
                commands = self.ai_agent.get_commands(
                    self.current_joint_states, self.imu_data)

                if commands is not None:
                    self.joint_cmd_pub.publish(commands)

                # Perform balance control if needed
                self.perform_balance_control()

    def perform_balance_control(self):
        """Implement basic balance control"""
        if self.imu_data is None:
            return

        # Simple balance control based on IMU data
        orientation = self.imu_data.orientation
        # Check if robot is tilting too much
        tilt_threshold = self.get_parameter('balance_threshold').value

        # Calculate tilt magnitude (simplified)
        tilt_magnitude = abs(orientation.z)  # Simplified approach

        if tilt_magnitude > tilt_threshold:
            self.get_logger().warn('Balance threshold exceeded - implementing corrective action')
            # Implement balance correction commands
            self.correct_balance()

    def correct_balance(self):
        """Apply balance correction commands"""
        cmd = JointState()
        cmd.name = ['left_ankle', 'right_ankle', 'left_hip', 'right_hip']
        cmd.position = [0.0] * len(cmd.name)

        # Apply corrective positions based on tilt
        if self.imu_data:
            # Simplified balance correction
            cmd.position[0] = -self.imu_data.orientation.y * 0.1  # Left ankle
            cmd.position[1] = -self.imu_data.orientation.y * 0.1  # Right ankle
            cmd.position[2] = self.imu_data.orientation.x * 0.05  # Left hip
            cmd.position[3] = self.imu_data.orientation.x * 0.05  # Right hip

            self.joint_cmd_pub.publish(cmd)

    def publish_status(self):
        """Publish system status"""
        status_msg = Bool()
        status_msg.data = (self.safety_ok and
                          self.operational_mode == 'active')
        self.status_pub.publish(status_msg)

    def publish_stop_command(self):
        """Publish zero commands to stop all movement"""
        cmd = JointState()
        if self.current_joint_states:
            cmd.name = self.current_joint_states.name
            cmd.position = [0.0] * len(cmd.name)
        else:
            # Default joint names for the robot
            cmd.name = [
                'neck_joint', 'left_shoulder_pitch', 'left_elbow',
                'right_shoulder_pitch', 'right_elbow', 'left_hip',
                'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle'
            ]
            cmd.position = [0.0] * len(cmd.name)

        self.joint_cmd_pub.publish(cmd)

    def activate_system_callback(self, request, response):
        """Activate the robot system"""
        with self.state_lock:
            if self.safety_ok:
                self.operational_mode = 'active'
                response.success = True
                response.message = 'System activated successfully'
                self.get_logger().info('System activated')
            else:
                response.success = False
                response.message = 'Cannot activate - safety violation'
                self.get_logger().error('Activation blocked due to safety violation')

        return response

    def standby_system_callback(self, request, response):
        """Put the system in standby mode"""
        with self.state_lock:
            self.operational_mode = 'standby'
            self.publish_stop_command()
            response.success = True
            response.message = 'System in standby mode'
            self.get_logger().info('System in standby')

        return response

    def emergency_stop_callback(self, request, response):
        """Emergency stop procedure"""
        with self.state_lock:
            self.operational_mode = 'emergency'
            self.publish_stop_command()
            response.success = True
            response.message = 'Emergency stop activated'
            self.get_logger().fatal('EMERGENCY STOP ACTIVATED')

        return response


class SimpleAIAgent:
    """
    Simple AI agent that interacts with the humanoid robot.

    This demonstrates the integration of AI concepts with robot control.
    """

    def __init__(self, parent_node):
        self.node = parent_node
        self.behavior_state = 'idle'
        self.time_counter = 0.0

    def get_commands(self, joint_states, imu_data):
        """Generate commands based on current state and sensor data"""
        if joint_states is None:
            return None

        cmd = JointState()
        cmd.name = joint_states.name
        cmd.position = [0.0] * len(cmd.name)

        # Update time counter
        self.time_counter += 0.01

        # Implement different behaviors based on state
        if self.behavior_state == 'idle':
            # Neutral position
            self.apply_neutral_positions(cmd)
        elif self.behavior_state == 'wave':
            # Wave arms in a pattern
            self.apply_wave_behavior(cmd)
        elif self.behavior_state == 'balance':
            # Maintain balance
            self.apply_balance_behavior(cmd, imu_data)

        # Update behavior periodically
        if int(self.time_counter) % 10 == 0:  # Every 10 seconds
            self.update_behavior_state()

        return cmd

    def apply_neutral_positions(self, cmd):
        """Apply neutral joint positions"""
        # For simplicity, keep everything at neutral (0.0)
        # In a real system, you'd have specific neutral poses
        pass

    def apply_wave_behavior(self, cmd):
        """Apply waving arm behavior"""
        # Find arm joint indices (simplified - in real system you'd have mappings)
        for i, name in enumerate(cmd.name):
            if 'shoulder' in name.lower():
                cmd.position[i] = math.sin(self.time_counter) * 0.5
            elif 'elbow' in name.lower():
                cmd.position[i] = math.cos(self.time_counter) * 0.3

    def apply_balance_behavior(self, cmd, imu_data):
        """Apply balance correction based on IMU data"""
        if imu_data is None:
            return

        # Apply corrections based on orientation
        cmd.position[0] = -imu_data.orientation.y * 0.1  # Ankle correction
        cmd.position[1] = -imu_data.orientation.y * 0.1  # Ankle correction

    def update_behavior_state(self):
        """Cycle through different behaviors"""
        if self.behavior_state == 'idle':
            self.behavior_state = 'wave'
        elif self.behavior_state == 'wave':
            self.behavior_state = 'balance'
        else:
            self.behavior_state = 'idle'

        self.node.get_logger().info(f'Behavior state changed to: {self.behavior_state}')


def main(args=None):
    """Main function to run the capstone controller"""
    rclpy.init(args=args)

    controller = CapstoneController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down capstone controller')
    finally:
        # Cleanup
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Safety Monitor Node

Create a dedicated safety monitoring node:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Bool
from builtin_interfaces.msg import Time
from threading import Lock
import numpy as np

class CapstoneSafetyMonitor(Node):
    """
    Safety monitor for the capstone humanoid robot project.

    Implements multiple safety layers and monitoring systems.
    """

    def __init__(self):
        super().__init__('capstone_safety_monitor')

        # Safety state
        self.safety_lock = Lock()
        self.system_safe = True
        self.violation_reasons = []
        self.last_valid_time = self.get_clock().now()

        # Declare safety parameters
        self.declare_parameter('max_joint_velocity', 5.0)
        self.declare_parameter('max_joint_effort', 100.0)
        self.declare_parameter('imu_angular_velocity_limit', 5.0)
        self.declare_parameter('safety_timeout', 0.5)
        self.declare_parameter('fall_threshold', 0.5)
        self.declare_parameter('collision_force_threshold', 50.0)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)

        # Publishers
        self.safety_pub = self.create_publisher(Bool, 'safety_ok', 10)

        # Timer for periodic safety checks
        self.safety_timer = self.create_timer(0.01, self.periodic_safety_check)

        self.get_logger().info('Capstone Safety Monitor initialized')

    def joint_callback(self, msg):
        """Monitor joint states for safety violations"""
        with self.safety_lock:
            # Check joint velocities
            if msg.velocity:
                max_vel = self.get_parameter('max_joint_velocity').value
                for i, vel in enumerate(msg.velocity):
                    if abs(vel) > max_vel:
                        self.log_safety_violation(
                            f'Joint {msg.name[i]} velocity exceeded: {vel} > {max_vel}')

            # Check joint efforts
            if msg.effort:
                max_effort = self.get_parameter('max_joint_effort').value
                for i, eff in enumerate(msg.effort):
                    if abs(eff) > max_effort:
                        self.log_safety_violation(
                            f'Joint {msg.name[i]} effort exceeded: {eff} > {max_effort}')

            # Update last valid time
            self.last_valid_time = self.get_clock().now()

    def imu_callback(self, msg):
        """Monitor IMU data for safety violations"""
        with self.safety_lock:
            # Check angular velocity limits
            max_ang_vel = self.get_parameter('imu_angular_velocity_limit').value
            ang_vel = msg.angular_velocity

            if (abs(ang_vel.x) > max_ang_vel or
                abs(ang_vel.y) > max_ang_vel or
                abs(ang_vel.z) > max_ang_vel):
                self.log_safety_violation(
                    f'IMU angular velocity exceeded: ({ang_vel.x}, {ang_vel.y}, {ang_vel.z})')

            # Check for potential fall based on orientation
            fall_threshold = self.get_parameter('fall_threshold').value
            orientation = msg.orientation

            # Simple fall detection (assuming z-axis is up)
            # In a real system, you'd use more sophisticated methods
            orientation_magnitude = np.sqrt(
                orientation.x**2 + orientation.y**2 + orientation.z**2)

            # Check if orientation vector deviates too much from vertical
            if abs(orientation.z / orientation_magnitude if orientation_magnitude != 0 else 0) < np.cos(fall_threshold):
                self.log_safety_violation('Potential fall detected from IMU data')

    def periodic_safety_check(self):
        """Perform periodic safety checks"""
        with self.safety_lock:
            # Check for sensor timeouts
            current_time = self.get_clock().now()
            timeout_duration = self.get_parameter('safety_timeout').value

            time_since_last_valid = (current_time - self.last_valid_time).nanoseconds / 1e9

            if time_since_last_valid > timeout_duration:
                self.log_safety_violation(f'Sensor timeout: no valid data for {time_since_last_valid:.2f}s')

            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = self.system_safe
            self.safety_pub.publish(safety_msg)

            # Log safety status periodically
            if int(current_time.nanoseconds / 1e9) % 5 == 0:  # Every 5 seconds
                self.get_logger().info(f'System safety status: {self.system_safe}')

    def log_safety_violation(self, reason):
        """Log a safety violation and update system status"""
        if self.system_safe:
            self.system_safe = False
            self.violation_reasons.append(reason)

            self.get_logger().fatal(f'SAFETY VIOLATION: {reason}')
            self.get_logger().fatal('SYSTEM SAFETY COMPROMISED - EMERGENCY PROCEDURES INITIATED')

            # In a real system, you would trigger emergency stop here
            # For this simulation, we just log the violation

    def reset_safety(self):
        """Reset safety system (only after addressing violations)"""
        with self.safety_lock:
            self.system_safe = True
            self.violation_reasons.clear()
            self.last_valid_time = self.get_clock().now()
            self.get_logger().info('Safety system reset - all clear')


def main(args=None):
    """Main function for safety monitor"""
    rclpy.init(args=args)

    safety_monitor = CapstoneSafetyMonitor()

    try:
        rclpy.spin(safety_monitor)
    except KeyboardInterrupt:
        safety_monitor.get_logger().info('Shutting down safety monitor')
    finally:
        safety_monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 4: Launch File

Create a launch file to bring up the entire system:

```python
# launch/capstone_project.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('capstone_humanoid')

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'robot_description':
                    os.path.join(pkg_share, 'urdf', 'capstone_humanoid.urdf')}
            ],
            output='screen'
        ),

        # Joint State Publisher (for GUI control during development)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # Main Controller Node
        Node(
            package='capstone_humanoid',
            executable='capstone_controller',
            name='capstone_controller',
            parameters=[
                {'control_frequency': 100},
                {'safety_timeout': 1.0},
                {'balance_threshold': 0.2}
            ],
            output='screen'
        ),

        # Safety Monitor Node
        Node(
            package='capstone_humanoid',
            executable='capstone_safety_monitor',
            name='capstone_safety_monitor',
            parameters=[
                {'max_joint_velocity': 5.0},
                {'max_joint_effort': 100.0},
                {'safety_timeout': 0.5}
            ],
            output='screen'
        ),

        # Gazebo Simulator (if running simulation)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                )
            )
        ),

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'capstone_humanoid',
                '-file', os.path.join(pkg_share, 'urdf', 'capstone_humanoid.urdf'),
                '-x', '0', '-y', '0', '-z', '1.0'
            ],
            output='screen'
        )
    ])
```

### Step 5: Testing and Validation

Create comprehensive tests for your system:

```python
#!/usr/bin/env python3

import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import time

class TestCapstoneSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize ROS context"""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS context"""
        rclpy.shutdown()

    def setUp(self):
        """Set up test environment"""
        self.test_node = Node('capstone_tester')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.test_node)

        # Create test subscribers
        self.safety_sub = self.test_node.create_subscription(
            Bool, 'safety_ok', self.safety_callback, 10)
        self.status_sub = self.test_node.create_subscription(
            Bool, 'system_status', self.status_callback, 10)

        self.safety_received = False
        self.status_received = False

    def safety_callback(self, msg):
        """Callback for safety messages"""
        self.safety_received = True
        self.last_safety = msg.data

    def status_callback(self, msg):
        """Callback for status messages"""
        self.status_received = True
        self.last_status = msg.data

    def test_safety_system(self):
        """Test that safety system publishes status"""
        # Wait for messages
        timeout = time.time() + 5.0  # 5 second timeout

        while not (self.safety_received and self.status_received):
            if time.time() > timeout:
                self.fail("Timeout waiting for safety messages")
            self.executor.spin_once(timeout_sec=0.1)

        # Verify that safety system is active
        self.assertTrue(self.safety_received, "No safety messages received")
        self.assertTrue(self.status_received, "No status messages received")

    def test_system_activation(self):
        """Test system activation service"""
        # Create client for activation service
        client = self.test_node.create_client(Trigger, 'activate_system')

        # Wait for service
        if not client.wait_for_service(timeout_sec=1.0):
            self.skipTest("Activation service not available")

        # Call service
        request = Trigger.Request()
        future = client.call_async(request)

        # Wait for response
        self.executor.spin_until_future_complete(future, timeout_sec=2.0)

        if future.done():
            response = future.result()
            self.assertTrue(response.success, f"Activation failed: {response.message}")
        else:
            self.fail("Activation service call timed out")

    def test_system_standby(self):
        """Test system standby service"""
        # Create client for standby service
        client = self.test_node.create_client(Trigger, 'standby_system')

        # Wait for service
        if not client.wait_for_service(timeout_sec=1.0):
            self.skipTest("Standby service not available")

        # Call service
        request = Trigger.Request()
        future = client.call_async(request)

        # Wait for response
        self.executor.spin_until_future_complete(future, timeout_sec=2.0)

        if future.done():
            response = future.result()
            self.assertTrue(response.success, f"Standby failed: {response.message}")
        else:
            self.fail("Standby service call timed out")


def main():
    """Run the tests"""
    unittest.main()


if __name__ == '__main__':
    main()
```

## Project Validation

### Testing Checklist

- [ ] Robot model loads correctly in Gazebo
- [ ] All joints move within expected ranges
- [ ] Safety monitor detects and responds to violations
- [ ] Main controller receives sensor data
- [ ] AI agent generates appropriate commands
- [ ] Balance control system responds to IMU data
- [ ] Emergency stop functions properly
- [ ] System activates and deactivates safely
- [ ] All ROS 2 nodes communicate properly
- [ ] Performance meets requirements

### Performance Metrics

- Control loop frequency: Target 100Hz
- Sensor data latency: < 10ms
- Safety response time: < 1ms
- System startup time: < 30 seconds

## Summary

The Module 1 capstone project demonstrates the integration of all Physical AI foundations concepts:

1. **Distributed Architecture**: Using ROS 2 nodes, topics, and services
2. **Robot Modeling**: Creating accurate URDF models with proper physics
3. **AI Integration**: Implementing AI agents that interact with the robot
4. **Safety Systems**: Multiple layers of safety monitoring and enforcement
5. **Simulation Integration**: Connecting to Gazebo for realistic testing
6. **Best Practices**: Following ROS 2 development standards

Successfully completing this project validates your understanding of Physical AI foundations and prepares you for more advanced topics in the subsequent modules. The skills developed here form the basis for building complex humanoid robot systems that can operate safely and effectively in the physical world.