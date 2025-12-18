---
title: Chapter 5 - Data Flow from Sensors to Actuators
description: Understanding the complete data flow in humanoid robot systems from sensing to action
sidebar_position: 5
---

# Chapter 5: Data Flow from Sensors to Actuators

## Learning Objectives

After completing this chapter, you should be able to:
- Trace the complete data flow from sensors to actuators in a humanoid robot
- Understand the timing and synchronization requirements
- Design efficient data processing pipelines

## Introduction

In humanoid robots, data flows through multiple layers of processing from sensors that perceive the environment to actuators that execute physical actions. Understanding this complete data flow is essential for building responsive and reliable robotic systems.

## Sensor Data Acquisition

The first step in the data flow involves acquiring data from various sensors:

### Types of Sensors
- **Proprioceptive sensors**: Joint encoders, IMU, force/torque sensors
- **Exteroceptive sensors**: Cameras, LiDAR, microphones, touch sensors
- **Environmental sensors**: Temperature, pressure, humidity sensors

### Sensor Data Processing Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Subscribe to sensor data
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)

        # Publish processed sensor data
        self.processed_pub = self.create_publisher(
            Float64MultiArray, 'processed_sensors', 10)

    def joint_callback(self, msg):
        # Process joint state data
        processed_data = self.process_joint_data(msg)
        self.publish_processed_data(processed_data)

    def imu_callback(self, msg):
        # Process IMU data
        processed_data = self.process_imu_data(msg)
        self.publish_processed_data(processed_data)

    def process_joint_data(self, joint_state):
        # Apply filtering and calibration
        filtered_positions = self.filter_positions(joint_state.position)
        return filtered_positions

    def process_imu_data(self, imu_data):
        # Extract orientation and angular velocity
        orientation = imu_data.orientation
        angular_velocity = imu_data.angular_velocity
        return [orientation, angular_velocity]
```

## Perception and State Estimation

After sensor data acquisition, the robot needs to estimate its current state:

### State Estimation Pipeline
- **Sensor fusion**: Combining data from multiple sensors
- **Filtering**: Applying Kalman filters or particle filters
- **State prediction**: Estimating current and future states

## Decision Making and Planning

Based on the processed sensor data and estimated state, the robot makes decisions:

### Planning Hierarchy
- **High-level planning**: Task planning and goal selection
- **Mid-level planning**: Path planning and trajectory generation
- **Low-level planning**: Joint trajectory generation

```python
class DecisionMaker(Node):
    def __init__(self):
        super().__init__('decision_maker')

        # Subscribe to processed sensor data
        self.sensor_sub = self.create_subscription(
            Float64MultiArray, 'processed_sensors',
            self.sensor_callback, 10)

        # Publish commands to actuators
        self.command_pub = self.create_publisher(
            JointState, 'joint_commands', 10)

    def sensor_callback(self, msg):
        # Make decisions based on sensor data
        command = self.make_decision(msg.data)
        self.command_pub.publish(command)

    def make_decision(self, sensor_data):
        # Implement decision-making logic
        # This could involve state machines, neural networks, etc.
        pass
```

## Actuator Command Execution

The final step is sending commands to actuators:

### Actuator Control
- **Position control**: Setting joint positions
- **Velocity control**: Controlling joint velocities
- **Torque control**: Controlling applied forces

### Control Loop Implementation

```python
class ActuatorController(Node):
    def __init__(self):
        super().__init__('actuator_controller')

        # Subscribe to commands
        self.command_sub = self.create_subscription(
            JointState, 'joint_commands',
            self.command_callback, 10)

        # Publish to hardware interface
        self.hardware_pub = self.create_publisher(
            JointState, 'hardware_commands', 10)

    def command_callback(self, msg):
        # Apply control algorithms
        hardware_commands = self.apply_control_law(msg)
        self.hardware_pub.publish(hardware_commands)

    def apply_control_law(self, commands):
        # Implement PID control, model predictive control, etc.
        pass
```

## Real-time Considerations

For humanoid robots, timing is critical:

### Timing Requirements
- **Sensor acquisition**: Typically 100-1000 Hz
- **State estimation**: 100-500 Hz
- **Control loops**: 100-2000 Hz
- **High-level planning**: 1-10 Hz

### Synchronization

```python
import threading
import time

class RealTimeScheduler:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_sensor_time = time.time()

    def synchronize_data_flow(self):
        # Ensure data consistency across the pipeline
        with self.lock:
            current_time = time.time()
            if current_time - self.last_sensor_time > 0.01:  # 10ms
                # Process new sensor data
                self.process_sensors()
                self.last_sensor_time = current_time
```

## Summary

The data flow from sensors to actuators in humanoid robots involves multiple processing stages with specific timing and synchronization requirements. Understanding this complete pipeline is essential for building responsive and reliable robotic systems.