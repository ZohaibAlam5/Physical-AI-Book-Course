---
title: Chapter 10 - Simulation Basics with Gazebo
description: Introduction to robot simulation using Gazebo for humanoid robotics
sidebar_position: 10
---

# Chapter 10: Simulation Basics with Gazebo

## Learning Objectives

After completing this chapter, you should be able to:
- Set up and configure Gazebo for humanoid robot simulation
- Create basic simulation environments
- Implement controllers for simulated humanoid robots
- Debug and troubleshoot simulation issues

## Introduction

Gazebo is a powerful open-source robotics simulator that provides realistic physics simulation, sensor simulation, and rendering capabilities. For humanoid robots, Gazebo offers the ability to test control algorithms, develop behaviors, and validate designs in a safe virtual environment before deployment on real hardware.

## Installing and Setting Up Gazebo

### Prerequisites

Gazebo comes with ROS 2 installations, but you can install it separately:

```bash
# On Ubuntu
sudo apt update
sudo apt install gazebo11 libgazebo11-dev

# For ROS 2 Humble Hawksbill
sudo apt install ros-humble-gazebo-*
```

### Basic Gazebo Launch

```bash
# Launch Gazebo with default empty world
gazebo

# Launch with a specific world
gazebo worlds/empty.world
```

## Gazebo Configuration for Humanoid Robots

### Creating a World File

Create a world file that defines your simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="humanoid_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a sky -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Environment objects -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.5 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.5 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Place for your humanoid robot -->
    <!-- This will be populated when launching with robot model -->

  </world>
</sdf>
```

## Integrating Humanoid Robot with Gazebo

### Adding Gazebo Plugins to URDF

To make your robot work in Gazebo, you need to add Gazebo-specific plugins to your URDF:

```xml
<?xml version="1.0"?>
<robot name="humanoid_gazebo" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include the basic robot model -->
  <xacro:include filename="$(find humanoid_description)/urdf/humanoid.urdf.xacro" />

  <!-- Gazebo-specific material definitions -->
  <gazebo reference="torso">
    <material>Gazebo/Red</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_upper_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="right_upper_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Gazebo plugin for joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <joint_name>left_shoulder_joint</joint_name>
      <joint_name>right_shoulder_joint</joint_name>
      <joint_name>left_elbow_joint</joint_name>
      <joint_name>right_elbow_joint</joint_name>
      <joint_name>left_hip_joint</joint_name>
      <joint_name>right_hip_joint</joint_name>
      <joint_name>left_knee_joint</joint_name>
      <joint_name>right_knee_joint</joint_name>
      <joint_name>left_ankle_joint</joint_name>
      <joint_name>right_ankle_joint</joint_name>
    </plugin>
  </gazebo>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- IMU sensor plugin -->
  <gazebo reference="torso">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.1</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.1</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.1</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

  <!-- Camera sensor plugin -->
  <gazebo reference="head">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>head_camera_optical_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Force/Torque sensor plugin -->
  <gazebo reference="left_ankle_joint">
    <sensor name="left_ankle_ft_sensor" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>child</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </gazebo>

  <gazebo reference="right_ankle_joint">
    <sensor name="right_ankle_ft_sensor" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>child</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </gazebo>

</robot>
```

## Launching Robot in Gazebo

### Launch File for Robot Spawn

Create a launch file to spawn your robot in Gazebo:

```python
# launch/humanoid_gazebo.launch.py
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get URDF via xacro
    robot_description_path = os.path.join(
        get_package_share_directory('humanoid_description'),
        'urdf',
        'humanoid_gazebo.urdf.xacro'
    )

    # Read URDF
    with open(robot_description_path, 'r') as infp:
        robot_desc = infp.read()

    # Robot State Publisher node
    params = {'robot_description': robot_desc}
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # Gazebo server
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [ThisLaunchFileDir(), '/gzserver.launch.py']
        )
    )

    # Gazebo client
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [ThisLaunchFileDir(), '/gzclient.launch.py']
        )
    )

    # Spawn entity node
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'humanoid_robot'],
        output='screen'
    )

    return LaunchDescription([
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_entity,
    ])
```

## Implementing Controllers

### Joint Position Controller

Create a controller configuration file:

```yaml
# config/humanoid_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_arm_controller:
      type: position_controllers/JointGroupPositionController

    right_arm_controller:
      type: position_controllers/JointGroupPositionController

    leg_controller:
      type: position_controllers/JointGroupPositionController

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_joint
      - left_elbow_joint

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_joint
      - right_elbow_joint

leg_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
      - left_ankle_joint
      - right_hip_joint
      - right_knee_joint
      - right_ankle_joint
```

### Controller Spawning Node

```python
# scripts/spawn_controllers.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController

class ControllerSpawner(Node):
    def __init__(self):
        super().__init__('controller_spawner')

        # Create client for controller manager
        self.client = self.create_client(
            SwitchController,
            '/controller_manager/switch_controller'
        )

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for controller manager service...')

        self.spawn_controllers()

    def spawn_controllers(self):
        """Spawn and activate controllers"""
        request = SwitchController.Request()

        # Activate controllers
        request.start_controllers = [
            'joint_state_broadcaster',
            'left_arm_controller',
            'right_arm_controller',
            'leg_controller'
        ]
        request.stop_controllers = []
        request.strictness = SwitchController.Request.BEST_EFFORT

        future = self.client.call_async(request)
        future.add_done_callback(self.controller_spawn_callback)

    def controller_spawn_callback(self, future):
        """Callback when controllers are spawned"""
        response = future.result()
        if response.ok:
            self.get_logger().info('Controllers activated successfully')
        else:
            self.get_logger().error('Failed to activate controllers')

def main(args=None):
    rclpy.init(args=args)
    spawner = ControllerSpawner()

    try:
        rclpy.spin(spawner)
    except KeyboardInterrupt:
        spawner.get_logger().info('Shutting down controller spawner')
    finally:
        spawner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Basic Simulation Example

### Simple Movement Controller

```python
# scripts/basic_sim_controller.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import math

class BasicSimController(Node):
    def __init__(self):
        super().__init__('basic_sim_controller')

        # Publishers for joint commands
        self.left_arm_pub = self.create_publisher(
            Float64MultiArray,
            '/left_arm_controller/commands',
            10
        )
        self.right_arm_pub = self.create_publisher(
            Float64MultiArray,
            '/right_arm_controller/commands',
            10
        )
        self.leg_pub = self.create_publisher(
            Float64MultiArray,
            '/leg_controller/commands',
            10
        )

        # Subscriber for joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Internal state
        self.joint_positions = {}
        self.time_counter = 0.0

        self.get_logger().info('Basic Simulation Controller initialized')

    def joint_state_callback(self, msg):
        """Update internal joint state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop"""
        # Update time counter
        self.time_counter += 0.01

        # Send commands to left arm (wave motion)
        left_arm_cmd = Float64MultiArray()
        left_arm_cmd.data = [
            math.sin(self.time_counter) * 0.5,    # Shoulder
            math.cos(self.time_counter) * 0.3     # Elbow
        ]
        self.left_arm_pub.publish(left_arm_cmd)

        # Send commands to right arm (opposite wave)
        right_arm_cmd = Float64MultiArray()
        right_arm_cmd.data = [
            math.sin(self.time_counter + math.pi) * 0.5,  # Shoulder
            math.cos(self.time_counter + math.pi) * 0.3   # Elbow
        ]
        self.right_arm_pub.publish(right_arm_cmd)

        # Send commands to legs (standing position with slight movement)
        leg_cmd = Float64MultiArray()
        leg_cmd.data = [
            0.0,  # Left hip
            0.0,  # Left knee
            0.0,  # Left ankle
            0.0,  # Right hip
            0.0,  # Right knee
            0.0   # Right ankle
        ]
        self.leg_pub.publish(leg_cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = BasicSimController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down simulation controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Debugging Simulation Issues

### Common Problems and Solutions

1. **Robot Falls Through Ground**:
   - Check that all links have proper inertial properties
   - Verify collision geometries are defined
   - Adjust physics parameters in world file

2. **Joints Don't Respond**:
   - Ensure controllers are properly loaded and running
   - Check topic names match between controller and publishers
   - Verify joint names in controller configuration match URDF

3. **Simulation Runs Slowly**:
   - Reduce physics update rate
   - Simplify collision geometries
   - Use fewer sensors or reduce update rates

4. **Robot Oscillates Uncontrollably**:
   - Adjust controller gains
   - Increase damping in URDF
   - Reduce simulation step size

### Diagnostic Tools

```bash
# Check if controllers are loaded
ros2 control list_controllers

# Monitor joint states
ros2 topic echo /joint_states

# Check TF frames
ros2 run tf2_tools view_frames

# Monitor controller state
ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers
```

## Advanced Simulation Features

### Terrain Generation

Create custom terrains for testing locomotion:

```xml
<!-- In world file -->
<model name="uneven_terrain">
  <static>true</static>
  <link name="terrain_link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>model://terrain/heightmap.png</uri>
          <size>10 10 2</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>model://terrain/heightmap.png</uri>
          <size>10 10 2</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

### Multi-Robot Simulation

Simulate multiple robots in the same environment:

```python
# launch/multi_robot_gazebo.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Robot 1
    robot1_spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot1',
            '-topic', 'robot1_description',
            '-x', '0.0', '-y', '0.0', '-z', '1.0'
        ],
        output='screen'
    )

    # Robot 2
    robot2_spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot2',
            '-topic', 'robot2_description',
            '-x', '2.0', '-y', '0.0', '-z', '1.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        robot1_spawner,
        robot2_spawner,
    ])
```

## Performance Optimization

### Reducing Computational Load

1. **Physics Optimization**:
   - Use simpler collision shapes (boxes instead of meshes)
   - Adjust physics parameters for speed vs accuracy trade-off
   - Reduce update rates for non-critical sensors

2. **Visualization Optimization**:
   - Turn off rendering when not needed for testing
   - Reduce model complexity in visual elements
   - Use wireframe mode for debugging

3. **Sensor Optimization**:
   - Reduce sensor update rates when possible
   - Use simpler sensor models during development
   - Limit sensor ranges to necessary values

## Summary

Gazebo provides a powerful platform for simulating humanoid robots before deploying to real hardware. By properly configuring your URDF with Gazebo plugins and implementing appropriate controllers, you can create realistic simulations that accurately represent the behavior of your physical robot. Remember to validate your simulation results with real-world tests to ensure accuracy.