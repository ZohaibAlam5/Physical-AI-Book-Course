---
title: Chapter 3 - Gazebo for Robot Simulation
description: Comprehensive guide to using Gazebo for humanoid robot simulation
sidebar_position: 3
---

# Chapter 3: Gazebo for Robot Simulation

## Learning Objectives

After completing this chapter, you should be able to:
- Set up and configure Gazebo for humanoid robot simulation
- Create and customize Gazebo worlds for robotics applications
- Integrate Gazebo with ROS 2 for realistic robot simulation
- Utilize Gazebo's sensors, physics, and visualization capabilities
- Troubleshoot common Gazebo simulation issues

## Introduction

Gazebo is one of the most widely-used simulation environments in robotics research and development. Originally developed by the Open Source Robotics Foundation (OSRF), Gazebo provides a comprehensive platform for simulating robots in realistic environments. For Physical AI and humanoid robotics, Gazebo offers the necessary physics accuracy, sensor simulation, and visualization capabilities to create meaningful simulation environments.

## Gazebo Architecture and Components

### Core Components

Gazebo consists of several key components that work together to provide a complete simulation environment:

#### 1. Physics Engine
- **ODE (Open Dynamics Engine)**: Default physics engine for rigid body simulation
- **Bullet**: Alternative physics engine with advanced features
- **Simbody**: Multi-body dynamics engine
- **DART**: Dynamic Animation and Robotics Toolkit

#### 2. Rendering Engine
- **OGRE**: Scene graph and rendering
- **OpenGL**: Graphics rendering pipeline
- **GUI**: Qt-based user interface

#### 3. Communication Layer
- **Transport**: ZeroMQ-based messaging system
- **Messages**: Protobuf-based message definitions
- **Services**: RPC mechanisms for simulation control

### Gazebo Libraries

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/common/common.hh>
#include <gazebo/rendering/rendering.hh>
```

## Installing and Setting Up Gazebo

### Installation

For ROS 2 integration, install Gazebo alongside your ROS distribution:

```bash
# On Ubuntu with ROS 2 Humble Hawksbill
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-dev

# Alternative: Install standalone Gazebo Garden (recommended)
sudo apt install gazebo-garden
```

### Basic Gazebo Commands

```bash
# Launch Gazebo with default empty world
gazebo

# Launch with a specific world
gazebo worlds/empty.world

# Launch with verbose output
gazebo -v 4

# Launch without GUI (for batch processing)
gazebo -s libgazebo_ros_init.so worlds/empty.world
```

## Creating Gazebo Worlds

### World File Structure

Gazebo worlds are defined using SDF (Simulation Description Format):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_simulation">
    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="ode" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
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

    <!-- Custom models -->
    <model name="my_robot">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="chassis">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.5 0.25</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.5 0.25</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
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

    <!-- Lighting -->
    <light name="sun_light" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -1.0</direction>
    </light>

  </world>
</sdf>
```

### Advanced World Configuration

#### Custom Terrain

```xml
<model name="custom_terrain">
  <static>true</static>
  <link name="terrain_link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>model://terrain/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>model://terrain/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
      <material>
        <script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/Dirt</name>
        </script>
      </material>
    </visual>
  </link>
</model>
```

#### Obstacles and Structures

```xml
<!-- Table obstacle -->
<model name="table">
  <pose>5 0 0 0 0 0</pose>
  <link name="table_base">
    <collision name="collision">
      <geometry>
        <box>
          <size>1.5 0.8 0.8</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.5 0.8 0.8</size>
        </box>
      </geometry>
      <material>
        <ambient>0.6 0.4 0.2 1</ambient>
        <diffuse>0.6 0.4 0.2 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>20.0</mass>
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

<!-- Stairs -->
<model name="stairs">
  <pose>8 0 0 0 0 0</pose>
  <link name="step_1">
    <collision name="collision">
      <geometry>
        <box>
          <size>1.0 0.3 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.0 0.3 0.1</size>
        </box>
      </geometry>
    </visual>
    <inertial>
      <mass>5.0</mass>
      <inertia>
        <ixx>0.1</ixx>
        <iyy>0.1</iyy>
        <izz>0.1</izz>
      </inertial>
    </inertial>
  </link>
  <!-- Additional steps would be added here -->
</model>
```

## Integrating with ROS 2

### Gazebo ROS Packages

The Gazebo ROS packages provide seamless integration between Gazebo and ROS 2:

```xml
<!-- package.xml -->
<depend>gazebo_ros_pkgs</depend>
<depend>gazebo_dev</depend>
<depend>gazebo_msgs</depend>
<depend>gazebo_plugins</depend>
```

### Spawning Robots in Gazebo

```python
# Python script to spawn robot in Gazebo
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import os

class RobotSpawner(Node):
    def __init__(self):
        super().__init__('robot_spawner')

        # Create client for spawn service
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')

        self.spawn_robot()

    def spawn_robot(self):
        """Spawn robot in Gazebo"""
        # Read robot URDF
        urdf_path = os.path.join(
            os.getenv('AMENT_PREFIX_PATH', ''),
            'share', 'my_robot_description', 'urdf', 'robot.urdf'
        )

        with open(urdf_path, 'r') as f:
            robot_xml = f.read()

        # Create spawn request
        request = SpawnEntity.Request()
        request.name = "my_humanoid_robot"
        request.xml = robot_xml
        request.robot_namespace = "humanoid"

        # Set initial pose
        initial_pose = Pose()
        initial_pose.position.x = 0.0
        initial_pose.position.y = 0.0
        initial_pose.position.z = 1.0  # Start slightly above ground
        request.initial_pose = initial_pose

        # Send spawn request
        future = self.spawn_client.call_async(request)
        future.add_done_callback(self.spawn_callback)

    def spawn_callback(self, future):
        """Handle spawn response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Robot spawned successfully!')
            else:
                self.get_logger().error(f'Failed to spawn robot: {response.status_message}')
        except Exception as e:
            self.get_logger().error(f'Spawn service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    spawner = RobotSpawner()

    try:
        rclpy.spin(spawner)
    except KeyboardInterrupt:
        spawner.get_logger().info('Shutting down robot spawner')
    finally:
        spawner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch Files for Gazebo Integration

```python
# launch/gazebo_simulation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'humanoid_world.world'
            ])
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open(
                PathJoinSubstitution([
                    FindPackageShare('my_robot_description'),
                    'urdf',
                    'robot.urdf'
                ])
            ).read()
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Joint state publisher (for GUI)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_gui': True}]
    )

    # Delay spawn until Gazebo is ready
    delayed_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gazebo,
            on_exit=[spawn_entity],
        )
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        delayed_spawn
    ])
```

## Gazebo Plugins for Robotics

### ROS Control Plugin

The gazebo_ros_control plugin enables ROS 2 control of simulated robots:

```xml
<!-- In robot URDF/XACRO -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    <legacyModeNS>true</legacyModeNS>
  </plugin>
</gazebo>
```

### Sensor Plugins

#### IMU Plugin

```xml
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
```

#### Camera Plugin

```xml
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
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head_camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100</max_depth>
      <robot_namespace>/humanoid_robot</robot_namespace>
    </plugin>
  </sensor>
</gazebo>
```

#### Force/Torque Sensor Plugin

```xml
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
```

## Advanced Gazebo Features

### Multi-Robot Simulation

Simulating multiple robots in the same environment:

```xml
<!-- worlds/multi_robot.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="multi_robot_world">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Robot 1 -->
    <model name="robot1">
      <pose>0 0 1 0 0 0</pose>
      <!-- Robot 1 URDF content -->
    </model>

    <!-- Robot 2 -->
    <model name="robot2">
      <pose>2 0 1 0 0 0</pose>
      <!-- Robot 2 URDF content -->
    </model>

    <!-- Robot 3 -->
    <model name="robot3">
      <pose>4 0 1 0 0 0</pose>
      <!-- Robot 3 URDF content -->
    </model>

    <physics name="ode" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

### Custom Controllers

Creating custom Gazebo controllers:

```cpp
// include/my_robot_gazebo/CustomController.hh
#ifndef CUSTOM_CONTROLLER_HH
#define CUSTOM_CONTROLLER_HH

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace gazebo
{
  class CustomController : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
    public: void OnUpdate(const common::UpdateInfo & _info);

    private: physics::ModelPtr model;
    private: event::ConnectionPtr update_connection;
    private: rclcpp::Node::SharedPtr node;
    private: rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr command_sub;
    private: std::vector<physics::JointPtr> joints;
    private: std::vector<double> joint_targets;
  };
}

#endif
```

```cpp
// src/CustomController.cc
#include "my_robot_gazebo/CustomController.hh"
#include <ignition/math/Pose3.hh>

namespace gazebo
{
  void CustomController::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    this->model = _model;

    // Initialize ROS 2 node
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }

    this->node = rclcpp::Node::make_shared("gazebo_custom_controller");

    // Subscribe to joint commands
    this->command_sub = this->node->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/joint_commands",
      10,
      std::bind(&CustomController::CommandCallback, this, std::placeholders::_1));

    // Get joints from model
    for (const auto & joint_name : {"joint1", "joint2", "joint3"}) {
      auto joint = this->model->GetJoint(joint_name);
      if (joint) {
        this->joints.push_back(joint);
        this->joint_targets.push_back(0.0);
      }
    }

    // Connect to update event
    this->update_connection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&CustomController::OnUpdate, this, std::placeholders::_1));
  }

  void CustomController::CommandCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    for (size_t i = 0; i < std::min(msg->data.size(), this->joint_targets.size()); ++i) {
      this->joint_targets[i] = msg->data[i];
    }
  }

  void CustomController::OnUpdate(const common::UpdateInfo & _info)
  {
    // Update joint positions
    for (size_t i = 0; i < this->joints.size(); ++i) {
      this->joints[i]->SetPosition(0, this->joint_targets[i]);
    }

    // Process ROS callbacks
    rclcpp::spin_some(this->node);
  }
}

GZ_REGISTER_MODEL_PLUGIN(gazebo::CustomController)
```

### Sensor Fusion in Simulation

Combining multiple sensors for enhanced perception:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Initialize state estimate
        self.position = np.zeros(3)  # x, y, z
        self.orientation = np.array([0, 0, 0, 1])  # quaternion
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        # Covariance matrices
        self.pose_covariance = np.eye(6) * 0.1
        self.twist_covariance = np.eye(6) * 0.1

        # Subscribers for different sensors
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Publisher for fused state
        self.fused_state_pub = self.create_publisher(
            Odometry, '/fused_odom', 10)

        # Timer for state publishing
        self.publish_timer = self.create_timer(0.01, self.publish_fused_state)

    def imu_callback(self, msg):
        """Process IMU data for attitude estimation"""
        # Update orientation using gyroscope integration
        dt = 0.01  # Assuming 100Hz update rate
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Integrate angular velocity to update orientation
        omega_norm = np.linalg.norm(gyro)
        if omega_norm > 1e-6:
            # Convert to quaternion increment
            half_angle = omega_norm * dt / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)

            dq = np.array([
                sin_half * gyro[0] / omega_norm,
                sin_half * gyro[1] / omega_norm,
                sin_half * gyro[2] / omega_norm,
                cos_half
            ])

            # Apply quaternion multiplication
            self.orientation = self.quaternion_multiply(self.orientation, dq)
            # Normalize quaternion
            self.orientation /= np.linalg.norm(self.orientation)

        # Update linear acceleration in world frame
        accel_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Convert to world frame
        rot_matrix = R.from_quat(self.orientation).as_matrix()
        accel_world = rot_matrix @ accel_body

        # Integrate acceleration to get velocity and position
        self.velocity += accel_world * dt
        self.position += self.velocity * dt

    def joint_callback(self, msg):
        """Process joint state data for odometry"""
        # Use forward kinematics to estimate position based on joint angles
        # This is a simplified example - in practice, you'd implement
        # full forward kinematics for your specific robot
        pass

    def odom_callback(self, msg):
        """Process wheel odometry data"""
        # Update position estimate with wheel odometry
        self.position[0:2] = [
            msg.pose.pose.position.x,
            msg.pose.pose.y
        ]

        # Update orientation with odometry orientation
        self.orientation = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])

    def publish_fused_state(self):
        """Publish fused state estimate"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set pose
        odom_msg.pose.pose.position.x = self.position[0]
        odom_msg.pose.pose.position.y = self.position[1]
        odom_msg.pose.pose.position.z = self.position[2]

        odom_msg.pose.pose.orientation.x = self.orientation[0]
        odom_msg.pose.pose.orientation.y = self.orientation[1]
        odom_msg.pose.pose.orientation.z = self.orientation[2]
        odom_msg.pose.pose.orientation.w = self.orientation[3]

        # Set covariance
        odom_msg.pose.covariance = self.pose_covariance.flatten().tolist()

        # Set twist (velocity)
        odom_msg.twist.twist.linear.x = self.velocity[0]
        odom_msg.twist.twist.linear.y = self.velocity[1]
        odom_msg.twist.twist.linear.z = self.velocity[2]

        odom_msg.twist.twist.angular.x = self.angular_velocity[0]
        odom_msg.twist.twist.angular.y = self.angular_velocity[1]
        odom_msg.twist.twist.angular.z = self.angular_velocity[2]

        odom_msg.twist.covariance = self.twist_covariance.flatten().tolist()

        self.fused_state_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down sensor fusion node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Physics Optimization

#### Reducing Computational Load

```xml
<!-- In world file -->
<physics name="fast_physics" type="ode">
  <!-- Larger timestep for faster simulation -->
  <max_step_size>0.002</max_step_size>

  <!-- Lower update rate for non-critical applications -->
  <real_time_update_rate>500</real_time_update_rate>

  <!-- Reduce solver iterations -->
  <ode>
    <solver>
      <iters>5</iters>  <!-- Reduced from default 20 -->
      <sor>1.0</sor>    <!-- Reduced from default 1.3 -->
    </solver>
  </ode>
</physics>
```

#### Simplified Collision Models

```xml
<!-- Use simpler collision geometry for performance -->
<link name="simplified_link">
  <!-- Use box instead of complex mesh for collision -->
  <collision name="collision">
    <geometry>
      <box>
        <size>0.2 0.1 0.1</size>
      </box>
    </geometry>
  </collision>

  <!-- Keep detailed mesh for visualization -->
  <visual name="visual">
    <geometry>
      <mesh>
        <uri>model://my_robot/meshes/detail_link.stl</uri>
      </mesh>
    </geometry>
  </visual>
</link>
```

### Visualization Optimization

#### Reducing Render Load

```bash
# Launch Gazebo without GUI for headless simulation
gzserver --verbose worlds/empty.world

# Or run simulation separately from visualization
# Terminal 1:
gzserver --verbose worlds/empty.world

# Terminal 2:
gzclient
```

#### Level of Detail (LOD)

```xml
<model name="detailed_model">
  <link name="visual_link">
    <visual name="visual">
      <!-- Multiple LOD levels -->
      <level>
        <threshold>10</threshold>  <!-- Distance in meters -->
        <mesh>
          <uri>model://my_robot/meshes/lod1.dae</uri>
        </mesh>
      </level>
      <level>
        <threshold>50</threshold>
        <mesh>
          <uri>model://my_robot/meshes/lod2.dae</uri>
        </mesh>
      </level>
      <level>
        <threshold>100</threshold>
        <mesh>
          <uri>model://my_robot/meshes/lod3.dae</uri>
        </mesh>
      </level>
    </visual>
  </link>
</model>
```

## Debugging and Troubleshooting

### Common Issues and Solutions

#### 1. Robot Falling Through Ground

**Symptoms**: Robot falls through the ground plane or other static objects.

**Causes and Solutions**:
- Missing collision geometries in URDF
- Incorrect inertial properties
- Physics parameters too loose

```xml
<!-- Ensure proper collision and inertial definitions -->
<link name="base_link">
  <collision>
    <geometry>
      <box>
        <size>0.5 0.5 0.1</size>
      </box>
    </geometry>
  </collision>
  <inertial>
    <mass value="10.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0"
             iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

#### 2. Joint Oscillation

**Symptoms**: Joints oscillate uncontrollably.

**Causes and Solutions**:
- High control gains
- Physics parameters need tuning
- Inertia mismatch

```xml
<!-- Tune physics parameters -->
<joint name="oscillating_joint" type="revolute">
  <dynamics>
    <damping>1.0</damping>    <!-- Add damping -->
    <friction>0.1</friction>  <!-- Add friction -->
  </dynamics>
</joint>
```

#### 3. Simulation Speed Issues

**Symptoms**: Simulation runs too slow or too fast.

**Solutions**:
- Adjust `real_time_factor` in physics configuration
- Reduce model complexity
- Optimize control loop frequencies

### Diagnostic Tools

#### Checking Model States

```bash
# Get model states
ros2 service call /gazebo/get_model_state gazebo_msgs/srv/GetModelState \
  "{name: 'my_robot', relative_entity_name: 'world'}"

# Get link states
ros2 service call /gazebo/get_link_state gazebo_msgs/srv/GetLinkState \
  "{link_name: 'base_link', relative_entity_name: 'world'}"
```

#### Monitoring Performance

```python
# Performance monitoring node
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetPhysicsProperties
from std_msgs.msg import Float32
import time

class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        self.physics_client = self.create_client(
            GetPhysicsProperties, '/gazebo/get_physics_properties')

        self.performance_pub = self.create_publisher(
            Float32, '/simulation_performance', 10)

        self.timer = self.create_timer(1.0, self.monitor_performance)

        self.last_time = time.time()

    def monitor_performance(self):
        """Monitor simulation performance"""
        current_time = time.time()
        elapsed = current_time - self.last_time

        # Calculate simulation speed factor
        if self.physics_client.service_is_ready():
            future = self.physics_client.call_async(GetPhysicsProperties.Request())
            # Process response to get real-time factor

        # Publish performance metric
        perf_msg = Float32()
        perf_msg.data = elapsed  # Time per iteration
        self.performance_pub.publish(perf_msg)

        self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    monitor = PerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

### 1. Progressive Complexity

Start with simple models and gradually add complexity:

1. Basic geometric shapes
2. Add detailed visual meshes
3. Implement realistic physics
4. Add sensors and controllers

### 2. Parameter Management

Use configuration files for physics parameters:

```yaml
# config/gazebo_params.yaml
gazebo:
  physics:
    max_step_size: 0.001
    real_time_factor: 1.0
    solver_iterations: 50
  sensors:
    imu_rate: 100
    camera_rate: 30
    lidar_rate: 10
```

### 3. Model Organization

Keep simulation models separate from real robot descriptions:

```
robot_description/
├── urdf/
│   ├── robot_real.urdf.xacro  # For real robot
│   └── robot_sim.urdf.xacro   # For simulation
├── meshes/
└── materials/
```

### 4. Validation Process

Always validate simulation results with real-world tests:

1. Simulate simple, predictable behaviors
2. Compare with analytical solutions
3. Validate with real robot data when available
4. Document reality gap characteristics

## Summary

Gazebo provides a comprehensive simulation environment for Physical AI and humanoid robotics development. By understanding its architecture, configuration options, and integration with ROS 2, you can create realistic and useful simulation environments. The key is to balance physical accuracy with computational performance, while ensuring that simulation results are validated against real-world behavior. Proper use of Gazebo can significantly accelerate robot development while reducing costs and safety risks.