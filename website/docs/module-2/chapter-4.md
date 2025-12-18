---
title: Chapter 4 - URDF/SDF Integration with Simulators
description: Comprehensive guide to integrating URDF models with simulation environments
sidebar_position: 4
---

# Chapter 4: URDF/SDF Integration with Simulators

## Learning Objectives

After completing this chapter, you should be able to:
- Understand the relationship between URDF and SDF formats
- Create proper Gazebo integration for URDF models
- Implement simulation-specific extensions to URDF
- Validate URDF/SDF models for simulation compatibility
- Troubleshoot common integration issues

## Introduction

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are complementary formats that serve different purposes in robotics development. URDF is primarily designed for describing robot kinematics and physical properties for ROS applications, while SDF is designed for simulation environments. However, for effective Physical AI development, it's crucial to understand how to properly integrate these formats to create simulation-ready robot models.

## Understanding URDF and SDF

### URDF vs SDF: Key Differences

| Aspect | URDF | SDF |
|--------|------|-----|
| **Primary Purpose** | Robot description for ROS | Simulation environment description |
| **Scope** | Single robot or mechanism | Entire simulation world |
| **Physics** | Basic inertial properties | Comprehensive physics engine integration |
| **Sensors** | Limited sensor description | Rich sensor simulation capabilities |
| **Plugins** | None | Extensive plugin system |
| **Environments** | Static scenes | Dynamic environments with lighting, physics, etc. |

### Complementary Nature

While URDF and SDF serve different purposes, they work together in simulation:

```xml
<!-- URDF describes the robot -->
<robot name="humanoid_robot">
  <link name="base_link">
    <inertial>...</inertial>
    <visual>...</visual>
    <collision>...</collision>
  </link>
  <joint name="joint1" type="revolute">...</joint>
</robot>

<!-- SDF can contain URDF or extend it with simulation features -->
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- URDF content goes here OR is included -->
    <include>
      <uri>model://humanoid_robot/model.urdf</uri>
    </include>

    <!-- Simulation-specific extensions -->
    <plugin name="controller" filename="libgazebo_ros_control.so">
      ...
    </plugin>
  </model>
</sdf>
```

## Gazebo-Specific Extensions to URDF

### The `<gazebo>` Tag

The `<gazebo>` tag allows you to add simulation-specific content to your URDF files:

```xml
<?xml version="1.0"?>
<robot name="simulated_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Standard URDF content -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.3 0.3 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo-specific material definition -->
  <gazebo reference="base_link">
    <material>Gazebo/Orange</material>
    <!-- Contact properties -->
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Joint definition -->
  <joint name="base_to_arm" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="arm_link">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
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
  </link>

  <!-- Gazebo plugin for the joint -->
  <gazebo reference="base_to_arm">
    <implicitSpringDamper>1</implicitSpringDamper>
  </gazebo>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Sensor definition -->
  <gazebo reference="base_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

</robot>
```

### Types of Gazebo Extensions

#### 1. Link-Specific Extensions

```xml
<gazebo reference="link_name">
  <!-- Material properties -->
  <material>Gazebo/Blue</material>

  <!-- Contact properties -->
  <mu1>0.5</mu1>           <!-- Primary friction coefficient -->
  <mu2>0.5</mu2>           <!-- Secondary friction coefficient -->
  <kp>1000000.0</kp>       <!-- Spring stiffness -->
  <kd>100.0</kd>           <!-- Damping coefficient -->

  <!-- Self-collision -->
  <self_collide>true</self_collide>

  <!-- Gravity -->
  <gravity>true</gravity>

  <!-- Linear and angular damping -->
  <linear_damping>0.0</linear_damping>
  <angular_damping>0.0</angular_damping>

  <!-- Maximum contacts -->
  <max_contacts>10</max_contacts>
</gazebo>
```

#### 2. Joint-Specific Extensions

```xml
<gazebo reference="joint_name">
  <!-- Spring and damper -->
  <implicitSpringDamper>1</implicitSpringDamper>

  <!-- Spring reference -->
  <springReference>0.0</springReference>

  <!-- Spring stiffness -->
  <springStiffness>0.0</springStiffness>

  <!-- Damping coefficient -->
  <springDissipation>0.0</springDissipation>
</gazebo>
```

#### 3. Global Extensions

```xml
<gazebo>
  <!-- ROS control plugin -->
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
  </plugin>

  <!-- Joint state publisher -->
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <joint_name>joint1</joint_name>
    <joint_name>joint2</joint_name>
  </plugin>
</gazebo>
```

## Advanced Integration Techniques

### Xacro Macros for Simulation Integration

Using Xacro to create reusable simulation components:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_sim">

  <!-- Property definitions -->
  <xacro:property name="PI" value="3.1415926535897931" />

  <!-- Macro for adding simulation plugins -->
  <xacro:macro name="gazebo_ros_control_plugin">
    <gazebo>
      <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
        <robotNamespace>/humanoid_robot</robotNamespace>
      </plugin>
    </gazebo>
  </xacro:macro>

  <!-- Macro for adding IMU sensor -->
  <xacro:macro name="imu_sensor" params="name parent_link topic update_rate *origin">
    <gazebo reference="${parent_link}">
      <sensor name="${name}" type="imu">
        <always_on>true</always_on>
        <update_rate>${update_rate}</update_rate>
        <visualize>false</visualize>
        <topic>${topic}</topic>
        <pose>0 0 0 0 0 0</pose>
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
          </angular_velocity>
          <linear_acceleration>
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
  </xacro:macro>

  <!-- Macro for adding camera sensor -->
  <xacro:macro name="camera_sensor" params="name parent_link topic update_rate *origin">
    <gazebo reference="${parent_link}">
      <sensor name="${name}" type="camera">
        <update_rate>${update_rate}</update_rate>
        <camera name="head">
          <horizontal_fov>1.3962634</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.05</near>
            <far>300</far>
          </clip>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.007</stddev>
          </noise>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <frame_name>${parent_link}_optical_frame</frame_name>
          <min_depth>0.05</min_depth>
          <max_depth>300</max_depth>
          <point_cloud_cutoff>0.05</point_cloud_cutoff>
          <point_cloud_cutoff_max>300</point_cloud_cutoff_max>
          <always_on>true</always_on>
          <update_rate>${update_rate}</update_rate>
          <image_topic_name>${topic}/image_raw</image_topic_name>
          <camera_info_topic_name>${topic}/camera_info</camera_info_topic_name>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:macro>

  <!-- Robot body -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Apply IMU macro -->
  <xacro:imu_sensor name="base_imu" parent_link="base_link"
                    topic="imu/data" update_rate="100">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:imu_sensor>

  <!-- Apply camera macro -->
  <xacro:camera_sensor name="head_camera" parent_link="base_link"
                       topic="camera/image_raw" update_rate="30">
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </xacro:camera_sensor>

  <!-- Add ROS control plugin -->
  <xacro:gazebo_ros_control_plugin/>

</robot>
```

### Transmission Elements for Control

Adding transmission elements for ROS control integration:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="controlled_robot">

  <!-- Joint definitions -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="upper_arm"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Transmission definition -->
  <transmission name="shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="shoulder_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="shoulder_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo plugin for transmission -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
      <controlPeriod>0.001</controlPeriod>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

</robot>
```

## SDF World Integration

### Converting URDF to SDF for Simulation

Sometimes it's beneficial to convert your URDF to SDF format for more complex simulation scenarios:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- Include URDF as a model -->
    <include>
      <uri>model://humanoid_robot/urdf/robot.urdf</uri>
    </include>

    <!-- Additional SDF elements -->
    <static>false</static>
    <self_collide>false</self_collide>
    <enable_wind>false</enable_wind>

    <!-- Model-level properties -->
    <pose>0 0 1 0 0 0</pose>

    <!-- Plugins -->
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </model>

  <!-- Additional world elements -->
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

  <physics name="default_physics" default="0" type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <gravity>0 0 -9.8</gravity>
  </physics>

</sdf>
```

### Multi-Model SDF Worlds

Creating complex simulation environments with multiple models:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="complex_humanoid_world">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Humanoid Robot 1 -->
    <model name="humanoid_1">
      <pose>-2 0 0.5 0 0 0</pose>
      <include>
        <uri>model://humanoid_robot/urdf/robot.urdf</uri>
      </include>
    </model>

    <!-- Humanoid Robot 2 -->
    <model name="humanoid_2">
      <pose>2 0 0.5 0 0 0</pose>
      <include>
        <uri>model://humanoid_robot/urdf/robot.urdf</uri>
      </include>
    </model>

    <!-- Environment obstacles -->
    <model name="table">
      <pose>0 3 0.5 0 0 0</pose>
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

    <!-- Physics configuration -->
    <physics name="ode" default="0" type="ode">
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

    <!-- Wind -->
    <wind>
      <linear_velocity>0.1 0 0</linear_velocity>
    </wind>

  </world>
</sdf>
```

## Sensor Integration

### Camera Sensors

Integrating camera sensors with proper ROS interfaces:

```xml
<gazebo reference="camera_mount">
  <sensor name="rgb_camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.05</near>
        <far>300</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.05</min_depth>
      <max_depth>300</max_depth>
      <point_cloud_cutoff>0.05</point_cloud_cutoff>
      <point_cloud_cutoff_max>300</point_cloud_cutoff_max>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <image_topic_name>camera/image_raw</image_topic_name>
      <camera_info_topic_name>camera/camera_info</camera_info_topic_name>
      <point_cloud_topic_name>camera/depth/points</point_cloud_topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Sensors

Integrating LiDAR sensors for perception:

```xml
<gazebo reference="lidar_mount">
  <sensor name="lidar_2d" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="gpu_lidar" filename="libgazebo_ros_gpu_lidar.so">
      <frame_name>lidar_frame</frame_name>
      <min_range>0.1</min_range>
      <max_range>30.0</max_range>
      <update_rate>10</update_rate>
      <topic>scan</topic>
    </plugin>
  </sensor>
</gazebo>
```

### Force/Torque Sensors

Integrating force/torque sensors for manipulation:

```xml
<gazebo reference="left_ankle_joint">
  <sensor name="left_ankle_ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
    <plugin name="ft_sensor" filename="libgazebo_ros_ft_sensor.so">
      <frame_name>left_ankle_link</frame_name>
      <topic>left_ankle/ft_sensor</topic>
    </plugin>
  </sensor>
</gazebo>
```

## Controller Integration

### Joint Position Controllers

Setting up joint position controllers:

```xml
<!-- In URDF with transmission -->
<transmission name="left_shoulder_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_shoulder_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- Controller configuration -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <controlPeriod>0.001</controlPeriod>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>
```

```yaml
# config/controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_arm_controller:
      type: position_controllers/JointGroupPositionController

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_joint
      - left_elbow_joint
      - left_wrist_joint
```

### Whole-Body Controllers

For humanoid robots, more complex controllers may be needed:

```xml
<!-- Whole-body controller plugin -->
<gazebo>
  <plugin name="whole_body_controller" filename="libwhole_body_controller.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <com_topic>center_of_mass</com_topic>
    <zmp_topic>zero_moment_point</zmp_topic>
    <balance_topic>balance_control</balance_topic>
  </plugin>
</gazebo>
```

## Validation and Testing

### URDF Validation Tools

Before integrating with simulation, validate your URDF:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Check for joint limits and kinematics
ros2 run urdf_parser check_urdf /path/to/robot.urdf

# Visualize in RViz
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat /path/to/robot.urdf)"
```

### SDF Validation

Validate SDF files:

```bash
# Use gazebo to check SDF validity
gz sdf -c /path/to/world.sdf

# Or use the command line tool
gz sdf -p /path/to/world.sdf
```

### Simulation Testing

Create a test script to validate simulation integration:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import time
import math

class SimulationValidator(Node):
    def __init__(self):
        super().__init__('simulation_validator')

        # Publishers for commanding the robot
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10)

        # Subscribers for sensor feedback
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Timer for validation tests
        self.test_timer = self.create_timer(0.1, self.run_validation_tests)

        # State tracking
        self.current_joint_states = None
        self.imu_data = None
        self.test_counter = 0

        self.get_logger().info('Simulation Validator initialized')

    def joint_state_callback(self, msg):
        """Receive joint state feedback"""
        self.current_joint_states = msg

    def imu_callback(self, msg):
        """Receive IMU feedback"""
        self.imu_data = msg

    def run_validation_tests(self):
        """Run various validation tests"""
        self.test_counter += 1

        if self.test_counter == 1:
            self.test_basic_movement()
        elif self.test_counter == 10:  # After 1 second
            self.test_sensor_feedback()
        elif self.test_counter == 20:  # After 2 seconds
            self.test_stability()
        elif self.test_counter > 30:
            self.test_counter = 0  # Reset for continuous testing

    def test_basic_movement(self):
        """Test basic joint movement"""
        if self.current_joint_states is None:
            return

        cmd = JointState()
        cmd.name = self.current_joint_states.name
        cmd.position = []

        # Send a simple sinusoidal command to first joint
        for i, name in enumerate(self.current_joint_states.name):
            if i == 0:  # Move first joint
                cmd.position.append(math.sin(self.get_clock().now().nanoseconds / 1e9) * 0.5)
            else:
                cmd.position.append(0.0)  # Keep others neutral

        self.joint_cmd_pub.publish(cmd)
        self.get_logger().info('Sent basic movement command')

    def test_sensor_feedback(self):
        """Verify sensor feedback is working"""
        if self.current_joint_states is not None:
            self.get_logger().info(f'Joint states received: {len(self.current_joint_states.name)} joints')

        if self.imu_data is not None:
            self.get_logger().info('IMU data is being received')

    def test_stability(self):
        """Test overall system stability"""
        if self.current_joint_states is not None:
            # Check for reasonable joint positions (not NaN or extremely large)
            for pos in self.current_joint_states.position:
                if math.isnan(pos) or abs(pos) > 100:
                    self.get_logger().error(f'Unreasonable joint position detected: {pos}')
                    return

            self.get_logger().info('Stability check passed - joint positions are reasonable')

def main(args=None):
    rclpy.init(args=args)
    validator = SimulationValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Shutting down simulation validator')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Common Integration Issues and Solutions

### Issue 1: Robot Falls Through Ground

**Cause**: Missing or incorrect collision geometries.

**Solution**:
```xml
<!-- Ensure all links have collision geometries -->
<link name="link_name">
  <collision>
    <geometry>
      <box>
        <size>0.1 0.1 0.1</size>  <!-- Or appropriate geometry -->
      </box>
    </geometry>
  </collision>
  <visual>
    <geometry>
      <mesh>
        <uri>model://meshes/link_mesh.dae</uri>
      </mesh>
    </geometry>
  </visual>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

### Issue 2: Joint Oscillation

**Cause**: Missing damping or improper physics parameters.

**Solution**:
```xml
<joint name="problematic_joint" type="revolute">
  <dynamics>
    <damping>1.0</damping>      <!-- Add damping -->
    <friction>0.1</friction>    <!-- Add static friction -->
  </dynamics>
</joint>

<!-- Or in Gazebo extension -->
<gazebo reference="problematic_joint">
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

### Issue 3: Sensor Not Publishing Data

**Cause**: Plugin not loaded or wrong topic names.

**Solution**:
```xml
<gazebo reference="sensor_link">
  <sensor name="my_camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
    </camera>
    <!-- Make sure plugin is specified correctly -->
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>sensor_link_optical</frame_name>
      <topic>camera/image_raw</topic>
      <update_rate>30</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Issue 4: ROS Control Not Working

**Cause**: Missing transmission or incorrect plugin configuration.

**Solution**:
```xml
<!-- Make sure transmission is defined -->
<transmission name="joint_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="motor_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </actuator>
</transmission>

<!-- And plugin is configured -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/robot_name</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>
```

## Performance Optimization

### Physics Optimization

```xml
<!-- Optimize physics parameters for performance -->
<physics name="fast_physics" type="ode">
  <max_step_size>0.002</max_step_size>  <!-- Increase from 0.001 if possible -->
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>500</real_time_update_rate>  <!-- Reduce from 1000 if needed -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>  <!-- Reduce if possible -->
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
```

### Model Optimization

```xml
<!-- Use simplified collision geometries for performance -->
<link name="performance_link">
  <!-- Use simple box for collision -->
  <collision>
    <geometry>
      <box>
        <size>0.2 0.1 0.1</size>
      </box>
    </geometry>
  </collision>

  <!-- Use detailed mesh for visualization -->
  <visual>
    <geometry>
      <mesh>
        <uri>model://detailed_mesh.stl</uri>
      </mesh>
    </geometry>
  </visual>
</link>
```

## Debugging Integration

### Checking Loaded Models

```bash
# List all models in simulation
gz model -m

# Check model states
gz model -s -m <model_name>

# Check topics
ros2 topic list | grep -E "(joint|imu|camera|scan)"
```

### Verifying Sensor Data

```bash
# Check if sensor topics are being published
ros2 topic echo /camera/image_raw --field header.stamp
ros2 topic echo /imu/data --field orientation
ros2 topic echo /scan --field ranges
```

## Best Practices

### 1. Separation of Concerns

Keep simulation-specific content separate from robot description:

```xml
<!-- robot.urdf.xacro - Pure robot description -->
<xacro:macro name="humanoid_robot">
  <!-- Pure URDF content -->
</xacro:macro>

<!-- robot.gazebo.xacro - Simulation extensions -->
<xacro:macro name="humanoid_robot_gazebo_extensions">
  <!-- Gazebo-specific content -->
</xacro:macro>
```

### 2. Parameterization

Use parameters to make models configurable:

```xml
<xacro:property name="robot_name" value="$(arg robot_name)" default="humanoid_robot" />
<xacro:property name="namespace" value="$(arg namespace)" default="" />

<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>${namespace}/${robot_name}</robotNamespace>
  </plugin>
</gazebo>
```

### 3. Validation Checks

Always validate models before deployment:

- Check for self-collisions
- Verify joint limits
- Test sensor functionality
- Validate physics behavior

## Summary

Proper URDF/SDF integration is crucial for effective Physical AI simulation. The key is understanding how these formats complement each other: URDF describes the robot's physical properties and kinematics, while SDF extends this with simulation-specific features like plugins, sensors, and environmental interactions. By following the patterns and best practices outlined in this chapter, you can create robust simulation environments that accurately represent your physical robot systems while providing the rich simulation capabilities needed for Physical AI development.