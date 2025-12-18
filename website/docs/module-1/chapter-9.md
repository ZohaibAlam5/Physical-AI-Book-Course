---
title: Chapter 9 - URDF Modeling Exercises
description: Practical exercises for creating and refining URDF models for humanoid robots
sidebar_position: 9
---

# Chapter 9: URDF Modeling Exercises

## Learning Objectives

After completing this chapter, you should be able to:
- Create URDF models for different humanoid robot configurations
- Validate URDF models for correctness and physical accuracy
- Optimize URDF models for simulation and real-world performance

## Introduction

This chapter provides hands-on exercises to practice creating and refining URDF models for humanoid robots. These exercises will help you gain practical experience with URDF modeling, from basic concepts to advanced techniques.

## Exercise 1: Basic Humanoid Model

Create a simple humanoid model with torso, head, arms, and legs.

### Exercise Description
Create a URDF file for a basic humanoid robot with the following specifications:
- Torso: Box shape, 0.3×0.2×0.5 meters
- Head: Sphere, radius 0.1 meters
- Arms: Each with upper arm (cylinder, 0.05m radius, 0.3m length) and lower arm (cylinder, 0.04m radius, 0.25m length)
- Legs: Each with upper leg (cylinder, 0.06m radius, 0.4m length), lower leg (cylinder, 0.05m radius, 0.4m length), and foot (box, 0.15×0.08×0.05 meters)

### Solution

```xml
<?xml version="1.0"?>
<robot name="basic_humanoid_exercise">
  <!-- Materials -->
  <material name="red">
    <color rgba="0.8 0.2 0.2 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.2 0.8 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="2.0"/>
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
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0" effort="15.0" velocity="2.0"/>
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
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.2 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="2.0"/>
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
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0" effort="15.0" velocity="2.0"/>
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
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.05 -0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="1.0" effort="30.0" velocity="1.5"/>
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

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="30.0" velocity="1.5"/>
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

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>
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
  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.05 -0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="1.0" effort="30.0" velocity="1.5"/>
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

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="30.0" velocity="1.5"/>
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

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>
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
</robot>
```

## Exercise 2: Adding Sensors to URDF

Enhance the basic humanoid model by adding sensors.

### Exercise Description
Add the following sensors to the basic humanoid model:
- IMU in the torso
- Camera in the head
- Force/torque sensors in the ankles
- Range sensors in the feet

### Solution Extension

```xml
  <!-- IMU sensor in torso -->
  <gazebo reference="torso">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <noise>
          <type>gaussian</type>
          <rate>
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </rate>
          <accel>
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </accel>
        </noise>
      </imu>
    </sensor>
  </gazebo>

  <!-- Camera in head -->
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
    </sensor>
  </gazebo>

  <!-- Force/Torque sensors in ankles -->
  <gazebo reference="left_ankle_joint">
    <sensor name="left_ankle_ft_sensor" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
    </sensor>
  </gazebo>

  <gazebo reference="right_ankle_joint">
    <sensor name="right_ankle_ft_sensor" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
    </sensor>
  </gazebo>

  <!-- Range sensors in feet -->
  <gazebo reference="left_foot">
    <sensor name="left_foot_contact_sensor" type="contact">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <contact>
        <collision>left_foot_collision</collision>
      </contact>
    </sensor>
  </gazebo>

  <gazebo reference="right_foot">
    <sensor name="right_foot_contact_sensor" type="contact">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <contact>
        <collision>right_foot_collision</collision>
      </contact>
    </sensor>
  </gazebo>
```

## Exercise 3: URDF with Transmission Elements

Add transmission elements to enable control of the robot joints.

### Exercise Description
Add transmission elements to the basic humanoid model to enable ROS control.

### Solution Extension

```xml
  <!-- Transmission for left shoulder joint -->
  <transmission name="left_shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_shoulder_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for right shoulder joint -->
  <transmission name="right_shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_shoulder_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_shoulder_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for left elbow joint -->
  <transmission name="left_elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_elbow_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_elbow_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for right elbow joint -->
  <transmission name="right_elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_elbow_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_elbow_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for left hip joint -->
  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for right hip joint -->
  <transmission name="right_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for left knee joint -->
  <transmission name="left_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_knee_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for right knee joint -->
  <transmission name="right_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_knee_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_knee_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for left ankle joint -->
  <transmission name="left_ankle_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_ankle_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_ankle_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Transmission for right ankle joint -->
  <transmission name="right_ankle_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_ankle_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_ankle_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
```

## Exercise 4: Using Xacro for Modular URDF

Convert the URDF to Xacro format for better organization and reusability.

### Exercise Description
Convert the basic humanoid URDF to Xacro format, using macros to modularize common elements like limbs.

### Solution (Xacro format)

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="basic_humanoid_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_size" value="0.3 0.2 0.5" />
  <xacro:property name="head_radius" value="0.1" />
  <xacro:property name="upper_arm_length" value="0.3" />
  <xacro:property name="upper_arm_radius" value="0.05" />
  <xacro:property name="lower_arm_length" value="0.25" />
  <xacro:property name="lower_arm_radius" value="0.04" />
  <xacro:property name="upper_leg_length" value="0.4" />
  <xacro:property name="upper_leg_radius" value="0.06" />
  <xacro:property name="lower_leg_length" value="0.4" />
  <xacro:property name="lower_leg_radius" value="0.05" />
  <xacro:property name="foot_size" value="0.15 0.08 0.05" />

  <!-- Materials -->
  <material name="red">
    <color rgba="0.8 0.2 0.2 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.2 0.8 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Macro for arm -->
  <xacro:macro name="arm" params="side parent x_offset y_offset z_offset">
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${x_offset} ${y_offset} ${z_offset}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="-1.57" upper="1.57" effort="20.0" velocity="2.0"/>
    </joint>

    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 -${upper_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="0" effort="15.0" velocity="2.0"/>
    </joint>

    <link name="${side}_lower_arm">
      <visual>
        <geometry>
          <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Macro for leg -->
  <xacro:macro name="leg" params="side parent x_offset y_offset z_offset">
    <joint name="${side}_hip_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_leg"/>
      <origin xyz="${x_offset} ${y_offset} ${z_offset}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="-0.5" upper="1.0" effort="30.0" velocity="1.5"/>
    </joint>

    <link name="${side}_upper_leg">
      <visual>
        <geometry>
          <cylinder radius="${upper_leg_radius}" length="${upper_leg_length}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${upper_leg_radius}" length="${upper_leg_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="3.0"/>
        <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_upper_leg"/>
      <child link="${side}_lower_leg"/>
      <origin xyz="0 0 -${upper_leg_length}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="0" upper="1.57" effort="30.0" velocity="1.5"/>
    </joint>

    <link name="${side}_lower_leg">
      <visual>
        <geometry>
          <cylinder radius="${lower_leg_radius}" length="${lower_leg_length}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${lower_leg_radius}" length="${lower_leg_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.5"/>
        <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <joint name="${side}_ankle_joint" type="revolute">
      <parent link="${side}_lower_leg"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -${lower_leg_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>
    </joint>

    <link name="${side}_foot">
      <visual>
        <geometry>
          <box size="${foot_size}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <box size="${foot_size}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="${torso_size}"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="${torso_size}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Arms -->
  <xacro:arm side="left" parent="torso" x_offset="0.2" y_offset="0.1" z_offset="0.1"/>
  <xacro:arm side="right" parent="torso" x_offset="-0.2" y_offset="0.1" z_offset="0.1"/>

  <!-- Legs -->
  <xacro:leg side="left" parent="torso" x_offset="0.05" y_offset="-0.1" z_offset="-0.25"/>
  <xacro:leg side="right" parent="torso" x_offset="-0.05" y_offset="-0.1" z_offset="-0.25"/>

</robot>
```

## Exercise 5: URDF Validation and Debugging

Learn how to validate and debug URDF models.

### Exercise Description
Practice using URDF validation tools and debugging common issues.

### Validation Commands

```bash
# Validate URDF syntax
check_urdf /path/to/robot.urdf

# Check for joint limits
ros2 run joint_state_publisher_gui joint_state_publisher_gui --ros-args -p use_gui:=true

# Visualize in RViz
ros2 run rviz2 rviz2

# Check robot state in Gazebo
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="/path/to/robot.urdf"
```

### Common Issues and Solutions

1. **Inertia Issues**:
   - Problem: Robot falls through the ground in simulation
   - Solution: Ensure all links have properly defined inertia elements

2. **Joint Limits**:
   - Problem: Joints move beyond physical constraints
   - Solution: Set appropriate limits in joint definitions

3. **Collision Issues**:
   - Problem: Robot parts pass through each other
   - Solution: Define proper collision geometries

4. **Visual Issues**:
   - Problem: Robot appears incorrectly in visualization
   - Solution: Check visual origins and geometries

## Summary

These exercises provide practical experience in creating and refining URDF models for humanoid robots. By working through these examples, you'll develop the skills needed to model complex robots for both simulation and real-world applications. Practice these exercises to become proficient in URDF modeling.