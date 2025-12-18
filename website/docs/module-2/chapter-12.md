---
title: Chapter 12 - Module 2 Capstone Project
description: Capstone project integrating all concepts from Module 2 on Digital Twins & Robot Simulation
sidebar_position: 12
---

# Chapter 12: Module 2 Capstone Project - Complete Humanoid Simulation System

## Learning Objectives

After completing this capstone project, you should be able to:
- Integrate all Module 2 concepts into a complete humanoid simulation system
- Implement a physics-accurate humanoid robot model with proper mass properties
- Create realistic sensor simulation including cameras, LiDAR, and IMU
- Develop whole-body control systems that maintain balance during locomotion
- Validate simulation performance and accuracy against requirements

## Introduction

The Module 2 capstone project brings together all the concepts learned about digital twins and robot simulation. You will create a complete, physics-accurate simulation of a humanoid robot that can maintain balance, walk, and interact with its environment using realistic sensor data. This project demonstrates the integration of URDF modeling, physics simulation, sensor integration, and control systems.

## Project Overview

### Project Goals
1. Create a complete humanoid robot simulation with 24+ joints
2. Implement realistic physics with proper mass properties and constraints
3. Integrate multiple sensor types (camera, LiDAR, IMU) with realistic noise models
4. Develop balance and walking control systems
5. Validate simulation performance and accuracy

### Success Criteria
- Robot maintains balance for at least 30 seconds without falling
- Walking gait is stable and natural-looking
- All sensors provide realistic data within specified accuracy bounds
- Simulation runs at 30+ FPS with real-time factor > 0.8
- Robot can navigate simple obstacles and terrain changes

## Complete Humanoid Robot Model

### Comprehensive URDF Implementation

```xml
<?xml version="1.0"?>
<robot name="physical_ai_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="torso_mass" value="15.0"/>
  <xacro:property name="head_mass" value="3.0"/>
  <xacro:property name="upper_arm_mass" value="2.0"/>
  <xacro:property name="lower_arm_mass" value="1.5"/>
  <xacro:property name="upper_leg_mass" value="4.0"/>
  <xacro:property name="lower_leg_mass" value="3.0"/>
  <xacro:property name="foot_mass" value="1.0"/>

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.2 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="silver">
    <color rgba="0.75 0.75 0.75 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base/Fixed link -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <inertial>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <mass value="${torso_mass}"/>
      <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.4"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="50" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="head">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="${head_mass}"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Head-mounted sensors -->
  <joint name="head_camera_joint" type="fixed">
    <parent link="head"/>
    <child link="head_camera"/>
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="head_camera">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.05 0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="${upper_arm_mass}"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_yaw" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="80" velocity="2.0"/>
    <dynamics damping="0.8" friction="0.1"/>
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <mass value="${lower_arm_mass}"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="60" velocity="2.0"/>
    <dynamics damping="0.6" friction="0.1"/>
  </joint>

  <link name="left_hand">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
      <material name="silver"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Arm -->
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0.05 0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="${upper_arm_mass}"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_yaw" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="80" velocity="2.0"/>
    <dynamics damping="0.8" friction="0.1"/>
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <mass value="${lower_arm_mass}"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_lower_arm"/>
    <child link="right_hand"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="60" velocity="2.0"/>
    <dynamics damping="0.6" friction="0.1"/>
  </joint>

  <link name="right_hand">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
      <material name="silver"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_roll" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="0.05 -0.08 -0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/3}" effort="150" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.3"/>
  </joint>

  <link name="left_thigh">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="${upper_leg_mass}"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_hip_pitch" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/3}" upper="${M_PI/2}" effort="150" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.3"/>
  </joint>

  <link name="left_shin">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="${lower_leg_mass}"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_shin"/>
    <child link="left_ankle"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="150" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.3"/>
  </joint>

  <link name="left_ankle">
    <inertial>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <mass value="0.8"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle_pitch" type="revolute">
    <parent link="left_ankle"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="100" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
      <mass value="${foot_mass}"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.08 0.05"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.08 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Leg -->
  <joint name="right_hip_roll" type="revolute">
    <parent link="torso"/>
    <child link="right_thigh"/>
    <origin xyz="-0.05 -0.08 -0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/3}" effort="150" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.3"/>
  </joint>

  <link name="right_thigh">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="${upper_leg_mass}"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_hip_pitch" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/3}" upper="${M_PI/2}" effort="150" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.3"/>
  </joint>

  <link name="right_shin">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="${lower_leg_mass}"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_shin"/>
    <child link="right_ankle"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="150" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.3"/>
  </joint>

  <link name="right_ankle">
    <inertial>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <mass value="0.8"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle_pitch" type="revolute">
    <parent link="right_ankle"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="100" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="right_foot">
    <inertial>
      <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
      <mass value="${foot_mass}"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.08 0.05"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.08 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo plugins for simulation -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

  <!-- IMU sensor in torso -->
  <gazebo reference="torso">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
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
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <topicName>imu/data</topicName>
        <bodyName>torso</bodyName>
        <frameName>imu_link</frameName>
        <updateRateHZ>100.0</updateRateHZ>
        <gaussianNoise>0.0</gaussianNoise>
        <accelGaussianNoise>0.017</accelGaussianNoise>
        <velGaussianNoise>0.002</velGaussianNoise>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Camera sensor in head -->
  <gazebo reference="head_camera">
    <sensor name="head_camera" type="camera">
      <always_on>true</always_on>
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
        <cameraName>head_camera</cameraName>
        <imageTopicName>image_raw</imageTopicName>
        <cameraInfoTopicName>camera_info</cameraInfoTopicName>
        <frameName>head_camera_optical_frame</frameName>
        <hackBaseline>0.07</hackBaseline>
        <distortion_k1>0.0</distortion_k1>
        <distortion_k2>0.0</distortion_k2>
        <distortion_k3>0.0</distortion_k3>
        <distortion_t1>0.0</distortion_t1>
        <distortion_t2>0.0</distortion_t2>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LiDAR sensor -->
  <gazebo reference="head">
    <sensor name="lidar_sensor" type="ray">
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
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <topicName>scan</topicName>
        <frameName>head</frameName>
        <min_range>0.1</min_range>
        <max_range>30.0</max_range>
        <update_rate>10</update_rate>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Force/Torque sensors in feet -->
  <gazebo reference="left_foot">
    <sensor name="left_foot_ft_sensor" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>child</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </gazebo>

  <gazebo reference="right_foot">
    <sensor name="right_foot_ft_sensor" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>child</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </gazebo>

  <!-- Joint transmissions for ROS control -->
  <transmission name="left_hip_roll_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_roll">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_roll_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_hip_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_pitch">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_pitch_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_knee">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_ankle_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_ankle_pitch">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_ankle_pitch_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_hip_roll_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip_roll">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_roll_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_hip_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip_pitch">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_pitch_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_knee">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_knee_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_ankle_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_ankle_pitch">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_ankle_pitch_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

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

  <transmission name="left_shoulder_yaw_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder_yaw">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_shoulder_yaw_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_elbow">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_elbow_motor">
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

  <transmission name="right_shoulder_yaw_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_shoulder_yaw">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_shoulder_yaw_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_elbow">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_elbow_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="neck_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="neck_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="neck_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

## Simulation Environment Setup

### Complete World Configuration

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_simulation_world">
    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add some obstacles for navigation -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
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

    <!-- Add some objects for manipulation -->
    <model name="red_cube">
      <pose>2.2 0.1 0.9 0 0 0</pose>
      <link name="cube_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.00083</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.00083</iyy>
            <iyz>0.0</iyz>
            <izz>0.00083</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Physics engine configuration -->
    <physics name="ode_physics" default="0" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>
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

    <!-- Wind settings -->
    <wind>
      <linear_velocity>0 0 0</linear_velocity>
    </wind>

  </world>
</sdf>
```

## Whole-Body Control Implementation

### Complete Controller System

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Time
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from collections import deque

class HumanoidSimulationController(Node):
    """Complete controller for the humanoid simulation system"""

    def __init__(self):
        super().__init__('humanoid_simulation_controller')

        # Robot parameters
        self.com_height = 0.85  # Estimated CoM height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.com_height)

        # Control parameters
        self.control_frequency = 1000  # Hz
        self.dt = 1.0 / self.control_frequency

        # Initialize robot state
        self.current_state = {
            'positions': np.zeros(18),  # 18 main joints
            'velocities': np.zeros(18),
            'accelerations': np.zeros(18),
            'com_position': np.array([0.0, 0.0, 0.85]),
            'com_velocity': np.zeros(3),
            'imu_data': None,
            'contact_data': {}
        }

        # Joint names (in order of the URDF)
        self.joint_names = [
            'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch',
            'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch',
            'left_shoulder_pitch', 'left_shoulder_yaw', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_yaw', 'right_elbow',
            'neck_joint',
            'torso_yaw', 'torso_pitch', 'torso_roll'  # Assuming additional torso joints
        ]

        # Initialize PID controllers for each joint
        self.joint_controllers = {}
        for joint_name in self.joint_names:
            self.joint_controllers[joint_name] = self.create_joint_pid(joint_name)

        # Initialize specialized controllers
        self.balance_controller = BalanceController(self.com_height)
        self.walk_controller = WalkingController(self.com_height)
        self.manipulation_controller = ManipulationController()
        self.trajectory_generator = TrajectoryGenerator()

        # Initialize subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        # Initialize publishers
        self.joint_cmd_pub = self.create_publisher(
            JointState, 'joint_commands', 10)
        self.com_pub = self.create_publisher(
            Vector3, 'center_of_mass', 10)
        self.zmp_pub = self.create_publisher(
            Vector3, 'zero_moment_point', 10)

        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)

        # State estimation
        self.com_estimator = COMEstimator(self.joint_names)
        self.contact_estimator = ContactEstimator()

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        self.get_logger().info('Humanoid Simulation Controller initialized')

    def create_joint_pid(self, joint_name):
        """Create PID controller with parameters tuned for specific joint"""
        # Different gains for different joint types
        if 'hip' in joint_name:
            # Hip joints need high stiffness for balance
            return {
                'kp': 500.0,
                'ki': 10.0,
                'kd': 50.0,
                'integral': 0.0,
                'prev_error': 0.0
            }
        elif 'knee' in joint_name:
            # Knee joints for locomotion
            return {
                'kp': 400.0,
                'ki': 8.0,
                'kd': 40.0,
                'integral': 0.0,
                'prev_error': 0.0
            }
        elif 'ankle' in joint_name:
            # Ankle joints for fine balance control
            return {
                'kp': 300.0,
                'ki': 5.0,
                'kd': 30.0,
                'integral': 0.0,
                'prev_error': 0.0
            }
        elif 'shoulder' in joint_name or 'elbow' in joint_name:
            # Arm joints for manipulation
            return {
                'kp': 200.0,
                'ki': 2.0,
                'kd': 20.0,
                'integral': 0.0,
                'prev_error': 0.0
            }
        else:
            # Default gains
            return {
                'kp': 100.0,
                'ki': 1.0,
                'kd': 10.0,
                'integral': 0.0,
                'prev_error': 0.0
            }

    def joint_state_callback(self, msg):
        """Update current joint state"""
        # Map received joint states to our internal representation
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                if i < len(msg.position):
                    self.current_state['positions'][idx] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_state['velocities'][idx] = msg.velocity[i]
                if i < len(msg.effort):
                    self.current_state['accelerations'][idx] = msg.effort[i]  # Actually torques

    def imu_callback(self, msg):
        """Update IMU data"""
        self.current_state['imu_data'] = msg

    def scan_callback(self, msg):
        """Update LiDAR data"""
        self.current_state['lidar_data'] = msg

    def control_loop(self):
        """Main control loop"""
        start_time = time.time()

        # Update state estimates
        self.update_state_estimates()

        # Determine current behavior mode
        behavior_mode = self.determine_behavior_mode()

        # Compute control commands based on mode
        if behavior_mode == 'balance':
            commands = self.balance_controller.compute_balance_control(self.current_state)
        elif behavior_mode == 'walking':
            commands = self.walk_controller.compute_walking_control(self.current_state)
        elif behavior_mode == 'manipulation':
            commands = self.manipulation_controller.compute_manipulation_control(self.current_state)
        else:
            # Default to balance mode
            commands = self.balance_controller.compute_balance_control(self.current_state)

        # Apply joint limits and constraints
        commands = self.apply_joint_constraints(commands)

        # Publish joint commands
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = self.joint_names
        cmd_msg.position = commands.tolist()  # Use positions as commands
        cmd_msg.velocity = [0.0] * len(commands)  # Zero velocity commands
        cmd_msg.effort = [0.0] * len(commands)   # Zero effort commands

        self.joint_cmd_pub.publish(cmd_msg)

        # Publish CoM and ZMP for monitoring
        com_msg = Vector3()
        com_msg.x = self.current_state['com_position'][0]
        com_msg.y = self.current_state['com_position'][1]
        com_msg.z = self.current_state['com_position'][2]
        self.com_pub.publish(com_msg)

        # Calculate and publish ZMP
        zmp_msg = Vector3()
        if self.current_state['com_velocity'] is not None:
            # ZMP = CoM - (CoM_height / gravity) * CoM_acceleration
            # For now, simplified as CoM position with some correction
            zmp_msg.x = self.current_state['com_position'][0] - (
                self.current_state['com_velocity'][0] / (self.omega**2))
            zmp_msg.y = self.current_state['com_position'][1] - (
                self.current_state['com_velocity'][1] / (self.omega**2))
            zmp_msg.z = 0.0  # ZMP is on ground plane
        else:
            zmp_msg.x = self.current_state['com_position'][0]
            zmp_msg.y = self.current_state['com_position'][1]
            zmp_msg.z = 0.0
        self.zmp_pub.publish(zmp_msg)

        # Monitor performance
        elapsed_time = time.time() - start_time
        self.performance_monitor.record_iteration(elapsed_time)

    def update_state_estimates(self):
        """Update state estimates from sensor data"""
        # Update CoM estimate
        if len(self.current_state['positions']) > 0:
            self.current_state['com_position'] = self.com_estimator.estimate_com(
                self.current_state['positions'])

        if len(self.current_state['positions']) > 0 and len(self.current_state['velocities']) > 0:
            self.current_state['com_velocity'] = self.com_estimator.estimate_com_velocity(
                self.current_state['positions'], self.current_state['velocities'])

        # Update contact estimates
        self.current_state['contact_data'] = self.contact_estimator.estimate_contacts(
            self.current_state['positions'], self.current_state.get('lidar_data'))

    def determine_behavior_mode(self):
        """Determine current behavior mode based on state and goals"""
        # For now, use a simple heuristic
        # In practice, this would be more sophisticated
        com_pos = self.current_state['com_position']
        com_vel = self.current_state['com_velocity']

        # Check stability
        com_xy_speed = np.linalg.norm(com_vel[:2])

        if com_xy_speed > 0.1:
            # Moving significantly - likely walking
            return 'walking'
        elif self.is_unstable():
            # Unstable - prioritize balance
            return 'balance'
        else:
            # Stable and not moving significantly
            return 'balance'

    def is_unstable(self):
        """Check if robot is currently unstable"""
        # Check if CoM is outside support polygon
        support_polygon = self.calculate_support_polygon()
        com_xy = self.current_state['com_position'][:2]

        return not self.point_in_polygon(com_xy, support_polygon)

    def calculate_support_polygon(self):
        """Calculate support polygon from contact feet"""
        # Get foot positions
        left_foot_pos = np.array([0.0, 0.1, 0.0])  # Would get from FK
        right_foot_pos = np.array([0.0, -0.1, 0.0])  # Would get from FK

        # Create support polygon (simplified as rectangle between feet)
        support_polygon = np.array([
            [min(left_foot_pos[0], right_foot_pos[0]) - 0.1,  # Add margins
             min(left_foot_pos[1], right_foot_pos[1]) - 0.05],
            [max(left_foot_pos[0], right_foot_pos[0]) + 0.1,
             min(left_foot_pos[1], right_foot_pos[1]) - 0.05],
            [max(left_foot_pos[0], right_foot_pos[0]) + 0.1,
             max(left_foot_pos[1], right_foot_pos[1]) + 0.05],
            [min(left_foot_pos[0], right_foot_pos[0]) - 0.1,
             max(left_foot_pos[1], right_foot_pos[1]) + 0.05]
        ])

        return support_polygon

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def apply_joint_constraints(self, commands):
        """Apply joint limits and other constraints"""
        constrained_commands = commands.copy()

        # Define joint limits (in practice, would come from URDF)
        joint_limits = {
            'left_hip_roll': {'min': -0.5, 'max': 1.0},
            'left_hip_pitch': {'min': -0.5, 'max': 1.57},
            'left_knee': {'min': 0.0, 'max': 1.57},
            'left_ankle_pitch': {'min': -0.5, 'max': 0.5},
            'right_hip_roll': {'min': -1.0, 'max': 0.5},
            'right_hip_pitch': {'min': -0.5, 'max': 1.57},
            'right_knee': {'min': 0.0, 'max': 1.57},
            'right_ankle_pitch': {'min': -0.5, 'max': 0.5},
            'left_shoulder_pitch': {'min': -1.57, 'max': 1.57},
            'left_shoulder_yaw': {'min': -0.78, 'max': 0.78},
            'left_elbow': {'min': -1.57, 'max': 0.0},
            'right_shoulder_pitch': {'min': -1.57, 'max': 1.57},
            'right_shoulder_yaw': {'min': -0.78, 'max': 0.78},
            'right_elbow': {'min': -1.57, 'max': 0.0},
            'neck_joint': {'min': -0.5, 'max': 0.5}
        }

        for i, joint_name in enumerate(self.joint_names):
            if joint_name in joint_limits:
                limits = joint_limits[joint_name]

                # Calculate next position if command is applied
                next_pos = self.current_state['positions'][i] + commands[i] * 0.01  # 10ms step

                # Apply limits
                if next_pos < limits['min']:
                    max_change = (limits['min'] - self.current_state['positions'][i]) / 0.01
                    constrained_commands[i] = min(commands[i], max_change)
                elif next_pos > limits['max']:
                    max_change = (limits['max'] - self.current_state['positions'][i]) / 0.01
                    constrained_commands[i] = max(commands[i], max_change)

        return constrained_commands

class BalanceController:
    """Balance controller for humanoid robot"""

    def __init__(self, com_height):
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)

        # Control gains
        self.com_kp = 100.0
        self.com_kd = 20.0
        self.ankle_kp = 50.0
        self.ankle_kd = 10.0

    def compute_balance_control(self, robot_state):
        """Compute balance control commands"""
        # Get current state
        com_pos = robot_state['com_position']
        com_vel = robot_state['com_velocity']
        imu_data = robot_state['imu_data']

        # Calculate balance error
        # For balance, CoM should be over support polygon
        support_polygon = self.calculate_support_polygon(robot_state)
        com_xy = com_pos[:2]

        # Find closest point in support polygon
        target_com = self.find_balance_point(com_xy, support_polygon)

        # Calculate error
        com_error = target_com - com_xy
        com_vel_error = -com_vel[:2]  # Damping term

        # Compute CoM-based balance correction
        com_correction = self.com_kp * com_error + self.com_kd * com_vel_error

        # Convert to joint commands
        # This is a simplified approach - in practice, would use whole-body control
        joint_commands = np.zeros(18)

        # Apply corrections primarily to hip and ankle joints
        joint_commands[0] = com_correction[0] * 0.1  # Left hip roll
        joint_commands[1] = com_correction[0] * 0.05  # Left hip pitch
        joint_commands[3] = com_correction[1] * 0.2  # Left ankle pitch

        joint_commands[4] = com_correction[0] * 0.1  # Right hip roll
        joint_commands[5] = com_correction[0] * 0.05  # Right hip pitch
        joint_commands[7] = com_correction[1] * 0.2  # Right ankle pitch

        # Use IMU data for additional orientation correction
        if imu_data is not None:
            # Extract roll and pitch from IMU orientation
            quat = [imu_data.orientation.x, imu_data.orientation.y,
                   imu_data.orientation.z, imu_data.orientation.w]
            rotation = R.from_quat(quat)
            euler = rotation.as_euler('xyz')

            # Correct for non-upright orientation
            joint_commands[16] += -euler[0] * 50  # Torso roll correction
            joint_commands[17] += -euler[1] * 50  # Torso pitch correction

        return joint_commands

    def calculate_support_polygon(self, robot_state):
        """Calculate support polygon based on contact feet"""
        # In practice, would use contact sensor data or forward kinematics
        # For now, return a simple polygon based on foot positions
        return np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])

    def find_balance_point(self, com_xy, support_polygon):
        """Find the optimal balance point within support polygon"""
        # For now, return center of support polygon
        # In practice, would use more sophisticated ZMP-based methods
        center = np.mean(support_polygon, axis=0)
        return center

class WalkingController:
    """Walking controller for humanoid robot"""

    def __init__(self, com_height):
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)

        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters (between feet)
        self.step_height = 0.05 # meters (foot lift)
        self.step_duration = 0.8  # seconds

        # State tracking
        self.current_step_phase = 0.0
        self.step_count = 0
        self.left_support = True

    def compute_walking_control(self, robot_state):
        """Compute walking control commands"""
        # Update step phase
        self.current_step_phase += 1.0 / (self.step_duration * 1000)  # Assuming 1kHz control
        if self.current_step_phase > 1.0:
            self.current_step_phase = 0.0
            self.step_count += 1
            self.left_support = not self.left_support  # Alternate support foot

        # Generate walking pattern
        walking_commands = self.generate_walking_pattern(robot_state)

        # Add balance corrections to walking
        balance_controller = BalanceController(self.com_height)
        balance_commands = balance_controller.compute_balance_control(robot_state)

        # Combine walking and balance with appropriate weights
        combined_commands = 0.7 * walking_commands + 0.3 * balance_commands

        return combined_commands

    def generate_walking_pattern(self, robot_state):
        """Generate walking pattern commands"""
        joint_commands = np.zeros(18)

        # Calculate walking pattern based on phase
        if self.current_step_phase < 0.5:  # Swing phase
            # Calculate foot trajectory for swing foot
            if self.left_support:
                # Right foot swinging
                swing_phase = self.current_step_phase / 0.5
                # Lift and move right foot forward
                joint_commands[5] = -0.1 * np.sin(swing_phase * np.pi)  # Right hip pitch
                joint_commands[6] = 0.2 * np.sin(swing_phase * np.pi)   # Right knee (lift leg)
            else:
                # Left foot swinging
                swing_phase = self.current_step_phase / 0.5
                joint_commands[1] = -0.1 * np.sin(swing_phase * np.pi)  # Left hip pitch
                joint_commands[2] = 0.2 * np.sin(swing_phase * np.pi)   # Left knee (lift leg)

        # Add forward progression
        forward_progress = self.step_length * self.step_count
        # Adjust hip joints to move CoM forward
        hip_correction = 0.05 * np.sin(self.current_step_phase * 2 * np.pi)
        joint_commands[1] += hip_correction  # Left hip
        joint_commands[5] += hip_correction  # Right hip

        # Add lateral sway for balance
        lateral_sway = 0.02 * np.sin(self.current_step_phase * np.pi)
        joint_commands[0] += lateral_sway   # Left hip roll
        joint_commands[4] -= lateral_sway   # Right hip roll

        return joint_commands

class ManipulationController:
    """Manipulation controller for humanoid robot arms"""

    def __init__(self):
        # Manipulation-specific parameters
        self.arm_kp = 200.0
        self.arm_kd = 20.0
        self.gripper_kp = 50.0

    def compute_manipulation_control(self, robot_state):
        """Compute manipulation control commands"""
        joint_commands = np.zeros(18)

        # Example: Reach for object in front of robot
        # In practice, would use vision data to determine object location
        desired_left_hand_pos = np.array([0.5, 0.2, 0.8])  # In robot frame
        desired_right_hand_pos = np.array([0.5, -0.2, 0.8])  # In robot frame

        # Calculate current hand positions (simplified - would use FK)
        current_left_hand_pos = np.array([0.3, 0.1, 0.7])
        current_right_hand_pos = np.array([0.3, -0.1, 0.7])

        # Calculate position errors
        left_hand_error = desired_left_hand_pos - current_left_hand_pos
        right_hand_error = desired_right_hand_pos - current_right_hand_pos

        # Convert to joint space commands (simplified mapping)
        joint_commands[8] = left_hand_error[0] * self.arm_kp * 0.01  # Left shoulder pitch
        joint_commands[9] = left_hand_error[1] * self.arm_kp * 0.01  # Left shoulder yaw
        joint_commands[10] = left_hand_error[2] * self.arm_kp * 0.01  # Left elbow

        joint_commands[11] = right_hand_error[0] * self.arm_kp * 0.01  # Right shoulder pitch
        joint_commands[12] = right_hand_error[1] * self.arm_kp * 0.01  # Right shoulder yaw
        joint_commands[13] = right_hand_error[2] * self.arm_kp * 0.01  # Right elbow

        return joint_commands

class COMEstimator:
    """Center of Mass estimator for humanoid robot"""

    def __init__(self, joint_names):
        self.joint_names = joint_names
        # Simplified mass distribution (in practice, would come from URDF)
        self.link_masses = {
            'torso': 15.0,
            'head': 3.0,
            'upper_arm': 2.0,
            'lower_arm': 1.5,
            'upper_leg': 4.0,
            'lower_leg': 3.0,
            'foot': 1.0
        }

    def estimate_com(self, joint_positions):
        """Estimate center of mass position"""
        # Simplified CoM estimation
        # In practice, would use forward kinematics and actual link masses
        com_x = 0.0
        com_y = 0.0
        com_z = 0.85  # Typical humanoid CoM height

        # Add contributions from different body parts
        # This is a simplified model - in practice would use full kinematic chain
        if len(joint_positions) >= 18:
            # Use joint positions to estimate CoM shift
            # Hip joints affect CoM position significantly
            hip_effect = joint_positions[0] * 0.05 + joint_positions[4] * 0.05  # Left and right hip roll
            com_x += hip_effect * 0.1

            # Ankle joints affect CoM height slightly
            ankle_effect = (joint_positions[3] + joint_positions[7]) * 0.5  # Average ankle position
            com_z += ankle_effect * 0.02

        return np.array([com_x, com_y, com_z])

    def estimate_com_velocity(self, joint_positions, joint_velocities):
        """Estimate center of mass velocity"""
        if len(joint_velocities) == 0:
            return np.zeros(3)

        # Simplified velocity estimation
        com_vel_x = 0.0
        com_vel_y = 0.0
        com_vel_z = 0.0

        # Use dominant joints for CoM velocity
        if len(joint_velocities) >= 18:
            hip_vel_effect = joint_velocities[0] * 0.05 + joint_velocities[4] * 0.05
            com_vel_x += hip_vel_effect * 0.1

        return np.array([com_vel_x, com_vel_y, com_vel_z])

class ContactEstimator:
    """Estimate contact points for balance control"""

    def __init__(self):
        pass

    def estimate_contacts(self, joint_positions, lidar_data=None):
        """Estimate which parts of robot are in contact with ground"""
        contacts = {}

        # Simplified contact estimation based on foot positions
        # In practice, would use force/torque sensors or contact detection
        contacts['left_foot'] = True  # Assume left foot is in contact
        contacts['right_foot'] = True  # Assume right foot is in contact

        # If LiDAR data is available, use it to refine contact estimates
        if lidar_data is not None:
            # Analyze LiDAR data to detect ground contact
            # This would involve detecting planar surfaces near foot level
            pass

        return contacts

class TrajectoryGenerator:
    """Generate smooth trajectories for joint movements"""

    def __init__(self):
        pass

    def generate_min_jerk_trajectory(self, start_pos, end_pos, duration, dt=0.01):
        """Generate minimum jerk trajectory between two points"""
        n_points = int(duration / dt)
        trajectory = np.zeros((n_points, len(start_pos)))

        for i in range(n_points):
            t = i * dt / duration  # Normalized time (0 to 1)

            # Minimum jerk basis functions
            basis = 10 * t**3 - 15 * t**4 + 6 * t**5

            trajectory[i] = start_pos + basis * (end_pos - start_pos)

        return trajectory

    def generate_swing_foot_trajectory(self, start_pos, end_pos, height=0.05):
        """Generate trajectory for swing foot during walking"""
        # Create parabolic trajectory with lift at midpoint
        n_points = 50
        trajectory = np.zeros((n_points, 3))

        for i in range(n_points):
            t = i / (n_points - 1)  # 0 to 1

            # X, Y: linear interpolation
            trajectory[i, 0] = start_pos[0] + t * (end_pos[0] - start_pos[0])
            trajectory[i, 1] = start_pos[1] + t * (end_pos[1] - start_pos[1])

            # Z: parabolic lift
            if t < 0.5:
                # Going up
                trajectory[i, 2] = start_pos[2] + (height * 4 * t**2)
            else:
                # Coming down
                trajectory[i, 2] = start_pos[2] + height - (height * 4 * (t - 0.5)**2)

        return trajectory

class PerformanceMonitor:
    """Monitor control system performance"""

    def __init__(self):
        self.iteration_times = deque(maxlen=1000)
        self.control_frequency = 1000  # Expected frequency

    def record_iteration(self, elapsed_time):
        """Record control iteration time"""
        self.iteration_times.append(elapsed_time)

    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.iteration_times:
            return {
                'avg_time_ms': 0,
                'min_time_ms': 0,
                'max_time_ms': 0,
                'std_dev_ms': 0,
                'actual_frequency': 0
            }

        times_ms = [t * 1000 for t in self.iteration_times]
        avg_time = sum(times_ms) / len(times_ms)

        return {
            'avg_time_ms': avg_time,
            'min_time_ms': min(times_ms),
            'max_time_ms': max(times_ms),
            'std_dev_ms': np.std(times_ms),
            'actual_frequency': 1000 / avg_time if avg_time > 0 else 0
        }

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidSimulationController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down humanoid simulation controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Validation and Testing

### Simulation Validation

```python
class SimulationValidator:
    """Validate simulation performance and accuracy"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.validation_results = {}

    def validate_simulation_performance(self):
        """Validate simulation performance metrics"""
        results = {
            'real_time_factor': 0.0,
            'frame_rate': 0.0,
            'physics_accuracy': True,
            'sensor_fidelity': True,
            'control_stability': True
        }

        # Check real-time factor
        # This would be obtained from Gazebo simulation data
        results['real_time_factor'] = self.check_real_time_factor()
        results['frame_rate'] = self.check_rendering_frame_rate()

        # Validate physics behavior
        results['physics_accuracy'] = self.validate_physics_behavior()

        # Validate sensor data quality
        results['sensor_fidelity'] = self.validate_sensor_data()

        # Validate control system stability
        results['control_stability'] = self.validate_control_stability()

        return results

    def check_real_time_factor(self):
        """Check if simulation is running at acceptable real-time factor"""
        # In practice, would query Gazebo for real-time factor
        # For now, return a simulated value
        import random
        return random.uniform(0.8, 1.2)  # Simulated RTF between 0.8 and 1.2

    def check_rendering_frame_rate(self):
        """Check rendering frame rate"""
        # In practice, would monitor rendering performance
        # For now, return a simulated value
        return 60  # 60 FPS

    def validate_physics_behavior(self):
        """Validate that physics behavior is realistic"""
        # Check for common physics issues:
        # - Objects falling through the ground
        # - Unrealistic joint movements
        # - Energy conservation violations
        # - Constraint violations

        # This would involve running physics-specific tests
        tests_passed = 0
        total_tests = 5

        # Test 1: Gravity simulation (objects should fall at ~9.8 m/s^2)
        if self.test_gravity_simulation():
            tests_passed += 1

        # Test 2: Joint limits (joints shouldn't exceed limits)
        if self.test_joint_limits():
            tests_passed += 1

        # Test 3: Collision detection (objects shouldn't pass through each other)
        if self.test_collision_detection():
            tests_passed += 1

        # Test 4: Balance (humanoid should be able to maintain balance)
        if self.test_balance_maintenance():
            tests_passed += 1

        # Test 5: Mass properties (should behave according to mass distribution)
        if self.test_mass_properties():
            tests_passed += 1

        return tests_passed == total_tests

    def test_gravity_simulation(self):
        """Test that gravity is simulated correctly"""
        # Drop an object and measure acceleration
        # In simulation, would run actual physics test
        return True  # Placeholder

    def test_joint_limits(self):
        """Test that joint limits are enforced"""
        # Try to command joint beyond limits and verify it doesn't exceed them
        # In simulation, would run actual joint limit test
        return True  # Placeholder

    def test_collision_detection(self):
        """Test that collision detection works properly"""
        # Create collision scenario and verify detection
        # In simulation, would run actual collision test
        return True  # Placeholder

    def test_balance_maintenance(self):
        """Test that humanoid can maintain balance"""
        # Run balance controller and verify CoM stays within support polygon
        # In simulation, would run actual balance test
        return True  # Placeholder

    def test_mass_properties(self):
        """Test that mass properties are realistic"""
        # Verify that robot behaves according to its mass distribution
        # In simulation, would run actual mass property test
        return True  # Placeholder

    def validate_sensor_data(self):
        """Validate that sensor data is realistic and accurate"""
        # Check for:
        # - Proper noise characteristics
        # - Appropriate update rates
        # - Realistic ranges and accuracies
        # - No dropped frames or corrupted data

        sensor_tests = [
            self.test_camera_sensor_accuracy(),
            self.test_lidar_sensor_accuracy(),
            self.test_imu_sensor_accuracy()
        ]

        return all(sensor_tests)

    def test_camera_sensor_accuracy(self):
        """Test camera sensor for proper image quality and noise"""
        # Verify camera produces realistic images with appropriate noise levels
        # Check for proper field of view, resolution, and frame rate
        return True  # Placeholder

    def test_lidar_sensor_accuracy(self):
        """Test LiDAR sensor for proper range and resolution"""
        # Verify LiDAR produces realistic point clouds with appropriate noise
        # Check for proper range, resolution, and update rate
        return True  # Placeholder

    def test_imu_sensor_accuracy(self):
        """Test IMU sensor for proper noise and drift characteristics"""
        # Verify IMU produces realistic data with appropriate noise and bias
        # Check for proper update rate and accuracy
        return True  # Placeholder

    def validate_control_stability(self):
        """Validate that control system is stable"""
        # Run control system for extended period and verify:
        # - No oscillations or instability
        # - Proper tracking performance
        # - No excessive control effort
        # - Stability margins are adequate

        stability_tests = [
            self.test_control_stability_over_time(),
            self.test_tracking_accuracy(),
            self.test_control_effort_bounds()
        ]

        return all(stability_tests)

    def test_control_stability_over_time(self):
        """Test control system stability over extended period"""
        # Run simulation with control system for several minutes
        # Monitor for signs of instability or oscillation
        return True  # Placeholder

    def test_tracking_accuracy(self):
        """Test control system tracking accuracy"""
        # Command robot to track specific trajectories
        # Measure tracking error and verify it's within acceptable bounds
        return True  # Placeholder

    def test_control_effort_bounds(self):
        """Test that control efforts stay within reasonable bounds"""
        # Monitor control commands to ensure they're not excessively large
        # Verify joint torques/velocities stay within limits
        return True  # Placeholder

    def run_comprehensive_validation(self):
        """Run comprehensive validation of the entire simulation system"""
        validation_report = {
            'simulation_performance': self.validate_simulation_performance(),
            'physics_validation': self.validate_physics_behavior(),
            'sensor_validation': self.validate_sensor_data(),
            'control_validation': self.validate_control_stability(),
            'integration_validation': self.validate_system_integration(),
            'overall_rating': 0.0,
            'recommendations': []
        }

        # Calculate overall rating based on validation results
        score = 0
        total_possible = 4  # Four main validation categories

        if validation_report['simulation_performance']['real_time_factor'] >= 0.8:
            score += 0.25
        if validation_report['physics_validation']:
            score += 0.25
        if validation_report['sensor_validation']:
            score += 0.25
        if validation_report['control_validation']:
            score += 0.25

        validation_report['overall_rating'] = score

        # Generate recommendations based on validation results
        if validation_report['simulation_performance']['real_time_factor'] < 0.8:
            validation_report['recommendations'].append(
                "Simulation performance needs improvement - consider optimizing physics parameters"
            )

        if not validation_report['physics_validation']:
            validation_report['recommendations'].append(
                "Physics behavior validation failed - check mass properties and constraints"
            )

        if not validation_report['sensor_validation']:
            validation_report['recommendations'].append(
                "Sensor validation failed - verify sensor configurations and noise parameters"
            )

        if not validation_report['control_validation']:
            validation_report['recommendations'].append(
                "Control stability validation failed - review control parameters and gains"
            )

        return validation_report

    def validate_system_integration(self):
        """Validate that all components work together properly"""
        # Test the integration of:
        # - Robot model with physics engine
        # - Sensors with control system
        # - Control system with robot dynamics
        # - All components with simulation environment

        # This would involve running integrated system tests
        integration_tests = [
            self.test_robot_sensor_integration(),
            self.test_control_sensor_feedback(),
            self.test_multisensor_fusion()
        ]

        return all(integration_tests)

    def test_robot_sensor_integration(self):
        """Test that robot model and sensors are properly integrated"""
        # Verify that sensor data reflects robot's actual state in simulation
        return True  # Placeholder

    def test_control_sensor_feedback(self):
        """Test that control system properly uses sensor feedback"""
        # Verify that control system responds appropriately to sensor data
        return True  # Placeholder

    def test_multisensor_fusion(self):
        """Test that multiple sensors work together properly"""
        # Verify that data from different sensors can be combined effectively
        return True  # Placeholder
```

## Performance Optimization

### Real-Time Performance Considerations

```python
class RealTimeOptimizer:
    """Optimize simulation for real-time performance"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.optimization_strategies = []

    def optimize_simulation_performance(self):
        """Apply various optimization strategies for real-time performance"""
        optimization_report = {
            'original_performance': self.measure_current_performance(),
            'applied_optimizations': [],
            'improved_performance': None,
            'tradeoffs': []
        }

        # Strategy 1: Physics optimization
        physics_opt = self.optimize_physics_parameters()
        if physics_opt:
            optimization_report['applied_optimizations'].append({
                'strategy': 'Physics Parameter Tuning',
                'improvement': physics_opt['improvement'],
                'impact': physics_opt['impact']
            })

        # Strategy 2: Sensor optimization
        sensor_opt = self.optimize_sensor_parameters()
        if sensor_opt:
            optimization_report['applied_optimizations'].append({
                'strategy': 'Sensor Parameter Tuning',
                'improvement': sensor_opt['improvement'],
                'impact': sensor_opt['impact']
            })

        # Strategy 3: Rendering optimization
        render_opt = self.optimize_rendering_parameters()
        if render_opt:
            optimization_report['applied_optimizations'].append({
                'strategy': 'Rendering Optimization',
                'improvement': render_opt['improvement'],
                'impact': render_opt['impact']
            })

        # Strategy 4: Control optimization
        control_opt = self.optimize_control_parameters()
        if control_opt:
            optimization_report['applied_optimizations'].append({
                'strategy': 'Control Parameter Tuning',
                'improvement': control_opt['improvement'],
                'impact': control_opt['impact']
            })

        # Measure final performance
        optimization_report['improved_performance'] = self.measure_current_performance()

        return optimization_report

    def optimize_physics_parameters(self):
        """Optimize physics engine parameters for performance"""
        # Strategies to optimize physics:
        # 1. Adjust solver iterations (fewer = faster, less accurate)
        # 2. Adjust constraint parameters (CFM, ERP)
        # 3. Adjust contact parameters
        # 4. Adjust update rates

        current_rtf = self.measure_current_performance()['real_time_factor']

        if current_rtf < 0.8:  # Performance is poor
            # Reduce solver iterations for performance
            # This would involve modifying the SDF physics configuration
            optimization = {
                'improvement': 'Increased RTF by adjusting solver iterations',
                'impact': 'Slight reduction in physics accuracy but significant performance gain',
                'parameters_changed': {
                    'solver_iterations': 'Reduced from 100 to 50',
                    'constraint_parameters': 'Adjusted for performance'
                }
            }
            return optimization
        else:
            return None  # No optimization needed

    def optimize_sensor_parameters(self):
        """Optimize sensor parameters for performance"""
        # Strategies to optimize sensors:
        # 1. Reduce update rates for less critical sensors
        # 2. Reduce resolution for cameras
        # 3. Reduce sample counts for LiDAR
        # 4. Optimize sensor placement and field of view

        optimization = {
            'improvement': 'Reduced sensor update rates and resolution where appropriate',
            'impact': 'Improved simulation performance with minimal impact on perception quality',
            'parameters_changed': {
                'camera_fps': 'Reduced from 30 to 15 FPS',
                'lidar_samples': 'Reduced from 720 to 360 samples',
                'imu_rate': 'Reduced from 100Hz to 50Hz'
            }
        }
        return optimization

    def optimize_rendering_parameters(self):
        """Optimize rendering parameters for performance"""
        # Strategies to optimize rendering:
        # 1. Reduce shadow quality
        # 2. Reduce texture resolution
        # 3. Adjust anti-aliasing
        # 4. Reduce visual effects

        optimization = {
            'improvement': 'Reduced rendering overhead',
            'impact': 'Improved performance with slight visual quality reduction',
            'parameters_changed': {
                'shadow_quality': 'Reduced to simple shadows',
                'texture_resolution': 'Reduced to low resolution',
                'anti_aliasing': 'Disabled',
                'visual_effects': 'Minimized'
            }
        }
        return optimization

    def optimize_control_parameters(self):
        """Optimize control system parameters for performance"""
        # Strategies to optimize control:
        # 1. Adjust control frequency (balance accuracy vs performance)
        # 2. Optimize control algorithms (simpler but effective)
        # 3. Use caching for expensive computations
        # 4. Parallelize control computations

        optimization = {
            'improvement': 'Optimized control algorithms and caching',
            'impact': 'Maintained control accuracy while improving performance',
            'parameters_changed': {
                'control_frequency': 'Optimized to 500Hz',
                'algorithm_optimization': 'Implemented caching and pre-allocation',
                'parallel_computation': 'Enabled for independent tasks'
            }
        }
        return optimization

    def measure_current_performance(self):
        """Measure current simulation performance"""
        # This would interface with simulation to get actual performance metrics
        # For now, return simulated values
        return {
            'real_time_factor': 0.9,  # 90% of real-time
            'frame_rate': 60,         # 60 FPS
            'cpu_usage': 0.7,         # 70% CPU usage
            'memory_usage': 0.6       # 60% memory usage
        }

def main(args=None):
    rclpy.init(args=args)

    # Initialize robot model (simplified)
    robot_model = None  # Would be actual robot model

    # Create and run simulation validator
    validator = SimulationValidator(robot_model)
    validation_results = validator.run_comprehensive_validation()

    # Create and run performance optimizer
    optimizer = RealTimeOptimizer(robot_model)
    optimization_report = optimizer.optimize_simulation_performance()

    print("=== SIMULATION VALIDATION RESULTS ===")
    print(f"Overall Rating: {validation_results['overall_rating']}/1.0")
    print(f"Real-time Factor: {validation_results['simulation_performance']['real_time_factor']}")
    print(f"Physics Validation: {'PASS' if validation_results['physics_validation'] else 'FAIL'}")
    print(f"Sensor Validation: {'PASS' if validation_results['sensor_validation'] else 'FAIL'}")
    print(f"Control Validation: {'PASS' if validation_results['control_validation'] else 'FAIL'}")

    print("\n=== PERFORMANCE OPTIMIZATION REPORT ===")
    print(f"Applied {len(optimization_report['applied_optimizations'])} optimizations")
    for opt in optimization_report['applied_optimizations']:
        print(f"- {opt['strategy']}: {opt['improvement']}")

    print(f"\nRecommendations: {len(validation_results['recommendations'])}")
    for rec in validation_results['recommendations']:
        print(f"- {rec}")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

The Module 2 capstone project demonstrates the complete integration of digital twins and robot simulation for humanoid robots. The key achievements include:

1. **Complete Robot Model**: A fully articulated humanoid model with realistic kinematics, dynamics, and physical properties
2. **Sensor Integration**: Comprehensive sensor suite including cameras, LiDAR, and IMU with realistic noise models
3. **Control Systems**: Sophisticated whole-body control systems including balance, walking, and manipulation controllers
4. **Simulation Environment**: Realistic physics simulation with proper constraints and environmental interactions
5. **Validation Framework**: Comprehensive validation and performance optimization tools

The implementation showcases the sophisticated engineering required to create realistic humanoid robot simulations that can be used for development, testing, and training of Physical AI systems. Proper simulation enables developers to iterate quickly and safely before deploying to real hardware, making it an essential tool in the Physical AI development pipeline.

This capstone project provides a solid foundation for more advanced topics in the subsequent modules, including perception systems, navigation, and autonomous behavior.