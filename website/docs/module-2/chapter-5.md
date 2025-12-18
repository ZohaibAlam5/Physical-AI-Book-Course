---
title: Chapter 5 - Sensor Simulation - Cameras, LiDAR, IMU
description: Comprehensive guide to simulating various sensors for humanoid robots
sidebar_position: 5
---

# Chapter 5: Sensor Simulation - Cameras, LiDAR, IMU

## Learning Objectives

After completing this chapter, you should be able to:
- Implement realistic camera sensor simulation for humanoid robots
- Configure LiDAR sensors with appropriate parameters for simulation
- Simulate IMU sensors with realistic noise models
- Integrate multiple sensor types in a unified simulation environment
- Validate sensor simulation against real-world characteristics

## Introduction

Sensor simulation is a critical component of Physical AI and humanoid robotics development. Realistic sensor simulation enables developers to test perception algorithms, validate control systems, and train AI agents in a safe, controlled environment before deploying to real hardware. This chapter covers the three most important sensor types for humanoid robots: cameras for vision-based perception, LiDAR for ranging and mapping, and IMUs for inertial measurement and balance control.

## Camera Simulation

### Types of Camera Sensors

Robots use various types of cameras depending on their application:

#### 1. RGB Cameras
- Provide color images for object recognition and scene understanding
- Essential for vision-based navigation and manipulation
- Used for human-robot interaction and facial recognition

#### 2. Depth Cameras
- Provide 3D information about the environment
- Critical for navigation, obstacle detection, and manipulation
- Enable 3D reconstruction and spatial awareness

#### 3. Stereo Cameras
- Provide depth information through stereo vision
- More accurate than structured light depth sensors
- Useful for precise distance measurements

### Camera Parameters and Characteristics

#### Resolution and Field of View

Camera sensors in simulation must match real-world characteristics:

```xml
<gazebo reference="camera_mount">
  <sensor name="rgb_camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <!-- Horizontal field of view (in radians) -->
      <horizontal_fov>1.3962634</horizontal_fov>  <!-- 80 degrees -->

      <!-- Image resolution -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>

      <!-- Clipping distances -->
      <clip>
        <near>0.1</near>    <!-- Near clipping plane -->
        <far>100.0</far>    <!-- Far clipping plane -->
      </clip>

      <!-- Noise model -->
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>

    <!-- ROS plugin -->
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
      <point_cloud_cutoff>0.1</point_cloud_cutoff>
      <point_cloud_cutoff_max>100.0</point_cloud_cutoff_max>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <image_topic_name>camera/image_raw</image_topic_name>
      <camera_info_topic_name>camera/camera_info</camera_info_topic_name>
      <point_cloud_topic_name>camera/depth/points</point_cloud_topic_name>
    </plugin>
  </sensor>
</gazebo>
```

#### Camera Intrinsics

Real cameras have specific intrinsic parameters that must be simulated:

```xml
<!-- In camera plugin -->
<plugin name="camera_controller" filename="libgazebo_ros_camera.so">
  <frame_name>camera_optical_frame</frame_name>

  <!-- Camera intrinsics -->
  <cx>320.0</cx>        <!-- Principal point x -->
  <cy>240.0</cy>        <!-- Principal point y -->
  <fx>525.0</fx>        <!-- Focal length x -->
  <fy>525.0</fy>        <!-- Focal length y -->

  <!-- Distortion coefficients -->
  <distortion_k1>0.0</distortion_k1>
  <distortion_k2>0.0</distortion_k2>
  <distortion_p1>0.0</distortion_p1>
  <distortion_p2>0.0</distortion_p2>
  <distortion_k3>0.0</distortion_k3>

  <!-- Other parameters -->
  <min_depth>0.1</min_depth>
  <max_depth>100.0</max_depth>
  <update_rate>30</update_rate>
  <topic>camera/image_raw</topic>
</plugin>
```

### Depth Camera Simulation

Depth cameras provide crucial 3D information for humanoid robots:

```xml
<gazebo reference="depth_camera_mount">
  <sensor name="depth_camera" type="depth">
    <update_rate>30</update_rate>
    <camera name="depth_head_camera">
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>320</width>
        <height>240</height>
        <format>R_FLOAT32</format>  <!-- Raw depth values -->
      </image>
      <clip>
        <near>0.3</near>    <!-- Closer than 0.3m is unreliable -->
        <far>10.0</far>     <!-- Beyond 10m is noisy -->
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>  <!-- 1cm accuracy at 1m -->
      </noise>
    </camera>

    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.1</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <point_cloud_cutoff>0.3</point_cloud_cutoff>
      <point_cloud_cutoff_max>10.0</point_cloud_cutoff_max>
      <frame_name>depth_camera_optical_frame</frame_name>
      <point_cloud_topic_name>camera/depth/points</point_cloud_topic_name>
      <depth_image_topic_name>camera/depth/image_raw</depth_image_topic_name>
      <depth_image_camera_info_topic_name>camera/depth/camera_info</depth_image_camera_info_topic_name>
      <image_topic_name>camera/rgb/image_raw</image_topic_name>
      <camera_info_topic_name>camera/rgb/camera_info</camera_info_topic_name>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Stereo Camera Simulation

Stereo cameras provide precise depth information:

```xml
<!-- Left camera -->
<gazebo reference="left_camera_mount">
  <sensor name="left_camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="left_camera">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="left_camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>left_camera_optical_frame</frame_name>
      <update_rate>30</update_rate>
      <topic>stereo/left/image_raw</topic>
      <camera_info_topic>stereo/left/camera_info</camera_info_topic>
    </plugin>
  </sensor>
</gazebo>

<!-- Right camera -->
<gazebo reference="right_camera_mount">
  <sensor name="right_camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="right_camera">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="right_camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>right_camera_optical_frame</frame_name>
      <update_rate>30</update_rate>
      <topic>stereo/right/image_raw</topic>
      <camera_info_topic>stereo/right/camera_info</camera_info_topic>
    </plugin>
  </sensor>
</gazebo>
```

### Camera Noise Modeling

Real cameras have various types of noise that must be simulated:

```xml
<!-- Different noise models for different camera types -->
<sensor name="high_quality_camera" type="camera">
  <camera name="hq_camera">
    <noise>
      <!-- Gaussian noise for CCD/CMOS sensors -->
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.005</stddev>  <!-- Low noise for high-quality camera -->
    </noise>
  </camera>
</sensor>

<sensor name="low_light_camera" type="camera">
  <camera name="ll_camera">
    <noise>
      <!-- Higher noise for low-light conditions -->
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.02</stddev>  <!-- Higher noise for low-light camera -->
    </noise>
  </camera>
</sensor>

<!-- Thermal camera noise -->
<sensor name="thermal_camera" type="thermal_camera">
  <camera name="thermal_cam">
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.5</stddev>  <!-- Temperature-based noise -->
    </noise>
  </camera>
</sensor>
```

## LiDAR Simulation

### Types of LiDAR Sensors

LiDAR sensors come in various configurations for different applications:

#### 1. 2D LiDAR
- Single horizontal scan line
- Used for 2D mapping and navigation
- Common on wheeled robots

#### 2. 3D LiDAR
- Multiple scan planes
- Provides full 3D point cloud
- Essential for humanoid navigation

#### 3. Solid-State LiDAR
- No moving parts
- Compact and reliable
- Emerging technology for robotics

### 2D LiDAR Configuration

```xml
<gazebo reference="lidar_2d_mount">
  <sensor name="lidar_2d" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>           <!-- Number of samples per revolution -->
          <resolution>1</resolution>       <!-- Resolution of samples -->
          <min_angle>-3.14159</min_angle>  <!-- -π radians = -180 degrees -->
          <max_angle>3.14159</max_angle>   <!-- π radians = 180 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>                    <!-- Minimum detectable range -->
        <max>30.0</max>                   <!-- Maximum detectable range -->
        <resolution>0.01</resolution>     <!-- Range resolution -->
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

### 3D LiDAR Configuration

For humanoid robots, 3D LiDAR provides comprehensive environmental awareness:

```xml
<gazebo reference="lidar_3d_mount">
  <sensor name="velodyne_vlp16" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>1800</samples>         <!-- Samples per revolution -->
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -π -->
          <max_angle>3.14159</max_angle>  <!-- π -->
        </horizontal>
        <vertical>
          <samples>16</samples>           <!-- 16 laser beams -->
          <resolution>1</resolution>
          <min_angle>-0.2618</min_angle>  <!-- -15 degrees -->
          <max_angle>0.2618</max_angle>   <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.2</min>                   <!-- Min range -->
        <max>100.0</max>                 <!-- Max range -->
        <resolution>0.001</resolution>    <!-- Range resolution -->
      </range>
    </ray>

    <plugin name="velodyne_driver" filename="libgazebo_ros_velodyne_gpu_lidar.so">
      <frame_name>velodyne</frame_name>
      <min_range>0.2</min_range>
      <max_range>100.0</max_range>
      <topic>velodyne_points</topic>
      <hmin_angle>-3.14159</hmin_angle>
      <hmax_angle>3.14159</hmax_angle>
      <vmin_angle>-0.2618</vmin_angle>
      <vmax_angle>0.2618</vmax_angle>
      <samples>1800</samples>
      <update_rate>10</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Noise and Accuracy Modeling

Real LiDAR sensors have various error sources that must be simulated:

```xml
<gazebo reference="lidar_with_noise">
  <sensor name="lidar_noisy" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>1081</samples>
          <resolution>1</resolution>
          <min_angle>-2.35619</min_angle>  <!-- -135 degrees -->
          <max_angle>2.35619</max_angle>   <!-- 135 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.08</min>
        <max>30.0</max>
        <resolution>0.001</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
      </noise>
    </ray>

    <plugin name="noisy_lidar" filename="libgazebo_ros_lidar.so">
      <topic>scan_noisy</topic>
      <frame_name>lidar_frame</frame_name>
      <min_range>0.08</min_range>
      <max_range>30.0</max_range>
      <update_rate>10</update_rate>
      <gaussian_noise>0.01</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### Multi-Beam LiDAR Configuration

Advanced LiDAR configurations with multiple beams:

```xml
<gazebo reference="multi_beam_lidar">
  <sensor name="ouster_lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>1024</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>64</samples>           <!-- 64 beams for Ouster OS1 -->
          <resolution>1</resolution>
          <min_angle>-0.5236</min_angle>  <!-- -30 degrees -->
          <max_angle>0.5236</max_angle>   <!-- 30 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>120.0</max>                  <!-- Extended range -->
        <resolution>0.002</resolution>    <!-- 2mm resolution -->
      </range>
    </ray>

    <plugin name="ouster_driver" filename="libgazebo_ros_ouster_lidar.so">
      <frame_name>os_lidar</frame_name>
      <min_range>0.1</min_range>
      <max_range>120.0</max_range>
      <topic>os_cloud</topic>
      <update_rate>10</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

## IMU Simulation

### IMU Fundamentals

IMUs (Inertial Measurement Units) are critical for humanoid robot balance and navigation:

#### IMU Components
- **Accelerometer**: Measures linear acceleration
- **Gyroscope**: Measures angular velocity
- **Magnetometer**: Measures magnetic field (compass)

#### IMU Applications in Humanoid Robots
- Balance control and stabilization
- Orientation estimation
- Motion tracking
- Navigation and localization

### IMU Sensor Configuration

```xml
<gazebo reference="imu_mount">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <!-- Gyroscope noise characteristics -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>          <!-- 0.2 mrad/s -->
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

      <!-- Accelerometer noise characteristics -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>        <!-- 17 mg -->
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
      <frame_name>imu_link</frame_name>
      <topic>imu/data</topic>
      <serviceName>imu/service</serviceName>
      <gaussianNoise>0.0</gaussianNoise>
      <updateRateHZ>100.0</updateRateHZ>
    </plugin>
  </sensor>
</gazebo>
```

### High-Precision IMU Configuration

For applications requiring high precision:

```xml
<gazebo reference="high_precision_imu">
  <sensor name="precision_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>500</update_rate>         <!-- Higher update rate -->
    <visualize>true</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1e-5</stddev>          <!-- Very low noise -->
            <bias_mean>0.0</bias_mean>
            <bias_stddev>1e-6</bias_stddev>
            <dynamic_bias_std>1e-6</dynamic_bias_std>
            <dynamic_bias_correlation_time>100</dynamic_bias_correlation_time>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1e-5</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>1e-6</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1e-5</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>1e-6</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1e-4</stddev>          <!-- Very low noise -->
            <bias_mean>0.0</bias_mean>
            <bias_stddev>1e-5</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1e-4</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>1e-5</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1e-4</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>1e-5</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Magnetometer Integration

For compass functionality:

```xml
<gazebo reference="magnetic_sensor">
  <sensor name="magnetometer" type="magnetometer">
    <always_on>true</always_on>
    <update_rate>50</update_rate>
    <visualize>false</visualize>
    <magnetometer>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>6e-6</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>6e-6</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>6e-6</stddev>
        </noise>
      </z>
    </magnetometer>

    <plugin name="mag_controller" filename="libgazebo_ros_mag.so">
      <topic>imu/mag</topic>
      <frame_name>mag_link</frame_name>
      <gaussian_noise>6e-6</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## Multi-Sensor Integration

### Sensor Fusion Simulation

Combining multiple sensors for enhanced perception:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="multi_sensor_humanoid">

  <!-- Robot body -->
  <link name="base_link">
    <inertial>
      <mass value="50.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </collision>
  </link>

  <!-- Head link with sensors -->
  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
  </joint>

  <link name="head_link">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Camera in head -->
  <joint name="camera_joint" type="fixed">
    <parent link="head_link"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="1e-5" ixy="0.0" ixz="0.0" iyy="1e-5" iyz="0.0" izz="1e-5"/>
    </inertial>
  </link>

  <!-- IMU in head -->
  <joint name="imu_joint" type="fixed">
    <parent link="head_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1e-6" ixy="0.0" ixz="0.0" iyy="1e-6" iyz="0.0" izz="1e-6"/>
    </inertial>
  </link>

  <!-- LiDAR mount -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.1 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="1e-4" ixy="0.0" ixz="0.0" iyy="1e-4" iyz="0.0" izz="1e-4"/>
    </inertial>
  </link>

  <!-- Gazebo sensor definitions -->
  <!-- Camera -->
  <gazebo reference="camera_link">
    <sensor name="front_camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="front_cam">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>30</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_optical_frame</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>30</max_depth>
        <topic>camera/image_raw</topic>
        <camera_info_topic>camera/camera_info</camera_info_topic>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU -->
  <gazebo reference="imu_link">
    <sensor name="head_imu" type="imu">
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
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <frame_name>imu_link</frame_name>
        <topic>imu/data</topic>
        <updateRateHZ>100.0</updateRateHZ>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LiDAR -->
  <gazebo reference="lidar_link">
    <sensor name="front_lidar" type="ray">
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
      <plugin name="lidar_controller" filename="libgazebo_ros_lidar.so">
        <topic>scan</topic>
        <frame_name>lidar_link</frame_name>
        <min_range>0.1</min_range>
        <max_range>30.0</max_range>
        <update_rate>10</update_rate>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Sensor Data Processing

### Real-Time Sensor Data Processing

Processing sensor data in real-time for humanoid applications:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import tf2_ros
from geometry_msgs.msg import TransformStamped

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Latest sensor data
        self.latest_image = None
        self.latest_imu = None
        self.latest_scan = None

        # TF broadcaster for sensor transformations
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        # Processed data publishers
        self.processed_image_pub = self.create_publisher(
            Image, 'camera/processed_image', 10)
        self.fused_data_pub = self.create_publisher(
            Imu, 'sensor_fusion/imu_fused', 10)

        # Timer for processing
        self.process_timer = self.create_timer(0.033, self.process_sensors)  # ~30Hz

        self.get_logger().info('Sensor Processor initialized')

    def image_callback(self, msg):
        """Process camera image data"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process image (example: edge detection)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Convert back to ROS format
            processed_msg = self.cv_bridge.cv2_to_imgmsg(edges, encoding='mono8')
            processed_msg.header = msg.header

            self.latest_image = processed_msg
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def imu_callback(self, msg):
        """Process IMU data"""
        # Store latest IMU data
        self.latest_imu = msg

        # Process orientation data
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Convert to Euler angles for easier interpretation
        rotation = R.from_quat(orientation)
        euler_angles = rotation.as_euler('xyz', degrees=True)

        # Log orientation for monitoring
        self.get_logger().debug(f'Orientation: {euler_angles}')

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.latest_scan = msg

        # Process scan data (example: find nearest obstacle)
        if len(msg.ranges) > 0:
            valid_ranges = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
            if valid_ranges:
                min_distance = min(valid_ranges)
                self.get_logger().debug(f'Min distance: {min_distance:.2f}m')

    def process_sensors(self):
        """Process all sensor data and generate fused outputs"""
        if self.latest_image is not None:
            self.processed_image_pub.publish(self.latest_image)

        if self.latest_imu is not None and self.latest_scan is not None:
            # Create fused data message
            fused_msg = self.fuse_sensor_data(self.latest_imu, self.latest_scan)
            self.fused_data_pub.publish(fused_msg)

    def fuse_sensor_data(self, imu_data, scan_data):
        """Fuse IMU and LiDAR data for improved perception"""
        # Create a fused IMU message with additional metadata
        fused_imu = Imu()
        fused_imu.header = imu_data.header
        fused_imu.orientation = imu_data.orientation
        fused_imu.orientation_covariance = imu_data.orientation_covariance
        fused_imu.angular_velocity = imu_data.angular_velocity
        fused_imu.angular_velocity_covariance = imu_data.angular_velocity_covariance
        fused_imu.linear_acceleration = imu_data.linear_acceleration
        fused_imu.linear_acceleration_covariance = imu_data.linear_acceleration_covariance

        # Add LiDAR-derived information as metadata
        # In practice, this would involve more sophisticated fusion algorithms
        if len(scan_data.ranges) > 0:
            valid_ranges = [r for r in scan_data.ranges if scan_data.range_min <= r <= scan_data.range_max]
            if valid_ranges:
                min_dist = min(valid_ranges)
                # Add this as a custom field or in a separate fused message
                # For now, we'll just log it
                self.get_logger().debug(f'Fused perception: nearest obstacle {min_dist:.2f}m')

        return fused_imu

    def broadcast_transforms(self):
        """Broadcast sensor frame transforms"""
        # Example transform for camera
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'head_link'
        t.child_frame_id = 'camera_optical_frame'

        t.transform.translation.x = 0.05
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Optical frame rotation (to align with camera convention)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down sensor processor')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Validation and Calibration

### Sensor Validation Tools

Validating that sensors behave correctly in simulation:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Float64
import numpy as np
from collections import deque
import statistics

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Data storage for validation
        self.imu_data_history = deque(maxlen=1000)
        self.scan_data_history = deque(maxlen=100)
        self.image_stats_history = deque(maxlen=100)

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.validate_imu, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.validate_scan, 10)
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.validate_image, 10)

        # Validation result publishers
        self.imu_validation_pub = self.create_publisher(
            Float64, 'validation/imu_stability', 10)
        self.scan_validation_pub = self.create_publisher(
            Float64, 'validation/scan_consistency', 10)

        # Timer for validation reporting
        self.validation_timer = self.create_timer(1.0, self.report_validation)

        self.get_logger().info('Sensor Validator initialized')

    def validate_imu(self, msg):
        """Validate IMU data for stability and range"""
        # Check for reasonable values
        angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        linear_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Store for statistical analysis
        self.imu_data_history.append({
            'angular_velocity': np.linalg.norm(angular_vel),
            'linear_acceleration': np.linalg.norm(linear_acc),
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        })

        # Check for extreme values
        if np.any(np.abs(angular_vel) > 100) or np.any(np.abs(linear_acc) > 100):
            self.get_logger().warn('IMU values outside expected range')

    def validate_scan(self, msg):
        """Validate LiDAR scan data"""
        # Check for reasonable ranges
        valid_ranges = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]

        # Store statistics
        if valid_ranges:
            self.scan_data_history.append({
                'min_range': min(valid_ranges),
                'max_range': max(valid_ranges),
                'valid_points': len(valid_ranges),
                'total_points': len(msg.ranges)
            })

    def validate_image(self, msg):
        """Validate camera image data"""
        try:
            # Convert to numpy array for analysis
            if msg.encoding in ['rgb8', 'bgr8']:
                # Calculate basic statistics
                image_data = np.frombuffer(msg.data, dtype=np.uint8)
                image_data = image_data.reshape((msg.height, msg.width, 3))

                # Calculate mean and std deviation
                mean_intensity = np.mean(image_data)
                std_intensity = np.std(image_data)

                self.image_stats_history.append({
                    'mean': mean_intensity,
                    'std': std_intensity,
                    'shape': image_data.shape
                })

        except Exception as e:
            self.get_logger().error(f'Image validation error: {e}')

    def report_validation(self):
        """Report validation statistics"""
        if self.imu_data_history:
            angular_velocities = [d['angular_velocity'] for d in self.imu_data_history]
            linear_accels = [d['linear_acceleration'] for d in self.imu_data_history]

            avg_angular_vel = sum(angular_velocities) / len(angular_velocities)
            avg_linear_accel = sum(linear_accels) / len(linear_accels)

            # Publish stability metric (lower is better)
            stability_metric = (avg_angular_vel + avg_linear_accel) / 2.0
            stability_msg = Float64()
            stability_msg.data = stability_metric
            self.imu_validation_pub.publish(stability_msg)

            self.get_logger().info(f'IMU Stability: {stability_metric:.4f}')

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Shutting down sensor validator')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Sensor Performance Tuning

Optimizing sensor simulation for better performance:

```xml
<!-- Optimize camera performance -->
<gazebo reference="optimized_camera">
  <sensor name="perf_camera" type="camera">
    <update_rate>15</update_rate>  <!-- Reduce update rate if high FPS not needed -->
    <camera name="optimized_cam">
      <horizontal_fov>1.047</horizontal_fov>  <!-- Narrower FOV reduces rendering load -->
      <image>
        <width>320</width>    <!-- Lower resolution for better performance -->
        <height>240</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>20.0</far>       <!-- Reduce far clip for performance -->
      </clip>
    </camera>
    <plugin name="perf_camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>20.0</max_depth>
      <update_rate>15</update_rate>  <!-- Match sensor update rate -->
      <topic>camera/performance_image</topic>
    </plugin>
  </sensor>
</gazebo>

<!-- Optimize LiDAR performance -->
<gazebo reference="optimized_lidar">
  <sensor name="perf_lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>        <!-- Half the samples for performance -->
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>15.0</max>                 <!-- Reduced range for performance -->
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="perf_lidar_controller" filename="libgazebo_ros_lidar.so">
      <topic>scan/performance</topic>
      <frame_name>lidar_frame</frame_name>
      <min_range>0.1</min_range>
      <max_range>15.0</max_range>
      <update_rate>5</update_rate>       <!-- Lower update rate for performance -->
    </plugin>
  </sensor>
</gazebo>
```

## Troubleshooting Common Issues

### 1. Sensor Data Not Publishing

**Symptoms**: No data on sensor topics despite simulation running.

**Causes and Solutions**:
- Plugin not loaded: Check Gazebo logs for plugin loading errors
- Wrong topic names: Verify topic names match expected subscriptions
- Frame names: Ensure frame names match TF expectations

### 2. High CPU Usage

**Symptoms**: Simulation runs slowly or CPU usage is very high.

**Solutions**:
- Reduce sensor update rates
- Lower image resolutions
- Reduce LiDAR samples
- Disable visualization when not needed

### 3. Inconsistent Data

**Symptoms**: Sensor data appears noisy or inconsistent.

**Solutions**:
- Check noise parameters
- Verify timing synchronization
- Validate sensor mounting positions

## Best Practices

### 1. Realistic Noise Modeling

Always include realistic noise models that match real sensors:

```xml
<!-- Example of realistic sensor noise -->
<sensor name="realistic_camera" type="camera">
  <camera name="real_cam">
    <noise>
      <type>gaussian</type>
      <!-- Use real sensor specifications -->
      <mean>0.0</mean>
      <stddev>0.01</stddev>  <!-- Based on real camera characteristics -->
    </noise>
  </camera>
</sensor>
```

### 2. Sensor Placement Consideration

Place sensors considering their real-world constraints:

```xml
<!-- Proper optical frame placement -->
<joint name="camera_optical_joint" type="fixed">
  <parent link="camera_link"/>
  <child link="camera_optical_frame"/>
  <!-- Rotate to optical frame convention -->
  <origin xyz="0 0 0" rpy="-1.57079632679 0 -1.57079632679"/>
</joint>

<link name="camera_optical_frame"/>
```

### 3. Performance vs. Accuracy Trade-offs

Balance simulation accuracy with computational performance based on application needs.

## Summary

Sensor simulation is fundamental to Physical AI development, providing the perception capabilities that enable robots to understand and interact with their environment. The key to effective sensor simulation lies in:

1. **Realistic modeling**: Accurately simulating sensor characteristics, noise, and limitations
2. **Proper integration**: Ensuring sensors are correctly mounted and calibrated
3. **Performance optimization**: Balancing accuracy with computational efficiency
4. **Validation**: Continuously validating simulation results against real-world behavior

By mastering these sensor simulation techniques, you'll be able to develop and test sophisticated Physical AI systems that can perceive and navigate the physical world effectively.