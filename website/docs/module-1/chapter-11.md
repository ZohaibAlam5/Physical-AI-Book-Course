---
title: Chapter 11 - ROS 2 Best Practices for Physical AI
description: Best practices for developing ROS 2 applications in Physical AI and humanoid robotics
sidebar_position: 11
---

# Chapter 11: ROS 2 Best Practices for Physical AI

## Learning Objectives

After completing this chapter, you should be able to:
- Apply best practices for ROS 2 development in Physical AI applications
- Structure ROS 2 packages for humanoid robotics
- Implement efficient and maintainable code patterns
- Follow safety and reliability guidelines for robot systems

## Introduction

Developing ROS 2 applications for Physical AI and humanoid robotics requires adherence to specific best practices to ensure safety, reliability, and maintainability. This chapter covers the essential best practices that every Physical AI developer should follow.

## Package Structure Best Practices

### Recommended Package Layout

For humanoid robotics applications, follow this recommended package structure:

```
humanoid_robot/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── controllers.yaml
│   ├── sensors.yaml
│   └── parameters.yaml
├── launch/
│   ├── robot.launch.py
│   ├── simulation.launch.py
│   └── bringup.launch.py
├── src/
│   ├── controllers/
│   │   ├── balance_controller.cpp
│   │   ├── walking_controller.cpp
│   │   └── manipulation_controller.cpp
│   ├── perception/
│   │   ├── object_detector.cpp
│   │   ├── pose_estimator.cpp
│   │   └── environment_mapper.cpp
│   └── behavior/
│       ├── state_machine.cpp
│       ├── planner.cpp
│       └── executor.cpp
├── include/
│   └── humanoid_robot/
│       ├── controllers/
│       ├── perception/
│       └── behavior/
├── scripts/
│   ├── calibration.py
│   ├── diagnostics.py
│   └── utilities.py
├── urdf/
│   ├── robot.urdf.xacro
│   └── materials.xacro
├── meshes/
│   └── visual/
└── test/
    ├── test_controllers.cpp
    ├── test_perception.cpp
    └── integration_tests.py
```

### Naming Conventions

Follow consistent naming conventions:

```python
# Python nodes should be lowercase with underscores
class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller_node')

        # Topics: lowercase with underscores
        self.imu_sub = self.create_subscription(
            Imu, 'sensors/imu/data', self.imu_callback, 10)

        # Services: lowercase with underscores
        self.calibrate_srv = self.create_service(
            Trigger, 'calibration/trigger', self.calibrate_callback)

        # Parameters: lowercase with underscores
        self.declare_parameter('control_frequency', 100)
```

## Node Design Best Practices

### Proper Node Initialization

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from builtin_interfaces.msg import Time
import traceback

class RobustHumanoidNode(Node):
    def __init__(self):
        # Use a descriptive node name
        super().__init__('humanoid_control_node')

        # Declare parameters early with defaults
        self.declare_parameter('control_frequency', 100)
        self.declare_parameter('safety_timeout', 1.0)
        self.declare_parameter('emergency_stop', False)

        # Initialize internal state
        self.initialized = False
        self.emergency_active = False

        # Set up QoS profiles appropriately
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        control_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Initialize subscribers, publishers, services
        self.setup_communication_interfaces()

        # Initialize timers
        self.setup_timers()

        # Finalize initialization
        self.initialized = True
        self.get_logger().info('Humanoid Control Node initialized successfully')

    def setup_communication_interfaces(self):
        """Initialize all communication interfaces"""
        try:
            # Subscribers
            self.joint_state_sub = self.create_subscription(
                JointState, 'joint_states', self.joint_state_callback, 10)

            self.imu_sub = self.create_subscription(
                Imu, 'sensors/imu/data_raw', self.imu_callback, 10)

            # Publishers
            self.joint_cmd_pub = self.create_publisher(
                JointState, 'joint_commands', 10)

            # Services
            self.emergency_stop_srv = self.create_service(
                Trigger, 'emergency_stop', self.emergency_stop_callback)

        except Exception as e:
            self.get_logger().error(f'Failed to setup communication interfaces: {e}')
            raise

    def setup_timers(self):
        """Initialize all timers"""
        control_freq = self.get_parameter('control_frequency').value

        self.control_timer = self.create_timer(
            1.0/control_freq, self.control_loop,
            clock=self.get_clock())

    def joint_state_callback(self, msg):
        """Process joint state messages"""
        if not self.initialized:
            return

        try:
            # Process the message
            self.update_robot_state(msg)
        except Exception as e:
            self.get_logger().error(f'Error in joint state callback: {e}')
            self.handle_error(e)

    def control_loop(self):
        """Main control loop"""
        if not self.initialized or self.emergency_active:
            return

        try:
            # Main control logic
            commands = self.compute_control_commands()
            self.joint_cmd_pub.publish(commands)
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
            self.activate_emergency_stop()

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop service calls"""
        self.activate_emergency_stop()
        response.success = True
        response.message = "Emergency stop activated"
        return response

    def activate_emergency_stop(self):
        """Activate emergency stop procedures"""
        self.emergency_active = True
        # Stop all motion
        stop_cmd = JointState()
        # Publish stop commands
        self.joint_cmd_pub.publish(stop_cmd)
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')

    def update_robot_state(self, joint_state_msg):
        """Update internal robot state representation"""
        # Implement state update logic
        pass

    def compute_control_commands(self):
        """Compute control commands based on current state"""
        # Implement control logic
        return JointState()

    def handle_error(self, error):
        """Generic error handling"""
        self.get_logger().error(f'Handling error: {error}')
        traceback.print_exc()
```

## Memory Management and Performance

### Efficient Message Handling

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from collections import deque
import numpy as np

class EfficientMessageHandler(Node):
    def __init__(self):
        super().__init__('efficient_handler')

        # Use deques for efficient history management
        self.joint_history = deque(maxlen=100)  # Keep last 100 messages

        # Pre-allocate numpy arrays for numerical computations
        self.joint_buffer = np.zeros(20, dtype=np.float64)  # Pre-allocated buffer

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.optimized_callback, 10)

    def optimized_callback(self, msg):
        """Optimized message callback"""
        # Avoid copying data unnecessarily
        self.update_joint_buffer(msg)

        # Process with numpy for efficiency
        if len(self.joint_history) > 10:
            velocity_estimate = self.estimate_velocity()

        # Store efficiently
        self.joint_history.append({
            'timestamp': msg.header.stamp,
            'positions': msg.position.copy()  # Only copy when necessary
        })

    def update_joint_buffer(self, msg):
        """Efficiently update pre-allocated buffer"""
        # Direct assignment for efficiency
        for i, pos in enumerate(msg.position[:len(self.joint_buffer)]):
            self.joint_buffer[i] = pos

    def estimate_velocity(self):
        """Estimate joint velocities from history"""
        if len(self.joint_history) < 2:
            return np.zeros(len(self.joint_buffer))

        # Use numpy for efficient computation
        prev_positions = np.array(self.joint_history[-2]['positions'])
        curr_positions = np.array(self.joint_history[-1]['positions'])

        dt = (self.joint_history[-1]['timestamp'].nanosec -
              self.joint_history[-2]['timestamp'].nanosec) / 1e9

        if dt <= 0:
            return np.zeros(len(self.joint_buffer))

        return (curr_positions - prev_positions) / dt
```

## Safety and Reliability Patterns

### Safety Monitor Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration
from threading import Lock

class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Safety state
        self.safety_lock = Lock()
        self.safety_ok = True
        self.emergency_reason = ""
        self.last_valid_time = self.get_clock().now()

        # Safety parameters
        self.declare_parameter('max_joint_velocity', 5.0)  # rad/s
        self.declare_parameter('max_imu_angular_velocity', 10.0)  # rad/s
        self.declare_parameter('safety_timeout', 0.5)  # seconds
        self.declare_parameter('fall_threshold', 0.3)  # radians from upright

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'sensors/imu/data', self.imu_callback, 10)

        # Publishers
        self.safety_pub = self.create_publisher(Bool, 'safety_ok', 10)
        self.emergency_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.01, self.safety_check)  # 100Hz

        self.get_logger().info('Safety Monitor initialized')

    def joint_callback(self, msg):
        """Monitor joint states for safety violations"""
        with self.safety_lock:
            # Check for excessive velocities
            if hasattr(msg, 'velocity') and msg.velocity:
                max_vel = self.get_parameter('max_joint_velocity').value
                for vel in msg.velocity:
                    if abs(vel) > max_vel:
                        self.trigger_safety_violation(
                            f'Joint velocity exceeded: {vel} > {max_vel}')
                        return

            # Update last valid time
            self.last_valid_time = self.get_clock().now()

    def imu_callback(self, msg):
        """Monitor IMU data for safety violations"""
        with self.safety_lock:
            # Check for excessive angular velocities
            ang_vel = msg.angular_velocity
            max_ang_vel = self.get_parameter('max_imu_angular_velocity').value

            if (abs(ang_vel.x) > max_ang_vel or
                abs(ang_vel.y) > max_ang_vel or
                abs(ang_vel.z) > max_ang_vel):
                self.trigger_safety_violation(
                    f'Angular velocity exceeded: ({ang_vel.x}, {ang_vel.y}, {ang_vel.z})')
                return

            # Check for potential fall
            fall_threshold = self.get_parameter('fall_threshold').value
            orientation = msg.orientation

            # Simple upright check (assuming z-axis is up)
            if abs(orientation.z) < abs(orientation.w) * np.cos(fall_threshold):
                self.trigger_safety_violation('Potential fall detected')
                return

    def safety_check(self):
        """Periodic safety checks"""
        with self.safety_lock:
            # Check for timeouts
            current_time = self.get_clock().now()
            timeout = Duration()
            timeout.sec = int(self.get_parameter('safety_timeout').value)
            timeout.nanosec = int((self.get_parameter('safety_timeout').value % 1) * 1e9)

            if (current_time - self.last_valid_time).nanoseconds > timeout.nanosec:
                if self.safety_ok:
                    self.trigger_safety_violation('Sensor timeout')
                return

            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = self.safety_ok
            self.safety_pub.publish(safety_msg)

    def trigger_safety_violation(self, reason):
        """Trigger safety violation procedures"""
        if not self.safety_ok:
            return  # Already in safety violation state

        self.safety_ok = False
        self.emergency_reason = reason

        # Publish emergency stop
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_pub.publish(emergency_msg)

        self.get_logger().fatal(f'SAFETY VIOLATION: {reason}')

        # Additional safety procedures could be implemented here
        # Such as stopping all controllers, logging, alerting operators, etc.
```

## Parameter Management

### Centralized Parameter Handling

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, FloatingPointRange
from typing import Dict, Any, Union

class ParameterManager(Node):
    def __init__(self):
        super().__init__('parameter_manager')

        # Define parameter schemas
        self.parameter_schemas = {
            'control.gains.position': {
                'type': rclpy.Parameter.Type.DOUBLE,
                'default': 10.0,
                'description': 'Position control gain',
                'range': FloatingPointRange(from_value=0.0, to_value=100.0)
            },
            'control.gains.velocity': {
                'type': rclpy.Parameter.Type.DOUBLE,
                'default': 1.0,
                'description': 'Velocity control gain',
                'range': FloatingPointRange(from_value=0.0, to_value=10.0)
            },
            'safety.max_torque': {
                'type': rclpy.Parameter.Type.DOUBLE,
                'default': 100.0,
                'description': 'Maximum allowed torque',
                'range': FloatingPointRange(from_value=0.0, to_value=500.0)
            },
            'control.frequency': {
                'type': rclpy.Parameter.Type.INTEGER,
                'default': 100,
                'description': 'Control loop frequency',
                'range': IntegerRange(from_value=10, to_value=1000)
            }
        }

        # Declare all parameters
        self.declare_parameters_with_schema()

        # Setup parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info('Parameter Manager initialized')

    def declare_parameters_with_schema(self):
        """Declare parameters with proper descriptors"""
        for param_name, schema in self.parameter_schemas.items():
            descriptor = ParameterDescriptor()
            descriptor.description = schema['description']

            if 'range' in schema:
                if isinstance(schema['range'], FloatingPointRange):
                    descriptor.floating_point_range = [schema['range']]
                elif isinstance(schema['range'], IntegerRange):
                    descriptor.integer_range = [schema['range']]

            self.declare_parameter(
                param_name,
                schema['default'],
                descriptor=descriptor
            )

    def parameter_callback(self, params):
        """Validate parameter changes"""
        successful_params = []

        for param in params:
            if param.name in self.parameter_schemas:
                schema = self.parameter_schemas[param.name]

                # Type validation
                if param.type_ != schema['type']:
                    self.get_logger().error(
                        f'Invalid type for {param.name}: expected {schema["type"]}, got {param.type_}'
                    )
                    return False

                # Range validation
                if 'range' in schema:
                    range_val = schema['range']
                    if isinstance(range_val, FloatingPointRange):
                        if not (range_val.from_value <= param.value <= range_val.to_value):
                            self.get_logger().error(
                                f'Value {param.value} out of range for {param.name}'
                            )
                            return False
                    elif isinstance(range_val, IntegerRange):
                        if not (range_val.from_value <= param.value <= range_val.to_value):
                            self.get_logger().error(
                                f'Value {param.value} out of range for {param.name}'
                            )
                            return False

                successful_params.append(param)
            else:
                self.get_logger().warning(f'Unknown parameter: {param.name}')

        return successful_params

    def get_control_gain(self, gain_type: str) -> float:
        """Get control gain with proper error handling"""
        param_name = f'control.gains.{gain_type}'

        if param_name not in self.parameter_schemas:
            self.get_logger().error(f'Unknown gain type: {gain_type}')
            return self.parameter_schemas['control.gains.position']['default']

        return self.get_parameter(param_name).value

    def get_safety_limit(self, limit_type: str) -> float:
        """Get safety limit with proper error handling"""
        param_name = f'safety.max_{limit_type}'

        if param_name not in self.parameter_schemas:
            self.get_logger().error(f'Unknown limit type: {limit_type}')
            return 0.0

        return self.get_parameter(param_name).value
```

## Logging and Diagnostics

### Structured Logging

```python
import rclpy
from rclpy.node import Node
from diagnostic_updater import Updater, DiagnosticStatusWrapper
from diagnostic_msgs.msg import DiagnosticArray
import json
from datetime import datetime

class DiagnosticsNode(Node):
    def __init__(self):
        super().__init__('diagnostics_node')

        # Initialize diagnostic updater
        self.diag_updater = Updater(self)
        self.diag_updater.setHardwareID('humanoid_robot_v1.0')

        # Add diagnostic tasks
        self.diag_updater.add('Joint Health', self.joint_health_diagnostic)
        self.diag_updater.add('Sensor Status', self.sensor_status_diagnostic)
        self.diag_updater.add('Control Performance', self.control_performance_diagnostic)

        # Setup diagnostic timer
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

        # Performance tracking
        self.control_loop_times = []
        self.message_counts = {}

        self.get_logger().info('Diagnostics Node initialized')

    def joint_health_diagnostic(self, stat):
        """Diagnostic for joint health"""
        try:
            # Simulate checking joint health
            joint_temp_warning = False
            joint_error_count = 0

            # Add key values for monitoring
            stat.add('Joint Count', 24)  # Example: 24 joints
            stat.add('Temperature Warnings', 0)
            stat.add('Errors', 0)

            # Determine status
            if joint_error_count > 5:
                stat.summary(DiagnosticStatusWrapper.ERROR, "Multiple joint errors detected")
            elif joint_temp_warning:
                stat.summary(DiagnosticStatusWrapper.WARN, "Joint temperature warnings")
            else:
                stat.summary(DiagnosticStatusWrapper.OK, "All joints healthy")

        except Exception as e:
            stat.summary(DiagnosticStatusWrapper.ERROR, f"Diagnostic error: {e}")

        return stat

    def sensor_status_diagnostic(self, stat):
        """Diagnostic for sensor status"""
        try:
            # Simulate checking sensor status
            sensors_ok = 8  # Out of total sensors
            total_sensors = 10

            stat.add('Sensors OK', sensors_ok)
            stat.add('Total Sensors', total_sensors)
            stat.add('Health Percentage', f'{(sensors_ok/total_sensors)*100:.1f}%')

            if sensors_ok < total_sensors * 0.8:  # Less than 80% OK
                stat.summary(DiagnosticStatusWrapper.ERROR, "Critical sensor failures")
            elif sensors_ok < total_sensors * 0.95:  # Less than 95% OK
                stat.summary(DiagnosticStatusWrapper.WARN, "Some sensor issues")
            else:
                stat.summary(DiagnosticStatusWrapper.OK, "All sensors nominal")

        except Exception as e:
            stat.summary(DiagnosticStatusWrapper.ERROR, f"Diagnostic error: {e}")

        return stat

    def control_performance_diagnostic(self, stat):
        """Diagnostic for control performance"""
        try:
            # Calculate average control loop time
            if self.control_loop_times:
                avg_time = sum(self.control_loop_times) / len(self.control_loop_times)
                max_time = max(self.control_loop_times)

                stat.add('Average Loop Time (ms)', f'{avg_time*1000:.2f}')
                stat.add('Max Loop Time (ms)', f'{max_time*1000:.2f}')
                stat.add('Loop Count', len(self.control_loop_times))

                # Performance thresholds
                if avg_time > 0.02:  # 20ms threshold
                    stat.summary(DiagnosticStatusWrapper.WARN, "Control loop slow")
                else:
                    stat.summary(DiagnosticStatusWrapper.OK, "Control performance nominal")
            else:
                stat.summary(DiagnosticStatusWrapper.WARN, "No performance data available")

        except Exception as e:
            stat.summary(DiagnosticStatusWrapper.ERROR, f"Diagnostic error: {e}")

        return stat

    def publish_diagnostics(self):
        """Publish diagnostic information"""
        try:
            self.diag_updater.update()
        except Exception as e:
            self.get_logger().error(f'Diagnostic publishing error: {e}')

    def log_structured_event(self, event_type: str, details: dict):
        """Log structured events for analysis"""
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'node': self.get_name(),
            'details': details
        }

        # Log as JSON string for easy parsing
        self.get_logger().info(f'EVENT: {json.dumps(event_data)}')
```

## Testing and Validation

### Comprehensive Testing Approach

```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import time

class TestHumanoidController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize ROS context once for all tests"""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS context after all tests"""
        rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.node = Node('test_humanoid_controller')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        # Create test publishers and subscribers
        self.cmd_pub = self.node.create_publisher(
            Float64MultiArray, 'test_commands', 10)
        self.state_sub = self.node.create_subscription(
            JointState, 'test_states', self.state_callback, 10)

        self.received_states = []

    def state_callback(self, msg):
        """Collect received joint states"""
        self.received_states.append(msg)

    def test_joint_command_processing(self):
        """Test that joint commands are processed correctly"""
        # Create a test command
        cmd = Float64MultiArray()
        cmd.data = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example joint positions

        # Publish command
        self.cmd_pub.publish(cmd)

        # Wait for processing
        start_time = time.time()
        while len(self.received_states) == 0 and time.time() - start_time < 1.0:
            self.executor.spin_once(timeout_sec=0.1)

        # Verify that states were received
        self.assertGreater(len(self.received_states), 0)

    def test_safety_limits(self):
        """Test that safety limits are enforced"""
        # This would test that extreme commands are limited
        pass

    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        # This would test that emergency stop works correctly
        pass

# Integration test class
class TestHumanoidIntegration(unittest.TestCase):
    def test_full_system_integration(self):
        """Test integration of multiple components"""
        # This would test the integration of controllers,
        # perception, and behavior components
        pass

if __name__ == '__main__':
    unittest.main()
```

## Documentation and Code Quality

### API Documentation Standards

```python
from typing import Optional, List, Tuple
import numpy as np

class HumanoidKinematics:
    """
    Humanoid robot kinematics calculator.

    This class provides forward and inverse kinematics solutions for
    humanoid robots with standard anthropomorphic structure.

    Example:
        >>> kinematics = HumanoidKinematics()
        >>> ee_pose = kinematics.forward_kinematics([0.1, 0.2, 0.3])
        >>> joint_angles = kinematics.inverse_kinematics(ee_pose)

    Attributes:
        robot_config: Dictionary containing robot-specific parameters
        workspace_bounds: Tuple of (x_min, x_max, y_min, y_max, z_min, z_max)
    """

    def __init__(self, robot_config: Optional[dict] = None):
        """
        Initialize the kinematics calculator.

        Args:
            robot_config: Optional dictionary with robot-specific parameters.
                         If None, uses default parameters for generic humanoid.

        Raises:
            ValueError: If robot_config contains invalid parameters
            TypeError: If robot_config is not a dictionary
        """
        if robot_config is not None and not isinstance(robot_config, dict):
            raise TypeError("robot_config must be a dictionary or None")

        self.robot_config = robot_config or self._get_default_config()
        self.workspace_bounds = self._calculate_workspace()

    def forward_kinematics(self,
                          joint_angles: List[float],
                          link_name: str = 'end_effector'
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward kinematics for given joint angles.

        Computes the position and orientation of a specified link given
        the joint angles of the robot.

        Args:
            joint_angles: List of joint angles in radians
            link_name: Name of the link to calculate FK for

        Returns:
            Tuple of (position, orientation) where:
            - position: 3-element numpy array [x, y, z]
            - orientation: 4-element numpy array [qx, qy, qz, qw] (quaternion)

        Raises:
            ValueError: If joint_angles length doesn't match expected
            ValueError: If link_name is not valid
            RuntimeError: If FK calculation fails
        """
        # Implementation would go here
        pass

    def inverse_kinematics(self,
                          target_pose: Tuple[np.ndarray, np.ndarray],
                          seed_angles: Optional[List[float]] = None,
                          max_iterations: int = 100
                          ) -> List[float]:
        """
        Calculate inverse kinematics for target pose.

        Finds joint angles that achieve a target end-effector pose.

        Args:
            target_pose: Tuple of (position, orientation) where:
                        - position: 3-element array [x, y, z]
                        - orientation: 4-element array [qx, qy, qz, qw]
            seed_angles: Initial guess for joint angles (optional)
            max_iterations: Maximum number of iterations for IK solver

        Returns:
            List of joint angles that achieve the target pose, or empty list
            if no solution is found

        Raises:
            ValueError: If target_pose is malformed
            RuntimeError: If IK solver fails to converge
        """
        # Implementation would go here
        pass

    def _get_default_config(self) -> dict:
        """Get default robot configuration."""
        return {
            'arm_length': 0.3,
            'leg_length': 0.4,
            'torso_height': 0.5,
            'joint_limits': {
                'shoulder': (-1.57, 1.57),
                'elbow': (-1.57, 0.0),
                'hip': (-0.5, 1.0),
                'knee': (0.0, 1.57),
                'ankle': (-0.5, 0.5)
            }
        }
```

## Summary

Following these best practices will help you develop robust, safe, and maintainable ROS 2 applications for Physical AI and humanoid robotics. The key areas to focus on include:

1. **Proper node structure** with error handling and lifecycle management
2. **Efficient memory management** and performance optimization
3. **Safety and reliability** with monitoring and emergency procedures
4. **Parameter management** with validation and ranges
5. **Comprehensive logging and diagnostics** for system monitoring
6. **Thorough testing** with unit and integration tests
7. **Good documentation** with clear APIs and examples

These practices will help ensure your Physical AI systems are reliable, safe, and maintainable for long-term operation.