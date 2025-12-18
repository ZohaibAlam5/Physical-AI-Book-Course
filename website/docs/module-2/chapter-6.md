---
title: Chapter 6 - Control Architecture for Humanoid Systems
description: Comprehensive control architecture for humanoid robot systems
sidebar_position: 6
---

# Chapter 6: Control Architecture for Humanoid Systems

## Learning Objectives

After completing this chapter, you should be able to:
- Design hierarchical control architectures for humanoid robots
- Implement feedback control systems with appropriate stability margins
- Integrate multiple control loops for simultaneous task execution
- Apply advanced control techniques like MPC and LQR to humanoid systems
- Design robust control systems that handle uncertainties and disturbances

## Introduction

Control architecture for humanoid robots is significantly more complex than for simpler robotic systems due to their many degrees of freedom, underactuated nature, and need to maintain balance while performing tasks. A well-designed control architecture must handle multiple simultaneous objectives: maintaining balance, executing manipulation tasks, avoiding obstacles, and respecting joint limits. This chapter explores the fundamental principles and practical implementations of control systems for humanoid robots.

## Hierarchical Control Architecture

### Multi-Level Control Structure

Humanoid robots require a multi-level control architecture to handle the complexity of simultaneous balance and task execution:

```
┌─────────────────────────────────────────────────────────┐
│                    High-Level Planner                   │
│              (Trajectories, Behaviors)                  │
├─────────────────────────────────────────────────────────┤
│                   Task-Level Control                    │
│        (Balance, Manipulation, Locomotion)              │
├─────────────────────────────────────────────────────────┤
│                  Joint-Level Control                    │
│        (Individual Joint Control, Torques)              │
├─────────────────────────────────────────────────────────┤
│                 Hardware Interface                      │
│          (Motor Drivers, Encoders, etc.)                │
└─────────────────────────────────────────────────────────┘
```

### Control Hierarchy Implementation

```python
import numpy as np
from abc import ABC, abstractmethod
import time
from scipy.linalg import solve_continuous_are
from scipy.signal import lti, lsim

class ControlLayer(ABC):
    """Abstract base class for control layers"""

    def __init__(self, name, update_rate):
        self.name = name
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        self.last_update = time.time()

    @abstractmethod
    def compute_commands(self, state, reference):
        """Compute control commands based on state and reference"""
        pass

    def update_if_ready(self, state, reference):
        """Update control if sufficient time has passed"""
        current_time = time.time()
        if current_time - self.last_update >= self.dt:
            commands = self.compute_commands(state, reference)
            self.last_update = current_time
            return commands
        return None

class HighLevelPlanner(ControlLayer):
    """High-level trajectory planner and behavior selector"""

    def __init__(self):
        super().__init__("HighLevelPlanner", 10)  # 10Hz planning
        self.current_behavior = "stand"
        self.trajectory_generators = {
            "walk": self.generate_walk_trajectory,
            "reach": self.generate_reach_trajectory,
            "balance": self.generate_balance_trajectory,
            "stand": self.generate_stand_trajectory
        }

    def compute_commands(self, state, reference):
        """Generate high-level trajectory references"""
        if reference.get('behavior') != self.current_behavior:
            self.current_behavior = reference['behavior']

        # Generate trajectory based on current behavior
        trajectory = self.trajectory_generators[self.current_behavior](state, reference)

        return {
            'trajectory': trajectory,
            'behavior': self.current_behavior,
            'next_behavior': reference.get('next_behavior', self.current_behavior)
        }

    def generate_walk_trajectory(self, state, reference):
        """Generate walking trajectory"""
        # Calculate step timing and foot placements
        step_length = reference.get('step_length', 0.3)
        step_height = reference.get('step_height', 0.05)
        step_duration = reference.get('step_duration', 0.8)

        # Calculate number of steps needed
        distance_to_travel = reference.get('distance', 1.0)
        n_steps = int(distance_to_travel / step_length)

        trajectory = []
        current_time = 0

        for i in range(n_steps):
            # Calculate foot positions for this step
            left_foot = np.array([
                i * step_length,
                0.1 if i % 2 == 0 else -0.1,  # Alternate feet positions
                0.0
            ])

            right_foot = np.array([
                i * step_length,
                -0.1 if i % 2 == 0 else 0.1,  # Alternate feet positions
                0.0
            ])

            # Calculate CoM trajectory to maintain balance
            com_x = i * step_length + step_length / 2  # Lead the step
            com_y = 0.0  # Center between feet
            com_z = 0.8  # Maintain height

            trajectory.append({
                'time': current_time,
                'left_foot': left_foot,
                'right_foot': right_foot,
                'com_position': np.array([com_x, com_y, com_z]),
                'zmp_reference': np.array([com_x, com_y])  # Simplified ZMP reference
            })

            current_time += step_duration

        return trajectory

    def generate_reach_trajectory(self, state, reference):
        """Generate reaching trajectory"""
        target_pos = np.array(reference['target_position'])
        current_ee_pos = reference['current_end_effector_position']

        # Generate smooth trajectory to target
        duration = reference.get('duration', 2.0)
        n_points = int(duration * self.update_rate)

        trajectory = []
        for i in range(n_points):
            t = i / (n_points - 1) if n_points > 1 else 0

            # Cubic interpolation for smooth motion
            smooth_t = 3 * t**2 - 2 * t**3
            current_pos = (1 - smooth_t) * current_ee_pos + smooth_t * target_pos

            trajectory.append({
                'time': i * self.dt,
                'end_effector_position': current_pos,
                'end_effector_orientation': reference.get('target_orientation', [0, 0, 0, 1])
            })

        return trajectory

    def generate_balance_trajectory(self, state, reference):
        """Generate balance maintenance trajectory"""
        # Keep CoM over support polygon
        support_center = reference['support_polygon_center']

        trajectory = [{
            'time': 0,
            'com_position': np.array([support_center[0], support_center[1], 0.8]),
            'zmp_reference': support_center
        }]

        return trajectory

    def generate_stand_trajectory(self, state, reference):
        """Generate standing position trajectory"""
        # Maintain neutral standing pose
        neutral_com = np.array([0, 0, 0.8])

        trajectory = [{
            'time': 0,
            'com_position': neutral_com,
            'zmp_reference': np.array([0, 0])
        }]

        return trajectory

class TaskLevelController(ControlLayer):
    """Task-level controller for balance, manipulation, and locomotion"""

    def __init__(self, robot_model):
        super().__init__("TaskLevelController", 100)  # 100Hz task control
        self.robot_model = robot_model
        self.gravity = 9.81

        # Balance control parameters
        self.com_kp = 50.0
        self.com_kd = 10.0
        self.zmp_kp = 100.0
        self.zmp_kd = 20.0

        # Manipulation control parameters
        self.ee_kp = 300.0
        self.ee_kd = 30.0

        # State estimators
        self.com_estimator = CenterOfMassEstimator(robot_model)
        self.zmp_calculator = ZMPCalculator(robot_model)

    def compute_commands(self, state, reference):
        """Compute task-level commands"""
        # Extract current state
        joint_positions = state['positions']
        joint_velocities = state['velocities']

        # Calculate current CoM and ZMP
        current_com = self.com_estimator.estimate_com(joint_positions)
        current_com_vel = self.com_estimator.estimate_com_velocity(joint_positions, joint_velocities)
        current_zmp = self.zmp_calculator.calculate_zmp(current_com, current_com_vel)

        # Determine which tasks to execute based on behavior
        behavior = reference.get('behavior', 'stand')

        task_commands = {}

        if behavior in ['walk', 'balance', 'stand']:
            # Balance control task
            balance_commands = self.compute_balance_control(
                current_com, current_com_vel, reference)
            task_commands['balance'] = balance_commands

        if behavior == 'reach':
            # Manipulation task
            manipulation_commands = self.compute_manipulation_control(
                state, reference)
            task_commands['manipulation'] = manipulation_commands

        # Combine tasks using priority-based approach
        combined_commands = self.combine_tasks_by_priority(task_commands, state)

        return combined_commands

    def compute_balance_control(self, current_com, current_com_vel, reference):
        """Compute balance control commands"""
        # Get reference CoM and ZMP
        ref_com = reference['com_position']
        ref_zmp = reference['zmp_reference']

        # Calculate balance errors
        com_error = ref_com[:2] - current_com[:2]  # Only x,y for balance
        com_vel_error = -current_com_vel[:2]  # Damping term (desired vel = 0)

        # Calculate balance correction using inverted pendulum model
        com_correction = (
            self.com_kp * com_error +
            self.com_kd * com_vel_error
        )

        # Also consider ZMP error
        zmp_error = ref_zmp - current_zmp
        zmp_correction = self.zmp_kp * zmp_error

        # Combine corrections
        total_balance_correction = 0.7 * com_correction + 0.3 * zmp_correction

        # Convert to joint space using CoM Jacobian transpose
        com_jacobian = self.robot_model.com_jacobian(joint_positions)
        balance_joints = com_jacobian[:2, :].T @ total_balance_correction

        return {
            'correction': total_balance_correction,
            'joint_commands': balance_joints,
            'type': 'balance'
        }

    def compute_manipulation_control(self, state, reference):
        """Compute manipulation control commands"""
        # Get current end-effector pose
        ee_link = reference.get('end_effector_link', 'right_hand')
        current_ee_pose = self.robot_model.forward_kinematics_for_link(
            state['positions'], ee_link)

        # Get reference pose
        ref_ee_pos = reference['end_effector_position']
        ref_ee_orient = reference.get('end_effector_orientation', [0, 0, 0, 1])

        # Calculate errors
        pos_error = ref_ee_pos - current_ee_pose['position']

        # For orientation error, use quaternion difference
        from scipy.spatial.transform import Rotation as R
        current_rot = R.from_quat(current_ee_pose['orientation'])
        desired_rot = R.from_quat(ref_ee_orient)
        relative_rot = desired_rot * current_rot.inv()
        orient_error = relative_rot.as_rotvec()

        # Calculate correction
        pos_correction = self.ee_kp * pos_error
        orient_correction = self.ee_kp * orient_error * 0.1  # Scale orientation correction

        total_correction = np.concatenate([pos_correction, orient_correction])

        # Convert to joint space using end-effector Jacobian
        ee_jacobian = self.robot_model.jacobian_for_link(state['positions'], ee_link)
        manipulation_joints = ee_jacobian.T @ total_correction

        return {
            'correction': total_correction,
            'joint_commands': manipulation_joints,
            'type': 'manipulation'
        }

    def combine_tasks_by_priority(self, task_commands, state):
        """Combine multiple tasks using priority-based null space projection"""
        # Start with highest priority task (balance)
        total_commands = np.zeros(len(state['positions']))

        # Apply balance commands first
        if 'balance' in task_commands:
            total_commands += task_commands['balance']['joint_commands']

            # Calculate null space of balance task
            balance_jacobian = self.robot_model.com_jacobian(state['positions'])[:2, :]
            balance_null_space = self.compute_null_space_projection(balance_jacobian)

            # Apply manipulation in balance null space
            if 'manipulation' in task_commands:
                manipulation_in_null = balance_null_space @ task_commands['manipulation']['joint_commands']
                total_commands += manipulation_in_null

        elif 'manipulation' in task_commands:
            # If no balance task, just apply manipulation
            total_commands += task_commands['manipulation']['joint_commands']

        return total_commands

    def compute_null_space_projection(self, jacobian):
        """Compute null space projection matrix"""
        try:
            j_pinv = np.linalg.pinv(jacobian)
            return np.eye(jacobian.shape[1]) - j_pinv @ jacobian
        except np.linalg.LinAlgError:
            # If pseudo-inverse fails, return identity (no null space projection)
            return np.eye(jacobian.shape[1])

class JointLevelController(ControlLayer):
    """Low-level joint controller for individual joint control"""

    def __init__(self, robot_model):
        super().__init__("JointLevelController", 1000)  # 1000Hz joint control
        self.robot_model = robot_model

        # Joint PID controllers
        self.joint_pids = {}
        joint_names = robot_model.get_joint_names()

        for joint_name in joint_names:
            self.joint_pids[joint_name] = {
                'kp': 100.0,  # Position gain
                'ki': 1.0,    # Integral gain
                'kd': 10.0,   # Derivative gain
                'integral': 0.0,
                'prev_error': 0.0
            }

        # Feedforward compensation
        self.gravity_compensation = GravityCompensator(robot_model)
        self.coriolis_compensation = CoriolisCompensator(robot_model)

    def compute_commands(self, state, reference):
        """Compute joint-level commands"""
        joint_positions = state['positions']
        joint_velocities = state['velocities']

        # Get desired joint positions from higher level
        desired_positions = reference.get('joint_positions', joint_positions)
        desired_velocities = reference.get('joint_velocities', np.zeros_like(joint_positions))
        desired_accelerations = reference.get('joint_accelerations', np.zeros_like(joint_positions))

        # Compute joint control commands
        joint_torques = np.zeros(len(joint_positions))

        for i, joint_name in enumerate(self.robot_model.get_joint_names()):
            # Calculate tracking errors
            pos_error = desired_positions[i] - joint_positions[i]
            vel_error = desired_velocities[i] - joint_velocities[i]

            # Update integral term
            self.joint_pids[joint_name]['integral'] += pos_error * self.dt

            # Calculate PID output
            p_term = self.joint_pids[joint_name]['kp'] * pos_error
            i_term = self.joint_pids[joint_name]['ki'] * self.joint_pids[joint_name]['integral']
            d_term = self.joint_pids[joint_name]['kd'] * vel_error

            pid_output = p_term + i_term + d_term

            # Add feedforward compensation
            gravity_comp = self.gravity_compensation.compensate(joint_positions)[i]
            coriolis_comp = self.coriolis_compensation.compensate(joint_positions, joint_velocities)[i]

            # Calculate commanded torque
            joint_torques[i] = pid_output + gravity_comp + coriolis_comp

            # Update previous error for derivative term
            self.joint_pids[joint_name]['prev_error'] = pos_error

        return {
            'joint_torques': joint_torques,
            'desired_positions': desired_positions,
            'desired_velocities': desired_velocities
        }

    def reset_integrators(self):
        """Reset PID integral terms"""
        for joint_name in self.joint_pids:
            self.joint_pids[joint_name]['integral'] = 0.0
            self.joint_pids[joint_name]['prev_error'] = 0.0

class CenterOfMassEstimator:
    """Estimate center of mass position and velocity"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def estimate_com(self, joint_positions):
        """Estimate CoM position from joint configuration"""
        return self.robot_model.calculate_com_position(joint_positions)

    def estimate_com_velocity(self, joint_positions, joint_velocities):
        """Estimate CoM velocity from joint configuration and velocities"""
        return self.robot_model.calculate_com_velocity(joint_positions, joint_velocities)

class ZMPCalculator:
    """Calculate Zero Moment Point from CoM information"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.com_height = 0.8  # Assumed CoM height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.com_height)

    def calculate_zmp(self, com_position, com_velocity):
        """Calculate ZMP using inverted pendulum model"""
        # ZMP_x = CoM_x - (CoM_height / gravity) * CoM_acc_x
        # For control, we often use: ZMP ≈ CoM - CoM_vel/omega
        # But for estimation from current state, we assume CoM_acc ≈ 0 in steady state

        # Simplified calculation: ZMP is approximately at CoM projection
        # For more accurate calculation, we'd need acceleration data
        return com_position[:2]  # x,y coordinates only

class GravityCompensator:
    """Compensate for gravity effects in joint control"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def compensate(self, joint_positions):
        """Calculate gravity compensation torques"""
        return self.robot_model.gravity_compensation(joint_positions)

class CoriolisCompensator:
    """Compensate for Coriolis and centrifugal effects"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def compensate(self, joint_positions, joint_velocities):
        """Calculate Coriolis compensation torques"""
        return self.robot_model.coriolis_compensation(joint_positions, joint_velocities)

class HumanoidControlArchitecture:
    """Main control architecture class that orchestrates all control layers"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

        # Initialize control layers
        self.high_level_planner = HighLevelPlanner()
        self.task_level_controller = TaskLevelController(robot_model)
        self.joint_level_controller = JointLevelController(robot_model)

        # State estimation
        self.state_estimator = StateEstimator(robot_model)

        # Safety monitors
        self.safety_monitor = SafetyMonitor(robot_model)

        # Timing control
        self.last_control_time = time.time()

    def compute_control_commands(self, sensor_data, user_reference):
        """Main control loop that computes commands for the robot"""
        current_time = time.time()
        dt = current_time - self.last_control_time
        self.last_control_time = current_time

        # Estimate current state from sensor data
        current_state = self.state_estimator.estimate_state(sensor_data)

        # Plan high-level trajectory
        high_level_output = self.high_level_planner.update_if_ready(
            current_state, user_reference)

        if high_level_output is None:
            # Use last known trajectory if planning not ready
            high_level_output = getattr(self.high_level_planner, '_last_trajectory', {})

        # Update high-level trajectory
        self.high_level_planner._last_trajectory = high_level_output

        # Compute task-level commands
        task_reference = {
            'behavior': high_level_output['behavior'],
            'trajectory': high_level_output.get('trajectory', []),
            **high_level_output.get('trajectory', [{}])[0] if high_level_output.get('trajectory') else {}
        }

        task_commands = self.task_level_controller.update_if_ready(
            current_state, task_reference)

        if task_commands is None:
            # Use last computed commands if task control not ready
            task_commands = getattr(self.task_level_controller, '_last_commands', {})

        # Update last commands
        self.task_level_controller._last_commands = task_commands

        # Compute joint-level commands
        joint_reference = {
            'joint_positions': task_commands.get('joint_positions', current_state['positions']),
            'joint_velocities': task_commands.get('joint_velocities', np.zeros_like(current_state['positions'])),
            'joint_accelerations': task_commands.get('joint_accelerations', np.zeros_like(current_state['positions']))
        }

        joint_commands = self.joint_level_controller.update_if_ready(
            current_state, joint_reference)

        if joint_commands is None:
            # Use zero commands if joint control not ready
            joint_commands = {
                'joint_torques': np.zeros(len(current_state['positions']))
            }

        # Apply safety checks
        safe_commands = self.safety_monitor.check_safety(
            joint_commands, current_state, sensor_data)

        # Validate commands
        validated_commands = self.validate_commands(safe_commands, current_state)

        return validated_commands

    def validate_commands(self, commands, current_state):
        """Validate control commands for safety and feasibility"""
        # Check for NaN or infinite values
        if np.any(np.isnan(commands['joint_torques'])) or np.any(np.isinf(commands['joint_torques'])):
            print("Warning: Control commands contain NaN or infinite values. Zeroing commands.")
            commands['joint_torques'] = np.zeros_like(commands['joint_torques'])

        # Check torque limits
        joint_limits = self.robot_model.get_joint_limits()
        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            max_torque = limits.get('max_torque', 100.0)
            commands['joint_torques'][i] = np.clip(
                commands['joint_torques'][i], -max_torque, max_torque)

        # Check for excessive velocity commands
        max_velocity = 2.0  # rad/s
        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            max_vel = limits.get('max_velocity', max_velocity)
            if abs(commands['joint_velocities'][i]) > max_vel:
                commands['joint_velocities'][i] = np.sign(commands['joint_velocities'][i]) * max_vel

        return commands

class StateEstimator:
    """Estimate robot state from sensor data"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def estimate_state(self, sensor_data):
        """Estimate robot state from sensor measurements"""
        # Get joint positions from encoder data
        joint_positions = sensor_data.get('joint_positions', np.zeros(self.robot_model.get_num_joints()))
        joint_velocities = sensor_data.get('joint_velocities', np.zeros_like(joint_positions))
        joint_accelerations = sensor_data.get('joint_accelerations', np.zeros_like(joint_positions))

        # Get IMU data for orientation
        imu_data = sensor_data.get('imu', {})
        orientation = imu_data.get('orientation', [0, 0, 0, 1])
        angular_velocity = imu_data.get('angular_velocity', [0, 0, 0])
        linear_acceleration = imu_data.get('linear_acceleration', [0, 0, 0])

        # Estimate base position and velocity using kinematic integration
        # This is simplified - in practice, would use more sophisticated estimation
        base_pose = self.estimate_base_pose(joint_positions, imu_data)

        return {
            'positions': joint_positions,
            'velocities': joint_velocities,
            'accelerations': joint_accelerations,
            'base_pose': base_pose,
            'orientation': orientation,
            'angular_velocity': angular_velocity,
            'linear_acceleration': linear_acceleration
        }

    def estimate_base_pose(self, joint_positions, imu_data):
        """Estimate base pose from joint positions and IMU"""
        # This would typically use forward kinematics with contact estimation
        # For now, return a simplified estimate
        return {
            'position': np.array([0, 0, 0]),
            'orientation': np.array(imu_data.get('orientation', [0, 0, 0, 1]))
        }

class SafetyMonitor:
    """Monitor safety conditions and limit commands"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.safety_thresholds = {
            'joint_position': 0.1,  # rad
            'joint_velocity': 0.1,  # rad/s
            'torque': 0.1,          # N*m
            'com_deviation': 0.1,   # m
            'balance_angle': 0.3    # rad
        }

    def check_safety(self, commands, state, sensor_data):
        """Check if commands are safe to execute"""
        safe_commands = commands.copy()

        # Check for excessive joint positions
        joint_limits = self.robot_model.get_joint_limits()
        current_positions = state['positions']

        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            # Check if commanded position is within limits
            if safe_commands['joint_positions'][i] < limits['min'] or \
               safe_commands['joint_positions'][i] > limits['max']:
                # Reduce command magnitude to stay within limits
                safe_commands['joint_positions'][i] = np.clip(
                    safe_commands['joint_positions'][i],
                    limits['min'], limits['max'])

        # Check for excessive torques
        max_torques = [limits.get('max_torque', 100.0) for _, limits in joint_limits.items()]
        safe_commands['joint_torques'] = np.clip(
            safe_commands['joint_torques'],
            -np.array(max_torques), np.array(max_torques))

        # Check for balance conditions
        if 'imu' in sensor_data:
            imu_orientation = sensor_data['imu'].get('orientation', [0, 0, 0, 1])
            from scipy.spatial.transform import Rotation as R
            rot = R.from_quat(imu_orientation)
            euler_angles = rot.as_euler('xyz')

            # If robot is tilting too much, reduce all commands
            if abs(euler_angles[0]) > self.safety_thresholds['balance_angle'] or \
               abs(euler_angles[1]) > self.safety_thresholds['balance_angle']:
                # Scale down all commands
                safe_commands['joint_torques'] *= 0.5
                safe_commands['joint_positions'] = state['positions'] + \
                    (safe_commands['joint_positions'] - state['positions']) * 0.5

        return safe_commands
```

## Feedback Control Systems

### PID Control Fundamentals

PID (Proportional-Integral-Derivative) control is fundamental to humanoid robot control:

```python
class PIDController:
    """PID controller for humanoid robot joints"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        # Internal state
        self.integral_error = 0.0
        self.previous_error = 0.0

        # Anti-windup limits
        self.integral_limit = 10.0
        self.output_limit = 100.0

    def update(self, desired_value, current_value, feedforward=0.0):
        """Update PID controller and return control output"""
        error = desired_value - current_value

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error,
                                    -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral_error

        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        d_term = self.kd * derivative

        # Total output
        output = p_term + i_term + d_term + feedforward

        # Apply output limits
        output = np.clip(output, -self.output_limit, self.output_limit)

        # Store for next iteration
        self.previous_error = error

        return output

    def reset(self):
        """Reset controller internal state"""
        self.integral_error = 0.0
        self.previous_error = 0.0

class JointPIDController:
    """PID controller for individual robot joints"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.joint_controllers = {}

        # Initialize PID controllers for each joint
        joint_names = robot_model.get_joint_names()
        joint_limits = robot_model.get_joint_limits()

        for joint_name in joint_names:
            limits = joint_limits[joint_name]

            # Set gains based on joint characteristics
            if 'hip' in joint_name or 'knee' in joint_name or 'ankle' in joint_name:
                # Leg joints - higher stiffness for balance
                kp = 500.0
                ki = 10.0
                kd = 50.0
            elif 'shoulder' in joint_name or 'elbow' in joint_name:
                # Arm joints - medium stiffness for manipulation
                kp = 300.0
                ki = 5.0
                kd = 30.0
            else:
                # Default gains
                kp = 200.0
                ki = 2.0
                kd = 20.0

            self.joint_controllers[joint_name] = PIDController(kp, ki, kd, dt=0.001)

    def compute_joint_commands(self, desired_positions, desired_velocities,
                             current_positions, current_velocities):
        """Compute joint commands using PID control"""
        commands = np.zeros(len(current_positions))

        for i, joint_name in enumerate(self.robot_model.get_joint_names()):
            # Get current and desired values
            current_pos = current_positions[i]
            current_vel = current_velocities[i]
            desired_pos = desired_positions[i]
            desired_vel = desired_velocities[i]

            # Position control
            pos_error = desired_pos - current_pos
            pos_cmd = self.joint_controllers[joint_name].update(
                desired_pos, current_pos, feedforward=0.0)

            # Velocity control (if needed)
            vel_error = desired_vel - current_vel
            vel_cmd = self.joint_controllers[joint_name].update(
                desired_vel, current_vel, feedforward=pos_cmd)

            commands[i] = vel_cmd

        return commands

class BalancePIDController:
    """PID controller specifically for balance control"""

    def __init__(self, com_height=0.8):
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)

        # PID gains for balance (tuned for inverted pendulum)
        self.com_pid_x = PIDController(kp=200.0, ki=10.0, kd=40.0, dt=0.01)
        self.com_pid_y = PIDController(kp=200.0, ki=10.0, kd=40.0, dt=0.01)

        # ZMP control gains
        self.zmp_pid_x = PIDController(kp=100.0, ki=5.0, kd=20.0, dt=0.01)
        self.zmp_pid_y = PIDController(kp=100.0, ki=5.0, kd=20.0, dt=0.01)

    def compute_balance_commands(self, current_com, current_com_vel,
                                desired_com, support_polygon_center):
        """Compute balance commands using PID control"""
        # Calculate errors
        com_error_x = desired_com[0] - current_com[0]
        com_error_y = desired_com[1] - current_com[1]

        # Calculate ZMP reference (desired ZMP should be near support polygon center)
        desired_zmp = support_polygon_center

        # Calculate current ZMP (approximated)
        current_zmp_x = current_com[0] - current_com_vel[0] / (self.omega**2)
        current_zmp_y = current_com[1] - current_com_vel[1] / (self.omega**2)

        zmp_error_x = desired_zmp[0] - current_zmp_x
        zmp_error_y = desired_zmp[1] - current_zmp_y

        # Compute balance corrections
        balance_cmd_x = (
            self.com_pid_x.update(desired_com[0], current_com[0]) +
            self.zmp_pid_x.update(desired_zmp[0], current_zmp_x)
        )

        balance_cmd_y = (
            self.com_pid_y.update(desired_com[1], current_com[1]) +
            self.zmp_pid_y.update(desired_zmp[1], current_zmp_y)
        )

        # Convert to joint commands using CoM Jacobian transpose
        # This is a simplified approach - in practice, would use more sophisticated methods
        balance_commands = np.array([balance_cmd_x, balance_cmd_y, 0, 0, 0, 0])  # Only x,y for balance

        return balance_commands
```

## Advanced Control Techniques

### Linear Quadratic Regulator (LQR)

LQR is an optimal control technique particularly useful for humanoid balance:

```python
class LQRController:
    """Linear Quadratic Regulator controller for humanoid balance"""

    def __init__(self, robot_model, com_height=0.8):
        self.robot_model = robot_model
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)

        # State: [com_x, com_y, com_z, com_dx, com_dy, com_dz, theta_x, theta_y, theta_z, omega_x, omega_y, omega_z]
        # Control: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        self.state_dim = 12
        self.control_dim = 6

        # Design LQR matrices
        self.Q = self.design_state_weight_matrix()
        self.R = self.design_control_weight_matrix()

        # Solve algebraic Riccati equation to get optimal gain matrix
        self.K = self.solve_lqr_gain()

    def design_state_weight_matrix(self):
        """Design state weighting matrix Q"""
        Q = np.zeros((self.state_dim, self.state_dim))

        # Weight for CoM position (x, y, z)
        Q[0, 0] = 100.0  # CoM x position error
        Q[1, 1] = 100.0  # CoM y position error
        Q[2, 2] = 10.0   # CoM z position error (less critical)

        # Weight for CoM velocity
        Q[3, 3] = 50.0   # CoM x velocity error
        Q[4, 4] = 50.0   # CoM y velocity error
        Q[5, 5] = 5.0    # CoM z velocity error

        # Weight for orientation
        Q[6, 6] = 200.0  # Roll error
        Q[7, 7] = 200.0  # Pitch error
        Q[8, 8] = 50.0   # Yaw error (less critical for balance)

        # Weight for angular velocity
        Q[9, 9] = 100.0  # Roll rate error
        Q[10, 10] = 100.0 # Pitch rate error
        Q[11, 11] = 25.0  # Yaw rate error

        return Q

    def design_control_weight_matrix(self):
        """Design control weighting matrix R"""
        R = np.zeros((self.control_dim, self.control_dim))

        # Weight for forces (prefer smaller forces)
        R[0, 0] = 0.1    # Force x
        R[1, 1] = 0.1    # Force y
        R[2, 2] = 0.05   # Force z (less penalty for vertical forces)

        # Weight for torques (prefer smaller torques)
        R[3, 3] = 0.05   # Torque x (roll)
        R[4, 4] = 0.05   # Torque y (pitch)
        R[5, 5] = 0.2    # Torque z (yaw - higher penalty to avoid spinning)

        return R

    def solve_lqr_gain(self):
        """Solve the LQR problem to find optimal gain matrix"""
        # Linearized system matrices for inverted pendulum model
        # State: [x, y, z, x_dot, y_dot, z_dot, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]

        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, self.control_dim))

        # Simplified linearized dynamics for inverted pendulum
        # dx/dt = x_dot
        # dy/dt = y_dot
        # dz/dt = z_dot
        # d^2x/dt^2 = omega^2 * (x - zmp_x)  (simplified inverted pendulum)
        # d^2y/dt^2 = omega^2 * (y - zmp_y)
        # d^2z/dt^2 = 0 (constant height assumption)

        # Fill A matrix (continuous time)
        # Position derivatives
        A[0, 3] = 1.0  # dx/dt = x_dot
        A[1, 4] = 1.0  # dy/dt = y_dot
        A[2, 5] = 1.0  # dz/dt = z_dot

        # Velocity derivatives (linearized inverted pendulum)
        A[3, 0] = self.omega**2  # d^2x/dt^2 = omega^2 * x
        A[4, 1] = self.omega**2  # d^2y/dt^2 = omega^2 * y
        A[3, 6] = -self.omega**2 * 0.1  # Coupling with roll (simplified)
        A[4, 7] = -self.omega**2 * 0.1  # Coupling with pitch (simplified)

        # Orientation derivatives (simplified)
        A[6, 9] = 1.0  # d(roll)/dt = roll_rate
        A[7, 10] = 1.0 # d(pitch)/dt = pitch_rate
        A[8, 11] = 1.0 # d(yaw)/dt = yaw_rate

        # Angular velocity derivatives (simplified - assuming direct torque control)
        A[9, 3] = 0.0  # For now, direct torque control
        A[10, 4] = 0.0
        A[11, 5] = 0.0

        # Fill B matrix (control input matrix)
        # Direct control of accelerations
        B[3, 0] = 1.0  # Force x affects x acceleration
        B[4, 1] = 1.0  # Force y affects y acceleration
        B[5, 2] = 1.0  # Force z affects z acceleration
        B[9, 3] = 1.0  # Torque x affects roll rate
        B[10, 4] = 1.0 # Torque y affects pitch rate
        B[11, 5] = 1.0 # Torque z affects yaw rate

        # Discretize the system
        dt = 0.01  # 100Hz control
        I = np.eye(self.state_dim)
        A_d = I + A * dt
        B_d = B * dt

        # Solve discrete-time LQR
        try:
            # Solve the discrete algebraic Riccati equation
            from scipy.linalg import solve_discrete_are
            P = solve_discrete_are(A_d.T, B_d.T, self.Q, self.R)

            # Calculate optimal gain matrix
            K = np.linalg.inv(B_d.T @ P @ B_d + self.R) @ B_d.T @ P @ A_d

            return K
        except:
            # If ARE solver fails, return a simple PD gain
            print("Warning: LQR Riccati equation solver failed. Using PD controller.")
            K_pd = np.zeros((self.control_dim, self.state_dim))
            K_pd[0, 0] = 100.0  # Position x
            K_pd[1, 1] = 100.0  # Position y
            K_pd[3, 6] = 50.0   # Roll
            K_pd[4, 7] = 50.0   # Pitch
            return K_pd

    def compute_control(self, state_error):
        """Compute LQR control output"""
        # State error should be a 12-dimensional vector
        if len(state_error) != self.state_dim:
            raise ValueError(f"State error must be {self.state_dim}-dimensional")

        # Apply LQR control law: u = -K * x
        control_output = -self.K @ state_error

        return control_output

    def get_state_vector(self, current_state, desired_state):
        """Get state error vector for LQR control"""
        # Current state vector [com_pos, com_vel, orientation, angular_vel]
        current_com_pos = current_state['com_position']
        current_com_vel = current_state['com_velocity']
        current_orientation = current_state['orientation']
        current_angular_vel = current_state['angular_velocity']

        # Desired state vector
        desired_com_pos = desired_state['com_position']
        desired_com_vel = desired_state.get('com_velocity', np.zeros(3))
        desired_orientation = desired_state['orientation']
        desired_angular_vel = desired_state.get('angular_velocity', np.zeros(3))

        # Calculate errors
        com_pos_error = desired_com_pos - current_com_pos
        com_vel_error = desired_com_vel - current_com_vel
        orientation_error = self.quaternion_error(current_orientation, desired_orientation)
        angular_vel_error = desired_angular_vel - current_angular_vel

        # Combine into state error vector
        state_error = np.concatenate([
            com_pos_error,
            com_vel_error,
            orientation_error,
            angular_vel_error
        ])

        return state_error

    def quaternion_error(self, q1, q2):
        """Calculate orientation error as rotation vector"""
        from scipy.spatial.transform import Rotation as R
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        relative_rotation = r2 * r1.inv()
        return relative_rotation.as_rotvec()

class LQRBalanceController:
    """LQR-based balance controller for humanoid robots"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.lqr_controller = LQRController(robot_model)

        # Support polygon tracking
        self.support_polygon = None
        self.desired_com_trajectory = None
        self.trajectory_idx = 0

    def compute_balance_control(self, current_state, desired_com=None):
        """Compute balance control using LQR"""
        if desired_com is None:
            # Default to keeping CoM over support polygon center
            if self.support_polygon is not None:
                support_center = np.mean(self.support_polygon, axis=0)
                desired_com = np.array([support_center[0], support_center[1], 0.8])
            else:
                desired_com = np.array([0, 0, 0.8])  # Default standing position

        # Create desired state
        desired_state = {
            'com_position': desired_com,
            'com_velocity': np.array([0, 0, 0]),  # Want zero velocity
            'orientation': np.array([0, 0, 0, 1]),  # Upright orientation
            'angular_velocity': np.array([0, 0, 0])  # No angular velocity
        }

        # Get state error
        state_error = self.lqr_controller.get_state_vector(current_state, desired_state)

        # Compute control
        control_output = self.lqr_controller.compute_control(state_error)

        # Convert force/torque commands to joint commands
        joint_commands = self.map_force_to_joints(control_output, current_state)

        return joint_commands

    def map_force_to_joints(self, force_torque, current_state):
        """Map force/torque commands to joint commands"""
        # This is a simplified mapping - in practice, would use full dynamics model
        joint_commands = np.zeros(len(current_state['positions']))

        # Distribute forces to joints based on their role in balance
        joint_names = self.robot_model.get_joint_names()

        for i, joint_name in enumerate(joint_names):
            if 'hip' in joint_name or 'ankle' in joint_name:
                # Hip and ankle joints are critical for balance
                if 'pitch' in joint_name:
                    # Pitch joints respond to forward/backward forces
                    joint_commands[i] = force_torque[0] * 0.1  # Map x-force to pitch
                elif 'roll' in joint_name:
                    # Roll joints respond to lateral forces
                    joint_commands[i] = force_torque[1] * 0.1  # Map y-force to roll
                elif 'yaw' in joint_name:
                    # Yaw joints respond to yaw torques
                    joint_commands[i] = force_torque[5] * 0.05  # Map z-torque to yaw
            elif 'shoulder' in joint_name:
                # Shoulder joints for upper body balance
                if 'pitch' in joint_name:
                    joint_commands[i] = force_torque[0] * 0.02  # Smaller contribution
                elif 'roll' in joint_name:
                    joint_commands[i] = force_torque[1] * 0.02  # Smaller contribution

        return joint_commands
```

### Model Predictive Control (MPC)

MPC is particularly useful for humanoid robots because it can handle constraints explicitly:

```python
class ModelPredictiveController:
    """Model Predictive Controller for humanoid balance and motion"""

    def __init__(self, robot_model, prediction_horizon=20, dt=0.01):
        self.robot_model = robot_model
        self.prediction_horizon = prediction_horizon
        self.dt = dt

        # State: [com_x, com_y, com_z, com_dx, com_dy, com_dz]
        # Control: [zmp_x, zmp_y] (virtual actuator approach)
        self.state_dim = 6
        self.control_dim = 2

        # Prediction model (simplified inverted pendulum)
        self.A, self.B = self.create_prediction_model()

        # MPC weights
        self.Q = np.diag([100, 100, 10, 10, 10, 5])  # State weights
        self.R = np.diag([0.1, 0.1])                 # Control weights
        self.P = np.diag([500, 500, 50, 50, 50, 25])  # Terminal weights

        # Constraints
        self.zmp_min = np.array([-0.2, -0.1])  # m
        self.zmp_max = np.array([0.2, 0.1])   # m

        # Support polygon constraints (would be updated based on foot positions)
        self.support_polygon_min = np.array([-0.15, -0.08])
        self.support_polygon_max = np.array([0.15, 0.08])

    def create_prediction_model(self):
        """Create linear prediction model matrices A and B"""
        # Inverted pendulum model: x_ddot = omega^2 * (x - zmp_x)
        # State: [x, y, z, x_dot, y_dot, z_dot]
        omega = np.sqrt(9.81 / 0.8)  # Assuming 0.8m CoM height

        A = np.array([
            [1, 0, 0, self.dt, 0, 0],                    # x += x_dot * dt
            [0, 1, 0, 0, self.dt, 0],                    # y += y_dot * dt
            [0, 0, 1, 0, 0, self.dt],                    # z += z_dot * dt
            [0, 0, 0, 1, 0, 0],                          # x_dot unchanged (simplified)
            [0, 0, 0, 0, 1, 0],                          # y_dot unchanged (simplified)
            [0, 0, 0, 0, 0, 1]                           # z_dot unchanged (simplified)
        ])

        # For the inverted pendulum dynamics, we need to add the acceleration terms
        # d^2x/dt^2 = omega^2 * (x - zmp_x) => x_dot += omega^2 * (x - zmp_x) * dt
        # This creates coupling between position and velocity
        A[3, 0] = omega**2 * self.dt**2  # x affects x_dot
        A[4, 1] = omega**2 * self.dt**2  # y affects y_dot

        # Control input matrix (how ZMP affects CoM)
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-omega**2 * self.dt**2, 0],  # ZMP_x affects x_dot
            [0, -omega**2 * self.dt**2],  # ZMP_y affects y_dot
            [0, 0]
        ])

        return A, B

    def solve_mpc(self, current_state, reference_trajectory, support_polygon=None):
        """
        Solve MPC optimization problem

        Args:
            current_state: Current state vector [x, y, z, x_dot, y_dot, z_dot]
            reference_trajectory: Reference trajectory over prediction horizon
            support_polygon: Current support polygon constraints

        Returns:
            Optimal control sequence (first element is applied to robot)
        """
        try:
            import cvxpy as cp

            # Define optimization variables
            X = cp.Variable((self.state_dim, self.prediction_horizon + 1))  # State trajectory
            U = cp.Variable((self.control_dim, self.prediction_horizon))   # Control trajectory

            # Cost function
            cost = 0

            # Running cost (tracking error and control effort)
            for k in range(self.prediction_horizon):
                state_error = X[:, k] - reference_trajectory[k]
                control_effort = U[:, k]

                cost += cp.quad_form(state_error, self.Q)
                cost += cp.quad_form(control_effort, self.R)

            # Terminal cost
            terminal_error = X[:, self.prediction_horizon] - reference_trajectory[-1]
            cost += cp.quad_form(terminal_error, self.P)

            # Constraints
            constraints = []

            # Initial state constraint
            constraints.append(X[:, 0] == current_state)

            # System dynamics constraints
            for k in range(self.prediction_horizon):
                constraints.append(X[:, k+1] == self.A @ X[:, k] + self.B @ U[:, k])

            # ZMP constraints (must be within support polygon)
            if support_polygon is not None:
                for k in range(self.prediction_horizon):
                    # Simplified support polygon constraint as bounding box
                    constraints.append(U[0, k] >= support_polygon[0, 0])  # x_min
                    constraints.append(U[0, k] <= support_polygon[1, 0])  # x_max
                    constraints.append(U[1, k] >= support_polygon[0, 1])  # y_min
                    constraints.append(U[1, k] <= support_polygon[1, 1])  # y_max
            else:
                # Default support polygon constraints
                for k in range(self.prediction_horizon):
                    constraints.append(U[0, k] >= self.support_polygon_min[0])
                    constraints.append(U[0, k] <= self.support_polygon_max[0])
                    constraints.append(U[1, k] >= self.support_polygon_min[1])
                    constraints.append(U[1, k] <= self.support_polygon_max[1])

            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status == cp.OPTIMAL:
                # Return first control in sequence
                return U.value[:, 0]
            else:
                print(f"MPC optimization failed with status: {problem.status}")
                # Return zero control if optimization fails
                return np.zeros(self.control_dim)

        except ImportError:
            # If CVXPY is not available, use simplified approach
            print("CVXPY not available, using simplified MPC approach")
            return self.simplified_mpc_approach(current_state, reference_trajectory)

    def simplified_mpc_approach(self, current_state, reference_trajectory):
        """Simplified MPC approach when CVXPY is not available"""
        # Use a simple feedback control approach that considers future states
        current_com = current_state[:2]  # x, y
        current_com_vel = current_state[3:4]  # x_dot, y_dot

        # Calculate error to first reference point
        ref_com = reference_trajectory[0][:2]
        ref_com_vel = reference_trajectory[0][3:4]

        # Simple feedback control
        kp = 100.0
        kd = 20.0

        com_error = ref_com - current_com
        vel_error = ref_com_vel - current_com_vel

        zmp_command = kp * com_error + kd * vel_error

        # Apply constraints
        zmp_command[0] = np.clip(zmp_command[0],
                                self.support_polygon_min[0], self.support_polygon_max[0])
        zmp_command[1] = np.clip(zmp_command[1],
                                self.support_polygon_min[1], self.support_polygon_max[1])

        return zmp_command

    def get_reference_trajectory(self, current_state, behavior='standing'):
        """Generate reference trajectory based on behavior"""
        reference_trajectory = []

        if behavior == 'standing':
            # Keep CoM at neutral position
            neutral_com = np.array([0, 0, 0.8])
            neutral_com_vel = np.array([0, 0, 0])

            for k in range(self.prediction_horizon + 1):
                ref_state = np.zeros(self.state_dim)
                ref_state[:3] = neutral_com
                ref_state[3:] = neutral_com_vel
                reference_trajectory.append(ref_state)

        elif behavior == 'walking':
            # Generate walking pattern with CoM tracking
            step_length = 0.3
            step_duration = 0.8

            for k in range(self.prediction_horizon + 1):
                time_step = k * self.dt

                # Calculate expected CoM position during walking
                step_num = int(time_step / step_duration)
                time_in_step = time_step % step_duration

                # CoM moves forward with slight lateral sway
                com_x = step_num * step_length + (time_in_step / step_duration) * step_length
                com_y = 0.05 * np.sin(2 * np.pi * time_step / step_duration)  # Slight sway
                com_z = 0.8  # Maintain height

                ref_state = np.zeros(self.state_dim)
                ref_state[:3] = [com_x, com_y, com_z]

                # Calculate expected velocities
                if k > 0:
                    prev_ref = reference_trajectory[k-1]
                    ref_state[3:] = (ref_state[:3] - prev_ref[:3]) / self.dt
                else:
                    ref_state[3:] = [step_length / step_duration, 0, 0]  # Forward velocity

                reference_trajectory.append(ref_state)

        return np.array(reference_trajectory)

    def compute_mpc_control(self, current_state, behavior='standing', support_polygon=None):
        """Compute MPC control for given behavior"""
        # Get reference trajectory
        reference_trajectory = self.get_reference_trajectory(current_state, behavior)

        # Solve MPC problem
        zmp_command = self.solve_mpc(current_state, reference_trajectory, support_polygon)

        # Convert ZMP command to joint commands
        joint_commands = self.zmp_to_joints(zmp_command, current_state)

        return joint_commands

    def zmp_to_joints(self, zmp_command, current_state):
        """Convert ZMP commands to joint commands"""
        # This would use inverse kinematics or whole-body control
        # to determine how to move joints to achieve desired ZMP

        joint_commands = np.zeros(len(current_state['positions']))

        # Simplified approach: adjust ankle and hip joints to move ZMP
        joint_names = self.robot_model.get_joint_names()

        for i, joint_name in enumerate(joint_names):
            if 'ankle' in joint_name:
                # Ankle joints are primary for ZMP control
                if 'pitch' in joint_name:
                    # Forward/backward ZMP affects ankle pitch
                    joint_commands[i] = zmp_command[0] * 50  # Gain for ankle control
                elif 'roll' in joint_name:
                    # Lateral ZMP affects ankle roll
                    joint_commands[i] = zmp_command[1] * 50  # Gain for ankle control
            elif 'hip' in joint_name:
                # Hip joints as secondary for ZMP control
                if 'pitch' in joint_name:
                    joint_commands[i] = zmp_command[0] * 20  # Smaller gain
                elif 'roll' in joint_name:
                    joint_commands[i] = zmp_command[1] * 20  # Smaller gain
            elif 'torso' in joint_name:
                # Torso joints for additional balance
                if 'pitch' in joint_name:
                    joint_commands[i] = -zmp_command[0] * 10  # Counteract forward/backward
                elif 'roll' in joint_name:
                    joint_commands[i] = -zmp_command[1] * 10  # Counteract lateral

        return joint_commands

class MPCBalanceController:
    """MPC-based balance controller integrated with humanoid control architecture"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.mpc_controller = ModelPredictiveController(robot_model)

        # Support polygon estimator
        self.support_polygon_estimator = SupportPolygonEstimator(robot_model)

    def compute_control(self, current_state, behavior='standing'):
        """Compute MPC-based balance control"""
        # Estimate current support polygon
        support_polygon = self.support_polygon_estimator.estimate(
            current_state['positions'], current_state.get('contact_data'))

        # Get current CoM state
        current_com_pos = self.robot_model.calculate_com_position(current_state['positions'])
        current_com_vel = self.robot_model.calculate_com_velocity(
            current_state['positions'], current_state['velocities'])

        # Pack into state vector for MPC
        mpc_state = np.zeros(6)
        mpc_state[:3] = current_com_pos
        mpc_state[3:] = current_com_vel

        # Compute MPC control
        joint_commands = self.mpc_controller.compute_mpc_control(
            mpc_state, behavior, support_polygon)

        return joint_commands

class SupportPolygonEstimator:
    """Estimate support polygon based on contact information"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def estimate(self, joint_positions, contact_data=None):
        """Estimate support polygon from contact information"""
        if contact_data is not None and 'contacts' in contact_data:
            # Use contact sensor data to determine support polygon
            support_points = []
            for contact in contact_data['contacts']:
                if contact['in_contact']:
                    support_points.append(contact['position'][:2])  # x, y only

            if len(support_points) >= 3:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(support_points)
                    return np.array(support_points)[hull.vertices]
                except:
                    pass

        # Fallback: estimate from foot positions
        left_foot_pos = self.robot_model.forward_kinematics_for_link(
            joint_positions, 'left_foot')['position']
        right_foot_pos = self.robot_model.forward_kinematics_for_link(
            joint_positions, 'right_foot')['position']

        # Create conservative support polygon
        support_polygon = np.array([
            [min(left_foot_pos[0], right_foot_pos[0]) - 0.1,  # Conservative bounds
             min(left_foot_pos[1], right_foot_pos[1]) - 0.05],
            [max(left_foot_pos[0], right_foot_pos[0]) + 0.1,
             max(left_foot_pos[1], right_foot_pos[1]) + 0.05]
        ])

        return support_polygon
```

## Integration with Control Architecture

### Unified Controller with Multiple Control Modes

```python
class MultiModalController:
    """Controller that can switch between different control modes"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

        # Initialize different control modes
        self.pid_controller = JointPIDController(robot_model)
        self.lqr_controller = LQRBalanceController(robot_model)
        self.mpc_controller = MPCBalanceController(robot_model)
        self.task_controller = TaskLevelController(robot_model)

        # Current control mode
        self.current_mode = 'pid'  # 'pid', 'lqr', 'mpc', 'task'

        # Mode switching parameters
        self.mode_switch_thresholds = {
            'balance_error': 0.1,      # Switch to balance mode if CoM error > 10cm
            'disturbance_magnitude': 5.0,  # Switch to robust mode if disturbance > 5N
            'contact_changes': 2       # Switch if more than 2 contacts change
        }

    def compute_control(self, current_state, reference, sensor_data=None):
        """Compute control based on current mode and conditions"""

        # Determine if mode switch is needed
        new_mode = self.determine_control_mode(current_state, sensor_data)
        if new_mode != self.current_mode:
            self.switch_control_mode(new_mode)

        # Compute control based on current mode
        if self.current_mode == 'pid':
            return self.compute_pid_control(current_state, reference)
        elif self.current_mode == 'lqr':
            return self.compute_lqr_control(current_state, reference)
        elif self.current_mode == 'mpc':
            return self.compute_mpc_control(current_state, reference, sensor_data)
        elif self.current_mode == 'task':
            return self.compute_task_control(current_state, reference)
        else:
            # Default to PID if mode is unknown
            return self.compute_pid_control(current_state, reference)

    def determine_control_mode(self, current_state, sensor_data):
        """Determine appropriate control mode based on current conditions"""

        # Check balance stability
        com_pos = self.robot_model.calculate_com_position(current_state['positions'])
        support_center = self.estimate_support_center(sensor_data)

        balance_error = np.linalg.norm(com_pos[:2] - support_center)

        # Check for external disturbances
        disturbance_magnitude = self.estimate_disturbance_magnitude(sensor_data)

        # Check contact stability
        contact_changes = self.count_contact_changes(sensor_data)

        # Decision logic
        if balance_error > self.mode_switch_thresholds['balance_error']:
            # High balance error - use balance-oriented controller
            if disturbance_magnitude > self.mode_switch_thresholds['disturbance_magnitude']:
                return 'mpc'  # Use MPC for disturbance rejection
            else:
                return 'lqr'  # Use LQR for balance
        elif contact_changes > self.mode_switch_thresholds['contact_changes']:
            # Many contact changes - use MPC for constraint handling
            return 'mpc'
        else:
            # Stable conditions - use standard PID
            return 'pid'

    def estimate_support_center(self, sensor_data):
        """Estimate support polygon center from sensor data"""
        if sensor_data and 'contact_data' in sensor_data:
            contacts = sensor_data['contact_data']
            in_contact_points = [c['position'][:2] for c in contacts if c.get('in_contact', False)]
            if in_contact_points:
                return np.mean(in_contact_points, axis=0)

        # Fallback: estimate from foot positions
        return np.array([0.0, 0.0])

    def estimate_disturbance_magnitude(self, sensor_data):
        """Estimate magnitude of external disturbances"""
        if sensor_data and 'wrench' in sensor_data:
            external_wrench = sensor_data['wrench']
            return np.linalg.norm(external_wrench[:3])  # Force magnitude
        return 0.0

    def count_contact_changes(self, sensor_data):
        """Count number of contact changes"""
        if sensor_data and 'contact_data' in sensor_data:
            contacts = sensor_data['contact_data']
            return sum(1 for c in contacts if c.get('changed', False))
        return 0

    def switch_control_mode(self, new_mode):
        """Switch to new control mode"""
        print(f"Switching control mode from {self.current_mode} to {new_mode}")

        # Perform any necessary mode transition operations
        if self.current_mode == 'mpc' and new_mode != 'mpc':
            # Reset MPC internal state if leaving MPC mode
            pass

        self.current_mode = new_mode

    def compute_pid_control(self, current_state, reference):
        """Compute PID-based control"""
        desired_positions = reference.get('joint_positions', current_state['positions'])
        desired_velocities = reference.get('joint_velocities', np.zeros_like(current_state['positions']))

        return self.pid_controller.compute_joint_commands(
            desired_positions, desired_velocities,
            current_state['positions'], current_state['velocities'])

    def compute_lqr_control(self, current_state, reference):
        """Compute LQR-based balance control"""
        # Create desired state for LQR
        desired_state = {
            'com_position': reference.get('com_position', np.array([0, 0, 0.8])),
            'orientation': reference.get('orientation', np.array([0, 0, 0, 1]))
        }

        return self.lqr_controller.compute_balance_control(current_state, desired_state)

    def compute_mpc_control(self, current_state, reference, sensor_data):
        """Compute MPC-based control"""
        behavior = reference.get('behavior', 'standing')
        return self.mpc_controller.compute_control(current_state, behavior)

    def compute_task_control(self, current_state, reference):
        """Compute task-based control"""
        return self.task_controller.compute_commands(current_state, reference)
```

## Performance Considerations

### Real-Time Control Optimization

```python
class RealTimeController:
    """Controller optimized for real-time performance"""

    def __init__(self, robot_model, control_frequency=1000):
        self.robot_model = robot_model
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # Pre-allocate arrays to avoid memory allocation during control
        self.pre_allocated = {
            'joint_positions': np.zeros(robot_model.get_num_joints()),
            'joint_velocities': np.zeros(robot_model.get_num_joints()),
            'joint_commands': np.zeros(robot_model.get_num_joints()),
            'mass_matrix': np.zeros((robot_model.get_num_joints(), robot_model.get_num_joints())),
            'coriolis': np.zeros(robot_model.get_num_joints()),
            'gravity': np.zeros(robot_model.get_num_joints()),
            'jacobian': np.zeros((6, robot_model.get_num_joints())),
            'task_error': np.zeros(6)
        }

        # Control timing
        self.last_control_time = time.time()
        self.control_period = 1.0 / control_frequency

        # Performance monitoring
        self.control_timing = {
            'min_time': float('inf'),
            'max_time': 0,
            'avg_time': 0,
            'count': 0
        }

    def compute_control_real_time(self, current_state, reference):
        """Compute control with real-time performance optimizations"""
        start_time = time.time()

        # Validate inputs
        if len(current_state['positions']) != len(self.pre_allocated['joint_positions']):
            raise ValueError("State dimension mismatch")

        # Copy inputs to pre-allocated arrays (avoids memory allocation)
        np.copyto(self.pre_allocated['joint_positions'], current_state['positions'])
        np.copyto(self.pre_allocated['joint_velocities'], current_state['velocities'])

        # Compute control using optimized methods
        commands = self.optimized_control_computation(
            self.pre_allocated['joint_positions'],
            self.pre_allocated['joint_velocities'],
            reference)

        # Store timing information
        elapsed_time = time.time() - start_time
        self.update_timing_statistics(elapsed_time)

        # Check if control loop is meeting timing requirements
        if elapsed_time > self.control_period * 0.8:  # 80% of period
            print(f"Warning: Control computation taking {elapsed_time*1000:.1f}ms, "
                  f"period is {self.control_period*1000:.1f}ms")

        return commands

    def optimized_control_computation(self, joint_positions, joint_velocities, reference):
        """Optimized control computation using pre-allocated arrays"""

        # Get desired positions and velocities
        desired_positions = reference.get('joint_positions', joint_positions)
        desired_velocities = reference.get('joint_velocities', np.zeros_like(joint_positions))

        # Use pre-allocated array for commands
        commands = self.pre_allocated['joint_commands']

        # Compute simple PD control (fastest option)
        kp = 100.0
        kd = 10.0

        pos_error = desired_positions - joint_positions
        vel_error = desired_velocities - joint_velocities

        commands[:] = kp * pos_error + kd * vel_error

        # Add gravity compensation if needed (only for static positions)
        if np.linalg.norm(desired_velocities) < 0.01:  # Nearly static
            gravity_comp = self.robot_model.gravity_compensation(joint_positions)
            commands += gravity_comp

        return commands

    def update_timing_statistics(self, elapsed_time):
        """Update control timing statistics"""
        self.control_timing['min_time'] = min(self.control_timing['min_time'], elapsed_time)
        self.control_timing['max_time'] = max(self.control_timing['max_time'], elapsed_time)

        # Update average using exponential moving average
        if self.control_timing['count'] == 0:
            self.control_timing['avg_time'] = elapsed_time
        else:
            # Use alpha = 0.01 for slow adaptation
            alpha = 0.01
            self.control_timing['avg_time'] = (
                alpha * elapsed_time + (1 - alpha) * self.control_timing['avg_time'])

        self.control_timing['count'] += 1

    def get_performance_report(self):
        """Get performance timing report"""
        return {
            'frequency_actual': 1.0 / self.control_timing['avg_time'] if self.control_timing['avg_time'] > 0 else 0,
            'min_time_ms': self.control_timing['min_time'] * 1000,
            'max_time_ms': self.control_timing['max_time'] * 1000,
            'avg_time_ms': self.control_timing['avg_time'] * 1000,
            'period_ms': self.control_period * 1000,
            'timing_margin': (self.control_period - self.control_timing['max_time']) * 1000
        }

    def adaptive_control_frequency(self, performance_report):
        """Adjust control frequency based on performance"""
        # If timing margin is too small, reduce frequency
        if performance_report['timing_margin'] < 1.0:  # Less than 1ms margin
            new_freq = max(100, int(self.control_frequency * 0.9))  # Reduce by 10%, min 100Hz
            print(f"Reducing control frequency to {new_freq}Hz due to timing constraints")
            return new_freq
        elif performance_report['timing_margin'] > 5.0:  # More than 5ms margin
            new_freq = min(2000, int(self.control_frequency * 1.1))  # Increase by 10%, max 2000Hz
            print(f"Increasing control frequency to {new_freq}Hz due to excess capacity")
            return new_freq
        else:
            return self.control_frequency  # Keep current frequency
```

## Summary

Control architecture for humanoid robots requires a sophisticated multi-layered approach that can handle the complex requirements of balance, manipulation, and locomotion simultaneously. The key components include:

1. **Hierarchical Control Structure**: High-level planning, task-level coordination, and low-level joint control working together.

2. **Feedback Control Systems**: PID controllers for basic tracking, with specialized gains for different joint types and functions.

3. **Advanced Control Techniques**: LQR for optimal balance control and MPC for constraint handling and predictive control.

4. **Sensor Integration**: Proper handling of camera, LiDAR, and IMU data for state estimation and control feedback.

5. **Performance Optimization**: Real-time considerations including pre-allocation, caching, and efficient algorithms.

The success of humanoid control systems depends on properly balancing these different control objectives while maintaining system stability and responsiveness. Modern approaches often combine multiple control techniques and adaptively switch between them based on current conditions and task requirements.