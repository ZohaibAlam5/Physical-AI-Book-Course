---
title: Chapter 7 - Whole-Body Control Integration
description: Integrating multiple control systems for coordinated humanoid robot behavior
sidebar_position: 7
---

# Chapter 7: Whole-Body Control Integration

## Learning Objectives

After completing this chapter, you should be able to:
- Integrate multiple control systems for coordinated humanoid behavior
- Implement priority-based task control with null-space projection
- Coordinate balance, manipulation, and locomotion simultaneously
- Handle control conflicts and constraint resolution
- Validate whole-body control system performance

## Introduction

Whole-body control integration is the art of coordinating multiple control objectives simultaneously in a humanoid robot system. Unlike single-task robots, humanoid robots must manage balance, manipulation, locomotion, and other objectives all at once. This chapter explores the sophisticated techniques required to integrate these competing control demands into a unified, stable control system.

## Control Integration Architecture

### Hierarchical Task Integration

The challenge in whole-body control is managing competing objectives with different priorities:

```
┌─────────────────────────────────────────────────────────────────┐
│                    High-Level Planner                           │
│              (Trajectories, Behaviors, Goals)                  │
├─────────────────────────────────────────────────────────────────┤
│                  Task Coordinator                               │
│        (Balance vs Manipulation vs Locomotion)                 │
├─────────────────────────────────────────────────────────────────┤
│                 Priority Resolver                               │
│      (Null-space projections, Constraint handling)             │
├─────────────────────────────────────────────────────────────────┤
│                 Joint-Level Control                             │
│        (Individual joint control, Torque generation)           │
├─────────────────────────────────────────────────────────────────┤
│                Hardware Interface                               │
│          (Motor drivers, encoders, actuators)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Control Integration Framework

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from typing import Dict, List, Tuple, Optional

class WholeBodyControlIntegrator:
    """Integrates multiple control systems for coordinated humanoid behavior"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.n_joints = robot_model.get_num_joints()

        # Initialize individual controllers
        self.balance_controller = BalanceController(robot_model)
        self.manipulation_controller = ManipulationController(robot_model)
        self.locomotion_controller = LocomotionController(robot_model)
        self.posture_controller = PostureController(robot_model)

        # Control priority hierarchy
        self.priority_levels = {
            'safety': 0,
            'balance': 1,
            'locomotion': 2,
            'manipulation': 3,
            'posture': 4,
            'energy': 5
        }

        # Null space projectors cache
        self.null_space_cache = {}

        # Timing control
        self.control_frequency = 1000  # Hz
        self.dt = 1.0 / self.control_frequency

        # Pre-allocated arrays for efficiency
        self.temp_arrays = {
            'joint_commands': np.zeros(self.n_joints),
            'null_projector': np.eye(self.n_joints),
            'task_jacobian': np.zeros((6, self.n_joints)),
            'task_error': np.zeros(6),
            'task_force': np.zeros(6)
        }

    def integrate_controls(self, current_state, high_level_goals):
        """
        Integrate multiple control objectives into unified joint commands

        Args:
            current_state: Current robot state (positions, velocities, accelerations)
            high_level_goals: High-level behavioral goals

        Returns:
            Joint commands for robot control
        """
        # Parse high-level goals
        behavior = high_level_goals.get('behavior', 'stand')
        target_pose = high_level_goals.get('target_pose')
        manipulation_task = high_level_goals.get('manipulation_task')
        locomotion_target = high_level_goals.get('locomotion_target')

        # Initialize total commands
        total_commands = np.zeros(self.n_joints)

        # Start with highest priority: Safety
        safety_commands = self.compute_safety_control(current_state)
        if np.any(safety_commands != 0):
            return safety_commands  # Emergency stop if safety violation

        # Next priority: Balance control (essential for humanoid stability)
        balance_commands = self.balance_controller.compute_balance_control(
            current_state, high_level_goals.get('balance_requirements'))
        total_commands += balance_commands

        # Compute null space of balance control
        balance_jacobian = self.robot_model.com_jacobian(current_state['positions'])
        balance_null_space = self.compute_null_space_projection(balance_jacobian[:2, :])

        # Apply locomotion in balance null space
        if locomotion_target:
            locomotion_commands = self.locomotion_controller.compute_locomotion_control(
                current_state, locomotion_target)
            # Project locomotion commands into balance null space
            locomotion_in_null = balance_null_space @ locomotion_commands
            total_commands += locomotion_in_null

        # Apply manipulation in balance+locomotion null space
        if manipulation_task:
            manipulation_commands = self.manipulation_controller.compute_manipulation_control(
                current_state, manipulation_task)

            # Calculate combined Jacobian for balance + locomotion
            combined_jac = np.vstack([
                balance_jacobian[:2, :],  # Balance Jacobian (x,y only)
                self.get_locomotion_jacobian(current_state)  # Locomotion Jacobian
            ])

            # Compute null space of combined high-priority tasks
            combined_null_space = self.compute_null_space_projection(combined_jac)

            # Apply manipulation in combined null space
            manipulation_in_null = combined_null_space @ manipulation_commands
            total_commands += manipulation_in_null

        # Apply posture control in lowest priority null space
        posture_commands = self.posture_controller.compute_posture_control(
            current_state, high_level_goals.get('posture_preferences', {}))

        # If there are multiple high-priority tasks, use their combined null space
        if locomotion_target and manipulation_task:
            combined_jac = np.vstack([
                balance_jacobian[:2, :],
                self.get_locomotion_jacobian(current_state),
                self.get_manipulation_jacobian(current_state, manipulation_task)
            ])
            posture_null_space = self.compute_null_space_projection(combined_jac)
        elif locomotion_target:
            combined_jac = np.vstack([
                balance_jacobian[:2, :],
                self.get_locomotion_jacobian(current_state)
            ])
            posture_null_space = self.compute_null_space_projection(combined_jac)
        elif manipulation_task:
            combined_jac = np.vstack([
                balance_jacobian[:2, :],
                self.get_manipulation_jacobian(current_state, manipulation_task)
            ])
            posture_null_space = self.compute_null_space_projection(combined_jac)
        else:
            posture_null_space = balance_null_space  # Only balance constraint

        posture_in_null = posture_null_space @ posture_commands
        total_commands += posture_in_null

        # Apply joint limits and constraints
        total_commands = self.apply_joint_constraints(total_commands, current_state)

        return total_commands

    def compute_safety_control(self, current_state):
        """Compute safety-related control commands"""
        # Check for safety violations
        joint_positions = current_state['positions']
        joint_velocities = current_state['velocities']

        safety_commands = np.zeros(self.n_joints)

        # Check joint limits
        joint_limits = self.robot_model.get_joint_limits()
        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            current_pos = joint_positions[i]
            current_vel = joint_velocities[i]

            # Check for approaching limits
            if current_pos < limits['min'] + 0.05:  # 5% from limit
                # Apply repulsive force to move away from limit
                safety_commands[i] = 100.0  # Positive torque to move away from min limit
            elif current_pos > limits['max'] - 0.05:  # 5% from limit
                # Apply repulsive force to move away from limit
                safety_commands[i] = -100.0  # Negative torque to move away from max limit

            # Check for excessive velocities
            max_vel = limits.get('max_velocity', 2.0)
            if abs(current_vel) > max_vel * 0.9:  # 90% of max velocity
                # Apply braking force
                brake_force = -np.sign(current_vel) * 50.0
                safety_commands[i] += brake_force

        # Check for balance safety (if CoM is far from support polygon)
        com_pos = self.robot_model.calculate_com_position(joint_positions)
        support_polygon = self.estimate_support_polygon(current_state)

        if not self.is_com_stable(com_pos[:2], support_polygon):
            # Emergency balance recovery
            balance_recovery = self.balance_controller.compute_emergency_balance_control(
                current_state, com_pos, support_polygon)
            return balance_recovery

        return safety_commands

    def compute_null_space_projection(self, jacobian):
        """Compute null space projection matrix for a given task Jacobian"""
        try:
            # Method 1: Using SVD (more numerically stable)
            U, s, Vt = np.linalg.svd(jacobian, full_matrices=False)

            # Calculate rank (number of significant singular values)
            rank = np.sum(s > 1e-6)

            if rank == 0:
                # If all singular values are zero, return identity
                return np.eye(jacobian.shape[1])

            # Null space is spanned by the right singular vectors corresponding to zero singular values
            # N = I - V_r * V_r^T where V_r contains the first r columns of V
            Vr = Vt[:rank, :].T  # First r columns of V (as rows of Vt transposed)
            null_projector = np.eye(jacobian.shape[1]) - Vr @ Vr.T

            return null_projector

        except np.linalg.LinAlgError:
            # If SVD fails, use pseudo-inverse method
            try:
                j_pinv = np.linalg.pinv(jacobian)
                return np.eye(jacobian.shape[1]) - j_pinv @ jacobian
            except:
                # If all methods fail, return identity matrix (no null space projection)
                return np.eye(jacobian.shape[1])

    def get_locomotion_jacobian(self, current_state):
        """Get Jacobian for locomotion tasks (simplified)"""
        # For locomotion, we might consider feet positions and CoM motion
        # This is a simplified example - in practice, would be more complex
        left_foot_jac = self.robot_model.jacobian_for_link(
            current_state['positions'], 'left_foot')
        right_foot_jac = self.robot_model.jacobian_for_link(
            current_state['positions'], 'right_foot')

        # Combine foot Jacobians for locomotion control
        # Using only position part (first 3 rows) for foot placement
        combined_jac = np.vstack([
            left_foot_jac[:3, :],   # Left foot position
            right_foot_jac[:3, :]   # Right foot position
        ])

        return combined_jac

    def get_manipulation_jacobian(self, current_state, manipulation_task):
        """Get Jacobian for manipulation tasks"""
        if 'end_effector' in manipulation_task:
            ee_link = manipulation_task['end_effector']
            ee_jac = self.robot_model.jacobian_for_link(
                current_state['positions'], ee_link)

            # Use position and orientation (first 6 rows) for full pose control
            return ee_jac
        else:
            # Default to right hand if no specific end effector specified
            ee_jac = self.robot_model.jacobian_for_link(
                current_state['positions'], 'right_hand')
            return ee_jac

    def apply_joint_constraints(self, commands, current_state):
        """Apply joint limits and other constraints to control commands"""
        joint_positions = current_state['positions']
        joint_limits = self.robot_model.get_joint_limits()

        constrained_commands = commands.copy()

        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            # Calculate what the next position would be with this command
            # This is a simplified approach - in practice, would consider dynamics
            next_position = joint_positions[i] + commands[i] * self.dt * 0.1  # Small step for safety

            # Apply soft limits to prevent exceeding joint bounds
            if next_position < limits['min']:
                # Reduce command magnitude to prevent limit violation
                max_change = (limits['min'] - joint_positions[i]) / (self.dt * 0.1)
                constrained_commands[i] = min(commands[i], max_change)
            elif next_position > limits['max']:
                # Reduce command magnitude to prevent limit violation
                max_change = (limits['max'] - joint_positions[i]) / (self.dt * 0.1)
                constrained_commands[i] = max(commands[i], max_change)

            # Apply velocity limits
            max_velocity = limits.get('max_velocity', 2.0)
            max_change_per_step = max_velocity * self.dt
            constrained_commands[i] = np.clip(
                constrained_commands[i],
                -max_change_per_step, max_change_per_step)

        return constrained_commands

    def is_com_stable(self, com_xy, support_polygon):
        """Check if center of mass is stable within support polygon"""
        if len(support_polygon) < 3:
            return False  # Need at least 3 points for a stable polygon

        # Use point-in-polygon test
        return self.point_in_polygon(com_xy, support_polygon)

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting algorithm"""
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

    def estimate_support_polygon(self, current_state):
        """Estimate current support polygon from contact information"""
        # Get foot positions
        left_foot_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'left_foot')['position']
        right_foot_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'right_foot')['position']

        # Create conservative support polygon
        foot_size = [0.15, 0.08]  # Typical humanoid foot size

        # Calculate support polygon as convex hull of feet + margins
        support_points = [
            [left_foot_pos[0] - foot_size[0]/2, left_foot_pos[1] - foot_size[1]/2],   # Left foot corners
            [left_foot_pos[0] + foot_size[0]/2, left_foot_pos[1] - foot_size[1]/2],
            [left_foot_pos[0] + foot_size[0]/2, left_foot_pos[1] + foot_size[1]/2],
            [left_foot_pos[0] - foot_size[0]/2, left_foot_pos[1] + foot_size[1]/2],
            [right_foot_pos[0] - foot_size[0]/2, right_foot_pos[1] - foot_size[1]/2], # Right foot corners
            [right_foot_pos[0] + foot_size[0]/2, right_foot_pos[1] - foot_size[1]/2],
            [right_foot_pos[0] + foot_size[0]/2, right_foot_pos[1] + foot_size[1]/2],
            [right_foot_pos[0] - foot_size[0]/2, right_foot_pos[1] + foot_size[1]/2]
        ]

        # For now, return bounding box - in practice, would compute convex hull
        support_polygon = np.array([
            [min(p[0] for p in support_points), min(p[1] for p in support_points)],  # Bottom-left
            [max(p[0] for p in support_points), min(p[1] for p in support_points)],  # Bottom-right
            [max(p[0] for p in support_points), max(p[1] for p in support_points)],  # Top-right
            [min(p[0] for p in support_points), max(p[1] for p in support_points)]   # Top-left
        ])

        return support_polygon
```

## Priority-Based Task Control

### Task Prioritization Strategies

```python
class PriorityBasedController:
    """Handles priority-based task control with proper null-space management"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.n_joints = robot_model.get_num_joints()

    def execute_priority_control(self, current_state, tasks):
        """
        Execute tasks in priority order with proper null-space management

        Args:
            current_state: Current robot state
            tasks: List of tasks with priority levels

        Returns:
            Joint commands combining all tasks
        """
        # Sort tasks by priority (lower number = higher priority)
        sorted_tasks = sorted(tasks, key=lambda x: x['priority'])

        # Initialize commands and null space
        total_commands = np.zeros(self.n_joints)
        current_null_space = np.eye(self.n_joints)  # Start with full DOF availability

        for task in sorted_tasks:
            # Get task Jacobian and project into current null space
            task_jacobian = self.get_task_jacobian(task, current_state)
            projected_jacobian = task_jacobian @ current_null_space

            # Calculate task-space commands
            task_commands = self.compute_task_command(task, current_state)

            # Convert to joint space
            if projected_jacobian.shape[0] <= projected_jacobian.shape[1]:
                # Underdetermined system - use damped least squares
                damping = 0.01
                joint_commands = np.linalg.solve(
                    projected_jacobian.T @ projected_jacobian + damping * np.eye(projected_jacobian.shape[1]),
                    projected_jacobian.T @ task_commands
                )
            else:
                # Overdetermined system - use pseudo-inverse
                joint_commands = np.linalg.pinv(projected_jacobian) @ task_commands

            # Apply commands in current null space
            total_commands += current_null_space.T @ joint_commands

            # Update null space to exclude this task's DOFs
            task_null_space = self.compute_task_null_space(projected_jacobian)
            current_null_space = task_null_space @ current_null_space

        return total_commands

    def get_task_jacobian(self, task, current_state):
        """Get appropriate Jacobian for a task"""
        task_type = task['type']

        if task_type == 'balance':
            # Use CoM Jacobian for balance tasks
            com_jac = self.robot_model.com_jacobian(current_state['positions'])
            return com_jac[:2, :]  # Only x,y for balance
        elif task_type == 'manipulation':
            # Use end-effector Jacobian for manipulation
            ee_link = task.get('end_effector', 'right_hand')
            ee_jac = self.robot_model.jacobian_for_link(current_state['positions'], ee_link)
            return ee_jac  # Full 6-DOF for manipulation
        elif task_type == 'locomotion':
            # Use foot Jacobian for locomotion
            foot_link = task.get('foot', 'left_foot')
            foot_jac = self.robot_model.jacobian_for_link(current_state['positions'], foot_link)
            return foot_jac[:3, :]  # Only position for foot placement
        elif task_type == 'posture':
            # Use identity for joint-space posture tasks
            return np.eye(self.n_joints)
        else:
            # Default to identity
            return np.eye(self.n_joints)

    def compute_task_command(self, task, current_state):
        """Compute task-space command for a specific task"""
        task_type = task['type']

        if task_type == 'balance':
            return self.compute_balance_command(task, current_state)
        elif task_type == 'manipulation':
            return self.compute_manipulation_command(task, current_state)
        elif task_type == 'locomotion':
            return self.compute_locomotion_command(task, current_state)
        elif task_type == 'posture':
            return self.compute_posture_command(task, current_state)
        else:
            return np.zeros(task.get('dimension', 6))

    def compute_balance_command(self, task, current_state):
        """Compute balance task command"""
        # Get current CoM state
        current_com = self.robot_model.calculate_com_position(current_state['positions'])
        current_com_vel = self.robot_model.calculate_com_velocity(
            current_state['positions'], current_state['velocities'])

        # Get desired CoM state
        desired_com = task.get('desired_com', np.array([0, 0, 0.8]))
        desired_com_vel = task.get('desired_com_vel', np.array([0, 0, 0]))

        # Calculate errors
        com_error = desired_com[:2] - current_com[:2]  # Only x,y for balance
        com_vel_error = desired_com_vel[:2] - current_com_vel[:2]

        # Control gains
        kp = task.get('kp', 100.0)
        kd = task.get('kd', 20.0)

        # Compute balance correction
        balance_cmd = kp * com_error + kd * com_vel_error

        return balance_cmd

    def compute_manipulation_command(self, task, current_state):
        """Compute manipulation task command"""
        # Get current end-effector pose
        ee_link = task.get('end_effector', 'right_hand')
        current_ee_pose = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], ee_link)

        # Get desired pose
        desired_pose = task['desired_pose']

        # Calculate position error
        pos_error = desired_pose['position'] - current_ee_pose['position']

        # Calculate orientation error
        current_quat = current_ee_pose['orientation']
        desired_quat = desired_pose['orientation']
        quat_error = self.quaternion_difference(current_quat, desired_quat)

        # Combine position and orientation errors
        task_error = np.concatenate([pos_error, quat_error])

        # Control gains
        kp = task.get('kp', 300.0)
        kd = task.get('kd', 30.0)

        # Calculate desired velocity
        desired_vel = kp * task_error + kd * task.get('desired_vel', np.zeros(6))

        return desired_vel

    def compute_locomotion_command(self, task, current_state):
        """Compute locomotion task command"""
        # Get current foot pose
        foot_link = task.get('foot', 'left_foot')
        current_foot_pose = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], foot_link)

        # Get desired foot pose
        desired_foot_pose = task['desired_pose']

        # Calculate position error (for locomotion, we mainly care about position)
        pos_error = desired_foot_pose['position'] - current_foot_pose['position']

        # Control gains
        kp = task.get('kp', 200.0)
        kd = task.get('kd', 20.0)

        # Calculate desired velocity
        desired_vel = kp * pos_error[:3] + kd * task.get('desired_vel', np.zeros(3))

        return desired_vel

    def compute_posture_command(self, task, current_state):
        """Compute posture task command"""
        # Get desired joint positions
        desired_positions = task.get('desired_positions', current_state['positions'])

        # Calculate joint position errors
        pos_errors = desired_positions - current_state['positions']

        # Control gains
        kp = task.get('kp', 50.0)

        # Calculate desired joint velocities
        desired_velocities = kp * pos_errors

        return desired_velocities

    def compute_task_null_space(self, jacobian):
        """Compute null space projector for a task"""
        try:
            j_pinv = np.linalg.pinv(jacobian)
            return np.eye(jacobian.shape[1]) - j_pinv @ jacobian
        except np.linalg.LinAlgError:
            # If pseudo-inverse fails, return identity (no null space projection)
            return np.eye(jacobian.shape[1])

    def quaternion_difference(self, q1, q2):
        """Calculate difference between two quaternions as rotation vector"""
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        relative_rotation = r2 * r1.inv()
        return relative_rotation.as_rotvec()
```

## Constraint Resolution

### Handling Conflicting Objectives

```python
class ConstraintResolver:
    """Resolves conflicts between competing control objectives"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.n_joints = robot_model.get_num_joints()

    def resolve_conflicts(self, task_commands, task_jacobians, current_state):
        """
        Resolve conflicts between multiple simultaneous tasks

        Args:
            task_commands: List of task-space commands
            task_jacobians: List of task Jacobians
            current_state: Current robot state

        Returns:
            Resolved joint commands
        """
        # Check for constraint conflicts
        conflicts = self.detect_conflicts(task_jacobians, task_commands)

        if not conflicts:
            # No conflicts, use standard prioritized control
            return self.standard_prioritized_control(task_commands, task_jacobians)

        # Resolve conflicts using optimization-based approach
        return self.optimization_based_resolution(
            task_commands, task_jacobians, current_state, conflicts)

    def detect_conflicts(self, task_jacobians, task_commands):
        """Detect potential conflicts between tasks"""
        conflicts = []

        # For each pair of tasks, check if their Jacobians have overlapping null spaces
        for i in range(len(task_jacobians)):
            for j in range(i + 1, len(task_jacobians)):
                jac_i = task_jacobians[i]
                jac_j = task_jacobians[j]

                # Check if tasks operate in similar DOF spaces
                # Calculate the overlap between task spaces
                try:
                    # Project task j into task i's null space
                    null_i = self.compute_null_space(jac_i)
                    projected_j_in_i_null = null_i @ jac_j.T

                    # If projection is significant, there might be conflict
                    if np.linalg.norm(projected_j_in_i_null) > 0.1:
                        conflicts.append((i, j))
                except:
                    # If computation fails, assume no conflict for safety
                    continue

        return conflicts

    def standard_prioritized_control(self, task_commands, task_jacobians):
        """Standard prioritized control without conflicts"""
        joint_commands = np.zeros(self.n_joints)
        current_null_space = np.eye(self.n_joints)

        for i, (jac, cmd) in enumerate(zip(task_jacobians, task_commands)):
            # Project task into current null space
            projected_jac = jac @ current_null_space

            # Calculate joint commands for this task
            if projected_jac.shape[0] <= projected_jac.shape[1]:
                damping = 0.01
                task_joint_cmd = np.linalg.solve(
                    projected_jac.T @ projected_jac + damping * np.eye(projected_jac.shape[1]),
                    projected_jac.T @ cmd
                )
            else:
                task_joint_cmd = np.linalg.pinv(projected_jac) @ cmd

            # Apply in current null space
            joint_commands += current_null_space.T @ task_joint_cmd

            # Update null space
            task_null = self.compute_null_space(projected_jac)
            current_null_space = task_null @ current_null_space

        return joint_commands

    def optimization_based_resolution(self, task_commands, task_jacobians,
                                    current_state, conflicts):
        """Resolve conflicts using optimization"""
        try:
            from scipy.optimize import minimize
            import cvxpy as cp

            # Use CVXPY for quadratic programming approach
            # Define optimization variables
            joint_vars = cp.Variable(self.n_joints)

            # Define objective function: minimize weighted sum of task errors
            total_cost = 0

            for i, (jac, cmd) in enumerate(zip(task_jacobians, task_commands)):
                task_error = jac @ joint_vars - cmd
                weight = 1.0  # Could be task-specific
                total_cost += weight * cp.sum_squares(task_error)

            # Add regularization to prefer small joint changes
            current_joints = current_state['positions']
            regularization = 0.01 * cp.sum_squares(joint_vars - current_joints)
            total_cost += regularization

            # Define constraints
            constraints = []

            # Joint limit constraints
            joint_limits = self.robot_model.get_joint_limits()
            for i, (joint_name, limits) in enumerate(joint_limits.items()):
                constraints.append(joint_vars[i] >= limits['min'])
                constraints.append(joint_vars[i] <= limits['max'])

            # Velocity constraints
            max_velocity = 2.0  # rad/s
            for i in range(self.n_joints):
                # Assume we want to limit velocity change
                constraints.append(cp.abs(joint_vars[i] - current_joints[i]) <= max_velocity * self.dt)

            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(total_cost), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status == cp.OPTIMAL:
                return joint_vars.value
            else:
                # If optimization fails, fall back to standard prioritized control
                print(f"Optimization failed with status: {problem.status}, using fallback")
                return self.standard_prioritized_control(task_commands, task_jacobians)

        except ImportError:
            # If CVXPY is not available, use simplified conflict resolution
            return self.simplified_conflict_resolution(task_commands, task_jacobians, conflicts)

    def simplified_conflict_resolution(self, task_commands, task_jacobians, conflicts):
        """Simplified conflict resolution when optimization libraries not available"""
        # For conflicting tasks, reduce their influence proportionally
        adjusted_commands = [cmd.copy() for cmd in task_commands]

        for task_i, task_j in conflicts:
            # Reduce influence of lower priority task
            priority_diff = abs(task_commands[task_j].get('priority', 3) -
                              task_commands[task_i].get('priority', 3))

            if priority_diff == 0:
                # Same priority - split influence equally
                reduction_factor = 0.5
            elif task_commands[task_j]['priority'] > task_commands[task_i]['priority']:
                # task_j is lower priority - reduce more
                reduction_factor = 0.3
            else:
                # task_i is lower priority - reduce it instead
                reduction_factor = 0.3
                adjusted_commands[task_i] *= 0.7  # Keep 70% influence
                continue

            adjusted_commands[task_j] *= reduction_factor

        # Apply standard prioritized control with adjusted commands
        return self.standard_prioritized_control(adjusted_commands, task_jacobians)

    def compute_null_space(self, jacobian):
        """Compute null space projector for a Jacobian"""
        try:
            U, s, Vt = np.linalg.svd(jacobian, full_matrices=False)
            rank = np.sum(s > 1e-6)

            if rank == 0:
                return np.eye(jacobian.shape[1])

            Vr = Vt[:rank, :].T
            null_projector = np.eye(jacobian.shape[1]) - Vr @ Vr.T

            return null_projector
        except np.linalg.LinAlgError:
            try:
                j_pinv = np.linalg.pinv(jacobian)
                return np.eye(jacobian.shape[1]) - j_pinv @ jacobian
            except:
                return np.eye(jacobian.shape[1])

    def validate_solution(self, joint_commands, current_state):
        """Validate that the solution respects all constraints"""
        # Check joint limits
        joint_limits = self.robot_model.get_joint_limits()
        current_positions = current_state['positions']

        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            new_position = current_positions[i] + joint_commands[i] * 0.01  # 10ms step

            if new_position < limits['min'] or new_position > limits['max']:
                print(f"Warning: Joint {joint_name} would exceed limits: {new_position} (limits: {limits['min']} to {limits['max']})")
                # Limit the command to respect joint limits
                max_change = min(
                    (limits['max'] - current_positions[i]) / 0.01,
                    (current_positions[i] - limits['min']) / 0.01
                )
                joint_commands[i] = np.clip(joint_commands[i], -max_change, max_change)

        # Check for NaN or infinite values
        if np.any(np.isnan(joint_commands)) or np.any(np.isinf(joint_commands)):
            print("Error: Joint commands contain NaN or infinite values. Returning zero commands.")
            return np.zeros_like(joint_commands)

        return joint_commands
```

## Balance-Manipulation Coordination

### Coordinated Control Strategies

```python
class BalanceManipulationCoordinator:
    """Coordinates balance and manipulation tasks"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.n_joints = robot_model.get_num_joints()

        # Balance and manipulation controllers
        self.balance_controller = BalanceController(robot_model)
        self.manipulation_controller = ManipulationController(robot_model)

        # Coordination parameters
        self.balance_weight = 0.7  # Weight given to balance vs manipulation
        self.manipulation_weight = 0.3  # Weight given to manipulation vs balance
        self.com_shift_threshold = 0.1  # Threshold for CoM shift that triggers coordination

    def coordinate_balance_manipulation(self, current_state, manipulation_task,
                                      balance_requirements=None):
        """
        Coordinate balance and manipulation tasks to prevent instability

        Args:
            current_state: Current robot state
            manipulation_task: Desired manipulation task
            balance_requirements: Balance constraints (optional)

        Returns:
            Coordinated joint commands
        """
        # Calculate current CoM and stability
        current_com = self.robot_model.calculate_com_position(current_state['positions'])
        support_polygon = self.estimate_support_polygon(current_state)
        is_stable = self.is_com_stable(current_com[:2], support_polygon)

        # Estimate CoM shift due to manipulation
        estimated_com_shift = self.estimate_com_shift_for_manipulation(
            manipulation_task, current_state)

        # Check if manipulation will compromise balance
        if not is_stable or np.linalg.norm(estimated_com_shift) > self.com_shift_threshold:
            # Need to coordinate balance and manipulation
            return self.coordinated_control(current_state, manipulation_task,
                                         balance_requirements)
        else:
            # Manipulation is safe, can proceed with standard approach
            manipulation_cmds = self.manipulation_controller.compute_manipulation_control(
                current_state, manipulation_task)

            # Still apply some balance compensation
            balance_cmds = self.balance_controller.compute_balance_control(
                current_state, balance_requirements)

            # Combine with appropriate weights
            total_cmds = (self.balance_weight * balance_cmds +
                         self.manipulation_weight * manipulation_cmds)

            return total_cmds

    def estimate_com_shift_for_manipulation(self, manipulation_task, current_state):
        """Estimate how manipulation will shift the center of mass"""
        # This is a simplified estimation
        # In practice, would use more sophisticated forward dynamics

        if 'end_effector' in manipulation_task and 'desired_pose' in manipulation_task:
            ee_link = manipulation_task['end_effector']

            # Get current and desired end-effector positions
            current_ee_pos = self.robot_model.forward_kinematics_for_link(
                current_state['positions'], ee_link)['position']
            desired_ee_pos = manipulation_task['desired_pose']['position']

            # Estimate CoM shift based on end-effector movement
            ee_displacement = desired_ee_pos - current_ee_pos

            # Simplified model: CoM shifts proportionally to EE displacement
            # with a factor based on the robot's kinematic structure
            com_shift_factor = 0.1  # Empirical factor
            estimated_com_shift = ee_displacement[:2] * com_shift_factor  # Only x,y components

            return estimated_com_shift
        else:
            return np.zeros(2)

    def coordinated_control(self, current_state, manipulation_task, balance_requirements):
        """Perform coordinated balance-manipulation control"""
        try:
            from scipy.optimize import minimize

            def objective(joint_delta):
                """Minimize both manipulation error and balance error"""
                # Calculate new joint positions
                new_joints = current_state['positions'] + joint_delta

                # Calculate manipulation error
                manip_error = self.calculate_manipulation_error(
                    new_joints, manipulation_task)

                # Calculate balance error
                new_com = self.robot_model.calculate_com_position(new_joints)
                balance_error = self.calculate_balance_error(
                    new_com, balance_requirements)

                # Weighted combination of errors
                total_error = (self.manipulation_weight * np.sum(manip_error**2) +
                              self.balance_weight * np.sum(balance_error**2))

                # Add regularization for smooth motion
                total_error += 0.01 * np.sum(joint_delta**2)

                return total_error

            # Define constraints
            def joint_limit_constraint(joint_delta):
                """Ensure joint limits are respected"""
                new_joints = current_state['positions'] + joint_delta
                joint_limits = self.robot_model.get_joint_limits()

                violations = []
                for i, (joint_name, limits) in enumerate(joint_limits.items()):
                    if new_joints[i] < limits['min'] or new_joints[i] > limits['max']:
                        violations.append(1.0)  # Return positive value if violated
                    else:
                        violations.append(0.0)

                return -np.array(violations)  # Negative because constraint >= 0

            # Initial guess (small random movement)
            initial_guess = np.zeros(self.n_joints)

            # Define bounds based on joint limits
            joint_limits = self.robot_model.get_joint_limits()
            bounds = []
            max_change = 0.1  # Maximum change per step (0.1 rad)

            for joint_name, limits in joint_limits.items():
                bounds.append((-max_change, max_change))

            # Solve optimization problem
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': joint_limit_constraint},
                options={'maxiter': 100}
            )

            if result.success:
                return result.x
            else:
                # If optimization fails, use fallback approach
                return self.fallback_coordinated_control(
                    current_state, manipulation_task, balance_requirements)

        except ImportError:
            # If scipy is not available, use fallback approach
            return self.fallback_coordinated_control(
                current_state, manipulation_task, balance_requirements)

    def fallback_coordinated_control(self, current_state, manipulation_task, balance_requirements):
        """Fallback coordinated control when optimization libraries not available"""
        # Calculate manipulation commands
        manip_cmds = self.manipulation_controller.compute_manipulation_control(
            current_state, manipulation_task)

        # Calculate balance commands
        balance_cmds = self.balance_controller.compute_balance_control(
            current_state, balance_requirements)

        # Use priority-based approach with null-space projection
        manip_jac = self.get_manipulation_jacobian(current_state, manipulation_task)
        balance_jac = self.robot_model.com_jacobian(current_state['positions'])[:2, :]

        # First, apply balance commands
        total_cmds = balance_cmds

        # Then, apply manipulation in balance null space
        try:
            balance_null = self.compute_balance_null_space(balance_jac)
            manip_in_null = balance_null @ manip_cmds
            total_cmds += manip_in_null
        except:
            # If null space computation fails, blend commands by weight
            total_cmds = self.balance_weight * balance_cmds + self.manipulation_weight * manip_cmds

        return total_cmds

    def calculate_manipulation_error(self, joint_positions, manipulation_task):
        """Calculate manipulation task error"""
        if 'end_effector' in manipulation_task and 'desired_pose' in manipulation_task:
            ee_link = manipulation_task['end_effector']
            current_ee_pose = self.robot_model.forward_kinematics_for_link(
                joint_positions, ee_link)
            desired_ee_pose = manipulation_task['desired_pose']

            # Position error
            pos_error = desired_ee_pose['position'] - current_ee_pose['position']

            # Orientation error
            current_quat = current_ee_pose['orientation']
            desired_quat = desired_ee_pose['orientation']
            quat_error = self.quaternion_difference(current_quat, desired_quat)

            return np.concatenate([pos_error, quat_error])
        else:
            return np.zeros(6)

    def calculate_balance_error(self, com_position, balance_requirements):
        """Calculate balance error"""
        if balance_requirements and 'desired_com' in balance_requirements:
            desired_com = balance_requirements['desired_com']
            return desired_com[:2] - com_position[:2]  # Only x,y for balance
        else:
            # Default to center of support polygon
            support_center = np.array([0.0, 0.0])  # Would be calculated from feet positions
            return support_center - com_position[:2]

    def get_manipulation_jacobian(self, current_state, manipulation_task):
        """Get manipulation task Jacobian"""
        ee_link = manipulation_task.get('end_effector', 'right_hand')
        ee_jac = self.robot_model.jacobian_for_link(current_state['positions'], ee_link)
        return ee_jac

    def compute_balance_null_space(self, balance_jacobian):
        """Compute null space for balance task"""
        try:
            U, s, Vt = np.linalg.svd(balance_jacobian, full_matrices=False)
            rank = np.sum(s > 1e-6)

            if rank == 0:
                return np.eye(balance_jacobian.shape[1])

            Vr = Vt[:rank, :].T
            null_projector = np.eye(balance_jacobian.shape[1]) - Vr @ Vr.T

            return null_projector
        except:
            j_pinv = np.linalg.pinv(balance_jacobian)
            return np.eye(balance_jacobian.shape[1]) - j_pinv @ balance_jacobian

    def estimate_support_polygon(self, current_state):
        """Estimate support polygon from current state"""
        # Get foot positions
        left_foot_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'left_foot')['position'][:2]
        right_foot_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'right_foot')['position'][:2]

        # Create support polygon (simplified as rectangle between feet)
        support_polygon = np.array([
            [min(left_foot_pos[0], right_foot_pos[0]) - 0.05,  # Add small margins
             min(left_foot_pos[1], right_foot_pos[1]) - 0.05],
            [max(left_foot_pos[0], right_foot_pos[0]) + 0.05,
             min(left_foot_pos[1], right_foot_pos[1]) - 0.05],
            [max(left_foot_pos[0], right_foot_pos[0]) + 0.05,
             max(left_foot_pos[1], right_foot_pos[1]) + 0.05],
            [min(left_foot_pos[0], right_foot_pos[0]) - 0.05,
             max(left_foot_pos[1], right_foot_pos[1]) + 0.05]
        ])

        return support_polygon

    def is_com_stable(self, com_xy, support_polygon):
        """Check if CoM is stable within support polygon"""
        return self.point_in_polygon(com_xy, support_polygon)

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

    def quaternion_difference(self, q1, q2):
        """Calculate quaternion difference as rotation vector"""
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        relative_rotation = r2 * r1.inv()
        return relative_rotation.as_rotvec()
```

## Performance Optimization

### Efficient Integration Techniques

```python
class OptimizedWholeBodyIntegrator:
    """Performance-optimized whole-body control integration"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.n_joints = robot_model.get_num_joints()

        # Pre-allocate arrays to avoid allocation during control
        self.pre_allocated = {
            'joint_positions': np.zeros(self.n_joints),
            'joint_velocities': np.zeros(self.n_joints),
            'joint_commands': np.zeros(self.n_joints),
            'mass_matrix': np.zeros((self.n_joints, self.n_joints)),
            'com_jacobian': np.zeros((3, self.n_joints)),
            'ee_jacobian': np.zeros((6, self.n_joints)),
            'null_projector': np.zeros((self.n_joints, self.n_joints)),
            'task_error': np.zeros(6),
            'temp_vector': np.zeros(self.n_joints)
        }

        # Caching for expensive computations
        self.jacobian_cache = {}
        self.mass_matrix_cache = {}
        self.forward_kinematics_cache = {}

        # Control timing
        self.control_frequency = 1000  # Hz
        self.dt = 1.0 / self.control_frequency

        # Performance monitoring
        self.performance_stats = {
            'control_calls': 0,
            'avg_time': 0.0,
            'max_time': 0.0,
            'min_time': float('inf'),
            'cache_hits': 0,
            'cache_misses': 0
        }

    def integrate_controls_optimized(self, current_state, high_level_goals):
        """Optimized control integration using pre-allocated arrays and caching"""
        start_time = time.time()

        # Use pre-allocated arrays to avoid memory allocation
        joint_positions = self.pre_allocated['joint_positions']
        joint_velocities = self.pre_allocated['joint_velocities']
        joint_commands = self.pre_allocated['joint_commands']

        # Copy current state to pre-allocated arrays
        np.copyto(joint_positions, current_state['positions'])
        np.copyto(joint_velocities, current_state['velocities'])

        # Initialize commands
        joint_commands.fill(0.0)

        # Process tasks in priority order
        tasks = self.parse_high_level_goals(high_level_goals)

        # Start with full DOF available
        current_null_space = np.eye(self.n_joints, out=self.pre_allocated['null_projector'])

        for task in tasks:
            # Get task Jacobian (using caching)
            task_jac = self.get_cached_jacobian_optimized(task, joint_positions)

            # Project into current null space
            projected_jac = self.multiply_matrices_optimized(current_null_space, task_jac.T).T

            # Calculate task command
            task_cmd = self.compute_task_command_optimized(task, current_state)

            # Solve for joint commands in projected space
            joint_delta = self.solve_task_inverse_optimized(projected_jac, task_cmd)

            # Apply to total commands in null space
            self.apply_command_in_null_space_optimized(
                joint_commands, current_null_space.T, joint_delta)

            # Update null space
            task_null = self.compute_task_null_space_optimized(projected_jac)
            self.update_null_space_optimized(current_null_space, task_null)

        # Apply constraints
        self.apply_joint_constraints_optimized(joint_commands, current_state)

        # Update performance stats
        elapsed_time = time.time() - start_time
        self.update_performance_stats(elapsed_time)

        return joint_commands.copy()

    def get_cached_jacobian_optimized(self, task, joint_positions):
        """Get cached Jacobian matrix for performance"""
        # Create cache key based on joint configuration and task
        joint_hash = self.hash_joints_optimized(joint_positions)
        cache_key = (joint_hash, task['type'], task.get('link', ''))

        if cache_key not in self.jacobian_cache:
            self.jacobian_cache[cache_key] = self.compute_task_jacobian(task, joint_positions)

            # Limit cache size
            if len(self.jacobian_cache) > 100:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.jacobian_cache))
                del self.jacobian_cache[oldest_key]

            self.performance_stats['cache_misses'] += 1
        else:
            self.performance_stats['cache_hits'] += 1

        return self.jacobian_cache[cache_key]

    def hash_joints_optimized(self, joint_positions):
        """Create hash of joint positions for caching"""
        # Round to reduce sensitivity to tiny changes
        rounded_joints = tuple(np.round(joint_positions, decimals=4))
        return hash(rounded_joints)

    def multiply_matrices_optimized(self, A, B, out=None):
        """Optimized matrix multiplication with output array"""
        if out is None:
            return A @ B
        else:
            np.matmul(A, B, out=out)
            return out

    def solve_task_inverse_optimized(self, jacobian, task_command):
        """Optimized solution of task inverse using pre-allocated arrays"""
        temp_result = self.pre_allocated['temp_vector']

        if jacobian.shape[0] <= jacobian.shape[1]:
            # Use damped least squares for underdetermined systems
            damping = 0.01
            JT = jacobian.T
            A = JT @ jacobian + damping * np.eye(jacobian.shape[1])
            b = JT @ task_command

            # Solve in-place to avoid allocation
            try:
                joint_delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                joint_delta = np.linalg.pinv(jacobian) @ task_command
        else:
            # Use pseudo-inverse for overdetermined systems
            joint_delta = np.linalg.pinv(jacobian) @ task_command

        return joint_delta

    def apply_command_in_null_space_optimized(self, total_commands, null_space_T, task_command):
        """Apply task command in null space using optimized operations"""
        # Use BLAS-level operation for efficiency
        result = null_space_T @ task_command
        total_commands += result

    def compute_task_null_space_optimized(self, jacobian):
        """Optimized null space computation"""
        # Use pre-allocated temporary arrays
        U = self.pre_allocated['temp_vector'][:jacobian.shape[0]]
        VT = self.pre_allocated['temp_vector'][:jacobian.shape[1]]

        try:
            # Use SVD for numerical stability
            U, s, Vt = np.linalg.svd(jacobian, full_matrices=False, compute_uv=True)
            rank = np.sum(s > 1e-6)

            null_projector = self.pre_allocated['null_projector']

            if rank == 0:
                null_projector[:] = np.eye(jacobian.shape[1])
            else:
                # Calculate null space projector: I - V_r * V_r^T
                Vr = Vt[:rank, :].T
                np.eye(jacobian.shape[1], out=null_projector)
                temp_result = Vr @ Vr.T
                null_projector -= temp_result

            return null_projector
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse method
            j_pinv = np.linalg.pinv(jacobian)
            result = np.eye(jacobian.shape[1]) - j_pinv @ jacobian
            return result

    def update_null_space_optimized(self, current_null, task_null):
        """Update current null space with task null space"""
        # Compute: current_null = task_null @ current_null
        temp = current_null.copy()
        current_null[:] = task_null @ temp

    def apply_joint_constraints_optimized(self, commands, current_state):
        """Optimized joint constraint application"""
        joint_limits = self.robot_model.get_joint_limits()

        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            # Calculate next position with command
            next_pos = current_state['positions'][i] + commands[i] * self.dt * 0.1

            # Apply position limits
            if next_pos < limits['min']:
                max_change = (limits['min'] - current_state['positions'][i]) / (self.dt * 0.1)
                commands[i] = min(commands[i], max_change)
            elif next_pos > limits['max']:
                max_change = (limits['max'] - current_state['positions'][i]) / (self.dt * 0.1)
                commands[i] = max(commands[i], max_change)

            # Apply velocity limits
            max_vel = limits.get('max_velocity', 2.0)
            max_change = max_vel * self.dt
            commands[i] = np.clip(commands[i], -max_change, max_change)

    def update_performance_stats(self, elapsed_time):
        """Update control performance statistics"""
        self.performance_stats['control_calls'] += 1
        self.performance_stats['avg_time'] = (
            (self.performance_stats['avg_time'] * (self.performance_stats['control_calls'] - 1) + elapsed_time) /
            self.performance_stats['control_calls']
        )
        self.performance_stats['max_time'] = max(self.performance_stats['max_time'], elapsed_time)
        self.performance_stats['min_time'] = min(self.performance_stats['min_time'], elapsed_time)

    def get_performance_report(self):
        """Get performance statistics report"""
        total_cache_accesses = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        cache_hit_rate = (self.performance_stats['cache_hits'] / total_cache_accesses * 100
                         if total_cache_accesses > 0 else 0)

        return {
            'control_frequency_actual': 1.0 / self.performance_stats['avg_time'] if self.performance_stats['avg_time'] > 0 else 0,
            'avg_control_time_ms': self.performance_stats['avg_time'] * 1000,
            'max_control_time_ms': self.performance_stats['max_time'] * 1000,
            'min_control_time_ms': self.performance_stats['min_time'] * 1000,
            'cache_hit_rate_percent': cache_hit_rate,
            'total_control_calls': self.performance_stats['control_calls']
        }

    def parse_high_level_goals(self, goals):
        """Parse high-level goals into prioritized tasks"""
        tasks = []

        # Balance task (highest priority)
        if 'balance' in goals or 'stability' in goals:
            tasks.append({
                'type': 'balance',
                'priority': 1,
                'link': 'com',
                'desired_state': goals.get('balance', {}),
                'weight': 1.0
            })

        # Locomotion task (high priority)
        if 'locomotion' in goals or 'walking' in goals:
            tasks.append({
                'type': 'locomotion',
                'priority': 2,
                'link': goals.get('foot', 'left_foot'),
                'desired_state': goals.get('locomotion', {}),
                'weight': 0.8
            })

        # Manipulation task (medium priority)
        if 'manipulation' in goals or 'arm' in goals:
            tasks.append({
                'type': 'manipulation',
                'priority': 3,
                'link': goals.get('end_effector', 'right_hand'),
                'desired_state': goals.get('manipulation', {}),
                'weight': 0.6
            })

        # Posture task (low priority)
        if 'posture' in goals or 'configuration' in goals:
            tasks.append({
                'type': 'posture',
                'priority': 4,
                'link': 'joints',
                'desired_state': goals.get('posture', {}),
                'weight': 0.3
            })

        # Sort by priority
        return sorted(tasks, key=lambda x: x['priority'])

    def compute_task_jacobian(self, task, joint_positions):
        """Compute task-specific Jacobian"""
        task_type = task['type']
        link_name = task.get('link', '')

        if task_type == 'balance':
            com_jac = self.robot_model.com_jacobian(joint_positions)
            return com_jac[:2, :]  # Only x,y for balance
        elif task_type == 'manipulation':
            ee_jac = self.robot_model.jacobian_for_link(joint_positions, link_name)
            return ee_jac  # Full 6-DOF for manipulation
        elif task_type == 'locomotion':
            foot_jac = self.robot_model.jacobian_for_link(joint_positions, link_name)
            return foot_jac[:3, :]  # Only position for foot placement
        elif task_type == 'posture':
            return np.eye(self.n_joints)  # Identity for joint-space tasks
        else:
            return np.eye(self.n_joints)

    def compute_task_command_optimized(self, task, current_state):
        """Compute optimized task command"""
        task_type = task['type']
        desired_state = task.get('desired_state', {})

        if task_type == 'balance':
            return self.compute_balance_command_optimized(task, current_state)
        elif task_type == 'manipulation':
            return self.compute_manipulation_command_optimized(task, current_state)
        elif task_type == 'locomotion':
            return self.compute_locomotion_command_optimized(task, current_state)
        elif task_type == 'posture':
            return self.compute_posture_command_optimized(task, current_state)
        else:
            return np.zeros(6)
```

## Validation and Testing

### Control System Validation

```python
class ControlValidationSystem:
    """System for validating whole-body control integration"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def validate_balance_manipulation_integration(self, current_state, test_scenario):
        """Validate balance-manipulation coordination"""
        validation_results = {
            'stability_preserved': True,
            'manipulation_accuracy': 0.0,
            'balance_recovery_capability': True,
            'constraint_satisfaction': True
        }

        # Test 1: Verify stability is maintained during manipulation
        initial_com = self.robot_model.calculate_com_position(current_state['positions'])
        initial_support_polygon = self.estimate_support_polygon(current_state)

        # Execute manipulation task
        manip_task = {
            'type': 'manipulation',
            'end_effector': 'right_hand',
            'desired_pose': {
                'position': np.array([0.5, 0.2, 0.8]),
                'orientation': np.array([0, 0, 0, 1])
            }
        }

        commands = self.integrate_controls(current_state, {
            'manipulation_task': manip_task,
            'balance_requirements': {'desired_com': initial_com}
        })

        # Simulate command execution (simplified)
        new_positions = current_state['positions'] + commands * 0.01  # 10ms integration

        # Check if CoM remains stable
        new_com = self.robot_model.calculate_com_position(new_positions)
        new_support_polygon = self.estimate_support_polygon({'positions': new_positions})

        if not self.is_com_stable(new_com[:2], new_support_polygon):
            validation_results['stability_preserved'] = False

        # Test 2: Verify manipulation accuracy
        new_ee_pos = self.robot_model.forward_kinematics_for_link(
            new_positions, 'right_hand')['position']
        desired_ee_pos = manip_task['desired_pose']['position']
        manip_error = np.linalg.norm(new_ee_pos - desired_ee_pos)
        validation_results['manipulation_accuracy'] = manip_error

        # Test 3: Verify balance recovery capability
        # Perturb robot and check if balance is recovered
        perturbed_positions = new_positions.copy()
        perturbed_positions[0] += 0.1  # Artificial perturbation
        perturbed_state = {'positions': perturbed_positions, 'velocities': current_state['velocities']}

        balance_recovery_commands = self.integrate_controls(perturbed_state, {
            'behavior': 'balance_recovery'
        })

        # Check if recovery commands are appropriate
        if np.linalg.norm(balance_recovery_commands) < 0.1:
            validation_results['balance_recovery_capability'] = False

        # Test 4: Verify constraint satisfaction
        joint_limits = self.robot_model.get_joint_limits()
        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            if new_positions[i] < limits['min'] or new_positions[i] > limits['max']:
                validation_results['constraint_satisfaction'] = False
                break

        return validation_results

    def validate_task_coordination(self, current_state, task_combinations):
        """Validate coordination between different task types"""
        validation_results = {
            'task_interference': False,
            'priority_respected': True,
            'smooth_transitions': True,
            'performance_metrics': {}
        }

        for task_combo in task_combinations:
            # Execute combined tasks
            commands = self.integrate_controls(current_state, task_combo)

            # Check for excessive joint commands (sign of interference)
            if np.max(np.abs(commands)) > 50.0:  # Threshold for excessive commands
                validation_results['task_interference'] = True

            # Validate that high-priority tasks are satisfied
            # This would involve checking specific task objectives
            pass

        return validation_results

    def run_comprehensive_validation(self, current_state):
        """Run comprehensive validation of control system"""
        validation_suite = {
            'balance_manipulation': self.validate_balance_manipulation_integration(
                current_state, 'standard_scenario'),
            'locomotion_stability': self.validate_locomotion_balance_integration(
                current_state, 'walking_scenario'),
            'multi_task_coordination': self.validate_task_coordination(
                current_state, self.generate_test_task_combinations()),
            'performance_under_load': self.validate_performance_metrics(current_state)
        }

        # Aggregate results
        overall_success = all(
            result.get('stability_preserved', True) and
            result.get('constraint_satisfaction', True)
            for result in validation_suite.values()
            if isinstance(result, dict)
        )

        return {
            'overall_success': overall_success,
            'individual_results': validation_suite,
            'recommendations': self.generate_recommendations(validation_suite)
        }

    def generate_test_task_combinations(self):
        """Generate test combinations for validation"""
        return [
            {
                'balance_requirements': {'desired_com': np.array([0, 0, 0.8])},
                'manipulation_task': {
                    'end_effector': 'left_hand',
                    'desired_pose': {'position': np.array([0.3, 0.1, 0.7]), 'orientation': [0, 0, 0, 1]}
                }
            },
            {
                'balance_requirements': {'desired_com': np.array([0.1, 0.05, 0.8])},
                'locomotion_target': {'foot': 'right_foot', 'position': np.array([0.4, -0.1, 0])}
            },
            {
                'balance_requirements': {'desired_com': np.array([0, 0, 0.8])},
                'manipulation_task': {
                    'end_effector': 'right_hand',
                    'desired_pose': {'position': np.array([0.6, 0.3, 0.9]), 'orientation': [0, 0, 0, 1]}
                },
                'locomotion_target': {'foot': 'left_foot', 'position': np.array([0.3, 0.15, 0])}
            }
        ]

    def validate_locomotion_balance_integration(self, current_state, scenario):
        """Validate locomotion and balance integration"""
        # This would test that when executing stepping motions, balance is maintained
        results = {
            'balance_maintained': True,
            'step_accuracy': 0.0,
            'transition_smoothness': True
        }

        # For locomotion validation, check that CoM follows appropriate trajectory
        # during stepping motions while maintaining stability

        return results

    def validate_performance_metrics(self, current_state):
        """Validate performance metrics"""
        results = {
            'control_frequency': 0,
            'computation_time': 0,
            'memory_usage': 0,
            'numerical_stability': True
        }

        # Test computation time under various scenarios
        import time
        start = time.time()
        for _ in range(100):
            commands = self.integrate_controls(current_state, {'behavior': 'stand'})
        end = time.time()

        avg_time = (end - start) / 100
        results['computation_time'] = avg_time
        results['control_frequency'] = 1.0 / avg_time if avg_time > 0 else 0

        # Check for numerical stability (no NaN or infinite values)
        if np.any(np.isnan(commands)) or np.any(np.isinf(commands)):
            results['numerical_stability'] = False

        return results

    def generate_recommendations(self, validation_results):
        """Generate recommendations based on validation results"""
        recommendations = []

        if not validation_results['balance_manipulation']['stability_preserved']:
            recommendations.append(
                "Balance control needs improvement - consider increasing balance gains")

        if validation_results['balance_manipulation']['manipulation_accuracy'] > 0.05:  # 5cm error threshold
            recommendations.append(
                "Manipulation accuracy needs improvement - consider retuning task-specific gains")

        if not validation_results['balance_manipulation']['constraint_satisfaction']:
            recommendations.append(
                "Constraint handling needs improvement - verify joint limit enforcement")

        if validation_results['balance_manipulation']['computation_time'] > 0.01:  # 10ms threshold
            recommendations.append(
                "Performance needs optimization - consider algorithm improvements or caching")

        return recommendations
```

## Summary

Whole-body control integration is the sophisticated orchestration of multiple simultaneous control objectives in humanoid robots. The key aspects covered in this chapter include:

1. **Hierarchical Control**: Organizing control tasks by priority to ensure critical functions (balance) are maintained while achieving other objectives (manipulation, locomotion).

2. **Null-Space Projection**: Using mathematical techniques to allow lower-priority tasks to operate in the null space of higher-priority tasks, preventing interference.

3. **Constraint Resolution**: Handling conflicts between competing objectives using optimization techniques or priority-based approaches.

4. **Balance-Manipulation Coordination**: Ensuring that manipulation tasks don't compromise balance stability through coordinated control strategies.

5. **Performance Optimization**: Implementing efficient algorithms with caching and pre-allocation to meet real-time control requirements.

The success of humanoid robots depends critically on the seamless integration of these control systems, allowing the robot to perform complex tasks while maintaining stability and safety. Proper whole-body control integration enables humanoid robots to operate effectively in dynamic environments with multiple simultaneous objectives.