---
title: Chapter 8 - Balance Control and Locomotion
description: Advanced balance control and locomotion techniques for humanoid robots
sidebar_position: 8
---

# Chapter 8: Balance Control and Locomotion

## Learning Objectives

After completing this chapter, you should be able to:
- Understand the fundamental principles of balance control in humanoid robots
- Implement Center of Mass (CoM) and Zero Moment Point (ZMP) based balance control
- Design walking pattern generators for bipedal locomotion
- Apply advanced control techniques like Capture Point and MPC for balance
- Integrate balance control with locomotion for stable walking

## Introduction

Balance control is one of the most challenging aspects of humanoid robotics. Unlike wheeled robots that maintain continuous contact with the ground, humanoid robots must dynamically manage their balance during both static poses and dynamic locomotion. This chapter covers the essential theories and practical implementations for achieving stable balance and locomotion in humanoid robots.

## Balance Fundamentals

### Center of Mass (CoM) and Center of Pressure (CoP)

The Center of Mass (CoM) is the point where the total mass of the robot can be considered to be concentrated. For balance, the CoM must remain within the support polygon defined by the contact points with the ground.

```python
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class CenterOfMassCalculator:
    """Calculate and analyze Center of Mass for humanoid robots"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def calculate_com_position(self, joint_positions):
        """Calculate CoM position from joint positions"""
        total_mass = 0.0
        weighted_sum = np.zeros(3)

        # Get link poses and masses
        link_poses = self.robot_model.forward_kinematics(joint_positions)
        link_masses = self.robot_model.get_link_masses()

        for link_name, pose in link_poses.items():
            if link_name in link_masses:
                mass = link_masses[link_name]
                # Calculate CoM of this link in world frame
                link_com_local = self.robot_model.get_link_com_offset(link_name)
                link_rotation = self.quaternion_to_rotation_matrix(pose['orientation'])
                link_com_world = pose['position'] + link_rotation @ link_com_local

                weighted_sum += mass * link_com_world
                total_mass += mass

        if total_mass > 0:
            com_position = weighted_sum / total_mass
        else:
            com_position = np.zeros(3)

        return com_position

    def calculate_com_jacobian(self, joint_positions):
        """Calculate Jacobian matrix relating joint velocities to CoM velocity"""
        # This is a simplified implementation
        # In practice, this would use analytical derivatives or numerical methods

        n_joints = len(joint_positions)
        com_jacobian = np.zeros((3, n_joints))  # [dx, dy, dz] / [djoint1, djoint2, ...]

        # Calculate effect of each joint on CoM position
        current_com = self.calculate_com_position(joint_positions)

        # Small perturbation for numerical derivative
        epsilon = 1e-6

        for i in range(n_joints):
            # Perturb joint position
            perturbed_positions = joint_positions.copy()
            perturbed_positions[i] += epsilon

            # Calculate new CoM position
            perturbed_com = self.calculate_com_position(perturbed_positions)

            # Calculate derivative (change in CoM per change in joint)
            com_jacobian[:, i] = (perturbed_com - current_com) / epsilon

        return com_jacobian

    def calculate_support_polygon(self, contact_points):
        """Calculate support polygon from contact points"""
        if len(contact_points) < 3:
            # For 2 points, use line segment
            if len(contact_points) == 2:
                return self.calculate_line_support(contact_points)
            # For 1 point, use point (though not stable)
            elif len(contact_points) == 1:
                return contact_points
        else:
            # For 3+ points, use convex hull
            points_2d = np.array([[p[0], p[1]] for p in contact_points])
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]

    def calculate_line_support(self, contact_points):
        """Calculate support polygon for 2 contact points (line segment)"""
        # For 2 points, support region is line segment with small width
        p1, p2 = contact_points
        midpoint = (p1 + p2) / 2
        direction = p2 - p1
        direction_perp = np.array([-direction[1], direction[0], 0])  # Perpendicular in x-y plane
        direction_perp = direction_perp / np.linalg.norm(direction_perp)

        # Create small rectangle around line segment
        width = 0.05  # 5cm width
        offset = direction_perp * width / 2

        return np.array([
            p1 + offset,
            p2 + offset,
            p2 - offset,
            p1 - offset
        ])

    def is_stable(self, com_position, contact_points):
        """Check if CoM is within support polygon"""
        if len(contact_points) == 0:
            return False

        if len(contact_points) == 1:
            # Point contact - check if CoM is at contact point (within tolerance)
            contact = contact_points[0]
            distance = np.linalg.norm(com_position[:2] - contact[:2])
            return distance < 0.05  # 5cm tolerance

        if len(contact_points) == 2:
            # Line contact - check if CoM is near line segment
            return self.is_point_near_line_segment(
                com_position[:2], contact_points[0][:2], contact_points[1][:2])

        # For 3+ points, use polygon containment
        support_polygon = self.calculate_support_polygon(contact_points)
        return self.is_point_in_convex_polygon(
            com_position[:2], support_polygon)

    def is_point_near_line_segment(self, point, line_start, line_end, tolerance=0.05):
        """Check if point is near line segment"""
        # Calculate distance from point to line segment
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:
            # Line is actually a point
            return np.linalg.norm(point - line_start) < tolerance

        # Project point onto line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = line_start + t * line_vec
        distance = np.linalg.norm(point - projection)

        return distance < tolerance

    def is_point_in_convex_polygon(self, point, polygon):
        """Check if point is inside convex polygon using cross product method"""
        n = len(polygon)

        # Check if point is on same side of all edges
        sign = None
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Calculate cross product to determine which side of edge point is on
            cross_product = ((p2[0] - p1[0]) * (point[1] - p1[1]) -
                           (p2[1] - p1[1]) * (point[0] - p1[0]))

            current_sign = 1 if cross_product >= 0 else -1

            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                # Point is on different sides of edges, so outside polygon
                return False

        return True

    def calculate_stability_margin(self, com_position, contact_points):
        """Calculate stability margin (distance from CoM to support polygon boundary)"""
        if len(contact_points) == 0:
            return -float('inf')

        if len(contact_points) == 1:
            # For point contact, return distance to contact point
            contact = contact_points[0]
            return -np.linalg.norm(com_position[:2] - contact[:2])

        if len(contact_points) == 2:
            # For line contact, return distance to line segment
            return -self.distance_to_line_segment(
                com_position[:2], contact_points[0][:2], contact_points[1][:2])

        # For polygon support, calculate distance to boundary
        support_polygon = self.calculate_support_polygon(contact_points)

        # Find minimum distance to any edge of polygon
        min_distance = float('inf')
        for i in range(len(support_polygon)):
            p1 = support_polygon[i]
            p2 = support_polygon[(i + 1) % len(support_polygon)]

            distance = self.distance_point_to_line_segment(
                com_position[:2], p1, p2)
            min_distance = min(min_distance, distance)

        # If CoM is inside polygon, return positive distance to boundary
        # If CoM is outside polygon, return negative distance
        is_inside = self.is_point_in_convex_polygon(com_position[:2], support_polygon)
        return min_distance if is_inside else -min_distance

    def distance_point_to_line_segment(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:
            return np.linalg.norm(point - line_start)

        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection)

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def plot_balance_analysis(self, com_position, contact_points, title="Balance Analysis"):
        """Plot CoM position relative to support polygon"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot support polygon
        if len(contact_points) >= 3:
            support_polygon = self.calculate_support_polygon(contact_points)
            polygon_x = np.append(support_polygon[:, 0], support_polygon[0, 0])
            polygon_y = np.append(support_polygon[:, 1], support_polygon[0, 1])
            ax.plot(polygon_x, polygon_y, 'b-', linewidth=2, label='Support Polygon')

            # Fill support polygon
            ax.fill(support_polygon[:, 0], support_polygon[:, 1], 'lightblue', alpha=0.3)
        elif len(contact_points) == 2:
            # Draw line segment for 2 contacts
            points = np.array(contact_points)
            ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Support Line')
        elif len(contact_points) == 1:
            # Draw point for single contact
            ax.plot(contact_points[0][0], contact_points[0][1], 'bo', markersize=10, label='Support Point')

        # Plot CoM
        ax.plot(com_position[0], com_position[1], 'ro', markersize=10, label='CoM')

        # Add stability margin info
        margin = self.calculate_stability_margin(com_position, contact_points)
        ax.text(0.02, 0.98, f'Stability Margin: {margin:.3f}m', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        plt.show()
```

### Zero Moment Point (ZMP) Theory

The Zero Moment Point is a crucial concept in humanoid balance control. It represents the point on the ground where the net moment of the ground reaction force is zero.

```python
class ZMPController:
    """Zero Moment Point based balance controller"""

    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height  # Height of center of mass above ground
        self.gravity = gravity
        self.omega = np.sqrt(self.gravity / self.com_height)  # Natural frequency of inverted pendulum

        # ZMP tracking controller parameters
        self.kp = 100.0  # Proportional gain
        self.kd = 20.0   # Derivative gain

        # State estimation
        self.com_position = np.zeros(3)
        self.com_velocity = np.zeros(3)
        self.com_acceleration = np.zeros(3)

        # Desired trajectories
        self.desired_zmp = np.zeros(2)
        self.desired_com = np.zeros(3)

    def calculate_zmp(self, com_pos, com_acc):
        """Calculate Zero Moment Point from CoM position and acceleration"""
        # ZMP_x = CoM_x - (CoM_height / gravity) * CoM_acc_x
        # ZMP_y = CoM_y - (CoM_height / gravity) * CoM_acc_y
        zmp_x = com_pos[0] - (self.com_height / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (self.com_height / self.gravity) * com_acc[1]

        return np.array([zmp_x, zmp_y])

    def calculate_com_from_zmp(self, zmp_pos, zmp_vel, zmp_acc):
        """Calculate required CoM position to achieve desired ZMP"""
        # Inverted pendulum relationship:
        # CoM_pos = ZMP_pos + (CoM_height / gravity) * CoM_acc
        # For control, we want: CoM_acc = (CoM_pos - ZMP_pos) * (gravity / CoM_height)

        # Desired CoM acceleration to track ZMP
        com_acc_x = (self.com_position[0] - zmp_pos[0]) * (self.gravity / self.com_height)
        com_acc_y = (self.com_position[1] - zmp_pos[1]) * (self.gravity / self.com_height)

        # Add feedback to track ZMP trajectory
        zmp_error = zmp_pos - self.calculate_zmp(self.com_position, self.com_acceleration)
        com_acc_x += self.kp * zmp_error[0] + self.kd * zmp_vel[0]
        com_acc_y += self.kp * zmp_error[1] + self.kd * zmp_vel[1]

        # Integrate to get desired CoM position
        dt = 0.01  # Assume 100Hz control
        self.com_velocity[0] += com_acc_x * dt
        self.com_velocity[1] += com_acc_y * dt

        desired_com_pos = self.com_position.copy()
        desired_com_pos[0] += self.com_velocity[0] * dt
        desired_com_pos[1] += self.com_velocity[1] * dt

        return desired_com_pos

    def compute_balance_control(self, current_com, current_com_vel, desired_zmp):
        """Compute balance control commands to track desired ZMP"""
        # Calculate current ZMP
        current_zmp = self.calculate_zmp(current_com, self.com_acceleration)

        # Calculate ZMP error
        zmp_error = desired_zmp - current_zmp

        # Calculate CoM correction to achieve desired ZMP
        # Using inverted pendulum model
        com_correction_x = zmp_error[0] + (self.com_height / self.gravity) * self.com_acceleration[0]
        com_correction_y = zmp_error[1] + (self.com_height / self.gravity) * self.com_acceleration[1]

        # Apply feedback control
        com_correction_x += self.kp * zmp_error[0] + self.kd * current_com_vel[0]
        com_correction_y += self.kp * zmp_error[1] + self.kd * current_com_vel[1]

        # Calculate desired CoM acceleration
        desired_com_acc = np.array([
            (com_correction_x - current_com[0]) * (self.gravity / self.com_height),
            (com_correction_y - current_com[1]) * (self.gravity / self.com_height),
            0  # No vertical acceleration
        ])

        # Integrate to get desired CoM velocity and position
        dt = 0.01  # Control timestep
        desired_com_vel = current_com_vel + desired_com_acc * dt
        desired_com_pos = current_com + desired_com_vel * dt

        return desired_com_pos, desired_com_vel, desired_com_acc

    def generate_zmp_trajectory(self, start_zmp, goal_zmp, duration, step_type='double_support'):
        """Generate ZMP trajectory for walking"""
        # Number of steps based on control frequency (100Hz)
        n_steps = int(duration * 100)
        t = np.linspace(0, duration, n_steps)

        if step_type == 'double_support':
            # During double support, ZMP moves from one foot to the other
            zmp_trajectory = np.zeros((n_steps, 2))
            for i in range(n_steps):
                # Cubic interpolation for smooth transition
                progress = min(1.0, t[i] / duration)
                cubic_progress = 3 * progress**2 - 2 * progress**3  # Smooth S-curve

                zmp_trajectory[i] = (1 - cubic_progress) * start_zmp + cubic_progress * goal_zmp

        elif step_type == 'single_support':
            # During single support, ZMP follows a more complex pattern
            # This is a simplified version - real walking has more complex ZMP patterns
            zmp_trajectory = np.zeros((n_steps, 2))
            for i in range(n_steps):
                progress = min(1.0, t[i] / duration)

                # ZMP moves forward but also shifts laterally for balance
                zmp_trajectory[i, 0] = start_zmp[0] + (goal_zmp[0] - start_zmp[0]) * progress
                zmp_trajectory[i, 1] = start_zmp[1] + (goal_zmp[1] - start_zmp[1]) * progress

                # Add small lateral adjustments for balance during single support
                # This creates the typical side-to-side sway during walking
                lateral_adjustment = 0.02 * np.sin(progress * np.pi)  # Peak in middle
                zmp_trajectory[i, 1] += lateral_adjustment

        return zmp_trajectory

    def is_zmp_stable(self, zmp_pos, support_polygon):
        """Check if ZMP is within support polygon"""
        if len(support_polygon) < 3:
            return False

        # Check if ZMP is inside support polygon
        return self.is_point_in_convex_polygon(zmp_pos, support_polygon[:, :2])

    def is_point_in_convex_polygon(self, point, polygon):
        """Check if point is inside convex polygon"""
        # Use cross product method
        n = len(polygon)
        sign = None

        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Calculate cross product to determine which side of edge point is on
            cross_product = ((p2[0] - p1[0]) * (point[1] - p1[1]) -
                           (p2[1] - p1[1]) * (point[0] - p1[0]))

            current_sign = 1 if cross_product >= 0 else -1

            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False  # Outside polygon

        return True  # Inside polygon
```

## Inverted Pendulum Models

### Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model is a simplified model that assumes constant CoM height:

```python
class LinearInvertedPendulumModel:
    """Linear Inverted Pendulum Model for balance control"""

    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(self.gravity / self.com_height)

    def compute_zmp_from_com(self, com_pos, com_acc):
        """Compute ZMP from CoM position and acceleration"""
        zmp_x = com_pos[0] - com_acc[0] / (self.omega**2)
        zmp_y = com_pos[1] - com_acc[1] / (self.omega**2)
        return np.array([zmp_x, zmp_y])

    def compute_com_from_zmp(self, zmp_pos, zmp_vel, zmp_acc):
        """Compute CoM trajectory that achieves desired ZMP"""
        # For LIPM, CoM trajectory can be computed analytically
        # dx/dt = omega^2 * (CoM - ZMP)
        # This gives us: CoM(t) = ZMP(t) + (CoM(0) - ZMP(0)) * exp(omega*t) + ...

        # Simplified implementation using feedback linearization
        # For tracking a ZMP trajectory, we use:
        # CoM_ddot = omega^2 * (CoM - ZMP) + ZMP_ddot

        # Calculate desired CoM acceleration
        com_acc_x = self.omega**2 * (com_pos[0] - zmp_pos[0]) + zmp_acc[0]
        com_acc_y = self.omega**2 * (com_pos[1] - zmp_pos[1]) + zmp_acc[1]

        return np.array([com_acc_x, com_acc_y, 0])

    def plan_com_trajectory(self, start_com, goal_com, zmp_trajectory):
        """Plan CoM trajectory that follows ZMP trajectory"""
        n_steps = len(zmp_trajectory)
        dt = 0.01  # 100Hz control

        com_trajectory = np.zeros((n_steps, 3))
        com_velocity = np.zeros((n_steps, 3))
        com_acceleration = np.zeros((n_steps, 3))

        # Initial conditions
        com_pos = start_com.copy()
        com_vel = np.zeros(3)

        for i in range(n_steps):
            # Calculate desired ZMP for this step
            desired_zmp = zmp_trajectory[i]

            # Calculate ZMP error
            current_zmp = self.compute_zmp_from_com(com_pos, com_acc)

            # Feedback control to track ZMP
            zmp_error = desired_zmp - current_zmp
            feedback_gain = 10.0  # Tunable parameter

            # Calculate control acceleration
            control_acc = feedback_gain * zmp_error

            # Total acceleration (to track ZMP + return to equilibrium)
            total_acc = self.omega**2 * (com_pos[:2] - desired_zmp) + control_acc

            # Update CoM state
            com_acc = np.array([total_acc[0], total_acc[1], 0])  # No vertical acceleration
            com_vel[:2] += com_acc[:2] * dt
            com_pos[:2] += com_vel[:2] * dt

            # Store trajectory
            com_trajectory[i] = com_pos.copy()
            com_velocity[i] = com_vel.copy()
            com_acceleration[i] = com_acc

        return com_trajectory, com_velocity, com_acceleration

class CapturePointController:
    """Capture Point based balance controller"""

    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(self.gravity / self.com_height)

    def calculate_capture_point(self, com_pos, com_vel):
        """Calculate capture point where robot would naturally fall"""
        # Capture point = CoM position + CoM velocity / omega
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega

        return np.array([cp_x, cp_y])

    def should_step(self, capture_point, support_polygon, safety_margin=0.1):
        """Determine if a step is needed to maintain balance"""
        # Check if capture point is outside support polygon with safety margin
        if len(support_polygon) < 3:
            return True  # Always step if insufficient support

        # Expand support polygon by safety margin
        expanded_polygon = self.expand_polygon(support_polygon, safety_margin)

        return not self.is_point_in_convex_polygon(capture_point, expanded_polygon)

    def calculate_step_location(self, capture_point, com_pos, support_polygon):
        """Calculate where to step to capture the capture point"""
        # Find the optimal step location to bring capture point back to stability

        # For simplicity, we'll use the center of support polygon as target
        # In practice, more sophisticated methods would be used
        support_center = np.mean(support_polygon, axis=0)

        # Calculate desired step location to move capture point toward center
        desired_capture = support_center

        # Calculate where to step to achieve desired capture point
        # Using inverted relationship: step_pos = capture_point - CoM_velocity / omega
        # But we want to control CoM velocity through stepping

        # Simplified approach: step in direction opposite to capture point offset
        offset = capture_point - support_center
        step_direction = -offset / np.linalg.norm(offset) if np.linalg.norm(offset) > 0.01 else np.array([0, 1])

        # Step distance based on capture point offset
        step_distance = min(0.3, np.linalg.norm(offset) * 0.8)  # Max 30cm step

        # Calculate step location
        step_location = com_pos[:2] + step_direction * step_distance

        return step_location

    def expand_polygon(self, polygon, margin):
        """Expand polygon by given margin"""
        # Simple expansion by moving vertices outward
        center = np.mean(polygon, axis=0)
        expanded = []

        for vertex in polygon:
            direction = vertex - center
            direction = direction / np.linalg.norm(direction)
            expanded_vertex = vertex + direction * margin
            expanded.append(expanded_vertex)

        return np.array(expanded)

    def is_point_in_convex_polygon(self, point, polygon):
        """Check if point is inside convex polygon"""
        # Use cross product method
        n = len(polygon)
        sign = None

        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Calculate cross product to determine which side of edge point is on
            cross_product = ((p2[0] - p1[0]) * (point[1] - p1[1]) -
                           (p2[1] - p1[1]) * (point[0] - p1[0]))

            current_sign = 1 if cross_product >= 0 else -1

            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False  # Outside polygon

        return True  # Inside polygon

    def compute_step_timing(self, com_pos, com_vel, desired_step_location):
        """Compute optimal timing for taking a step"""
        # Calculate time needed to prepare for step
        # This involves the dynamics of shifting weight to the stance leg

        # Simplified approach: use fixed step timing based on walking speed
        # In practice, this would be computed based on current dynamics
        step_timing = 0.8  # seconds

        return step_timing
```

## Walking Pattern Generation

### Footstep Planning

Generating stable footsteps for humanoid walking:

```python
class FootstepPlanner:
    """Plan stable footsteps for humanoid locomotion"""

    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05):
        self.step_length = step_length  # Forward step distance
        self.step_width = step_width    # Lateral foot separation
        self.step_height = step_height  # Foot lift height
        self.max_turn = 0.3             # Maximum turn per step (radians)

    def plan_straight_walk(self, start_pose, num_steps, step_length=None):
        """Plan footsteps for straight walking"""
        if step_length is None:
            step_length = self.step_length

        footsteps = []
        current_pose = start_pose.copy()

        # Start with left foot in front
        left_support = True

        for step_idx in range(num_steps):
            # Calculate next step position
            step_x = current_pose[0] + step_length * (step_idx + 1)
            step_y = current_pose[1]

            # Alternate between left and right foot
            if step_idx % 2 == 0:  # Right foot step (since left starts as support)
                step_y = current_pose[1] - self.step_width / 2  # Right foot
                foot_name = 'right'
            else:  # Left foot step
                step_y = current_pose[1] + self.step_width / 2  # Left foot
                foot_name = 'left'

            # Calculate step orientation (assuming straight walking)
            step_yaw = current_pose[2]

            # Create step pose
            step_pose = np.array([step_x, step_y, 0.0, 0.0, 0.0, step_yaw])
            footsteps.append({
                'pose': step_pose,
                'foot': foot_name,
                'step_number': step_idx,
                'support_leg': 'left' if left_support else 'right'
            })

            # Alternate support leg
            left_support = not left_support

        return footsteps

    def plan_turning_walk(self, start_pose, turn_angle, num_steps):
        """Plan footsteps for turning motion"""
        footsteps = []
        current_pose = start_pose.copy()

        # Calculate turn per step
        turn_per_step = turn_angle / num_steps

        # Start with left foot as support
        left_support = True

        for step_idx in range(num_steps):
            # Calculate step position and orientation
            # For turning, we follow a circular path
            turn_so_far = turn_per_step * (step_idx + 1)
            radius = self.step_length / (2 * np.sin(turn_per_step / 2)) if turn_per_step != 0 else float('inf')

            if radius != float('inf'):
                # Calculate position on circular arc
                step_x = start_pose[0] + radius * np.sin(turn_so_far)
                step_y = start_pose[1] - radius * (1 - np.cos(turn_so_far))
                step_yaw = start_pose[2] + turn_so_far
            else:
                # Straight line if no turn
                step_x = start_pose[0] + self.step_length * (step_idx + 1) * np.cos(start_pose[2])
                step_y = start_pose[1] + self.step_length * (step_idx + 1) * np.sin(start_pose[2])
                step_yaw = start_pose[2]

            # Alternate feet and adjust lateral position for turn
            if step_idx % 2 == 0:  # Right foot step
                # For right turn, right foot moves more laterally outward
                lateral_offset = -self.step_width / 2
                if turn_angle > 0:  # Right turn
                    lateral_offset *= 1.1  # Slightly wider for stability
            else:  # Left foot step
                # For right turn, left foot moves more laterally inward
                lateral_offset = self.step_width / 2
                if turn_angle > 0:  # Right turn
                    lateral_offset *= 0.9  # Slightly narrower

            # Apply lateral offset in rotated frame
            step_x += lateral_offset * np.cos(step_yaw + np.pi/2)
            step_y += lateral_offset * np.sin(step_yaw + np.pi/2)

            step_pose = np.array([step_x, step_y, 0.0, 0.0, 0.0, step_yaw])
            footsteps.append({
                'pose': step_pose,
                'foot': 'right' if step_idx % 2 == 0 else 'left',
                'step_number': step_idx,
                'support_leg': 'left' if left_support else 'right'
            })

            left_support = not left_support

        return footsteps

    def plan_stepping_in_place(self, start_pose, num_steps):
        """Plan footsteps for stepping in place"""
        footsteps = []
        base_pose = start_pose.copy()

        for step_idx in range(num_steps):
            # Alternate between left and right foot, keeping position roughly the same
            if step_idx % 2 == 0:  # Right foot step
                step_pose = base_pose.copy()
                step_pose[1] -= self.step_width / 2  # Move right foot laterally
            else:  # Left foot step
                step_pose = base_pose.copy()
                step_pose[1] += self.step_width / 2  # Move left foot laterally

            # Add small forward/backward motion for natural stepping
            forward_offset = 0.05 * np.sin(step_idx * np.pi / 2)  # Oscillating forward motion
            step_pose[0] += forward_offset

            footsteps.append({
                'pose': step_pose,
                'foot': 'right' if step_idx % 2 == 0 else 'left',
                'step_number': step_idx,
                'support_leg': 'left' if step_idx % 2 == 1 else 'right'  # Opposite of swing foot
            })

        return footsteps

    def validate_footstep(self, foot_pose, terrain_map=None, obstacle_map=None):
        """Validate that a footstep is safe and stable"""
        x, y, z, roll, pitch, yaw = foot_pose

        # Check terrain validity
        if terrain_map:
            terrain_height = terrain_map.get_height_at(x, y)
            if abs(z - terrain_height) > 0.1:  # Too high or low
                return False, "Footstep at invalid height"

        # Check for obstacles
        if obstacle_map:
            if obstacle_map.is_occupied(x, y, radius=0.1):  # 10cm radius check
                return False, "Footstep collides with obstacle"

        # Check for valid contact surface
        if terrain_map:
            slope = terrain_map.get_slope_at(x, y)
            if slope > 0.3:  # Too steep (16.7 degrees)
                return False, "Footstep on too steep surface"

        # Check if footstep is reachable by robot
        # This would involve inverse kinematics checks in practice
        if abs(x) > 2.0 or abs(y) > 1.0:  # Rough reachability check
            return False, "Footstep not reachable"

        return True, "Footstep is valid"

    def optimize_footsteps(self, footsteps, robot_model, terrain_map=None):
        """Optimize footsteps for stability and reachability"""
        optimized_footsteps = []

        for i, step in enumerate(footsteps):
            # Start with original step
            optimized_pose = step['pose'].copy()

            # Optimize for stability (move toward center of support polygon)
            if i > 0:
                # Consider previous step to maintain balance
                prev_step = footsteps[i-1]['pose']
                # Ensure step is within reasonable distance from previous step
                distance_from_prev = np.linalg.norm(
                    optimized_pose[:2] - prev_step[:2]
                )

                if distance_from_prev > 0.5:  # Too far
                    # Adjust to be closer to reasonable step distance
                    direction = (optimized_pose[:2] - prev_step[:2]) / distance_from_prev
                    optimized_pose[:2] = prev_step[:2] + direction * min(distance_from_prev, 0.4)

            # Validate optimized step
            is_valid, reason = self.validate_footstep(optimized_pose, terrain_map)
            if not is_valid:
                # Try to adjust step to make it valid
                adjusted_pose = self.adjust_footstep_for_validity(optimized_pose, terrain_map)
                if adjusted_pose is not None:
                    optimized_pose = adjusted_pose

            # Create optimized step
            optimized_step = step.copy()
            optimized_step['pose'] = optimized_pose
            optimized_footsteps.append(optimized_step)

        return optimized_footsteps

    def adjust_footstep_for_validity(self, original_pose, terrain_map):
        """Try to adjust a footstep to make it valid"""
        # Try small adjustments around original position
        adjustments = [
            [0, 0],      # Original position
            [0.05, 0],   # Small forward
            [-0.05, 0],  # Small backward
            [0, 0.05],   # Small right
            [0, -0.05],  # Small left
            [0.05, 0.05], # Forward-right diagonal
            [0.05, -0.05], # Forward-left diagonal
        ]

        for dx, dy in adjustments:
            adjusted_pose = original_pose.copy()
            adjusted_pose[0] += dx
            adjusted_pose[1] += dy

            is_valid, reason = self.validate_footstep(adjusted_pose, terrain_map)
            if is_valid:
                return adjusted_pose

        return None  # Could not find valid adjustment
```

### Walking Pattern Generator

Creating smooth walking trajectories:

```python
class WalkingPatternGenerator:
    """Generate complete walking patterns with CoM, ZMP, and foot trajectories"""

    def __init__(self, com_height=0.8, dt=0.01):
        self.com_height = com_height
        self.dt = dt
        self.lipm = LinearInvertedPendulumModel(com_height=com_height)

        # Walking parameters
        self.step_duration = 0.8  # seconds per step
        self.dsp_duration = 0.1   # double support phase duration
        self.ssp_duration = self.step_duration - self.dsp_duration  # single support phase

    def generate_walk_cycle(self, footsteps, walking_speed=0.3):
        """Generate complete walking pattern for given footsteps"""
        # Calculate timing parameters
        steps_per_second = walking_speed / self.step_length if self.step_length > 0 else 0.8
        step_duration = 1.0 / steps_per_second if steps_per_second > 0 else 0.8

        # Total trajectory duration
        total_duration = len(footsteps) * step_duration
        n_samples = int(total_duration / self.dt)

        # Initialize trajectory arrays
        time_array = np.linspace(0, total_duration, n_samples)
        com_trajectory = np.zeros((n_samples, 3))  # x, y, z
        zmp_trajectory = np.zeros((n_samples, 2))  # x, y
        left_foot_trajectory = np.zeros((n_samples, 6))  # x, y, z, roll, pitch, yaw
        right_foot_trajectory = np.zeros((n_samples, 6))

        # Current state
        current_com = np.array([0.0, 0.0, self.com_height])
        current_com_vel = np.zeros(3)
        current_com_acc = np.zeros(3)

        # Generate trajectory step by step
        step_idx = 0
        current_time = 0.0

        for i, t in enumerate(time_array):
            # Determine which step we're in
            current_step_idx = int(t / step_duration)
            if current_step_idx >= len(footsteps):
                current_step_idx = len(footsteps) - 1

            # Determine phase within step (0 to 1)
            time_in_step = t - (current_step_idx * step_duration)
            step_phase = time_in_step / step_duration

            # Determine support phase (double or single support)
            is_double_support = (step_phase < self.dsp_duration/step_duration or
                               step_phase > (1 - self.dsp_duration/step_duration))

            # Calculate current support polygon
            if is_double_support:
                # Both feet in contact
                if current_step_idx > 0:
                    prev_step = footsteps[max(0, current_step_idx-1)]
                    curr_step = footsteps[current_step_idx]
                    support_polygon = self.calculate_double_support_polygon(
                        prev_step['pose'], curr_step['pose'])
                else:
                    # Use first step and a default position
                    support_polygon = self.calculate_single_support_polygon(footsteps[0]['pose'])
            else:
                # Single support - use current stance foot
                stance_foot = self.get_stance_foot(footsteps, current_step_idx, step_phase)
                support_polygon = self.calculate_single_support_polygon(stance_foot)

            # Generate CoM trajectory for this phase
            desired_com, desired_com_vel, desired_com_acc = self.generate_com_trajectory_step(
                current_com, current_com_vel, support_polygon, step_phase, is_double_support)

            # Generate foot trajectories
            left_foot_pos, right_foot_pos = self.generate_foot_trajectories(
                footsteps, current_step_idx, step_phase, current_time)

            # Store in trajectories
            com_trajectory[i] = desired_com
            zmp_trajectory[i] = self.lipm.compute_zmp_from_com(desired_com, desired_com_acc)
            left_foot_trajectory[i] = left_foot_pos
            right_foot_trajectory[i] = right_foot_pos

            # Update state for next iteration
            current_com = desired_com
            current_com_vel = desired_com_vel

        return {
            'time': time_array,
            'com': com_trajectory,
            'zmp': zmp_trajectory,
            'left_foot': left_foot_trajectory,
            'right_foot': right_foot_trajectory,
            'footsteps': footsteps
        }

    def calculate_double_support_polygon(self, prev_step, curr_step):
        """Calculate support polygon during double support phase"""
        # For double support, support polygon is convex hull of both feet
        prev_pos = prev_step['pose'][:2]
        curr_pos = curr_step['pose'][:2]

        # Create polygon from both feet (simplified as rectangle)
        min_x = min(prev_pos[0], curr_pos[0]) - 0.1
        max_x = max(prev_pos[0], curr_pos[0]) + 0.1
        min_y = min(prev_pos[1], curr_pos[1]) - 0.1
        max_y = max(prev_pos[1], curr_pos[1]) + 0.1

        return np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

    def calculate_single_support_polygon(self, stance_foot_pose):
        """Calculate support polygon for single support"""
        # Simplified as rectangle around foot
        x, y = stance_foot_pose[:2]
        return np.array([
            [x - 0.1, y - 0.05],  # front-left of foot
            [x + 0.1, y - 0.05],  # front-right of foot
            [x + 0.1, y + 0.05],  # back-right of foot
            [x - 0.1, y + 0.05]   # back-left of foot
        ])

    def get_stance_foot(self, footsteps, step_idx, step_phase):
        """Determine which foot is stance foot at given phase"""
        # During single support, the stance foot alternates
        # If step_idx is even and phase is in single support, left foot is swing for right stance
        if step_idx % 2 == 0:  # Even steps: right foot moves, left is stance during SSP
            if step_phase < 0.5:  # First half of step
                return footsteps[step_idx-1]['pose'] if step_idx > 0 else footsteps[step_idx]['pose']
            else:  # Second half of step
                return footsteps[step_idx]['pose']
        else:  # Odd steps: left foot moves, right is stance during SSP
            if step_phase < 0.5:
                return footsteps[step_idx-1]['pose'] if step_idx > 0 else footsteps[step_idx]['pose']
            else:
                return footsteps[step_idx]['pose']

    def generate_com_trajectory_step(self, current_com, current_com_vel,
                                   support_polygon, step_phase, is_double_support):
        """Generate CoM trajectory for current step phase"""
        # Use inverted pendulum model to generate CoM trajectory
        # that maintains balance within support polygon

        if is_double_support:
            # During double support, move CoM toward center of support polygon
            support_center = np.mean(support_polygon, axis=0)
            desired_com = current_com.copy()
            desired_com[:2] = current_com[:2] * 0.9 + support_center * 0.1  # Gentle move toward center
        else:
            # During single support, follow inverted pendulum dynamics
            # with reference trajectory that moves CoM appropriately
            desired_com = self.generate_ssp_com_reference(
                current_com, support_polygon, step_phase)

        # Calculate velocity and acceleration based on desired position
        dt = 0.01
        desired_com_vel = (desired_com - current_com) / dt * 0.8 + current_com_vel * 0.2  # Smooth transition
        desired_com_acc = (desired_com_vel - current_com_vel) / dt

        # Maintain constant height
        desired_com[2] = self.com_height

        return desired_com, desired_com_vel, desired_com_acc

    def generate_ssp_com_reference(self, current_com, support_polygon, step_phase):
        """Generate reference CoM trajectory during single support phase"""
        # During single support, CoM typically moves laterally to prepare
        # for next step and forward to maintain momentum

        # Calculate target CoM position based on support polygon
        support_center = np.mean(support_polygon, axis=0)

        # During single support, CoM may oscillate slightly for dynamic balance
        lateral_oscillation = 0.02 * np.sin(step_phase * 2 * np.pi)  # Small lateral movement
        forward_progression = 0.1 * step_phase  # Forward movement with progression

        target_com = current_com.copy()
        target_com[0] = support_center[0] + forward_progression
        target_com[1] = support_center[1] + lateral_oscillation

        return target_com

    def generate_foot_trajectories(self, footsteps, step_idx, step_phase, current_time):
        """Generate foot trajectories for current step"""
        # Calculate foot positions based on step phase
        left_foot = np.zeros(6)  # x, y, z, roll, pitch, yaw
        right_foot = np.zeros(6)

        # Determine which foot is swing foot (moving) vs stance foot (stationary)
        if step_idx % 2 == 0:  # Even steps: right foot swings
            stance_foot = 'left'
            swing_foot = 'right'
        else:  # Odd steps: left foot swings
            stance_foot = 'right'
            swing_foot = 'left'

        # Set stance foot position (stationary during SSP)
        if step_idx > 0:
            if stance_foot == 'left':
                stance_step_idx = step_idx - 1 if step_idx % 2 == 0 else step_idx
            else:
                stance_step_idx = step_idx if step_idx % 2 == 0 else step_idx - 1

            if stance_step_idx < len(footsteps):
                if stance_foot == 'left':
                    left_foot[:3] = footsteps[stance_step_idx]['pose'][:3]
                    left_foot[5] = footsteps[stance_step_idx]['pose'][5]  # yaw
                else:
                    right_foot[:3] = footsteps[stance_step_idx]['pose'][:3]
                    right_foot[5] = footsteps[stance_step_idx]['pose'][5]  # yaw

        # Calculate swing foot trajectory
        if step_idx < len(footsteps):
            target_pose = footsteps[step_idx]['pose']

            if swing_foot == 'left':
                # Generate swing trajectory for left foot
                left_foot = self.generate_swing_trajectory(
                    left_foot, target_pose, step_phase, 'left')
            else:
                # Generate swing trajectory for right foot
                right_foot = self.generate_swing_trajectory(
                    right_foot, target_pose, step_phase, 'right')

        return left_foot, right_foot

    def generate_swing_trajectory(self, current_pose, target_pose, phase, foot_name):
        """Generate smooth swing trajectory for foot"""
        # Use cubic spline interpolation for smooth motion
        if phase < 0.5:  # First half: lift and move forward
            # Calculate lift height based on phase
            lift_factor = np.sin(phase * 2 * np.pi)  # Sinusoidal lift
            z_lift = self.step_height * lift_factor

            # Calculate forward progression
            x_progress = current_pose[0] + (target_pose[0] - current_pose[0]) * phase * 2
            y_progress = current_pose[1] + (target_pose[1] - current_pose[1]) * phase * 2
        else:  # Second half: lower to ground
            # Complete forward progression
            x_progress = target_pose[0]
            y_progress = target_pose[1]

            # Lower foot based on phase
            lower_factor = np.sin((phase - 0.5) * 2 * np.pi)  # Sinusoidal lower
            z_lift = self.step_height * (1 - lower_factor)

        # Create smooth trajectory
        swing_pose = target_pose.copy()
        swing_pose[0] = x_progress
        swing_pose[1] = y_progress
        swing_pose[2] = z_lift  # Apply lift

        return swing_pose

    def generate_zmp_trajectory_from_footsteps(self, footsteps):
        """Generate ZMP trajectory based on footsteps"""
        # During double support, ZMP moves from one foot to the other
        # During single support, ZMP stays within the stance foot's support region

        total_duration = len(footsteps) * self.step_duration
        n_samples = int(total_duration / self.dt)

        zmp_trajectory = np.zeros((n_samples, 2))
        time_array = np.linspace(0, total_duration, n_samples)

        for i, t in enumerate(time_array):
            step_idx = int(t / self.step_duration)
            if step_idx >= len(footsteps):
                step_idx = len(footsteps) - 1

            time_in_step = t - (step_idx * self.step_duration)
            step_phase = time_in_step / self.step_duration

            if step_phase < self.dsp_duration / self.step_duration:
                # First double support phase - ZMP moves from previous foot to current
                if step_idx > 0:
                    prev_step = footsteps[step_idx - 1]['pose']
                    curr_step = footsteps[step_idx]['pose']
                    progress = step_phase / (self.dsp_duration / self.step_duration)
                    zmp_trajectory[i] = (1 - progress) * prev_step[:2] + progress * curr_step[:2]
                else:
                    zmp_trajectory[i] = footsteps[step_idx]['pose'][:2]
            elif step_phase > (1 - self.dsp_duration / self.step_duration):
                # Second double support phase - ZMP moves to next foot
                if step_idx < len(footsteps) - 1:
                    curr_step = footsteps[step_idx]['pose']
                    next_step = footsteps[step_idx + 1]['pose']
                    progress = (step_phase - (1 - self.dsp_duration / self.step_duration)) / (self.dsp_duration / self.step_duration)
                    zmp_trajectory[i] = (1 - progress) * curr_step[:2] + progress * next_step[:2]
                else:
                    zmp_trajectory[i] = footsteps[step_idx]['pose'][:2]
            else:
                # Single support phase - ZMP stays in center of stance foot
                # with small oscillations for dynamic balance
                ssp_phase = (step_phase - self.dsp_duration / self.step_duration) / (1 - 2 * self.dsp_duration / self.step_duration)
                stance_pos = footsteps[step_idx]['pose'][:2]

                # Small oscillation for dynamic balance
                oscillation = 0.01 * np.array([np.sin(ssp_phase * 2 * np.pi), np.cos(ssp_phase * 2 * np.pi)])
                zmp_trajectory[i] = stance_pos + oscillation

        return zmp_trajectory
```

## Advanced Balance Control Techniques

### Model Predictive Control (MPC) for Balance

Model Predictive Control is a powerful technique for balance control that optimizes over a prediction horizon:

```python
import cvxpy as cp
import numpy as np

class ModelPredictiveBalanceController:
    """Model Predictive Controller for humanoid balance"""

    def __init__(self, com_height=0.8, prediction_horizon=20, dt=0.01):
        self.com_height = com_height
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.gravity = 9.81

        # MPC weights
        self.Q = np.diag([10.0, 10.0, 1.0])  # State cost (x, y, z)
        self.R = np.diag([0.1, 0.1])         # Control cost (ZMP x, y)
        self.P = np.diag([50.0, 50.0, 5.0]) # Terminal cost

        # System matrices for LIPM
        self.A, self.B = self.linear_inverted_pendulum_matrices()

    def linear_inverted_pendulum_matrices(self):
        """Calculate system matrices for Linear Inverted Pendulum Model"""
        omega = np.sqrt(self.gravity / self.com_height)

        # State: [x, y, z, x_dot, y_dot, z_dot]
        # For LIPM, we typically use [x, x_dot] and [y, y_dot] separately
        # Here we'll use [com_x, com_y, com_x_dot, com_y_dot] for simplicity

        A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        B = np.array([
            [self.dt**2 * omega**2 / 2, 0],
            [0, self.dt**2 * omega**2 / 2],
            [self.dt * omega**2, 0],
            [0, self.dt * omega**2]
        ])

        return A, B

    def solve_balance_mpc(self, current_state, reference_trajectory,
                         support_polygons, initial_zmp=None):
        """
        Solve MPC problem for balance control

        Args:
            current_state: Current CoM state [x, y, x_dot, y_dot]
            reference_trajectory: Reference CoM trajectory over prediction horizon
            support_polygons: Support polygons for each time step
            initial_zmp: Initial ZMP if available

        Returns:
            optimal_zmp_sequence: Sequence of optimal ZMP commands
        """
        # Define optimization variables
        # State variables for entire horizon
        X = cp.Variable((4, self.prediction_horizon + 1))  # [x, y, x_dot, y_dot] for each time step
        U = cp.Variable((2, self.prediction_horizon))      # [zmp_x, zmp_y] for each control step

        # Cost function
        cost = 0

        # Running cost: penalize deviation from reference and control effort
        for k in range(self.prediction_horizon):
            state_error = X[:, k] - reference_trajectory[k]
            cost += cp.quad_form(state_error, self.Q)
            cost += cp.quad_form(U[:, k], self.R)

        # Terminal cost: penalize final state error
        final_error = X[:, self.prediction_horizon] - reference_trajectory[-1]
        cost += cp.quad_form(final_error, self.P)

        # Constraints
        constraints = []

        # Initial state constraint
        constraints.append(X[:, 0] == current_state)

        # System dynamics constraints
        for k in range(self.prediction_horizon):
            constraints.append(X[:, k+1] == self.A @ X[:, k] + self.B @ U[:, k])

        # ZMP constraints (must be within support polygon at each time step)
        for k in range(self.prediction_horizon):
            if k < len(support_polygons):
                # For each time step, ZMP must be within the support polygon
                # We'll approximate this with bounding box constraints
                poly = support_polygons[k]
                min_x = np.min(poly[:, 0])
                max_x = np.max(poly[:, 0])
                min_y = np.min(poly[:, 1])
                max_y = np.max(poly[:, 1])

                constraints.append(U[0, k] >= min_x - 0.05)  # Add small safety margin
                constraints.append(U[0, k] <= max_x + 0.05)
                constraints.append(U[1, k] >= min_y - 0.05)
                constraints.append(U[1, k] <= max_y + 0.05)

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status == cp.OPTIMAL:
                # Extract optimal solution
                optimal_zmp_sequence = U.value.T  # Shape: (horizon, 2)
                return optimal_zmp_sequence
            else:
                # If optimization fails, return a safe default
                print(f"MPC optimization failed with status: {problem.status}")
                # Return ZMP at center of current support polygon
                if len(support_polygons) > 0:
                    current_poly = support_polygons[0]
                    center = np.mean(current_poly, axis=0)
                    return np.tile(center, (self.prediction_horizon, 1))
                else:
                    return np.zeros((self.prediction_horizon, 2))

        except Exception as e:
            print(f"MPC solver error: {e}")
            # Return safe default
            return np.zeros((self.prediction_horizon, 2))

    def update_reference_trajectory(self, current_state, goal_state, disturbance_prediction=None):
        """Update reference trajectory based on current state and goals"""
        reference_trajectory = np.zeros((self.prediction_horizon, 4))

        for k in range(self.prediction_horizon):
            # Simple linear interpolation toward goal
            alpha = k / self.prediction_horizon
            reference_trajectory[k, :2] = (1 - alpha) * current_state[:2] + alpha * goal_state[:2]
            reference_trajectory[k, 2:] = (1 - alpha) * current_state[2:] + alpha * goal_state[2:]

            # Add compensation for predicted disturbances if available
            if disturbance_prediction is not None:
                reference_trajectory[k, :2] += disturbance_prediction[k, :2] * 0.1  # Scaling factor

        return reference_trajectory

    def compute_balance_control_step(self, current_com, current_com_vel,
                                   desired_com_trajectory, support_polygons):
        """Compute single step of balance control using MPC"""
        # Construct current state vector
        current_state = np.concatenate([current_com[:2], current_com_vel[:2]])  # [x, y, x_dot, y_dot]

        # Get reference trajectory for prediction horizon
        ref_trajectory = self.update_reference_trajectory(
            current_state, desired_com_trajectory[0] if len(desired_com_trajectory) > 0 else current_state)

        # Solve MPC problem
        optimal_zmp_sequence = self.solve_balance_mpc(
            current_state, ref_trajectory, support_polygons)

        # Return first control action
        if optimal_zmp_sequence is not None and len(optimal_zmp_sequence) > 0:
            return optimal_zmp_sequence[0]  # Return first ZMP command
        else:
            # If MPC fails, return current ZMP
            return self.lipm.compute_zmp_from_com(current_com, np.zeros(3))[:2]

class WholeBodyBalanceController:
    """Whole-body balance controller that coordinates CoM and joint control"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.com_controller = ZMPController(com_height=0.8)
        self.capture_point_controller = CapturePointController(com_height=0.8)
        self.mpc_controller = ModelPredictiveBalanceController(com_height=0.8)

        # Joint impedance parameters
        self.impedance_gains = {
            'stiffness': 500.0,  # N/m for position control
            'damping': 20.0,     # Ns/m for velocity damping
            'mass': 5.0          # kg equivalent for acceleration
        }

    def compute_whole_body_balance_control(self, current_state, sensor_data):
        """
        Compute whole-body balance control considering CoM, ZMP, and joint control

        Args:
            current_state: Current robot state (joint positions, velocities)
            sensor_data: Sensor data (IMU, force/torque, joint sensors)

        Returns:
            joint_commands: Joint position/velocity/torque commands
        """
        # Extract relevant state information
        com_pos = self.robot_model.calculate_com_position(current_state['joint_positions'])
        com_vel = self.robot_model.calculate_com_velocity(current_state['joint_positions'],
                                                         current_state['joint_velocities'])
        imu_data = sensor_data.get('imu', {})
        ft_sensors = sensor_data.get('force_torque', {})

        # Calculate ZMP from force/torque sensors if available
        if 'left_ankle' in ft_sensors and 'right_ankle' in ft_sensors:
            zmp_meas = self.calculate_measured_zmp(ft_sensors)
        else:
            # Estimate ZMP from IMU and kinematics
            zmp_meas = self.estimate_zmp_from_imu(imu_data, com_pos)

        # Determine support foot and polygon
        support_polygon = self.calculate_support_polygon(sensor_data)

        # Decide on control strategy based on stability
        capture_point = self.capture_point_controller.calculate_capture_point(com_pos, com_vel)
        should_step = self.capture_point_controller.should_step(capture_point, support_polygon)

        if should_step:
            # Execute stepping strategy
            step_location = self.capture_point_controller.calculate_step_location(
                capture_point, com_pos, support_polygon)
            control_commands = self.compute_stepping_strategy(
                current_state, step_location, support_polygon)
        else:
            # Use balance control without stepping
            control_commands = self.compute_balance_without_stepping(
                current_state, com_pos, com_vel, zmp_meas, support_polygon)

        return control_commands

    def calculate_measured_zmp(self, ft_sensors):
        """Calculate ZMP from force/torque sensor measurements"""
        # ZMP_x = -tau_y / F_z, ZMP_y = tau_x / F_z
        # where tau is moment and F is force

        total_force = np.array([0.0, 0.0, 0.0])
        total_moment = np.array([0.0, 0.0, 0.0])

        # Sum forces and moments from both feet
        for foot, ft_data in ft_sensors.items():
            if 'force' in ft_data and 'moment' in ft_data:
                force = np.array(ft_data['force'])
                moment = np.array(ft_data['moment'])

                total_force += force
                total_moment += moment

        # Calculate ZMP (avoid division by zero)
        if abs(total_force[2]) > 1.0:  # Minimum force threshold
            zmp_x = -total_moment[1] / total_force[2]
            zmp_y = total_moment[0] / total_force[2]
            return np.array([zmp_x, zmp_y])
        else:
            # Return current CoM position as estimate if no reliable force measurement
            return np.array([0.0, 0.0])

    def estimate_zmp_from_imu(self, imu_data, com_pos):
        """Estimate ZMP from IMU data and CoM position"""
        if 'linear_acceleration' in imu_data and 'angular_velocity' in imu_data:
            # Simplified ZMP estimation using IMU data
            # This is an approximation - real ZMP requires force plate or F/T sensors
            linear_acc = np.array(imu_data['linear_acceleration'])
            angular_vel = np.array(imu_data['angular_velocity'])

            # Estimate ZMP using inverted pendulum model
            estimated_zmp_x = com_pos[0] - linear_acc[0] / self.com_controller.omega**2
            estimated_zmp_y = com_pos[1] - linear_acc[1] / self.com_controller.omega**2

            return np.array([estimated_zmp_x, estimated_zmp_y])
        else:
            return com_pos[:2]  # Use CoM as ZMP estimate

    def calculate_support_polygon(self, sensor_data):
        """Calculate support polygon from contact sensor data"""
        contact_points = []

        # Get contact information from force/torque sensors or contact sensors
        if 'contact_sensors' in sensor_data:
            for link_name, contact_info in sensor_data['contact_sensors'].items():
                if contact_info.get('contact', False):
                    # Get position of contact link
                    link_pos = self.robot_model.get_link_position(link_name)
                    contact_points.append(link_pos[:2])  # Use x,y coordinates
        else:
            # Fallback: use foot positions if no contact sensors
            for foot_link in ['left_foot', 'right_foot']:
                try:
                    foot_pos = self.robot_model.get_link_position(foot_link)
                    contact_points.append(foot_pos[:2])
                except:
                    continue

        if len(contact_points) == 0:
            return np.array([])

        if len(contact_points) == 1:
            # Single point support
            return np.array([contact_points[0]])
        elif len(contact_points) == 2:
            # Two points - create rectangle around them
            p1, p2 = contact_points
            center = (p1 + p2) / 2
            direction = p2 - p1
            perp_dir = np.array([-direction[1], direction[0]])
            perp_dir = perp_dir / np.linalg.norm(perp_dir) if np.linalg.norm(perp_dir) > 0 else np.array([1, 0])

            width = 0.1  # 10cm width
            offset = perp_dir * width / 2

            return np.array([p1 + offset, p2 + offset, p2 - offset, p1 - offset])
        else:
            # Multiple points - use convex hull
            from scipy.spatial import ConvexHull
            points_2d = np.array(contact_points)
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]

    def compute_stepping_strategy(self, current_state, step_location, support_polygon):
        """Compute control commands for executing a step"""
        # This is a simplified stepping strategy
        # In practice, this would involve complex whole-body motion planning

        # Plan step trajectory
        step_trajectory = self.plan_step_trajectory(step_location, current_state)

        # Generate joint commands to follow step trajectory
        joint_commands = self.generate_step_joint_commands(
            current_state, step_trajectory)

        return joint_commands

    def plan_step_trajectory(self, target_step_location, current_state):
        """Plan trajectory for stepping motion"""
        # Calculate swing foot trajectory to reach target location
        # This would involve complex whole-body planning in practice

        # For now, return a simplified plan
        current_left_foot = self.robot_model.forward_kinematics_for_link(
            current_state['joint_positions'], 'left_foot')
        current_right_foot = self.robot_model.forward_kinematics_for_link(
            current_state['joint_positions'], 'right_foot')

        # Determine which foot is swing foot (the one not in current support polygon)
        # This is a simplified determination
        stance_foot = 'right'  # Assume right foot is stance for this example
        swing_foot_pos = current_left_foot if stance_foot == 'right' else current_right_foot

        # Plan trajectory from current swing foot position to target step location
        step_trajectory = {
            'swing_foot_initial': swing_foot_pos[:3],
            'swing_foot_target': np.array([target_step_location[0], target_step_location[1], 0.0]),  # Assume z=0 for ground
            'swing_foot_trajectory': self.generate_swing_foot_trajectory(swing_foot_pos[:3], target_step_location)
        }

        return step_trajectory

    def generate_swing_foot_trajectory(self, start_pos, target_pos):
        """Generate smooth trajectory for swing foot"""
        # Use cubic spline for smooth foot trajectory
        # Rise to appropriate height, move horizontally, then descend

        # Calculate intermediate points
        mid_pos = (start_pos + target_pos) / 2
        mid_pos[2] = max(start_pos[2], target_pos[2]) + 0.1  # Lift foot 10cm above highest point

        # Generate trajectory points
        trajectory_points = []
        n_points = 20

        for i in range(n_points + 1):
            t = i / n_points  # Parameter from 0 to 1

            # Use cubic interpolation for smooth motion
            if t < 0.5:
                # First half: lift and move forward
                t_half = t * 2
                x = start_pos[0] + (mid_pos[0] - start_pos[0]) * t_half
                y = start_pos[1] + (mid_pos[1] - start_pos[1]) * t_half
                z = start_pos[2] + (mid_pos[2] - start_pos[2]) * (3*t_half**2 - 2*t_half**3)  # Smooth lift
            else:
                # Second half: descend to target
                t_half = (t - 0.5) * 2
                x = mid_pos[0] + (target_pos[0] - mid_pos[0]) * t_half
                y = mid_pos[1] + (target_pos[1] - mid_pos[1]) * t_half
                z = mid_pos[2] + (target_pos[2] - mid_pos[2]) * (3*t_half**2 - 2*t_half**3)  # Smooth descent

            trajectory_points.append(np.array([x, y, z]))

        return np.array(trajectory_points)

    def compute_balance_without_stepping(self, current_state, com_pos, com_vel, zmp_meas, support_polygon):
        """Compute balance control without stepping"""
        # Use MPC-based balance control
        current_com_state = np.concatenate([com_pos[:2], com_vel[:2]])

        # Create simple reference trajectory (try to maintain current CoM position)
        ref_trajectory = np.tile(current_com_state, (self.mpc_controller.prediction_horizon, 1))

        # Calculate support polygons for prediction horizon
        support_polygons = [support_polygon for _ in range(self.mpc_controller.prediction_horizon)]

        # Solve MPC problem
        optimal_zmp_sequence = self.mpc_controller.solve_balance_mpc(
            current_com_state, ref_trajectory, support_polygons)

        if optimal_zmp_sequence is not None and len(optimal_zmp_sequence) > 0:
            desired_zmp = optimal_zmp_sequence[0]
        else:
            desired_zmp = zmp_meas  # Fallback to measured ZMP

        # Use whole-body inverse kinematics to achieve desired CoM and ZMP
        joint_commands = self.compute_whole_body_control_for_balance(
            current_state, desired_zmp, com_pos)

        return joint_commands

    def compute_whole_body_control_for_balance(self, current_state, desired_zmp, desired_com):
        """Compute whole-body control to achieve desired ZMP and CoM"""
        # This would use advanced whole-body control techniques like:
        # - Operational Space Control
        # - Task-Priority Inverse Kinematics
        # - Hierarchical Optimization

        # Simplified implementation using weighted least squares IK
        current_joints = current_state['joint_positions']

        # Define multiple tasks with priorities:
        # 1. High priority: Maintain balance (ZMP control)
        # 2. Medium priority: Achieve desired CoM position
        # 3. Low priority: Maintain joint center positions

        # Calculate Jacobians for different tasks
        com_jacobian = self.robot_model.com_jacobian(current_joints)
        zmp_jacobian = self.calculate_zmp_jacobian(current_joints)
        stance_foot_jacobians = {
            'left': self.robot_model.jacobian_for_link(current_joints, 'left_foot'),
            'right': self.robot_model.jacobian_for_link(current_joints, 'right_foot')
        }

        # Define task errors
        current_com = self.robot_model.calculate_com_position(current_joints)
        current_zmp = self.calculate_measured_zmp_from_model(current_joints)

        com_error = desired_com[:2] - current_com[:2]
        zmp_error = desired_zmp - current_zmp

        # Use hierarchical inverse kinematics
        # (This is a simplified implementation - real systems use more sophisticated approaches)
        joint_delta = self.hierarchical_ik_solve(
            current_joints,
            {'com': (com_error, com_jacobian, 1.0),
             'zmp': (zmp_error, zmp_jacobian, 1.0)},
            self.robot_model.get_joint_limits())

        # Apply joint updates
        updated_joints = current_joints + joint_delta * 0.01  # Small step size

        # Ensure joint limits are satisfied
        joint_limits = self.robot_model.get_joint_limits()
        for i, (jnt_name, limits) in enumerate(joint_limits.items()):
            updated_joints[i] = np.clip(updated_joints[i], limits['min'], limits['max'])

        return updated_joints

    def calculate_zmp_jacobian(self, joint_positions):
        """Calculate Jacobian that relates joint velocities to ZMP velocities"""
        # The ZMP jacobian is more complex and depends on the current contact configuration
        # This is a simplified approximation

        # For a more accurate implementation, we would need to consider:
        # - Current contact points and their Jacobians
        # - Force distribution among contacts
        # - The relationship between joint torques and ground reaction forces

        # Use CoM Jacobian as approximation (valid for simple cases)
        return self.robot_model.com_jacobian(joint_positions)[:2, :]  # Only x,y components

    def hierarchical_ik_solve(self, current_joints, tasks, joint_limits):
        """Solve hierarchical inverse kinematics problem"""
        # Implementation of weighted least squares hierarchical IK
        # This is a simplified version - in practice, more sophisticated solvers are used

        n_joints = len(current_joints)
        joint_delta = np.zeros(n_joints)

        # Sort tasks by priority (would be provided in tasks dict)
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1][2], reverse=True)  # Priority is 3rd element

        for task_name, (error, jacobian, priority) in sorted_tasks:
            # Solve for this task while considering previous tasks
            if len(error) == 2:  # 2D task (like ZMP)
                error_3d = np.concatenate([error, [0]])  # Extend to 3D
            else:
                error_3d = error

            # Use damped least squares to solve for joint updates
            damping = 0.01
            j_jt = jacobian @ jacobian.T
            reg_matrix = damping * np.eye(len(error_3d))

            try:
                delta_theta = np.linalg.solve(j_jt + reg_matrix, jacobian @ error_3d)
                joint_delta += delta_theta
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudoinverse
                joint_delta += np.linalg.pinv(jacobian) @ error_3d

        return joint_delta

    def calculate_measured_zmp_from_model(self, joint_positions):
        """Calculate ZMP using robot model (simplified)"""
        # This would normally require force/torque sensors
        # For simulation, we can estimate based on contact forces
        # This is a placeholder implementation
        com_pos = self.robot_model.calculate_com_position(joint_positions)
        return com_pos[:2]  # Simplified - return CoM as ZMP estimate
```

## Sensor Fusion for Enhanced Balance

### Multi-Sensor Integration

Combining data from multiple sensors to improve balance control:

```python
class SensorFusionBalancer:
    """Fusion of multiple sensors for enhanced balance control"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

        # Sensor fusion parameters
        self.imu_weight = 0.7
        self.ft_weight = 0.3
        self.kalman_process_noise = 0.1
        self.kalman_measurement_noise = 0.05

        # State estimation (Extended Kalman Filter for nonlinear system)
        self.state_estimate = np.zeros(6)  # [com_x, com_y, com_z, com_dx, com_dy, com_dz]
        self.covariance = np.eye(6) * 0.1

    def fuse_sensor_data(self, imu_data, ft_data, joint_data):
        """Fuse data from multiple sensors to estimate robot state"""
        # Predict step (use model to predict next state)
        dt = 0.01  # Control timestep
        self.predict_state(dt)

        # Update step (incorporate sensor measurements)
        measurement = self.construct_measurement(imu_data, ft_data, joint_data)
        self.update_state(measurement)

        return self.state_estimate

    def predict_state(self, dt):
        """Predict next state using system model"""
        # Use simple dynamics model: x_dot = Ax + Bu
        # For balance, we can use inverted pendulum dynamics as approximation

        # State transition matrix (simplified inverted pendulum)
        omega = np.sqrt(9.81 / 0.8)  # sqrt(g/com_height)
        A = np.array([
            [1, 0, 0, dt, 0, 0],      # x = x + dx*dt
            [0, 1, 0, 0, dt, 0],      # y = y + dy*dt
            [0, 0, 1, 0, 0, dt],      # z = z + dz*dt
            [omega**2*dt, 0, 0, 1, 0, 0],  # dx = dx + omega^2*x*dt
            [0, omega**2*dt, 0, 0, 1, 0],  # dy = dy + omega^2*y*dt
            [0, 0, 0, 0, 0, 1]        # dz = dz (no change in z velocity assumed)
        ])

        # Process noise covariance
        Q = np.eye(6) * self.kalman_process_noise

        # Predict state and covariance
        self.state_estimate = A @ self.state_estimate
        self.covariance = A @ self.covariance @ A.T + Q

    def construct_measurement(self, imu_data, ft_data, joint_data):
        """Construct measurement vector from sensor data"""
        # Extract relevant information from sensors
        measurements = []

        # IMU measurements (orientation and angular velocity)
        if 'orientation' in imu_data:
            # Convert orientation to roll/pitch for balance estimation
            quat = np.array([imu_data['orientation'].w,
                           imu_data['orientation'].x,
                           imu_data['orientation'].y,
                           imu_data['orientation'].z])
            rotation = R.from_quat(quat)
            euler = rotation.as_euler('xyz')
            measurements.extend([euler[0], euler[1]])  # roll, pitch

        # Accelerometer measurements (for CoM acceleration estimation)
        if 'linear_acceleration' in imu_data:
            measurements.extend([
                imu_data['linear_acceleration'].x,
                imu_data['linear_acceleration'].y,
                imu_data['linear_acceleration'].z
            ])

        # Force/torque measurements (for ZMP estimation)
        if 'left_ankle' in ft_data and 'right_ankle' in ft_data:
            # Calculate ZMP from F/T sensors
            left_ft = ft_data['left_ankle']
            right_ft = ft_data['right_ankle']

            total_force = np.array(left_ft['force']) + np.array(right_ft['force'])
            total_moment = np.array(left_ft['moment']) + np.array(right_ft['moment'])

            if abs(total_force[2]) > 0.1:  # Check for sufficient normal force
                zmp_x = -total_moment[1] / total_force[2]
                zmp_y = total_moment[0] / total_force[2]
                measurements.extend([zmp_x, zmp_y])

        # Joint position measurements (for CoM calculation)
        if 'positions' in joint_data:
            com_pos = self.robot_model.calculate_com_position(joint_data['positions'])
            measurements.extend(com_pos[:2])  # x, y of CoM

        return np.array(measurements)

    def update_state(self, measurement):
        """Update state estimate with new measurement"""
        # Measurement matrix (relates state to measurements)
        # This is simplified - in practice, this would be more complex
        H = np.zeros((len(measurement), 6))
        H[0, 0] = 1  # CoM x measurement
        H[1, 1] = 1  # CoM y measurement
        if len(measurement) >= 4:
            H[2, 3] = 1  # CoM x velocity measurement
            H[3, 4] = 1  # CoM y velocity measurement

        # Measurement noise covariance
        R = np.eye(len(measurement)) * self.kalman_measurement_noise

        # Kalman gain
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Innovation (difference between measurement and prediction)
        predicted_measurement = H @ self.state_estimate
        innovation = measurement[:len(predicted_measurement)] - predicted_measurement

        # Update state estimate and covariance
        self.state_estimate = self.state_estimate + K @ innovation
        self.covariance = (np.eye(6) - K @ H) @ self.covariance

    def get_balance_state(self):
        """Get current balance state estimate"""
        return {
            'com_position': self.state_estimate[:3],
            'com_velocity': self.state_estimate[3:],
            'stability_margin': self.calculate_stability_margin(),
            'capture_point': self.estimate_capture_point()
        }

    def calculate_stability_margin(self):
        """Calculate current stability margin"""
        com_pos = self.state_estimate[:2]
        # This would use current support polygon from contact sensors
        # For now, return a placeholder
        return 0.1  # 10cm margin

    def estimate_capture_point(self):
        """Estimate current capture point"""
        com_pos = self.state_estimate[:2]
        com_vel = self.state_estimate[3:5]
        omega = np.sqrt(9.81 / 0.8)

        capture_point_x = com_pos[0] + com_vel[0] / omega
        capture_point_y = com_pos[1] + com_vel[1] / omega

        return np.array([capture_point_x, capture_point_y])
```

## Practical Implementation Tips

### Performance Considerations

For real-time balance control, consider these performance optimizations:

```python
class OptimizedBalanceController:
    """Optimized balance controller for real-time performance"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.dt = 0.01  # 100Hz control rate

        # Pre-allocate arrays to avoid allocation during control
        self.com_jacobian_cache = {}
        self.mass_matrix_cache = {}
        self.state_prediction = np.zeros(4)  # [com_x, com_y, com_dx, com_dy]

        # Use simplified models for real-time computation
        self.use_simplified_dynamics = True

    def compute_balance_control_fast(self, current_state, sensor_data):
        """Fast balance control computation for real-time use"""
        start_time = time.time()

        # Use cached kinematics when possible
        joint_positions = current_state['joint_positions']
        joint_hash = self.hash_joints(joint_positions)

        if joint_hash not in self.com_jacobian_cache:
            self.com_jacobian_cache[joint_hash] = self.robot_model.com_jacobian(joint_positions)
            # Limit cache size to prevent memory issues
            if len(self.com_jacobian_cache) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self.com_jacobian_cache))
                del self.com_jacobian_cache[oldest_key]

        com_jac = self.com_jacobian_cache[joint_hash]

        # Calculate CoM state
        com_pos = self.robot_model.calculate_com_position(joint_positions)
        com_vel = self.calculate_com_velocity_cached(joint_positions, current_state['joint_velocities'])

        # Simplified ZMP calculation
        zmp_est = self.estimate_zmp_simple(com_pos, com_vel)

        # Fast balance correction using pre-computed gains
        balance_correction = self.apply_balance_gains(com_pos, com_vel, zmp_est)

        # Apply joint limits and constraints
        corrected_joints = self.apply_joint_constraints(
            joint_positions, balance_correction, current_state['joint_limits'])

        computation_time = time.time() - start_time
        if computation_time > 0.008:  # Warning if taking more than 8ms (target: 10ms)
            print(f"Warning: Balance control took {computation_time*1000:.1f}ms")

        return corrected_joints

    def hash_joints(self, positions, precision=3):
        """Create hash for joint positions for caching"""
        rounded_positions = tuple(np.round(positions, decimals=precision))
        return hash(rounded_positions)

    def calculate_com_velocity_cached(self, joint_positions, joint_velocities):
        """Calculate CoM velocity using cached Jacobian"""
        joint_hash = self.hash_joints(joint_positions)
        if joint_hash in self.com_jacobian_cache:
            com_jac = self.com_jacobian_cache[joint_hash]
            com_vel = com_jac @ joint_velocities
            return com_vel[:3]  # Only return x, y, z components
        else:
            # Fallback to full calculation
            return self.robot_model.calculate_com_velocity(joint_positions, joint_velocities)

    def estimate_zmp_simple(self, com_pos, com_vel):
        """Fast ZMP estimation using simplified model"""
        # Use linear inverted pendulum model: ZMP = CoM - CoM_ddot/omega^2
        # But estimate CoM_ddot from velocity change
        if hasattr(self, 'prev_com_vel'):
            dt = 0.01
            com_acc = (com_vel - self.prev_com_vel) / dt
        else:
            com_acc = np.zeros(3)

        self.prev_com_vel = com_vel.copy()

        omega = np.sqrt(9.81 / 0.8)  # Fixed CoM height assumption
        zmp_x = com_pos[0] - com_acc[0] / (omega**2)
        zmp_y = com_pos[1] - com_acc[1] / (omega**2)

        return np.array([zmp_x, zmp_y])

    def apply_balance_gains(self, com_pos, com_vel, zmp_est):
        """Apply pre-computed balance control gains"""
        # Simple PD control on CoM position error
        desired_com = np.array([0.0, 0.0, 0.8])  # Desired CoM position (above support foot)

        com_error = desired_com[:2] - com_pos[:2]
        vel_error = -com_vel[:2]  # Negative because we want to damp velocity

        kp = 50.0  # Position gain
        kd = 10.0  # Velocity gain (damping)

        balance_control = kp * com_error + kd * vel_error

        # Convert to joint space using transpose Jacobian method
        joint_hash = self.hash_joints(self.current_joint_positions)
        com_jac = self.com_jacobian_cache.get(joint_hash,
                                            self.robot_model.com_jacobian(self.current_joint_positions))

        # Use transpose of CoM Jacobian for control allocation
        joint_correction = com_jac[:2, :].T @ balance_control  # Only x,y components

        return joint_correction

    def apply_joint_constraints(self, current_joints, joint_correction, joint_limits):
        """Apply joint limits and constraints efficiently"""
        corrected_joints = current_joints + joint_correction

        # Apply joint limits with soft saturation
        for i, (name, limits) in enumerate(joint_limits.items()):
            if corrected_joints[i] < limits['min']:
                # Apply soft limit with damping near boundaries
                overage = limits['min'] - corrected_joints[i]
                corrected_joints[i] = limits['min'] + 0.1 * overage
            elif corrected_joints[i] > limits['max']:
                overage = corrected_joints[i] - limits['max']
                corrected_joints[i] = limits['max'] - 0.1 * overage

        return corrected_joints

class AdaptiveBalanceController:
    """Balance controller that adapts to changing conditions"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.base_gains = {'kp': 100.0, 'kd': 20.0, 'ki': 1.0}
        self.adaptive_gains = self.base_gains.copy()
        self.error_integral = np.zeros(2)  # x, y errors
        self.previous_error = np.zeros(2)

        # Adaptation parameters
        self.adaptation_rate = 0.01
        self.stability_threshold = 0.05

    def compute_adaptive_control(self, current_state, sensor_data):
        """Compute adaptive balance control based on performance"""
        # Calculate current state errors
        com_pos = self.robot_model.calculate_com_position(current_state['joint_positions'])
        com_vel = self.robot_model.calculate_com_velocity(
            current_state['joint_positions'], current_state['joint_velocities'])

        # Desired CoM position (over support polygon center)
        desired_com = self.calculate_desired_com_position(sensor_data)
        error = desired_com[:2] - com_pos[:2]

        # Update error integral for adaptive control
        self.error_integral += error * 0.01  # dt = 0.01s
        error_derivative = (error - self.previous_error) / 0.01

        # Calculate PID control with current gains
        control_output = (self.adaptive_gains['kp'] * error +
                         self.adaptive_gains['ki'] * self.error_integral +
                         self.adaptive_gains['kd'] * error_derivative)

        # Adapt gains based on stability
        stability_measure = self.assess_stability(error, error_derivative)
        self.adapt_gains(stability_measure)

        self.previous_error = error.copy()

        # Convert to joint commands using inverse kinematics
        joint_commands = self.convert_control_to_joints(control_output, current_state)

        return joint_commands, control_output

    def calculate_desired_com_position(self, sensor_data):
        """Calculate desired CoM position based on support polygon"""
        # Find center of support polygon
        support_polygon = self.calculate_support_polygon(sensor_data)
        if len(support_polygon) > 0:
            desired_x = np.mean(support_polygon[:, 0])
            desired_y = np.mean(support_polygon[:, 1])
            return np.array([desired_x, desired_y, 0.8])  # Fixed height
        else:
            return np.array([0.0, 0.0, 0.8])

    def assess_stability(self, error, error_derivative):
        """Assess current stability based on error characteristics"""
        # Calculate stability metrics
        error_magnitude = np.linalg.norm(error)
        velocity_magnitude = np.linalg.norm(error_derivative)

        # Stability is worse with larger errors and higher velocities
        stability_score = 1.0 / (1.0 + error_magnitude + 0.1 * velocity_magnitude)
        return stability_score

    def adapt_gains(self, stability_measure):
        """Adapt control gains based on stability"""
        # Increase gains when system is unstable, decrease when overly stable
        gain_multiplier = 1.0 + self.adaptation_rate * (self.stability_threshold - stability_measure)

        # Apply gain adaptation with limits
        for gain_name in ['kp', 'ki', 'kd']:
            new_gain = self.base_gains[gain_name] * gain_multiplier
            # Limit gain changes to prevent instability
            self.adaptive_gains[gain_name] = np.clip(
                new_gain,
                self.base_gains[gain_name] * 0.5,  # Lower bound
                self.base_gains[gain_name] * 2.0   # Upper bound
            )

    def convert_control_to_joints(self, cartesian_control, current_state):
        """Convert Cartesian control to joint commands"""
        # Use transpose Jacobian method for simplicity
        # In practice, this would use more sophisticated inverse kinematics
        current_joints = current_state['joint_positions']

        # Get CoM Jacobian
        com_jacobian = self.robot_model.com_jacobian(current_joints)

        # Apply control using transpose method
        joint_control = com_jacobian[:2, :].T @ cartesian_control  # Only x,y components

        # Add to current joint positions
        joint_commands = current_joints + joint_control * 0.001  # Small step

        # Apply joint limits
        joint_limits = self.robot_model.get_joint_limits()
        for i, (name, limits) in enumerate(joint_limits.items()):
            joint_commands[i] = np.clip(joint_commands[i], limits['min'], limits['max'])

        return joint_commands
```

## Summary

Balance control and locomotion for humanoid robots require sophisticated integration of multiple control techniques, sensor fusion, and real-time optimization. The key components include:

1. **Mathematical Models**: Understanding inverted pendulum models, ZMP theory, and capture point concepts for balance control
2. **Sensor Integration**: Effectively combining IMU, force/torque, and vision data for state estimation
3. **Control Techniques**: Implementing both simple reactive controllers and advanced predictive methods like MPC
4. **Whole-Body Coordination**: Managing the complex interactions between multiple joints and balance objectives
5. **Performance Optimization**: Ensuring real-time execution for stable control

The successful implementation of these techniques enables humanoid robots to maintain balance during both static poses and dynamic locomotion, which is fundamental for Physical AI systems that must operate safely in the physical world. The balance between computational efficiency and control performance is crucial, as is the integration of multiple sensor modalities for robust state estimation.