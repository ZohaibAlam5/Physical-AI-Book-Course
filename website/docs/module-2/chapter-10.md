---
title: Chapter 10 - Manipulation and Grasping for Humanoids
description: Advanced techniques for manipulation and grasping with humanoid robots
sidebar_position: 10
---

# Chapter 10: Manipulation and Grasping for Humanoids

## Learning Objectives

After completing this chapter, you should be able to:
- Understand the unique challenges of manipulation for humanoid robots
- Implement kinematic and dynamic models for humanoid manipulation
- Design grasp planning and execution algorithms
- Integrate perception with manipulation for object interaction
- Apply advanced control techniques for dexterous manipulation

## Introduction

Humanoid robots possess a unique advantage in manipulation tasks due to their anthropomorphic design, which allows them to interact with environments and objects designed for human use. However, manipulation with humanoid robots presents several unique challenges including redundant kinematics, balance constraints during manipulation, and the need to coordinate multiple degrees of freedom while maintaining stability. This chapter explores the advanced techniques required for effective manipulation and grasping with humanoid robots.

## Kinematic Considerations for Humanoid Manipulation

### Humanoid Arm Kinematics

Humanoid robots typically have 7-DOF arms similar to human arms, providing redundancy for reaching and manipulation:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidArmKinematics:
    """Kinematic model for humanoid arm manipulation"""

    def __init__(self, arm_config):
        """
        Initialize with arm configuration parameters

        Args:
            arm_config: Dictionary containing arm parameters
        """
        self.arm_config = arm_config

        # Arm segment lengths
        self.shoulder_offset = arm_config.get('shoulder_offset', 0.2)  # Distance from torso to shoulder
        self.upper_arm_length = arm_config.get('upper_arm_length', 0.3)  # Length from shoulder to elbow
        self.lower_arm_length = arm_config.get('lower_arm_length', 0.25)  # Length from elbow to wrist
        self.hand_length = arm_config.get('hand_length', 0.1)  # Length from wrist to fingertip

        # Joint limits (in radians)
        self.joint_limits = {
            'shoulder_yaw': (-np.pi/3, np.pi/3),      # Side-to-side movement
            'shoulder_pitch': (-np.pi/2, np.pi/2),    # Forward/backward movement
            'shoulder_roll': (-np.pi, np.pi),         # Rotational movement
            'elbow_pitch': (-np.pi/2, 0),             # Elbow bend
            'elbow_yaw': (-np.pi/4, np.pi/4),         # Forearm rotation
            'wrist_pitch': (-np.pi/3, np.pi/3),       # Wrist up/down
            'wrist_yaw': (-np.pi/2, np.pi/2)          # Wrist side-to-side
        }

    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector position and orientation from joint angles

        Args:
            joint_angles: Array of 7 joint angles [shoulder_yaw, shoulder_pitch, shoulder_roll,
                         elbow_pitch, elbow_yaw, wrist_pitch, wrist_yaw]

        Returns:
            Dictionary with 'position' and 'orientation' keys
        """
        if len(joint_angles) != 7:
            raise ValueError("Expected 7 joint angles")

        # Extract individual joint angles
        shoulder_yaw, shoulder_pitch, shoulder_roll, elbow_pitch, elbow_yaw, wrist_pitch, wrist_yaw = joint_angles

        # Calculate transformation matrices for each joint
        # Shoulder transformations
        T_shoulder = self._translation_matrix(0, 0, self.shoulder_offset)
        T_shoulder = T_shoulder @ self._rotation_matrix('z', shoulder_yaw)
        T_shoulder = T_shoulder @ self._rotation_matrix('y', shoulder_pitch)
        T_shoulder = T_shoulder @ self._rotation_matrix('x', shoulder_roll)

        # Upper arm (to elbow)
        T_elbow = T_shoulder @ self._translation_matrix(0, 0, self.upper_arm_length)
        T_elbow = T_elbow @ self._rotation_matrix('x', elbow_pitch)

        # Lower arm (to wrist)
        T_wrist = T_elbow @ self._translation_matrix(0, 0, self.lower_arm_length)
        T_wrist = T_wrist @ self._rotation_matrix('z', elbow_yaw)
        T_wrist = T_wrist @ self._rotation_matrix('x', wrist_pitch)
        T_wrist = T_wrist @ self._rotation_matrix('z', wrist_yaw)

        # Hand (to end-effector)
        T_end_effector = T_wrist @ self._translation_matrix(0, 0, self.hand_length)

        # Extract position and orientation
        position = T_end_effector[:3, 3]
        orientation_matrix = T_end_effector[:3, :3]
        orientation_quat = R.from_matrix(orientation_matrix).as_quat()

        return {
            'position': position,
            'orientation': orientation_quat,
            'transformation_matrix': T_end_effector
        }

    def inverse_kinematics(self, target_pose, initial_guess=None, max_iterations=100, tolerance=1e-4):
        """
        Calculate joint angles to reach target pose using Jacobian-based method

        Args:
            target_pose: Dictionary with 'position' and 'orientation' keys
            initial_guess: Starting joint angles (optional)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Joint angles array or None if no solution found
        """
        if initial_guess is None:
            # Use neutral position as initial guess
            initial_guess = np.array([0, 0, 0, -0.5, 0, 0, 0])

        current_joints = initial_guess.copy()

        for iteration in range(max_iterations):
            # Calculate current end-effector pose
            current_pose = self.forward_kinematics(current_joints)

            # Calculate errors
            pos_error = target_pose['position'] - current_pose['position']
            orient_error = self._quaternion_difference(
                target_pose['orientation'], current_pose['orientation'])

            # Combine position and orientation errors
            error = np.concatenate([pos_error, orient_error])

            # Check for convergence
            if np.linalg.norm(error) < tolerance:
                return current_joints

            # Calculate Jacobian
            jacobian = self.calculate_jacobian(current_joints)

            # Apply joint limits
            joint_limits_violated = self._check_joint_limits_violation(current_joints)
            if joint_limits_violated:
                # Apply joint limit avoidance
                limit_correction = self._calculate_joint_limit_correction(current_joints)
                error = np.concatenate([pos_error, orient_error]) + limit_correction

            # Calculate joint velocity using damped least squares
            damping = 0.01
            joint_delta = np.linalg.solve(
                jacobian.T @ jacobian + damping * np.eye(7),
                jacobian.T @ error
            )

            # Apply small step
            current_joints += joint_delta * 0.1  # Small step size for stability

            # Apply joint limits
            current_joints = self._apply_joint_limits(current_joints)

        # If no convergence, return None
        return None

    def calculate_jacobian(self, joint_angles):
        """
        Calculate geometric Jacobian for the arm

        Returns:
            6x7 Jacobian matrix (linear + angular velocities)
        """
        # Use numerical differentiation for simplicity
        epsilon = 1e-6
        jacobian = np.zeros((6, 7))

        # Calculate column for each joint
        for i in range(7):
            # Positive perturbation
            joints_plus = joint_angles.copy()
            joints_plus[i] += epsilon
            pose_plus = self.forward_kinematics(joints_plus)

            # Negative perturbation
            joints_minus = joint_angles.copy()
            joints_minus[i] -= epsilon
            pose_minus = self.forward_kinematics(joints_minus)

            # Calculate derivative
            pos_diff = pose_plus['position'] - pose_minus['position']
            orient_diff = self._quaternion_difference(pose_minus['orientation'], pose_plus['orientation'])

            jacobian[:3, i] = pos_diff / (2 * epsilon)
            jacobian[3:, i] = orient_diff / (2 * epsilon)

        return jacobian

    def _translation_matrix(self, x, y, z):
        """Create translation transformation matrix"""
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    def _rotation_matrix(self, axis, angle):
        """Create rotation matrix around specified axis"""
        if axis == 'x':
            return np.array([
                [1, 0, 0, 0],
                [0, np.cos(angle), -np.sin(angle), 0],
                [0, np.sin(angle), np.cos(angle), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'z':
            return np.array([
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

    def _quaternion_difference(self, q1, q2):
        """Calculate difference between two quaternions"""
        # Convert to rotation vectors for small angle approximation
        r1 = R.from_quat(q1).as_rotvec()
        r2 = R.from_quat(q2).as_rotvec()
        return r2 - r1

    def _check_joint_limits_violation(self, joints):
        """Check if any joint is near limit"""
        for i, (joint_name, limits) in enumerate(self.joint_limits.items()):
            if joints[i] < limits[0] + 0.1 or joints[i] > limits[1] - 0.1:
                return True
        return False

    def _calculate_joint_limit_correction(self, joints):
        """Calculate correction to move joints away from limits"""
        correction = np.zeros(7)
        for i, (joint_name, limits) in enumerate(self.joint_limits.items()):
            if joints[i] < limits[0] + 0.1:
                correction[i] = 0.1  # Move away from lower limit
            elif joints[i] > limits[1] - 0.1:
                correction[i] = -0.1  # Move away from upper limit
        return correction

    def _apply_joint_limits(self, joints):
        """Apply joint limits to joint angles"""
        limited_joints = joints.copy()
        for i, (joint_name, limits) in enumerate(self.joint_limits.items()):
            limited_joints[i] = np.clip(limited_joints[i], limits[0], limits[1])
        return limited_joints

    def calculate_reachable_workspace(self, resolution=0.05):
        """Calculate reachable workspace using Monte Carlo sampling"""
        workspace_points = []

        for _ in range(10000):  # Sample 10000 random configurations
            # Generate random joint angles within limits
            random_joints = []
            for joint_name, limits in self.joint_limits.items():
                random_joints.append(np.random.uniform(limits[0], limits[1]))

            random_joints = np.array(random_joints)

            # Calculate end-effector position
            try:
                pose = self.forward_kinematics(random_joints)
                workspace_points.append(pose['position'])
            except:
                continue  # Skip invalid configurations

        return np.array(workspace_points)
```

### Whole-Body Kinematics for Humanoid Manipulation

When manipulating objects, humanoid robots must consider the entire body's kinematics:

```python
class WholeBodyKinematics:
    """Whole-body kinematics for humanoid robots during manipulation"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.arm_kinematics = HumanoidArmKinematics(robot_model.get_arm_config())

    def calculate_manipulation_jacobian(self, joint_angles, end_effector_link):
        """
        Calculate manipulation Jacobian that relates joint velocities to end-effector velocities
        considering the entire body including balance constraints
        """
        # Get full robot Jacobian
        full_jacobian = self.robot_model.full_jacobian(joint_angles)

        # Get end-effector index in Jacobian
        ee_index = self.robot_model.get_link_index(end_effector_link)

        # Extract 6xN Jacobian for end-effector (linear + angular velocities)
        ee_jacobian = full_jacobian[6*ee_index:6*(ee_index+1), :]

        # Add balance constraints
        com_jacobian = self.robot_model.com_jacobian(joint_angles)

        # Return combined Jacobian with balance constraints
        return {
            'manipulation': ee_jacobian,
            'center_of_mass': com_jacobian,
            'full': full_jacobian
        }

    def calculate_manipulability_ellipse(self, joint_angles, end_effector_link):
        """Calculate manipulability ellipse to assess dexterity"""
        jacobian = self.calculate_manipulation_jacobian(joint_angles, end_effector_link)['manipulation']

        # Calculate manipulability measure
        JJT = jacobian @ jacobian.T
        manipulability = np.sqrt(np.linalg.det(JJT))

        # Calculate principal axes of manipulability ellipse
        eigenvals, eigenvecs = np.linalg.eigh(JJT)

        return {
            'measure': manipulability,
            'axes': np.sqrt(eigenvals),
            'directions': eigenvecs,
            'condition_number': np.sqrt(eigenvals[-1] / eigenvals[0]) if eigenvals[0] > 0 else float('inf')
        }

    def find_optimal_manipulation_posture(self, target_pose, balance_constraints=None):
        """
        Find optimal arm posture that maximizes manipulability while satisfying constraints
        """
        # This would typically involve optimization, but here's a simplified approach

        # Start with basic IK solution
        initial_solution = self.arm_kinematics.inverse_kinematics(target_pose)

        if initial_solution is None:
            return None

        # Optimize for manipulability while considering balance
        best_solution = initial_solution
        best_manipulability = 0

        # Try multiple initial guesses to find global optimum
        for _ in range(10):
            random_guess = self._generate_random_valid_configuration()
            solution = self.arm_kinematics.inverse_kinematics(target_pose, random_guess)

            if solution is not None:
                # Calculate manipulability
                manipulability_data = self.calculate_manipulability_ellipse(
                    solution, 'right_hand' if 'right' in target_pose.get('link', '') else 'left_hand')

                current_manipulability = manipulability_data['measure']

                # Check if solution satisfies balance constraints
                if balance_constraints is None or self._check_balance_feasibility(solution, balance_constraints):
                    if current_manipulability > best_manipulability:
                        best_solution = solution
                        best_manipulability = current_manipulability

        return best_solution

    def _generate_random_valid_configuration(self):
        """Generate random joint configuration within limits"""
        config = []
        for joint_name, limits in self.arm_kinematics.joint_limits.items():
            config.append(np.random.uniform(limits[0], limits[1]))
        return np.array(config)

    def _check_balance_feasibility(self, joint_angles, balance_constraints):
        """Check if joint configuration satisfies balance constraints"""
        # Calculate CoM position with this configuration
        com_pos = self.robot_model.calculate_com_position(joint_angles)

        # Check if CoM is within balance constraints (e.g., support polygon)
        if 'support_polygon' in balance_constraints:
            return self._is_point_in_polygon(com_pos[:2], balance_constraints['support_polygon'])

        return True  # No balance constraints specified
```

## Grasp Planning and Execution

### Grasp Synthesis

Creating effective grasps for various objects:

```python
class GraspSynthesizer:
    """Synthesize grasps for objects based on geometry and task requirements"""

    def __init__(self, robot_hand_model):
        self.hand_model = robot_hand_model
        self.grasp_database = {}  # Precomputed grasps for common objects

    def synthesize_grasps(self, object_info, task_requirements=None):
        """
        Synthesize potential grasps for an object based on its properties

        Args:
            object_info: Dictionary containing object properties (shape, size, weight, etc.)
            task_requirements: Task-specific requirements (lifting, pouring, etc.)

        Returns:
            List of potential grasp poses ranked by quality
        """
        grasp_candidates = []

        # Generate grasp candidates based on object shape
        if object_info['shape'] == 'cylinder':
            grasp_candidates.extend(self._generate_cylinder_grasps(object_info))
        elif object_info['shape'] == 'box':
            grasp_candidates.extend(self._generate_box_grasps(object_info))
        elif object_info['shape'] == 'sphere':
            grasp_candidates.extend(self._generate_sphere_grasps(object_info))
        else:
            # Generic grasp synthesis for unknown shapes
            grasp_candidates.extend(self._generate_generic_grasps(object_info))

        # Rank grasps by quality metrics
        ranked_grasps = self._rank_grasps(grasp_candidates, object_info, task_requirements)

        return ranked_grasps

    def _generate_cylinder_grasps(self, object_info):
        """Generate grasps for cylindrical objects"""
        grasps = []

        # Define cylinder parameters
        radius = object_info['dimensions']['radius']
        height = object_info['dimensions']['height']
        center = object_info['position']

        # Generate power grasps (around the circumference)
        for angle in np.linspace(0, 2*np.pi, 12):  # 12 angular positions
            # Power grasp - hand wraps around cylinder
            grasp_pose = {
                'position': np.array([
                    center[0] + (radius + 0.02) * np.cos(angle),  # 2cm clearance
                    center[1] + (radius + 0.02) * np.sin(angle),
                    center[2]
                ]),
                'orientation': R.from_euler('xyz', [0, 0, angle]).as_quat(),
                'type': 'power',
                'quality': self._evaluate_power_grasp_quality(radius, object_info['weight'])
            }
            grasps.append(grasp_pose)

        # Generate pinch grasps (top/bottom)
        for z_offset in [-height/2, height/2]:  # Top and bottom
            for angle in np.linspace(0, 2*np.pi, 8):  # 8 angular positions
                # Pinch grasp - fingers oppose along diameter
                grasp_pose = {
                    'position': np.array([center[0], center[1], center[2] + z_offset]),
                    'orientation': R.from_euler('xyz', [np.pi/2, 0, angle]).as_quat(),  # Approach from side
                    'type': 'pinch',
                    'quality': self._evaluate_pinch_grasp_quality(radius, object_info['weight'])
                }
                grasps.append(grasp_pose)

        return grasps

    def _generate_box_grasps(self, object_info):
        """Generate grasps for box-shaped objects"""
        grasps = []

        # Get box dimensions
        dims = object_info['dimensions']
        center = object_info['position']

        # Generate corner grasps
        corner_offsets = [
            [dims['x']/2, dims['y']/2, dims['z']/2],
            [dims['x']/2, dims['y']/2, -dims['z']/2],
            [dims['x']/2, -dims['y']/2, dims['z']/2],
            [dims['x']/2, -dims['y']/2, -dims['z']/2],
            [-dims['x']/2, dims['y']/2, dims['z']/2],
            [-dims['x']/2, dims['y']/2, -dims['z']/2],
            [-dims['x']/2, -dims['y']/2, dims['z']/2],
            [-dims['x']/2, -dims['y']/2, -dims['z']/2]
        ]

        for offset in corner_offsets:
            # Corner grasp - approach from diagonal
            corner_pos = center + np.array(offset)
            approach_dir = -np.array(offset) / np.linalg.norm(offset)  # Approach from opposite direction

            # Calculate orientation to align hand with approach direction
            z_axis = approach_dir
            x_axis = np.array([1, 0, 0])  # Default hand orientation
            if abs(np.dot(z_axis, x_axis)) > 0.9:  # Avoid singularity
                x_axis = np.array([0, 1, 0])

            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            orientation = R.from_matrix(rotation_matrix).as_quat()

            grasp_pose = {
                'position': corner_pos,
                'orientation': orientation,
                'type': 'corner',
                'quality': self._evaluate_corner_grasp_quality(dims, object_info['weight'])
            }
            grasps.append(grasp_pose)

        # Generate face-centered grasps
        faces = [
            ([dims['x']/2, 0, 0], [1, 0, 0]),  # +X face
            ([-dims['x']/2, 0, 0], [-1, 0, 0]), # -X face
            ([0, dims['y']/2, 0], [0, 1, 0]),   # +Y face
            ([0, -dims['y']/2, 0], [0, -1, 0]), # -Y face
            ([0, 0, dims['z']/2], [0, 0, 1]),   # +Z face (top)
            ([0, 0, -dims['z']/2], [0, 0, -1])  # -Z face (bottom)
        ]

        for face_center_offset, approach_dir in faces:
            face_center = center + np.array(face_center_offset)

            # Calculate orientation for face grasp
            z_axis = np.array(approach_dir)
            x_axis = np.array([1, 0, 0])
            if abs(np.dot(z_axis, x_axis)) > 0.9:
                x_axis = np.array([0, 1, 0])

            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            orientation = R.from_matrix(rotation_matrix).as_quat()

            grasp_pose = {
                'position': face_center,
                'orientation': orientation,
                'type': 'face',
                'quality': self._evaluate_face_grasp_quality(dims, object_info['weight'])
            }
            grasps.append(grasp_pose)

        return grasps

    def _generate_sphere_grasps(self, object_info):
        """Generate grasps for spherical objects"""
        grasps = []

        radius = object_info['dimensions']['radius']
        center = object_info['position']

        # Generate spherical coordinate-based grasps
        for theta in np.linspace(0, np.pi, 6):  # Polar angle
            for phi in np.linspace(0, 2*np.pi, 8):  # Azimuthal angle
                # Calculate surface point
                x = center[0] + (radius + 0.02) * np.sin(theta) * np.cos(phi)
                y = center[1] + (radius + 0.02) * np.sin(theta) * np.sin(phi)
                z = center[2] + (radius + 0.02) * np.cos(theta)

                position = np.array([x, y, z])

                # Calculate orientation to point inward toward center
                approach_dir = (center - position)  # Vector pointing toward center
                approach_dir = approach_dir / np.linalg.norm(approach_dir)

                # Choose hand orientation (orthogonal to approach direction)
                z_axis = -approach_dir  # Hand approaches along this direction
                x_axis = np.array([1, 0, 0])

                # Avoid singularity when approach is along x-axis
                if abs(np.dot(z_axis, x_axis)) > 0.9:
                    x_axis = np.array([0, 1, 0])

                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)

                rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
                orientation = R.from_matrix(rotation_matrix).as_quat()

                grasp_pose = {
                    'position': position,
                    'orientation': orientation,
                    'type': 'sphere',
                    'quality': self._evaluate_sphere_grasp_quality(radius, object_info['weight'])
                }
                grasps.append(grasp_pose)

        return grasps

    def _rank_grasps(self, grasps, object_info, task_requirements):
        """Rank grasps by quality metrics"""
        ranked_grasps = []

        for grasp in grasps:
            quality_score = 0

            # Geometric quality (based on object properties)
            quality_score += self._calculate_geometric_quality(grasp, object_info)

            # Kinematic quality (reachability and dexterity)
            quality_score += self._calculate_kinematic_quality(grasp, object_info)

            # Task-specific quality (if provided)
            if task_requirements:
                quality_score += self._calculate_task_quality(grasp, task_requirements)

            # Stability quality (grasp stability)
            quality_score += self._calculate_stability_quality(grasp, object_info)

            grasp['quality_score'] = quality_score
            ranked_grasps.append(grasp)

        # Sort by quality score (descending)
        ranked_grasps.sort(key=lambda x: x['quality_score'], reverse=True)
        return ranked_grasps

    def _calculate_geometric_quality(self, grasp, object_info):
        """Calculate geometric quality of grasp"""
        # This would involve checking grasp contacts and object geometry
        # For now, return a simplified score
        if grasp['type'] == 'power':
            return 0.8  # Power grasps generally good for stability
        elif grasp['type'] == 'pinch':
            return 0.6  # Pinch grasps good for precision but less stable
        else:
            return 0.7  # Default score

    def _calculate_kinematic_quality(self, grasp, object_info):
        """Calculate kinematic quality (reachability and dexterity)"""
        # Check if grasp is kinematically reachable
        target_pose = {
            'position': grasp['position'],
            'orientation': grasp['orientation']
        }

        # Use inverse kinematics to check reachability
        ik_solution = self.hand_model.inverse_kinematics(target_pose)

        if ik_solution is None:
            return -1.0  # Not reachable

        # Calculate manipulability at this configuration
        manipulability = self.hand_model.calculate_manipulability(ik_solution)

        # Return normalized manipulability score
        return min(manipulability / 0.1, 1.0)  # Normalize to 0-1 range

    def _calculate_task_quality(self, grasp, task_requirements):
        """Calculate task-specific quality"""
        task_score = 0.0

        # Example task requirements: lifting, pouring, precision placement
        if 'task_type' in task_requirements:
            if task_requirements['task_type'] == 'lifting':
                # Prefer power grasps for lifting heavy objects
                if grasp['type'] in ['power', 'corner']:
                    task_score += 0.3
            elif task_requirements['task_type'] == 'precision':
                # Prefer pinch grasps for precision tasks
                if grasp['type'] == 'pinch':
                    task_score += 0.3
            elif task_requirements['task_type'] == 'pouring':
                # Prefer grasps that allow proper orientation for pouring
                if self._is_pouring_friendly_orientation(grasp['orientation']):
                    task_score += 0.3

        return task_score

    def _calculate_stability_quality(self, grasp, object_info):
        """Calculate grasp stability quality"""
        # Stability depends on friction, contact points, and object weight
        friction_coeff = object_info.get('friction_coefficient', 0.8)
        weight = object_info['weight']

        # Simple stability calculation
        stability_score = friction_coeff * 0.5  # Base stability on friction

        # Reduce stability for heavier objects
        max_stable_weight = 2.0  # kg - maximum stable grasp weight
        stability_score *= max(0, 1 - (weight - max_stable_weight) / max_stable_weight)

        return max(0, stability_score)

    def _is_pouring_friendly_orientation(self, orientation):
        """Check if orientation is suitable for pouring task"""
        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(orientation).as_matrix()

        # For pouring, we want the container opening to face downward
        # This depends on the container's geometry
        # Simplified check: z-axis of container should be tilted appropriately
        container_z_axis = rotation_matrix[:, 2]  # Third column is z-axis

        # Check if z-axis has appropriate downward component
        return container_z_axis[2] < -0.5  # At least 30 degrees downward tilt

    def evaluate_grasp_stability(self, grasp_pose, object_properties, hand_configuration):
        """Evaluate the stability of a specific grasp"""
        # Calculate contact points between hand and object
        contact_points = self._calculate_contact_points(grasp_pose, hand_configuration, object_properties)

        # Calculate grasp wrench space
        wrench_space = self._calculate_grasp_wrench_space(contact_points, object_properties)

        # Evaluate grasp quality metrics
        metrics = {
            'volume': self._calculate_wrench_space_volume(wrench_space),
            'condition': self._calculate_wrench_space_condition(wrench_space),
            'closure': self._check_force_closure(contact_points),
            'friction_cones': self._evaluate_friction_cones(contact_points, object_properties)
        }

        # Overall stability score
        stability_score = (
            0.3 * self._normalize_volume(metrics['volume']) +
            0.2 * self._normalize_condition(metrics['condition']) +
            0.3 * (1.0 if metrics['closure'] else 0.0) +
            0.2 * metrics['friction_cones']
        )

        return {
            'stability_score': stability_score,
            'metrics': metrics,
            'is_stable': stability_score > 0.5  # Threshold for stability
        }

    def _calculate_contact_points(self, grasp_pose, hand_configuration, object_properties):
        """Calculate contact points between hand and object"""
        # This would use detailed hand model and object geometry
        # For now, return simplified contact points
        contacts = []

        # Example: 3 contact points for a simple grasp
        for finger_idx in range(3):  # Assume 3 contact points
            # Calculate contact position based on grasp pose and finger position
            contact_pos = grasp_pose['position'] + np.random.rand(3) * 0.01  # Small random offset
            contact_normal = np.random.rand(3)  # Random normal (would be calculated from geometry)
            contact_normal = contact_normal / np.linalg.norm(contact_normal)

            contacts.append({
                'position': contact_pos,
                'normal': contact_normal,
                'friction_coeff': object_properties.get('friction_coefficient', 0.8)
            })

        return contacts

    def _calculate_grasp_wrench_space(self, contact_points, object_properties):
        """Calculate the grasp wrench space"""
        # This is a simplified implementation
        # In practice, this would involve complex convex hull calculations

        # For each contact point, calculate the wrench cone
        wrench_cones = []
        for contact in contact_points:
            # Calculate wrench basis for this contact
            normal = contact['normal']
            friction_coeff = contact['friction_coeff']

            # Create wrench space basis (simplified)
            wrench_basis = self._create_wrench_basis(normal, friction_coeff)
            wrench_cones.append(wrench_basis)

        # Combine all wrench cones
        combined_wrench_space = np.vstack(wrench_cones)

        return combined_wrench_space

    def _create_wrench_basis(self, normal, friction_coeff):
        """Create wrench basis for a contact point"""
        # Create tangent vectors orthogonal to normal
        t1 = np.array([1, 0, 0])
        if abs(np.dot(normal, t1)) > 0.9:
            t1 = np.array([0, 1, 0])

        t2 = np.cross(normal, t1)
        t2 = t2 / np.linalg.norm(t2)
        t1 = np.cross(t2, normal)
        t1 = t1 / np.linalg.norm(t1)

        # Create wrench basis vectors
        # Forces: normal force + friction forces
        normal_force = np.concatenate([normal, np.zeros(3)])  # Force component
        friction_force1 = np.concatenate([friction_coeff * t1, np.zeros(3)])
        friction_force2 = np.concatenate([friction_coeff * t2, np.zeros(3)])

        # Moments: due to contact location
        contact_pos = np.random.rand(3)  # Simplified contact position
        moment1 = np.concatenate([np.zeros(3), np.cross(contact_pos, friction_coeff * t1)])
        moment2 = np.concatenate([np.zeros(3), np.cross(contact_pos, friction_coeff * t2)])

        return np.array([normal_force, friction_force1, friction_force2, moment1, moment2])

    def _calculate_wrench_space_volume(self, wrench_space):
        """Calculate volume of wrench space (simplified)"""
        # In practice, this would calculate the volume of the convex hull
        # of the wrench space, which is computationally intensive
        # Here we use a simplified proxy
        try:
            # Calculate determinant of a subset of wrench space for volume proxy
            if wrench_space.shape[0] >= 6:
                subset = wrench_space[:6, :6]  # Take 6x6 subset
                volume_proxy = abs(np.linalg.det(subset))
                return volume_proxy
        except:
            pass

        return 0.01  # Default small value

    def _calculate_wrench_space_condition(self, wrench_space):
        """Calculate condition number of wrench space"""
        try:
            # Condition number indicates how well-conditioned the grasp is
            U, s, Vt = np.linalg.svd(wrench_space)
            condition_number = s[0] / s[-1] if s[-1] != 0 else float('inf')
            return 1.0 / (1.0 + condition_number)  # Convert to quality score (higher is better)
        except:
            return 0.0

    def _check_force_closure(self, contact_points):
        """Check if grasp has force closure"""
        # Force closure means the grasp can resist arbitrary external wrenches
        # This is a simplified check - full implementation would be more complex
        return len(contact_points) >= 3  # Minimum for force closure in 3D

    def _evaluate_friction_cones(self, contact_points, object_properties):
        """Evaluate friction cone constraints"""
        # Check if contact forces can be applied within friction limits
        valid_contacts = 0
        for contact in contact_points:
            # For now, just check if friction coefficient is reasonable
            if 0.1 <= contact['friction_coeff'] <= 1.0:
                valid_contacts += 1

        return valid_contacts / len(contact_points)  # Fraction of valid contacts
```

## Manipulation Control Strategies

### Operational Space Control for Manipulation

Implementing operational space control for precise manipulation:

```python
class OperationalSpaceController:
    """Operational space controller for humanoid manipulation"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.gravity = 9.81

    def compute_operational_space_control(self, current_state, desired_task_state,
                                        task_jacobian, task_selection_matrix=None):
        """
        Compute operational space control command

        Args:
            current_state: Current joint positions and velocities
            desired_task_state: Desired task space state and derivatives
            task_jacobian: Task Jacobian matrix
            task_selection_matrix: Selection matrix for specific task DOFs (optional)

        Returns:
            Joint torques for control
        """
        joint_positions = current_state['positions']
        joint_velocities = current_state['velocities']

        if task_selection_matrix is None:
            # Use identity matrix for all task DOFs
            task_selection_matrix = np.eye(task_jacobian.shape[0])

        # Calculate current task state
        current_task_state = self._forward_task_kinematics(joint_positions, task_jacobian)

        # Calculate task error
        position_error = desired_task_state['position'] - current_task_state['position']
        velocity_error = desired_task_state['velocity'] - current_task_state['velocity']

        # Calculate task-space acceleration command
        kp = desired_task_state.get('kp', 100.0)  # Position gain
        kd = desired_task_state.get('kd', 20.0)   # Velocity gain (damping)

        task_acceleration = (
            kp * position_error +
            kd * velocity_error +
            desired_task_state.get('acceleration', np.zeros_like(position_error))
        )

        # Calculate operational space mass matrix
        M = self.robot_model.mass_matrix(joint_positions)
        M_inv = np.linalg.inv(M)

        # Lambda = (J * M^-1 * J^T)^-1 (task space mass matrix)
        J_M_inv_JT = task_jacobian @ M_inv @ task_jacobian.T
        try:
            Lambda = np.linalg.inv(J_M_inv_JT)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            Lambda = np.linalg.pinv(J_M_inv_JT)

        # Calculate Coriolis and gravity compensation in task space
        h = self.robot_model.coriolis_gravity_forces(joint_positions, joint_velocities)
        J_transpose = task_jacobian.T

        # Calculate operational space control command
        task_force = Lambda @ task_acceleration

        # Convert to joint torques
        # tau = J^T * F + N^T * tau_null  (where N is null space projector)
        joint_torques = J_transpose @ task_force

        # Add null-space control to maintain secondary objectives (posture, balance)
        null_space_projector = np.eye(len(joint_positions)) - J_transpose @ np.linalg.pinv(task_jacobian)
        posture_torques = self._compute_posture_control(joint_positions, desired_task_state.get('neutral_configuration'))
        joint_torques += null_space_projector @ posture_torques

        return joint_torques

    def _forward_task_kinematics(self, joint_positions, task_jacobian):
        """Calculate current task space state"""
        # This would involve forward kinematics to get current task position
        # For now, return a simplified calculation
        current_position = np.zeros(task_jacobian.shape[0] // 2)  # Assuming 6 DOF task (position + orientation)
        current_velocity = np.zeros(task_jacobian.shape[0] // 2)  # Assuming 6 DOF task

        return {
            'position': current_position,
            'velocity': current_velocity
        }

    def _compute_posture_control(self, current_joints, neutral_configuration):
        """Compute null-space posture control"""
        if neutral_configuration is None:
            # Use default neutral configuration
            neutral_configuration = np.zeros(len(current_joints))

        # PD control in joint space for posture
        kp_posture = 10.0
        kd_posture = 1.0

        posture_error = neutral_configuration - current_joints
        posture_velocity = -self.current_joint_velocities  # Assuming we have this

        posture_torques = kp_posture * posture_error - kd_posture * posture_velocity

        return posture_torques

    def multi_task_control(self, current_state, tasks):
        """
        Perform multi-task control with priority-based task execution

        Args:
            current_state: Current robot state
            tasks: List of tasks with priorities

        Returns:
            Joint torques for control
        """
        joint_positions = current_state['positions']
        joint_velocities = current_state['velocities']

        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda x: x['priority'], reverse=True)

        # Initialize total torques
        total_torques = np.zeros(len(joint_positions))
        current_null_space = np.eye(len(joint_positions))

        for task in sorted_tasks:
            # Get task Jacobian and desired state
            task_jacobian = task['jacobian']
            desired_state = task['desired_state']

            # Project task Jacobian into current null space
            projected_jacobian = task_jacobian @ current_null_space

            # Compute task-space control
            task_control = self.compute_operational_space_control(
                current_state, desired_state, projected_jacobian)

            # Convert to joint torques in current null space
            joint_task_torques = current_null_space.T @ task_control

            # Add to total torques
            total_torques += joint_task_torques

            # Update null space projector
            # Calculate new null space projector after this task
            try:
                M = self.robot_model.mass_matrix(joint_positions)
                M_inv = np.linalg.inv(M)

                J_M_inv = projected_jacobian @ M_inv
                J_M_inv_JT = J_M_inv @ projected_jacobian.T

                # Regularized inverse to handle singularities
                reg_param = 0.001
                N_task = np.eye(len(joint_positions)) - M_inv @ J_M_inv.T @ np.linalg.inv(J_M_inv_JT + reg_param * np.eye(J_M_inv_JT.shape[0])) @ J_M_inv

                # Update current null space
                current_null_space = N_task @ current_null_space
            except np.linalg.LinAlgError:
                # If matrix is singular, don't update null space
                pass

        return total_torques

    def cartesian_impedance_control(self, current_state, desired_pose,
                                  stiffness_matrix=None, damping_matrix=None):
        """
        Implement Cartesian impedance control for compliant manipulation

        Args:
            current_state: Current robot state
            desired_pose: Desired Cartesian pose [position, orientation]
            stiffness_matrix: 6x6 stiffness matrix (optional)
            damping_matrix: 6x6 damping matrix (optional)

        Returns:
            Joint torques for impedance control
        """
        if stiffness_matrix is None:
            stiffness_matrix = np.diag([1000, 1000, 1000, 100, 100, 100])  # Position >> orientation

        if damping_matrix is None:
            # Critical damping: D = 2 * sqrt(K * M)
            # For now, use simple proportional relationship
            damping_matrix = 2 * np.sqrt(stiffness_matrix)

        # Get current end-effector pose
        current_pose = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'end_effector')

        # Calculate pose error
        pos_error = desired_pose['position'] - current_pose['position']

        # For orientation error, use rotation vector representation
        current_rot = R.from_quat(current_pose['orientation'])
        desired_rot = R.from_quat(desired_pose['orientation'])
        relative_rot = desired_rot * current_rot.inv()
        orientation_error = relative_rot.as_rotvec()

        pose_error = np.concatenate([pos_error, orientation_error])

        # Calculate Cartesian velocity
        jacobian = self.robot_model.jacobian_for_link(current_state['positions'], 'end_effector')
        current_cart_vel = jacobian @ current_state['velocities']

        # Calculate Cartesian forces
        cartesian_force = stiffness_matrix @ pose_error - damping_matrix @ current_cart_vel

        # Convert to joint torques using transpose Jacobian method
        joint_torques = jacobian.T @ cartesian_force

        # Add gravity compensation
        gravity_compensation = self.robot_model.gravity_compensation(current_state['positions'])
        joint_torques += gravity_compensation

        return joint_torques

    def admittance_control(self, current_state, external_wrench,
                          desired_admittance_matrix=None):
        """
        Implement admittance control for force-guided manipulation

        Args:
            current_state: Current robot state
            external_wrench: Measured external force/torque
            desired_admittance_matrix: 6x6 admittance matrix (optional)

        Returns:
            Joint torques for admittance control
        """
        if desired_admittance_matrix is None:
            # Default admittance (how much the robot yields to external forces)
            desired_admittance_matrix = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])  # More compliant in translation

        # Calculate desired motion based on external forces
        desired_cartesian_accel = desired_admittance_matrix @ external_wrench

        # Convert to joint space
        jacobian = self.robot_model.jacobian_for_link(current_state['positions'], 'end_effector')

        # Use pseudo-inverse to map Cartesian acceleration to joint acceleration
        joint_accel = np.linalg.pinv(jacobian) @ desired_cartesian_accel

        # Convert acceleration to torques using inverse dynamics
        M = self.robot_model.mass_matrix(current_state['positions'])
        C = self.robot_model.coriolis_matrix(current_state['positions'], current_state['velocities'])
        G = self.robot_model.gravity_vector(current_state['positions'])

        joint_torques = M @ joint_accel + C @ current_state['velocities'] + G

        return joint_torques

    def hybrid_force_position_control(self, current_state, desired_trajectory,
                                   force_control_dofs, position_control_dofs,
                                   desired_forces=None):
        """
        Implement hybrid force/position control for constrained manipulation

        Args:
            current_state: Current robot state
            desired_trajectory: Desired position trajectory
            force_control_dofs: Degrees of freedom to control with force
            position_control_dofs: Degrees of freedom to control with position
            desired_forces: Desired forces in force-controlled DOFs (optional)

        Returns:
            Joint torques for hybrid control
        """
        if desired_forces is None:
            desired_forces = np.zeros(len(force_control_dofs))

        # Get current state
        current_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'end_effector')['position']
        jacobian = self.robot_model.jacobian_for_link(current_state['positions'], 'end_effector')

        # Calculate position error for position-controlled DOFs
        pos_error = desired_trajectory['position'] - current_pos

        # Calculate joint torques for position control
        pos_torques = np.zeros(len(current_state['positions']))
        for i, dof_idx in enumerate(position_control_dofs):
            pos_torques += jacobian[dof_idx:dof_idx+1, :].T * pos_error[dof_idx] * 100  # Position gain

        # Calculate joint torques for force control
        force_torques = np.zeros(len(current_state['positions']))
        for i, dof_idx in enumerate(force_control_dofs):
            force_error = desired_forces[i] - self.measured_external_forces[dof_idx]
            force_torques += jacobian[dof_idx:dof_idx+1, :].T * force_error * 10  # Force gain

        # Combine position and force control torques
        total_torques = pos_torques + force_torques

        # Add gravity and Coriolis compensation
        h = self.robot_model.coriolis_gravity_forces(
            current_state['positions'], current_state['velocities'])
        total_torques += h

        return total_torques

    def calculate_manipulability_measures(self, joint_positions, end_effector_link):
        """Calculate various manipulability measures"""
        jacobian = self.robot_model.jacobian_for_link(joint_positions, end_effector_link)

        # Calculate different manipulability measures
        JJT = jacobian @ jacobian.T

        # Yoshikawa manipulability (kinematic dexterity)
        yoshikawa = np.sqrt(np.linalg.det(JJT))

        # Condition number (proximity to singularities)
        U, s, Vt = np.linalg.svd(JJT)
        condition_number = s[0] / s[-1] if s[-1] != 0 else float('inf')

        # Minimum singular value (local dexterity)
        min_singular_value = s[-1]

        # Manipulability ellipsoid axes
        ellipsoid_axes = np.sqrt(s)

        return {
            'yoshikawa': yoshikawa,
            'condition_number': condition_number,
            'min_singular': min_singular_value,
            'ellipsoid_axes': ellipsoid_axes,
            'is_healthy': min_singular_value > 0.01  # Healthy if not near singularity
        }

    def calculate_dynamic_manipulability(self, joint_positions, joint_velocities, end_effector_link):
        """Calculate dynamic manipulability considering robot dynamics"""
        jacobian = self.robot_model.jacobian_for_link(joint_positions, end_effector_link)
        M = self.robot_model.mass_matrix(joint_positions)

        # Calculate dynamic manipulability ellipsoid
        try:
            M_inv = np.linalg.inv(M)
            Lambda = np.linalg.inv(jacobian @ M_inv @ jacobian.T)

            # Dynamic manipulability measure
            dynamic_measure = np.sqrt(np.linalg.det(Lambda))

            # Eigenvalues of dynamic manipulability matrix
            eigenvals, _ = np.linalg.eigh(Lambda)

            return {
                'measure': dynamic_measure,
                'eigenvalues': eigenvals,
                'ellipsoid_axes': np.sqrt(eigenvals),
                'condition_number': eigenvals[-1] / eigenvals[0] if eigenvals[0] != 0 else float('inf')
            }
        except np.linalg.LinAlgError:
            return {
                'measure': 0.0,
                'eigenvalues': np.zeros(jacobian.shape[0]),
                'ellipsoid_axes': np.zeros(jacobian.shape[0]),
                'condition_number': float('inf')
            }
```

## Sensor Integration for Manipulation

### Vision-Guided Manipulation

Integrating camera sensors for visual servoing and object manipulation:

```python
import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class VisionGuidedManipulation:
    """Vision-guided manipulation system for humanoid robots"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.cv_bridge = CvBridge()

        # Camera parameters (these would typically come from camera_info topic)
        self.camera_matrix = np.array([
            [525.0, 0.0, 319.5],  # fx, 0, cx
            [0.0, 525.0, 239.5],  # 0, fy, cy
            [0.0, 0.0, 1.0]       # 0, 0, 1
        ])

        # Distortion coefficients
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # No distortion for simplicity

        # Object detection and tracking
        self.object_detectors = {}
        self.tracked_objects = {}

    def process_camera_image(self, image_msg, camera_info_msg=None):
        """Process camera image for object detection and tracking"""
        # Convert ROS image to OpenCV format
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Detect objects in the image
        detected_objects = self.detect_objects_in_image(cv_image)

        # Track objects over time
        self.update_object_tracking(detected_objects)

        return detected_objects

    def detect_objects_in_image(self, cv_image):
        """Detect objects in camera image"""
        # This would typically use a trained object detection model
        # For demonstration, we'll use a simple color-based detection

        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects (example: red cup)
        color_ranges = {
            'red_cup': ([0, 50, 50], [10, 255, 255]),    # Lower red range
            'red_cup2': ([170, 50, 50], [180, 255, 255]) # Upper red range
        }

        detected_objects = []

        for obj_name, (lower, upper) in color_ranges.items():
            # Create mask for color range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Filter by area to avoid noise
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center in image coordinates
                    center_u = x + w // 2
                    center_v = y + h // 2

                    # Calculate object depth (simplified - in real system would use stereo or depth camera)
                    # For now, assume a fixed depth or use size-based estimation
                    depth = self.estimate_depth_from_size(w, h, obj_name)

                    # Convert pixel coordinates to 3D world coordinates
                    object_position = self.pixel_to_world_coordinates(
                        center_u, center_v, depth, self.camera_matrix)

                    detected_objects.append({
                        'name': obj_name,
                        'position': object_position,
                        'pixel_coords': (center_u, center_v),
                        'bounding_box': (x, y, w, h),
                        'confidence': area / (cv_image.shape[0] * cv_image.shape[1])  # Simple confidence based on relative size
                    })

        return detected_objects

    def estimate_depth_from_size(self, pixel_width, pixel_height, object_name):
        """Estimate depth based on object size in pixels"""
        # This is a simplified depth estimation
        # In a real system, you would use stereo vision, structured light, or other depth sensing

        # Known object dimensions (in meters)
        known_sizes = {
            'red_cup': (0.08, 0.08, 0.10),  # width, height, depth
            'green_bottle': (0.06, 0.06, 0.25),
            'book': (0.20, 0.15, 0.02)
        }

        if object_name in known_sizes:
            # Use pinhole camera model: size_in_pixels = (actual_size * focal_length) / distance
            # Rearranging: distance = (actual_size * focal_length) / size_in_pixels

            # Use the average of width and height for depth estimation
            avg_pixel_size = (pixel_width + pixel_height) / 2
            actual_size = known_sizes[object_name][0]  # Use width as reference dimension
            focal_length = self.camera_matrix[0, 0]  # fx

            depth = (actual_size * focal_length) / avg_pixel_size
            return max(depth, 0.1)  # Minimum distance of 0.1m

        # If object size unknown, return default value
        return 1.0

    def pixel_to_world_coordinates(self, u, v, depth, camera_matrix):
        """Convert pixel coordinates + depth to world coordinates"""
        # Invert camera matrix
        K_inv = np.linalg.inv(camera_matrix)

        # Create homogeneous pixel coordinate
        pixel_homogeneous = np.array([u, v, 1.0])

        # Convert to normalized camera coordinates
        normalized_camera = K_inv @ pixel_homogeneous

        # Scale by depth to get 3D camera coordinates
        camera_coords = normalized_camera * depth

        # Convert from camera frame to world frame
        # This requires knowing the camera pose relative to the robot base
        # For simplicity, assume camera is at origin looking along -Z axis
        # In real system, would use TF transforms
        world_coords = camera_coords  # Simplified assumption

        return world_coords

    def update_object_tracking(self, detected_objects):
        """Update object tracking with new detections"""
        for detected_obj in detected_objects:
            obj_name = detected_obj['name']

            if obj_name in self.tracked_objects:
                # Update existing track
                old_track = self.tracked_objects[obj_name]
                new_position = detected_obj['position']

                # Calculate velocity (if we have previous position and timestamp)
                if 'prev_position' in old_track and 'timestamp' in old_track:
                    dt = detected_obj.get('timestamp', 0) - old_track['timestamp']
                    if dt > 0:
                        velocity = (new_position - old_track['prev_position']) / dt
                        old_track['velocity'] = velocity

                # Update track information
                old_track['position'] = new_position
                old_track['prev_position'] = new_position
                old_track['timestamp'] = detected_obj.get('timestamp', 0)
                old_track['confidence'] = detected_obj['confidence']
            else:
                # Create new track
                self.tracked_objects[obj_name] = {
                    'position': detected_obj['position'],
                    'prev_position': detected_obj['position'],
                    'velocity': np.zeros(3),
                    'timestamp': detected_obj.get('timestamp', 0),
                    'confidence': detected_obj['confidence'],
                    'track_id': len(self.tracked_objects)
                }

    def calculate_visual_servoing_commands(self, current_state, target_object_name,
                                         camera_frame='camera_link',
                                         end_effector_frame='end_effector'):
        """Calculate visual servoing commands to move end-effector toward target object"""
        if target_object_name not in self.tracked_objects:
            return None  # Target not visible

        # Get target object position in camera frame
        target_pos_camera = self.tracked_objects[target_object_name]['position']

        # Transform to robot base frame
        target_pos_base = self.transform_camera_to_base(
            target_pos_camera, camera_frame)

        # Get current end-effector position in base frame
        current_ee_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], end_effector_frame)['position']

        # Calculate error in Cartesian space
        pos_error = target_pos_base - current_ee_pos

        # Define visual servoing gains
        kv = 0.5  # Velocity gain for visual servoing

        # Calculate desired end-effector velocity
        desired_ee_vel = kv * pos_error

        # Convert Cartesian velocity to joint velocities using Jacobian
        jacobian = self.robot_model.jacobian_for_link(
            current_state['positions'], end_effector_frame)

        # Use pseudo-inverse to solve for joint velocities
        try:
            joint_velocities = np.linalg.pinv(jacobian[:3, :]) @ desired_ee_vel  # Only position part
        except np.linalg.LinAlgError:
            # If matrix is singular, use damped least squares
            damping = 0.01
            joint_velocities = np.linalg.solve(
                jacobian[:3, :].T @ jacobian[:3, :] + damping * np.eye(len(current_state['positions'])),
                jacobian[:3, :].T @ desired_ee_vel
            )

        return joint_velocities

    def transform_camera_to_base(self, camera_coords, camera_frame):
        """Transform coordinates from camera frame to robot base frame"""
        # This would typically use TF transforms
        # For now, return simplified transformation
        # In a real system, you would use tf2_ros to get the transform

        # Get camera pose relative to base (simplified)
        # This would come from URDF or TF tree
        camera_to_base_translation = np.array([0.2, 0.0, 1.5])  # Example: camera 20cm forward, 1.5m up from base
        camera_to_base_rotation = R.from_euler('xyz', [0, 0, 0]).as_matrix()  # No rotation for simplicity

        # Transform point
        world_coords = camera_to_base_rotation @ camera_coords + camera_to_base_translation

        return world_coords

    def perform_grasp_planning_with_vision(self, object_name, approach_direction='top_down'):
        """Plan grasp based on visual object information"""
        if object_name not in self.tracked_objects:
            return None

        object_info = self.tracked_objects[object_name]
        object_pos = object_info['position']

        # Determine grasp approach based on object properties and desired direction
        if approach_direction == 'top_down':
            # Approach from above (common for picking up objects)
            approach_pos = object_pos + np.array([0, 0, 0.2])  # 20cm above object
            approach_orient = R.from_euler('xyz', [0, np.pi, 0]).as_quat()  # Gripper pointing down
        elif approach_direction == 'side':
            # Approach from side
            approach_pos = object_pos + np.array([0.15, 0, 0])  # 15cm in front of object
            approach_orient = R.from_euler('xyz', [0, np.pi/2, 0]).as_quat()  # Gripper pointing toward object
        else:
            # Default approach
            approach_pos = object_pos + np.array([0, 0, 0.1])  # 10cm above object
            approach_orient = R.from_euler('xyz', [0, np.pi, 0]).as_quat()

        # Create grasp trajectory
        grasp_trajectory = [
            {
                'position': approach_pos,
                'orientation': approach_orient,
                'gripper_open': True,
                'description': 'Approach position'
            },
            {
                'position': object_pos + np.array([0, 0, 0.02]),  # Just above object
                'orientation': approach_orient,
                'gripper_open': True,
                'description': 'Pre-grasp position'
            },
            {
                'position': object_pos,  # At object level
                'orientation': approach_orient,
                'gripper_open': True,
                'description': 'Grasp position (gripper open)'
            },
            {
                'position': object_pos,
                'orientation': approach_orient,
                'gripper_open': False,
                'description': 'Grasp closure'
            },
            {
                'position': object_pos + np.array([0, 0, 0.05]),  # Lift object slightly
                'orientation': approach_orient,
                'gripper_open': False,
                'description': 'Lift position'
            }
        ]

        return grasp_trajectory

    def execute_grasp_trajectory(self, trajectory, current_state):
        """Execute a planned grasp trajectory"""
        commands = []

        for waypoint in trajectory:
            # Plan path to waypoint
            target_pose = {
                'position': waypoint['position'],
                'orientation': waypoint['orientation']
            }

            # Use inverse kinematics to find joint positions
            ik_solution = self.robot_model.inverse_kinematics_for_link(
                current_state['positions'], target_pose, 'end_effector')

            if ik_solution is not None:
                # Add gripper command if specified
                command = {
                    'joint_positions': ik_solution,
                    'gripper_command': 'close' if not waypoint['gripper_open'] else 'open',
                    'description': waypoint['description']
                }
                commands.append(command)
            else:
                # If IK fails, try with relaxed constraints or alternative approach
                print(f"IK failed for waypoint: {waypoint['description']}")

        return commands

    def detect_grasp_points_on_object(self, object_image_region):
        """Detect potential grasp points on an object using computer vision"""
        # This would typically use a CNN trained for grasp point detection
        # For now, implement a simple geometric approach

        # Analyze the object region to find good grasp points
        # Look for areas that are suitable for gripping

        # Convert to grayscale
        gray = cv2.cvtColor(object_image_region, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours of the object
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Get the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Find potential grasp points
        grasp_points = []

        # Calculate contour properties
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Consider center as a potential grasp point
            grasp_points.append({
                'x': cx,
                'y': cy,
                'type': 'center',
                'quality': 0.5
            })

        # Find extreme points
        leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

        # Add extreme points as potential grasp locations
        grasp_points.extend([
            {'x': leftmost[0], 'y': leftmost[1], 'type': 'side', 'quality': 0.7},
            {'x': rightmost[0], 'y': rightmost[1], 'type': 'side', 'quality': 0.7},
            {'x': topmost[0], 'y': topmost[1], 'type': 'top', 'quality': 0.6},
            {'x': bottommost[0], 'y': bottommost[1], 'type': 'bottom', 'quality': 0.6}
        ])

        # Filter grasp points based on object geometry and potential for successful grasp
        filtered_grasps = []
        for grasp in grasp_points:
            quality = self._evaluate_grasp_quality_at_point(grasp, largest_contour)
            if quality > 0.3:  # Threshold for acceptable grasp quality
                grasp['quality'] = quality
                filtered_grasps.append(grasp)

        return sorted(filtered_grasps, key=lambda x: x['quality'], reverse=True)

    def _evaluate_grasp_quality_at_point(self, grasp_point, contour):
        """Evaluate the quality of a grasp at a given point"""
        x, y = grasp_point['x'], grasp_point['y']

        # Calculate local curvature at the point
        # This is a simplified approach - in reality, you'd use more sophisticated methods
        point_idx = None
        min_dist = float('inf')

        for i, point in enumerate(contour):
            px, py = point[0]
            dist = (px - x)**2 + (py - y)**2
            if dist < min_dist:
                min_dist = dist
                point_idx = i

        if point_idx is None:
            return 0.0

        # Look at neighboring points to evaluate local geometry
        neighbor_idx1 = (point_idx - 5) % len(contour)
        neighbor_idx2 = (point_idx + 5) % len(contour)

        p1 = contour[neighbor_idx1][0]
        p2 = contour[point_idx][0]
        p3 = contour[neighbor_idx2][0]

        # Calculate angle at the point (more acute angles indicate edges/better grasp points)
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        )

        # Quality based on angle (edges are better for grasping)
        angle_quality = (np.pi - angle) / np.pi  # Higher for sharper angles

        # Also consider distance from center (edge grasps often better)
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Normalize by object size
            object_area = cv2.contourArea(contour)
            size_factor = np.sqrt(object_area) / 2
            normalized_distance = min(distance_from_center / size_factor, 1.0)
            center_distance_quality = normalized_distance
        else:
            center_distance_quality = 0.5

        # Combine factors
        quality = 0.6 * angle_quality + 0.4 * center_distance_quality

        return quality

    def integrate_vision_with_manipulation_control(self, current_state, vision_feedback):
        """Integrate visual feedback with manipulation control"""
        # This would implement visual servoing, object tracking, and adaptive grasping

        control_commands = []

        # If we have tracked objects, adjust manipulation based on visual feedback
        for obj_name, obj_data in self.tracked_objects.items():
            if obj_data['confidence'] > 0.5:  # Only use confident detections
                # Calculate correction based on visual feedback
                correction = self._calculate_visual_correction(obj_data, current_state)

                # Apply correction to manipulation commands
                control_commands.append({
                    'type': 'visual_correction',
                    'correction': correction,
                    'object': obj_name
                })

        return control_commands

    def _calculate_visual_correction(self, object_data, current_state):
        """Calculate correction based on visual tracking"""
        # Calculate the difference between expected and actual object position
        # This would trigger adaptive manipulation behaviors

        # For now, return a simple position correction
        current_ee_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'end_effector')['position']

        visual_error = object_data['position'] - current_ee_pos

        return visual_error * 0.1  # Small correction factor
```

### IMU Integration for Manipulation

Integrating IMU data for whole-body coordination during manipulation:

```python
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import scipy.spatial.transform as st

class IMUIntegratedManipulation:
    """Integrate IMU data for whole-body manipulation coordination"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.gravity = np.array([0, 0, -9.81])

        # IMU state estimation
        self.orientation_estimator = OrientationEstimator()
        self.bias_estimator = BiasEstimator()

        # Balance and manipulation coordination parameters
        self.balance_weight = 0.7
        self.manipulation_weight = 0.3
        self.compliance_threshold = 0.1  # Threshold for switching to compliant behavior

    def process_imu_data(self, imu_msg):
        """Process IMU data for state estimation"""
        # Extract IMU measurements
        angular_velocity = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])

        linear_acceleration = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        orientation = np.array([
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ])

        # Update orientation estimate
        self.orientation_estimator.update(
            angular_velocity, linear_acceleration, orientation)

        # Estimate biases
        self.bias_estimator.update(angular_velocity, linear_acceleration)

        return {
            'orientation': self.orientation_estimator.get_orientation(),
            'angular_velocity': angular_velocity - self.bias_estimator.get_gyro_bias(),
            'linear_acceleration': linear_acceleration - self.bias_estimator.get_accel_bias(),
            'gravity_compensated_acceleration': self.remove_gravity(linear_acceleration, orientation)
        }

    def remove_gravity(self, acceleration, orientation_quat):
        """Remove gravity component from accelerometer readings"""
        # Convert quaternion to rotation matrix
        rotation = R.from_quat(orientation_quat)

        # Rotate gravity vector to sensor frame
        gravity_sensor_frame = rotation.apply(self.gravity)

        # Remove gravity from accelerometer readings
        linear_acceleration_no_gravity = acceleration - gravity_sensor_frame

        return linear_acceleration_no_gravity

    def coordinate_balance_and_manipulation(self, current_state, imu_data, manipulation_command):
        """Coordinate balance and manipulation based on IMU feedback"""
        # Calculate current balance state from IMU data
        balance_state = self.assess_balance_state(imu_data)

        # Determine if we need to prioritize balance over manipulation
        if balance_state['stability_measure'] < self.compliance_threshold:
            # Prioritize balance - reduce manipulation aggressiveness
            adjusted_manipulation = self._reduce_manipulation_aggressiveness(
                manipulation_command, balance_state)
        else:
            # Normal manipulation with balance awareness
            adjusted_manipulation = self._integrate_balance_awareness(
                manipulation_command, balance_state)

        return adjusted_manipulation

    def assess_balance_state(self, imu_data):
        """Assess current balance state from IMU data"""
        # Calculate orientation relative to gravity
        current_orientation = imu_data['orientation']
        current_rot = R.from_quat(current_orientation)

        # Calculate body orientation relative to gravity
        gravity_direction = np.array([0, 0, -1])  # In body frame when upright
        body_up = current_rot.apply(gravity_direction)

        # Calculate tilt angles
        roll_tilt = np.arctan2(body_up[1], body_up[2])
        pitch_tilt = np.arctan2(-body_up[0], np.sqrt(body_up[1]**2 + body_up[2]**2))

        # Calculate angular velocity (indicates instability)
        angular_velocity_magnitude = np.linalg.norm(imu_data['angular_velocity'])

        # Calculate linear acceleration (indicates external forces)
        linear_accel_magnitude = np.linalg.norm(imu_data['linear_acceleration'])

        # Stability measure (lower = more unstable)
        stability_measure = np.exp(-(abs(roll_tilt) + abs(pitch_tilt) +
                                   angular_velocity_magnitude + linear_accel_magnitude))

        # Calculate capture point (where robot would fall if no corrective action taken)
        com_height = 0.8  # Assumed CoM height
        omega = np.sqrt(9.81 / com_height)
        com_velocity = self.estimate_com_velocity_from_imu(imu_data)
        capture_point = com_velocity / omega

        return {
            'roll_tilt': roll_tilt,
            'pitch_tilt': pitch_tilt,
            'angular_velocity': imu_data['angular_velocity'],
            'linear_acceleration': imu_data['linear_acceleration'],
            'stability_measure': stability_measure,
            'capture_point': capture_point,
            'is_unstable': stability_measure < self.compliance_threshold
        }

    def estimate_com_velocity_from_imu(self, imu_data):
        """Estimate CoM velocity from IMU data (simplified)"""
        # This is a simplified estimation - in reality, you'd need more complex fusion
        # with kinematic data from joint encoders

        # For now, return a simple estimate based on IMU integration
        # This would need to be much more sophisticated in practice
        return np.zeros(3)  # Placeholder

    def _reduce_manipulation_aggressiveness(self, manipulation_command, balance_state):
        """Reduce manipulation command aggressiveness when balance is compromised"""
        # Scale down manipulation commands based on instability
        instability_factor = 1.0 - balance_state['stability_measure']
        scaling_factor = max(0.1, 1.0 - instability_factor)  # At least 10% of original command

        # Apply scaling to position commands
        scaled_command = manipulation_command.copy()
        if 'position' in scaled_command:
            # Scale position command toward current position
            current_pos = self.robot_model.forward_kinematics_for_link(
                self.current_joint_positions, 'end_effector')['position']
            scaled_command['position'] = (
                scaling_factor * (scaled_command['position'] - current_pos) + current_pos
            )

        # Reduce velocity and acceleration commands
        if 'velocity' in scaled_command:
            scaled_command['velocity'] *= scaling_factor
        if 'acceleration' in scaled_command:
            scaled_command['acceleration'] *= scaling_factor

        return scaled_command

    def _integrate_balance_awareness(self, manipulation_command, balance_state):
        """Integrate balance awareness into manipulation command"""
        # Modify manipulation command to consider balance
        modified_command = manipulation_command.copy()

        # Adjust CoM position expectation based on manipulation
        if 'target_pose' in manipulation_command:
            # Calculate expected CoM shift due to manipulation
            expected_com_shift = self.estimate_com_shift_for_manipulation(
                manipulation_command['target_pose'])

            # Check if manipulation would compromise balance
            if self.would_compromise_balance(expected_com_shift, balance_state):
                # Adjust manipulation to maintain balance
                modified_command = self.adjust_manipulation_for_balance(
                    manipulation_command, balance_state)

        return modified_command

    def estimate_com_shift_for_manipulation(self, target_pose):
        """Estimate how manipulation will shift CoM"""
        # This would use kinematic model to estimate CoM shift
        # For now, return a simple estimate
        return np.zeros(3)  # Placeholder

    def would_compromise_balance(self, expected_com_shift, balance_state):
        """Check if manipulation would compromise balance"""
        # Estimate new CoM position after manipulation
        current_com = self.robot_model.calculate_com_position(self.current_joint_positions)
        new_com = current_com + expected_com_shift

        # Check if new CoM would be outside support polygon
        # This would require calculating the support polygon from current stance
        # For now, use a simplified check
        max_allowable_shift = 0.1  # 10cm threshold
        return np.linalg.norm(expected_com_shift) > max_allowable_shift

    def adjust_manipulation_for_balance(self, manipulation_command, balance_state):
        """Adjust manipulation command to maintain balance"""
        # This would implement balance-aware manipulation planning
        # For example, using whole-body control to coordinate manipulation with balance
        adjusted_command = manipulation_command.copy()

        # Modify target to keep CoM within stable region
        current_com = self.robot_model.calculate_com_position(self.current_joint_positions)
        support_center = self.estimate_support_center()  # Current support polygon center

        # Calculate allowable manipulation range
        max_shift_from_support = 0.08  # 8cm from support center

        # Adjust target if it would shift CoM too far
        expected_com_shift = self.estimate_com_shift_for_manipulation(
            manipulation_command['target_pose'])
        new_com = current_com + expected_com_shift

        if np.linalg.norm(new_com[:2] - support_center[:2]) > max_shift_from_support:
            # Scale down the manipulation to stay within balance limits
            scale_factor = max_shift_from_support / np.linalg.norm(new_com[:2] - support_center[:2])
            adjusted_target = manipulation_command['target_pose'].copy()

            # Move target closer to current position by scale factor
            current_ee_pos = self.robot_model.forward_kinematics_for_link(
                self.current_joint_positions, 'end_effector')['position']
            adjusted_target['position'] = (
                scale_factor * (adjusted_target['position'] - current_ee_pos) + current_ee_pos
            )

            adjusted_command['target_pose'] = adjusted_target

        return adjusted_command

    def estimate_support_center(self):
        """Estimate center of current support polygon"""
        # This would calculate from current foot positions
        # For now, return a placeholder
        return np.array([0.0, 0.0, 0.0])

    def compute_balance_compensatory_motions(self, balance_state, manipulation_command):
        """Compute compensatory motions to maintain balance during manipulation"""
        # Calculate compensatory motions to counteract manipulation effects
        compensatory_joints = np.zeros(len(self.current_joint_positions))

        if balance_state['is_unstable']:
            # Compute balance recovery motions
            balance_recovery = self._compute_balance_recovery_motion(balance_state)
            compensatory_joints += balance_recovery * self.balance_weight

        # Compute manipulation-compatible balance adjustments
        manipulation_balance = self._compute_manipulation_balance_adjustment(
            manipulation_command, balance_state)
        compensatory_joints += manipulation_balance * self.manipulation_weight

        return compensatory_joints

    def _compute_balance_recovery_motion(self, balance_state):
        """Compute joint motions for balance recovery"""
        # Calculate required body adjustments to restore balance
        required_body_motion = np.zeros(len(self.current_joint_positions))

        # Adjust trunk orientation based on tilt
        if abs(balance_state['roll_tilt']) > 0.05:  # 5 degrees threshold
            # Adjust trunk to counteract roll
            trunk_roll_idx = self.get_joint_index('torso_roll')
            if trunk_roll_idx is not None:
                required_body_motion[trunk_roll_idx] = -balance_state['roll_tilt'] * 0.5

        if abs(balance_state['pitch_tilt']) > 0.05:  # 5 degrees threshold
            # Adjust trunk to counteract pitch
            trunk_pitch_idx = self.get_joint_index('torso_pitch')
            if trunk_pitch_idx is not None:
                required_body_motion[trunk_pitch_idx] = -balance_state['pitch_tilt'] * 0.5

        # Adjust hip joints to shift CoM
        left_hip_roll_idx = self.get_joint_index('left_hip_roll')
        right_hip_roll_idx = self.get_joint_index('right_hip_roll')

        if left_hip_roll_idx is not None and right_hip_roll_idx is not None:
            # Counteract lateral CoM shift
            required_body_motion[left_hip_roll_idx] = -balance_state['capture_point'][1] * 0.1
            required_body_motion[right_hip_roll_idx] = -balance_state['capture_point'][1] * 0.1

        return required_body_motion

    def _compute_manipulation_balance_adjustment(self, manipulation_command, balance_state):
        """Compute balance adjustments compatible with ongoing manipulation"""
        # Calculate how to adjust balance without interfering with manipulation
        adjustment = np.zeros(len(self.current_joint_positions))

        # Use inverse kinematics to find balance-compatible joint configuration
        # that maintains manipulation goal

        if 'target_pose' in manipulation_command:
            target_pose = manipulation_command['target_pose']

            # Use whole-body IK to find configuration that achieves manipulation
            # while maintaining balance
            adjustment = self._whole_body_ik_for_balance_and_manipulation(
                target_pose, balance_state)

        return adjustment

    def _whole_body_ik_for_balance_and_manipulation(self, target_pose, balance_state):
        """Solve whole-body IK considering both manipulation and balance"""
        # This would implement whole-body inverse kinematics with multiple objectives
        # For now, return a simplified solution

        # Calculate desired CoM position for balance
        desired_com = self.calculate_balance_com_position(balance_state)

        # Use prioritized IK to achieve both goals
        # Higher priority: manipulation task
        # Lower priority: balance maintenance

        current_joints = self.current_joint_positions

        # First, solve for manipulation task
        manipulation_solution = self.robot_model.inverse_kinematics_for_link(
            current_joints, target_pose, 'end_effector')

        if manipulation_solution is not None:
            # Then, adjust for balance in null space
            balance_adjustment = self._compute_balance_adjustment_in_nullspace(
                current_joints, manipulation_solution, desired_com)

            return manipulation_solution + balance_adjustment
        else:
            return current_joints  # Return current if no solution found

    def calculate_balance_com_position(self, balance_state):
        """Calculate desired CoM position for balance"""
        # Calculate CoM position that maintains balance given current conditions
        # This would consider support polygon, capture point, and stability margins

        support_center = self.estimate_support_center()

        # Keep CoM close to support center with safety margin
        desired_com = support_center.copy()
        desired_com[2] = 0.8  # Maintain desired CoM height

        return desired_com

    def _compute_balance_adjustment_in_nullspace(self, current_joints, manipulation_joints, desired_com):
        """Compute balance adjustment in the null space of manipulation task"""
        # Calculate Jacobian for balance task (CoM control)
        com_jacobian = self.robot_model.com_jacobian(current_joints)

        # Calculate current and desired CoM
        current_com = self.robot_model.calculate_com_position(current_joints)
        com_error = desired_com - current_com

        # Project CoM Jacobian into null space of manipulation task
        # This is a simplified approach - in practice would use more sophisticated null space projection
        nullspace_adjustment = com_jacobian.T @ com_error * 0.01  # Small gain

        return nullspace_adjustment

    def get_joint_index(self, joint_name):
        """Get index of joint in joint array"""
        # This would look up the joint index from the robot model
        # For now, return a placeholder
        joint_names = self.robot_model.get_joint_names()
        try:
            return joint_names.index(joint_name)
        except ValueError:
            return None  # Joint not found

class OrientationEstimator:
    """Estimate orientation from IMU data using sensor fusion"""

    def __init__(self):
        self.orientation = np.array([0, 0, 0, 1])  # Quaternion [x, y, z, w]
        self.gyro_bias = np.zeros(3)
        self.process_noise = 1e-3
        self.measurement_noise = 1e-2

    def update(self, angular_velocity, linear_acceleration, magnetometer_reading=None):
        """Update orientation estimate with new IMU measurements"""
        dt = 0.01  # Assuming 100Hz update rate

        # Remove bias from gyro measurement
        corrected_angular_velocity = angular_velocity - self.gyro_bias

        # Integrate angular velocity to update orientation
        # Convert to quaternion increment
        omega_norm = np.linalg.norm(corrected_angular_velocity)

        if omega_norm > 1e-6:  # Avoid division by zero
            half_angle = omega_norm * dt / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)

            # Create rotation quaternion
            rotation_quat = np.array([
                corrected_angular_velocity[0] * sin_half / omega_norm,
                corrected_angular_velocity[1] * sin_half / omega_norm,
                corrected_angular_velocity[2] * sin_half / omega_norm,
                cos_half
            ])

            # Apply rotation to current orientation
            self.orientation = self._quaternion_multiply(self.orientation, rotation_quat)
        else:
            # No rotation, keep same orientation
            pass

        # Apply accelerometer correction (tilt correction)
        self._apply_accelerometer_correction(linear_acceleration)

        # Normalize quaternion
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([x, y, z, w])

    def _apply_accelerometer_correction(self, linear_acceleration):
        """Apply accelerometer-based tilt correction"""
        # Calculate expected gravity vector based on current orientation
        current_rot = R.from_quat(self.orientation)
        expected_gravity = current_rot.apply(np.array([0, 0, 9.81]))

        # Calculate measurement error
        measurement_error = linear_acceleration + expected_gravity  # +gravity because accelerometer measures -gravity when static

        # Apply correction with small gain
        correction_gain = 0.01
        correction_quat = self._vector_to_quaternion(
            correction_gain * measurement_error)

        # Apply correction
        self.orientation = self._quaternion_multiply(self.orientation, correction_quat)

        # Normalize
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

    def _vector_to_quaternion(self, vector):
        """Convert 3D vector to quaternion (for small rotations)"""
        angle = np.linalg.norm(vector)

        if angle > 1e-6:
            axis = vector / angle
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)

            return np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                np.cos(half_angle)
            ])
        else:
            return np.array([0, 0, 0, 1])

    def get_orientation(self):
        """Get current orientation estimate"""
        return self.orientation.copy()

class BiasEstimator:
    """Estimate and track IMU sensor biases"""

    def __init__(self):
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.gyro_bias_alpha = 0.001  # Learning rate for gyro bias
        self.accel_bias_alpha = 0.001  # Learning rate for accel bias

        # Assumptions for bias estimation
        self.expected_gravity = np.array([0, 0, 9.81])
        self.stationary_threshold = 0.1  # Threshold for considering robot stationary

    def update(self, angular_velocity, linear_acceleration):
        """Update bias estimates"""
        # Check if robot is likely stationary
        is_stationary = (np.linalg.norm(angular_velocity) < self.stationary_threshold and
                        abs(np.linalg.norm(linear_acceleration) - 9.81) < 0.5)

        if is_stationary:
            # Update gyro bias (should be near zero when stationary)
            self.gyro_bias = (1 - self.gyro_bias_alpha) * self.gyro_bias + \
                           self.gyro_bias_alpha * angular_velocity

            # Update accel bias (should measure gravity when stationary)
            measured_gravity = linear_acceleration
            accel_error = measured_gravity - self.expected_gravity
            self.accel_bias = (1 - self.accel_bias_alpha) * self.accel_bias + \
                            self.accel_bias_alpha * accel_error

    def get_gyro_bias(self):
        """Get current gyro bias estimate"""
        return self.gyro_bias.copy()

    def get_accel_bias(self):
        """Get current accel bias estimate"""
        return self.accel_bias.copy()
```

## LiDAR Integration for Manipulation

### 3D Perception for Manipulation

```python
from sensor_msgs.msg import LaserScan, PointCloud2
import struct

class LidarGuidedManipulation:
    """Use LiDAR data for manipulation planning and obstacle avoidance"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

        # LiDAR parameters
        self.min_range = 0.1
        self.max_range = 10.0
        self.angle_min = -np.pi
        self.angle_max = np.pi
        self.angle_increment = 0.01  # ~0.57 degrees

        # Object detection parameters
        self.cluster_distance_threshold = 0.1
        self.min_cluster_points = 5
        self.max_cluster_points = 1000

    def process_lidar_scan(self, scan_msg):
        """Process LiDAR scan to detect objects and obstacles"""
        # Convert scan to Cartesian points
        points = self.scan_to_cartesian(scan_msg)

        # Cluster points to identify objects
        clusters = self.cluster_points(points)

        # Filter clusters based on size and properties
        valid_clusters = self.filter_clusters(clusters)

        # Extract object information
        objects = self.extract_objects_from_clusters(valid_clusters)

        return objects

    def scan_to_cartesian(self, scan_msg):
        """Convert LiDAR scan to Cartesian coordinates"""
        points = []

        for i, range_val in enumerate(scan_msg.ranges):
            if self.min_range <= range_val <= self.max_range:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment

                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                z = 0.0  # Assume ground-level scan for now

                points.append(np.array([x, y, z]))

        return np.array(points)

    def cluster_points(self, points):
        """Cluster points using DBSCAN algorithm"""
        from sklearn.cluster import DBSCAN

        if len(points) < 2:
            return []

        clustering = DBSCAN(
            eps=self.cluster_distance_threshold,
            min_samples=self.min_cluster_points
        ).fit(points)

        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        clusters = []
        for cluster_id in range(n_clusters):
            cluster_points = points[labels == cluster_id]
            clusters.append(cluster_points)

        return clusters

    def filter_clusters(self, clusters):
        """Filter clusters based on size and properties"""
        valid_clusters = []

        for cluster in clusters:
            if len(cluster) >= self.min_cluster_points and len(cluster) <= self.max_cluster_points:
                # Calculate cluster properties
                centroid = np.mean(cluster, axis=0)
                size = np.max(cluster, axis=0) - np.min(cluster, axis=0)

                # Filter based on reasonable object sizes
                if np.all(size[:2] < 2.0) and np.all(size[:2] > 0.01):  # Between 1cm and 2m
                    valid_clusters.append({
                        'points': cluster,
                        'centroid': centroid,
                        'size': size,
                        'n_points': len(cluster)
                    })

        return valid_clusters

    def extract_objects_from_clusters(self, clusters):
        """Extract object information from clusters"""
        objects = []

        for i, cluster_info in enumerate(clusters):
            # Determine object type based on shape and size
            object_type = self.classify_object(cluster_info)

            objects.append({
                'id': f'object_{i}',
                'type': object_type,
                'position': cluster_info['centroid'],
                'size': cluster_info['size'],
                'confidence': min(1.0, cluster_info['n_points'] / 100),  # Normalize confidence
                'points': cluster_info['points']
            })

        return objects

    def classify_object(self, cluster_info):
        """Classify object based on geometric properties"""
        size = cluster_info['size']

        # Simple classification based on dimensions
        if size[0] < 0.1 and size[1] < 0.1 and size[2] > 0.1:
            return 'pole'  # Tall, thin object
        elif size[0] < 0.3 and size[1] < 0.3 and size[2] < 0.3:
            return 'box'   # Small cube-like object
        elif size[0] > 0.5 or size[1] > 0.5:
            return 'wall'  # Large planar surface
        else:
            return 'object'  # Generic object

    def plan_manipulation_around_obstacles(self, target_object, current_state, detected_objects):
        """Plan manipulation trajectory avoiding detected obstacles"""
        # Get target object position
        target_pos = target_object['position']

        # Get current end-effector position
        current_ee_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'end_effector')['position']

        # Create obstacle representation for path planning
        obstacles = self.create_obstacle_representation(detected_objects, target_object)

        # Plan path avoiding obstacles
        path = self.plan_path_around_obstacles(
            current_ee_pos, target_pos, obstacles)

        return path

    def create_obstacle_representation(self, detected_objects, target_object):
        """Create obstacle representation for path planning"""
        obstacles = []

        for obj in detected_objects:
            if obj['id'] != target_object['id']:  # Don't treat target as obstacle
                # Represent as bounding box
                bbox = {
                    'center': obj['position'],
                    'size': obj['size'],
                    'safety_margin': 0.05  # 5cm safety margin
                }
                obstacles.append(bbox)

        return obstacles

    def plan_path_around_obstacles(self, start_pos, goal_pos, obstacles):
        """Plan path around obstacles using RRT or similar algorithm"""
        # This is a simplified path planning implementation
        # In practice, you'd use a more sophisticated algorithm like RRT*, A*, or PRM

        # For now, implement a simple potential field approach
        path = [start_pos]

        current_pos = start_pos.copy()
        step_size = 0.02  # 2cm steps

        while np.linalg.norm(current_pos - goal_pos) > 0.05:  # 5cm threshold
            # Calculate desired direction
            direction = goal_pos - current_pos
            direction = direction / np.linalg.norm(direction)

            # Calculate step in desired direction
            next_pos = current_pos + direction * step_size

            # Check for obstacle collision
            collision_free = True
            for obstacle in obstacles:
                if self.check_collision(next_pos, obstacle):
                    # Apply repulsive force
                    obs_to_pos = next_pos - obstacle['center']
                    distance = np.linalg.norm(obs_to_pos[:2])  # Only x,y for 2D avoidance

                    if distance < (np.max(obstacle['size'][:2])/2 + obstacle['safety_margin']):
                        # Calculate repulsive direction (away from obstacle)
                        repulsion = obs_to_pos / distance if distance > 0 else np.array([1, 0, 0])

                        # Apply repulsive force perpendicular to desired direction
                        perpendicular = np.array([-direction[1], direction[0], 0])
                        if np.dot(repulsion[:2], perpendicular[:2]) < 0:
                            perpendicular = -perpendicular

                        next_pos = current_pos + (direction * step_size * 0.5 +
                                                perpendicular * step_size * 0.5)
                        collision_free = False
                        break

            if collision_free:
                current_pos = next_pos
            else:
                # If collision detected, try alternative direction
                # For simplicity, we'll just move toward goal with small adjustment
                current_pos = current_pos + direction * step_size * 0.5

            path.append(current_pos.copy())

            # Prevent infinite loops
            if len(path) > 1000:
                break

        path.append(goal_pos)  # Add final goal position
        return np.array(path)

    def check_collision(self, position, obstacle):
        """Check if position collides with obstacle"""
        # Check if position is within obstacle bounding box plus safety margin
        center = obstacle['center']
        size = obstacle['size']
        margin = obstacle['safety_margin']

        # Check each dimension
        for i in range(3):
            if abs(position[i] - center[i]) > (size[i]/2 + margin):
                return False  # Outside this dimension

        return True  # Inside all dimensions (collision)

    def integrate_lidar_with_manipulation_control(self, current_state, lidar_objects, manipulation_goal):
        """Integrate LiDAR feedback with manipulation control"""
        control_adjustments = np.zeros(len(current_state['positions']))

        # Check if any obstacles are in the manipulation path
        for obj in lidar_objects:
            if self.is_object_in_manipulation_path(obj, manipulation_goal, current_state):
                # Adjust manipulation to avoid obstacle
                avoidance_adjustment = self.calculate_obstacle_avoidance_adjustment(
                    obj, manipulation_goal, current_state)

                # Apply adjustment with appropriate weighting
                control_adjustments += avoidance_adjustment * 0.3  # 30% influence

        return control_adjustments

    def is_object_in_manipulation_path(self, object_info, manipulation_goal, current_state):
        """Check if object is in the path of manipulation"""
        current_ee_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'end_effector')['position']
        goal_pos = manipulation_goal['position']

        # Calculate path between current and goal positions
        path_vector = goal_pos - current_ee_pos
        path_length = np.linalg.norm(path_vector)

        if path_length < 0.01:  # Very short path
            return False

        # Normalize path vector
        path_unit = path_vector / path_length

        # Check if object is close to the path
        obj_to_start = object_info['position'] - current_ee_pos
        projection_length = np.dot(obj_to_start, path_unit)

        # Clamp projection to path segment
        projection_length = np.clip(projection_length, 0, path_length)
        closest_point_on_path = current_ee_pos + path_unit * projection_length

        # Calculate distance from object to path
        distance_to_path = np.linalg.norm(object_info['position'] - closest_point_on_path)

        # Consider object in path if within threshold
        threshold = max(0.1, object_info['size'][0]/2 + 0.05)  # Object size + safety margin
        return distance_to_path < threshold

    def calculate_obstacle_avoidance_adjustment(self, obstacle, manipulation_goal, current_state):
        """Calculate joint adjustments to avoid obstacle"""
        # Calculate avoidance direction (away from obstacle)
        current_ee_pos = self.robot_model.forward_kinematics_for_link(
            current_state['positions'], 'end_effector')['position']

        obstacle_direction = current_ee_pos - obstacle['position']
        obstacle_distance = np.linalg.norm(obstacle_direction)

        if obstacle_distance > 0:
            obstacle_direction = obstacle_direction / obstacle_distance
        else:
            obstacle_direction = np.array([1, 0, 0])  # Default direction

        # Calculate required avoidance displacement
        avoidance_distance = max(0.05, 0.1 - obstacle_distance)  # At least 5cm clearance

        # Convert Cartesian avoidance to joint space using Jacobian transpose
        jacobian = self.robot_model.jacobian_for_link(current_state['positions'], 'end_effector')

        # Use transpose Jacobian for simple inverse mapping
        avoidance_joints = jacobian[:3, :].T @ (obstacle_direction * avoidance_distance)

        return avoidance_joints * 0.1  # Small scaling factor for stability

    def create_occupancy_grid_from_lidar(self, scan_msg, grid_resolution=0.05):
        """Create occupancy grid from LiDAR scan for navigation planning"""
        # Calculate grid dimensions based on max range
        grid_size = int(2 * self.max_range / grid_resolution)
        occupancy_grid = np.zeros((grid_size, grid_size))

        # Convert scan points to grid coordinates
        points = self.scan_to_cartesian(scan_msg)

        for point in points:
            # Convert world coordinates to grid coordinates
            grid_x = int((point[0] + self.max_range) / grid_resolution)
            grid_y = int((point[1] + self.max_range) / grid_resolution)

            # Check bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                occupancy_grid[grid_x, grid_y] = 1  # Mark as occupied

        return occupancy_grid

    def detect_free_space_for_manipulation(self, occupancy_grid, robot_radius=0.3):
        """Detect free space around robot for manipulation planning"""
        # Find areas that are free of obstacles and large enough for manipulation
        free_spaces = []

        grid_size = occupancy_grid.shape[0]
        center_x, center_y = grid_size // 2, grid_size // 2  # Robot is at center

        # Use morphological operations to find connected free space
        from scipy.ndimage import binary_erosion, label

        # Invert occupancy grid (free = 1, occupied = 0)
        free_space = 1 - occupancy_grid

        # Erode to account for robot radius
        structuring_element = np.ones((int(robot_radius*2/grid_resolution),
                                     int(robot_radius*2/grid_resolution)))
        free_space_eroded = binary_erosion(free_space, structure=structuring_element)

        # Find connected components
        labeled_free_space, n_components = label(free_space_eroded)

        # Extract centroids of free space regions
        for i in range(1, n_components + 1):
            region_mask = (labeled_free_space == i)
            y_coords, x_coords = np.where(region_mask)

            centroid_x = np.mean(x_coords) * grid_resolution - self.max_range
            centroid_y = np.mean(y_coords) * grid_resolution - self.max_range

            # Only consider regions that are sufficiently large
            if len(x_coords) > (robot_radius / grid_resolution)**2:
                free_spaces.append({
                    'position': np.array([centroid_x, centroid_y, 0]),
                    'area': len(x_coords) * grid_resolution**2,
                    'valid': True
                })

        return free_spaces
```

## Multi-Sensor Fusion for Manipulation

### Sensor Fusion Framework

```python
class SensorFusionForManipulation:
    """Fusion framework for combining camera, LiDAR, and IMU data for manipulation"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

        # Sensor data buffers
        self.camera_data = None
        self.lidar_data = None
        self.imu_data = None

        # Timestamps for synchronization
        self.camera_timestamp = None
        self.lidar_timestamp = None
        self.imu_timestamp = None

        # State estimation
        self.state_estimator = ExtendedKalmanFilter()
        self.object_tracker = MultiObjectTracker()

    def update_sensor_data(self, camera_msg=None, lidar_msg=None, imu_msg=None):
        """Update sensor data and perform fusion"""
        if camera_msg is not None:
            self.camera_data = camera_msg
            self.camera_timestamp = camera_msg.header.stamp

        if lidar_msg is not None:
            self.lidar_data = lidar_msg
            self.lidar_timestamp = lidar_msg.header.stamp

        if imu_msg is not None:
            self.imu_data = self.process_imu_data(imu_msg)
            self.imu_timestamp = imu_msg.header.stamp

    def process_imu_data(self, imu_msg):
        """Process IMU data for fusion"""
        # This would use the IMU processing from earlier section
        # Return processed IMU data with orientation, angular velocity, etc.
        return {
            'orientation': np.array([imu_msg.orientation.x, imu_msg.orientation.y,
                                   imu_msg.orientation.z, imu_msg.orientation.w]),
            'angular_velocity': np.array([imu_msg.angular_velocity.x,
                                        imu_msg.angular_velocity.y,
                                        imu_msg.angular_velocity.z]),
            'linear_acceleration': np.array([imu_msg.linear_acceleration.x,
                                           imu_msg.linear_acceleration.y,
                                           imu_msg.linear_acceleration.z])
        }

    def fuse_sensor_data(self, current_state):
        """Fuse sensor data for manipulation planning"""
        # Perform temporal synchronization of sensor data
        synced_data = self.synchronize_sensor_data()

        # Perform spatial registration (transform all data to common frame)
        registered_data = self.register_sensor_data(synced_data)

        # Update state estimation
        self.update_state_estimate(registered_data, current_state)

        # Update object tracking
        self.update_object_tracking(registered_data)

        # Generate fused perception output
        fused_perception = self.generate_fused_perception(registered_data)

        return fused_perception

    def synchronize_sensor_data(self):
        """Synchronize sensor data temporally"""
        # Find common time window for all sensors
        if self.camera_timestamp is None or self.lidar_timestamp is None or self.imu_timestamp is None:
            return None

        # Use the most recent timestamp as reference
        latest_time = max(
            self.camera_timestamp.sec + self.camera_timestamp.nanosec * 1e-9,
            self.lidar_timestamp.sec + self.lidar_timestamp.nanosec * 1e-9,
            self.imu_timestamp.sec + self.imu_timestamp.nanosec * 1e-9
        )

        # For now, return current data (in practice, would interpolate to common time)
        return {
            'camera': self.camera_data,
            'lidar': self.lidar_data,
            'imu': self.imu_data
        }

    def register_sensor_data(self, synced_data):
        """Register sensor data to common coordinate frame"""
        # Transform all sensor data to robot base frame
        # This would use TF transforms in a real system
        registered = {}

        # Camera data transformation
        if synced_data['camera'] is not None:
            # Convert camera image to 3D points and transform to base frame
            camera_objects = self.process_camera_image(synced_data['camera'])
            registered['camera_objects'] = self.transform_to_base_frame(
                camera_objects, 'camera_link')

        # LiDAR data transformation
        if synced_data['lidar'] is not None:
            lidar_objects = self.process_lidar_scan(synced_data['lidar'])
            registered['lidar_objects'] = self.transform_to_base_frame(
                lidar_objects, 'lidar_link')

        # IMU data transformation
        if synced_data['imu'] is not None:
            registered['imu_data'] = synced_data['imu']

        return registered

    def transform_to_base_frame(self, objects, source_frame):
        """Transform objects from source frame to robot base frame"""
        # This would use TF transforms in a real system
        # For now, return objects unchanged
        return objects

    def update_state_estimate(self, registered_data, current_state):
        """Update robot state estimate using sensor fusion"""
        # Use extended Kalman filter or other fusion algorithm
        measurement_vector = self.create_measurement_vector(registered_data)

        # Predict state using robot kinematics
        predicted_state = self.predict_state(current_state)

        # Update state estimate with measurements
        updated_state = self.state_estimator.update(
            predicted_state, measurement_vector)

        return updated_state

    def create_measurement_vector(self, registered_data):
        """Create measurement vector from registered sensor data"""
        measurements = []

        # Add IMU-based orientation measurements
        if 'imu_data' in registered_data:
            imu_orient = registered_data['imu_data']['orientation']
            measurements.extend(imu_orient)

        # Add camera-based position measurements for tracked objects
        if 'camera_objects' in registered_data:
            for obj in registered_data['camera_objects']:
                measurements.extend(obj['position'])

        # Add LiDAR-based position measurements
        if 'lidar_objects' in registered_data:
            for obj in registered_data['lidar_objects']:
                measurements.extend(obj['position'])

        return np.array(measurements)

    def predict_state(self, current_state):
        """Predict next state using robot dynamics model"""
        # Use current joint positions, velocities, and controls to predict next state
        # This would involve forward dynamics simulation
        return current_state  # Simplified prediction

    def update_object_tracking(self, registered_data):
        """Update multi-object tracking using fused sensor data"""
        # Combine object detections from different sensors
        all_detections = []

        if 'camera_objects' in registered_data:
            all_detections.extend(registered_data['camera_objects'])

        if 'lidar_objects' in registered_data:
            all_detections.extend(registered_data['lidar_objects'])

        # Perform data association and track management
        self.object_tracker.update_tracks(all_detections)

    def generate_fused_perception(self, registered_data):
        """Generate fused perception output for manipulation"""
        # Combine information from all sensors
        fused_objects = []

        # Get tracked objects
        tracked_objects = self.object_tracker.get_tracked_objects()

        for obj_id, track in tracked_objects.items():
            # Combine measurements from different sensors
            fused_object = {
                'id': obj_id,
                'position': track['position'],
                'velocity': track['velocity'],
                'size': track['size'],
                'type': track['type'],
                'confidence': track['confidence'],
                'sensor_sources': track.get('sensor_sources', [])
            }

            fused_objects.append(fused_object)

        return {
            'objects': fused_objects,
            'robot_state': self.get_fused_robot_state(registered_data),
            'environment_map': self.create_fused_environment_map(registered_data)
        }

    def get_fused_robot_state(self, registered_data):
        """Get fused estimate of robot state"""
        # Combine joint encoder data with IMU data for best estimate
        # This would typically involve sensor fusion algorithms
        return {
            'position': np.array([0, 0, 0]),  # Would come from localization
            'orientation': registered_data.get('imu_data', {}).get('orientation', [0, 0, 0, 1]),
            'velocity': np.array([0, 0, 0]),
            'angular_velocity': registered_data.get('imu_data', {}).get('angular_velocity', [0, 0, 0])
        }

    def create_fused_environment_map(self, registered_data):
        """Create fused environment map from all sensors"""
        # Combine camera, LiDAR, and other sensor data into unified map
        # This would typically create a 3D occupancy grid or point cloud
        fused_map = {}

        # Add static obstacles from LiDAR
        if 'lidar_objects' in registered_data:
            fused_map['static_obstacles'] = registered_data['lidar_objects']

        # Add dynamic objects from camera tracking
        if 'camera_objects' in registered_data:
            fused_map['dynamic_objects'] = registered_data['camera_objects']

        # Add traversable areas based on fused data
        fused_map['traversable_areas'] = self.calculate_traversable_areas(registered_data)

        return fused_map

    def calculate_traversable_areas(self, registered_data):
        """Calculate areas suitable for manipulation based on fused sensor data"""
        # Determine free space that's accessible for manipulation
        # Consider robot reach, obstacle locations, and surface properties

        traversable_areas = []

        # For each object, determine if it's in a manipulable location
        all_objects = []
        if 'lidar_objects' in registered_data:
            all_objects.extend(registered_data['lidar_objects'])
        if 'camera_objects' in registered_data:
            all_objects.extend(registered_data['camera_objects'])

        for obj in all_objects:
            if self.is_manipulable_object(obj):
                traversable_areas.append({
                    'position': obj['position'],
                    'reachable': self.is_reachable(obj['position']),
                    'stable_surface': self.has_stable_surface_nearby(obj['position']),
                    'object_type': obj['type']
                })

        return traversable_areas

    def is_manipulable_object(self, obj):
        """Check if object is suitable for manipulation"""
        # Check if object size and position are appropriate for manipulation
        if obj['size'][0] > 0.5 or obj['size'][1] > 0.5 or obj['size'][2] > 0.5:
            return False  # Too large

        if obj['size'][0] < 0.01 or obj['size'][1] < 0.01 or obj['size'][2] < 0.01:
            return False  # Too small

        # Check if object is at appropriate height
        if obj['position'][2] < 0.1 or obj['position'][2] > 1.5:
            return False  # Too low or too high

        return True

    def is_reachable(self, position):
        """Check if position is reachable by robot arm"""
        # Use forward kinematics to check reachability
        # This would check if the position is within the robot's workspace
        current_joints = self.current_joint_positions  # Would come from current state
        ee_pos = self.robot_model.forward_kinematics_for_link(
            current_joints, 'end_effector')['position']

        # Simple distance check (in practice, would use full IK)
        distance = np.linalg.norm(position - ee_pos)
        return distance < 1.0  # Assuming 1m reach

    def has_stable_surface_nearby(self, position):
        """Check if there's a stable surface near the object for manipulation"""
        # Check LiDAR data for nearby planar surfaces (tables, floors, etc.)
        # This would look for clusters of points that form a stable surface
        return True  # Simplified check

class ExtendedKalmanFilter:
    """Simple EKF implementation for state estimation"""

    def __init__(self, state_dim=12, measurement_dim=6):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State: [position, orientation, velocity, angular_velocity]
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim) * 0.1

        # Process and measurement noise
        self.process_noise = np.eye(state_dim) * 0.01
        self.measurement_noise = np.eye(measurement_dim) * 0.1

    def predict(self, control_input=None, dt=0.01):
        """Predict next state"""
        # Simplified prediction - in practice would use detailed dynamics model
        # This is just a placeholder for the concept
        pass

    def update(self, predicted_state, measurement):
        """Update state estimate with measurement"""
        # Simplified update - in practice would implement full EKF equations
        return predicted_state  # Return prediction for now

class MultiObjectTracker:
    """Track multiple objects using sensor data"""

    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_track_age = 10  # Maximum age before track deletion
        self.min_hits_to_confirm = 3  # Minimum hits before track is confirmed

    def update_tracks(self, detections):
        """Update object tracks with new detections"""
        # Implement data association and tracking
        # This would typically use algorithms like Kalman filters or particle filters
        for detection in detections:
            # For now, just store the detection as a track
            if detection['id'] not in self.tracks:
                # Create new track
                self.tracks[detection['id']] = {
                    'position': detection['position'],
                    'velocity': np.zeros(3),
                    'size': detection['size'],
                    'type': detection['type'],
                    'hits': 1,
                    'age': 0,
                    'confirmed': False
                }
            else:
                # Update existing track
                track = self.tracks[detection['id']]
                track['position'] = detection['position']
                track['size'] = detection['size']
                track['age'] += 1
                track['hits'] += 1

                # Confirm track if it has enough hits
                if track['hits'] >= self.min_hits_to_confirm and not track['confirmed']:
                    track['confirmed'] = True

    def get_tracked_objects(self):
        """Get all confirmed tracked objects"""
        confirmed_tracks = {}
        for obj_id, track in self.tracks.items():
            if track['confirmed']:
                confirmed_tracks[obj_id] = track

        return confirmed_tracks
```

## Performance Optimization

### Sensor Data Processing Optimization

```python
class OptimizedSensorProcessing:
    """Optimized sensor processing for real-time manipulation"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

        # Pre-allocate arrays to avoid allocation during processing
        self.temp_arrays = {
            'scan_points': np.zeros((10000, 3)),
            'cluster_points': np.zeros((1000, 3)),
            'transform_matrix': np.eye(4)
        }

        # Threading for parallel processing
        import threading
        self.processing_lock = threading.Lock()

        # Downsampling parameters
        self.scan_downsample_factor = 2  # Process every 2nd scan point
        self.image_downsample_factor = 4  # Process quarter-resolution images

        # Caching for expensive computations
        self.jacobian_cache = {}
        self.kinematics_cache = {}

    def process_camera_image_optimized(self, image_msg):
        """Optimized camera image processing"""
        with self.processing_lock:
            # Downsample image for faster processing
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            if self.image_downsample_factor > 1:
                height, width = cv_image.shape[:2]
                new_width = width // self.image_downsample_factor
                new_height = height // self.image_downsample_factor
                cv_image = cv2.resize(cv_image, (new_width, new_height))

            # Use optimized detection algorithm
            detected_objects = self.fast_detect_objects_in_image(cv_image)

            # Upscale coordinates back to original image size
            for obj in detected_objects:
                obj['pixel_coords'] = (
                    obj['pixel_coords'][0] * self.image_downsample_factor,
                    obj['pixel_coords'][1] * self.image_downsample_factor
                )

            return detected_objects

    def fast_detect_objects_in_image(self, cv_image):
        """Fast object detection using optimized algorithms"""
        # Use faster but less accurate detection for real-time performance
        # This might use simpler color thresholding or template matching

        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Use predefined color ranges for common objects
        color_ranges = [
            ((0, 50, 50), (10, 255, 255)),    # Red (lower)
            ((170, 50, 50), (180, 255, 255)), # Red (upper)
            ((40, 50, 50), (80, 255, 255)),   # Green
            ((100, 50, 50), (130, 255, 255))  # Blue
        ]

        detected_objects = []

        for i, (lower, upper) in enumerate(color_ranges):
            # Create mask for color range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Use faster contour detection
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum area threshold (reduced for speed)
                    # Calculate bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_u = x + w // 2
                    center_v = y + h // 2

                    detected_objects.append({
                        'name': f'object_{i}',
                        'position': np.array([center_u, center_v, 0]),  # Will be converted to 3D later
                        'pixel_coords': (center_u, center_v),
                        'bounding_box': (x, y, w, h),
                        'confidence': min(1.0, area / 1000)  # Normalized by expected max area
                    })

        return detected_objects

    def process_lidar_scan_optimized(self, scan_msg):
        """Optimized LiDAR scan processing"""
        with self.processing_lock:
            # Pre-allocate result array
            points = self.temp_arrays['scan_points']
            valid_count = 0

            # Process scan with downsampling
            for i in range(0, len(scan_msg.ranges), self.scan_downsample_factor):
                range_val = scan_msg.ranges[i]
                if self.min_range <= range_val <= self.max_range:
                    angle = scan_msg.angle_min + i * scan_msg.angle_increment * self.scan_downsample_factor

                    x = range_val * np.cos(angle)
                    y = range_val * np.sin(angle)
                    z = 0.0

                    points[valid_count] = [x, y, z]
                    valid_count += 1

                    if valid_count >= len(points):
                        break  # Prevent overflow

            # Use optimized clustering algorithm
            clusters = self.fast_cluster_points(points[:valid_count])

            # Process clusters
            valid_clusters = self.filter_clusters_optimized(clusters)
            objects = self.extract_objects_optimized(valid_clusters)

            return objects

    def fast_cluster_points(self, points):
        """Fast clustering using simplified algorithm"""
        # Use grid-based clustering instead of DBSCAN for speed
        grid_size = 0.1  # 10cm grid cells

        # Create grid and assign points to cells
        grid = {}
        for point in points:
            grid_x = int(point[0] / grid_size)
            grid_y = int(point[1] / grid_size)
            grid_key = (grid_x, grid_y)

            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append(point)

        # Return clusters (grid cells with multiple points)
        clusters = []
        for cell_points in grid.values():
            if len(cell_points) >= self.min_cluster_points:
                clusters.append(np.array(cell_points))

        return clusters

    def filter_clusters_optimized(self, clusters):
        """Optimized cluster filtering"""
        valid_clusters = []

        for cluster in clusters:
            if len(cluster) >= self.min_cluster_points:
                centroid = np.mean(cluster, axis=0)
                size = np.max(cluster, axis=0) - np.min(cluster, axis=0)

                # Quick size check
                if all(s < 2.0 for s in size[:2]) and all(s > 0.01 for s in size[:2]):
                    valid_clusters.append({
                        'points': cluster,
                        'centroid': centroid,
                        'size': size,
                        'n_points': len(cluster)
                    })

        return valid_clusters

    def extract_objects_optimized(self, clusters):
        """Optimized object extraction"""
        objects = []

        for i, cluster_info in enumerate(clusters):
            # Simplified classification
            size = cluster_info['size']
            if size[0] < 0.1 and size[1] < 0.1 and size[2] > 0.1:
                obj_type = 'pole'
            elif size[0] < 0.3 and size[1] < 0.3 and size[2] < 0.3:
                obj_type = 'box'
            elif size[0] > 0.5 or size[1] > 0.5:
                obj_type = 'wall'
            else:
                obj_type = 'object'

            objects.append({
                'id': f'object_{i}',
                'type': obj_type,
                'position': cluster_info['centroid'],
                'size': cluster_info['size'],
                'confidence': min(1.0, cluster_info['n_points'] / 100),
                'points': cluster_info['points']
            })

        return objects

    def use_cached_kinematics(self, joint_positions, query_type):
        """Use cached kinematics results when possible"""
        # Create hash of joint positions
        joint_hash = tuple(np.round(joint_positions, decimals=3))

        if joint_hash in self.kinematics_cache:
            if query_type in self.kinematics_cache[joint_hash]:
                return self.kinematics_cache[joint_hash][query_type]

        # Calculate and cache result
        if query_type == 'forward_kinematics':
            result = self.robot_model.calculate_forward_kinematics(joint_positions)
        elif query_type == 'jacobian':
            result = self.robot_model.calculate_jacobian(joint_positions)
        elif query_type == 'com':
            result = self.robot_model.calculate_com_position(joint_positions)
        else:
            return None

        # Store in cache (limit cache size)
        if joint_hash not in self.kinematics_cache:
            self.kinematics_cache[joint_hash] = {}
        self.kinematics_cache[joint_hash][query_type] = result

        # Limit cache size to prevent memory issues
        if len(self.kinematics_cache) > 100:
            # Remove oldest entries (in a real implementation, use LRU)
            oldest_key = next(iter(self.kinematics_cache))
            del self.kinematics_cache[oldest_key]

        return result

    def multi_threaded_sensor_processing(self, sensor_messages):
        """Process multiple sensor messages in parallel"""
        import concurrent.futures

        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks for parallel processing
            futures = {}

            if 'camera' in sensor_messages:
                futures['camera'] = executor.submit(
                    self.process_camera_image_optimized, sensor_messages['camera'])

            if 'lidar' in sensor_messages:
                futures['lidar'] = executor.submit(
                    self.process_lidar_scan_optimized, sensor_messages['lidar'])

            if 'imu' in sensor_messages:
                futures['imu'] = executor.submit(
                    self.process_imu_data, sensor_messages['imu'])

            # Collect results
            for sensor_type, future in futures.items():
                try:
                    results[sensor_type] = future.result(timeout=0.1)  # 100ms timeout
                except concurrent.futures.TimeoutError:
                    print(f"Warning: {sensor_type} processing timed out")
                    results[sensor_type] = None

        return results
```

## Summary

Sensor simulation is a cornerstone of effective Physical AI development, especially for humanoid robots that must perceive and interact with their environment. This chapter has covered:

1. **Camera Simulation**: Techniques for realistic image generation, object detection, and visual servoing
2. **LiDAR Simulation**: Point cloud generation, clustering algorithms, and 3D environment mapping
3. **IMU Integration**: Orientation estimation, bias correction, and balance coordination
4. **Multi-Sensor Fusion**: Combining different sensor modalities for enhanced perception
5. **Performance Optimization**: Techniques for real-time sensor processing

The key to successful sensor simulation is balancing realism with computational efficiency. High-fidelity simulation provides more accurate training data but requires more computational resources. The goal is to create simulations that are realistic enough to bridge the reality gap while remaining efficient enough for practical development and testing workflows.

By properly integrating these sensors with your humanoid robot model and control systems, you can create sophisticated manipulation behaviors that respond appropriately to sensory feedback, enabling your robots to operate effectively in complex, dynamic environments.