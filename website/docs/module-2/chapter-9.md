---
title: Chapter 9 - Walking Pattern Generation and Gait Control
description: Advanced techniques for generating walking patterns and controlling humanoid gait
sidebar_position: 9
---

# Chapter 9: Walking Pattern Generation and Gait Control

## Learning Objectives

After completing this chapter, you should be able to:
- Generate stable walking patterns using various gait generation techniques
- Implement ZMP-based gait control for humanoid robots
- Design footstep planning algorithms for navigation
- Create smooth transition patterns between different walking gaits
- Integrate gait control with balance and locomotion systems

## Introduction

Walking pattern generation and gait control form the foundation of bipedal locomotion for humanoid robots. Unlike wheeled robots that can rely on continuous ground contact, humanoid robots must dynamically manage their balance while walking, alternating between single and double support phases. This chapter covers the fundamental techniques for generating stable and efficient walking patterns that enable humanoid robots to navigate their environment safely and effectively.

## Fundamentals of Humanoid Walking

### Walking Phases

Humanoid walking consists of several distinct phases:

1. **Double Support Phase (DSP)**: Both feet are in contact with the ground
2. **Single Support Phase (SSP)**: Only one foot is in contact with the ground
3. **Transition Phase**: Moving from DSP to SSP or vice versa

#### Timing Considerations

```python
class WalkingPhaseDetector:
    """Detect walking phases based on foot contact and timing"""

    def __init__(self, dsp_ratio=0.1, step_duration=0.8):
        self.dsp_ratio = dsp_ratio  # Ratio of DSP in each step
        self.step_duration = step_duration
        self.cycle_time = 0.0

    def detect_phase(self, time_in_cycle, left_contact, right_contact):
        """Detect current walking phase"""
        phase_progress = time_in_cycle % self.step_duration / self.step_duration

        if phase_progress < self.dsp_ratio or phase_progress > (1 - self.dsp_ratio):
            return 'DOUBLE_SUPPORT'
        else:
            # Determine which foot is in single support
            if phase_progress < 0.5:
                return 'LEFT_SINGLE_SUPPORT' if left_contact and not right_contact else 'RIGHT_SINGLE_SUPPORT'
            else:
                return 'RIGHT_SINGLE_SUPPORT' if right_contact and not left_contact else 'LEFT_SINGLE_SUPPORT'
```

### Gait Parameters

Key parameters that define walking gait:

```python
class GaitParameters:
    def __init__(self):
        # Basic gait parameters
        self.step_length = 0.30    # Forward step distance (meters)
        self.step_width = 0.20    # Lateral foot separation (meters)
        self.step_height = 0.05   # Foot lift height (meters)
        self.step_duration = 0.8  # Total time per step (seconds)

        # Support phase timing
        self.dsp_ratio = 0.1      # Double support phase ratio
        self.ssp_ratio = 0.8      # Single support phase ratio

        # Balance parameters
        self.com_height = 0.8     # Desired CoM height (meters)
        self.zmp_margin = 0.05    # Safety margin for ZMP (meters)

        # Walking speed parameters
        self.nominal_speed = 0.3  # Nominal walking speed (m/s)
        self.max_speed = 0.6      # Maximum safe walking speed (m/s)
        self.min_speed = 0.0      # Minimum speed (including standing)

    def adjust_for_terrain(self, terrain_slope, terrain_roughness):
        """Adjust gait parameters based on terrain characteristics"""
        adjusted_params = self.__dict__.copy()

        # Reduce step length on slopes
        adjusted_params['step_length'] *= (1.0 - abs(terrain_slope) * 0.3)

        # Increase DSP ratio on rough terrain
        adjusted_params['dsp_ratio'] = min(0.2, self.dsp_ratio + terrain_roughness * 0.1)

        # Increase step height for rough terrain
        adjusted_params['step_height'] = max(0.05, self.step_height + terrain_roughness * 0.02)

        # Reduce speed on challenging terrain
        speed_factor = (1.0 - abs(terrain_slope) * 0.2 - terrain_roughness * 0.1)
        adjusted_params['nominal_speed'] *= max(0.1, speed_factor)

        return adjusted_params
```

## ZMP-Based Walking Pattern Generation

### Preview Control Method

The preview control method uses future ZMP references to generate stable walking patterns:

```python
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import lfilter

class ZMPPreviewController:
    """ZMP preview controller for walking pattern generation"""

    def __init__(self, com_height=0.8, sampling_time=0.01, preview_window=2.0):
        self.com_height = com_height
        self.sampling_time = sampling_time
        self.preview_time = preview_window
        self.gravity = 9.81

        # Calculate preview steps
        self.preview_steps = int(preview_window / sampling_time)

        # Inverted pendulum parameters
        self.omega = np.sqrt(self.gravity / self.com_height)

        # State-space representation for inverted pendulum
        # State: [x, y, x_dot, y_dot]
        self.A = np.array([
            [1, 0, self.sampling_time, 0],
            [0, 1, 0, self.sampling_time],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.B = np.array([
            [self.sampling_time**2 * self.omega**2 / 2, 0],
            [0, self.sampling_time**2 * self.omega**2 / 2],
            [self.sampling_time * self.omega**2, 0],
            [0, self.sampling_time * self.omega**2]
        ])

        # Cost matrices for LQR
        self.Q = np.diag([100.0, 100.0, 1.0, 1.0])  # State cost
        self.R = np.diag([0.1, 0.1])                 # Control cost (ZMP)

        # Calculate LQR gains
        self.K, self.P = self.calculate_lqr_gains()

    def calculate_lqr_gains(self):
        """Calculate LQR gains for the system"""
        # Solve discrete algebraic Riccati equation
        P = solve_discrete_are(self.A.T, self.B.T, self.Q, self.R)

        # Calculate feedback gain
        K = np.linalg.inv(self.B.T @ P @ self.B + self.R) @ self.B.T @ P @ self.A

        return K, P

    def generate_walking_pattern(self, zmp_reference_trajectory, initial_state):
        """
        Generate CoM trajectory using ZMP preview control

        Args:
            zmp_reference_trajectory: Array of [x, y] ZMP references over time
            initial_state: Initial state [com_x, com_y, com_dx, com_dy]

        Returns:
            com_trajectory: Array of [x, y, z] CoM positions over time
        """
        n_steps = len(zmp_reference_trajectory)
        com_trajectory = np.zeros((n_steps, 3))

        # Initialize state
        state = initial_state.copy()
        com_trajectory[0, :2] = state[:2]  # x, y position
        com_trajectory[0, 2] = self.com_height  # z position (constant)

        # Implement preview control
        for i in range(1, n_steps):
            # Calculate error between current state and desired ZMP
            current_zmp = state[:2] - (state[2:4] / self.omega**2)
            desired_zmp = zmp_reference_trajectory[i]

            # Calculate control input using LQR + preview compensation
            state_error = np.concatenate([desired_zmp, [0, 0]]) - state
            control_input = -self.K @ state_error

            # Add preview compensation term
            preview_compensation = self.calculate_preview_compensation(
                zmp_reference_trajectory, i)

            control_input += preview_compensation

            # Update state using system dynamics
            state = self.A @ state + self.B @ control_input

            # Store CoM position
            com_trajectory[i, :2] = state[:2]
            com_trajectory[i, 2] = self.com_height

        return com_trajectory

    def calculate_preview_compensation(self, zmp_trajectory, current_idx):
        """Calculate preview compensation term"""
        compensation = np.zeros(2)

        # Sum over preview horizon with exponentially decaying weights
        for k in range(1, min(self.preview_steps, len(zmp_trajectory) - current_idx)):
            # Calculate preview gain for step k ahead
            Ak = np.linalg.matrix_power(self.A, k)
            Bk = self.calculate_matrix_power_sum(self.A, self.B, k)

            # Weight based on how far in the future (more recent = higher weight)
            weight = np.exp(-k * 0.1)  # Decay factor

            future_error = zmp_trajectory[current_idx + k] - zmp_trajectory[current_idx + k - 1]
            compensation += weight * (Ak @ self.B)[:2] @ future_error

        return compensation

    def calculate_matrix_power_sum(self, A, B, k):
        """Calculate sum of A^i * B for i = 0 to k-1"""
        result = B.copy()
        A_power = A.copy()

        for i in range(1, k):
            result += A_power @ B
            A_power = A_power @ A

        return result

    def generate_zmp_trajectory_for_walking(self, step_locations, step_times):
        """
        Generate ZMP reference trajectory for given footsteps

        Args:
            step_locations: Array of [x, y, theta] for each footstep
            step_times: Times when each step occurs

        Returns:
            zmp_trajectory: ZMP reference over time
        """
        total_time = step_times[-1]
        n_samples = int(total_time / self.sampling_time)
        time_vector = np.linspace(0, total_time, n_samples)

        zmp_trajectory = np.zeros((n_samples, 2))

        # Initialize with first support foot position
        current_support_foot = step_locations[0][:2]  # x, y of first step

        for i, t in enumerate(time_vector):
            # Determine current phase and support foot
            current_step_idx = 0
            for idx, step_time in enumerate(step_times):
                if t >= step_time:
                    current_step_idx = idx
                else:
                    break

            # Determine if in DSP or SSP
            if current_step_idx < len(step_times) - 1:
                next_step_time = step_times[current_step_idx + 1]
                time_in_step = t - step_times[current_step_idx]
                step_duration = next_step_time - step_times[current_step_idx]

                step_phase = time_in_step / step_duration

                if step_phase < self.dsp_ratio:
                    # First DSP - transition from previous support to current
                    if current_step_idx > 0:
                        prev_foot = step_locations[current_step_idx - 1][:2]
                        current_foot = step_locations[current_step_idx][:2]
                        alpha = step_phase / self.dsp_ratio
                        current_support_foot = (1 - alpha) * prev_foot + alpha * current_foot
                    else:
                        current_support_foot = step_locations[current_step_idx][:2]
                elif step_phase > (1 - self.dsp_ratio):
                    # Second DSP - transition to next support foot
                    next_foot = step_locations[current_step_idx + 1][:2]
                    alpha = (step_phase - (1 - self.dsp_ratio)) / self.dsp_ratio
                    current_support_foot = (1 - alpha) * step_locations[current_step_idx][:2] + alpha * next_foot
                else:
                    # SSP - ZMP follows support foot with small oscillations
                    current_support_foot = step_locations[current_step_idx][:2]

                    # Add small oscillation for dynamic balance
                    ssp_phase = (step_phase - self.dsp_ratio) / (1 - 2 * self.dsp_ratio)
                    oscillation = 0.01 * np.array([
                        np.sin(ssp_phase * 2 * np.pi),
                        np.cos(ssp_phase * 2 * np.pi)
                    ])
                    current_support_foot += oscillation

            zmp_trajectory[i] = current_support_foot

        return zmp_trajectory
```

## Footstep Planning Algorithms

### A* Based Footstep Planner

For navigating complex terrains:

```python
import heapq
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class FootstepNode:
    """Node for footstep planning"""
    x: float
    y: float
    yaw: float
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    parent: 'FootstepNode' = None

    def __lt__(self, other):
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)

class FootstepPlanner:
    """A* based footstep planner for humanoid navigation"""

    def __init__(self, step_length=0.3, step_width=0.2, max_turn=np.pi/4):
        self.step_length = step_length
        self.step_width = step_width
        self.max_turn = max_turn  # Maximum turn per step
        self.resolution = 0.1     # Grid resolution for planning
        self.collision_threshold = 0.1  # Minimum distance to obstacles

    def plan_footsteps(self, start_pose, goal_pose, terrain_map):
        """
        Plan footsteps from start to goal using A* algorithm

        Args:
            start_pose: [x, y, theta] of starting position
            goal_pose: [x, y, theta] of goal position
            terrain_map: Grid map with obstacle information

        Returns:
            List of footsteps [x, y, theta] or None if no path found
        """
        start_node = FootstepNode(
            x=start_pose[0], y=start_pose[1], yaw=start_pose[2],
            g_cost=0, h_cost=self.heuristic(start_pose, goal_pose)
        )

        open_list = [start_node]
        closed_set = set()

        # Convert to grid coordinates for faster lookup
        grid_start = self.pose_to_grid(start_pose)
        grid_goal = self.pose_to_grid(goal_pose)

        while open_list:
            current = heapq.heappop(open_list)

            # Check if we reached the goal (within tolerance)
            if self.pose_distance(current, goal_pose) < 0.3:  # 30cm tolerance
                return self.reconstruct_path(current)

            # Add current to closed set
            current_grid = self.pose_to_grid([current.x, current.y, current.yaw])
            closed_set.add(current_grid)

            # Generate possible next footsteps
            neighbors = self.generate_neighbors(current, terrain_map)

            for neighbor in neighbors:
                neighbor_grid = self.pose_to_grid([neighbor.x, neighbor.y, neighbor.yaw])

                if neighbor_grid in closed_set:
                    continue

                # Calculate tentative g_cost
                tentative_g = current.g_cost + self.step_cost(current, neighbor)

                # If this path to neighbor is better, update
                if tentative_g < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic([neighbor.x, neighbor.y, neighbor.yaw], goal_pose)

                    heapq.heappush(open_list, neighbor)

        return None  # No path found

    def generate_neighbors(self, current_node, terrain_map):
        """Generate possible next footsteps from current position"""
        neighbors = []

        # Possible step directions relative to current orientation
        step_directions = [
            (self.step_length, 0, 0),           # Forward
            (self.step_length/2, 0, 0),        # Short forward
            (0, self.step_width/2, np.pi/2),   # Left
            (0, -self.step_width/2, -np.pi/2), # Right
            (self.step_length/2, self.step_width/2, np.pi/4),   # Forward-left
            (self.step_length/2, -self.step_width/2, -np.pi/4), # Forward-right
        ]

        # Possible turns (keeping same position, changing orientation)
        turn_angles = [-self.max_turn/2, 0, self.max_turn/2]

        # Generate all possible steps
        for dx, dy, dtheta in step_directions:
            # Calculate new pose
            new_x = current_node.x + dx * np.cos(current_node.yaw) - dy * np.sin(current_node.yaw)
            new_y = current_node.y + dx * np.sin(current_node.yaw) + dy * np.cos(current_node.yaw)
            new_yaw = current_node.yaw + dtheta

            # Normalize yaw to [-pi, pi]
            new_yaw = ((new_yaw + np.pi) % (2 * np.pi)) - np.pi

            # Check if step is valid (collision-free)
            if self.is_valid_footstep([new_x, new_y, new_yaw], terrain_map):
                neighbor = FootstepNode(
                    x=new_x, y=new_y, yaw=new_yaw,
                    g_cost=float('inf'), h_cost=0
                )
                neighbors.append(neighbor)

        # Add turning in place
        for dtheta in turn_angles:
            if abs(dtheta) > 1e-6:  # Not a zero turn
                new_yaw = current_node.yaw + dtheta
                new_yaw = ((new_yaw + np.pi) % (2 * np.pi)) - np.pi

                if self.is_valid_footstep([current_node.x, current_node.y, new_yaw], terrain_map):
                    neighbor = FootstepNode(
                        x=current_node.x, y=current_node.y, yaw=new_yaw,
                        g_cost=float('inf'), h_cost=0
                    )
                    neighbors.append(neighbor)

        return neighbors

    def is_valid_footstep(self, pose, terrain_map):
        """Check if footstep is collision-free"""
        x, y, theta = pose

        # Check if position is within bounds
        if not terrain_map.is_valid_position(x, y):
            return False

        # Check for obstacles at foot position
        if terrain_map.is_occupied(x, y, radius=0.1):  # 10cm radius check
            return False

        # Check terrain slope at foot position
        slope = terrain_map.get_slope_at(x, y)
        if slope > 0.3:  # Too steep (16.7 degrees)
            return False

        # Check for sufficient clearance from obstacles
        clearance = terrain_map.get_clearance(x, y)
        if clearance < self.collision_threshold:
            return False

        return True

    def pose_distance(self, node, goal_pose):
        """Calculate distance between node and goal pose"""
        dx = node.x - goal_pose[0]
        dy = node.y - goal_pose[1]
        dist_2d = np.sqrt(dx*dx + dy*dy)

        # Also consider orientation difference
        dtheta = abs(node.yaw - goal_pose[2])
        dtheta = min(dtheta, 2*np.pi - dtheta)  # Handle angle wrapping

        # Weighted combination of position and orientation
        return dist_2d + 0.5 * dtheta

    def heuristic(self, current_pose, goal_pose):
        """Heuristic function for A* (Euclidean distance)"""
        dx = current_pose[0] - goal_pose[0]
        dy = current_pose[1] - goal_pose[1]
        return np.sqrt(dx*dx + dy*dy)

    def step_cost(self, from_node, to_node):
        """Calculate cost of taking a step from one node to another"""
        # Distance-based cost
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance_cost = np.sqrt(dx*dx + dy*dy)

        # Turn cost (penalize large turns)
        dtheta = abs(to_node.yaw - from_node.yaw)
        dtheta = min(dtheta, 2*np.pi - dtheta)
        turn_cost = 0.5 * dtheta

        # Height change cost (if terrain has elevation)
        # This would be implemented if using 3D terrain

        return distance_cost + turn_cost

    def reconstruct_path(self, goal_node):
        """Reconstruct path from goal node back to start"""
        path = []
        current = goal_node

        while current is not None:
            path.append([current.x, current.y, current.yaw])
            current = current.parent

        return path[::-1]  # Reverse to get start-to-goal order

    def pose_to_grid(self, pose):
        """Convert pose to grid coordinates for hashing"""
        x, y, theta = pose
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        # Quantize angle to reduce search space
        grid_theta = round(theta / (np.pi/8))  # Quantize to 22.5 degree increments
        return (grid_x, grid_y, grid_theta)
```

### Dynamic Walking Pattern Generation

For adaptive walking on varying terrains:

```python
class DynamicWalkingPatternGenerator:
    """Generate walking patterns that adapt to terrain and disturbances"""

    def __init__(self, base_params):
        self.base_params = base_params
        self.current_params = base_params.copy()

        # Online adaptation parameters
        self.adaptation_rate = 0.1
        self.stability_threshold = 0.1
        self.step_timing_buffer = []  # Store recent step timing data

    def generate_adaptive_walking_pattern(self, terrain_info, balance_state, desired_speed):
        """
        Generate walking pattern adapted to current conditions

        Args:
            terrain_info: Information about upcoming terrain
            balance_state: Current robot balance state
            desired_speed: Desired walking speed

        Returns:
            Walking pattern with timing and foot placements
        """
        # Adjust parameters based on terrain and balance
        self.adapt_parameters(terrain_info, balance_state, desired_speed)

        # Generate base walking pattern
        base_pattern = self.generate_base_pattern(desired_speed)

        # Apply terrain adaptations
        adapted_pattern = self.apply_terrain_adaptations(base_pattern, terrain_info)

        # Apply balance corrections
        final_pattern = self.apply_balance_corrections(adapted_pattern, balance_state)

        return final_pattern

    def adapt_parameters(self, terrain_info, balance_state, desired_speed):
        """Adapt walking parameters based on current conditions"""
        # Start with base parameters
        self.current_params = self.base_params.copy()

        # Adapt to terrain roughness
        if terrain_info.get('roughness', 0) > 0.1:
            self.current_params['step_height'] *= 1.5  # Higher steps on rough terrain
            self.current_params['dsp_ratio'] *= 1.2    # Longer double support
            self.current_params['step_length'] *= 0.8  # Shorter steps

        # Adapt to terrain slope
        slope = terrain_info.get('slope', 0)
        if abs(slope) > 0.1:  # Significant slope
            self.current_params['step_length'] *= (1.0 - abs(slope) * 0.3)
            # Adjust CoM height for stability on slopes
            self.current_params['com_height'] *= (1.0 - abs(slope) * 0.1)

        # Adapt to balance state (if robot is unstable)
        if balance_state.get('stability_margin', 0.1) < self.stability_threshold:
            self.current_params['step_width'] *= 1.1  # Wider stance for stability
            self.current_params['dsp_ratio'] *= 1.1  # More double support
            self.current_params['nominal_speed'] *= 0.8  # Slower for stability

        # Limit speed to safe range
        self.current_params['nominal_speed'] = np.clip(
            desired_speed, 0.1, self.current_params['max_speed'])

    def generate_base_pattern(self, desired_speed):
        """Generate basic walking pattern at desired speed"""
        # Calculate step timing based on desired speed
        step_frequency = desired_speed / self.current_params['step_length']
        step_duration = 1.0 / step_frequency if step_frequency > 0 else 0.8

        # Generate footsteps in a straight line
        n_steps = 10  # Generate 10 steps ahead
        footsteps = []

        current_x = 0.0
        current_y = 0.0  # Start with feet centered
        current_yaw = 0.0

        # Start with left foot
        left_support = True

        for step_idx in range(n_steps):
            if step_idx % 2 == 0:  # Right foot step (left is support)
                foot_x = current_x + self.current_params['step_length']
                foot_y = current_y - self.current_params['step_width'] / 2
                foot_yaw = current_yaw
            else:  # Left foot step (right is support)
                foot_x = current_x + self.current_params['step_length']
                foot_y = current_y + self.current_params['step_width'] / 2
                foot_yaw = current_yaw

            footsteps.append({
                'position': [foot_x, foot_y, 0.0],
                'orientation': foot_yaw,
                'timing': step_idx * step_duration,
                'support_leg': 'left' if left_support else 'right'
            })

            # Update current position (for next step planning)
            current_x += self.current_params['step_length']
            left_support = not left_support

        # Calculate corresponding ZMP trajectory
        zmp_trajectory = self.generate_zmp_trajectory_from_footsteps(
            footsteps, step_duration)

        return {
            'footsteps': footsteps,
            'zmp_trajectory': zmp_trajectory,
            'step_duration': step_duration
        }

    def apply_terrain_adaptations(self, base_pattern, terrain_info):
        """Apply terrain-specific adaptations to walking pattern"""
        adapted_pattern = base_pattern.copy()

        # Modify footsteps based on terrain features
        for i, step in enumerate(adapted_pattern['footsteps']):
            x, y, z = step['position']

            # Adjust for terrain elevation
            if 'elevation_map' in terrain_info:
                terrain_elevation = terrain_info['elevation_map'].get_height(x, y)
                step['position'][2] = terrain_elevation + 0.05  # 5cm above ground

            # Adjust for obstacles
            if 'obstacle_map' in terrain_info:
                # Check for nearby obstacles and adjust foot placement
                clearance = terrain_info['obstacle_map'].get_clearance(x, y)
                if clearance < 0.15:  # Very close to obstacle
                    # Shift foot placement laterally
                    if i % 2 == 0:  # Right foot - move right more
                        step['position'][1] -= 0.05
                    else:  # Left foot - move left more
                        step['position'][1] += 0.05

            # Adjust for slippery surfaces
            if terrain_info.get('friction_coefficient', 1.0) < 0.3:
                # Reduce step length and increase DSP ratio
                step['position'][0] -= 0.02  # Take shorter steps
                step['timing'] += 0.05      # Longer step duration

        # Regenerate ZMP trajectory with adapted footsteps
        adapted_pattern['zmp_trajectory'] = self.generate_zmp_trajectory_from_footsteps(
            adapted_pattern['footsteps'], adapted_pattern['step_duration'])

        return adapted_pattern

    def apply_balance_corrections(self, pattern, balance_state):
        """Apply real-time balance corrections to walking pattern"""
        corrected_pattern = pattern.copy()

        # Check if immediate correction is needed
        if balance_state.get('capture_point_deviation', 1.0) > 0.05:  # 5cm from ZMP
            # Adjust next few footsteps to recover balance
            n_correction_steps = min(3, len(corrected_pattern['footsteps']))

            for i in range(n_correction_steps):
                step = corrected_pattern['footsteps'][i]

                # Calculate correction needed
                cp_deviation = balance_state.get('capture_point_deviation', np.array([0, 0]))

                # Move foot toward capture point
                correction_factor = 0.5  # Apply 50% of correction
                step['position'][0] -= cp_deviation[0] * correction_factor
                step['position'][1] -= cp_deviation[1] * correction_factor

        # Adjust ZMP trajectory for balance recovery
        corrected_pattern['zmp_trajectory'] = self.generate_corrected_zmp_trajectory(
            corrected_pattern['footsteps'], corrected_pattern['step_duration'], balance_state)

        return corrected_pattern

    def generate_corrected_zmp_trajectory(self, footsteps, step_duration, balance_state):
        """Generate ZMP trajectory with balance correction"""
        # Start with normal ZMP trajectory
        base_zmp = self.generate_zmp_trajectory_from_footsteps(footsteps, step_duration)

        # Apply balance correction if needed
        if balance_state.get('com_position') is not None:
            com_pos = np.array(balance_state['com_position'])
            com_vel = np.array(balance_state.get('com_velocity', [0, 0, 0]))

            # Calculate current ZMP error
            current_zmp = com_pos[:2] - com_vel[:2] / self.omega**2
            desired_zmp = balance_state.get('desired_zmp', current_zmp)

            # Blend normal ZMP with correction
            correction_strength = min(1.0, balance_state.get('instability_measure', 0))
            corrected_zmp = (1 - correction_strength) * base_zmp + correction_strength * desired_zmp

            return corrected_zmp

        return base_zmp

    def calculate_ankle_adjustments(self, balance_state, current_step_phase):
        """Calculate ankle adjustments for balance recovery"""
        # Calculate required ankle torques to correct balance
        com_pos = balance_state.get('com_position', np.array([0, 0, 0.8]))
        com_vel = balance_state.get('com_velocity', np.array([0, 0, 0]))

        # Calculate current ZMP
        current_zmp = com_pos[:2] - com_vel[:2] / self.omega**2

        # Calculate desired ZMP based on support polygon
        support_polygon = balance_state.get('support_polygon', np.array([[0, 0]]))
        desired_zmp = self.calculate_desired_zmp(current_zmp, support_polygon)

        # Calculate ankle adjustments to move ZMP toward desired
        zmp_error = desired_zmp - current_zmp
        ankle_adjustments = self.map_zmp_error_to_ankle_adjustments(zmp_error)

        return ankle_adjustments

    def map_zmp_error_to_ankle_adjustments(self, zmp_error):
        """Map ZMP error to required ankle adjustments"""
        # Simple mapping - in practice, this would use more sophisticated inverse kinematics
        # and consideration of the robot's specific kinematics

        # Convert ZMP error to required ankle moments
        ankle_roll_adjustment = zmp_error[1] * 50  # Proportional gain
        ankle_pitch_adjustment = -zmp_error[0] * 50  # Note sign inversion

        return {
            'left_ankle_roll': ankle_roll_adjustment,
            'left_ankle_pitch': ankle_pitch_adjustment,
            'right_ankle_roll': -ankle_roll_adjustment,  # Opposite for right foot
            'right_ankle_pitch': ankle_pitch_adjustment
        }
```

## Advanced Gait Patterns

### Different Walking Gaits

Different walking patterns for various speeds and conditions:

```python
class GaitSelector:
    """Select appropriate gait based on speed and conditions"""

    def __init__(self):
        self.gaits = {
            'crawl': self.generate_crawl_gait,
            'creep': self.generate_creep_gait,
            'walk': self.generate_walk_gait,
            'fast_walk': self.generate_fast_walk_gait,
            'run': self.generate_run_gait
        }

    def select_gait(self, desired_speed, terrain_type, stability_requirements):
        """Select appropriate gait based on conditions"""
        if desired_speed < 0.1:
            return 'stand'
        elif desired_speed < 0.3:
            return 'creep'  # Slow, stable walking
        elif desired_speed < 0.8:
            if stability_requirements == 'high':
                return 'creep'  # Use stable gait even at higher speed
            else:
                return 'walk'  # Normal walking
        elif desired_speed < 1.5:
            return 'fast_walk'  # Faster walking
        else:
            return 'run'  # Running gait (if supported)

    def generate_walk_gait(self, params):
        """Generate normal walking gait pattern"""
        # Normal walking with standard DSP/SSP ratios
        gait_pattern = {
            'dsp_ratio': 0.1,
            'ssp_ratio': 0.8,
            'step_length': params.get('step_length', 0.3),
            'step_width': params.get('step_width', 0.2),
            'step_height': params.get('step_height', 0.05),
            'step_duration': 1.0 / (params.get('nominal_speed', 0.3) / params.get('step_length', 0.3)),
            'com_trajectory': self.generate_com_trajectory(params, 'walk'),
            'zmp_profile': self.generate_zmp_profile(params, 'walk')
        }
        return gait_pattern

    def generate_fast_walk_gait(self, params):
        """Generate fast walking gait pattern"""
        # Faster walking with reduced DSP for longer steps
        gait_pattern = {
            'dsp_ratio': 0.05,  # Shorter double support
            'ssp_ratio': 0.85,  # Longer single support
            'step_length': params.get('step_length', 0.3) * 1.2,  # Longer steps
            'step_width': params.get('step_width', 0.2),  # Keep similar width
            'step_height': params.get('step_height', 0.05) * 1.1,  # Slightly higher for faster clearance
            'step_duration': 1.0 / (params.get('nominal_speed', 0.6) / (params.get('step_length', 0.3) * 1.2)),
            'com_trajectory': self.generate_com_trajectory(params, 'fast_walk'),
            'zmp_profile': self.generate_zmp_profile(params, 'fast_walk')
        }
        return gait_pattern

    def generate_creep_gait(self, params):
        """Generate slow, stable creep gait"""
        # Creep gait with continuous support
        gait_pattern = {
            'dsp_ratio': 0.3,  # Much longer double support for stability
            'ssp_ratio': 0.4,  # Shorter single support
            'step_length': params.get('step_length', 0.3) * 0.7,  # Shorter steps for stability
            'step_width': params.get('step_width', 0.2) * 1.2,  # Wider stance
            'step_height': params.get('step_height', 0.05) * 0.8,  # Lower lift for stability
            'step_duration': 1.0 / (params.get('nominal_speed', 0.15) / (params.get('step_length', 0.3) * 0.7)),
            'com_trajectory': self.generate_com_trajectory(params, 'creep'),
            'zmp_profile': self.generate_zmp_profile(params, 'creep')
        }
        return gait_pattern

    def generate_com_trajectory(self, params, gait_type):
        """Generate CoM trajectory for specific gait"""
        if gait_type == 'creep':
            # Very stable CoM motion with minimal oscillation
            return self.generate_stable_com_trajectory(params)
        elif gait_type == 'fast_walk':
            # More dynamic CoM motion to match faster pace
            return self.generate_dynamic_com_trajectory(params)
        else:
            # Standard walking CoM motion
            return self.generate_standard_com_trajectory(params)

    def generate_standard_com_trajectory(self, params):
        """Generate standard CoM trajectory"""
        # Create smooth CoM trajectory with small oscillations
        dt = 0.01  # 100Hz control
        step_duration = params['step_duration']
        n_steps = int(2 * step_duration / dt)  # 2 steps worth of trajectory

        time_vec = np.linspace(0, 2 * step_duration, n_steps)
        com_trajectory = np.zeros((n_steps, 3))  # x, y, z

        for i, t in enumerate(time_vec):
            # Forward progression
            com_trajectory[i, 0] = params['step_length'] * t / step_duration

            # Lateral sway (to maintain balance during single support)
            step_phase = (t % step_duration) / step_duration
            if int(t / step_duration) % 2 == 0:  # Right foot support
                # CoM shifts left during right foot support
                lateral_sway = -0.02 * np.sin(step_phase * np.pi)  # Peak in middle of SSP
            else:  # Left foot support
                # CoM shifts right during left foot support
                lateral_sway = 0.02 * np.sin(step_phase * np.pi)

            com_trajectory[i, 1] = lateral_sway

            # Vertical motion (small oscillation for natural walking)
            vertical_motion = 0.01 * np.sin(2 * np.pi * t / step_duration)
            com_trajectory[i, 2] = params['com_height'] + vertical_motion

        return com_trajectory

    def generate_zmp_profile(self, params, gait_type):
        """Generate ZMP profile for specific gait"""
        dt = 0.01
        step_duration = params['step_duration']
        n_steps = int(2 * step_duration / dt)

        time_vec = np.linspace(0, 2 * step_duration, n_steps)
        zmp_profile = np.zeros((n_steps, 2))

        for i, t in enumerate(time_vec):
            step_idx = int(t / step_duration)
            time_in_step = t - (step_idx * step_duration)
            step_phase = time_in_step / step_duration

            if gait_type == 'creep':
                # Smooth ZMP transitions with minimal oscillation
                zmp_profile[i] = self.calculate_creep_zmp(params, step_phase, step_idx)
            elif gait_type == 'fast_walk':
                # More dynamic ZMP motion to match faster pace
                zmp_profile[i] = self.calculate_fast_walk_zmp(params, step_phase, step_idx)
            else:
                # Standard walking ZMP pattern
                zmp_profile[i] = self.calculate_standard_zmp(params, step_phase, step_idx)

        return zmp_profile

    def calculate_standard_zmp(self, params, step_phase, step_idx):
        """Calculate standard walking ZMP"""
        # During DSP, ZMP moves between feet
        if step_phase < params['dsp_ratio'] or step_phase > (1 - params['dsp_ratio']):
            if step_idx % 2 == 0:  # Even steps: transitioning to right foot
                if step_phase < params['dsp_ratio']:  # First DSP
                    alpha = step_phase / params['dsp_ratio']
                    # Move from left foot position to right foot position
                    left_pos = np.array([step_idx * params['step_length'], params['step_width']/2])
                    right_pos = np.array([(step_idx + 1) * params['step_length'], -params['step_width']/2])
                    return (1 - alpha) * left_pos + alpha * right_pos
                else:  # Second DSP
                    alpha = (step_phase - (1 - params['dsp_ratio'])) / params['dsp_ratio']
                    right_pos = np.array([(step_idx + 1) * params['step_length'], -params['step_width']/2])
                    next_left_pos = np.array([(step_idx + 1) * params['step_length'], params['step_width']/2])
                    return (1 - alpha) * right_pos + alpha * next_left_pos
            else:  # Odd steps: transitioning to left foot
                # Similar logic for odd steps
                if step_phase < params['dsp_ratio']:
                    alpha = step_phase / params['dsp_ratio']
                    right_pos = np.array([step_idx * params['step_length'], -params['step_width']/2])
                    left_pos = np.array([(step_idx + 1) * params['step_length'], params['step_width']/2])
                    return (1 - alpha) * right_pos + alpha * left_pos
                else:
                    alpha = (step_phase - (1 - params['dsp_ratio'])) / params['dsp_ratio']
                    left_pos = np.array([(step_idx + 1) * params['step_length'], params['step_width']/2])
                    next_right_pos = np.array([(step_idx + 1) * params['step_length'], -params['step_width']/2])
                    return (1 - alpha) * left_pos + alpha * next_right_pos
        else:
            # During SSP, ZMP stays near the support foot with small oscillations
            ssp_phase = (step_phase - params['dsp_ratio']) / (1 - 2 * params['dsp_ratio'])

            if step_idx % 2 == 0:  # Right foot support
                support_pos = np.array([(step_idx + 0.5) * params['step_length'], -params['step_width']/2])
            else:  # Left foot support
                support_pos = np.array([(step_idx + 0.5) * params['step_length'], params['step_width']/2])

            # Add small oscillation for dynamic balance
            oscillation = 0.01 * np.array([
                np.sin(ssp_phase * np.pi),
                np.cos(ssp_phase * np.pi)
            ])

            return support_pos + oscillation

    def generate_turning_pattern(self, turn_angle, params):
        """Generate walking pattern for turning"""
        # Calculate number of steps needed for turn
        steps_for_turn = max(2, int(abs(turn_angle) / (np.pi/4)))  # At least 2 steps, ~45deg per step

        # Generate turning footsteps
        footsteps = []
        current_yaw = 0.0

        for step_idx in range(steps_for_turn):
            # Calculate turn per step
            turn_per_step = turn_angle / steps_for_turn

            if step_idx % 2 == 0:  # Right foot step
                # Move forward and turn
                step_x = (step_idx // 2) * params['step_length'] * np.cos(current_yaw + turn_per_step/2)
                step_y = (step_idx // 2) * params['step_length'] * np.sin(current_yaw + turn_per_step/2)
                step_yaw = current_yaw + turn_per_step
            else:  # Left foot step
                # Move forward and turn, with lateral offset
                step_x = ((step_idx + 1) // 2) * params['step_length'] * np.cos(current_yaw + turn_per_step/2)
                step_y = ((step_idx + 1) // 2) * params['step_length'] * np.sin(current_yaw + turn_per_step/2)
                # Add lateral offset with turning adjustment
                lateral_offset = params['step_width'] / 2
                if turn_angle > 0:  # Right turn
                    step_y += lateral_offset * np.cos(turn_per_step) - 0.02 * np.sin(turn_per_step)  # Adjust for turn
                else:  # Left turn
                    step_y += lateral_offset * np.cos(turn_per_step) + 0.02 * np.sin(turn_per_step)

                step_yaw = current_yaw + turn_per_step

            footsteps.append({
                'position': [step_x, step_y, 0.0],
                'orientation': step_yaw,
                'timing': step_idx * params['step_duration'],
                'support_leg': 'left' if step_idx % 2 == 0 else 'right'
            })

            current_yaw = step_yaw

        return {
            'footsteps': footsteps,
            'zmp_trajectory': self.generate_zmp_trajectory_from_footsteps(footsteps, params['step_duration']),
            'step_duration': params['step_duration']
        }
```

## Integration with Control Systems

### Whole-Body Walking Controller

Integrating walking patterns with whole-body control:

```python
class WholeBodyWalkingController:
    """Whole-body controller for integrated walking and balance"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.gait_generator = DynamicWalkingPatternGenerator(GaitParameters())
        self.balance_controller = ZMPPreviewController()

        # Task priorities for whole-body control
        self.task_weights = {
            'com_tracking': 1000.0,
            'foot_placement': 100.0,
            'posture': 10.0,
            'balance': 500.0
        }

    def compute_walking_control(self, current_state, sensor_data, desired_speed, desired_direction):
        """Compute whole-body walking control commands"""
        # Get current state
        joint_positions = current_state['joint_positions']
        joint_velocities = current_state['joint_velocities']

        # Estimate current CoM and ZMP
        current_com = self.robot_model.calculate_com_position(joint_positions)
        current_com_vel = self.robot_model.calculate_com_velocity(joint_positions, joint_velocities)
        current_zmp = current_com[:2] - current_com_vel[:2] / self.balance_controller.omega**2

        # Get current support polygon
        support_polygon = self.estimate_support_polygon(sensor_data)

        # Generate walking pattern based on desired speed and direction
        terrain_info = self.get_terrain_information(sensor_data)
        balance_state = {
            'com_position': current_com,
            'com_velocity': current_com_vel,
            'current_zmp': current_zmp,
            'support_polygon': support_polygon,
            'stability_margin': self.calculate_stability_margin(current_zmp, support_polygon)
        }

        walking_pattern = self.gait_generator.generate_adaptive_walking_pattern(
            terrain_info, balance_state, desired_speed)

        # Calculate desired CoM trajectory from walking pattern
        desired_com_trajectory = self.calculate_com_from_zmp_trajectory(
            walking_pattern['zmp_trajectory'])

        # Use whole-body inverse kinematics to achieve both CoM and foot tracking
        control_commands = self.whole_body_ik_solve(
            joint_positions, joint_velocities,
            desired_com_trajectory[0],  # Next desired CoM position
            walking_pattern['footsteps'],  # Desired foot positions
            balance_state)  # Current balance state for constraints

        return control_commands, walking_pattern

    def whole_body_ik_solve(self, current_joints, current_velocities,
                           desired_com, desired_footsteps, balance_state):
        """Solve whole-body inverse kinematics with multiple tasks"""
        # Define multiple tasks with priorities
        tasks = {
            'com_task': {
                'desired': desired_com,
                'jacobian': self.robot_model.com_jacobian(current_joints),
                'weight': self.task_weights['com_tracking'],
                'priority': 1
            },
            'left_foot_task': {
                'desired': self.get_next_desired_foot_position('left', desired_footsteps),
                'jacobian': self.robot_model.jacobian_for_link(current_joints, 'left_foot'),
                'weight': self.task_weights['foot_placement'],
                'priority': 1
            },
            'right_foot_task': {
                'desired': self.get_next_desired_foot_position('right', desired_footsteps),
                'jacobian': self.robot_model.jacobian_for_link(current_joints, 'right_foot'),
                'weight': self.task_weights['foot_placement'],
                'priority': 1
            },
            'balance_task': {
                'desired': self.calculate_balance_posture(current_joints, balance_state),
                'jacobian': self.calculate_balance_jacobian(current_joints),
                'weight': self.task_weights['balance'],
                'priority': 2  # Lower priority than position tasks
            },
            'posture_task': {
                'desired': self.get_desired_posture(current_joints),
                'jacobian': self.calculate_posture_jacobian(current_joints),
                'weight': self.task_weights['posture'],
                'priority': 3  # Lowest priority
            }
        }

        # Solve using hierarchical inverse kinematics
        joint_deltas = self.hierarchical_ik_solver(current_joints, tasks)

        # Apply joint limits and generate commands
        new_joints = current_joints + joint_deltas

        # Apply joint limits
        joint_limits = self.robot_model.get_joint_limits()
        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            new_joints[i] = np.clip(new_joints[i], limits['min'], limits['max'])

        return new_joints

    def hierarchical_ik_solver(self, current_joints, tasks):
        """Solve hierarchical inverse kinematics problem"""
        n_joints = len(current_joints)
        joint_deltas = np.zeros(n_joints)

        # Sort tasks by priority
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1]['priority'])

        # Process tasks in order of priority
        for task_name, task_info in sorted_tasks:
            desired = task_info['desired']
            jacobian = task_info['jacobian']
            weight = task_info['weight']

            # Calculate error
            if task_name == 'com_task':
                current_value = self.robot_model.calculate_com_position(current_joints)
            elif 'foot_task' in task_name:
                foot_name = task_name.split('_')[0]
                current_value = self.robot_model.forward_kinematics_for_link(current_joints, f'{foot_name}_foot')
                current_value = current_value[:3]  # Position only
            else:
                # For other tasks, assume current value is current joint positions
                current_value = current_joints

            error = desired - current_value

            # Apply weighted least squares with null-space projection
            # First, solve for the highest priority task
            if task_info['priority'] == min(task['priority'] for task in tasks.values()):
                # Primary task - solve directly
                weighted_jac = jacobian * np.sqrt(weight)
                weighted_error = error * np.sqrt(weight)

                # Use damped least squares to avoid singularities
                damping = 0.01
                joint_delta = np.linalg.solve(
                    weighted_jac.T @ weighted_jac + damping * np.eye(n_joints),
                    weighted_jac.T @ weighted_error
                )

                joint_deltas = joint_delta
            else:
                # Secondary task - solve in null space of higher priority tasks
                # Calculate null space projection matrix
                I = np.eye(n_joints)
                null_space_proj = I - np.linalg.pinv(jacobian) @ jacobian

                # Solve secondary task in null space
                weighted_jac = (jacobian @ null_space_proj) * np.sqrt(weight)
                weighted_error = error * np.sqrt(weight)

                if weighted_jac.shape[0] > 0 and weighted_jac.shape[1] > 0:
                    try:
                        secondary_delta = np.linalg.solve(
                            weighted_jac.T @ weighted_jac + damping * np.eye(n_joints),
                            weighted_jac.T @ weighted_error
                        )

                        # Add to existing deltas while maintaining higher priority solutions
                        joint_deltas = null_space_proj @ joint_deltas + secondary_delta
                    except np.linalg.LinAlgError:
                        # If matrix is singular, use pseudoinverse
                        secondary_delta = np.linalg.pinv(weighted_jac) @ weighted_error
                        joint_deltas = null_space_proj @ joint_deltas + secondary_delta

        return joint_deltas

    def calculate_balance_posture(self, current_joints, balance_state):
        """Calculate desired joint positions for balance"""
        # Based on current balance state, calculate corrective posture
        com_pos = balance_state['com_position']
        current_zmp = balance_state['current_zmp']
        support_polygon = balance_state['support_polygon']

        # Calculate balance correction
        zmp_error = current_zmp - self.calculate_desired_zmp_in_support(support_polygon)

        # Map ZMP error to posture adjustments
        posture_correction = np.zeros(len(current_joints))

        # Adjust trunk and hip joints based on ZMP error
        # This is a simplified mapping - in practice, this would use more sophisticated biomechanics
        trunk_joint_idx = self.get_joint_index_by_name('torso_pitch', current_joints)
        if trunk_joint_idx is not None:
            posture_correction[trunk_joint_idx] = -zmp_error[0] * 0.5  # Pitch to correct forward/backward

        left_hip_idx = self.get_joint_index_by_name('left_hip_roll', current_joints)
        if left_hip_idx is not None:
            posture_correction[left_hip_idx] = -zmp_error[1] * 0.3  # Roll to correct lateral

        right_hip_idx = self.get_joint_index_by_name('right_hip_roll', current_joints)
        if right_hip_idx is not None:
            posture_correction[right_hip_idx] = -zmp_error[1] * 0.3  # Roll to correct lateral

        return current_joints + posture_correction

    def get_joint_index_by_name(self, joint_name, current_joints):
        """Get index of joint by name in joint array"""
        joint_names = self.robot_model.get_joint_names()
        try:
            return joint_names.index(joint_name)
        except ValueError:
            return None  # Joint not found

    def estimate_support_polygon(self, sensor_data):
        """Estimate current support polygon from sensor data"""
        support_points = []

        # Check force/torque sensors for contact
        ft_sensors = sensor_data.get('force_torque', {})

        for foot_name in ['left_foot', 'right_foot']:
            if foot_name in ft_sensors:
                force_magnitude = np.linalg.norm(ft_sensors[foot_name].get('force', [0, 0, 0]))
                if force_magnitude > 5.0:  # Threshold for contact detection
                    # Get foot position
                    foot_pos = self.robot_model.forward_kinematics_for_link(
                        sensor_data['joint_positions'], foot_name)
                    support_points.append(foot_pos[:2])  # x, y only

        if len(support_points) == 0:
            # No contact detected - use default (very small polygon)
            return np.array([[0, 0]])
        elif len(support_points) == 1:
            # Single support - use foot contact area
            contact_point = support_points[0]
            return np.array([
                [contact_point[0] - 0.05, contact_point[1] - 0.025],
                [contact_point[0] + 0.05, contact_point[1] - 0.025],
                [contact_point[0] + 0.05, contact_point[1] + 0.025],
                [contact_point[0] - 0.05, contact_point[1] + 0.025]
            ])
        else:
            # Double support - convex hull of contact points
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(support_points)
                return np.array(support_points)[hull.vertices]
            except:
                return np.array(support_points)

    def calculate_stability_margin(self, zmp_pos, support_polygon):
        """Calculate stability margin (distance from ZMP to support polygon boundary)"""
        if len(support_polygon) < 3:
            return -1.0  # Unstable if less than triangular support

        # Check if ZMP is inside support polygon
        if self.is_point_in_convex_polygon(zmp_pos, support_polygon):
            # Find minimum distance to any edge
            min_distance = float('inf')
            for i in range(len(support_polygon)):
                p1 = support_polygon[i]
                p2 = support_polygon[(i + 1) % len(support_polygon)]

                # Calculate distance from ZMP to edge
                distance = self.distance_point_to_line_segment(zmp_pos, p1, p2)
                min_distance = min(min_distance, distance)

            return min_distance
        else:
            # ZMP is outside - return negative distance
            min_distance = float('inf')
            for i in range(len(support_polygon)):
                p1 = support_polygon[i]
                p2 = support_polygon[(i + 1) % len(support_polygon)]

                distance = self.distance_point_to_line_segment(zmp_pos, p1, p2)
                min_distance = min(min_distance, distance)

            return -min_distance

    def is_point_in_convex_polygon(self, point, polygon):
        """Check if point is inside convex polygon using cross product method"""
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
                return False  # Point is outside polygon

        return True

    def distance_point_to_line_segment(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:
            return np.linalg.norm(point - line_start)

        # Calculate projection of point onto line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = line_start + t * line_vec

        return np.linalg.norm(point - projection)

    def get_terrain_information(self, sensor_data):
        """Extract terrain information from sensor data"""
        terrain_info = {
            'roughness': 0.0,
            'slope': 0.0,
            'friction_coefficient': 0.8,
            'obstacles_ahead': [],
            'surface_type': 'flat_ground'
        }

        # Use LIDAR or camera data to detect terrain features
        if 'lidar_scan' in sensor_data:
            scan_data = sensor_data['lidar_scan']
            # Analyze scan for ground slope and obstacles
            terrain_info['slope'] = self.analyze_ground_slope(scan_data)
            terrain_info['roughness'] = self.analyze_surface_roughness(scan_data)
            terrain_info['obstacles_ahead'] = self.detect_obstacles(scan_data)

        return terrain_info

    def calculate_desired_zmp_in_support(self, support_polygon):
        """Calculate desired ZMP position within support polygon"""
        # For stability, aim for center of support polygon
        if len(support_polygon) >= 3:
            return np.mean(support_polygon, axis=0)
        elif len(support_polygon) == 2:
            # For line support, use midpoint
            return (support_polygon[0] + support_polygon[1]) / 2
        else:
            # For single point, use that point
            return support_polygon[0] if len(support_polygon) > 0 else np.array([0.0, 0.0])
```

## Performance Optimization

### Efficient Gait Computation

```python
class OptimizedWalkingController:
    """Optimized walking controller for real-time performance"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.lookahead_steps = 10  # Steps to plan ahead
        self.control_frequency = 100  # Hz
        self.dt = 1.0 / self.control_frequency

        # Pre-compute commonly used values
        self.omega = np.sqrt(9.81 / 0.8)  # Assuming 0.8m CoM height
        self.omega_sq = self.omega ** 2

        # Cache for expensive computations
        self.jacobian_cache = {}
        self.mass_matrix_cache = {}

        # Simplified models for real-time computation
        self.use_linearized_model = True

    def compute_gait_step_fast(self, current_state, desired_speed, step_count):
        """Fast computation of single gait step"""
        # Use simplified kinematic model for real-time performance
        current_com = self.robot_model.calculate_com_position(current_state['joint_positions'])
        current_com_vel = self.robot_model.calculate_com_velocity(
            current_state['joint_positions'], current_state['joint_velocities'])

        # Calculate current ZMP
        current_zmp = current_com[:2] - current_com_vel[:2] / self.omega_sq

        # Generate next step based on desired speed and current state
        next_foot_position = self.predict_next_foot_position(
            current_state, desired_speed, step_count)

        # Calculate ZMP reference for this step
        zmp_reference = self.calculate_step_zmp_reference(
            current_zmp, next_foot_position, current_state)

        # Generate CoM trajectory to achieve desired ZMP
        desired_com = self.calculate_com_for_zmp(zmp_reference, current_com_vel)

        # Use fast inverse kinematics to achieve desired poses
        joint_commands = self.fast_inverse_kinematics(
            current_state['joint_positions'], desired_com, next_foot_position)

        return joint_commands, zmp_reference

    def predict_next_foot_position(self, current_state, desired_speed, step_count):
        """Fast prediction of next foot position"""
        # Simple kinematic prediction based on current pose and desired speed
        current_pose = self.estimate_base_pose(current_state)

        # Calculate step timing based on speed
        step_length = min(self.base_step_length * (desired_speed / self.nominal_speed), self.max_step_length)
        step_width = self.base_step_width

        # Alternate steps between left and right foot
        if step_count % 2 == 0:  # Right foot step
            # Move right foot forward and slightly to the right
            next_x = current_pose[0] + step_length
            next_y = current_pose[1] - step_width / 2
        else:  # Left foot step
            # Move left foot forward and slightly to the left
            next_x = current_pose[0] + step_length
            next_y = current_pose[1] + step_width / 2

        # Apply turning if needed
        if abs(self.desired_turn_rate) > 0.001:
            # Calculate turning effect on foot placement
            turn_offset_x = -step_width/2 * np.sin(self.desired_turn_rate * self.dt * step_count)
            turn_offset_y = step_width/2 * (1 - np.cos(self.desired_turn_rate * self.dt * step_count))

            next_x += turn_offset_x
            next_y += turn_offset_y

        return np.array([next_x, next_y, 0.0])

    def calculate_step_zmp_reference(self, current_zmp, next_foot_pos, current_state):
        """Calculate ZMP reference for current step"""
        # Simple ZMP pattern for walking:
        # 1. During DSP: move ZMP from current support to new support
        # 2. During SSP: keep ZMP near support foot

        # For real-time efficiency, use simplified pattern
        step_phase = self.get_current_step_phase()

        if step_phase < self.dsp_ratio:  # First DSP
            # Move from current ZMP to midway between feet
            mid_point = (current_zmp + next_foot_pos[:2]) / 2
            progress = step_phase / self.dsp_ratio
            return (1 - progress) * current_zmp + progress * mid_point
        elif step_phase > (1 - self.dsp_ratio):  # Second DSP
            # Move toward next foot position
            mid_point = (current_zmp + next_foot_pos[:2]) / 2
            progress = (step_phase - (1 - self.dsp_ratio)) / self.dsp_ratio
            return (1 - progress) * mid_point + progress * next_foot_pos[:2]
        else:  # SSP
            # Keep ZMP near current support foot with small oscillation
            ssp_phase = (step_phase - self.dsp_ratio) / (1 - 2 * self.dsp_ratio)
            support_foot = self.get_current_support_foot_position(current_state)

            # Add small oscillation for dynamic balance
            oscillation = 0.01 * np.array([
                np.sin(ssp_phase * np.pi),
                np.cos(ssp_phase * np.pi)
            ])

            return support_foot[:2] + oscillation

    def fast_inverse_kinematics(self, current_joints, desired_com, desired_foot_pos):
        """Fast inverse kinematics using simplified approach"""
        # Use Jacobian transpose method for speed
        # This is less accurate than full inverse kinematics but much faster

        # Calculate current CoM and foot positions
        current_com = self.robot_model.calculate_com_position(current_joints)
        current_left_foot = self.robot_model.forward_kinematics_for_link(current_joints, 'left_foot')
        current_right_foot = self.robot_model.forward_kinematics_for_link(current_joints, 'right_foot')

        # Calculate errors
        com_error = desired_com - current_com
        foot_error = desired_foot_pos - current_left_foot[:3]  # Assuming left foot for example

        # Get Jacobians (use cached if available)
        joint_hash = hash(tuple(np.round(current_joints, 3)))
        if joint_hash in self.jacobian_cache:
            com_jac = self.jacobian_cache[joint_hash]['com']
            foot_jac = self.jacobian_cache[joint_hash]['foot']
        else:
            com_jac = self.robot_model.com_jacobian(current_joints)
            foot_jac = self.robot_model.jacobian_for_link(current_joints, 'left_foot')

            # Cache for future use (limit cache size)
            if len(self.jacobian_cache) > 50:  # Limit cache size
                # Remove oldest entry
                oldest_key = next(iter(self.jacobian_cache))
                del self.jacobian_cache[oldest_key]

            self.jacobian_cache[joint_hash] = {
                'com': com_jac,
                'foot': foot_jac
            }

        # Apply Jacobian transpose method
        com_correction = 0.1 * com_jac.T @ com_error  # Small gain for stability
        foot_correction = 0.05 * foot_jac.T @ foot_error  # Even smaller gain

        # Combine corrections with weights
        total_correction = 0.7 * com_correction + 0.3 * foot_correction

        # Apply joint limits
        new_joints = current_joints + total_correction
        joint_limits = self.robot_model.get_joint_limits()

        for i, (joint_name, limits) in enumerate(joint_limits.items()):
            new_joints[i] = np.clip(new_joints[i], limits['min'], limits['max'])

        return new_joints

    def estimate_base_pose(self, current_state):
        """Fast estimation of robot base pose"""
        # Use IMU data for orientation and forward kinematics for position
        joint_positions = current_state['joint_positions']

        # Get torso/base link position
        base_pos = self.robot_model.forward_kinematics_for_link(joint_positions, 'torso')

        # Estimate orientation from IMU if available, otherwise assume upright
        imu_data = current_state.get('imu_data', {})
        if 'orientation' in imu_data:
            # Convert quaternion to Euler angles if needed
            orientation = imu_data['orientation']
            # Use z-axis as forward direction
            base_yaw = np.arctan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                                1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z))
        else:
            base_yaw = 0.0  # Assume facing forward

        return np.array([base_pos[0], base_pos[1], base_yaw])

    def get_current_step_phase(self):
        """Get current phase within the step cycle"""
        # This would use timing information from the controller
        # For now, return a placeholder
        return 0.5  # Middle of step

    def get_current_support_foot_position(self, current_state):
        """Get position of current support foot"""
        # Determine from contact sensors or step timing
        # This is a simplified implementation
        step_count = int(self.get_current_time() / self.step_duration)

        if step_count % 2 == 0:  # Right foot is support
            foot_pos = self.robot_model.forward_kinematics_for_link(
                current_state['joint_positions'], 'right_foot')
        else:  # Left foot is support
            foot_pos = self.robot_model.forward_kinematics_for_link(
                current_state['joint_positions'], 'left_foot')

        return foot_pos

    def get_current_time(self):
        """Get current simulation time"""
        # This would interface with the simulation clock
        import time
        return time.time() % 1000  # Placeholder
```

## Validation and Testing

### Walking Pattern Validation

```python
class WalkingPatternValidator:
    """Validate walking patterns for stability and feasibility"""

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def validate_walking_pattern(self, walking_pattern):
        """Validate entire walking pattern"""
        issues = []

        # Check ZMP stability
        zmp_stable = self.check_zmp_stability(walking_pattern['zmp_trajectory'],
                                            walking_pattern['footsteps'])
        if not zmp_stable:
            issues.append("ZMP trajectory goes outside support polygon")

        # Check step feasibility
        step_feasible = self.check_step_feasibility(walking_pattern['footsteps'])
        if not step_feasible:
            issues.append("Some footsteps are not kinematically feasible")

        # Check CoM continuity
        com_continuous = self.check_com_continuity(walking_pattern['com_trajectory'])
        if not com_continuous:
            issues.append("CoM trajectory has discontinuous jumps")

        # Check dynamic consistency
        dynamic_consistent = self.check_dynamic_consistency(
            walking_pattern['com_trajectory'], walking_pattern['zmp_trajectory'])
        if not dynamic_consistent:
            issues.append("CoM and ZMP trajectories are dynamically inconsistent")

        return len(issues) == 0, issues

    def check_zmp_stability(self, zmp_trajectory, footsteps):
        """Check if ZMP remains within support polygon throughout trajectory"""
        for i, zmp in enumerate(zmp_trajectory):
            # Determine support polygon at this time step
            support_polygon = self.calculate_support_polygon_at_time(i, footsteps)

            # Check if ZMP is within support polygon
            if not self.is_point_in_convex_polygon(zmp, support_polygon):
                return False

        return True

    def calculate_support_polygon_at_time(self, time_step, footsteps):
        """Calculate support polygon based on footstep timing"""
        # This would determine which feet are in contact at given time
        # For now, return a simple implementation
        if time_step % 2 == 0:
            # Single support with left foot
            return np.array([[-0.05, -0.025], [0.05, -0.025], [0.05, 0.025], [-0.05, 0.025]])
        else:
            # Single support with right foot
            return np.array([[-0.05, -0.025], [0.05, -0.025], [0.05, 0.025], [-0.05, 0.025]])

    def check_step_feasibility(self, footsteps):
        """Check if footsteps are kinematically feasible"""
        for i in range(1, len(footsteps)):
            prev_pos = np.array(footsteps[i-1]['position'][:2])
            curr_pos = np.array(footsteps[i]['position'][:2])

            # Check if step is within robot's reach capability
            step_distance = np.linalg.norm(curr_pos - prev_pos)
            if step_distance > self.robot_model.get_max_step_distance():
                return False

            # Check if step direction is reasonable
            if i > 1:
                prev_step_dir = np.array(footsteps[i-1]['position'][:2]) - np.array(footsteps[i-2]['position'][:2])
                curr_step_dir = curr_pos - prev_pos
                turn_angle = np.arccos(
                    np.clip(np.dot(prev_step_dir, curr_step_dir) /
                           (np.linalg.norm(prev_step_dir) * np.linalg.norm(curr_step_dir)), -1, 1)
                )
                # Check if turn is too sharp
                if abs(turn_angle) > np.pi / 3:  # More than 60 degrees
                    return False

        return True

    def check_com_continuity(self, com_trajectory):
        """Check if CoM trajectory is continuous"""
        for i in range(1, len(com_trajectory)):
            prev_pos = com_trajectory[i-1]
            curr_pos = com_trajectory[i]

            # Check for large jumps (indicating discontinuity)
            displacement = np.linalg.norm(curr_pos - prev_pos)
            if displacement > 0.1:  # More than 10cm jump in one time step
                return False

        return True

    def check_dynamic_consistency(self, com_trajectory, zmp_trajectory):
        """Check if CoM and ZMP trajectories are dynamically consistent"""
        if len(com_trajectory) != len(zmp_trajectory):
            return False

        # Check if ZMP = CoM - CoM_ddot/omega^2 relationship holds approximately
        omega = np.sqrt(9.81 / 0.8)  # Assuming 0.8m CoM height

        for i in range(2, len(com_trajectory)-2):  # Skip edges for numerical differentiation
            # Calculate CoM acceleration using central difference
            dt = 0.01  # Assuming 100Hz control
            com_acc = (com_trajectory[i-1] - 2*com_trajectory[i] + com_trajectory[i+1]) / (dt**2)

            # Calculate expected ZMP from inverted pendulum model
            expected_zmp = com_trajectory[i][:2] - com_acc[:2] / (omega**2)
            actual_zmp = zmp_trajectory[i]

            # Check if they match within tolerance
            if np.linalg.norm(expected_zmp - actual_zmp) > 0.05:  # 5cm tolerance
                return False

        return True

    def simulate_walking_step(self, initial_state, walking_pattern, step_duration):
        """Simulate one walking step to validate stability"""
        # This would run a short simulation to check if the pattern is stable
        # For now, return True as a placeholder
        return True

    def generate_stability_report(self, walking_pattern):
        """Generate detailed stability report for walking pattern"""
        report = {
            'zmp_margin': self.calculate_zmp_margin(walking_pattern),
            'step_stability': self.calculate_step_stability(walking_pattern),
            'com_excursion': self.calculate_com_excursion(walking_pattern),
            'dynamic_balance': self.calculate_dynamic_balance(walking_pattern)
        }
        return report

    def calculate_zmp_margin(self, walking_pattern):
        """Calculate minimum distance from ZMP to support polygon boundary"""
        min_distance = float('inf')

        for i, zmp in enumerate(walking_pattern['zmp_trajectory']):
            support_polygon = self.calculate_support_polygon_at_time(i, walking_pattern['footsteps'])

            if self.is_point_in_convex_polygon(zmp, support_polygon):
                # Calculate distance to closest edge
                for j in range(len(support_polygon)):
                    p1 = support_polygon[j]
                    p2 = support_polygon[(j + 1) % len(support_polygon)]
                    distance = self.distance_point_to_line_segment(zmp, p1, p2)
                    min_distance = min(min_distance, distance)
            else:
                # ZMP is outside - return negative distance
                return -self.distance_to_closest_edge(zmp, support_polygon)

        return min_distance
```

## Summary

Walking pattern generation and gait control are fundamental to humanoid robotics, requiring careful integration of kinematics, dynamics, and real-time control. The key aspects covered in this chapter include:

1. **ZMP-Based Control**: Using Zero Moment Point theory to generate stable walking patterns
2. **Footstep Planning**: Algorithms for determining appropriate foot placements
3. **Sensor Integration**: Incorporating multiple sensor types for enhanced stability
4. **Adaptive Control**: Adjusting gait parameters based on terrain and balance state
5. **Performance Optimization**: Techniques for real-time gait computation
6. **Validation**: Methods for ensuring gait stability and feasibility

The successful implementation of these techniques enables humanoid robots to walk stably in various conditions while maintaining balance and avoiding falls. Proper gait control is essential for the practical deployment of Physical AI systems in real-world environments.