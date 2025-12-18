---
title: "Chapter 5 - Navigation and Path Planning for Humanoids"
description: "Advanced navigation algorithms and path planning techniques specifically designed for bipedal humanoid robots operating in complex environments"
sidebar_label: "Chapter 5 - Navigation and Path Planning for Humanoids"
---

# Navigation and Path Planning for Humanoids

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement navigation algorithms that account for humanoid-specific constraints
- Design path planning systems that consider balance and stability
- Apply sampling-based and optimization-based path planning methods
- Integrate navigation with whole-body control for stable locomotion
- Handle dynamic obstacles and replanning in real-time
- Implement humanoid-specific navigation behaviors and gaits

## Introduction

Navigation for humanoid robots presents unique challenges compared to wheeled or tracked robots. Humanoid robots must maintain balance while navigating, which introduces constraints on movement patterns, speed, and path selection. The bipedal nature of locomotion requires careful consideration of center of mass (CoM) dynamics, Zero Moment Point (ZMP) stability, and footstep planning.

Unlike wheeled robots that can move in any direction with simple wheel commands, humanoid robots must plan complex sequences of steps while maintaining dynamic balance. This chapter explores navigation algorithms specifically designed for humanoid robots, addressing the unique challenges of bipedal locomotion in complex environments.

## Humanoid-Specific Navigation Constraints

### Balance and Stability Considerations

Humanoid navigation must account for the fundamental balance constraints of bipedal locomotion:

```python
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class Footstep:
    """Represents a single footstep for humanoid navigation"""
    position: np.ndarray  # [x, y, z]
    orientation: float    # yaw angle in radians
    foot: str             # 'left' or 'right'
    support_time: float   # time this foot supports the body
    timestamp: float

@dataclass
class BalanceConstraint:
    """Balance constraints for humanoid navigation"""
    zmp_limits: Tuple[float, float]  # [min_x, max_x] in meters
    com_height: float               # Center of mass height
    step_width: float              # Distance between feet
    max_step_length: float         # Maximum step length
    max_step_width: float          # Maximum lateral step distance
    balance_margin: float          # Safety margin for balance

class HumanoidNavigator:
    """Navigation system optimized for humanoid robots"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.balance_constraints = self._initialize_balance_constraints()
        self.footstep_planner = FootstepPlanner(robot_config)
        self.path_optimizer = PathOptimizer(robot_config)
        self.local_planner = LocalNavigator(robot_config)

    def _initialize_balance_constraints(self) -> BalanceConstraint:
        """Initialize balance constraints based on robot configuration"""
        return BalanceConstraint(
            zmp_limits=(-0.1, 0.1),  # ZMP should stay within foot boundaries
            com_height=self.robot_config.get('com_height', 0.8),
            step_width=self.robot_config.get('step_width', 0.3),
            max_step_length=self.robot_config.get('max_step_length', 0.4),
            max_step_width=self.robot_config.get('max_step_width', 0.2),
            balance_margin=0.05  # 5cm safety margin
        )

    def plan_navigation_path(self, start_pose, goal_pose, environment_map):
        """Plan navigation path considering humanoid balance constraints"""
        # First, plan high-level path using traditional methods
        high_level_path = self._plan_high_level_path(
            start_pose, goal_pose, environment_map
        )

        # Then, convert to footstep plan that maintains balance
        footstep_plan = self._convert_to_footsteps(
            high_level_path, start_pose
        )

        # Optimize footsteps for stability and efficiency
        optimized_plan = self._optimize_footsteps(
            footstep_plan, environment_map
        )

        return optimized_plan

    def _plan_high_level_path(self, start_pose, goal_pose, environment_map):
        """Plan initial path using traditional path planning"""
        # Use A* or RRT* for initial path planning
        # This path will be refined for humanoid constraints later
        path = self._a_star_search(start_pose[:2], goal_pose[:2], environment_map)
        return path

    def _convert_to_footsteps(self, path, start_pose):
        """Convert continuous path to discrete footsteps"""
        footsteps = []

        # Start with current foot positions
        left_foot_pos = start_pose[:3].copy()
        right_foot_pos = start_pose[:3].copy()
        right_foot_pos[1] += self.balance_constraints.step_width  # Offset for stance width

        current_left_support = True  # Start with left foot support

        # Generate footsteps along the path
        for i in range(len(path) - 1):
            # Calculate desired step direction
            direction = path[i + 1] - path[i]
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction = direction / distance  # Normalize

            # Calculate next foot position based on balance constraints
            if current_left_support:
                # Right foot moves forward
                next_right_pos = left_foot_pos + direction * min(
                    distance, self.balance_constraints.max_step_length
                )
                next_right_pos[1] += np.random.uniform(
                    -self.balance_constraints.max_step_width,
                    self.balance_constraints.max_step_width
                )
                next_right_pos[2] = 0  # Ground level

                footsteps.append(Footstep(
                    position=next_right_pos,
                    orientation=np.arctan2(direction[1], direction[0]),
                    foot='right',
                    support_time=0.8,  # 800ms support time
                    timestamp=i * 0.8  # Assuming 800ms per step
                ))

                right_foot_pos = next_right_pos
                current_left_support = False
            else:
                # Left foot moves forward
                next_left_pos = right_foot_pos + direction * min(
                    distance, self.balance_constraints.max_step_length
                )
                next_left_pos[1] += np.random.uniform(
                    -self.balance_constraints.max_step_width,
                    self.balance_constraints.max_step_width
                )
                next_left_pos[2] = 0  # Ground level

                footsteps.append(Footstep(
                    position=next_left_pos,
                    orientation=np.arctan2(direction[1], direction[0]),
                    foot='left',
                    support_time=0.8,
                    timestamp=i * 0.8
                ))

                left_foot_pos = next_left_pos
                current_left_support = True

        return footsteps

    def _optimize_footsteps(self, footsteps, environment_map):
        """Optimize footsteps for stability and obstacle avoidance"""
        optimized_footsteps = []

        for i, step in enumerate(footsteps):
            # Check if step is valid (not in collision)
            if self._is_step_valid(step, environment_map):
                # Optimize step position for stability
                optimized_pos = self._optimize_step_position(
                    step, footsteps, i, environment_map
                )

                optimized_step = Footstep(
                    position=optimized_pos,
                    orientation=step.orientation,
                    foot=step.foot,
                    support_time=step.support_time,
                    timestamp=step.timestamp
                )

                optimized_footsteps.append(optimized_step)
            else:
                # Try to find alternative step position
                alternative_step = self._find_alternative_step(
                    step, environment_map, footsteps, i
                )

                if alternative_step:
                    optimized_footsteps.append(alternative_step)

        return optimized_footsteps

    def _is_step_valid(self, step, environment_map):
        """Check if a footstep is valid (no collision, stable)"""
        # Check collision with obstacles
        if self._check_collision(step.position, environment_map):
            return False

        # Check if step maintains balance
        if not self._check_balance_stability(step):
            return False

        return True

    def _check_collision(self, position, environment_map):
        """Check if position collides with obstacles"""
        # This is a simplified collision check
        # In practice, you'd use the actual environment map
        x, y, z = position

        # Check if position is in obstacle region
        # This would interface with your actual map representation
        return False  # Simplified - assume no collision for now

    def _check_balance_stability(self, step):
        """Check if footstep maintains balance"""
        # Check if ZMP stays within support polygon
        # For simplicity, assume the step is stable if it's within bounds
        x, y, z = step.position

        # Check if step is within reasonable bounds
        if (abs(x) > 5.0 or abs(y) > 5.0 or
            z < -0.1 or z > 0.1):  # Ground level with tolerance
            return False

        return True
```

### Gait Planning and Footstep Sequences

Humanoid navigation requires careful planning of gait patterns and footstep sequences:

```python
class GaitPlanner:
    """Plan gait patterns for humanoid navigation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.gait_patterns = self._initialize_gait_patterns()

    def _initialize_gait_patterns(self):
        """Initialize different gait patterns"""
        return {
            'walk': {
                'step_length': 0.3,
                'step_height': 0.05,
                'step_time': 0.8,
                'stance_width': 0.25,
                'swing_time_ratio': 0.4  # 40% of step time for swing
            },
            'trot': {
                'step_length': 0.4,
                'step_height': 0.08,
                'step_time': 0.6,
                'stance_width': 0.2,
                'swing_time_ratio': 0.3
            },
            'crawl': {
                'step_length': 0.15,
                'step_height': 0.03,
                'step_time': 1.2,
                'stance_width': 0.3,
                'swing_time_ratio': 0.5
            }
        }

    def plan_gait_sequence(self, path, start_pose, gait_type='walk'):
        """Plan gait sequence for following a path"""
        gait_params = self.gait_patterns[gait_type]

        footsteps = []
        current_pose = start_pose.copy()

        # Calculate required steps based on path
        for i in range(len(path) - 1):
            # Determine step direction and magnitude
            direction = path[i + 1] - path[i]
            distance = np.linalg.norm(direction)

            if distance > 0.05:  # Minimum step threshold
                direction = direction / distance

                # Generate step based on gait parameters
                step = self._generate_step(
                    current_pose, direction, gait_params, i
                )

                footsteps.append(step)
                current_pose[:2] = path[i + 1][:2]  # Update position

        return footsteps

    def _generate_step(self, current_pose, direction, gait_params, step_index):
        """Generate a single step based on gait parameters"""
        # Calculate step position
        step_length = gait_params['step_length']
        step_pos = current_pose[:3].copy()
        step_pos[:2] += direction[:2] * step_length

        # Alternate feet
        foot = 'left' if step_index % 2 == 0 else 'right'

        # Calculate orientation
        orientation = np.arctan2(direction[1], direction[0])

        return Footstep(
            position=step_pos,
            orientation=orientation,
            foot=foot,
            support_time=gait_params['step_time'],
            timestamp=step_index * gait_params['step_time']
        )

    def adjust_gait_for_terrain(self, footsteps, terrain_info):
        """Adjust gait parameters based on terrain characteristics"""
        adjusted_footsteps = []

        for step in footsteps:
            # Modify step based on terrain
            adjusted_step = self._adjust_step_for_terrain(step, terrain_info)
            adjusted_footsteps.append(adjusted_step)

        return adjusted_footsteps

    def _adjust_step_for_terrain(self, step, terrain_info):
        """Adjust individual step based on terrain"""
        # Get terrain properties at step location
        terrain_type = self._get_terrain_type(step.position, terrain_info)

        # Adjust step parameters based on terrain
        if terrain_type == 'rough':
            # Use more conservative step parameters
            adjusted_pos = step.position.copy()
            adjusted_pos[2] += 0.02  # Slightly higher step for rough terrain
        elif terrain_type == 'slippery':
            # Use shorter, more stable steps
            adjusted_pos = step.position.copy()
            # Reduce step length effect by adjusting position slightly
        elif terrain_type == 'stairs':
            # Adjust for step height
            adjusted_pos = step.position.copy()
            adjusted_pos[2] += 0.15  # Typical step height
        else:
            adjusted_pos = step.position.copy()

        return Footstep(
            position=adjusted_pos,
            orientation=step.orientation,
            foot=step.foot,
            support_time=step.support_time,
            timestamp=step.timestamp
        )

    def _get_terrain_type(self, position, terrain_info):
        """Determine terrain type at given position"""
        # Simplified terrain classification
        # In practice, this would use actual terrain analysis
        x, y, z = position

        # Example terrain classification based on position
        if abs(z) > 0.1:
            return 'stairs'
        elif np.random.random() < 0.1:  # 10% chance of rough terrain
            return 'rough'
        elif np.random.random() < 0.05:  # 5% chance of slippery
            return 'slippery'
        else:
            return 'flat'
```

## Sampling-Based Path Planning for Humanoids

### Footstep-Aware RRT*

Traditional sampling-based planners need modification for humanoid constraints:

```python
class FootstepRRT:
    """RRT* planner adapted for humanoid footstep planning"""

    def __init__(self, robot_config, environment_map):
        self.robot_config = robot_config
        self.environment_map = environment_map
        self.start_footsteps = None
        self.goal_footsteps = None
        self.tree = {}
        self.path_cost = {}
        self.balance_checker = BalanceConstraintChecker(robot_config)

    def plan_footstep_path(self, start_footsteps, goal_footsteps):
        """Plan footstep path using RRT* approach"""
        self.start_footsteps = start_footsteps
        self.goal_footsteps = goal_footsteps

        # Initialize tree with start footsteps
        start_node = self._create_node(start_footsteps)
        self.tree[start_node.id] = start_node
        self.path_cost[start_node.id] = 0.0

        # Grow tree toward goal
        for iteration in range(1000):  # Max iterations
            # Sample random footsteps
            random_footsteps = self._sample_random_footsteps()

            # Find nearest node in tree
            nearest_node = self._find_nearest_node(random_footsteps)

            # Extend tree toward random footsteps
            new_footsteps = self._extend_toward(
                nearest_node.footsteps, random_footsteps
            )

            if new_footsteps and self._is_valid_footsteps(new_footsteps):
                # Add new node to tree
                new_node = self._create_node(new_footsteps)
                cost = self._calculate_cost(nearest_node, new_node)

                if cost < self.path_cost.get(new_node.id, float('inf')):
                    self.tree[new_node.id] = new_node
                    self.path_cost[new_node.id] = cost
                    self.tree[new_node.id].parent = nearest_node.id

                    # Rewire tree for optimization
                    self._rewire(new_node)

        # Extract path to goal
        goal_node = self._find_goal_node()
        if goal_node:
            return self._extract_path(goal_node)

        return None

    def _sample_random_footsteps(self):
        """Sample random valid footsteps"""
        # Sample footsteps in the configuration space
        # This is simplified - in practice, you'd sample more intelligently
        num_steps = 5  # Plan 5 steps ahead
        footsteps = []

        for i in range(num_steps):
            # Sample position near current area
            x = np.random.uniform(-2.0, 2.0)
            y = np.random.uniform(-2.0, 2.0)
            z = 0.0  # Ground level

            # Alternate feet
            foot = 'left' if i % 2 == 0 else 'right'

            # Random orientation
            orientation = np.random.uniform(-np.pi, np.pi)

            footsteps.append(Footstep(
                position=np.array([x, y, z]),
                orientation=orientation,
                foot=foot,
                support_time=0.8,
                timestamp=i * 0.8
            ))

        return footsteps

    def _find_nearest_node(self, footsteps):
        """Find nearest node in tree to given footsteps"""
        min_distance = float('inf')
        nearest_node = None

        for node_id, node in self.tree.items():
            distance = self._calculate_footstep_distance(
                footsteps, node.footsteps
            )

            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def _extend_toward(self, from_footsteps, to_footsteps):
        """Extend tree from one set of footsteps toward another"""
        # This would implement the actual extension logic
        # For humanoid navigation, this involves complex balance-aware steps
        # Simplified implementation:

        if len(from_footsteps) == 0 or len(to_footsteps) == 0:
            return None

        # Create intermediate footsteps that are a step toward the goal
        extended_footsteps = from_footsteps.copy()

        # Add one more step toward the goal
        if len(to_footsteps) > len(from_footsteps):
            next_step = to_footsteps[len(from_footsteps)]
            extended_footsteps.append(next_step)

        # Ensure balance constraints are maintained
        if self.balance_checker.check_sequence_balance(extended_footsteps):
            return extended_footsteps

        return None

    def _is_valid_footsteps(self, footsteps):
        """Check if footsteps are valid (collision-free and balanced)"""
        # Check collision avoidance
        for step in footsteps:
            if self._check_step_collision(step):
                return False

        # Check balance constraints
        if not self.balance_checker.check_sequence_balance(footsteps):
            return False

        return True

    def _check_step_collision(self, step):
        """Check if step collides with obstacles"""
        # Simplified collision check
        x, y, z = step.position
        return False  # Simplified - assume no collision

    def _calculate_cost(self, from_node, to_node):
        """Calculate cost of transitioning between nodes"""
        # Cost includes distance, balance stability, and obstacle proximity
        distance_cost = self._calculate_footstep_distance(
            from_node.footsteps, to_node.footsteps
        )

        # Add balance stability cost
        balance_cost = self.balance_checker.calculate_balance_cost(
            to_node.footsteps
        )

        # Add obstacle proximity cost
        obstacle_cost = self._calculate_obstacle_cost(to_node.footsteps)

        return distance_cost + balance_cost + obstacle_cost

    def _calculate_obstacle_cost(self, footsteps):
        """Calculate cost based on proximity to obstacles"""
        total_cost = 0.0

        for step in footsteps:
            # Calculate distance to nearest obstacle
            obstacle_dist = self._distance_to_nearest_obstacle(step.position)

            if obstacle_dist < 0.5:  # Within 50cm of obstacle
                # High cost for being close to obstacles
                total_cost += (0.5 - obstacle_dist) * 10.0

        return total_cost

    def _distance_to_nearest_obstacle(self, position):
        """Calculate distance to nearest obstacle"""
        # Simplified - in practice, use actual map data
        return 1.0  # Return default distance

    def _calculate_footstep_distance(self, footsteps1, footsteps2):
        """Calculate distance between two sets of footsteps"""
        if len(footsteps1) == 0 or len(footsteps2) == 0:
            return float('inf')

        # Calculate total distance between corresponding steps
        total_distance = 0.0
        min_len = min(len(footsteps1), len(footsteps2))

        for i in range(min_len):
            pos1 = footsteps1[i].position
            pos2 = footsteps2[i].position
            total_distance += np.linalg.norm(pos1 - pos2)

        return total_distance

    def _create_node(self, footsteps):
        """Create a tree node from footsteps"""
        return type('Node', (), {
            'id': id(footsteps),
            'footsteps': footsteps,
            'parent': None
        })()

    def _rewire(self, new_node):
        """Rewire tree to optimize path"""
        # Find nearby nodes that could have lower cost through new node
        for node_id, node in self.tree.items():
            if node_id != new_node.id:
                # Calculate cost through new node
                cost_through_new = (
                    self.path_cost[new_node.id] +
                    self._calculate_cost(new_node, node)
                )

                # Update if better path found
                if cost_through_new < self.path_cost.get(node_id, float('inf')):
                    self.path_cost[node_id] = cost_through_new
                    node.parent = new_node.id

    def _find_goal_node(self):
        """Find node that reaches the goal"""
        # This would find a node with footsteps close to goal
        # Simplified implementation
        for node_id, node in self.tree.items():
            if self._is_at_goal(node.footsteps):
                return node
        return None

    def _is_at_goal(self, footsteps):
        """Check if footsteps reach the goal"""
        # Simplified goal check
        return len(footsteps) >= len(self.goal_footsteps)

    def _extract_path(self, goal_node):
        """Extract path from goal node back to start"""
        path = []
        current_node = goal_node

        while current_node.parent is not None:
            path.append(current_node.footsteps)
            # Find parent node
            for node_id, node in self.tree.items():
                if node_id == current_node.parent:
                    current_node = node
                    break

        path.append(self.start_footsteps)
        return list(reversed(path))

class BalanceConstraintChecker:
    """Check balance constraints for footstep sequences"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.com_height = robot_config.get('com_height', 0.8)
        self.foot_size = robot_config.get('foot_size', [0.2, 0.1])  # [length, width]

    def check_sequence_balance(self, footsteps):
        """Check if footstep sequence maintains balance"""
        if len(footsteps) < 2:
            return True

        # Check balance at each transition
        for i in range(len(footsteps) - 1):
            if not self._check_step_transition_balance(
                footsteps[i], footsteps[i + 1]
            ):
                return False

        return True

    def _check_step_transition_balance(self, current_step, next_step):
        """Check balance during transition between steps"""
        # Calculate support polygon for current stance
        support_polygon = self._calculate_support_polygon(current_step, next_step)

        # Calculate expected CoM position during transition
        com_position = self._estimate_com_during_transition(
            current_step, next_step
        )

        # Check if CoM stays within support polygon
        return self._point_in_polygon(com_position[:2], support_polygon)

    def _calculate_support_polygon(self, current_step, next_step):
        """Calculate support polygon during step transition"""
        # For double support phase, support polygon is convex hull
        # of both feet
        foot1_corners = self._get_foot_corners(current_step)
        foot2_corners = self._get_foot_corners(next_step)

        all_corners = foot1_corners + foot2_corners
        return self._convex_hull(all_corners)

    def _get_foot_corners(self, step):
        """Get corner positions of foot in world coordinates"""
        center = step.position
        length, width = self.foot_size

        # Calculate corner positions relative to foot center
        corners = [
            center + np.array([length/2, width/2, 0]),
            center + np.array([length/2, -width/2, 0]),
            center + np.array([-length/2, -width/2, 0]),
            center + np.array([-length/2, width/2, 0])
        ]

        # Apply orientation
        cos_yaw = np.cos(step.orientation)
        sin_yaw = np.sin(step.orientation)

        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        rotated_corners = []
        for corner in corners:
            relative_corner = corner - center
            rotated_relative = rotation_matrix @ relative_corner
            rotated_corners.append(center + rotated_relative)

        return [c[:2] for c in rotated_corners]  # Return x,y coordinates

    def _convex_hull(self, points):
        """Calculate convex hull of points (simplified)"""
        # This is a simplified convex hull calculation
        # In practice, use scipy.spatial.ConvexHull
        return points

    def _estimate_com_during_transition(self, current_step, next_step):
        """Estimate CoM position during step transition"""
        # Simplified CoM estimation
        # In practice, this would use inverted pendulum model
        return (current_step.position + next_step.position) / 2

    def _point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        # Ray casting algorithm
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

    def calculate_balance_cost(self, footsteps):
        """Calculate cost based on balance stability"""
        if len(footsteps) < 2:
            return 0.0

        total_cost = 0.0
        for i in range(len(footsteps) - 1):
            # Calculate balance margin for each transition
            margin = self._calculate_balance_margin(
                footsteps[i], footsteps[i + 1]
            )
            # Higher cost for smaller balance margins
            total_cost += max(0, 0.1 - margin) * 100.0  # Penalty for low margin

        return total_cost

    def _calculate_balance_margin(self, current_step, next_step):
        """Calculate balance margin during transition"""
        # Calculate distance from CoM to edge of support polygon
        support_polygon = self._calculate_support_polygon(current_step, next_step)
        com_pos = self._estimate_com_during_transition(current_step, next_step)

        # Find minimum distance from CoM to polygon edge
        min_distance = float('inf')
        for i in range(len(support_polygon)):
            p1 = support_polygon[i]
            p2 = support_polygon[(i + 1) % len(support_polygon)]

            # Calculate distance from point to line segment
            dist = self._point_to_line_distance(com_pos[:2], p1, p2)
            min_distance = min(min_distance, dist)

        return min_distance

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate line length
        line_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_len == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2)

        # Calculate projection
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_len**2)))

        # Calculate closest point
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
```

## Local Navigation and Obstacle Avoidance

### Dynamic Window Approach for Humanoids

```python
class DynamicWindowNavigator:
    """Dynamic Window Approach adapted for humanoid navigation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.balance_constraints = self._initialize_balance_constraints()
        self.trajectory_evaluator = TrajectoryEvaluator(robot_config)

    def _initialize_balance_constraints(self):
        """Initialize constraints for humanoid movement"""
        return {
            'max_forward_step': 0.4,
            'max_backward_step': 0.2,
            'max_lateral_step': 0.15,
            'max_rotation': np.pi/4,  # 45 degrees
            'min_step_time': 0.5,
            'max_step_time': 1.5
        }

    def compute_velocity_commands(self, current_state, goal, obstacles):
        """Compute optimal velocity commands using DWA"""
        # Generate possible trajectories
        possible_trajectories = self._generate_trajectories(
            current_state, goal, obstacles
        )

        # Evaluate trajectories
        best_trajectory = self._evaluate_trajectories(
            possible_trajectories, current_state, goal, obstacles
        )

        return best_trajectory

    def _generate_trajectories(self, current_state, goal, obstacles):
        """Generate possible movement trajectories"""
        trajectories = []

        # Sample different step parameters
        step_lengths = np.linspace(0.1, self.balance_constraints['max_forward_step'], 5)
        step_widths = np.linspace(-self.balance_constraints['max_lateral_step'],
                                  self.balance_constraints['max_lateral_step'], 5)
        step_orientations = np.linspace(-self.balance_constraints['max_rotation'],
                                       self.balance_constraints['max_rotation'], 5)
        step_times = np.linspace(self.balance_constraints['min_step_time'],
                                self.balance_constraints['max_step_time'], 3)

        for length in step_lengths:
            for width in step_widths:
                for orientation in step_orientations:
                    for time in step_times:
                        # Create trajectory with these parameters
                        trajectory = self._create_trajectory(
                            current_state, length, width, orientation, time
                        )

                        if trajectory and self._is_trajectory_valid(trajectory, obstacles):
                            trajectories.append(trajectory)

        return trajectories

    def _create_trajectory(self, current_state, length, width, orientation, time):
        """Create a trajectory with given parameters"""
        # Calculate next foot position based on parameters
        current_pos = current_state['position']
        current_orientation = current_state['orientation']

        # Calculate step direction in world frame
        step_direction = np.array([
            length * np.cos(current_orientation + orientation),
            length * np.sin(current_orientation + orientation)
        ])

        # Add lateral offset
        lateral_offset = np.array([
            -width * np.sin(current_orientation + orientation),
            width * np.cos(current_orientation + orientation)
        ])

        next_pos = current_pos[:2] + step_direction + lateral_offset
        next_pos_3d = np.array([next_pos[0], next_pos[1], current_pos[2]])

        # Create footstep
        foot = 'left' if current_state.get('current_support_foot', 'left') == 'right' else 'right'

        step = Footstep(
            position=next_pos_3d,
            orientation=current_orientation + orientation,
            foot=foot,
            support_time=time,
            timestamp=current_state.get('timestamp', 0) + time
        )

        return [step]

    def _is_trajectory_valid(self, trajectory, obstacles):
        """Check if trajectory is valid (collision-free and balanced)"""
        for step in trajectory:
            # Check collision
            if self._check_collision(step.position, obstacles):
                return False

            # Check balance constraints
            if not self._check_balance(step):
                return False

        return True

    def _check_collision(self, position, obstacles):
        """Check if position collides with obstacles"""
        pos_2d = position[:2]

        for obstacle in obstacles:
            obs_pos = obstacle['position'][:2]
            obs_radius = obstacle.get('radius', 0.1)

            distance = np.linalg.norm(pos_2d - obs_pos)
            if distance < obs_radius + 0.1:  # Robot radius + safety margin
                return True

        return False

    def _check_balance(self, step):
        """Check if step maintains balance"""
        # Check if step is within reasonable bounds
        x, y, z = step.position

        if (abs(x) > 10.0 or abs(y) > 10.0 or
            z < -0.2 or z > 0.2):  # Ground level with tolerance
            return False

        return True

    def _evaluate_trajectories(self, trajectories, current_state, goal, obstacles):
        """Evaluate and select best trajectory"""
        best_trajectory = None
        best_score = float('-inf')

        for trajectory in trajectories:
            score = self._evaluate_trajectory(
                trajectory, current_state, goal, obstacles
            )

            if score > best_score:
                best_score = score
                best_trajectory = trajectory

        return best_trajectory

    def _evaluate_trajectory(self, trajectory, current_state, goal, obstacles):
        """Evaluate a single trajectory"""
        if not trajectory:
            return float('-inf')

        # Calculate scores for different criteria
        goal_score = self._calculate_goal_score(trajectory, goal)
        obs_score = self._calculate_obstacle_score(trajectory, obstacles)
        balance_score = self._calculate_balance_score(trajectory)

        # Weighted combination of scores
        total_score = (
            0.5 * goal_score +    # Goal achievement is important
            0.3 * obs_score +     # Obstacle avoidance
            0.2 * balance_score   # Balance maintenance
        )

        return total_score

    def _calculate_goal_score(self, trajectory, goal):
        """Calculate score based on progress toward goal"""
        if not trajectory:
            return 0

        last_step = trajectory[-1]
        goal_pos = goal[:2]
        step_pos = last_step.position[:2]

        # Distance to goal from this step
        dist_to_goal = np.linalg.norm(step_pos - goal_pos)

        # Invert so closer is better (add small constant to avoid division by zero)
        return 1.0 / (dist_to_goal + 0.1)

    def _calculate_obstacle_score(self, trajectory, obstacles):
        """Calculate score based on obstacle proximity"""
        if not trajectory:
            return 0

        min_distance = float('inf')

        for step in trajectory:
            step_pos = step.position[:2]

            for obstacle in obstacles:
                obs_pos = obstacle['position'][:2]
                distance = np.linalg.norm(step_pos - obs_pos)
                min_distance = min(min_distance, distance)

        # Return distance as score (higher is better)
        return min_distance if min_distance != float('inf') else 10.0

    def _calculate_balance_score(self, trajectory):
        """Calculate score based on balance maintenance"""
        if not trajectory:
            return 0

        # For now, return a simple balance score
        # In practice, this would involve more complex balance analysis
        return 1.0  # Assume all valid trajectories are equally balanced
```

## Integration with Whole-Body Control

### Navigation-Control Interface

```python
class NavigationControlInterface:
    """Interface between navigation and whole-body control"""

    def __init__(self, robot_config, navigation_system, control_system):
        self.robot_config = robot_config
        self.navigation_system = navigation_system
        self.control_system = control_system
        self.footstep_executor = FootstepExecutor(robot_config)
        self.balance_controller = BalanceController(robot_config)

    def execute_navigation_plan(self, plan):
        """Execute a navigation plan with whole-body control"""
        for step in plan:
            # Execute individual footstep
            success = self._execute_footstep(step)

            if not success:
                # Handle execution failure
                return self._handle_execution_failure(step, plan)

        return True

    def _execute_footstep(self, footstep):
        """Execute a single footstep with balance control"""
        # First, plan the swing trajectory
        swing_trajectory = self._plan_swing_trajectory(footstep)

        # Execute with balance controller active
        success = self.control_system.execute_trajectory(
            swing_trajectory,
            balance_controller=self.balance_controller
        )

        if success:
            # Update support foot
            self.balance_controller.update_support_foot(footstep.foot)

        return success

    def _plan_swing_trajectory(self, footstep):
        """Plan swing trajectory for footstep"""
        # Current support foot position
        current_support_pos = self._get_current_support_position()

        # Plan swing path (simple parabolic trajectory)
        swing_points = self._generate_swing_path(
            current_support_pos, footstep.position
        )

        return swing_points

    def _generate_swing_path(self, start_pos, end_pos):
        """Generate swing path with appropriate clearance"""
        # Calculate intermediate points for swing trajectory
        num_points = 10
        trajectory = []

        # Calculate swing height (clearance over obstacles)
        swing_height = 0.1  # 10cm clearance

        for i in range(num_points + 1):
            t = i / num_points  # Parameter from 0 to 1

            # Linear interpolation for x,y position
            pos_2d = (1 - t) * start_pos[:2] + t * end_pos[:2]

            # Parabolic trajectory for z (lift and lower foot)
            z = (1 - t) * start_pos[2] + t * end_pos[2]
            z += swing_height * np.sin(np.pi * t)  # Arc motion

            full_pos = np.array([pos_2d[0], pos_2d[1], z])
            trajectory.append(full_pos)

        return trajectory

    def _get_current_support_position(self):
        """Get current position of support foot"""
        # This would interface with actual robot state
        # For simulation, return a reasonable default
        return np.array([0.0, 0.0, 0.0])

    def _handle_execution_failure(self, failed_step, plan):
        """Handle failure to execute a step"""
        # Try to replan from current position
        current_pos = self._get_current_robot_position()
        remaining_goal = plan[-1].position if plan else current_pos

        # Generate new plan from current position
        new_plan = self.navigation_system.replan(
            current_pos, remaining_goal
        )

        if new_plan:
            # Execute new plan
            return self.execute_navigation_plan(new_plan)
        else:
            # Unable to replan, stop navigation
            return False

    def _get_current_robot_position(self):
        """Get current robot position from state estimation"""
        # This would interface with actual state estimation
        # For now, return a simulated position
        return np.array([0.0, 0.0, 0.8])  # x, y, com_height

class FootstepExecutor:
    """Execute footstep plans with proper timing and control"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.current_step_time = 0.8
        self.swing_time_ratio = 0.4  # 40% of step time for swing

    def execute_step_sequence(self, footsteps, control_interface):
        """Execute sequence of footsteps"""
        for i, step in enumerate(footsteps):
            # Execute single step
            success = self._execute_single_step(step, control_interface)

            if not success:
                return False

        return True

    def _execute_single_step(self, step, control_interface):
        """Execute a single footstep"""
        # Phase 1: Preparation (shift weight to stance foot)
        self._prepare_step(step, control_interface)

        # Phase 2: Swing (move swing foot to target)
        self._swing_foot(step, control_interface)

        # Phase 3: Landing (place foot and transfer weight)
        self._land_foot(step, control_interface)

        return True  # Simplified success check

    def _prepare_step(self, step, control_interface):
        """Prepare for step by shifting weight"""
        # Shift center of mass toward stance foot
        # This would involve whole-body control commands
        pass

    def _swing_foot(self, step, control_interface):
        """Swing foot to target position"""
        # Execute swing trajectory
        # This would involve joint control commands
        pass

    def _land_foot(self, step, control_interface):
        """Land foot and transfer weight"""
        # Place foot and shift weight
        # This would involve impact control and weight transfer
        pass

class BalanceController:
    """Maintain balance during navigation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.current_support_foot = 'left'
        self.com_reference = np.array([0.0, 0.0, 0.8])
        self.zmp_reference = np.array([0.0, 0.0])

    def update_support_foot(self, new_support_foot):
        """Update which foot is in support"""
        self.current_support_foot = new_support_foot

    def compute_balance_control(self, current_state):
        """Compute balance control commands"""
        # Calculate current CoM and ZMP
        current_com = current_state.get('com_position', self.com_reference)
        current_zmp = current_state.get('zmp_position', self.zmp_reference)

        # Compute control to maintain balance
        com_error = self.com_reference - current_com
        zmp_error = self.zmp_reference - current_zmp[:2]

        # Simple proportional control (in practice, use more sophisticated methods)
        com_control = 10.0 * com_error  # Stiffness parameter
        zmp_control = 5.0 * zmp_error   # Stiffness parameter

        return {
            'com_control': com_control,
            'zmp_control': zmp_control
        }
```

## Practical Implementation Considerations

### Real-Time Performance Optimization

```python
class OptimizedHumanoidNavigator:
    """Optimized navigation system for real-time humanoid operation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.navigation_cache = {}
        self.multi_resolution_maps = {}
        self.parallel_processors = self._initialize_parallel_processing()

    def _initialize_parallel_processing(self):
        """Initialize parallel processing capabilities"""
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor

        return {
            'path_planning': ThreadPoolExecutor(max_workers=2),
            'collision_checking': ThreadPoolExecutor(max_workers=3),
            'balance_verification': ThreadPoolExecutor(max_workers=2)
        }

    def navigate_with_obstacle_avoidance(self, start_pose, goal_pose,
                                       dynamic_obstacles, static_map):
        """Navigate with real-time obstacle avoidance"""
        # Use multi-resolution approach for efficiency
        coarse_path = self._plan_coarse_path(start_pose, goal_pose, static_map)

        if not coarse_path:
            return None

        # Plan detailed footsteps for current window
        current_window = self._get_navigation_window(start_pose, coarse_path)
        detailed_plan = self._plan_detailed_path(
            start_pose, current_window, dynamic_obstacles
        )

        return detailed_plan

    def _plan_coarse_path(self, start, goal, static_map):
        """Plan coarse path on low-resolution map"""
        # Use A* on coarse map for global planning
        if 'coarse' not in self.multi_resolution_maps:
            self.multi_resolution_maps['coarse'] = self._create_coarse_map(static_map)

        coarse_map = self.multi_resolution_maps['coarse']
        return self._a_star_coarse(start, goal, coarse_map)

    def _plan_detailed_path(self, start_pose, path_segment, dynamic_obstacles):
        """Plan detailed footsteps for current path segment"""
        # Convert path segment to footstep plan
        footsteps = self._convert_path_to_footsteps(path_segment, start_pose)

        # Check for dynamic obstacles
        safe_footsteps = self._avoid_dynamic_obstacles(
            footsteps, dynamic_obstacles
        )

        return safe_footsteps

    def _avoid_dynamic_obstacles(self, footsteps, dynamic_obstacles):
        """Modify footsteps to avoid moving obstacles"""
        safe_footsteps = []

        for step in footsteps:
            # Predict obstacle positions at step time
            future_obstacles = self._predict_obstacle_positions(
                step.timestamp, dynamic_obstacles
            )

            # Check for conflicts
            if self._is_step_safe(step, future_obstacles):
                safe_footsteps.append(step)
            else:
                # Find alternative step
                alternative = self._find_safe_alternative_step(
                    step, future_obstacles
                )

                if alternative:
                    safe_footsteps.append(alternative)
                else:
                    # Cannot find safe alternative, stop
                    break

        return safe_footsteps

    def _predict_obstacle_positions(self, timestamp, dynamic_obstacles):
        """Predict obstacle positions at future time"""
        predicted_obstacles = []

        for obstacle in dynamic_obstacles:
            # Simple constant velocity prediction
            current_pos = obstacle['position']
            velocity = obstacle.get('velocity', np.array([0.0, 0.0, 0.0]))
            time_diff = timestamp - obstacle.get('timestamp', 0)

            predicted_pos = current_pos + velocity * time_diff

            predicted_obstacles.append({
                'position': predicted_pos,
                'radius': obstacle.get('radius', 0.1)
            })

        return predicted_obstacles

    def _is_step_safe(self, step, predicted_obstacles):
        """Check if step is safe from predicted obstacles"""
        step_pos = step.position[:2]

        for obstacle in predicted_obstacles:
            obs_pos = obstacle['position'][:2]
            distance = np.linalg.norm(step_pos - obs_pos)

            if distance < obstacle['radius'] + 0.15:  # Safety margin
                return False

        return True

    def _find_safe_alternative_step(self, original_step, obstacles):
        """Find alternative step position that avoids obstacles"""
        # Try different positions around original step
        step_pos = original_step.position
        search_radius = 0.3  # 30cm search radius
        num_samples = 16

        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            offset = np.array([
                search_radius * np.cos(angle),
                search_radius * np.sin(angle),
                0.0
            ])

            candidate_pos = step_pos + offset

            # Check if candidate is safe
            candidate_step = Footstep(
                position=candidate_pos,
                orientation=original_step.orientation,
                foot=original_step.foot,
                support_time=original_step.support_time,
                timestamp=original_step.timestamp
            )

            if self._is_step_safe(candidate_step, obstacles):
                return candidate_step

        return None  # No safe alternative found
```

## Assessment Questions

1. Explain the key differences between navigation for wheeled robots and humanoid robots, focusing on balance constraints.

2. Design a footstep planning algorithm that can handle narrow passages while maintaining stability.

3. Implement a dynamic window approach specifically adapted for humanoid step planning.

4. Compare different gait patterns (walk, trot, crawl) and their applications in humanoid navigation.

5. Design a navigation system that can handle both static and dynamic obstacle avoidance for humanoid robots.

## Practice Exercises

1. **Footstep Optimization**: Implement an algorithm that optimizes a sequence of footsteps for energy efficiency while maintaining balance.

2. **Terrain Adaptation**: Create a navigation system that adjusts gait parameters based on terrain type (stairs, slopes, rough ground).

3. **Multi-Modal Navigation**: Design a system that can switch between different navigation modes (walking, crawling, climbing) based on environmental constraints.

4. **Failure Recovery**: Implement a navigation system that can detect and recover from balance loss during locomotion.

## Summary

Navigation and path planning for humanoid robots requires specialized approaches that account for the unique challenges of bipedal locomotion. This chapter covered:

- Balance and stability constraints specific to humanoid navigation
- Footstep planning algorithms that maintain dynamic balance
- Sampling-based methods adapted for humanoid constraints
- Local navigation and obstacle avoidance techniques
- Integration with whole-body control systems
- Real-time performance optimization strategies

The integration of navigation with balance control enables humanoid robots to move safely and efficiently through complex environments while maintaining stability throughout the locomotion process.