---
title: "Chapter 7 - Human-Robot Interaction and Social Navigation"
description: "Designing humanoid robots that can interact naturally with humans and navigate social spaces according to human social conventions"
sidebar_label: "Chapter 7 - Human-Robot Interaction and Social Navigation"
---

# Human-Robot Interaction and Social Navigation

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement social navigation algorithms that respect human spatial conventions
- Design human-robot interaction systems for natural communication
- Apply proxemics principles to robot behavior in human environments
- Create socially aware path planning that considers human comfort
- Implement gesture recognition and generation for humanoid robots
- Develop emotional expression systems for humanoid robots
- Design multi-modal interaction interfaces for human-robot communication

## Introduction

Human-robot interaction (HRI) and social navigation are critical capabilities for humanoid robots operating in human environments. Unlike industrial robots that work in isolated spaces, humanoid robots must navigate complex social situations where they interact with people, respect social conventions, and adapt their behavior to human expectations.

Social navigation goes beyond simple obstacle avoidance to include understanding and respecting human social spaces, conventions, and expectations. A humanoid robot must know when to maintain eye contact, how to pass by people safely, when to wait for others to proceed, and how to communicate its intentions clearly.

This chapter explores the implementation of social navigation algorithms and human-robot interaction systems that enable humanoid robots to operate naturally and safely in human environments.

## Proxemics and Personal Space

### Understanding Human Spatial Zones

Proxemics, the study of human spatial relationships, is fundamental to social navigation:

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math
from datetime import datetime
import cv2

class ProxemicZone(Enum):
    """Social distance zones based on proxemics research"""
    INTIMATE = "intimate"      # 0-0.5m - close relationships
    PERSONAL = "personal"      # 0.5-1.2m - friends, family
    SOCIAL = "social"          # 1.2-3.6m - strangers, acquaintances
    PUBLIC = "public"          # 3.6m+ - public speaking

@dataclass
class HumanProxemics:
    """Represents human spatial behavior and preferences"""
    zone: ProxemicZone
    distance_range: Tuple[float, float]  # min, max distance in meters
    comfort_level: float  # 0.0-1.0
    cultural_context: str  # cultural background affecting preferences
    relationship_type: str  # family, friend, stranger, authority, etc.

class ProxemicsManager:
    """Manage human proxemics for social navigation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.zone_distances = {
            ProxemicZone.INTIMATE: (0.0, 0.5),
            ProxemicZone.PERSONAL: (0.5, 1.2),
            ProxemicZone.SOCIAL: (1.2, 3.6),
            ProxemicZone.PUBLIC: (3.6, float('inf'))
        }

        # Cultural defaults (can be customized)
        self.cultural_defaults = {
            'western': {
                'personal_space_multiplier': 1.0,
                'eye_contact_duration': 1.5,  # seconds
                'approach_angle_preference': 30  # degrees
            },
            'eastern': {
                'personal_space_multiplier': 1.2,
                'eye_contact_duration': 1.0,
                'approach_angle_preference': 45
            }
        }

    def get_comfortable_distance(self, relationship_type: str, cultural_context: str = 'western') -> float:
        """Get comfortable distance based on relationship and culture"""
        base_distance = self.zone_distances[ProxemicZone.SOCIAL][0]  # Default to social distance

        if relationship_type == 'stranger':
            multiplier = self.cultural_defaults[cultural_context]['personal_space_multiplier']
            return base_distance * multiplier
        elif relationship_type == 'acquaintance':
            return base_distance
        elif relationship_type == 'friend':
            return self.zone_distances[ProxemicZone.PERSONAL][0]
        elif relationship_type == 'family':
            return self.zone_distances[ProxemicZone.INTIMATE][1] * 0.8  # Slightly more than intimate min

        return base_distance

    def calculate_social_force(self, human_position: np.ndarray, robot_position: np.ndarray,
                              relationship_type: str, cultural_context: str = 'western') -> np.ndarray:
        """Calculate social force based on proxemics"""
        # Vector from robot to human
        displacement = human_position - robot_position
        distance = np.linalg.norm(displacement)

        if distance == 0:
            return np.array([0.0, 0.0, 0.0])

        # Normalize direction
        direction = displacement / distance

        # Get comfortable distance
        comfortable_dist = self.get_comfortable_distance(relationship_type, cultural_context)

        # Calculate force magnitude based on distance
        if distance < comfortable_dist * 0.5:
            # Too close - strong repulsive force
            force_magnitude = 10.0 / (distance + 0.1)
        elif distance < comfortable_dist:
            # Within comfort zone - moderate repulsive force
            force_magnitude = 5.0 * (comfortable_dist - distance) / comfortable_dist
        elif distance > comfortable_dist * 2:
            # Too far - weak attractive force (if following)
            force_magnitude = -2.0 / distance
        else:
            # At comfortable distance - minimal force
            force_magnitude = 0.1

        return force_magnitude * direction

    def is_in_personal_space(self, human_pos: np.ndarray, robot_pos: np.ndarray,
                           relationship_type: str) -> bool:
        """Check if robot is in human's personal space"""
        distance = np.linalg.norm(human_pos - robot_pos)

        if relationship_type == 'stranger':
            return distance < self.zone_distances[ProxemicZone.SOCIAL][0]
        elif relationship_type == 'acquaintance':
            return distance < self.zone_distances[ProxemicZone.SOCIAL][0]
        elif relationship_type == 'friend':
            return distance < self.zone_distances[ProxemicZone.PERSONAL][0]
        elif relationship_type == 'family':
            return distance < self.zone_distances[ProxemicZone.INTIMATE][1]

        return distance < self.zone_distances[ProxemicZone.SOCIAL][0]

    def get_approach_recommendation(self, human_pos: np.ndarray, robot_pos: np.ndarray,
                                  relationship_type: str, cultural_context: str = 'western') -> Dict:
        """Get recommendation for approaching a human"""
        distance = np.linalg.norm(human_pos - robot_pos)
        comfortable_dist = self.get_comfortable_distance(relationship_type, cultural_context)

        approach_info = {
            'recommended_distance': comfortable_dist,
            'distance_status': 'too_close' if distance < comfortable_dist * 0.8 else
                             'too_far' if distance > comfortable_dist * 1.5 else
                             'comfortable',
            'recommended_action': 'move_back' if distance < comfortable_dist * 0.8 else
                                'move_closer' if distance > comfortable_dist * 1.5 else
                                'maintain_distance',
            'approach_angle': self.cultural_defaults[cultural_context]['approach_angle_preference']
        }

        return approach_info
```

### Social Force Model for Navigation

```python
class SocialForceModel:
    """Social force model for human-aware navigation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.proxemics_manager = ProxemicsManager(robot_config)
        self.social_force_params = {
            'human_repulsion_strength': 10.0,
            'human_repulsion_range': 2.0,
            'goal_attraction_strength': 5.0,
            'obstacle_repulsion_strength': 8.0,
            'social_convention_strength': 3.0
        }

    def calculate_total_social_force(self, robot_state: Dict, humans: List[Dict],
                                   goal_position: np.ndarray, obstacles: List[Dict]) -> np.ndarray:
        """Calculate total social force on robot"""
        total_force = np.zeros(3)

        # Goal attraction force
        goal_force = self._calculate_goal_force(
            robot_state['position'], goal_position
        )
        total_force += goal_force

        # Human repulsion forces
        for human in humans:
            human_force = self._calculate_human_force(
                robot_state['position'],
                human['position'],
                human.get('relationship_type', 'stranger'),
                human.get('cultural_context', 'western')
            )
            total_force += human_force

        # Obstacle avoidance forces
        for obstacle in obstacles:
            obstacle_force = self._calculate_obstacle_force(
                robot_state['position'], obstacle['position']
            )
            total_force += obstacle_force

        # Social convention forces (e.g., walk on right side)
        convention_force = self._calculate_social_convention_force(robot_state)
        total_force += convention_force

        return total_force

    def _calculate_goal_force(self, robot_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """Calculate force toward goal"""
        direction = goal_pos - robot_pos
        distance = np.linalg.norm(direction)

        if distance == 0:
            return np.zeros(3)

        direction_normalized = direction / distance
        force_magnitude = min(distance * self.social_force_params['goal_attraction_strength'], 5.0)

        return force_magnitude * direction_normalized

    def _calculate_human_force(self, robot_pos: np.ndarray, human_pos: np.ndarray,
                             relationship_type: str, cultural_context: str) -> np.ndarray:
        """Calculate social force from human"""
        displacement = human_pos - robot_pos
        distance = np.linalg.norm(displacement)

        if distance == 0:
            return np.zeros(3)

        direction = displacement / distance

        # Use proxemics manager to get appropriate force
        proxemic_force = self.proxemics_manager.calculate_social_force(
            human_pos, robot_pos, relationship_type, cultural_context
        )

        return proxemic_force

    def _calculate_obstacle_force(self, robot_pos: np.ndarray, obstacle_pos: np.ndarray) -> np.ndarray:
        """Calculate repulsive force from obstacle"""
        displacement = obstacle_pos - robot_pos
        distance = np.linalg.norm(displacement)

        if distance == 0:
            return np.zeros(3)

        direction = displacement / distance
        force_magnitude = (self.social_force_params['obstacle_repulsion_strength'] /
                          (distance ** 2 + 0.1))

        return force_magnitude * direction

    def _calculate_social_convention_force(self, robot_state: Dict) -> np.ndarray:
        """Calculate force based on social conventions (e.g., walk on right)"""
        # Simple model: encourage walking on right side of passages
        # In a real implementation, this would consider hallway geometry, traffic patterns, etc.
        convention_force = np.array([0.0, 0.0, 0.0])

        # Example: if near a wall, encourage keeping right
        if robot_state.get('near_wall', False):
            convention_force[1] = -1.0  # Slight push toward center/right

        return self.social_force_params['social_convention_strength'] * convention_force

    def adjust_robot_velocity(self, current_velocity: np.ndarray, social_force: np.ndarray,
                            max_velocity: float = 1.0) -> np.ndarray:
        """Adjust robot velocity based on social forces"""
        # Apply social force to current velocity
        new_velocity = current_velocity + 0.1 * social_force[:2]  # Only x,y components

        # Limit speed
        speed = np.linalg.norm(new_velocity)
        if speed > max_velocity:
            new_velocity = (new_velocity / speed) * max_velocity

        # Ensure minimum speed to keep moving
        if speed < 0.1 and np.linalg.norm(social_force[:2]) > 1.0:
            new_velocity = (social_force[:2] / np.linalg.norm(social_force[:2])) * 0.3

        return np.array([new_velocity[0], new_velocity[1], current_state.get('z_velocity', 0)])
```

## Social Navigation Algorithms

### Human-Aware Path Planning

```python
class HumanAwarePathPlanner:
    """Path planner that considers humans in the environment"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.proxemics_manager = ProxemicsManager(robot_config)
        self.social_force_model = SocialForceModel(robot_config)
        self.navigation_map = None
        self.social_cost_map = None

    def plan_path_with_humans(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                            humans: List[Dict], obstacles: List[Dict]) -> List[np.ndarray]:
        """Plan path that considers humans and social constraints"""
        # Create social cost map based on human positions
        self._create_social_cost_map(humans, start_pos, goal_pos)

        # Use A* with social costs
        path = self._social_aware_a_star(start_pos, goal_pos, humans, obstacles)

        return path

    def _create_social_cost_map(self, humans: List[Dict], start_pos: np.ndarray,
                              goal_pos: np.ndarray):
        """Create cost map that includes social costs"""
        # This would create a grid-based cost map
        # For simplicity, we'll define a function that calculates social cost
        self.social_cost_map = lambda pos: self._calculate_social_cost(pos, humans)

    def _calculate_social_cost(self, position: np.ndarray, humans: List[Dict]) -> float:
        """Calculate social cost of being at a position"""
        total_cost = 0.0

        for human in humans:
            distance = np.linalg.norm(position - human['position'])

            # Higher cost when too close to humans
            if distance < 1.0:  # Within 1 meter
                total_cost += 100.0 / (distance + 0.1)
            elif distance < 2.0:  # Within 2 meters
                total_cost += 10.0 / (distance + 0.1)
            elif distance > 5.0:  # Far away
                # Slight penalty for being too far from social interaction
                total_cost += 1.0

        return total_cost

    def _social_aware_a_star(self, start: np.ndarray, goal: np.ndarray,
                           humans: List[Dict], obstacles: List[Dict]) -> List[np.ndarray]:
        """A* path planning with social cost consideration"""
        import heapq

        # Convert to grid coordinates if needed
        def pos_to_grid(pos):
            return (int(pos[0] * 10), int(pos[1] * 10))  # 10cm resolution

        def grid_to_pos(grid_pos):
            return np.array([grid_pos[0] / 10.0, grid_pos[1] / 10.0, 0.0])

        start_grid = pos_to_grid(start)
        goal_grid = pos_to_grid(goal)

        # A* algorithm with social cost
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        visited = set()

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = [grid_to_pos(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(grid_to_pos(current))
                return list(reversed(path))

            visited.add(current)

            # Get neighbors
            neighbors = self._get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                # Check collision with obstacles
                neighbor_pos = grid_to_pos(neighbor)
                if self._check_collision(neighbor_pos, obstacles):
                    continue

                # Calculate tentative g_score
                tentative_g_score = g_score.get(current, float('inf')) + 1.0

                # Add social cost
                social_cost = self._calculate_social_cost(neighbor_pos, humans)
                tentative_g_score += social_cost * 0.1  # Weight social cost

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def _get_neighbors(self, grid_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((grid_pos[0] + dx, grid_pos[1] + dy))
        return neighbors

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _check_collision(self, position: np.ndarray, obstacles: List[Dict]) -> bool:
        """Check if position collides with obstacles"""
        for obstacle in obstacles:
            distance = np.linalg.norm(position - obstacle['position'])
            if distance < obstacle.get('radius', 0.1) + 0.1:  # Robot radius + safety margin
                return True
        return False

    def generate_social_navigation_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                                      humans: List[Dict], task_context: str) -> Dict:
        """Generate comprehensive social navigation plan"""
        # Plan main path
        path = self.plan_path_with_humans(start_pos, goal_pos, humans, [])

        # Add social behaviors along the path
        social_plan = {
            'main_path': path,
            'social_behaviors': [],
            'safety_margins': [],
            'interaction_points': []
        }

        # Identify points where interaction might be needed
        for i, waypoint in enumerate(path):
            nearby_humans = self._get_nearby_humans(waypoint, humans, radius=2.0)

            if nearby_humans:
                behavior = self._determine_social_behavior(waypoint, nearby_humans, task_context)
                social_plan['social_behaviors'].append({
                    'waypoint_index': i,
                    'position': waypoint,
                    'behavior': behavior,
                    'nearby_humans': nearby_humans
                })

        return social_plan

    def _get_nearby_humans(self, position: np.ndarray, humans: List[Dict],
                         radius: float) -> List[Dict]:
        """Get humans within radius of position"""
        nearby = []
        for human in humans:
            distance = np.linalg.norm(position - human['position'])
            if distance <= radius:
                nearby.append(human)
        return nearby

    def _determine_social_behavior(self, position: np.ndarray, nearby_humans: List[Dict],
                                 task_context: str) -> str:
        """Determine appropriate social behavior at position"""
        if task_context == 'greeting':
            return 'approach_and_greet'
        elif task_context == 'passing':
            if len(nearby_humans) == 1:
                return 'yield_and_pass'
            else:
                return 'wait_for_clear_path'
        elif task_context == 'following':
            return 'maintain_appropriate_distance'
        elif task_context == 'avoiding':
            return 'increase_distance'
        else:
            # Default behavior based on human density
            if len(nearby_humans) > 2:
                return 'wait_for_space'
            else:
                return 'normal_navigation'
```

### Socially-Aware Obstacle Avoidance

```python
class SocialObstacleAvoider:
    """Obstacle avoidance that considers social context"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.proxemics_manager = ProxemicsManager(robot_config)
        self.velocity_obstacles = []
        self.social_zones = {}

    def avoid_with_social_awareness(self, robot_state: Dict, humans: List[Dict],
                                  obstacles: List[Dict], goal_velocity: np.ndarray) -> np.ndarray:
        """Avoid obstacles while maintaining social appropriateness"""
        # Calculate preferred velocity toward goal
        preferred_velocity = goal_velocity.copy()

        # Apply social constraints
        social_adjusted_velocity = self._apply_social_constraints(
            robot_state, humans, preferred_velocity
        )

        # Apply collision avoidance
        collision_free_velocity = self._apply_collision_avoidance(
            robot_state, humans, obstacles, social_adjusted_velocity
        )

        return collision_free_velocity

    def _apply_social_constraints(self, robot_state: Dict, humans: List[Dict],
                                preferred_velocity: np.ndarray) -> np.ndarray:
        """Apply social constraints to preferred velocity"""
        current_pos = robot_state['position']
        adjusted_velocity = preferred_velocity.copy()

        for human in humans:
            distance = np.linalg.norm(current_pos - human['position'])

            # If too close, slow down and change direction
            comfortable_dist = self.proxemics_manager.get_comfortable_distance(
                human.get('relationship_type', 'stranger'),
                human.get('cultural_context', 'western')
            )

            if distance < comfortable_dist:
                # Calculate avoidance direction
                avoidance_direction = current_pos - human['position']
                avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)

                # Apply avoidance force proportional to closeness
                avoidance_strength = max(0, (comfortable_dist - distance) / comfortable_dist)
                adjusted_velocity += avoidance_direction * avoidance_strength * 0.5

        return adjusted_velocity

    def _apply_collision_avoidance(self, robot_state: Dict, humans: List[Dict],
                                 obstacles: List[Dict], preferred_velocity: np.ndarray) -> np.ndarray:
        """Apply collision avoidance using Velocity Obstacle method with social awareness"""
        current_pos = robot_state['position']
        current_vel = robot_state.get('velocity', np.zeros(3))

        # Create velocity obstacles for humans
        human_vos = self._create_human_velocity_obstacles(current_pos, current_vel, humans)

        # Create velocity obstacles for static obstacles
        obstacle_vos = self._create_obstacle_velocity_obstacles(current_pos, current_vel, obstacles)

        # Combine all velocity obstacles
        all_vos = human_vos + obstacle_vos

        # Find collision-free velocity
        collision_free_velocity = self._find_collision_free_velocity(
            current_vel, preferred_velocity, all_vos
        )

        return collision_free_velocity

    def _create_human_velocity_obstacles(self, robot_pos: np.ndarray, robot_vel: np.ndarray,
                                       humans: List[Dict]) -> List[Dict]:
        """Create velocity obstacles for humans"""
        vos = []

        for human in humans:
            # Vector from robot to human
            relative_pos = human['position'] - robot_pos
            distance = np.linalg.norm(relative_pos)

            # Calculate velocity obstacle parameters
            if distance > 0:
                # Unit vector toward human
                n = relative_pos / distance

                # Calculate aperture angle of velocity obstacle
                robot_radius = 0.3  # Approximate robot radius
                human_radius = 0.3  # Approximate human radius
                combined_radius = robot_radius + human_radius

                # Aperture angle
                sin_half_aperture = min(1.0, combined_radius / distance)
                aperture_angle = 2 * np.arcsin(sin_half_aperture)

                # Velocity obstacle boundary
                vo_boundary = {
                    'center': human['position'] - robot_pos,  # Relative position
                    'normal': n,
                    'aperture_angle': aperture_angle,
                    'distance': distance
                }

                vos.append(vo_boundary)

        return vos

    def _create_obstacle_velocity_obstacles(self, robot_pos: np.ndarray, robot_vel: np.ndarray,
                                          obstacles: List[Dict]) -> List[Dict]:
        """Create velocity obstacles for static obstacles"""
        vos = []

        for obstacle in obstacles:
            relative_pos = obstacle['position'] - robot_pos
            distance = np.linalg.norm(relative_pos)

            if distance > 0:
                n = relative_pos / distance
                obstacle_radius = obstacle.get('radius', 0.2)
                robot_radius = 0.3
                combined_radius = robot_radius + obstacle_radius

                sin_half_aperture = min(1.0, combined_radius / distance)
                aperture_angle = 2 * np.arcsin(sin_half_aperture)

                vo_boundary = {
                    'center': obstacle['position'] - robot_pos,
                    'normal': n,
                    'aperture_angle': aperture_angle,
                    'distance': distance,
                    'type': 'static'
                }

                vos.append(vo_boundary)

        return vos

    def _find_collision_free_velocity(self, current_vel: np.ndarray,
                                    preferred_vel: np.ndarray,
                                    velocity_obstacles: List[Dict]) -> np.ndarray:
        """Find collision-free velocity using sampling method"""
        # Define search space around preferred velocity
        search_radius = 1.0
        best_velocity = current_vel.copy()
        min_cost = float('inf')

        # Sample velocities in the neighborhood of preferred velocity
        for i in range(100):  # Sample 100 velocities
            # Generate random velocity in neighborhood
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0, search_radius)

            sample_velocity = preferred_vel.copy()
            sample_velocity[0] += speed * np.cos(angle)
            sample_velocity[1] += speed * np.sin(angle)

            # Check if velocity is in any velocity obstacle
            in_collision = False
            for vo in velocity_obstacles:
                if self._is_velocity_in_obstacle(sample_velocity, vo):
                    in_collision = True
                    break

            if not in_collision:
                # Calculate cost (distance from preferred velocity)
                cost = np.linalg.norm(sample_velocity - preferred_vel)

                if cost < min_cost:
                    min_cost = cost
                    best_velocity = sample_velocity

        # If no collision-free velocity found, try to find least bad option
        if min_cost == float('inf'):
            best_velocity = self._find_least_bad_velocity(preferred_vel, velocity_obstacles)

        return best_velocity

    def _is_velocity_in_obstacle(self, velocity: np.ndarray, vo: Dict) -> bool:
        """Check if velocity is within velocity obstacle"""
        # Calculate angle between velocity and obstacle normal
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude == 0:
            return False  # Zero velocity is not in obstacle

        vel_direction = velocity / vel_magnitude
        normal = vo['normal']

        # Calculate angle
        cos_angle = np.dot(vel_direction, normal)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Check if within aperture
        return angle < vo['aperture_angle'] / 2

    def _find_least_bad_velocity(self, preferred_vel: np.ndarray,
                               velocity_obstacles: List[Dict]) -> np.ndarray:
        """Find the velocity that collides with the fewest obstacles"""
        # For simplicity, return preferred velocity with reduced speed
        speed = np.linalg.norm(preferred_vel)
        if speed > 0.2:  # If we have significant preferred speed
            direction = preferred_vel / speed
            return direction * 0.2  # Move slowly
        else:
            return preferred_vel * 0.5  # Otherwise, just reduce existing velocity
```

## Human-Robot Interaction Systems

### Multi-Modal Communication Interface

```python
class MultiModalHRIInterface:
    """Multi-modal interface for human-robot interaction"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.speech_recognizer = SpeechRecognizer(robot_config)
        self.speech_synthesizer = SpeechSynthesizer(robot_config)
        self.gesture_recognizer = GestureRecognizer(robot_config)
        self.gesture_generator = GestureGenerator(robot_config)
        self.face_tracker = FaceTracker(robot_config)
        self.emotion_detector = EmotionDetector(robot_config)
        self.intent_classifier = IntentClassifier(robot_config)

    def process_human_interaction(self, sensor_data: Dict) -> Dict:
        """Process multi-modal human input and generate response"""
        interaction_result = {
            'detected_speech': None,
            'detected_gestures': [],
            'detected_faces': [],
            'recognized_emotions': [],
            'understood_intent': None,
            'robot_response': None
        }

        # Process speech input
        if 'audio' in sensor_data:
            speech_text = self.speech_recognizer.recognize_speech(sensor_data['audio'])
            interaction_result['detected_speech'] = speech_text

            if speech_text:
                intent = self.intent_classifier.classify_intent(speech_text)
                interaction_result['understood_intent'] = intent

        # Process gesture input
        if 'video' in sensor_data:
            gestures = self.gesture_recognizer.recognize_gestures(sensor_data['video'])
            interaction_result['detected_gestures'] = gestures

        # Process face tracking
        faces = self.face_tracker.track_faces(sensor_data.get('video', None))
        interaction_result['detected_faces'] = faces

        # Process emotion detection
        if 'video' in sensor_data:
            emotions = self.emotion_detector.detect_emotions(sensor_data['video'])
            interaction_result['recognized_emotions'] = emotions

        # Generate appropriate response
        response = self._generate_robot_response(interaction_result)
        interaction_result['robot_response'] = response

        return interaction_result

    def _generate_robot_response(self, interaction_data: Dict) -> Dict:
        """Generate robot response based on interaction data"""
        response = {
            'speech_output': '',
            'gesture_output': [],
            'face_expression': 'neutral',
            'action_plan': []
        }

        # Determine response based on intent and context
        intent = interaction_data.get('understood_intent')
        faces = interaction_data.get('detected_faces', [])
        emotions = interaction_data.get('recognized_emotions', [])

        if intent:
            if intent['type'] == 'greeting':
                response['speech_output'] = self._generate_greeting_response(faces)
                response['gesture_output'] = ['wave']
                response['face_expression'] = 'happy'
            elif intent['type'] == 'navigation_query':
                response['speech_output'] = self._generate_navigation_response(intent['destination'])
                response['gesture_output'] = ['point_direction']
            elif intent['type'] == 'assistance_request':
                response['speech_output'] = self._generate_assistance_response(intent['request'])
                response['gesture_output'] = ['nod']
            else:
                response['speech_output'] = "I understand you're talking to me. How can I help?"
                response['gesture_output'] = ['attention_pose']

        # Add emotional adaptation
        if emotions:
            dominant_emotion = self._get_dominant_emotion(emotions)
            response['face_expression'] = self._map_emotion_to_expression(dominant_emotion)

        # Generate action plan
        response['action_plan'] = self._generate_action_plan(intent, response)

        return response

    def _generate_greeting_response(self, faces: List[Dict]) -> str:
        """Generate appropriate greeting response"""
        if len(faces) == 1:
            return "Hello! It's nice to meet you."
        elif len(faces) > 1:
            return "Hello everyone! How can I help you today?"
        else:
            return "Hello! I'm here to assist."

    def _generate_navigation_response(self, destination: str) -> str:
        """Generate navigation response"""
        return f"I can help you find the {destination}. Please follow me."

    def _generate_assistance_response(self, request: str) -> str:
        """Generate assistance response"""
        return f"I can help you with {request}. Please follow these steps..."

    def _get_dominant_emotion(self, emotions: List[Dict]) -> str:
        """Get the most prominent emotion from detected emotions"""
        if not emotions:
            return 'neutral'

        # For simplicity, return the emotion with highest confidence
        return max(emotions, key=lambda x: x.get('confidence', 0)).get('emotion', 'neutral')

    def _map_emotion_to_expression(self, emotion: str) -> str:
        """Map human emotion to appropriate robot expression"""
        emotion_mapping = {
            'happy': 'happy',
            'sad': 'concerned',
            'angry': 'concerned',
            'surprised': 'surprised',
            'fear': 'concerned',
            'disgust': 'concerned',
            'neutral': 'neutral'
        }

        return emotion_mapping.get(emotion, 'neutral')

    def _generate_action_plan(self, intent: Dict, response: Dict) -> List[Dict]:
        """Generate action plan based on intent and response"""
        action_plan = []

        if intent and intent['type'] == 'navigation_query':
            # Add navigation action
            action_plan.append({
                'type': 'navigate',
                'destination': intent.get('destination', ''),
                'follow_mode': True
            })

        # Add gesture actions
        for gesture in response['gesture_output']:
            action_plan.append({
                'type': 'gesture',
                'gesture_type': gesture,
                'timing': 'concurrent'
            })

        # Add speech action
        if response['speech_output']:
            action_plan.append({
                'type': 'speak',
                'text': response['speech_output'],
                'timing': 'concurrent'
            })

        return action_plan

    def execute_interaction_response(self, response: Dict):
        """Execute the robot's response to human interaction"""
        # Execute speech
        if response.get('speech_output'):
            self.speech_synthesizer.speak(response['speech_output'])

        # Execute gestures
        for gesture in response.get('gesture_output', []):
            self.gesture_generator.execute_gesture(gesture)

        # Set face expression
        self._set_face_expression(response.get('face_expression', 'neutral'))

        # Execute action plan
        for action in response.get('action_plan', []):
            self._execute_action(action)

    def _set_face_expression(self, expression: str):
        """Set robot's face expression"""
        # This would control the robot's facial expression system
        print(f"Setting face expression to: {expression}")

    def _execute_action(self, action: Dict):
        """Execute a specific action"""
        action_type = action['type']

        if action_type == 'navigate':
            # Execute navigation
            print(f"Navigating to {action['destination']}")
        elif action_type == 'gesture':
            # Execute gesture
            print(f"Performing gesture: {action['gesture_type']}")
        elif action_type == 'speak':
            # Speak text (already handled above)
            pass
```

### Gesture Recognition and Generation

```python
class GestureRecognizer:
    """Recognize human gestures for interaction"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.gesture_models = self._load_gesture_models()
        self.tracking_history = {}

    def _load_gesture_models(self):
        """Load pre-trained gesture recognition models"""
        # In practice, this would load actual ML models
        # For this example, we'll define gesture templates
        return {
            'wave': self._create_wave_template(),
            'point': self._create_point_template(),
            'come_here': self._create_come_here_template(),
            'stop': self._create_stop_template(),
            'follow_me': self._create_follow_me_template()
        }

    def _create_wave_template(self):
        """Create template for wave gesture"""
        # Simplified template - in practice, this would be more complex
        return {
            'name': 'wave',
            'key_poses': [
                {'arm_angle': 90, 'hand_position': 'side'},
                {'arm_angle': 180, 'hand_position': 'front'},
                {'arm_angle': 90, 'hand_position': 'side'}
            ],
            'temporal_pattern': 'oscillatory',
            'confidence_threshold': 0.7
        }

    def _create_point_template(self):
        """Create template for pointing gesture"""
        return {
            'name': 'point',
            'key_poses': [
                {'arm_angle': 45, 'hand_orientation': 'extended'},
                {'arm_angle': 45, 'hand_orientation': 'extended', 'duration': 0.5}
            ],
            'temporal_pattern': 'sustained',
            'confidence_threshold': 0.8
        }

    def _create_come_here_template(self):
        """Create template for 'come here' gesture"""
        return {
            'name': 'come_here',
            'key_poses': [
                {'arm_angle': 90, 'hand_shape': 'open_palm'},
                {'arm_angle': 45, 'hand_shape': 'beckoning', 'direction': 'toward_robot'}
            ],
            'temporal_pattern': 'repetitive',
            'confidence_threshold': 0.75
        }

    def _create_stop_template(self):
        """Create template for stop gesture"""
        return {
            'name': 'stop',
            'key_poses': [
                {'arm_angle': 90, 'hand_orientation': 'palm_forward'},
                {'arm_angle': 90, 'hand_orientation': 'palm_forward', 'duration': 1.0}
            ],
            'temporal_pattern': 'sustained',
            'confidence_threshold': 0.8
        }

    def _create_follow_me_template(self):
        """Create template for 'follow me' gesture"""
        return {
            'name': 'follow_me',
            'key_poses': [
                {'arm_angle': 45, 'hand_orientation': 'extended'},
                {'arm_angle': 0, 'hand_orientation': 'extended', 'direction': 'forward'},
                {'arm_angle': 45, 'hand_orientation': 'extended'}
            ],
            'temporal_pattern': 'directional',
            'confidence_threshold': 0.7
        }

    def recognize_gestures(self, video_frames: List[np.ndarray]) -> List[Dict]:
        """Recognize gestures from video frames"""
        recognized_gestures = []

        # Extract pose information from frames
        poses = self._extract_poses(video_frames)

        # Compare to gesture templates
        for gesture_name, template in self.gesture_models.items():
            confidence = self._match_gesture_template(poses, template)

            if confidence > template['confidence_threshold']:
                recognized_gestures.append({
                    'gesture': gesture_name,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })

        return recognized_gestures

    def _extract_poses(self, video_frames: List[np.ndarray]) -> List[Dict]:
        """Extract pose information from video frames"""
        # This would use pose estimation algorithms
        # For this example, we'll simulate pose extraction
        poses = []

        for i, frame in enumerate(video_frames):
            # Simulate pose extraction
            pose = {
                'frame_index': i,
                'shoulder_left': np.random.rand(2) * 100 + 100,  # Simulated coordinates
                'shoulder_right': np.random.rand(2) * 100 + 150,
                'elbow_left': np.random.rand(2) * 100 + 120,
                'elbow_right': np.random.rand(2) * 100 + 170,
                'wrist_left': np.random.rand(2) * 100 + 140,
                'wrist_right': np.random.rand(2) * 100 + 190,
                'timestamp': datetime.now()
            }
            poses.append(pose)

        return poses

    def _match_gesture_template(self, poses: List[Dict], template: Dict) -> float:
        """Match poses to gesture template"""
        # This would implement actual gesture matching algorithm
        # For this example, we'll use a simplified approach
        if not poses:
            return 0.0

        # Calculate how well poses match template
        match_score = 0.0
        total_checks = 0

        for pose in poses:
            # Check each key pose in template
            for key_pose in template['key_poses']:
                # Simplified matching
                match_score += 0.1  # Add score for each pose
                total_checks += 1

        if total_checks > 0:
            match_score = match_score / total_checks
        else:
            match_score = 0.0

        # Apply temporal pattern matching
        if template['temporal_pattern'] == 'oscillatory':
            # Check for oscillatory movement
            oscillation_score = self._check_oscillatory_pattern(poses)
            match_score = (match_score + oscillation_score) / 2

        return min(match_score, 1.0)

    def _check_oscillatory_pattern(self, poses: List[Dict]) -> float:
        """Check if poses show oscillatory pattern"""
        # Simplified oscillation detection
        if len(poses) < 3:
            return 0.0

        # Calculate movement variance
        movements = []
        for i in range(1, len(poses)):
            pos1 = np.array(list(poses[i-1].get('wrist_right', [0, 0])))
            pos2 = np.array(list(poses[i].get('wrist_right', [0, 0])))
            movement = np.linalg.norm(pos2 - pos1)
            movements.append(movement)

        # Check for oscillation pattern
        if len(movements) >= 3:
            avg_movement = sum(movements) / len(movements)
            # More oscillatory if movements vary significantly
            variance = sum((m - avg_movement)**2 for m in movements) / len(movements)
            return min(variance * 2, 1.0)

        return 0.0

class GestureGenerator:
    """Generate robot gestures for interaction"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.gesture_sequences = self._define_gesture_sequences()

    def _define_gesture_sequences(self):
        """Define gesture sequences for robot"""
        return {
            'wave': {
                'name': 'wave',
                'sequence': [
                    {'joint': 'right_arm', 'position': [0, 0, 0], 'duration': 0.2},
                    {'joint': 'right_arm', 'position': [90, 0, 0], 'duration': 0.1},
                    {'joint': 'right_arm', 'position': [180, 0, 0], 'duration': 0.2},
                    {'joint': 'right_arm', 'position': [90, 0, 0], 'duration': 0.1},
                    {'joint': 'right_arm', 'position': [0, 0, 0], 'duration': 0.2}
                ]
            },
            'point_direction': {
                'name': 'point_direction',
                'sequence': [
                    {'joint': 'right_arm', 'position': [45, 0, 0], 'duration': 0.3},
                    {'joint': 'right_arm', 'position': [45, 0, 0], 'duration': 1.0},  # Hold
                    {'joint': 'right_arm', 'position': [0, 0, 0], 'duration': 0.3}
                ]
            },
            'nod': {
                'name': 'nod',
                'sequence': [
                    {'joint': 'head', 'position': [0, 0, 0], 'duration': 0.1},
                    {'joint': 'head', 'position': [0, -10, 0], 'duration': 0.1},
                    {'joint': 'head', 'position': [0, 0, 0], 'duration': 0.1},
                    {'joint': 'head', 'position': [0, -10, 0], 'duration': 0.1},
                    {'joint': 'head', 'position': [0, 0, 0], 'duration': 0.1}
                ]
            },
            'attention_pose': {
                'name': 'attention_pose',
                'sequence': [
                    {'joint': 'right_arm', 'position': [30, 0, 0], 'duration': 0.2},
                    {'joint': 'left_arm', 'position': [30, 0, 0], 'duration': 0.2},
                    {'joint': 'head', 'position': [0, 0, 0], 'duration': 0.1}
                ]
            }
        }

    def execute_gesture(self, gesture_type: str):
        """Execute a specific gesture"""
        if gesture_type in self.gesture_sequences:
            sequence = self.gesture_sequences[gesture_type]['sequence']

            for step in sequence:
                self._move_joint(step['joint'], step['position'])
                self._wait(step['duration'])
        else:
            print(f"Gesture '{gesture_type}' not defined")

    def _move_joint(self, joint_name: str, position: List[float]):
        """Move robot joint to specified position"""
        # This would interface with robot's joint control system
        print(f"Moving {joint_name} to {position}")

    def _wait(self, duration: float):
        """Wait for specified duration"""
        import time
        time.sleep(duration)

    def generate_contextual_gesture(self, context: Dict) -> str:
        """Generate appropriate gesture based on context"""
        if context.get('interaction_type') == 'greeting':
            return 'wave'
        elif context.get('request_type') == 'direction':
            return 'point_direction'
        elif context.get('request_type') == 'acknowledgment':
            return 'nod'
        elif context.get('state') == 'ready':
            return 'attention_pose'
        else:
            return 'attention_pose'  # Default gesture
```

## Emotional Expression and Recognition

### Emotion Detection and Response

```python
class EmotionDetector:
    """Detect human emotions for social interaction"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.emotion_models = self._load_emotion_models()

    def _load_emotion_models(self):
        """Load emotion detection models"""
        # In practice, this would load actual ML models
        # For this example, we'll define detection rules
        return {
            'happy': self._create_happy_detector(),
            'sad': self._create_sad_detector(),
            'angry': self._create_angry_detector(),
            'surprised': self._create_surprised_detector(),
            'neutral': self._create_neutral_detector()
        }

    def _create_happy_detector(self):
        """Create detector for happy emotion"""
        return {
            'facial_features': {
                'mouth_curvature': (0.3, 1.0),  # Smiling
                'eye_wrinkles': (0.2, 1.0),     # Crow's feet
                'eyebrow_position': (-0.1, 0.3) # Slightly raised
            },
            'confidence_boost': 1.2
        }

    def _create_sad_detector(self):
        """Create detector for sad emotion"""
        return {
            'facial_features': {
                'mouth_curvature': (-1.0, -0.2),  # Frowning
                'eyebrow_position': (-0.5, -0.1), # Furrowed
                'eye_openness': (0.1, 0.6)       # Droopy eyes
            },
            'confidence_boost': 1.1
        }

    def _create_angry_detector(self):
        """Create detector for angry emotion"""
        return {
            'facial_features': {
                'eyebrow_position': (-0.7, -0.2), # Knitted brows
                'mouth_tension': (0.6, 1.0),      # Tight lips
                'jaw_clench': (0.5, 1.0)         # Clenched jaw
            },
            'confidence_boost': 1.3
        }

    def _create_surprised_detector(self):
        """Create detector for surprised emotion"""
        return {
            'facial_features': {
                'eye_openness': (0.8, 1.0),      # Wide eyes
                'mouth_openness': (0.4, 1.0),     # Open mouth
                'eyebrow_height': (0.5, 1.0)     # Raised eyebrows
            },
            'confidence_boost': 1.4
        }

    def _create_neutral_detector(self):
        """Create detector for neutral emotion"""
        return {
            'facial_features': {
                'mouth_curvature': (-0.2, 0.2),   # Neutral mouth
                'eye_openness': (0.4, 0.7),       # Normal eyes
                'eyebrow_position': (-0.1, 0.1)  # Neutral brows
            },
            'confidence_boost': 1.0
        }

    def detect_emotions(self, video_frames: List[np.ndarray]) -> List[Dict]:
        """Detect emotions from video frames"""
        detected_emotions = []

        for frame in video_frames:
            # Extract facial features from frame
            facial_features = self._extract_facial_features(frame)

            # Classify emotion
            emotion_result = self._classify_emotion(facial_features)

            if emotion_result:
                detected_emotions.append(emotion_result)

        return detected_emotions

    def _extract_facial_features(self, frame: np.ndarray) -> Dict:
        """Extract facial features from image frame"""
        # This would use facial landmark detection
        # For this example, we'll simulate feature extraction
        features = {
            'mouth_curvature': np.random.uniform(-0.5, 0.8),
            'eye_openness': np.random.uniform(0.3, 0.9),
            'eyebrow_position': np.random.uniform(-0.3, 0.4),
            'mouth_openness': np.random.uniform(0.0, 0.6),
            'eye_wrinkles': np.random.uniform(0.0, 0.5),
            'jaw_clench': np.random.uniform(0.0, 0.3)
        }

        return features

    def _classify_emotion(self, features: Dict) -> Dict:
        """Classify emotion based on facial features"""
        emotion_scores = {}

        for emotion, model in self.emotion_models.items():
            score = self._calculate_emotion_score(features, model)
            emotion_scores[emotion] = score

        # Find emotion with highest score
        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[best_emotion]

            # Only return if confidence is above threshold
            if confidence > 0.6:
                return {
                    'emotion': best_emotion,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }

        return None

    def _calculate_emotion_score(self, features: Dict, model: Dict) -> float:
        """Calculate emotion score based on features and model"""
        score = 0.0
        feature_count = 0

        for feature_name, (min_val, max_val) in model['facial_features'].items():
            if feature_name in features:
                feature_val = features[feature_name]

                # Normalize feature value to [0, 1] range
                if min_val != max_val:
                    normalized_val = max(0, min(1, (feature_val - min_val) / (max_val - min_val)))
                    score += normalized_val
                    feature_count += 1

        if feature_count > 0:
            score = score / feature_count
            # Apply confidence boost
            score = min(score * model.get('confidence_boost', 1.0), 1.0)

        return score

class EmotionExpressionSystem:
    """System for expressing emotions through robot behavior"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.emotion_behaviors = self._define_emotion_behaviors()

    def _define_emotion_behaviors(self):
        """Define behaviors for different emotions"""
        return {
            'happy': {
                'head_movement': {'speed': 0.8, 'amplitude': 5, 'pattern': 'gentle_nod'},
                'arm_movement': {'speed': 0.6, 'amplitude': 10, 'pattern': 'light_wave'},
                'speech_prosody': {'pitch': 1.2, 'speed': 1.1, 'volume': 1.05},
                'facial_expression': 'smiling',
                'posture': 'open_and_welcoming'
            },
            'sad': {
                'head_movement': {'speed': 0.3, 'amplitude': 2, 'pattern': 'slow_dip'},
                'arm_movement': {'speed': 0.2, 'amplitude': 5, 'pattern': 'minimal'},
                'speech_prosody': {'pitch': 0.8, 'speed': 0.9, 'volume': 0.9},
                'facial_expression': 'concerned',
                'posture': 'closed_and_considerate'
            },
            'angry': {
                'head_movement': {'speed': 0.9, 'amplitude': 8, 'pattern': 'sharp_turns'},
                'arm_movement': {'speed': 0.8, 'amplitude': 15, 'pattern': 'assertive_gestures'},
                'speech_prosody': {'pitch': 1.1, 'speed': 1.3, 'volume': 1.2},
                'facial_expression': 'concerned',
                'posture': 'alert_and_attentive'
            },
            'surprised': {
                'head_movement': {'speed': 1.0, 'amplitude': 10, 'pattern': 'quick_turn'},
                'arm_movement': {'speed': 0.9, 'amplitude': 12, 'pattern': 'open_gesture'},
                'speech_prosody': {'pitch': 1.4, 'speed': 1.0, 'volume': 1.1},
                'facial_expression': 'surprised',
                'posture': 'attentive_and_reactive'
            },
            'neutral': {
                'head_movement': {'speed': 0.5, 'amplitude': 3, 'pattern': 'steady'},
                'arm_movement': {'speed': 0.4, 'amplitude': 8, 'pattern': 'controlled'},
                'speech_prosody': {'pitch': 1.0, 'speed': 1.0, 'volume': 1.0},
                'facial_expression': 'neutral',
                'posture': 'balanced_and_ready'
            },
            'concerned': {
                'head_movement': {'speed': 0.4, 'amplitude': 4, 'pattern': 'gentle_tilt'},
                'arm_movement': {'speed': 0.3, 'amplitude': 6, 'pattern': 'soothing_gesture'},
                'speech_prosody': {'pitch': 0.9, 'speed': 0.8, 'volume': 0.95},
                'facial_expression': 'empathetic',
                'posture': 'supportive_and_caring'
            }
        }

    def express_emotion(self, emotion: str, intensity: float = 1.0):
        """Express an emotion through robot behavior"""
        if emotion in self.emotion_behaviors:
            behavior = self.emotion_behaviors[emotion]

            # Apply behavior with intensity scaling
            self._apply_head_movement(behavior['head_movement'], intensity)
            self._apply_arm_movement(behavior['arm_movement'], intensity)
            self._apply_speech_prosody(behavior['speech_prosody'], intensity)
            self._apply_facial_expression(behavior['facial_expression'])
            self._apply_posture(behavior['posture'])
        else:
            # Default to neutral if emotion not defined
            self.express_emotion('neutral')

    def _apply_head_movement(self, head_params: Dict, intensity: float):
        """Apply head movement pattern"""
        speed = head_params['speed'] * intensity
        amplitude = head_params['amplitude'] * intensity
        pattern = head_params['pattern']

        print(f"Applying head movement: {pattern}, speed={speed}, amplitude={amplitude}")

    def _apply_arm_movement(self, arm_params: Dict, intensity: float):
        """Apply arm movement pattern"""
        speed = arm_params['speed'] * intensity
        amplitude = arm_params['amplitude'] * intensity
        pattern = arm_params['pattern']

        print(f"Applying arm movement: {pattern}, speed={speed}, amplitude={amplitude}")

    def _apply_speech_prosody(self, speech_params: Dict, intensity: float):
        """Apply speech prosody modifications"""
        pitch = speech_params['pitch'] * intensity
        speed = speech_params['speed'] * intensity
        volume = speech_params['volume'] * intensity

        print(f"Modifying speech: pitch={pitch}, speed={speed}, volume={volume}")

    def _apply_facial_expression(self, expression: str):
        """Apply facial expression"""
        print(f"Setting facial expression to: {expression}")

    def _apply_posture(self, posture: str):
        """Apply body posture"""
        print(f"Adjusting posture to: {posture}")

    def respond_to_human_emotion(self, human_emotion: str):
        """Generate appropriate emotional response to human emotion"""
        response_mapping = {
            'happy': 'happy',
            'sad': 'concerned',
            'angry': 'concerned',
            'surprised': 'surprised',
            'neutral': 'neutral'
        }

        response_emotion = response_mapping.get(human_emotion, 'neutral')
        self.express_emotion(response_emotion, intensity=0.8)
```

## Practical Implementation Considerations

### Social Navigation Integration

```python
class IntegratedSocialNavigation:
    """Integration of all social navigation and interaction components"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.proxemics_manager = ProxemicsManager(robot_config)
        self.social_force_model = SocialForceModel(robot_config)
        self.human_aware_planner = HumanAwarePathPlanner(robot_config)
        self.obstacle_avoider = SocialObstacleAvoider(robot_config)
        self.hri_interface = MultiModalHRIInterface(robot_config)
        self.emotion_system = EmotionExpressionSystem(robot_config)

    def execute_social_navigation_task(self, goal: str, context: Dict = None):
        """Execute a complete social navigation task"""
        task_result = {
            'navigation_success': False,
            'interaction_success': False,
            'social_compliance': True,
            'task_completed': False,
            'interaction_log': []
        }

        # Parse goal and context
        goal_pos = self._parse_goal_position(goal)
        humans = context.get('humans', []) if context else []

        # Plan socially-aware path
        social_plan = self.human_aware_planner.generate_social_navigation_plan(
            self._get_robot_position(), goal_pos, humans, 'navigation'
        )

        # Execute navigation with social awareness
        navigation_result = self._execute_social_navigation(
            social_plan['main_path'], humans
        )

        task_result['navigation_success'] = navigation_result['success']
        task_result['social_compliance'] = navigation_result['social_compliance']

        # Handle interactions along the way
        for behavior in social_plan['social_behaviors']:
            interaction = self._handle_social_interaction(
                behavior['position'], behavior['behavior'], humans
            )
            task_result['interaction_log'].append(interaction)

        # Check if task is completed
        final_distance = np.linalg.norm(
            self._get_robot_position() - goal_pos
        )
        task_result['task_completed'] = final_distance < 1.0  # Within 1 meter

        return task_result

    def _parse_goal_position(self, goal_description: str) -> np.ndarray:
        """Parse goal description to get position"""
        # This would use the cognitive map or other location system
        # For now, return a default position
        if 'kitchen' in goal_description.lower():
            return np.array([5.0, 3.0, 0.0])
        elif 'living room' in goal_description.lower():
            return np.array([2.0, 1.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0.0])  # Default home position

    def _get_robot_position(self) -> np.ndarray:
        """Get current robot position"""
        # This would interface with localization system
        return np.array([0.0, 0.0, 0.0])  # Default starting position

    def _execute_social_navigation(self, path: List[np.ndarray],
                                 humans: List[Dict]) -> Dict:
        """Execute navigation while maintaining social awareness"""
        result = {
            'success': True,
            'social_compliance': True,
            'path_followed': [],
            'social_violations': []
        }

        for i, waypoint in enumerate(path):
            # Check for humans near current position
            nearby_humans = self._get_nearby_humans(waypoint, humans, radius=3.0)

            # Calculate social forces
            robot_state = {'position': waypoint}
            social_force = self.social_force_model.calculate_total_social_force(
                robot_state, nearby_humans, path[-1], []
            )

            # Adjust movement based on social forces
            if np.linalg.norm(social_force) > 5.0:
                # Significant social force - adjust behavior
                result['social_violations'].append({
                    'waypoint_index': i,
                    'position': waypoint,
                    'force_magnitude': np.linalg.norm(social_force),
                    'recommended_action': 'wait_or_detour'
                })

            # Record path followed
            result['path_followed'].append(waypoint)

        return result

    def _get_nearby_humans(self, position: np.ndarray, humans: List[Dict],
                         radius: float) -> List[Dict]:
        """Get humans within radius of position"""
        nearby = []
        for human in humans:
            distance = np.linalg.norm(position - human['position'])
            if distance <= radius:
                nearby.append(human)
        return nearby

    def _handle_social_interaction(self, position: np.ndarray,
                                 behavior: str, humans: List[Dict]) -> Dict:
        """Handle social interaction at specific position"""
        interaction_result = {
            'position': position,
            'behavior_executed': behavior,
            'humans_interacted': len(humans),
            'response_generated': False,
            'interaction_quality': 0.0
        }

        if behavior == 'approach_and_greet':
            # Approach humans and greet them
            if humans:
                closest_human = min(humans, key=lambda h:
                    np.linalg.norm(position - h['position']))

                # Generate greeting interaction
                greeting_result = self._generate_greeting_interaction(closest_human)
                interaction_result.update(greeting_result)

        elif behavior == 'yield_and_pass':
            # Yield to humans before passing
            self._execute_yield_behavior(humans)
            interaction_result['response_generated'] = True

        elif behavior == 'wait_for_clear_path':
            # Wait for path to clear
            self._execute_wait_behavior(humans)
            interaction_result['response_generated'] = True

        return interaction_result

    def _generate_greeting_interaction(self, human: Dict) -> Dict:
        """Generate greeting interaction with human"""
        # Simulate greeting interaction
        interaction = {
            'response_generated': True,
            'interaction_quality': 0.9,  # High quality greeting
            'greeting_type': 'wave_and_hello',
            'emotional_response': 'positive'
        }

        # Express positive emotion
        self.emotion_system.express_emotion('happy', intensity=0.7)

        return interaction

    def _execute_yield_behavior(self, humans: List[Dict]):
        """Execute yielding behavior"""
        print("Robot yielding to humans...")
        # This would involve stopping and waiting
        self.emotion_system.express_emotion('neutral', intensity=0.5)

    def _execute_wait_behavior(self, humans: List[Dict]):
        """Execute waiting behavior"""
        print("Robot waiting for clear path...")
        # This would involve pausing navigation
        self.emotion_system.express_emotion('patient', intensity=0.6)

    def handle_unexpected_social_situation(self, sensor_data: Dict) -> Dict:
        """Handle unexpected social situations detected by sensors"""
        # Process sensor data to detect social situation
        interaction_data = self.hri_interface.process_human_interaction(sensor_data)

        # Determine appropriate response
        response = self._determine_situation_response(interaction_data)

        # Execute response
        self.hri_interface.execute_interaction_response(response)

        return {
            'situation_handled': True,
            'response_executed': response,
            'interaction_data': interaction_data
        }

    def _determine_situation_response(self, interaction_data: Dict) -> Dict:
        """Determine appropriate response to social situation"""
        response = {
            'speech_output': '',
            'gesture_output': [],
            'face_expression': 'neutral',
            'action_plan': []
        }

        # Check for specific social situations
        if interaction_data['detected_speech']:
            # Someone is talking to the robot
            response['speech_output'] = "Hello! How can I help you?"
            response['gesture_output'] = ['wave']
            response['face_expression'] = 'happy'
        elif interaction_data['detected_gestures']:
            # Someone made a gesture toward the robot
            gesture = interaction_data['detected_gestures'][0]['gesture']
            if gesture == 'wave':
                response['gesture_output'] = ['wave_back']
                response['speech_output'] = "Hello!"
                response['face_expression'] = 'happy'
            elif gesture == 'point':
                response['speech_output'] = "I see you're pointing. How can I assist?"
                response['gesture_output'] = ['nod']
        elif interaction_data['recognized_emotions']:
            # Someone appears to have strong emotions
            emotion = interaction_data['recognized_emotions'][0]['emotion']
            self.emotion_system.respond_to_human_emotion(emotion)
            response['face_expression'] = self._emotion_to_expression(emotion)

        return response

    def _emotion_to_expression(self, emotion: str) -> str:
        """Map emotion to appropriate facial expression"""
        mapping = {
            'happy': 'happy',
            'sad': 'concerned',
            'angry': 'concerned',
            'surprised': 'surprised',
            'fear': 'concerned',
            'disgust': 'concerned',
            'neutral': 'neutral'
        }
        return mapping.get(emotion, 'neutral')

    def evaluate_social_behavior(self, interaction_log: List[Dict]) -> Dict:
        """Evaluate the quality of social behavior"""
        evaluation = {
            'social_compliance_score': 0.0,
            'interaction_quality': 0.0,
            'human_comfort_level': 0.0,
            'recommendations': []
        }

        if not interaction_log:
            return evaluation

        # Calculate social compliance score
        compliant_interactions = sum(
            1 for interaction in interaction_log
            if interaction.get('response_generated', False)
        )
        evaluation['social_compliance_score'] = compliant_interactions / len(interaction_log)

        # Calculate interaction quality
        total_quality = sum(
            interaction.get('interaction_quality', 0.0)
            for interaction in interaction_log
        )
        evaluation['interaction_quality'] = total_quality / len(interaction_log) if interaction_log else 0.0

        # Generate recommendations for improvement
        if evaluation['social_compliance_score'] < 0.8:
            evaluation['recommendations'].append(
                "Improve response rate to social situations"
            )

        if evaluation['interaction_quality'] < 0.7:
            evaluation['recommendations'].append(
                "Enhance quality of social responses"
            )

        return evaluation
```

## Assessment Questions

1. Explain the concept of proxemics and its importance in human-robot interaction. How would you implement proxemics-aware behavior in a humanoid robot?

2. Design a social force model that can handle multiple humans with different relationships (stranger, friend, family) in the environment.

3. Implement a gesture recognition system that can identify common human gestures like waving, pointing, and beckoning.

4. Create a multi-modal interaction system that can process speech, gestures, and facial expressions simultaneously.

5. Design an emotional expression system that allows a humanoid robot to convey appropriate emotions based on social context.

## Practice Exercises

1. **Social Navigation**: Implement a path planning algorithm that considers human spatial preferences and social conventions.

2. **Gesture Recognition**: Create a gesture recognition system using computer vision techniques to identify human gestures.

3. **Emotion Detection**: Develop a facial expression recognition system that can detect basic human emotions.

4. **Interaction Sequences**: Design interaction sequences for common HRI scenarios like greeting, requesting assistance, and providing information.

## Summary

Human-robot interaction and social navigation are essential capabilities for humanoid robots operating in human environments. This chapter covered:

- Proxemics principles and personal space management for appropriate human-robot distance
- Social force models that consider humans as dynamic obstacles with social constraints
- Human-aware path planning algorithms that account for social conventions
- Multi-modal interaction systems that process speech, gestures, and facial expressions
- Gesture recognition and generation for natural communication
- Emotional expression and recognition systems
- Integration of all components into a cohesive social navigation system

The implementation of these capabilities enables humanoid robots to operate naturally and safely in human environments, respecting social conventions and engaging in appropriate interactions with people.