---
sidebar_position: 9
title: Chapter 9 - Motion Planning from Language
---

# Chapter 9 - Motion Planning from Language

In this chapter, we explore the complex task of translating natural language commands into executable motion plans for robotic systems. Motion planning from language bridges the gap between high-level human instructions and low-level robot control, enabling intuitive human-robot interaction and task execution.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the challenges of translating natural language to motion plans
- Implement semantic parsing for motion planning commands
- Design hierarchical motion planning systems that interpret language
- Integrate language understanding with robotic motion planning
- Evaluate and optimize motion plans generated from language
- Handle ambiguity and uncertainty in natural language commands

## Introduction to Language-Guided Motion Planning

Language-guided motion planning involves converting natural language instructions into sequences of robot actions that achieve the desired goal. This process requires understanding the semantics of language, mapping them to spatial and temporal concepts, and generating executable motion plans that are safe, efficient, and goal-oriented.

### Key Challenges

1. **Semantic Ambiguity**: Natural language often contains ambiguous references that require contextual understanding
2. **Spatial Reasoning**: Translating linguistic spatial concepts to geometric representations
3. **Temporal Planning**: Understanding temporal relationships in language for sequential planning
4. **Uncertainty Handling**: Managing uncertainty in both language interpretation and environment perception
5. **Multi-Modal Integration**: Combining linguistic, visual, and spatial information

## Semantic Parsing for Motion Planning

Semantic parsing converts natural language commands into structured representations that can be used for motion planning:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

@dataclass
class MotionCommand:
    """Structured representation of a motion command"""
    action_type: str
    parameters: Dict[str, Any]
    spatial_reference: Optional[str] = None
    temporal_constraints: Optional[Dict] = None
    confidence: float = 1.0

class ActionType(Enum):
    """Types of motion actions"""
    MOVE = "move"
    GRASP = "grasp"
    PLACE = "place"
    NAVIGATE = "navigate"
    ROTATE = "rotate"
    APPROACH = "approach"
    AVOID = "avoid"

class SemanticParser(nn.Module):
    """Semantic parser for motion planning commands"""

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.action_classifier = nn.Linear(hidden_dim, len(ActionType))
        self.parameter_extractor = nn.Linear(hidden_dim, 64)  # For extracting parameters
        self.spatial_encoder = nn.Linear(hidden_dim, 128)    # For spatial reasoning

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse tokens into motion command structure"""
        embedded = self.embedding(tokens)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Get final hidden state for classification
        final_hidden = hidden[-1]  # Last layer

        # Classify action type
        action_logits = self.action_classifier(final_hidden)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Extract parameters
        params = self.parameter_extractor(final_hidden)

        # Encode spatial information
        spatial_info = self.spatial_encoder(final_hidden)

        return {
            'action_probs': action_probs,
            'parameters': params,
            'spatial_info': spatial_info
        }

    def parse_command(self, text: str, tokenizer) -> MotionCommand:
        """Parse natural language text into motion command"""
        # Tokenize input
        tokens = tokenizer(text)
        token_tensor = torch.tensor([tokens])

        # Get parsing results
        results = self.forward(token_tensor)

        # Determine action type
        action_idx = torch.argmax(results['action_probs'], dim=-1).item()
        action_type = list(ActionType)[action_idx]

        # Extract parameters
        params = self._extract_parameters(results['parameters'], text)

        # Extract spatial reference
        spatial_ref = self._extract_spatial_reference(text)

        # Create motion command
        command = MotionCommand(
            action_type=action_type.value,
            parameters=params,
            spatial_reference=spatial_ref
        )

        return command

    def _extract_parameters(self, param_tensor: torch.Tensor, text: str) -> Dict[str, Any]:
        """Extract parameters from tensor and text"""
        # This would implement parameter extraction logic
        # For now, we'll use a simple approach based on text patterns
        params = {}

        # Extract numeric values
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            params['magnitude'] = float(numbers[0]) if numbers[0] else 1.0

        # Extract direction from text
        directions = ['forward', 'backward', 'left', 'right', 'up', 'down']
        for direction in directions:
            if direction in text.lower():
                params['direction'] = direction
                break

        # Extract object if present
        object_match = re.search(r'the (\w+)', text.lower())
        if object_match:
            params['object'] = object_match.group(1)

        return params

    def _extract_spatial_reference(self, text: str) -> Optional[str]:
        """Extract spatial reference from text"""
        # Look for spatial references like "to the table", "near the door"
        spatial_patterns = [
            r'to the (\w+)',
            r'at the (\w+)',
            r'near the (\w+)',
            r'by the (\w+)',
            r'next to the (\w+)'
        ]

        for pattern in spatial_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)

        return None

# Example command parser with more sophisticated logic
class AdvancedCommandParser:
    """Advanced parser for complex motion commands"""

    def __init__(self):
        self.action_keywords = {
            ActionType.MOVE: ['move', 'go', 'walk', 'drive', 'navigate'],
            ActionType.GRASP: ['grasp', 'pick', 'take', 'grab', 'hold'],
            ActionType.PLACE: ['place', 'put', 'set', 'drop', 'release'],
            ActionType.NAVIGATE: ['navigate', 'go to', 'head to', 'move to'],
            ActionType.ROTATE: ['turn', 'rotate', 'pivot', 'spin'],
            ActionType.APPROACH: ['approach', 'get close to', 'move toward'],
            ActionType.AVOID: ['avoid', 'steer clear of', 'go around']
        }

        self.spatial_keywords = {
            'directions': ['forward', 'backward', 'left', 'right', 'up', 'down', 'forward', 'back'],
            'distances': ['meters', 'feet', 'inches', 'centimeters'],
            'locations': ['table', 'chair', 'door', 'window', 'kitchen', 'bedroom', 'living room']
        }

    def parse(self, text: str) -> Optional[MotionCommand]:
        """Parse complex motion command from text"""
        text_lower = text.lower().strip()

        # Identify action type
        action_type = self._identify_action_type(text_lower)
        if not action_type:
            return None

        # Extract parameters
        params = self._extract_detailed_parameters(text_lower)

        # Extract spatial reference
        spatial_ref = self._extract_detailed_spatial_reference(text_lower)

        # Determine confidence based on completeness of parsing
        confidence = self._calculate_confidence(text_lower, action_type, params)

        return MotionCommand(
            action_type=action_type.value,
            parameters=params,
            spatial_reference=spatial_ref,
            confidence=confidence
        )

    def _identify_action_type(self, text: str) -> Optional[ActionType]:
        """Identify action type from text"""
        for action_type, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return action_type
        return None

    def _extract_detailed_parameters(self, text: str) -> Dict[str, Any]:
        """Extract detailed parameters from text"""
        params = {}

        # Extract numeric values (distances, angles, etc.)
        numbers = re.findall(r'(\d+\.?\d*)\s*(meters?|feet|inches?|centimeters?|degrees?|cm|ft|m)', text)
        if numbers:
            for num, unit in numbers:
                value = float(num)
                if 'meter' in unit or unit == 'm':
                    params['distance'] = value
                elif 'degree' in unit or unit == 'cm':
                    params['angle'] = value
                elif 'foot' in unit or unit == 'ft':
                    params['distance'] = value * 0.3048  # Convert to meters

        # Extract direction
        for direction in self.spatial_keywords['directions']:
            if direction in text:
                params['direction'] = direction
                break

        # Extract object
        object_patterns = [
            r'pick up the (\w+)',
            r'grasp the (\w+)',
            r'take the (\w+)',
            r'move to the (\w+)',
            r'go to the (\w+)'
        ]
        for pattern in object_patterns:
            match = re.search(pattern, text)
            if match:
                params['object'] = match.group(1)
                break

        return params

    def _extract_detailed_spatial_reference(self, text: str) -> Optional[str]:
        """Extract detailed spatial reference from text"""
        # Look for spatial references with more complex patterns
        patterns = [
            r'to the (\w+)',
            r'at the (\w+)',
            r'near the (\w+)',
            r'by the (\w+)',
            r'next to the (\w+)',
            r'beside the (\w+)',
            r'around the (\w+)',
            r'toward the (\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _calculate_confidence(self, text: str, action_type: ActionType, params: Dict) -> float:
        """Calculate confidence in parsing result"""
        confidence = 0.5  # Base confidence

        # Increase confidence if action type is identified
        if action_type:
            confidence += 0.3

        # Increase confidence if key parameters are found
        if params:
            confidence += 0.2 * min(len(params), 3) / 3

        # Cap at 1.0
        return min(confidence, 1.0)

# Example tokenizer for the semantic parser
class SimpleTokenizer:
    """Simple tokenizer for motion planning commands"""

    def __init__(self):
        self.vocab = {
            '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3,
            'move': 4, 'go': 5, 'to': 6, 'the': 7, 'and': 8,
            'grasp': 9, 'pick': 10, 'up': 11, 'place': 12,
            'turn': 13, 'left': 14, 'right': 15, 'forward': 16,
            'backward': 17, 'navigate': 18, 'approach': 19,
            'avoid': 20, 'object': 21, 'table': 22, 'chair': 23
        }

    def __call__(self, text: str) -> List[int]:
        """Tokenize text"""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, self.vocab['<UNK>']))
        return tokens
```

## Spatial Reasoning and Geometric Mapping

Converting linguistic spatial concepts to geometric representations is crucial for motion planning:

```python
import numpy as np
from scipy.spatial import distance
from typing import Union

class SpatialReasoner:
    """Reason about spatial relationships from language"""

    def __init__(self):
        self.spatial_relations = {
            'near': self._near_relation,
            'far': self._far_relation,
            'left_of': self._left_of_relation,
            'right_of': self._right_of_relation,
            'in_front_of': self._in_front_of_relation,
            'behind': self._behind_relation,
            'above': self._above_relation,
            'below': self._below_relation,
            'next_to': self._next_to_relation
        }

    def interpret_spatial_command(self, command: MotionCommand, robot_pose: np.ndarray,
                                 object_poses: Dict[str, np.ndarray]) -> np.ndarray:
        """Interpret spatial command and return target pose"""
        if command.spatial_reference and command.spatial_reference in object_poses:
            target_object_pose = object_poses[command.spatial_reference]

            # Apply spatial relation based on command parameters
            target_pose = self._apply_spatial_relation(
                robot_pose, target_object_pose, command
            )

            return target_pose

        # If no spatial reference, return robot pose or calculate from parameters
        return self._calculate_pose_from_parameters(robot_pose, command)

    def _apply_spatial_relation(self, robot_pose: np.ndarray, target_pose: np.ndarray,
                                command: MotionCommand) -> np.ndarray:
        """Apply spatial relation to calculate target pose"""
        relation = self._infer_spatial_relation(command)
        params = command.parameters

        if relation == 'near':
            return self._near_relation(robot_pose, target_pose, params)
        elif relation == 'left_of':
            return self._left_of_relation(robot_pose, target_pose, params)
        elif relation == 'right_of':
            return self._right_of_relation(robot_pose, target_pose, params)
        elif relation == 'in_front_of':
            return self._in_front_of_relation(robot_pose, target_pose, params)
        elif relation == 'behind':
            return self._behind_relation(robot_pose, target_pose, params)
        else:
            return target_pose  # Default to target pose

    def _infer_spatial_relation(self, command: MotionCommand) -> str:
        """Infer spatial relation from command"""
        # This would use more sophisticated inference in practice
        # For now, we'll use simple heuristics
        if 'left' in command.parameters:
            return 'left_of'
        elif 'right' in command.parameters:
            return 'right_of'
        elif 'forward' in command.parameters or 'front' in command.parameters:
            return 'in_front_of'
        elif 'back' in command.parameters or 'behind' in command.parameters:
            return 'behind'
        elif 'near' in command.parameters or command.parameters.get('distance', 1.0) < 1.0:
            return 'near'
        else:
            return 'near'  # Default

    def _near_relation(self, robot_pose: np.ndarray, target_pose: np.ndarray,
                       params: Dict) -> np.ndarray:
        """Calculate pose 'near' target"""
        distance_val = params.get('distance', 0.5)  # Default 0.5m

        # Calculate vector from robot to target
        direction_vector = target_pose[:3] - robot_pose[:3]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate new position
        new_position = target_pose[:3] - direction_vector * distance_val

        # Keep orientation from target or robot
        new_pose = np.copy(target_pose)
        new_pose[:3] = new_position

        return new_pose

    def _left_of_relation(self, robot_pose: np.ndarray, target_pose: np.ndarray,
                          params: Dict) -> np.ndarray:
        """Calculate pose 'left of' target"""
        distance_val = params.get('distance', 0.5)

        # Calculate left direction (perpendicular to robot-target vector)
        forward_vector = target_pose[:3] - robot_pose[:3]
        forward_vector = forward_vector / np.linalg.norm(forward_vector)

        # Calculate left vector (assuming Z is up)
        left_vector = np.array([-forward_vector[1], forward_vector[0], 0])
        left_vector = left_vector / np.linalg.norm(left_vector)

        # Calculate new position
        new_position = target_pose[:3] + left_vector * distance_val

        new_pose = np.copy(target_pose)
        new_pose[:3] = new_position

        return new_pose

    def _right_of_relation(self, robot_pose: np.ndarray, target_pose: np.ndarray,
                           params: Dict) -> np.ndarray:
        """Calculate pose 'right of' target"""
        distance_val = params.get('distance', 0.5)

        # Calculate right direction (opposite of left)
        forward_vector = target_pose[:3] - robot_pose[:3]
        forward_vector = forward_vector / np.linalg.norm(forward_vector)

        # Calculate right vector (assuming Z is up)
        right_vector = np.array([forward_vector[1], -forward_vector[0], 0])
        right_vector = right_vector / np.linalg.norm(right_vector)

        # Calculate new position
        new_position = target_pose[:3] + right_vector * distance_val

        new_pose = np.copy(target_pose)
        new_pose[:3] = new_position

        return new_pose

    def _in_front_of_relation(self, robot_pose: np.ndarray, target_pose: np.ndarray,
                              params: Dict) -> np.ndarray:
        """Calculate pose 'in front of' target"""
        distance_val = params.get('distance', 1.0)

        # Calculate vector from target to robot
        direction_vector = robot_pose[:3] - target_pose[:3]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate new position in front of target
        new_position = target_pose[:3] + direction_vector * distance_val

        new_pose = np.copy(target_pose)
        new_pose[:3] = new_position

        return new_pose

    def _behind_relation(self, robot_pose: np.ndarray, target_pose: np.ndarray,
                         params: Dict) -> np.ndarray:
        """Calculate pose 'behind' target"""
        distance_val = params.get('distance', 1.0)

        # Calculate vector from robot to target
        direction_vector = target_pose[:3] - robot_pose[:3]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate new position behind target
        new_position = target_pose[:3] + direction_vector * distance_val

        new_pose = np.copy(target_pose)
        new_pose[:3] = new_position

        return new_pose

    def _calculate_pose_from_parameters(self, robot_pose: np.ndarray,
                                        command: MotionCommand) -> np.ndarray:
        """Calculate pose from command parameters"""
        new_pose = np.copy(robot_pose)

        if command.action_type == ActionType.MOVE.value:
            direction = command.parameters.get('direction', 'forward')
            distance = command.parameters.get('distance', 1.0)

            if direction == 'forward':
                new_pose[0] += distance
            elif direction == 'backward':
                new_pose[0] -= distance
            elif direction == 'left':
                new_pose[1] += distance
            elif direction == 'right':
                new_pose[1] -= distance
            elif direction == 'up':
                new_pose[2] += distance
            elif direction == 'down':
                new_pose[2] -= distance

        return new_pose

# Coordinate system converter
class CoordinateSystemConverter:
    """Convert between different coordinate systems"""

    def __init__(self):
        self.reference_frames = {}

    def world_to_robot_frame(self, world_pose: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """Convert world coordinates to robot's local frame"""
        # Calculate transformation matrix
        R = self._rotation_matrix_from_pose(robot_pose)
        t = robot_pose[:3]

        # Transform point
        world_point = world_pose[:3]
        robot_point = R.T @ (world_point - t)

        # Combine with orientation
        robot_pose_local = np.copy(world_pose)
        robot_pose_local[:3] = robot_point

        return robot_pose_local

    def robot_to_world_frame(self, robot_pose: np.ndarray, robot_base_pose: np.ndarray) -> np.ndarray:
        """Convert robot's local frame to world coordinates"""
        R = self._rotation_matrix_from_pose(robot_base_pose)
        t = robot_base_pose[:3]

        # Transform point
        local_point = robot_pose[:3]
        world_point = R @ local_point + t

        # Combine with orientation
        world_pose = np.copy(robot_pose)
        world_pose[:3] = world_point

        return world_pose

    def _rotation_matrix_from_pose(self, pose: np.ndarray) -> np.ndarray:
        """Extract rotation matrix from pose (assuming pose contains rotation parameters)"""
        # This is a simplified version - in practice, this would depend on rotation representation
        # (quaternion, Euler angles, etc.)
        return np.eye(3)  # Identity for now
```

## Hierarchical Motion Planning from Language

Hierarchical planning breaks down complex language commands into manageable subtasks:

```python
from queue import Queue
import heapq

class HierarchicalMotionPlanner:
    """Hierarchical motion planner for language-guided tasks"""

    def __init__(self):
        self.task_decomposer = TaskDecomposer()
        self.low_level_planner = LowLevelMotionPlanner()
        self.high_level_planner = HighLevelTaskPlanner()

    def plan_from_language(self, command: str) -> List[MotionCommand]:
        """Plan motion from natural language command"""
        # Parse the command
        parser = AdvancedCommandParser()
        parsed_command = parser.parse(command)

        if not parsed_command:
            raise ValueError(f"Could not parse command: {command}")

        # Decompose high-level task
        subtasks = self.task_decomposer.decompose_task(parsed_command)

        # Plan each subtask
        motion_plan = []
        for subtask in subtasks:
            subplan = self._plan_subtask(subtask)
            motion_plan.extend(subplan)

        return motion_plan

    def _plan_subtask(self, subtask: MotionCommand) -> List[MotionCommand]:
        """Plan a single subtask"""
        if self._is_primitive_action(subtask):
            # For primitive actions, use low-level planner
            return [subtask]
        else:
            # Decompose further if needed
            sub_subtasks = self.task_decomposer.decompose_task(subtask)
            plan = []
            for sub_subtask in sub_subtasks:
                plan.extend(self._plan_subtask(sub_subtask))
            return plan

    def _is_primitive_action(self, command: MotionCommand) -> bool:
        """Check if command is a primitive action"""
        primitive_actions = [
            ActionType.MOVE.value,
            ActionType.ROTATE.value,
            ActionType.GRASP.value,
            ActionType.PLACE.value
        ]
        return command.action_type in primitive_actions

class TaskDecomposer:
    """Decompose high-level tasks into subtasks"""

    def __init__(self):
        self.decomposition_rules = {
            'pick and place': self._decompose_pick_and_place,
            'navigate to': self._decompose_navigate_to,
            'go around': self._decompose_go_around,
            'approach and grasp': self._decompose_approach_and_grasp
        }

    def decompose_task(self, command: MotionCommand) -> List[MotionCommand]:
        """Decompose a task into subtasks"""
        # Check for compound commands
        command_text = f"{command.action_type} {command.spatial_reference}".lower()

        for pattern, decomposer in self.decomposition_rules.items():
            if pattern in command_text:
                return decomposer(command)

        # If no specific rule applies, return the command as is
        return [command]

    def _decompose_pick_and_place(self, command: MotionCommand) -> List[MotionCommand]:
        """Decompose pick and place task"""
        object_name = command.parameters.get('object', 'object')
        place_location = command.spatial_reference

        return [
            MotionCommand(
                action_type=ActionType.NAVIGATE.value,
                parameters={'target': f'{object_name}'},
                spatial_reference=object_name
            ),
            MotionCommand(
                action_type=ActionType.APPROACH.value,
                parameters={'object': object_name},
                spatial_reference=object_name
            ),
            MotionCommand(
                action_type=ActionType.GRASP.value,
                parameters={'object': object_name},
                spatial_reference=object_name
            ),
            MotionCommand(
                action_type=ActionType.NAVIGATE.value,
                parameters={'target': place_location},
                spatial_reference=place_location
            ),
            MotionCommand(
                action_type=ActionType.PLACE.value,
                parameters={'object': object_name},
                spatial_reference=place_location
            )
        ]

    def _decompose_navigate_to(self, command: MotionCommand) -> List[MotionCommand]:
        """Decompose navigate to task"""
        target = command.spatial_reference or command.parameters.get('target')

        return [
            MotionCommand(
                action_type=ActionType.NAVIGATE.value,
                parameters={'target': target},
                spatial_reference=target
            )
        ]

    def _decompose_go_around(self, command: MotionCommand) -> List[MotionCommand]:
        """Decompose go around task"""
        obstacle = command.spatial_reference or command.parameters.get('obstacle')

        return [
            MotionCommand(
                action_type=ActionType.AVOID.value,
                parameters={'obstacle': obstacle},
                spatial_reference=obstacle
            )
        ]

    def _decompose_approach_and_grasp(self, command: MotionCommand) -> List[MotionCommand]:
        """Decompose approach and grasp task"""
        object_name = command.parameters.get('object', 'object')

        return [
            MotionCommand(
                action_type=ActionType.APPROACH.value,
                parameters={'object': object_name},
                spatial_reference=object_name
            ),
            MotionCommand(
                action_type=ActionType.GRASP.value,
                parameters={'object': object_name},
                spatial_reference=object_name
            )
        ]

class LowLevelMotionPlanner:
    """Low-level motion planner for primitive actions"""

    def __init__(self):
        self.path_planner = PathPlanner()
        self.trajectory_generator = TrajectoryGenerator()

    def plan_primitive_action(self, command: MotionCommand,
                             start_pose: np.ndarray,
                             environment_map) -> List[np.ndarray]:
        """Plan primitive action as trajectory"""
        if command.action_type == ActionType.MOVE.value:
            return self._plan_move_action(command, start_pose, environment_map)
        elif command.action_type == ActionType.ROTATE.value:
            return self._plan_rotate_action(command, start_pose)
        elif command.action_type == ActionType.GRASP.value:
            return self._plan_grasp_action(command, start_pose, environment_map)
        elif command.action_type == ActionType.PLACE.value:
            return self._plan_place_action(command, start_pose, environment_map)
        else:
            # For other actions, return a simple trajectory to target
            target_pose = self._calculate_target_pose(command, start_pose, environment_map)
            return self.path_planner.plan_path(start_pose, target_pose, environment_map)

    def _plan_move_action(self, command: MotionCommand,
                         start_pose: np.ndarray,
                         environment_map) -> List[np.ndarray]:
        """Plan move action"""
        # Calculate target based on direction and distance
        direction = command.parameters.get('direction', 'forward')
        distance = command.parameters.get('distance', 1.0)

        target_pose = np.copy(start_pose)
        if direction == 'forward':
            target_pose[0] += distance
        elif direction == 'backward':
            target_pose[0] -= distance
        elif direction == 'left':
            target_pose[1] += distance
        elif direction == 'right':
            target_pose[1] -= distance
        elif direction == 'up':
            target_pose[2] += distance
        elif direction == 'down':
            target_pose[2] -= distance

        # Plan path to target
        path = self.path_planner.plan_path(start_pose, target_pose, environment_map)
        return path

    def _plan_rotate_action(self, command: MotionCommand,
                           start_pose: np.ndarray) -> List[np.ndarray]:
        """Plan rotation action"""
        angle = command.parameters.get('angle', 90.0)  # degrees

        # For rotation, we might just return the final pose
        # or generate intermediate poses for smooth rotation
        final_pose = np.copy(start_pose)
        # This would modify the orientation part of the pose
        # Implementation depends on rotation representation

        return [start_pose, final_pose]

    def _plan_grasp_action(self, command: MotionCommand,
                          start_pose: np.ndarray,
                          environment_map) -> List[np.ndarray]:
        """Plan grasp action"""
        object_name = command.parameters.get('object', 'object')

        # Find object pose in environment
        object_pose = environment_map.get_object_pose(object_name)
        if object_pose is None:
            raise ValueError(f"Object {object_name} not found in environment")

        # Plan approach to object
        approach_pose = self._calculate_approach_pose(object_pose)
        path_to_object = self.path_planner.plan_path(start_pose, approach_pose, environment_map)

        # Plan grasp motion (simplified)
        grasp_poses = [approach_pose, object_pose]  # Simplified

        return path_to_object + grasp_poses

    def _plan_place_action(self, command: MotionCommand,
                          start_pose: np.ndarray,
                          environment_map) -> List[np.ndarray]:
        """Plan place action"""
        target_location = command.spatial_reference

        # Find target location in environment
        target_pose = environment_map.get_location_pose(target_location)
        if target_pose is None:
            raise ValueError(f"Location {target_location} not found in environment")

        # Plan path to target location
        path = self.path_planner.plan_path(start_pose, target_pose, environment_map)

        # Add placement motion
        place_poses = [target_pose]  # Simplified

        return path + place_poses

    def _calculate_target_pose(self, command: MotionCommand,
                              start_pose: np.ndarray,
                              environment_map) -> np.ndarray:
        """Calculate target pose from command"""
        if command.spatial_reference:
            # Look up target in environment
            target_pose = environment_map.get_location_pose(command.spatial_reference)
            if target_pose is not None:
                return target_pose

        # If no spatial reference, calculate from parameters
        return self._calculate_pose_from_params(command, start_pose)

    def _calculate_pose_from_params(self, command: MotionCommand,
                                   start_pose: np.ndarray) -> np.ndarray:
        """Calculate pose from command parameters"""
        target_pose = np.copy(start_pose)

        # Apply any position changes from parameters
        if 'x' in command.parameters:
            target_pose[0] = command.parameters['x']
        if 'y' in command.parameters:
            target_pose[1] = command.parameters['y']
        if 'z' in command.parameters:
            target_pose[2] = command.parameters['z']

        return target_pose

    def _calculate_approach_pose(self, object_pose: np.ndarray) -> np.ndarray:
        """Calculate approach pose for grasping"""
        approach_pose = np.copy(object_pose)

        # Move back from object to avoid collision during approach
        approach_distance = 0.2  # 20 cm from object
        approach_pose[0] -= approach_distance  # Assuming forward approach

        return approach_pose

class HighLevelTaskPlanner:
    """High-level task planner for complex language commands"""

    def __init__(self):
        self.task_network = TaskNetwork()

    def plan_complex_task(self, commands: List[MotionCommand]) -> List[MotionCommand]:
        """Plan complex task with dependencies"""
        # Build task dependency graph
        dependency_graph = self._build_dependency_graph(commands)

        # Topologically sort tasks
        sorted_tasks = self._topological_sort(dependency_graph)

        return sorted_tasks

    def _build_dependency_graph(self, commands: List[MotionCommand]) -> Dict:
        """Build dependency graph for tasks"""
        graph = {i: [] for i in range(len(commands))}  # Task ID to dependencies

        # Simple dependency rules:
        # - Place must happen after grasp
        # - Approach must happen before grasp
        # - Navigation must happen before approach

        for i, cmd in enumerate(commands):
            if cmd.action_type == ActionType.PLACE.value:
                # Find corresponding grasp action
                for j, prev_cmd in enumerate(commands[:i]):
                    if (prev_cmd.action_type == ActionType.GRASP.value and
                        prev_cmd.parameters.get('object') == cmd.parameters.get('object')):
                        graph[i].append(j)

        return graph

    def _topological_sort(self, graph: Dict) -> List[MotionCommand]:
        """Topologically sort tasks based on dependencies"""
        # This is a simplified implementation
        # In practice, this would use Kahn's algorithm or DFS
        sorted_indices = list(range(len(graph)))
        return sorted_indices  # Placeholder

class TaskNetwork:
    """Network representation of tasks and dependencies"""
    pass

# Path planning for motion execution
class PathPlanner:
    """Path planner for motion execution"""

    def __init__(self):
        self.planning_algorithm = 'a_star'  # Default algorithm

    def plan_path(self, start: np.ndarray, goal: np.ndarray,
                  environment_map) -> List[np.ndarray]:
        """Plan path from start to goal"""
        if self.planning_algorithm == 'a_star':
            return self._a_star_plan(start, goal, environment_map)
        elif self.planning_algorithm == 'rrt':
            return self._rrt_plan(start, goal, environment_map)
        else:
            return [start, goal]  # Direct path as fallback

    def _a_star_plan(self, start: np.ndarray, goal: np.ndarray,
                     environment_map) -> List[np.ndarray]:
        """A* path planning implementation"""
        # This is a simplified version of A* algorithm
        # In practice, this would use a proper A* implementation
        # with grid-based or continuous space representation

        # For now, return a straight line path
        # with some intermediate points
        path = [start]

        # Calculate direction vector
        direction = goal[:3] - start[:3]
        distance = np.linalg.norm(direction)

        if distance > 0.1:  # If distance is significant
            num_intermediate = max(1, int(distance / 0.1))  # 10cm steps
            for i in range(1, num_intermediate):
                intermediate = start + (direction * i / num_intermediate)
                path.append(intermediate)

        path.append(goal)
        return path

    def _rrt_plan(self, start: np.ndarray, goal: np.ndarray,
                  environment_map) -> List[np.ndarray]:
        """RRT path planning implementation"""
        # Simplified RRT implementation
        return [start, goal]  # Placeholder

class TrajectoryGenerator:
    """Generate smooth trajectories from paths"""

    def __init__(self):
        self.smoothing_factor = 0.1

    def generate_trajectory(self, path: List[np.ndarray],
                           velocity_profile: str = 'trapezoidal') -> List[np.ndarray]:
        """Generate smooth trajectory from path"""
        if velocity_profile == 'trapezoidal':
            return self._generate_trapezoidal_trajectory(path)
        else:
            return path  # Return path as is if no smoothing needed

    def _generate_trapezoidal_trajectory(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Generate trajectory with trapezoidal velocity profile"""
        # Add intermediate points for smooth motion
        smooth_path = []

        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]

            # Add the start point
            smooth_path.append(start_point)

            # If the distance between points is large, add intermediate points
            dist = np.linalg.norm(end_point[:3] - start_point[:3])
            if dist > 0.2:  # If more than 20cm apart
                num_points = int(dist / 0.05)  # 5cm steps
                for j in range(1, num_points):
                    intermediate = start_point + (end_point - start_point) * j / num_points
                    smooth_path.append(intermediate)

        # Add the final point
        if path:
            smooth_path.append(path[-1])

        return smooth_path
```

## Integration with Robotic Systems

Integrating language-guided motion planning with robotic systems requires careful coordination:

```python
import threading
import time
from collections import deque

class LanguageGuidedMotionSystem:
    """Complete system for language-guided motion planning"""

    def __init__(self, robot_controller, environment_map):
        self.robot_controller = robot_controller
        self.environment_map = environment_map
        self.motion_planner = HierarchicalMotionPlanner()
        self.spatial_reasoner = SpatialReasoner()
        self.safety_monitor = SafetyMonitor()

        # Communication queues
        self.command_queue = Queue()
        self.status_queue = Queue()

        # State tracking
        self.current_plan = []
        self.current_step = 0
        self.is_executing = False
        self.execution_thread = None

    def execute_language_command(self, command_text: str):
        """Execute a language command end-to-end"""
        try:
            # Plan the motion
            motion_plan = self.motion_planner.plan_from_language(command_text)

            # Validate the plan
            if not self.safety_monitor.validate_plan(motion_plan, self.environment_map):
                raise ValueError("Motion plan failed safety validation")

            # Execute the plan
            self._execute_plan(motion_plan)

            return {"status": "success", "plan": motion_plan}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _execute_plan(self, plan: List[MotionCommand]):
        """Execute a motion plan step by step"""
        self.current_plan = plan
        self.current_step = 0
        self.is_executing = True

        # Get current robot pose
        current_pose = self.robot_controller.get_current_pose()

        for i, command in enumerate(plan):
            if not self.is_executing:
                break

            # Update current step
            self.current_step = i

            # Execute the command
            self._execute_single_command(command, current_pose)

            # Update current pose after execution
            current_pose = self.robot_controller.get_current_pose()

        self.is_executing = False

    def _execute_single_command(self, command: MotionCommand, current_pose: np.ndarray):
        """Execute a single motion command"""
        if command.action_type == ActionType.NAVIGATE.value:
            self._execute_navigation_command(command, current_pose)
        elif command.action_type == ActionType.GRASP.value:
            self._execute_grasp_command(command)
        elif command.action_type == ActionType.PLACE.value:
            self._execute_place_command(command)
        elif command.action_type == ActionType.MOVE.value:
            self._execute_move_command(command, current_pose)
        elif command.action_type == ActionType.ROTATE.value:
            self._execute_rotate_command(command)
        elif command.action_type == ActionType.APPROACH.value:
            self._execute_approach_command(command, current_pose)
        elif command.action_type == ActionType.AVOID.value:
            self._execute_avoid_command(command, current_pose)

    def _execute_navigation_command(self, command: MotionCommand, current_pose: np.ndarray):
        """Execute navigation command"""
        # Determine target pose based on spatial reference
        if command.spatial_reference:
            target_objects = self.environment_map.get_all_objects()
            if command.spatial_reference in target_objects:
                target_pose = target_objects[command.spatial_reference]
            else:
                # If object not found, try to interpret as location
                target_pose = self.environment_map.get_location_pose(command.spatial_reference)

                if target_pose is None:
                    # Calculate from parameters
                    target_pose = self.spatial_reasoner._calculate_pose_from_parameters(
                        current_pose, command
                    )
        else:
            target_pose = self.spatial_reasoner._calculate_pose_from_parameters(
                current_pose, command
            )

        # Plan and execute path
        path_planner = PathPlanner()
        path = path_planner.plan_path(current_pose, target_pose, self.environment_map)

        # Execute the path
        self.robot_controller.follow_path(path)

    def _execute_grasp_command(self, command: MotionCommand):
        """Execute grasp command"""
        object_name = command.parameters.get('object', 'object')

        # Find object in environment
        target_objects = self.environment_map.get_all_objects()
        if object_name in target_objects:
            object_pose = target_objects[object_name]
            self.robot_controller.grasp_object_at_pose(object_pose, object_name)
        else:
            # Try to find object based on other parameters
            self.robot_controller.search_for_object(object_name)

    def _execute_place_command(self, command: MotionCommand):
        """Execute place command"""
        object_name = command.parameters.get('object', 'object')
        location = command.spatial_reference

        if location:
            # Find location in environment
            location_pose = self.environment_map.get_location_pose(location)
            if location_pose is not None:
                self.robot_controller.place_object_at_pose(object_name, location_pose)
            else:
                # Use current pose if location not found
                current_pose = self.robot_controller.get_current_pose()
                self.robot_controller.place_object_at_pose(object_name, current_pose)
        else:
            # Use current pose if no location specified
            current_pose = self.robot_controller.get_current_pose()
            self.robot_controller.place_object_at_pose(object_name, current_pose)

    def _execute_move_command(self, command: MotionCommand, current_pose: np.ndarray):
        """Execute move command"""
        # Calculate target pose from parameters
        target_pose = self.spatial_reasoner._calculate_pose_from_parameters(
            current_pose, command
        )

        # Move to target
        self.robot_controller.move_to_pose(target_pose)

    def _execute_rotate_command(self, command: MotionCommand):
        """Execute rotate command"""
        angle = command.parameters.get('angle', 90.0)
        self.robot_controller.rotate_by_angle(angle)

    def _execute_approach_command(self, command: MotionCommand, current_pose: np.ndarray):
        """Execute approach command"""
        object_name = command.parameters.get('object', 'object')

        # Find object in environment
        target_objects = self.environment_map.get_all_objects()
        if object_name in target_objects:
            object_pose = target_objects[object_name]

            # Calculate approach pose
            approach_pose = self._calculate_approach_pose(object_pose)
            self.robot_controller.move_to_pose(approach_pose)
        else:
            # Search for object
            self.robot_controller.search_for_object(object_name)

    def _execute_avoid_command(self, command: MotionCommand, current_pose: np.ndarray):
        """Execute avoid command"""
        obstacle_name = command.parameters.get('obstacle', 'obstacle')

        # Find obstacle in environment
        target_objects = self.environment_map.get_all_objects()
        if obstacle_name in target_objects:
            obstacle_pose = target_objects[obstacle_name]

            # Calculate path that avoids obstacle
            path_planner = PathPlanner()
            # This would implement obstacle avoidance planning
            # For now, we'll just move around randomly
            avoidance_pose = current_pose.copy()
            avoidance_pose[0] += 1.0  # Move right to avoid
            self.robot_controller.move_to_pose(avoidance_pose)

    def _calculate_approach_pose(self, object_pose: np.ndarray) -> np.ndarray:
        """Calculate approach pose for an object"""
        approach_pose = object_pose.copy()
        # Move back from object to avoid collision during approach
        approach_pose[0] -= 0.3  # 30cm from object
        return approach_pose

    def cancel_execution(self):
        """Cancel current plan execution"""
        self.is_executing = False
        self.robot_controller.stop_motion()

    def get_execution_status(self) -> Dict:
        """Get current execution status"""
        return {
            "is_executing": self.is_executing,
            "current_step": self.current_step,
            "total_steps": len(self.current_plan),
            "plan": self.current_plan
        }

class MockRobotController:
    """Mock robot controller for demonstration"""

    def __init__(self):
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw

    def get_current_pose(self) -> np.ndarray:
        """Get current robot pose"""
        return self.current_pose

    def follow_path(self, path: List[np.ndarray]):
        """Follow a path of poses"""
        for pose in path:
            self.move_to_pose(pose)
            time.sleep(0.1)  # Simulate movement time

    def move_to_pose(self, pose: np.ndarray):
        """Move to a specific pose"""
        print(f"Moving to pose: {pose[:3]}")
        self.current_pose = pose

    def grasp_object_at_pose(self, pose: np.ndarray, object_name: str):
        """Grasp an object at a specific pose"""
        print(f"Grasping {object_name} at {pose[:3]}")
        time.sleep(1)  # Simulate grasp time

    def place_object_at_pose(self, object_name: str, pose: np.ndarray):
        """Place an object at a specific pose"""
        print(f"Placing {object_name} at {pose[:3]}")
        time.sleep(1)  # Simulate placement time

    def rotate_by_angle(self, angle: float):
        """Rotate by a specific angle"""
        print(f"Rotating by {angle} degrees")
        time.sleep(0.5)

    def search_for_object(self, object_name: str):
        """Search for an object"""
        print(f"Searching for {object_name}")
        time.sleep(2)

    def stop_motion(self):
        """Stop current motion"""
        print("Stopping motion")

class MockEnvironmentMap:
    """Mock environment map for demonstration"""

    def __init__(self):
        self.objects = {
            'table': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'chair': np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            'book': np.array([1.1, 0.1, 0.8, 0.0, 0.0, 0.0])
        }
        self.locations = {
            'kitchen': np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'bedroom': np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        }

    def get_object_pose(self, object_name: str) -> Optional[np.ndarray]:
        """Get pose of an object"""
        return self.objects.get(object_name)

    def get_location_pose(self, location_name: str) -> Optional[np.ndarray]:
        """Get pose of a location"""
        return self.locations.get(location_name)

    def get_all_objects(self) -> Dict[str, np.ndarray]:
        """Get all objects in environment"""
        return self.objects

class SafetyMonitor:
    """Monitor safety for motion plans"""

    def __init__(self):
        self.safety_thresholds = {
            'min_distance_to_obstacle': 0.3,  # meters
            'max_velocity': 1.0,  # m/s
            'max_acceleration': 2.0  # m/s^2
        }

    def validate_plan(self, plan: List[MotionCommand], environment_map) -> bool:
        """Validate a motion plan for safety"""
        # Check each command in the plan
        for command in plan:
            if not self._validate_command(command, environment_map):
                return False
        return True

    def _validate_command(self, command: MotionCommand, environment_map) -> bool:
        """Validate a single command"""
        # For navigation commands, check if path is collision-free
        if command.action_type in [ActionType.NAVIGATE.value, ActionType.MOVE.value, ActionType.APPROACH.value]:
            # This would check for collisions in the planned path
            # For now, we'll assume it's safe
            return True

        # For grasp commands, check if object is accessible
        if command.action_type == ActionType.GRASP.value:
            object_name = command.parameters.get('object')
            if object_name:
                object_pose = environment_map.get_object_pose(object_name)
                if object_pose is not None:
                    # Check if object is within reachable workspace
                    # For now, assume it's reachable
                    return True

        return True
```

## Handling Ambiguity and Uncertainty

Real-world language commands often contain ambiguity that must be resolved:

```python
class AmbiguityResolver:
    """Resolve ambiguity in language commands"""

    def __init__(self):
        self.context_resolver = ContextResolver()
        self.disambiguation_strategies = [
            self._ask_for_clarification,
            self._use_default_interpretation,
            self._infer_from_environment
        ]

    def resolve_ambiguity(self, command: MotionCommand, context: Dict,
                         environment_map) -> MotionCommand:
        """Resolve ambiguity in a command"""
        resolved_command = command

        # Check for ambiguous spatial references
        if self._has_ambiguous_reference(command, environment_map):
            for strategy in self.disambiguation_strategies:
                resolved_command = strategy(command, context, environment_map)
                if resolved_command.spatial_reference:
                    break

        # Check for ambiguous object references
        if self._has_ambiguous_object(command, environment_map):
            for strategy in self.disambiguation_strategies:
                resolved_command = strategy(command, context, environment_map)
                if 'object' in resolved_command.parameters:
                    break

        return resolved_command

    def _has_ambiguous_reference(self, command: MotionCommand, environment_map) -> bool:
        """Check if command has ambiguous spatial reference"""
        if command.spatial_reference:
            # Check if there are multiple objects with the same name
            all_objects = environment_map.get_all_objects()
            matches = [name for name in all_objects.keys()
                      if command.spatial_reference.lower() in name.lower()]
            return len(matches) > 1
        return False

    def _has_ambiguous_object(self, command: MotionCommand, environment_map) -> bool:
        """Check if command has ambiguous object reference"""
        obj_name = command.parameters.get('object')
        if obj_name:
            all_objects = environment_map.get_all_objects()
            matches = [name for name in all_objects.keys()
                      if obj_name.lower() in name.lower()]
            return len(matches) > 1
        return False

    def _ask_for_clarification(self, command: MotionCommand, context: Dict,
                              environment_map) -> MotionCommand:
        """Ask for clarification of ambiguous command"""
        # This would involve human interaction
        # For simulation, we'll use context to disambiguate
        if command.spatial_reference:
            # Use context to find the most likely referent
            likely_object = self.context_resolver.find_likely_referent(
                command.spatial_reference, context, environment_map
            )
            if likely_object:
                command.spatial_reference = likely_object

        return command

    def _use_default_interpretation(self, command: MotionCommand, context: Dict,
                                   environment_map) -> MotionCommand:
        """Use default interpretation for ambiguous command"""
        # For spatial references, choose the closest object
        if command.spatial_reference:
            closest_object = self._find_closest_object(
                command.spatial_reference, context, environment_map
            )
            if closest_object:
                command.spatial_reference = closest_object

        return command

    def _infer_from_environment(self, command: MotionCommand, context: Dict,
                               environment_map) -> MotionCommand:
        """Infer meaning from environmental context"""
        # Use environmental cues to disambiguate
        # This might involve looking for objects recently mentioned or visible
        return command

    def _find_closest_object(self, reference: str, context: Dict,
                            environment_map) -> Optional[str]:
        """Find the closest object matching the reference"""
        robot_pose = context.get('robot_pose', np.array([0, 0, 0]))
        all_objects = environment_map.get_all_objects()

        closest_distance = float('inf')
        closest_object = None

        for obj_name, obj_pose in all_objects.items():
            if reference.lower() in obj_name.lower():
                dist = np.linalg.norm(robot_pose[:3] - obj_pose[:3])
                if dist < closest_distance:
                    closest_distance = dist
                    closest_object = obj_name

        return closest_object

class ContextResolver:
    """Resolve references using context"""

    def __init__(self):
        self.conversation_history = deque(maxlen=10)
        self.object_attention_history = deque(maxlen=20)

    def find_likely_referent(self, reference: str, context: Dict,
                            environment_map) -> Optional[str]:
        """Find the most likely object referent for a spatial reference"""
        # Check conversation history for recently mentioned objects
        for item in reversed(self.conversation_history):
            if reference.lower() in item.lower():
                # Check if this object exists in the environment
                all_objects = environment_map.get_all_objects()
                for obj_name in all_objects.keys():
                    if reference.lower() in obj_name.lower():
                        return obj_name

        # Check object attention history
        for obj_name in reversed(self.object_attention_history):
            if reference.lower() in obj_name.lower():
                return obj_name

        # If no context match, return the first object that matches
        all_objects = environment_map.get_all_objects()
        for obj_name in all_objects.keys():
            if reference.lower() in obj_name.lower():
                return obj_name

        return None

    def update_context(self, command: MotionCommand, result: Dict):
        """Update context based on command and result"""
        if command.spatial_reference:
            self.conversation_history.append(command.spatial_reference)
        if 'object' in command.parameters:
            self.conversation_history.append(command.parameters['object'])
            self.object_attention_history.append(command.parameters['object'])

# Uncertainty-aware motion planning
class UncertaintyAwarePlanner:
    """Motion planner that accounts for uncertainty"""

    def __init__(self):
        self.uncertainty_threshold = 0.7  # Minimum confidence threshold

    def plan_with_uncertainty(self, command: MotionCommand,
                             environment_map,
                             uncertainty_map: Dict[str, float]) -> List[np.ndarray]:
        """Plan motion considering uncertainty in environment"""
        # Adjust planning based on uncertainty
        if command.spatial_reference and command.spatial_reference in uncertainty_map:
            uncertainty = uncertainty_map[command.spatial_reference]

            if uncertainty > self.uncertainty_threshold:
                # High uncertainty - use conservative planning
                return self._conservative_plan(command, environment_map)
            else:
                # Low uncertainty - use normal planning
                planner = PathPlanner()
                start_pose = np.array([0, 0, 0, 0, 0, 0])  # Current robot pose
                target_pose = environment_map.get_location_pose(command.spatial_reference)
                if target_pose is not None:
                    return planner.plan_path(start_pose, target_pose, environment_map)

        # Default planning
        planner = PathPlanner()
        start_pose = np.array([0, 0, 0, 0, 0, 0])
        target_pose = np.array([1, 1, 0, 0, 0, 0])  # Default target
        return planner.plan_path(start_pose, target_pose, environment_map)

    def _conservative_plan(self, command: MotionCommand,
                          environment_map) -> List[np.ndarray]:
        """Create a conservative motion plan for uncertain situations"""
        # For uncertain situations, plan a safer path
        # This might involve going to a known safe location first
        # or using more cautious movement parameters

        # For now, return a simple path to a safe location
        safe_location = np.array([0, 0, 0, 0, 0, 0])  # Home position
        start_pose = np.array([0, 0, 0, 0, 0, 0])

        planner = PathPlanner()
        return planner.plan_path(start_pose, safe_location, environment_map)

# Example of the complete system in action
def example_language_guided_motion():
    """Example of language-guided motion planning system"""

    # Create mock components
    robot_controller = MockRobotController()
    environment_map = MockEnvironmentMap()

    # Create the main system
    system = LanguageGuidedMotionSystem(robot_controller, environment_map)

    # Example commands
    commands = [
        "Navigate to the table",
        "Grasp the book",
        "Place the book in the kitchen"
    ]

    # Execute each command
    for cmd in commands:
        print(f"\nExecuting command: '{cmd}'")
        result = system.execute_language_command(cmd)
        print(f"Result: {result['status']}")

        if result['status'] == 'error':
            print(f"Error: {result['error']}")

    # Show execution status
    status = system.get_execution_status()
    print(f"\nExecution status: {status}")

    return system

if __name__ == "__main__":
    system = example_language_guided_motion()
    print("\nLanguage-guided motion example completed")
```

## Evaluation and Optimization

Evaluating and optimizing language-guided motion planning systems:

```python
class MotionPlanningEvaluator:
    """Evaluator for language-guided motion planning"""

    def __init__(self):
        self.metrics = {
            'success_rate': 0.0,
            'planning_time': 0.0,
            'execution_time': 0.0,
            'path_efficiency': 0.0,
            'language_understanding_accuracy': 0.0
        }

    def evaluate_system(self, system, test_commands, expected_outcomes):
        """Evaluate the motion planning system"""
        total_commands = len(test_commands)
        successful_executions = 0
        total_planning_time = 0.0
        total_execution_time = 0.0
        total_path_efficiency = 0.0

        for cmd, expected in zip(test_commands, expected_outcomes):
            start_time = time.time()

            # Execute command
            result = system.execute_language_command(cmd)

            planning_time = time.time() - start_time
            total_planning_time += planning_time

            if result['status'] == 'success':
                successful_executions += 1

                # Measure execution time (would need to track this in real system)
                execution_time = 5.0  # Placeholder
                total_execution_time += execution_time

                # Measure path efficiency (ratio of actual path to optimal path)
                path_efficiency = 0.85  # Placeholder
                total_path_efficiency += path_efficiency

        # Calculate metrics
        self.metrics['success_rate'] = successful_executions / total_commands if total_commands > 0 else 0.0
        self.metrics['planning_time'] = total_planning_time / total_commands if total_commands > 0 else 0.0
        self.metrics['execution_time'] = total_execution_time / successful_executions if successful_executions > 0 else 0.0
        self.metrics['path_efficiency'] = total_path_efficiency / successful_executions if successful_executions > 0 else 0.0

        return self.metrics

    def generate_report(self):
        """Generate evaluation report"""
        report = f"""
Language-Guided Motion Planning Evaluation Report:
==================================================
- Success Rate: {self.metrics['success_rate']:.2%}
- Average Planning Time: {self.metrics['planning_time']:.3f}s
- Average Execution Time: {self.metrics['execution_time']:.3f}s
- Path Efficiency: {self.metrics['path_efficiency']:.2%}

Performance Rating: {'Excellent' if self.metrics['success_rate'] > 0.9 else 'Good' if self.metrics['success_rate'] > 0.7 else 'Needs Improvement'}
        """
        return report

# Optimization techniques
class MotionPlannerOptimizer:
    """Optimizer for language-guided motion planning"""

    def __init__(self):
        self.optimization_strategies = [
            self._optimize_parsing_speed,
            self._optimize_path_planning,
            self._optimize_execution
        ]

    def optimize_system(self, planner):
        """Apply optimization strategies to the planner"""
        for strategy in self.optimization_strategies:
            strategy(planner)

    def _optimize_parsing_speed(self, planner):
        """Optimize parsing speed"""
        # Use cached parsing results for common commands
        # Implement more efficient parsing algorithms
        pass

    def _optimize_path_planning(self, planner):
        """Optimize path planning"""
        # Use hierarchical path planning
        # Implement path smoothing
        # Use dynamic replanning for changing environments
        pass

    def _optimize_execution(self, planner):
        """Optimize execution"""
        # Parallel execution of independent subtasks
        # Predictive execution based on context
        # Adaptive execution speed based on confidence
        pass

# Benchmarking for language-guided motion planning
class MotionPlanningBenchmark:
    """Benchmark for language-guided motion planning"""

    def __init__(self):
        self.benchmarks = {
            'simple_navigation': self._benchmark_simple_navigation,
            'object_manipulation': self._benchmark_object_manipulation,
            'complex_task_execution': self._benchmark_complex_task_execution,
            'ambiguity_resolution': self._benchmark_ambiguity_resolution
        }

    def run_benchmark(self, system, benchmark_name):
        """Run a specific benchmark"""
        if benchmark_name in self.benchmarks:
            return self.benchmarks[benchmark_name](system)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def _benchmark_simple_navigation(self, system):
        """Benchmark simple navigation tasks"""
        commands = [
            "Go to the table",
            "Move forward 2 meters",
            "Turn left 90 degrees"
        ]

        evaluator = MotionPlanningEvaluator()
        return evaluator.evaluate_system(system, commands, [None]*len(commands))

    def _benchmark_object_manipulation(self, system):
        """Benchmark object manipulation tasks"""
        commands = [
            "Grasp the book",
            "Place the book on the table",
            "Pick up the cup and place it in the kitchen"
        ]

        evaluator = MotionPlanningEvaluator()
        return evaluator.evaluate_system(system, commands, [None]*len(commands))

    def _benchmark_complex_task_execution(self, system):
        """Benchmark complex multi-step tasks"""
        commands = [
            "Go to the kitchen, pick up the apple, and bring it to the dining table",
            "Navigate to the bedroom, find the shoes, and place them in the closet"
        ]

        evaluator = MotionPlanningEvaluator()
        return evaluator.evaluate_system(system, commands, [None]*len(commands))

    def _benchmark_ambiguity_resolution(self, system):
        """Benchmark ambiguity resolution"""
        commands = [
            "Go to the table"  # Ambiguous if multiple tables exist
        ]

        evaluator = MotionPlanningEvaluator()
        return evaluator.evaluate_system(system, commands, [None]*len(commands))
```

## Chapter Summary

In this chapter, we explored motion planning from language, covering:

- Semantic parsing techniques for converting natural language to structured commands
- Spatial reasoning and geometric mapping for understanding spatial relationships
- Hierarchical motion planning systems that decompose complex tasks
- Integration with robotic systems for end-to-end execution
- Techniques for handling ambiguity and uncertainty in language commands
- Evaluation and optimization strategies for language-guided motion planning

Language-guided motion planning enables intuitive human-robot interaction by allowing users to specify tasks using natural language. The challenge lies in accurately interpreting linguistic commands and translating them into executable robot actions while handling ambiguity and uncertainty.

## Next Steps

In the next chapter, we'll explore human-robot communication systems, examining how robots can engage in natural conversations and provide feedback to users during task execution.

## Exercises

1. **Implementation Challenge**: Implement a semantic parser that converts natural language commands to motion plans for a specific robotic platform.

2. **System Design**: Design a hierarchical motion planning system that can handle complex multi-step tasks specified in natural language.

3. **Ambiguity Resolution**: Implement an ambiguity resolution system that can handle uncertain or ambiguous language commands in a real environment.

4. **Integration Task**: Integrate a language understanding module with a robot's navigation system to enable natural language navigation commands.

5. **Evaluation Challenge**: Create a benchmark for evaluating language-guided motion planning systems across different complexity levels and environmental conditions.