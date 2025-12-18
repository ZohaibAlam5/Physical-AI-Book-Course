---
sidebar_position: 3
title: Chapter 3 - Natural Language to Task Plans
---

# Chapter 3 - Natural Language to Task Plans

In this chapter, we explore how natural language commands are transformed into structured task plans that humanoid robots can execute. This transformation is a critical component of the Vision-Language-Action paradigm, bridging the gap between human communication and robotic action execution. We'll examine the challenges, techniques, and implementation strategies for creating effective natural language to task planning systems.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the process of converting natural language commands to structured task plans
- Design hierarchical task planning systems for humanoid robots
- Implement semantic parsing for natural language commands
- Create task decomposition and sequencing algorithms
- Evaluate the effectiveness of natural language to task planning systems
- Integrate task planning with other VLA components

## Introduction to Natural Language to Task Planning

Natural language to task planning is the process of transforming human language commands into executable robotic tasks. This involves understanding the intent behind the language, decomposing complex tasks into simpler subtasks, and generating a sequence of actions that the robot can execute to fulfill the command.

The challenge lies in the ambiguity and flexibility of natural language compared to the structured requirements of robotic task execution. Humans use language that is often context-dependent, ambiguous, and high-level, while robots require precise, structured instructions.

### The Transformation Pipeline

The transformation from natural language to task plans typically involves several stages:

1. **Semantic Parsing**: Understanding the meaning and structure of the command
2. **Intent Recognition**: Identifying the user's goal or intention
3. **Task Decomposition**: Breaking complex tasks into manageable subtasks
4. **Action Sequencing**: Ordering actions in a logical sequence
5. **Constraint Checking**: Ensuring the plan is feasible given robot capabilities
6. **Plan Refinement**: Optimizing the plan for efficiency and safety

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re

class TaskType(Enum):
    """Types of tasks that can be generated from natural language"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    INFORMATION = "information"
    SYSTEM = "system"
    COMPOSITE = "composite"

class TaskStatus(Enum):
    """Status of a task in the planning/execution pipeline"""
    PENDING = "pending"
    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class NLCommand:
    """Structure for representing a natural language command"""
    text: str
    confidence: float
    timestamp: float
    source: str = "voice"  # voice, text, etc.

@dataclass
class TaskPlan:
    """Structure for representing a structured task plan"""
    task_id: str
    task_type: TaskType
    description: str
    subtasks: List['TaskPlan']
    parameters: Dict[str, Any]
    priority: int
    estimated_duration: float
    dependencies: List[str]  # IDs of tasks that must complete first
    status: TaskStatus = TaskStatus.PENDING

@dataclass
class Action:
    """Structure for representing a low-level robotic action"""
    action_type: str
    parameters: Dict[str, Any]
    duration: float
    preconditions: List[str]
    effects: List[str]

class SemanticParser(nn.Module):
    """Parses natural language commands into semantic structures"""

    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embedding for word representations
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # LSTM for sequence processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Linear layers for different semantic components
        self.intent_classifier = nn.Linear(hidden_dim, 6)  # 6 task types
        self.param_extractor = nn.Linear(hidden_dim, 128)  # Parameter extraction

        # Attention mechanism for important word identification
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def forward(self, command: str) -> Dict[str, Any]:
        """Parse a natural language command"""
        # Tokenize the command
        tokens = self._tokenize(command)

        # Convert to embeddings
        embedded = self.embedding(tokens)

        # Process with LSTM
        lstm_out, _ = self.lstm(embedded.unsqueeze(0))

        # Apply attention
        attended, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )

        # Classify intent
        intent_logits = self.intent_classifier(attended.mean(dim=0))
        intent_probs = torch.softmax(intent_logits, dim=-1)

        # Extract parameters
        params = self.param_extractor(attended.mean(dim=0))

        # Return semantic structure
        return {
            'intent': torch.argmax(intent_probs).item(),
            'confidence': torch.max(intent_probs).item(),
            'parameters': params.detach().numpy(),
            'important_tokens': self._identify_important_tokens(attended, tokens)
        }

    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization for demonstration"""
        # In practice, this would use a proper tokenizer
        # For now, we'll use a simple approach with fixed vocabulary
        words = text.lower().split()
        # Convert to token IDs (simplified)
        token_ids = []
        for word in words:
            # This is a simplified mapping - in practice, you'd use a vocabulary
            token_ids.append(hash(word) % 10000)  # Simplified vocabulary mapping
        return torch.tensor(token_ids)

    def _identify_important_tokens(self, attention_weights: torch.Tensor, tokens: torch.Tensor) -> List[int]:
        """Identify the most important tokens based on attention weights"""
        # Calculate importance scores
        importance = torch.mean(attention_weights, dim=1)
        _, top_indices = torch.topk(importance.squeeze(), min(5, len(tokens)))
        return top_indices.tolist()

class IntentRecognizer:
    """Recognizes user intent from parsed semantic structures"""

    def __init__(self):
        self.intent_keywords = {
            TaskType.NAVIGATION: ['go', 'move', 'walk', 'navigate', 'step', 'go to', 'move to'],
            TaskType.MANIPULATION: ['pick', 'grasp', 'take', 'hold', 'put', 'place', 'grab', 'lift'],
            TaskType.INTERACTION: ['talk', 'speak', 'greet', 'hello', 'hi', 'interact', 'communicate'],
            TaskType.INFORMATION: ['what', 'where', 'when', 'how', 'tell', 'describe', 'show'],
            TaskType.SYSTEM: ['stop', 'pause', 'resume', 'start', 'end', 'system', 'robot']
        }

    def recognize_intent(self, parsed_command: Dict[str, Any], original_text: str) -> Tuple[TaskType, float]:
        """Recognize intent from parsed command and original text"""
        # Check for keywords in the original text
        text_lower = original_text.lower()

        best_intent = TaskType.INFORMATION  # Default
        best_confidence = 0.0

        for task_type, keywords in self.intent_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_matches > 0:
                confidence = min(1.0, keyword_matches / len(keywords))
                if confidence > best_confidence:
                    best_intent = task_type
                    best_confidence = confidence

        # Use the neural network's prediction as well
        nn_confidence = parsed_command.get('confidence', 0.0)
        nn_intent_idx = parsed_command.get('intent', 0)

        # Map neural network output to task types
        nn_intent_map = [
            TaskType.NAVIGATION, TaskType.MANIPULATION, TaskType.INTERACTION,
            TaskType.INFORMATION, TaskType.SYSTEM, TaskType.COMPOSITE
        ]

        if nn_intent_idx < len(nn_intent_map):
            nn_intent = nn_intent_map[nn_intent_idx]
            # Combine keyword and neural network confidence
            combined_confidence = max(best_confidence, nn_confidence)
            combined_intent = nn_intent if nn_confidence > best_confidence else best_intent
        else:
            combined_confidence = best_confidence
            combined_intent = best_intent

        return combined_intent, combined_confidence

class TaskDecomposer:
    """Decomposes complex tasks into subtasks"""

    def __init__(self):
        self.decomposition_rules = {
            TaskType.NAVIGATION: self._decompose_navigation,
            TaskType.MANIPULATION: self._decompose_manipulation,
            TaskType.INTERACTION: self._decompose_interaction,
            TaskType.INFORMATION: self._decompose_information,
            TaskType.SYSTEM: self._decompose_system,
            TaskType.COMPOSITE: self._decompose_composite
        }

    def decompose_task(self, command: NLCommand, intent: TaskType, params: Dict[str, Any]) -> TaskPlan:
        """Decompose a command into a task plan"""
        if intent in self.decomposition_rules:
            return self.decomposition_rules[intent](command, params)
        else:
            # Default decomposition for unknown task types
            return TaskPlan(
                task_id=f"task_{hash(command.text)}",
                task_type=TaskType.INFORMATION,
                description=command.text,
                subtasks=[],
                parameters=params,
                priority=3,
                estimated_duration=1.0,
                dependencies=[]
            )

    def _decompose_navigation(self, command: NLCommand, params: Dict[str, Any]) -> TaskPlan:
        """Decompose navigation tasks"""
        # Extract destination from parameters
        destination = params.get('destination', 'unknown location')

        # Create subtasks for navigation
        subtasks = [
            TaskPlan(
                task_id=f"nav_scan_{hash(destination)}",
                task_type=TaskType.INFORMATION,
                description=f"Scan environment for path to {destination}",
                subtasks=[],
                parameters={'destination': destination},
                priority=5,
                estimated_duration=2.0,
                dependencies=[]
            ),
            TaskPlan(
                task_id=f"nav_plan_{hash(destination)}",
                task_type=TaskType.NAVIGATION,
                description=f"Plan path to {destination}",
                subtasks=[],
                parameters={'destination': destination},
                priority=5,
                estimated_duration=1.0,
                dependencies=[f"nav_scan_{hash(destination)}"]
            ),
            TaskPlan(
                task_id=f"nav_execute_{hash(destination)}",
                task_type=TaskType.NAVIGATION,
                description=f"Navigate to {destination}",
                subtasks=[],
                parameters={'destination': destination},
                priority=5,
                estimated_duration=10.0,  # This would be calculated based on distance
                dependencies=[f"nav_plan_{hash(destination)}"]
            )
        ]

        return TaskPlan(
            task_id=f"nav_main_{hash(destination)}",
            task_type=TaskType.NAVIGATION,
            description=f"Navigate to {destination}",
            subtasks=subtasks,
            parameters=params,
            priority=5,
            estimated_duration=13.0,
            dependencies=[]
        )

    def _decompose_manipulation(self, command: NLCommand, params: Dict[str, Any]) -> TaskPlan:
        """Decompose manipulation tasks"""
        # Extract object from parameters
        obj = params.get('object', 'unknown object')

        # Create subtasks for manipulation
        subtasks = [
            TaskPlan(
                task_id=f"manip_locate_{hash(obj)}",
                task_type=TaskType.INFORMATION,
                description=f"Locate {obj}",
                subtasks=[],
                parameters={'object': obj},
                priority=6,
                estimated_duration=3.0,
                dependencies=[]
            ),
            TaskPlan(
                task_id=f"manip_approach_{hash(obj)}",
                task_type=TaskType.NAVIGATION,
                description=f"Approach {obj}",
                subtasks=[],
                parameters={'object': obj},
                priority=6,
                estimated_duration=5.0,
                dependencies=[f"manip_locate_{hash(obj)}"]
            ),
            TaskPlan(
                task_id=f"manip_grasp_{hash(obj)}",
                task_type=TaskType.MANIPULATION,
                description=f"Grasp {obj}",
                subtasks=[],
                parameters={'object': obj},
                priority=6,
                estimated_duration=2.0,
                dependencies=[f"manip_approach_{hash(obj)}"]
            )
        ]

        return TaskPlan(
            task_id=f"manip_main_{hash(obj)}",
            task_type=TaskType.MANIPULATION,
            description=f"Manipulate {obj}",
            subtasks=subtasks,
            parameters=params,
            priority=6,
            estimated_duration=10.0,
            dependencies=[]
        )

    def _decompose_interaction(self, command: NLCommand, params: Dict[str, Any]) -> TaskPlan:
        """Decompose interaction tasks"""
        return TaskPlan(
            task_id=f"interact_{hash(command.text)}",
            task_type=TaskType.INTERACTION,
            description=command.text,
            subtasks=[],
            parameters=params,
            priority=2,
            estimated_duration=5.0,
            dependencies=[]
        )

    def _decompose_information(self, command: NLCommand, params: Dict[str, Any]) -> TaskPlan:
        """Decompose information tasks"""
        return TaskPlan(
            task_id=f"info_{hash(command.text)}",
            task_type=TaskType.INFORMATION,
            description=command.text,
            subtasks=[],
            parameters=params,
            priority=1,
            estimated_duration=2.0,
            dependencies=[]
        )

    def _decompose_system(self, command: NLCommand, params: Dict[str, Any]) -> TaskPlan:
        """Decompose system tasks"""
        return TaskPlan(
            task_id=f"system_{hash(command.text)}",
            task_type=TaskType.SYSTEM,
            description=command.text,
            subtasks=[],
            parameters=params,
            priority=8,
            estimated_duration=1.0,
            dependencies=[]
        )

    def _decompose_composite(self, command: NLCommand, params: Dict[str, Any]) -> TaskPlan:
        """Decompose composite tasks"""
        # For composite tasks, we'd need more complex logic
        # This is a simplified example
        return TaskPlan(
            task_id=f"composite_{hash(command.text)}",
            task_type=TaskType.COMPOSITE,
            description=command.text,
            subtasks=[],
            parameters=params,
            priority=4,
            estimated_duration=5.0,
            dependencies=[]
        )

class TaskPlanner:
    """Main component for converting natural language to task plans"""

    def __init__(self):
        self.semantic_parser = SemanticParser()
        self.intent_recognizer = IntentRecognizer()
        self.task_decomposer = TaskDecomposer()

    def plan_from_command(self, command: NLCommand) -> Optional[TaskPlan]:
        """Convert a natural language command to a task plan"""
        # Step 1: Parse the command semantically
        parsed = self.semantic_parser(command.text)

        # Step 2: Recognize the intent
        intent, confidence = self.intent_recognizer.recognize_intent(parsed, command.text)

        # Step 3: Check confidence threshold
        if confidence < 0.5:  # Threshold for acceptable confidence
            print(f"Command '{command.text}' has low confidence ({confidence:.2f}), skipping")
            return None

        # Step 4: Decompose into task plan
        task_plan = self.task_decomposer.decompose_task(command, intent, parsed)

        # Step 5: Set confidence in the task plan
        task_plan.parameters['command_confidence'] = confidence

        return task_plan

# Example usage
def example_task_planning():
    """Example of how to use the task planning system"""
    # Initialize the planner
    planner = TaskPlanner()

    # Example commands to process
    commands = [
        NLCommand("Please go to the kitchen", 0.9, 0.0),
        NLCommand("Pick up the red cup", 0.85, 1.0),
        NLCommand("Tell me the time", 0.95, 2.0),
        NLCommand("Stop all activities", 0.98, 3.0)
    ]

    print("Natural Language to Task Planning Example")
    print("=" * 50)

    for cmd in commands:
        print(f"\nProcessing command: '{cmd.text}' (confidence: {cmd.confidence:.2f})")

        # Plan the task
        task_plan = planner.plan_from_command(cmd)

        if task_plan:
            print(f"  Task Type: {task_plan.task_type.value}")
            print(f"  Description: {task_plan.description}")
            print(f"  Estimated Duration: {task_plan.estimated_duration}s")
            print(f"  Priority: {task_plan.priority}")
            print(f"  Subtasks: {len(task_plan.subtasks)}")

            for i, subtask in enumerate(task_plan.subtasks):
                print(f"    {i+1}. {subtask.description} ({subtask.task_type.value})")
        else:
            print("  Could not generate task plan - confidence too low or parsing failed")

if __name__ == "__main__":
    example_task_planning()
```

## Hierarchical Task Planning

Hierarchical task planning is essential for managing complex tasks that require multiple levels of abstraction. This approach breaks down high-level goals into manageable subtasks, creating a tree-like structure of tasks and subtasks.

### Task Hierarchy Structure

The hierarchical structure typically includes:

1. **High-Level Goals**: The overall objective (e.g., "Prepare dinner")
2. **Intermediate Tasks**: Major steps toward the goal (e.g., "Gather ingredients", "Cook meal")
3. **Primitive Actions**: Basic actions the robot can execute (e.g., "Move to counter", "Grasp knife")

```python
class HierarchicalTaskPlanner:
    """Implements hierarchical task planning for complex commands"""

    def __init__(self):
        self.task_library = {
            'prepare_dinner': {
                'description': 'Prepare a complete dinner meal',
                'subtasks': [
                    'gather_ingredients',
                    'prepare_ingredients',
                    'cook_main_dish',
                    'set_table',
                    'serve_meal'
                ]
            },
            'clean_room': {
                'description': 'Clean a room',
                'subtasks': [
                    'remove_trash',
                    'dust_surfaces',
                    'vacuum_floor',
                    'organize_items'
                ]
            }
        }

    def create_hierarchical_plan(self, goal: str) -> Optional[TaskPlan]:
        """Create a hierarchical plan for a high-level goal"""
        if goal not in self.task_library:
            return None

        goal_info = self.task_library[goal]
        subtasks = []

        for subtask_name in goal_info['subtasks']:
            # Create subtask based on the subtask name
            subtask = self._create_subtask(subtask_name)
            if subtask:
                subtasks.append(subtask)

        return TaskPlan(
            task_id=f"goal_{hash(goal)}",
            task_type=TaskType.COMPOSITE,
            description=goal_info['description'],
            subtasks=subtasks,
            parameters={'goal': goal},
            priority=7,
            estimated_duration=sum(st.estimated_duration for st in subtasks) if subtasks else 10.0,
            dependencies=[]
        )

    def _create_subtask(self, subtask_name: str) -> TaskPlan:
        """Create a specific subtask based on its name"""
        # Map subtask names to specific task plans
        subtask_mapping = {
            'gather_ingredients': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_1",
                task_type=TaskType.NAVIGATION,
                description=f"Gather ingredients for meal",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=6,
                estimated_duration=5.0,
                dependencies=[]
            ),
            'prepare_ingredients': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_2",
                task_type=TaskType.MANIPULATION,
                description=f"Prepare ingredients for cooking",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=6,
                estimated_duration=8.0,
                dependencies=[]
            ),
            'cook_main_dish': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_3",
                task_type=TaskType.MANIPULATION,
                description=f"Cook the main dish",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=7,
                estimated_duration=15.0,
                dependencies=[]
            ),
            'set_table': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_4",
                task_type=TaskType.MANIPULATION,
                description=f"Set the dining table",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=5,
                estimated_duration=5.0,
                dependencies=[]
            ),
            'serve_meal': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_5",
                task_type=TaskType.MANIPULATION,
                description=f"Serve the prepared meal",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=6,
                estimated_duration=3.0,
                dependencies=[]
            ),
            'remove_trash': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_1",
                task_type=TaskType.MANIPULATION,
                description=f"Remove trash from room",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=5,
                estimated_duration=3.0,
                dependencies=[]
            ),
            'dust_surfaces': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_2",
                task_type=TaskType.MANIPULATION,
                description=f"Dust surfaces in room",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=4,
                estimated_duration=10.0,
                dependencies=[]
            ),
            'vacuum_floor': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_3",
                task_type=TaskType.NAVIGATION,
                description=f"Vacuum the floor",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=5,
                estimated_duration=15.0,
                dependencies=[]
            ),
            'organize_items': TaskPlan(
                task_id=f"subtask_{hash(subtask_name)}_4",
                task_type=TaskType.MANIPULATION,
                description=f"Organize items in room",
                subtasks=[],
                parameters={'task': subtask_name},
                priority=4,
                estimated_duration=12.0,
                dependencies=[]
            )
        }

        return subtask_mapping.get(subtask_name, TaskPlan(
            task_id=f"subtask_{hash(subtask_name)}",
            task_type=TaskType.INFORMATION,
            description=f"Perform {subtask_name}",
            subtasks=[],
            parameters={'task': subtask_name},
            priority=3,
            estimated_duration=5.0,
            dependencies=[]
        ))

    def refine_plan(self, plan: TaskPlan, context: Dict[str, Any]) -> TaskPlan:
        """Refine a plan based on current context and constraints"""
        # Update estimated durations based on context
        if 'robot_speed' in context:
            speed_factor = context['robot_speed']
            plan.estimated_duration *= speed_factor
            for subtask in plan.subtasks:
                subtask.estimated_duration *= speed_factor

        # Adjust priorities based on context
        if 'urgency' in context and context['urgency'] == 'high':
            plan.priority = min(10, plan.priority + 2)

        return plan
```

## Semantic Parsing and Understanding

Semantic parsing is the process of converting natural language into formal representations that can be processed by planning algorithms. This involves understanding not just the words, but their relationships and meanings in the context of robotic tasks.

### Context-Aware Parsing

Effective semantic parsing must consider the context in which commands are given, including:

- Current robot state and location
- Environmental conditions
- Previous interactions
- User preferences and history

```python
class ContextualSemanticParser:
    """Semantic parser that considers context for better understanding"""

    def __init__(self):
        self.context = {
            'robot_location': [0, 0, 0],
            'robot_orientation': [0, 0, 0, 1],
            'visible_objects': [],
            'recent_commands': [],
            'user_preferences': {}
        }

    def parse_with_context(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse command considering the provided context"""
        self.context.update(context)

        # Perform basic parsing
        parsed = self._basic_parse(command)

        # Apply contextual disambiguation
        disambiguated = self._apply_contextual_disambiguation(parsed, command)

        # Resolve references based on context
        resolved = self._resolve_contextual_references(disambiguated)

        return resolved

    def _basic_parse(self, command: str) -> Dict[str, Any]:
        """Perform basic semantic parsing"""
        # Extract entities and relationships
        entities = self._extract_entities(command)
        relationships = self._extract_relationships(command)

        return {
            'entities': entities,
            'relationships': relationships,
            'action': self._identify_action(command),
            'spatial_info': self._extract_spatial_info(command)
        }

    def _extract_entities(self, command: str) -> List[Dict[str, Any]]:
        """Extract named entities from the command"""
        entities = []

        # Simple pattern matching for objects
        object_patterns = [
            r'the (\w+) (cup|bottle|box|book|object)',
            r'(\w+) (cup|bottle|box|book|object)',
            r'(red|blue|green|yellow) (\w+)',
        ]

        for pattern in object_patterns:
            matches = re.findall(pattern, command, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        entities.append({
                            'type': 'object',
                            'name': f"{match[0]} {match[1]}",
                            'properties': {'color': match[0] if match[0].lower() in ['red', 'blue', 'green', 'yellow'] else None}
                        })
                    elif len(match) == 1:
                        entities.append({
                            'type': 'object',
                            'name': match[0],
                            'properties': {}
                        })

        # Extract locations
        location_patterns = [
            r'to the? (\w+)',
            r'at the? (\w+)',
            r'in the? (\w+)'
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, command, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'location',
                    'name': match,
                    'properties': {}
                })

        return entities

    def _extract_relationships(self, command: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []

        # Spatial relationships
        spatial_relations = ['on', 'in', 'under', 'next to', 'near', 'at', 'to']
        for relation in spatial_relations:
            if relation in command.lower():
                relationships.append({
                    'type': 'spatial',
                    'relation': relation,
                    'command': command
                })

        return relationships

    def _identify_action(self, command: str) -> str:
        """Identify the main action in the command"""
        action_keywords = {
            'navigation': ['go', 'move', 'walk', 'navigate', 'step'],
            'manipulation': ['pick', 'grasp', 'take', 'hold', 'put', 'place', 'grab', 'lift'],
            'interaction': ['talk', 'speak', 'greet', 'hello', 'hi'],
            'information': ['what', 'where', 'when', 'how', 'tell', 'describe', 'show']
        }

        for action_type, keywords in action_keywords.items():
            for keyword in keywords:
                if keyword in command.lower():
                    return action_type

        return 'unknown'

    def _extract_spatial_info(self, command: str) -> Dict[str, Any]:
        """Extract spatial information from the command"""
        spatial_info = {}

        # Directional information
        directions = ['north', 'south', 'east', 'west', 'left', 'right', 'forward', 'backward', 'up', 'down']
        for direction in directions:
            if direction in command.lower():
                spatial_info['direction'] = direction

        # Distance information
        distance_patterns = [r'(\d+(?:\.\d+)?) (meters?|feet|steps?)']
        for pattern in distance_patterns:
            matches = re.findall(pattern, command, re.IGNORECASE)
            for match in matches:
                spatial_info['distance'] = float(match[0])
                spatial_info['unit'] = match[1]

        return spatial_info

    def _apply_contextual_disambiguation(self, parsed: Dict[str, Any], command: str) -> Dict[str, Any]:
        """Apply contextual information to disambiguate the parse"""
        # If there are multiple objects mentioned, use context to determine which one
        if len(parsed['entities']) > 1:
            # Check if any objects are in the visible_objects context
            visible_objects = self.context.get('visible_objects', [])
            for entity in parsed['entities']:
                if entity['name'] in visible_objects:
                    entity['confidence'] = 0.9
                else:
                    entity['confidence'] = 0.5  # Lower confidence for non-visible objects

        return parsed

    def _resolve_contextual_references(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve contextual references like "it", "there", "here" """
        # Replace demonstrative pronouns with contextually appropriate references
        if 'robot_location' in self.context:
            # "here" refers to robot's current location
            pass  # In a real implementation, this would resolve "here" to coordinates

        # "it" refers to the most recently mentioned object
        recent_commands = self.context.get('recent_commands', [])
        if recent_commands:
            # Logic to determine what "it" refers to based on previous commands
            pass

        return parsed
```

## Task Sequencing and Scheduling

Once tasks are identified and decomposed, they must be sequenced and scheduled appropriately to ensure efficient and safe execution.

### Sequential vs. Parallel Execution

Some tasks can be executed in parallel, while others must be executed sequentially. The task planner must determine the optimal execution order based on dependencies and resource constraints.

```python
class TaskScheduler:
    """Schedules tasks for execution considering dependencies and resources"""

    def __init__(self):
        self.resources = {
            'manipulator': True,  # True if available
            'navigation_system': True,
            'speakers': True,
            'microphones': True
        }

    def schedule_tasks(self, task_plan: TaskPlan) -> List[Tuple[float, TaskPlan]]:
        """Schedule tasks with appropriate timing"""
        scheduled_tasks = []
        current_time = 0.0

        # Process tasks in dependency order
        ready_tasks = self._get_ready_tasks(task_plan, [])
        completed_tasks = set()

        while ready_tasks:
            # Find tasks that can be executed in parallel
            executable_tasks = self._find_executable_tasks(ready_tasks, self.resources)

            if not executable_tasks:
                # No tasks can be executed, advance time to when a resource becomes free
                current_time = self._advance_time_for_resource_availability()
                continue

            # Execute tasks in parallel
            for task in executable_tasks:
                start_time = current_time
                end_time = start_time + task.estimated_duration

                scheduled_tasks.append((start_time, task))

                # Update resource availability
                self._allocate_resources(task)

                # Mark task as completed
                completed_tasks.add(task.task_id)

                # Update time to the end of the longest task
                current_time = max(current_time, end_time)

            # Get new ready tasks
            ready_tasks = self._get_ready_tasks(task_plan, list(completed_tasks))

        return scheduled_tasks

    def _get_ready_tasks(self, task_plan: TaskPlan, completed_tasks: List[str]) -> List[TaskPlan]:
        """Get tasks that are ready to be executed"""
        ready_tasks = []

        for task in task_plan.subtasks:
            # Check if all dependencies are completed
            all_deps_met = all(dep in completed_tasks for dep in task.dependencies)

            # Check if task hasn't been completed yet
            if all_deps_met and task.task_id not in completed_tasks:
                ready_tasks.append(task)

        return ready_tasks

    def _find_executable_tasks(self, tasks: List[TaskPlan], resources: Dict[str, bool]) -> List[TaskPlan]:
        """Find tasks that can be executed given available resources"""
        executable_tasks = []

        # Simple resource model - each task type requires specific resources
        resource_requirements = {
            TaskType.NAVIGATION: ['navigation_system'],
            TaskType.MANIPULATION: ['manipulator'],
            TaskType.INTERACTION: ['speakers', 'microphones'],
            TaskType.INFORMATION: ['speakers'],
            TaskType.SYSTEM: []
        }

        for task in tasks:
            # Check if required resources are available
            reqs = resource_requirements.get(task.task_type, [])
            if all(resources.get(req, True) for req in reqs):
                executable_tasks.append(task)

        return executable_tasks

    def _allocate_resources(self, task: TaskPlan):
        """Allocate resources for a task"""
        # In a real system, this would reserve resources
        # For this example, we'll just mark them as unavailable temporarily
        pass

    def _advance_time_for_resource_availability(self) -> float:
        """Advance time to when resources become available"""
        # This would be implemented based on actual resource availability
        return 0.1  # Small time increment
```

## Error Handling and Plan Adaptation

Robotic systems must handle situations where planned tasks fail or need to be adapted based on changing conditions.

### Plan Repair Mechanisms

When a task fails, the system should attempt to repair the plan rather than abandoning the entire goal.

```python
class PlanRepairer:
    """Repairs failed plans and adapts to changing conditions"""

    def __init__(self):
        self.repair_strategies = {
            'navigation_failure': self._repair_navigation_failure,
            'manipulation_failure': self._repair_manipulation_failure,
            'object_not_found': self._repair_object_not_found,
            'resource_unavailable': self._repair_resource_unavailable
        }

    def repair_plan(self, failed_task: TaskPlan, failure_reason: str, original_plan: TaskPlan) -> Optional[TaskPlan]:
        """Repair a plan after a task failure"""
        if failure_reason in self.repair_strategies:
            return self.repair_strategies[failure_reason](failed_task, original_plan)
        else:
            # Default repair strategy
            return self._generic_repair(failed_task, original_plan)

    def _repair_navigation_failure(self, failed_task: TaskPlan, original_plan: TaskPlan) -> Optional[TaskPlan]:
        """Repair plan after navigation failure"""
        # Try alternative path
        new_subtasks = []
        for subtask in original_plan.subtasks:
            if subtask.task_id == failed_task.task_id:
                # Create alternative navigation task
                alt_task = TaskPlan(
                    task_id=f"alt_{subtask.task_id}",
                    task_type=subtask.task_type,
                    description=f"Alternative navigation to {subtask.description}",
                    subtasks=[],
                    parameters={**subtask.parameters, 'alternative': True},
                    priority=subtask.priority,
                    estimated_duration=subtask.estimated_duration * 1.5,  # Longer due to alternative route
                    dependencies=subtask.dependencies
                )
                new_subtasks.append(alt_task)
            else:
                new_subtasks.append(subtask)

        return TaskPlan(
            task_id=original_plan.task_id,
            task_type=original_plan.task_type,
            description=original_plan.description,
            subtasks=new_subtasks,
            parameters=original_plan.parameters,
            priority=original_plan.priority,
            estimated_duration=original_plan.estimated_duration * 1.2,  # Account for detour
            dependencies=original_plan.dependencies
        )

    def _repair_manipulation_failure(self, failed_task: TaskPlan, original_plan: TaskPlan) -> Optional[TaskPlan]:
        """Repair plan after manipulation failure"""
        # Try different grasp strategy or approach
        new_subtasks = []
        for subtask in original_plan.subtasks:
            if subtask.task_id == failed_task.task_id:
                # Create alternative manipulation task
                alt_task = TaskPlan(
                    task_id=f"alt_{subtask.task_id}",
                    task_type=subtask.task_type,
                    description=f"Alternative manipulation of {subtask.description}",
                    subtasks=[],
                    parameters={**subtask.parameters, 'alternative_grasp': True},
                    priority=subtask.priority,
                    estimated_duration=subtask.estimated_duration * 1.3,  # Longer due to careful approach
                    dependencies=subtask.dependencies
                )
                new_subtasks.append(alt_task)
            else:
                new_subtasks.append(subtask)

        return TaskPlan(
            task_id=original_plan.task_id,
            task_type=original_plan.task_type,
            description=original_plan.description,
            subtasks=new_subtasks,
            parameters=original_plan.parameters,
            priority=original_plan.priority,
            estimated_duration=original_plan.estimated_duration * 1.1,
            dependencies=original_plan.dependencies
        )

    def _repair_object_not_found(self, failed_task: TaskPlan, original_plan: TaskPlan) -> Optional[TaskPlan]:
        """Repair plan when expected object is not found"""
        # Expand search area or try alternative objects
        new_subtasks = []
        for subtask in original_plan.subtasks:
            if subtask.task_id == failed_task.task_id:
                # Add search task before manipulation
                search_task = TaskPlan(
                    task_id=f"search_{subtask.task_id}",
                    task_type=TaskType.INFORMATION,
                    description=f"Search for {subtask.parameters.get('object', 'object')}",
                    subtasks=[],
                    parameters={**subtask.parameters, 'search_area': 'expanded'},
                    priority=subtask.priority,
                    estimated_duration=5.0,
                    dependencies=subtask.dependencies
                )

                # Update the original task to depend on search
                manip_task = TaskPlan(
                    task_id=subtask.task_id,
                    task_type=subtask.task_type,
                    description=subtask.description,
                    subtasks=[],
                    parameters=subtask.parameters,
                    priority=subtask.priority,
                    estimated_duration=subtask.estimated_duration,
                    dependencies=[search_task.task_id]
                )

                new_subtasks.extend([search_task, manip_task])
            else:
                new_subtasks.append(subtask)

        return TaskPlan(
            task_id=original_plan.task_id,
            task_type=original_plan.task_type,
            description=original_plan.description,
            subtasks=new_subtasks,
            parameters=original_plan.parameters,
            priority=original_plan.priority,
            estimated_duration=original_plan.estimated_duration + 5.0,
            dependencies=original_plan.dependencies
        )

    def _repair_resource_unavailable(self, failed_task: TaskPlan, original_plan: TaskPlan) -> Optional[TaskPlan]:
        """Repair plan when required resource is unavailable"""
        # Reschedule task for when resource is available
        new_subtasks = []
        for subtask in original_plan.subtasks:
            if subtask.task_id == failed_task.task_id:
                # Reschedule with lower priority
                rescheduled_task = TaskPlan(
                    task_id=f"resched_{subtask.task_id}",
                    task_type=subtask.task_type,
                    description=f"Rescheduled {subtask.description}",
                    subtasks=[],
                    parameters={**subtask.parameters, 'rescheduled': True},
                    priority=max(1, subtask.priority - 1),  # Lower priority
                    estimated_duration=subtask.estimated_duration,
                    dependencies=subtask.dependencies
                )
                new_subtasks.append(rescheduled_task)
            else:
                new_subtasks.append(subtask)

        return TaskPlan(
            task_id=original_plan.task_id,
            task_type=original_plan.task_type,
            description=original_plan.description,
            subtasks=new_subtasks,
            parameters=original_plan.parameters,
            priority=original_plan.priority,
            estimated_duration=original_plan.estimated_duration,
            dependencies=original_plan.dependencies
        )

    def _generic_repair(self, failed_task: TaskPlan, original_plan: TaskPlan) -> Optional[TaskPlan]:
        """Generic repair strategy"""
        # Simply remove the failed task and continue with the rest
        remaining_subtasks = [st for st in original_plan.subtasks if st.task_id != failed_task.task_id]

        return TaskPlan(
            task_id=original_plan.task_id,
            task_type=original_plan.task_type,
            description=f"Modified {original_plan.description} (excluding failed task)",
            subtasks=remaining_subtasks,
            parameters=original_plan.parameters,
            priority=original_plan.priority,
            estimated_duration=sum(st.estimated_duration for st in remaining_subtasks),
            dependencies=original_plan.dependencies
        )
```

## Integration with VLA System

The natural language to task planning system must be tightly integrated with the broader VLA system to enable coordinated behavior.

### Coordination with Vision and Action Systems

The task planning system needs to communicate with vision and action systems to ensure plans are executable and to adapt plans based on real-time information.

```python
class VLATaskCoordinator:
    """Coordinates task planning with vision and action systems"""

    def __init__(self):
        self.task_planner = TaskPlanner()
        self.hierarchical_planner = HierarchicalTaskPlanner()
        self.scheduler = TaskScheduler()
        self.repairer = PlanRepairer()

        # Current system state
        self.current_plan = None
        self.executing_tasks = []
        self.completed_tasks = []

    def process_command(self, command: NLCommand, context: Dict[str, Any]) -> Optional[TaskPlan]:
        """Process a natural language command through the VLA system"""
        # Step 1: Create initial task plan
        initial_plan = self.task_planner.plan_from_command(command)
        if not initial_plan:
            return None

        # Step 2: Refine plan with context
        contextual_parser = ContextualSemanticParser()
        parsed_with_context = contextual_parser.parse_with_context(command.text, context)

        refined_plan = self.hierarchical_planner.refine_plan(initial_plan, context)

        # Step 3: Schedule tasks
        scheduled_tasks = self.scheduler.schedule_tasks(refined_plan)

        # Step 4: Return the final plan
        self.current_plan = refined_plan
        return refined_plan

    def handle_execution_feedback(self, task_id: str, status: TaskStatus, feedback: Dict[str, Any]) -> Optional[TaskPlan]:
        """Handle feedback from task execution and adapt plan if needed"""
        if status == TaskStatus.FAILED:
            # Find the failed task
            failed_task = self._find_task_by_id(self.current_plan, task_id)
            if failed_task:
                # Repair the plan
                failure_reason = feedback.get('failure_reason', 'unknown')
                repaired_plan = self.repairer.repair_plan(failed_task, failure_reason, self.current_plan)
                if repaired_plan:
                    self.current_plan = repaired_plan
                    return repaired_plan

        elif status == TaskStatus.COMPLETED:
            # Mark task as completed
            self.completed_tasks.append(task_id)

        return self.current_plan

    def _find_task_by_id(self, plan: TaskPlan, task_id: str) -> Optional[TaskPlan]:
        """Find a task in a plan by its ID"""
        if plan.task_id == task_id:
            return plan

        for subtask in plan.subtasks:
            found = self._find_task_by_id(subtask, task_id)
            if found:
                return found

        return None
```

## Evaluation Metrics

Evaluating natural language to task planning systems requires metrics that assess both the quality of the plans and their effectiveness in achieving user goals.

### Plan Quality Metrics

1. **Success Rate**: Percentage of tasks successfully completed
2. **Plan Efficiency**: Ratio of actual to estimated task duration
3. **User Satisfaction**: Subjective measure of how well the robot understood and executed commands
4. **Adaptability**: Ability to handle unexpected situations and repair plans

### Quantitative Evaluation

```python
class TaskPlanningEvaluator:
    """Evaluates the performance of task planning systems"""

    def __init__(self):
        self.metrics = {
            'success_rate': 0.0,
            'efficiency_ratio': 1.0,
            'average_plan_depth': 0.0,
            'repair_frequency': 0.0
        }

    def evaluate_plan_execution(self, plan: TaskPlan, execution_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate a plan based on its execution log"""
        metrics = {}

        # Calculate success rate
        completed_tasks = [log for log in execution_log if log.get('status') == 'completed']
        failed_tasks = [log for log in execution_log if log.get('status') == 'failed']

        total_tasks = len(plan.subtasks) if plan.subtasks else 1
        completed_count = len(completed_tasks)

        metrics['success_rate'] = completed_count / total_tasks if total_tasks > 0 else 0.0

        # Calculate efficiency ratio
        if completed_tasks:
            actual_duration = sum(log.get('actual_duration', 0) for log in completed_tasks)
            estimated_duration = sum(st.estimated_duration for st in plan.subtasks if st.status == TaskStatus.COMPLETED)
            metrics['efficiency_ratio'] = actual_duration / estimated_duration if estimated_duration > 0 else float('inf')
        else:
            metrics['efficiency_ratio'] = float('inf')

        # Calculate average plan depth (how decomposed the plan is)
        metrics['average_plan_depth'] = self._calculate_plan_depth(plan)

        # Calculate repair frequency
        repair_events = [log for log in execution_log if log.get('event_type') == 'plan_repair']
        metrics['repair_frequency'] = len(repair_events) / len(execution_log) if execution_log else 0.0

        return metrics

    def _calculate_plan_depth(self, plan: TaskPlan) -> float:
        """Calculate the average depth of the task plan"""
        def get_depth(task_plan, current_depth=0):
            if not task_plan.subtasks:
                return current_depth
            return max(get_depth(subtask, current_depth + 1) for subtask in task_plan.subtasks)

        return get_depth(plan)

    def evaluate_natural_language_understanding(self, commands: List[NLCommand], expected_intents: List[TaskType]) -> Dict[str, float]:
        """Evaluate the semantic parsing and intent recognition"""
        correct_predictions = 0
        total_commands = len(commands)

        for i, command in enumerate(commands):
            if i < len(expected_intents):
                # Parse the command
                parsed = self._parse_command(command.text)

                # Recognize intent
                intent_recognizer = IntentRecognizer()
                intent, _ = intent_recognizer.recognize_intent(parsed, command.text)

                if intent == expected_intents[i]:
                    correct_predictions += 1

        return {
            'understanding_accuracy': correct_predictions / total_commands if total_commands > 0 else 0.0,
            'total_evaluated': total_commands
        }

    def _parse_command(self, text: str) -> Dict[str, Any]:
        """Parse command for evaluation purposes"""
        # Use the semantic parser
        parser = SemanticParser()
        return parser(text)
```

## Implementation Considerations

### Performance Optimization

Natural language to task planning systems must operate efficiently to maintain responsive interaction:

1. **Caching**: Store results of expensive parsing operations
2. **Preprocessing**: Pre-analyze common command patterns
3. **Parallel Processing**: Execute independent tasks simultaneously
4. **Incremental Updates**: Update plans incrementally rather than regenerating

### Safety and Validation

Safety is paramount in robotic systems:

1. **Constraint Checking**: Verify plans don't violate safety constraints
2. **Human Oversight**: Provide mechanisms for human intervention
3. **Fail-Safe Mechanisms**: Ensure robots can safely stop if something goes wrong
4. **Validation Layers**: Multiple layers of validation before executing commands

## Chapter Summary

This chapter explored the transformation of natural language commands into structured task plans for humanoid robots. We covered:

- The pipeline from natural language to task planning: semantic parsing, intent recognition, task decomposition, and action sequencing
- Hierarchical task planning for managing complex goals
- Context-aware semantic parsing for better understanding
- Task scheduling and coordination mechanisms
- Error handling and plan adaptation strategies
- Integration with the broader VLA system
- Evaluation metrics for assessing system performance

Effective natural language to task planning is crucial for intuitive human-robot interaction, requiring sophisticated natural language processing, planning algorithms, and system integration. The success of these systems depends on balancing understanding flexibility with execution reliability.

## Next Steps

In the next chapter, we'll explore how large language models are leveraged for cognitive planning in humanoid robots, examining the integration of advanced AI techniques with physical robot control.

## Exercises

1. **Implementation Challenge**: Implement a simple natural language to task planning system using the structures provided. Test it with various command types and evaluate its performance.

2. **Hierarchical Planning**: Design a hierarchical plan for a complex task like "Prepare a simple meal" and implement the decomposition logic.

3. **Context Integration**: Enhance the semantic parser to better incorporate contextual information for disambiguation.

4. **Plan Repair**: Implement additional plan repair strategies for different types of failures.

5. **Evaluation Framework**: Create a comprehensive evaluation framework for task planning systems and test it with various scenarios.