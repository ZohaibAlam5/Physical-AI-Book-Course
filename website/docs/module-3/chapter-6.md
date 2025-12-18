---
title: "Chapter 6 - Cognitive Mapping and Spatial Reasoning"
description: "Building cognitive maps and spatial reasoning capabilities for humanoid robots to understand and navigate complex environments"
sidebar_label: "Chapter 6 - Cognitive Mapping and Spatial Reasoning"
---

# Cognitive Mapping and Spatial Reasoning

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement cognitive mapping systems for humanoid robots
- Design spatial reasoning algorithms that enable environment understanding
- Create semantic maps that combine geometric and semantic information
- Implement topological mapping for efficient path planning
- Develop spatial memory systems for long-term navigation
- Apply graph-based reasoning for complex spatial tasks

## Introduction

Cognitive mapping and spatial reasoning form the foundation of intelligent navigation for humanoid robots. Unlike simple metric maps that only represent geometric relationships, cognitive maps incorporate semantic information, affordances, and higher-level spatial concepts that enable robots to reason about their environment in human-like ways.

For humanoid robots, cognitive mapping is particularly important because these robots must navigate complex, human-designed environments where understanding the meaning and function of spaces is as important as knowing their geometric properties. A humanoid robot should understand that a "kitchen" is a place for food preparation, that "stairs" require special locomotion patterns, and that "doors" are passable openings that may need to be opened.

This chapter explores the implementation of cognitive mapping systems that enable humanoid robots to build rich, meaningful representations of their environment and use these representations for intelligent navigation and interaction.

## Cognitive Map Architecture

### Hierarchical Spatial Representation

Cognitive maps use a hierarchical structure that represents space at multiple levels of abstraction:

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import networkx as nx
from scipy.spatial import cKDTree
import json
from datetime import datetime

class SpatialEntityType(Enum):
    """Types of entities in cognitive maps"""
    PLACE = "place"
    OBJECT = "object"
    PATH = "path"
    REGION = "region"
    LANDMARK = "landmark"
    AFFORDANCE = "affordance"

@dataclass
class SpatialEntity:
    """Base class for spatial entities in cognitive maps"""
    id: str
    entity_type: SpatialEntityType
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw] or quaternion
    properties: Dict[str, Any]
    semantic_labels: List[str]
    confidence: float
    timestamp: datetime

@dataclass
class PlaceEntity(SpatialEntity):
    """Represents a meaningful place or room"""
    place_type: str  # 'kitchen', 'bedroom', 'office', etc.
    area: float
    accessibility: Dict[str, bool]  # accessible from other places
    functional_properties: Dict[str, Any]  # what activities happen here

@dataclass
class PathEntity(SpatialEntity):
    """Represents a navigable path between locations"""
    path_type: str  # 'corridor', 'doorway', 'staircase', etc.
    connectivity: List[str]  # connects to which places
    traversability: Dict[str, float]  # difficulty for different gaits
    width: float

@dataclass
class ObjectEntity(SpatialEntity):
    """Represents an object in the environment"""
    object_type: str  # 'chair', 'table', 'door', etc.
    size: np.ndarray  # [length, width, height]
    affordances: List[str]  # 'sit_on', 'grasp', 'open', etc.
    stability: float  # likelihood of moving

class CognitiveMap:
    """Hierarchical cognitive map for humanoid robots"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.spatial_entities = {}  # id -> SpatialEntity
        self.topological_graph = nx.Graph()  # Topological connections
        self.metric_map = None  # Detailed geometric map
        self.semantic_layer = {}  # Semantic annotations
        self.spatial_memory = SpatialMemory()
        self.hierarchy_levels = {
            'metric': 0,      # Detailed geometry
            'object': 1,      # Objects and their properties
            'place': 2,       # Rooms and areas
            'topological': 3, # Connectivity between places
            'semantic': 4     # Meaning and function
        }

    def add_entity(self, entity: SpatialEntity):
        """Add a spatial entity to the cognitive map"""
        self.spatial_entities[entity.id] = entity

        # Update different map layers based on entity type
        if entity.entity_type == SpatialEntityType.PLACE:
            self._add_place_entity(entity)
        elif entity.entity_type == SpatialEntityType.PATH:
            self._add_path_entity(entity)
        elif entity.entity_type == SpatialEntityType.OBJECT:
            self._add_object_entity(entity)

        # Update spatial memory
        self.spatial_memory.update(entity)

    def _add_place_entity(self, entity: SpatialEntity):
        """Add place entity to topological graph"""
        place_entity = entity
        self.topological_graph.add_node(
            entity.id,
            type='place',
            position=entity.position,
            place_type=place_entity.place_type,
            area=place_entity.area
        )

    def _add_path_entity(self, entity: SpatialEntity):
        """Add path entity to connect places"""
        path_entity = entity
        self.topological_graph.add_node(
            entity.id,
            type='path',
            position=entity.position,
            path_type=path_entity.path_type,
            width=path_entity.width
        )

        # Connect to adjacent places
        for connected_id in path_entity.connectivity:
            if connected_id in self.topological_graph.nodes:
                self.topological_graph.add_edge(
                    entity.id,
                    connected_id,
                    weight=self._calculate_path_weight(entity, connected_id)
                )

    def _add_object_entity(self, entity: SpatialEntity):
        """Add object entity to semantic layer"""
        object_entity = entity
        self.semantic_layer[entity.id] = {
            'type': object_entity.object_type,
            'size': object_entity.size,
            'affordances': object_entity.affordances,
            'position': entity.position
        }

    def _calculate_path_weight(self, path_entity, connected_place_id):
        """Calculate path weight based on traversability and distance"""
        connected_place = self.spatial_entities[connected_place_id]
        distance = np.linalg.norm(path_entity.position - connected_place.position)

        # Factor in traversability for humanoid robot
        gait = self.robot_config.get('preferred_gait', 'walk')
        traversability = path_entity.traversability.get(gait, 1.0)

        return distance * traversability

    def get_reachable_places(self, start_place_id: str, max_distance: float = 10.0):
        """Get places reachable within max_distance"""
        try:
            # Use Dijkstra's algorithm on the topological graph
            lengths = nx.single_source_dijkstra_path_length(
                self.topological_graph, start_place_id, cutoff=max_distance
            )
            return list(lengths.keys())
        except nx.NetworkXNoPath:
            return []

    def find_path(self, start_place_id: str, goal_place_id: str):
        """Find path between two places in the cognitive map"""
        try:
            path = nx.shortest_path(
                self.topological_graph, start_place_id, goal_place_id
            )
            return path
        except nx.NetworkXNoPath:
            return None

    def get_place_by_type(self, place_type: str):
        """Get all places of a specific type"""
        places = []
        for entity_id, entity in self.spatial_entities.items():
            if (entity.entity_type == SpatialEntityType.PLACE and
                hasattr(entity, 'place_type') and
                entity.place_type == place_type):
                places.append(entity)
        return places

    def get_objects_by_affordance(self, affordance: str):
        """Get all objects that have a specific affordance"""
        objects = []
        for entity_id, entity in self.spatial_entities.items():
            if (entity.entity_type == SpatialEntityType.OBJECT and
                hasattr(entity, 'affordances') and
                affordance in entity.affordances):
                objects.append(entity)
        return objects

    def update_entity_confidence(self, entity_id: str, new_confidence: float):
        """Update confidence of a spatial entity"""
        if entity_id in self.spatial_entities:
            self.spatial_entities[entity_id].confidence = new_confidence
            self.spatial_entities[entity_id].timestamp = datetime.now()

    def remove_unreliable_entities(self, min_confidence: float = 0.5):
        """Remove entities with confidence below threshold"""
        unreliable_ids = [
            eid for eid, entity in self.spatial_entities.items()
            if entity.confidence < min_confidence
        ]

        for entity_id in unreliable_ids:
            del self.spatial_entities[entity_id]

            # Remove from topological graph if present
            if entity_id in self.topological_graph.nodes:
                self.topological_graph.remove_node(entity_id)
```

### Spatial Memory System

The spatial memory system maintains long-term knowledge about the environment:

```python
class SpatialMemory:
    """Long-term spatial memory for humanoid robots"""

    def __init__(self):
        self.memory_nodes = {}  # location -> memory entry
        self.episodic_memory = []  # sequence of experiences
        self.semantic_memory = {}  # general knowledge about space
        self.spatial_relations = {}  # relationships between entities
        self.forgetting_factor = 0.95  # How quickly memories fade

    def update(self, spatial_entity: SpatialEntity):
        """Update spatial memory with new entity"""
        location_key = self._get_location_key(spatial_entity.position)

        if location_key not in self.memory_nodes:
            self.memory_nodes[location_key] = {
                'entities': [],
                'last_seen': datetime.now(),
                'frequency': 0,
                'stability': 1.0  # How stable this location is
            }

        # Add entity to location
        self.memory_nodes[location_key]['entities'].append(spatial_entity)
        self.memory_nodes[location_key]['last_seen'] = datetime.now()
        self.memory_nodes[location_key]['frequency'] += 1

        # Update stability based on frequency of visits
        self.memory_nodes[location_key]['stability'] = min(
            1.0, self.memory_nodes[location_key]['frequency'] * 0.1
        )

    def _get_location_key(self, position: np.ndarray, resolution: float = 0.5):
        """Create a location key by discretizing position"""
        discrete_pos = (position / resolution).astype(int)
        return tuple(discrete_pos)

    def recall_entities_in_region(self, center_pos: np.ndarray, radius: float):
        """Recall entities in a spatial region"""
        entities = []
        center_key = self._get_location_key(center_pos)
        search_resolution = 1.0  # Resolution for search

        # Calculate search bounds
        min_bounds = (center_pos - radius) / search_resolution
        max_bounds = (center_pos + radius) / search_resolution

        for loc_key, memory_entry in self.memory_nodes.items():
            loc_pos = np.array(loc_key) * search_resolution

            if np.linalg.norm(loc_pos - center_pos) <= radius:
                entities.extend(memory_entry['entities'])

        return entities

    def predict_entity_location(self, entity_type: str, context: Dict):
        """Predict likely location of an entity based on context"""
        # Use semantic memory to predict likely locations
        if entity_type in self.semantic_memory:
            # Get typical locations for this entity type
            typical_locations = self.semantic_memory[entity_type].get('typical_locations', [])

            # Filter based on current context
            context_filtered = self._filter_by_context(typical_locations, context)
            return context_filtered

        return []

    def _filter_by_context(self, locations, context):
        """Filter locations based on contextual information"""
        # This would implement context-based filtering
        # For example, if context indicates 'kitchen', prefer kitchen locations
        return locations

    def update_semantic_knowledge(self, entity_type: str, location: str, frequency: float):
        """Update semantic knowledge about entity-location relationships"""
        if entity_type not in self.semantic_memory:
            self.semantic_memory[entity_type] = {
                'typical_locations': {},
                'affordances': [],
                'properties': {}
            }

        if location not in self.semantic_memory[entity_type]['typical_locations']:
            self.semantic_memory[entity_type]['typical_locations'][location] = 0

        self.semantic_memory[entity_type]['typical_locations'][location] += frequency

    def decay_memory(self):
        """Apply forgetting to old memories"""
        current_time = datetime.now()

        # Remove old episodic memories
        self.episodic_memory = [
            episode for episode in self.episodic_memory
            if (current_time - episode['timestamp']).days < 30  # Keep 30 days
        ]

        # Reduce frequency of infrequently visited locations
        for location_key, memory_entry in self.memory_nodes.items():
            time_since_visit = (current_time - memory_entry['last_seen']).days
            if time_since_visit > 0:
                memory_entry['frequency'] *= (self.forgetting_factor ** time_since_visit)

                # Remove if frequency becomes too low
                if memory_entry['frequency'] < 0.1:
                    del self.memory_nodes[location_key]

    def get_contextual_map(self, context: Dict):
        """Get map adapted to current context"""
        # Return a map that emphasizes relevant entities based on context
        contextual_entities = {}

        for entity_id, entity in self.memory_nodes.items():
            if self._is_relevant_to_context(entity, context):
                contextual_entities[entity_id] = entity

        return contextual_entities

    def _is_relevant_to_context(self, memory_entry, context):
        """Check if memory entry is relevant to current context"""
        # This would implement context-aware relevance checking
        # For example, in 'searching_for_food' context, kitchen-related entities are more relevant
        return True  # Simplified
```

## Semantic Mapping and Understanding

### Semantic Scene Graphs

Semantic scene graphs represent objects and their relationships:

```python
class SemanticSceneGraph:
    """Semantic scene graph for spatial reasoning"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.graph = nx.MultiDiGraph()
        self.object_categories = self._initialize_categories()
        self.spatial_relations = [
            'on', 'in', 'near', 'above', 'below', 'beside', 'between'
        ]

    def _initialize_categories(self):
        """Initialize object category hierarchy"""
        return {
            'furniture': ['chair', 'table', 'sofa', 'bed', 'cabinet'],
            'appliances': ['refrigerator', 'microwave', 'oven', 'dishwasher'],
            'architectural': ['door', 'window', 'wall', 'floor', 'ceiling'],
            'navigation': ['staircase', 'elevator', 'corridor', 'room'],
            'interactive': ['handle', 'button', 'switch', 'knob']
        }

    def add_object(self, obj_id: str, obj_type: str, position: np.ndarray):
        """Add object to semantic graph"""
        category = self._get_category(obj_type)

        self.graph.add_node(
            obj_id,
            type=obj_type,
            category=category,
            position=position,
            properties={}
        )

    def _get_category(self, obj_type: str):
        """Get category for object type"""
        for category, types in self.object_categories.items():
            if obj_type.lower() in types:
                return category
        return 'other'

    def add_relationship(self, subject_id: str, relation: str, object_id: str, confidence: float = 1.0):
        """Add spatial relationship between objects"""
        if relation not in self.spatial_relations:
            raise ValueError(f"Unknown spatial relation: {relation}")

        self.graph.add_edge(
            subject_id,
            object_id,
            relation=relation,
            confidence=confidence
        )

    def get_objects_by_category(self, category: str):
        """Get all objects of a specific category"""
        objects = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('category') == category:
                objects.append(node_id)
        return objects

    def get_relationships(self, obj_id: str, relation: str = None):
        """Get relationships for an object"""
        relationships = []

        # Outgoing relationships
        for target in self.graph.successors(obj_id):
            for key, edge_data in self.graph[obj_id][target].items():
                if relation is None or edge_data.get('relation') == relation:
                    relationships.append({
                        'subject': obj_id,
                        'relation': edge_data['relation'],
                        'object': target,
                        'confidence': edge_data.get('confidence', 1.0)
                    })

        # Incoming relationships
        for source in self.graph.predecessors(obj_id):
            for key, edge_data in self.graph[source][obj_id].items():
                if relation is None or edge_data.get('relation') == relation:
                    relationships.append({
                        'subject': source,
                        'relation': edge_data['relation'],
                        'object': obj_id,
                        'confidence': edge_data.get('confidence', 1.0)
                    })

        return relationships

    def find_path_with_semantics(self, start_obj: str, end_obj: str, constraints: Dict):
        """Find path between objects considering semantic constraints"""
        # Create subgraph with only valid nodes based on constraints
        valid_nodes = self._filter_nodes_by_constraints(constraints)

        # Create subgraph
        subgraph = self.graph.subgraph(valid_nodes)

        try:
            path = nx.shortest_path(subgraph, start_obj, end_obj)
            return path
        except nx.NetworkXNoPath:
            return None

    def _filter_nodes_by_constraints(self, constraints: Dict):
        """Filter nodes based on semantic constraints"""
        valid_nodes = []

        for node_id, attrs in self.graph.nodes(data=True):
            if self._node_satisfies_constraints(node_id, attrs, constraints):
                valid_nodes.append(node_id)

        return valid_nodes

    def _node_satisfies_constraints(self, node_id, attrs, constraints):
        """Check if node satisfies semantic constraints"""
        # Check if node has required properties
        required_types = constraints.get('required_types', [])
        if required_types and not any(t in attrs.get('type', '') for t in required_types):
            return False

        # Check if node doesn't have forbidden properties
        forbidden_types = constraints.get('forbidden_types', [])
        if any(t in attrs.get('type', '') for t in forbidden_types):
            return False

        return True

    def infer_missing_relationships(self):
        """Infer likely relationships based on spatial proximity and common patterns"""
        # This would implement relationship inference
        # For example, if two objects are close and one is typically 'on' the other
        inferred_relationships = []

        # Get all object pairs that are close
        close_pairs = self._get_close_object_pairs(distance_threshold=1.0)

        for obj1, obj2 in close_pairs:
            # Check if there's a likely relationship based on object types
            likely_relations = self._get_likely_relationships(
                self.graph.nodes[obj1]['type'],
                self.graph.nodes[obj2]['type']
            )

            for relation in likely_relations:
                # Add inferred relationship with low confidence
                self.add_relationship(obj1, relation, obj2, confidence=0.3)
                inferred_relationships.append((obj1, relation, obj2))

        return inferred_relationships

    def _get_close_object_pairs(self, distance_threshold: float):
        """Get pairs of objects that are spatially close"""
        close_pairs = []

        positions = {}
        for node_id, attrs in self.graph.nodes(data=True):
            if 'position' in attrs:
                positions[node_id] = attrs['position']

        # Check all pairs for proximity
        node_list = list(positions.keys())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                obj1, obj2 = node_list[i], node_list[j]
                dist = np.linalg.norm(positions[obj1] - positions[obj2])

                if dist <= distance_threshold:
                    close_pairs.append((obj1, obj2))

        return close_pairs

    def _get_likely_relationships(self, type1: str, type2: str):
        """Get likely relationships between object types"""
        # Define common relationship patterns
        patterns = {
            ('mug', 'table'): ['on'],
            ('book', 'shelf'): ['on'],
            ('chair', 'table'): ['beside', 'near'],
            ('refrigerator', 'kitchen'): ['in'],
            ('bed', 'bedroom'): ['in'],
            ('door', 'room'): ['in', 'beside']
        }

        # Check both directions
        if (type1, type2) in patterns:
            return patterns[(type1, type2)]
        elif (type2, type1) in patterns:
            return patterns[(type2, type1)]

        return []

    def get_semantic_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, task_context: str):
        """Get semantic path that considers task context"""
        # This would implement semantic path planning
        # For example, when looking for food, prioritize kitchen-related objects
        semantic_path = []

        if task_context == 'find_food':
            # Prioritize kitchen objects and food-related paths
            kitchen_objects = self.get_objects_by_category('appliances')
            food_objects = ['refrigerator', 'microwave', 'counter']
        elif task_context == 'find_seat':
            # Prioritize seating objects
            seating_objects = self.get_objects_by_category('furniture')

        # Implementation would find path through relevant semantic nodes
        return semantic_path
```

## Topological Mapping for Navigation

### Topological Graph Construction

Topological maps represent connectivity between places rather than detailed geometry:

```python
class TopologicalMapper:
    """Create and maintain topological maps for navigation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.topological_graph = nx.Graph()
        self.visited_places = set()
        self.path_segments = {}
        self.topological_threshold = 2.0  # meters between topological nodes
        self.min_visit_frequency = 3  # minimum visits to create node

    def update_topological_map(self, robot_pose, sensor_data):
        """Update topological map with current sensor data"""
        current_place = self._identify_current_place(robot_pose, sensor_data)

        if current_place and self._should_create_node(robot_pose, current_place):
            node_id = self._create_topological_node(robot_pose, current_place)

            # Connect to nearby nodes
            self._connect_to_nearby_nodes(robot_pose, node_id)

            # Store path segment if this is a new connection
            self._update_path_segments(robot_pose, node_id)

    def _identify_current_place(self, robot_pose, sensor_data):
        """Identify what kind of place the robot is currently in"""
        # Analyze sensor data to identify place type
        # This could use:
        # - Object detection to identify room contents
        # - Architectural features
        # - Semantic segmentation
        # - Previous map knowledge

        place_type = self._classify_place(sensor_data)
        place_features = self._extract_place_features(sensor_data)

        return {
            'type': place_type,
            'features': place_features,
            'pose': robot_pose
        }

    def _classify_place(self, sensor_data):
        """Classify place type based on sensor data"""
        # Use detected objects and environmental features
        detected_objects = sensor_data.get('objects', [])

        # Count objects of different types
        object_counts = {}
        for obj in detected_objects:
            obj_type = obj.get('type', 'unknown')
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

        # Classify based on object composition
        if object_counts.get('bed', 0) > 0:
            return 'bedroom'
        elif object_counts.get('kitchen_counter', 0) > 0 or object_counts.get('refrigerator', 0) > 0:
            return 'kitchen'
        elif object_counts.get('desk', 0) > 0 or object_counts.get('computer', 0) > 0:
            return 'office'
        elif object_counts.get('sofa', 0) > 0:
            return 'living_room'
        elif object_counts.get('door', 0) > 2:
            return 'corridor'
        else:
            return 'unknown'

    def _extract_place_features(self, sensor_data):
        """Extract features that characterize the place"""
        features = {
            'area': self._estimate_area(sensor_data),
            'shape': self._estimate_shape(sensor_data),
            'obstacles': len(sensor_data.get('obstacles', [])),
            'entrances': self._count_entrances(sensor_data),
            'furniture_count': len([o for o in sensor_data.get('objects', [])
                                  if o.get('type') in ['chair', 'table', 'sofa', 'bed']])
        }
        return features

    def _should_create_node(self, robot_pose, current_place):
        """Determine if a new topological node should be created"""
        # Check if this location is far enough from existing nodes
        for node_id in self.topological_graph.nodes():
            existing_pose = self.topological_graph.nodes[node_id]['pose']
            distance = np.linalg.norm(robot_pose[:2] - existing_pose[:2])

            if distance < self.topological_threshold:
                # Close to existing node, update its properties
                self._update_existing_node(node_id, current_place)
                return False

        # Check if this place has been visited enough times
        place_key = self._get_place_key(current_place)
        visit_count = self.visited_places.get(place_key, 0)

        return visit_count >= self.min_visit_frequency

    def _get_place_key(self, place_info):
        """Get a key for identifying similar places"""
        # Use place type and rough location
        pos = place_info['pose']
        location_key = (int(pos[0] / 5), int(pos[1] / 5))  # 5m grid
        return f"{place_info['type']}_{location_key}"

    def _create_topological_node(self, robot_pose, place_info):
        """Create a new topological node"""
        node_id = f"node_{len(self.topological_graph.nodes())}"

        self.topological_graph.add_node(
            node_id,
            pose=robot_pose,
            place_type=place_info['type'],
            features=place_info['features'],
            visit_count=1,
            last_visited=datetime.now()
        )

        # Update visit count
        place_key = self._get_place_key(place_info)
        self.visited_places[place_key] = self.visited_places.get(place_key, 0) + 1

        return node_id

    def _connect_to_nearby_nodes(self, robot_pose, new_node_id):
        """Connect new node to nearby nodes"""
        for node_id in list(self.topological_graph.nodes()):
            if node_id == new_node_id:
                continue

            existing_pose = self.topological_graph.nodes[node_id]['pose']
            distance = np.linalg.norm(robot_pose[:2] - existing_pose[:2])

            # Connect if nodes are close enough and of compatible types
            # (e.g., don't connect kitchen to bedroom directly unless there's a path)
            if distance < self.topological_threshold * 2:
                # Check if connection is reasonable based on place types
                new_place_type = self.topological_graph.nodes[new_node_id]['place_type']
                existing_place_type = self.topological_graph.nodes[node_id]['place_type']

                if self._are_places_connectable(new_place_type, existing_place_type):
                    self.topological_graph.add_edge(
                        new_node_id,
                        node_id,
                        weight=distance,
                        traversability=self._calculate_traversability(new_node_id, node_id)
                    )

    def _are_places_connectable(self, place1, place2):
        """Check if two places should be connected in topological map"""
        # Define which place types can be directly connected
        connectable_pairs = [
            ('corridor', 'kitchen'),
            ('corridor', 'bedroom'),
            ('corridor', 'living_room'),
            ('corridor', 'office'),
            ('kitchen', 'dining_room'),
            ('living_room', 'kitchen'),
            ('bedroom', 'bathroom')
        ]

        # Check both directions
        return ((place1, place2) in connectable_pairs or
                (place2, place1) in connectable_pairs or
                place1 == place2)  # Same place type can be connected if separate instances

    def _calculate_traversability(self, node1_id, node2_id):
        """Calculate how easy it is to traverse between nodes"""
        # Consider place types, obstacles, and other factors
        node1_type = self.topological_graph.nodes[node1_id]['place_type']
        node2_type = self.topological_graph.nodes[node2_id]['place_type']

        # Different place type transitions may have different costs
        transition_costs = {
            ('corridor', 'room'): 1.0,
            ('kitchen', 'living_room'): 1.0,
            ('bedroom', 'bathroom'): 1.0,
            ('kitchen', 'bedroom'): 1.5,  # Less likely direct connection
        }

        cost_key = (node1_type, node2_type) if node1_type <= node2_type else (node2_type, node1_type)
        base_cost = transition_costs.get(cost_key, 1.0)

        # Get distance
        pos1 = self.topological_graph.nodes[node1_id]['pose']
        pos2 = self.topological_graph.nodes[node2_id]['pose']
        distance = np.linalg.norm(pos1[:2] - pos2[:2])

        return base_cost * distance

    def _update_existing_node(self, node_id, current_place):
        """Update properties of existing node"""
        self.topological_graph.nodes[node_id]['visit_count'] += 1
        self.topological_graph.nodes[node_id]['last_visited'] = datetime.now()

        # Update features if they've changed
        old_features = self.topological_graph.nodes[node_id]['features']
        new_features = current_place['features']

        # Merge features (this is simplified)
        for key, value in new_features.items():
            if isinstance(value, (int, float)):
                old_features[key] = (old_features.get(key, 0) + value) / 2
            else:
                old_features[key] = value

    def find_topological_path(self, start_place_type, goal_place_type):
        """Find path between place types in topological map"""
        # Find all nodes of start type
        start_nodes = [
            node_id for node_id, attrs in self.topological_graph.nodes(data=True)
            if attrs.get('place_type') == start_place_type
        ]

        # Find all nodes of goal type
        goal_nodes = [
            node_id for node_id, attrs in self.topological_graph.nodes(data=True)
            if attrs.get('place_type') == goal_place_type
        ]

        if not start_nodes or not goal_nodes:
            return None

        # Find shortest path between any start and goal node
        best_path = None
        best_length = float('inf')

        for start_node in start_nodes:
            for goal_node in goal_nodes:
                try:
                    path = nx.shortest_path(
                        self.topological_graph, start_node, goal_node, weight='weight'
                    )
                    length = nx.shortest_path_length(
                        self.topological_graph, start_node, goal_node, weight='weight'
                    )

                    if length < best_length:
                        best_length = length
                        best_path = path
                except nx.NetworkXNoPath:
                    continue

        return best_path

    def get_navigation_strategy(self, start_node, goal_node):
        """Get navigation strategy for traversing topological path"""
        try:
            path = nx.shortest_path(self.topological_graph, start_node, goal_node)

            strategy = []
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]

                strategy.append({
                    'from': current_node,
                    'to': next_node,
                    'edge_data': self.topological_graph[current_node][next_node],
                    'local_goal': self.topological_graph.nodes[next_node]['pose']
                })

            return strategy
        except nx.NetworkXNoPath:
            return None
```

## Spatial Reasoning and Query Processing

### Spatial Query System

The spatial query system allows the robot to reason about space and answer questions:

```python
class SpatialQueryProcessor:
    """Process spatial queries and reasoning tasks"""

    def __init__(self, cognitive_map: CognitiveMap):
        self.cognitive_map = cognitive_map
        self.query_templates = self._initialize_query_templates()

    def _initialize_query_templates(self):
        """Initialize templates for common spatial queries"""
        return {
            'location_query': [
                'where is the {object}?',
                'find the {object}',
                'locate {object}'
            ],
            'path_query': [
                'how to get from {start} to {end}?',
                'path from {start} to {end}',
                'navigate to {end} from {start}'
            ],
            'relationship_query': [
                'what is {relation} the {object}?',
                'find objects {relation} to {object}',
                'what objects are near {object}?'
            ],
            'spatial_reasoning': [
                'can I {action} in {location}?',
                'is {object1} accessible from {object2}?',
                'what can I do with {object}?'
            ]
        }

    def process_query(self, query: str):
        """Process a spatial query and return answer"""
        query_type, params = self._parse_query(query)

        if query_type == 'location':
            return self._process_location_query(params)
        elif query_type == 'path':
            return self._process_path_query(params)
        elif query_type == 'relationship':
            return self._process_relationship_query(params)
        elif query_type == 'reasoning':
            return self._process_reasoning_query(params)
        else:
            return self._process_general_query(query)

    def _parse_query(self, query: str):
        """Parse query to determine type and parameters"""
        query_lower = query.lower()

        # Location queries
        if any(keyword in query_lower for keyword in ['where', 'find', 'locate']):
            # Extract object name
            import re
            object_match = re.search(r'(?:the\s+|a\s+|an\s+)(\w+)', query_lower)
            if object_match:
                return 'location', {'object': object_match.group(1)}

        # Path queries
        if any(keyword in query_lower for keyword in ['how to get', 'path', 'navigate', 'from']):
            # Extract start and end locations
            import re
            from_match = re.search(r'from\s+(\w+)', query_lower)
            to_match = re.search(r'to\s+(\w+)', query_lower)

            if from_match and to_match:
                return 'path', {
                    'start': from_match.group(1),
                    'end': to_match.group(1)
                }

        # Relationship queries
        if any(keyword in query_lower for keyword in ['near', 'beside', 'next to', 'by']):
            return 'relationship', {'query': query_lower}

        # Default to general query
        return 'general', {'query': query}

    def _process_location_query(self, params):
        """Process location query"""
        object_type = params.get('object', '')

        # Find all objects of this type
        entities = []
        for entity_id, entity in self.cognitive_map.spatial_entities.items():
            if (entity.entity_type == SpatialEntityType.OBJECT and
                object_type.lower() in entity.properties.get('type', '').lower()):
                entities.append({
                    'id': entity_id,
                    'position': entity.position,
                    'confidence': entity.confidence
                })

        return {
            'type': 'location',
            'entities': entities,
            'count': len(entities)
        }

    def _process_path_query(self, params):
        """Process path query"""
        start = params.get('start', '')
        end = params.get('end', '')

        # Find places that match the descriptions
        start_places = self.cognitive_map.get_place_by_type(start)
        end_places = self.cognitive_map.get_place_by_type(end)

        if not start_places or not end_places:
            return {'type': 'path', 'path': None, 'reason': 'Places not found'}

        # Use topological graph to find path
        start_id = start_places[0].id
        end_id = end_places[0].id

        path = self.cognitive_map.find_path(start_id, end_id)

        return {
            'type': 'path',
            'path': path,
            'start_place': start_id,
            'end_place': end_id
        }

    def _process_relationship_query(self, params):
        """Process relationship query"""
        query = params.get('query', '')

        # Extract relationship and object
        import re
        # Simple pattern matching for relationships
        near_pattern = r'(?:near|beside|next to|by)\s+(?:the\s+)?(\w+)'
        near_match = re.search(near_pattern, query)

        if near_match:
            target_object = near_match.group(1)

            # Find the target object
            target_entities = []
            for entity_id, entity in self.cognitive_map.spatial_entities.items():
                if target_object.lower() in entity.properties.get('type', '').lower():
                    target_entities.append(entity)

            if target_entities:
                target_pos = target_entities[0].position
                nearby_entities = []

                # Find entities within a certain distance
                for entity_id, entity in self.cognitive_map.spatial_entities.items():
                    if np.linalg.norm(entity.position - target_pos) < 2.0:  # 2m radius
                        nearby_entities.append({
                            'id': entity_id,
                            'type': entity.properties.get('type', 'unknown'),
                            'distance': np.linalg.norm(entity.position - target_pos)
                        })

                return {
                    'type': 'relationship',
                    'target': target_object,
                    'nearby_entities': nearby_entities
                }

        return {'type': 'relationship', 'result': 'Could not parse relationship query'}

    def _process_reasoning_query(self, params):
        """Process spatial reasoning query"""
        # This would implement more complex spatial reasoning
        # For example: "Can I sit in the kitchen?" or "Is the door accessible?"
        return {'type': 'reasoning', 'result': 'Spatial reasoning not implemented'}

    def _process_general_query(self, query):
        """Process general query that doesn't match templates"""
        # Use semantic search or other general methods
        return {'type': 'general', 'query': query, 'result': 'Query processed'}

    def answer_spatial_questions(self, questions: List[str]):
        """Answer multiple spatial questions"""
        answers = []

        for question in questions:
            answer = self.process_query(question)
            answers.append(answer)

        return answers

    def infer_spatial_knowledge(self):
        """Infer new spatial knowledge from existing map"""
        # This would implement spatial inference
        # For example: if A is near B and B is near C, then A might be near C
        # Or: if objects of type X are usually in rooms of type Y,
        # then a new instance of X might be in a Y room

        inferences = []

        # Example inference: transitivity of nearness
        for entity_id, entity in self.cognitive_map.spatial_entities.items():
            # Find entities that are near this entity
            nearby_entities = self._find_nearby_entities(entity.position, 3.0)

            for nearby_entity in nearby_entities:
                # Check if there are other entities near the nearby entity
                second_degree_nearby = self._find_nearby_entities(
                    nearby_entity.position, 3.0
                )

                # Add transitive relationships if they don't already exist
                for second_entity in second_degree_nearby:
                    if (second_entity.id != entity.id and
                        self._is_spatially_related(entity, second_entity)):
                        inferences.append({
                            'subject': entity.id,
                            'relationship': 'near',
                            'object': second_entity.id,
                            'confidence': 0.5  # Inferred relationship
                        })

        return inferences

    def _find_nearby_entities(self, position: np.ndarray, radius: float):
        """Find entities within a certain radius of a position"""
        nearby = []

        for entity_id, entity in self.cognitive_map.spatial_entities.items():
            distance = np.linalg.norm(entity.position - position)
            if distance <= radius:
                nearby.append(entity)

        return nearby

    def _is_spatially_related(self, entity1, entity2):
        """Check if two entities should be considered spatially related"""
        # Implement logic to determine if entities should be related
        # This could consider object types, spatial proximity, semantic similarity, etc.
        distance = np.linalg.norm(entity1.position - entity2.position)
        return distance < 5.0  # Within 5 meters
```

## Integration with Navigation and Planning

### Cognitive Navigation System

```python
class CognitiveNavigationSystem:
    """Navigation system that uses cognitive maps for intelligent path planning"""

    def __init__(self, robot_config, cognitive_map: CognitiveMap):
        self.robot_config = robot_config
        self.cognitive_map = cognitive_map
        self.topological_mapper = TopologicalMapper(robot_config)
        self.query_processor = SpatialQueryProcessor(cognitive_map)
        self.path_executor = PathExecutor(robot_config)
        self.context_manager = ContextManager()

    def navigate_with_cognitive_map(self, goal_description: str, context: Dict = None):
        """Navigate to goal using cognitive map understanding"""
        # Parse goal description to get specific location
        goal_info = self._interpret_goal_description(goal_description)

        if goal_info['type'] == 'object':
            # Find the specific object
            goal_positions = self._find_object_positions(goal_info['object_type'])
        elif goal_info['type'] == 'place':
            # Find places of the specified type
            places = self.cognitive_map.get_place_by_type(goal_info['place_type'])
            goal_positions = [place.position for place in places]
        else:
            # Use topological navigation
            path = self.cognitive_map.find_path(goal_info['start'], goal_info['end'])
            if path:
                return self._execute_topological_path(path)

        if not goal_positions:
            # If specific location not found, use semantic reasoning
            goal_positions = self._infer_goal_location(goal_description, context)

        # Plan path to nearest goal position
        current_pos = self._get_current_position()
        nearest_goal = min(goal_positions,
                          key=lambda pos: np.linalg.norm(pos[:2] - current_pos[:2]))

        # Execute navigation to goal
        return self.path_executor.navigate_to_position(nearest_goal)

    def _interpret_goal_description(self, description: str):
        """Interpret natural language goal description"""
        description_lower = description.lower()

        # Check if it's asking for an object
        objects = ['chair', 'table', 'bed', 'sofa', 'kitchen', 'refrigerator', 'microwave']
        for obj in objects:
            if obj in description_lower:
                return {'type': 'object', 'object_type': obj}

        # Check if it's asking for a place
        places = ['kitchen', 'bedroom', 'office', 'living room', 'bathroom', 'corridor']
        for place in places:
            if place in description_lower.replace(' ', '_'):
                return {'type': 'place', 'place_type': place.replace(' ', '_')}

        # Default to position-based navigation
        return {'type': 'position', 'description': description}

    def _find_object_positions(self, object_type: str):
        """Find positions of objects of specified type"""
        positions = []

        for entity_id, entity in self.cognitive_map.spatial_entities.items():
            if (entity.entity_type == SpatialEntityType.OBJECT and
                object_type.lower() in entity.properties.get('type', '').lower()):
                positions.append(entity.position)

        return positions

    def _infer_goal_location(self, description: str, context: Dict):
        """Infer likely location based on description and context"""
        # Use spatial query processor to find relevant information
        query_result = self.query_processor.process_query(description)

        if query_result['type'] == 'location' and query_result.get('entities'):
            return [entity['position'] for entity in query_result['entities']]

        # Use context to infer location
        if context:
            likely_places = self._get_likely_places_from_context(context, description)
            return [place.position for place in likely_places]

        # Default to home position or last known safe location
        return [np.array([0.0, 0.0, 0.0])]

    def _get_likely_places_from_context(self, context: Dict, description: str):
        """Get likely places based on current context"""
        current_place = context.get('current_place', 'unknown')

        # Define likely transitions
        likely_transitions = {
            'kitchen': ['refrigerator', 'microwave', 'counter'],
            'bedroom': ['bed', 'wardrobe', 'nightstand'],
            'office': ['desk', 'computer', 'chair'],
            'living_room': ['sofa', 'tv', 'coffee_table']
        }

        if current_place in likely_transitions:
            target_objects = likely_transitions[current_place]
            for obj in target_objects:
                if obj in description:
                    # Return places of current type
                    return self.cognitive_map.get_place_by_type(current_place)

        return []

    def _execute_topological_path(self, path):
        """Execute navigation along topological path"""
        results = []

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            # Get pose for next node
            next_pose = self.cognitive_map.topological_graph.nodes[next_node]['position']

            # Navigate to next node
            result = self.path_executor.navigate_to_position(next_pose)
            results.append(result)

            if not result['success']:
                # Handle navigation failure
                return self._handle_navigation_failure(current_node, next_node, results)

        return {'success': True, 'path_completed': True, 'results': results}

    def _handle_navigation_failure(self, current_node, next_node, previous_results):
        """Handle failure during topological navigation"""
        # Try alternative paths
        alternative_paths = list(nx.all_simple_paths(
            self.cognitive_map.topological_graph,
            current_node, next_node, cutoff=3  # Limit path length
        ))

        if len(alternative_paths) > 1:
            # Try second-best path
            alternative_path = alternative_paths[1]
            return self._execute_topological_path(alternative_path)
        else:
            # No alternatives, return failure
            return {
                'success': False,
                'reason': 'No alternative paths available',
                'partial_results': previous_results
            }

    def update_cognitive_map(self, sensor_data, robot_pose):
        """Update cognitive map with new sensor information"""
        # Update topological map
        self.topological_mapper.update_topological_map(robot_pose, sensor_data)

        # Update spatial entities
        self._update_spatial_entities(sensor_data, robot_pose)

        # Update spatial memory
        self.cognitive_map.spatial_memory.update_from_sensor_data(sensor_data, robot_pose)

    def _update_spatial_entities(self, sensor_data, robot_pose):
        """Update spatial entities based on sensor data"""
        # Process detected objects
        for obj_data in sensor_data.get('objects', []):
            obj_id = obj_data.get('id', f"obj_{len(self.cognitive_map.spatial_entities)}")

            # Convert relative position to global position
            obj_pos = robot_pose[:3] + obj_data['position']  # This is simplified

            entity = ObjectEntity(
                id=obj_id,
                entity_type=SpatialEntityType.OBJECT,
                position=obj_pos,
                orientation=np.array([0, 0, 0]),  # Simplified
                properties={'type': obj_data.get('type', 'unknown')},
                semantic_labels=[obj_data.get('type', 'unknown')],
                confidence=obj_data.get('confidence', 0.8),
                timestamp=datetime.now()
            )

            self.cognitive_map.add_entity(entity)

    def get_explainable_navigation(self, goal_description: str):
        """Get navigation explanation for human operators"""
        # Plan navigation and generate explanation
        goal_info = self._interpret_goal_description(goal_description)

        explanation = {
            'goal': goal_description,
            'goal_interpretation': goal_info,
            'navigation_strategy': 'cognitive_map_based',
            'steps': []
        }

        if goal_info['type'] == 'place':
            # Find path through topological map
            places = self.cognitive_map.get_place_by_type(goal_info['place_type'])
            if places:
                path = self.cognitive_map.find_path('current_location', places[0].id)
                if path:
                    explanation['steps'] = self._explain_path(path)

        return explanation

    def _explain_path(self, path):
        """Generate human-readable explanation of path"""
        steps = []

        for i, node_id in enumerate(path):
            node_data = self.cognitive_map.topological_graph.nodes[node_id]
            step_info = {
                'step': i + 1,
                'location': node_id,
                'place_type': node_data.get('place_type', 'unknown'),
                'action': f"Traverse to {node_data.get('place_type', 'location')}"
            }
            steps.append(step_info)

        return steps

class ContextManager:
    """Manage contextual information for cognitive navigation"""

    def __init__(self):
        self.current_context = {}
        self.context_history = []

    def update_context(self, new_context: Dict):
        """Update current context"""
        self.current_context.update(new_context)
        self.context_history.append({
            'context': self.current_context.copy(),
            'timestamp': datetime.now()
        })

    def get_context(self) -> Dict:
        """Get current context"""
        return self.current_context

    def get_temporal_context(self) -> Dict:
        """Get temporal aspects of context"""
        return {
            'time_of_day': datetime.now().strftime('%H:%M'),
            'day_of_week': datetime.now().strftime('%A'),
            'activity_pattern': self._infer_activity_pattern()
        }

    def _infer_activity_pattern(self):
        """Infer current activity pattern from context"""
        # This would analyze patterns in movement, object interactions, etc.
        return 'unknown'

class PathExecutor:
    """Execute navigation paths"""

    def __init__(self, robot_config):
        self.robot_config = robot_config

    def navigate_to_position(self, target_position):
        """Navigate to specific position"""
        # This would interface with actual navigation system
        # For simulation, return success
        return {
            'success': True,
            'target': target_position,
            'actual_path': [target_position],  # Simplified
            'time_taken': 0.0,
            'energy_consumed': 0.0
        }
```

## Practical Implementation Considerations

### Memory Management and Optimization

```python
class OptimizedCognitiveMap:
    """Memory-optimized cognitive map for long-term operation"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.spatial_entities = {}  # Dictionary with LRU cache
        self.topological_graph = nx.Graph()
        self.spatial_memory = SpatialMemory()

        # Memory limits
        self.max_entities = 1000
        self.max_age_days = 30
        self.min_confidence = 0.3

        # Spatial indexing for fast queries
        self.spatial_index = None
        self._rebuild_spatial_index()

    def _rebuild_spatial_index(self):
        """Rebuild spatial index for fast queries"""
        positions = []
        entity_ids = []

        for entity_id, entity in self.spatial_entities.items():
            positions.append(entity.position[:2])  # Only x,y for 2D indexing
            entity_ids.append(entity_id)

        if positions:
            self.spatial_index = cKDTree(positions)
            self._entity_ids = entity_ids
        else:
            self.spatial_index = None
            self._entity_ids = []

    def add_entity_with_cleanup(self, entity: SpatialEntity):
        """Add entity with automatic cleanup of old/low-confidence entities"""
        # Add new entity
        self.spatial_entities[entity.id] = entity

        # Periodic cleanup
        if len(self.spatial_entities) > self.max_entities * 0.8:
            self._cleanup_old_entities()

        # Rebuild spatial index if needed
        self._rebuild_spatial_index()

    def _cleanup_old_entities(self):
        """Remove old or low-confidence entities to free memory"""
        current_time = datetime.now()
        entities_to_remove = []

        for entity_id, entity in self.spatial_entities.items():
            # Remove low-confidence entities
            if entity.confidence < self.min_confidence:
                entities_to_remove.append(entity_id)
                continue

            # Remove old entities
            age_days = (current_time - entity.timestamp).days
            if age_days > self.max_age_days:
                entities_to_remove.append(entity_id)

        # Remove entities
        for entity_id in entities_to_remove:
            if entity_id in self.spatial_entities:
                del self.spatial_entities[entity_id]

                # Remove from topological graph if present
                if entity_id in self.topological_graph.nodes:
                    self.topological_graph.remove_node(entity_id)

    def find_nearby_entities(self, position: np.ndarray, radius: float):
        """Find entities within radius efficiently using spatial index"""
        if self.spatial_index is None:
            return []

        # Query spatial index
        x, y = position[:2]
        indices = self.spatial_index.query_ball_point([x, y], radius)

        # Get corresponding entity IDs
        nearby_entities = []
        for idx in indices:
            if idx < len(self._entity_ids):
                entity_id = self._entity_ids[idx]
                if entity_id in self.spatial_entities:
                    nearby_entities.append(self.spatial_entities[entity_id])

        return nearby_entities

    def get_memory_usage_report(self):
        """Get report on memory usage and optimization opportunities"""
        total_entities = len(self.spatial_entities)
        low_confidence_count = sum(
            1 for entity in self.spatial_entities.values()
            if entity.confidence < self.min_confidence
        )

        avg_age = 0
        if self.spatial_entities:
            current_time = datetime.now()
            total_age = sum(
                (current_time - entity.timestamp).days
                for entity in self.spatial_entities.values()
            )
            avg_age = total_age / len(self.spatial_entities)

        return {
            'total_entities': total_entities,
            'max_allowed': self.max_entities,
            'low_confidence_entities': low_confidence_count,
            'average_age_days': avg_age,
            'spatial_index_valid': self.spatial_index is not None
        }

    def optimize_for_task(self, task_requirements: Dict):
        """Optimize map for specific task requirements"""
        # Prioritize entities relevant to current task
        task_entities = task_requirements.get('required_entities', [])

        for entity_id, entity in self.spatial_entities.items():
            if any(task_entity in entity.properties.get('type', '')
                   for task_entity in task_entities):
                # Increase priority/importance of task-relevant entities
                entity.confidence = min(1.0, entity.confidence + 0.2)
```

## Assessment Questions

1. Explain the difference between metric maps and cognitive maps in robotics. What advantages do cognitive maps provide for humanoid robots?

2. Design a hierarchical spatial representation system that can handle multiple levels of abstraction for a humanoid robot navigating an office building.

3. Implement a semantic scene graph that can represent relationships between objects and infer new relationships based on spatial proximity.

4. Create a topological mapping algorithm that can identify and connect meaningful places in an environment.

5. Design a spatial query system that can answer natural language questions about the environment.

## Practice Exercises

1. **Map Building**: Implement a system that builds a cognitive map incrementally as a robot explores an unknown environment.

2. **Semantic Inference**: Create an algorithm that can infer semantic relationships between objects based on their spatial arrangement.

3. **Context-Aware Navigation**: Design a navigation system that changes its behavior based on the current context (time of day, task, environment).

4. **Memory Management**: Implement memory optimization techniques for long-term cognitive mapping in resource-constrained robots.

## Summary

Cognitive mapping and spatial reasoning enable humanoid robots to understand and navigate complex environments in meaningful ways. This chapter covered:

- Hierarchical spatial representation combining metric, semantic, and topological information
- Spatial memory systems that retain and update environmental knowledge over time
- Semantic scene graphs that represent objects and their relationships
- Topological mapping for efficient path planning between meaningful locations
- Spatial query processing systems that enable natural interaction with the map
- Integration of cognitive mapping with navigation and planning systems
- Memory optimization techniques for long-term operation

The combination of these techniques allows humanoid robots to build rich, meaningful representations of their environment that support intelligent navigation, interaction, and task execution in human-designed spaces.