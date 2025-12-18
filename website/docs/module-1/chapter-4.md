---
title: Chapter 4 - Distributed Systems - Nodes, Topics, Services
description: Understanding distributed systems concepts in ROS 2 for humanoid robots
sidebar_position: 4
---

# Chapter 4: Distributed Systems - Nodes, Topics, Services

## Learning Objectives

After completing this chapter, you should be able to:
- Understand the distributed nature of ROS 2 systems
- Implement nodes that communicate via topics and services
- Design distributed architectures for humanoid robot systems

## Introduction

ROS 2 is fundamentally a distributed computing framework that enables different components of a humanoid robot to communicate and coordinate effectively. This distributed architecture allows for:

- Scalable robot systems
- Fault tolerance
- Modularity in design
- Parallel processing capabilities

## Nodes in ROS 2

Nodes are the basic execution units in ROS 2. Each node typically represents a specific function or capability of the robot. Key aspects of nodes include:

- **Encapsulation**: Each node encapsulates specific functionality
- **Communication**: Nodes communicate with other nodes through topics, services, and actions
- **Lifecycle**: Nodes have a well-defined lifecycle that can be managed programmatically

### Creating Nodes

```python
import rclpy
from rclpy.node import Node

class HumanoidNode(Node):
    def __init__(self):
        super().__init__('humanoid_node')
        self.get_logger().info('Humanoid node initialized')
```

## Topics - Publish-Subscribe Communication

Topics implement a publish-subscribe communication pattern that is ideal for continuous data streams such as:

- Sensor readings
- Robot state information
- Camera images
- Joint positions

### Topic Publishers and Subscribers

```python
# Publisher example
publisher = self.create_publisher(JointState, 'joint_states', 10)

# Subscriber example
subscriber = self.create_subscription(
    JointState,
    'joint_states',
    self.joint_state_callback,
    10
)
```

## Services - Request-Response Communication

Services implement a request-response communication pattern that is ideal for:

- Configuration requests
- One-time commands
- Querying robot state
- Triggering specific actions

### Service Implementation

```python
# Service server
self.srv = self.create_service(SetJointPosition, 'set_joint_position', self.set_joint_position_callback)

# Service client
self.cli = self.create_client(SetJointPosition, 'set_joint_position')
```

## Practical Application

Here's a complete example of a distributed system for a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class HumanoidCommunicationNode(Node):
    def __init__(self):
        super().__init__('humanoid_communication')

        # Publisher for joint states
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Publisher for robot status
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)

        # Subscriber for commands
        self.command_subscriber = self.create_subscription(
            String,
            'robot_commands',
            self.command_callback,
            10
        )

        # Timer for periodic updates
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('Humanoid communication node started')

    def command_callback(self, msg):
        self.get_logger().info(f'Received command: {msg.data}')
        # Process the command and update robot state

    def timer_callback(self):
        # Publish joint states
        joint_msg = JointState()
        joint_msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        joint_msg.position = [0.0, 0.0, 0.0]  # Example positions
        self.joint_publisher.publish(joint_msg)

        # Publish status
        status_msg = String()
        status_msg.data = 'Operational'
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidCommunicationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Understanding distributed systems in ROS 2 is crucial for developing humanoid robots. The node-topic-service architecture provides a robust foundation for building complex, modular, and scalable robotic systems.