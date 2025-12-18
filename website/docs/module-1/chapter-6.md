---
title: Chapter 6 - Python AI Agents with rclpy
description: Creating AI agents in Python that interact with humanoid robots using rclpy
sidebar_position: 6
---

# Chapter 6: Python AI Agents with rclpy

## Learning Objectives

After completing this chapter, you should be able to:
- Create Python-based AI agents that interact with humanoid robots
- Use rclpy to implement ROS 2 nodes for AI functionality
- Design agent architectures that integrate with robot systems

## Introduction

Python is one of the most popular languages for AI and robotics development. The rclpy library provides Python bindings for ROS 2, enabling the creation of sophisticated AI agents that can interact with humanoid robots. This chapter explores how to build intelligent agents that can perceive, reason, and act through robotic systems.

## Setting Up rclpy

Before creating AI agents, we need to understand the rclpy basics:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

# Initialize the ROS client library
rclpy.init()

# Create a node
node = Node('ai_agent')

# Create publishers and subscribers
publisher = node.create_publisher(String, 'robot_commands', 10)
subscriber = node.create_subscription(
    JointState, 'joint_states', lambda msg: print(msg), 10)

# Spin the node
rclpy.spin(node)

# Cleanup
node.destroy_node()
rclpy.shutdown()
```

## Basic AI Agent Structure

An AI agent typically follows the perception-action cycle:

### Simple AI Agent Implementation

```python
import rclpy
from rclpy.node import Node
import numpy as np

class SimpleAIAgent(Node):
    def __init__(self):
        super().__init__('simple_ai_agent')

        # Perception: Subscribe to sensor data
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)

        # Action: Publish commands to actuators
        self.command_pub = self.create_publisher(
            JointState, 'joint_commands', 10)

        # Internal state
        self.current_state = None
        self.goal_state = None

        # Timer for decision making
        self.timer = self.create_timer(0.1, self.decision_callback)

        self.get_logger().info('Simple AI Agent initialized')

    def joint_callback(self, msg):
        """Process joint state data"""
        self.current_state = {
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        }

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = {
            'orientation': msg.orientation,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration
        }

    def decision_callback(self):
        """Main decision-making loop"""
        if self.current_state is not None:
            action = self.decide_action()
            if action is not None:
                self.command_pub.publish(action)

    def decide_action(self):
        """Implement decision-making logic"""
        # Simple example: maintain upright position
        if self.imu_data:
            # Check if robot is upright
            orientation = self.imu_data['orientation']
            # Simplified logic for balance
            if abs(orientation.z) < 0.7:  # Not upright
                return self.generate_balance_command()
        return None

    def generate_balance_command(self):
        """Generate commands to maintain balance"""
        cmd = JointState()
        cmd.position = [0.0] * len(self.current_state['positions'])  # Return to neutral
        return cmd
```

## State-Based AI Agents

For more complex behaviors, we can implement state-based agents:

### State Machine Implementation

```python
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    WALKING = 2
    BALANCING = 3
    INTERACTING = 4

class StateBasedAgent(Node):
    def __init__(self):
        super().__init__('state_based_agent')

        # Initialize state
        self.current_state = RobotState.IDLE
        self.previous_state = None

        # Subscribe to sensor data
        self.sensor_sub = self.create_subscription(
            JointState, 'joint_states', self.sensor_callback, 10)

        # Create publishers for different types of commands
        self.walk_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)

        # State transition timer
        self.timer = self.create_timer(0.05, self.state_machine)

        self.get_logger().info('State-Based AI Agent initialized')

    def sensor_callback(self, msg):
        """Process sensor data and determine state transitions"""
        # Logic to determine if state should change
        if self.should_transition_to_balancing(msg):
            self.current_state = RobotState.BALANCING
        elif self.should_transition_to_walking(msg):
            self.current_state = RobotState.WALKING
        # etc.

    def state_machine(self):
        """Execute behavior based on current state"""
        if self.current_state != self.previous_state:
            self.on_state_enter()

        if self.current_state == RobotState.IDLE:
            self.execute_idle_behavior()
        elif self.current_state == RobotState.WALKING:
            self.execute_walking_behavior()
        elif self.current_state == RobotState.BALANCING:
            self.execute_balancing_behavior()
        elif self.current_state == RobotState.INTERACTING:
            self.execute_interacting_behavior()

        self.previous_state = self.current_state

    def execute_idle_behavior(self):
        """Behavior when robot is idle"""
        # Publish neutral joint positions
        cmd = JointState()
        cmd.position = [0.0] * 12  # Assuming 12 joints
        self.joint_pub.publish(cmd)

    def execute_walking_behavior(self):
        """Behavior when robot is walking"""
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward
        cmd.angular.z = 0.0  # No rotation
        self.walk_pub.publish(cmd)

    def execute_balancing_behavior(self):
        """Behavior when robot needs to balance"""
        # Implement balance control logic
        pass

    def execute_interacting_behavior(self):
        """Behavior when robot is interacting"""
        # Implement interaction logic
        pass

    def should_transition_to_balancing(self, sensor_data):
        """Determine if robot should transition to balancing state"""
        # Check if robot is falling
        return False  # Simplified for example

    def should_transition_to_walking(self, sensor_data):
        """Determine if robot should transition to walking state"""
        # Check if robot should walk
        return False  # Simplified for example

    def on_state_enter(self):
        """Called when entering a new state"""
        self.get_logger().info(f'State transition: {self.previous_state} -> {self.current_state}')
```

## Learning-Based AI Agents

We can also implement AI agents that learn from experience:

### Simple Learning Agent

```python
class LearningAgent(Node):
    def __init__(self):
        super().__init__('learning_agent')

        # Subscribe to sensor data
        self.sensor_sub = self.create_subscription(
            JointState, 'joint_states', self.sensor_callback, 10)

        # Subscribe to reward signals
        self.reward_sub = self.create_subscription(
            Float64, 'reward', self.reward_callback, 10)

        # Publisher for actions
        self.action_pub = self.create_publisher(
            JointState, 'joint_commands', 10)

        # Simple Q-learning parameters
        self.q_table = {}  # State-action value table
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate

        # Timer for learning updates
        self.timer = self.create_timer(0.1, self.learning_callback)

        self.get_logger().info('Learning AI Agent initialized')

    def sensor_callback(self, msg):
        """Process sensor data and select action"""
        state = self.extract_state(msg)
        action = self.select_action(state)

        # Execute action
        command = self.action_to_command(action)
        self.action_pub.publish(command)

        # Store for learning
        self.previous_state = state
        self.previous_action = action

    def reward_callback(self, msg):
        """Process reward signal"""
        self.current_reward = msg.data

    def extract_state(self, joint_state):
        """Extract discrete state from continuous sensor data"""
        # Discretize joint positions
        positions = joint_state.position
        state = tuple([int(p * 10) for p in positions])  # Simple discretization
        return state

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice([0, 1, 2, 3])  # Example actions
        else:
            # Exploit: best known action
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                # Initialize Q-values for this state
                self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0}
                return 0

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0, 1: 0, 2: 0, 3: 0}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def learning_callback(self):
        """Update learning based on previous action and reward"""
        if hasattr(self, 'previous_state') and hasattr(self, 'current_reward'):
            self.update_q_value(
                self.previous_state,
                self.previous_action,
                self.current_reward,
                self.extract_state_from_latest()
            )
            self.current_reward = 0  # Reset reward

    def extract_state_from_latest(self):
        """Extract current state (simplified)"""
        # In a real implementation, this would access the latest sensor data
        return (0, 0, 0)  # Placeholder

    def action_to_command(self, action):
        """Convert discrete action to robot command"""
        cmd = JointState()
        if action == 0:
            cmd.position = [0.0, 0.0, 0.0]  # Neutral
        elif action == 1:
            cmd.position = [0.1, 0.0, 0.0]  # Small adjustment
        # Add more actions as needed
        return cmd
```

## Integration with Robot Systems

AI agents need to be properly integrated with the robot's control systems:

### Complete Agent Example

```python
def main(args=None):
    rclpy.init(args=args)

    # Create the AI agent
    agent = StateBasedAgent()

    try:
        # Spin the node to process callbacks
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Interrupted by user')
    finally:
        # Clean up
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Python-based AI agents using rclpy provide a powerful way to implement intelligent behavior for humanoid robots. By understanding the perception-action cycle and implementing appropriate agent architectures, we can create sophisticated robotic systems that can learn, adapt, and interact with their environment.