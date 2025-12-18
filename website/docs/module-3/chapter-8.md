---
title: "Chapter 8 - Learning and Adaptation in Humanoid Robots"
description: "Implementing machine learning and adaptation mechanisms for humanoid robots to learn from experience and adapt to new situations"
sidebar_label: "Chapter 8 - Learning and Adaptation in Humanoid Robots"
---

# Learning and Adaptation in Humanoid Robots

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement reinforcement learning algorithms for humanoid robot control
- Design adaptation mechanisms for handling environmental changes
- Apply imitation learning for skill acquisition from human demonstrations
- Create learning systems that improve robot performance over time
- Implement meta-learning for rapid adaptation to new tasks
- Design curiosity-driven exploration systems for autonomous learning
- Apply transfer learning techniques between different humanoid robots

## Introduction

Learning and adaptation are fundamental capabilities that enable humanoid robots to operate effectively in dynamic, unpredictable environments. Unlike traditional robots that rely on pre-programmed behaviors, humanoid robots must continuously learn from their experiences, adapt to new situations, and improve their performance over time.

The complexity of humanoid robots, with their many degrees of freedom and the need to maintain balance while performing tasks, makes learning-based approaches particularly valuable. These robots must learn to walk on different terrains, interact with various objects, navigate social situations, and adapt their behavior based on feedback from the environment and humans.

This chapter explores various learning and adaptation mechanisms specifically designed for humanoid robots, including reinforcement learning for control, imitation learning for skill acquisition, and meta-learning for rapid adaptation to new tasks.

## Reinforcement Learning for Humanoid Control

### Deep Reinforcement Learning Framework

Reinforcement learning is particularly well-suited for humanoid robot control due to the continuous action and state spaces involved:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque
import gym
from typing import List, Tuple, Dict, Any
import math

class HumanoidActorCritic(nn.Module):
    """Actor-Critic network for humanoid robot control"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(HumanoidActorCritic, self).__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are bounded to [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Action standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get action and value"""
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.log_std)

        # Create normal distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action for training"""
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.log_std)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value

class HumanoidReinforcementLearner:
    """Reinforcement learning system for humanoid robots"""

    def __init__(self, robot_config: Dict, state_dim: int, action_dim: int):
        self.robot_config = robot_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.actor_critic = HumanoidActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)

        # Experience replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 64

        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.95    # GAE parameter
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _ = self.actor_critic.get_action(state_tensor)

        return action.cpu().numpy()[0]

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self) -> float:
        """Update policy using collected experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([transition[0] for transition in batch]).to(self.device)
        actions = torch.FloatTensor([transition[1] for transition in batch]).to(self.device)
        rewards = torch.FloatTensor([transition[2] for transition in batch]).to(self.device)
        next_states = torch.FloatTensor([transition[3] for transition in batch]).to(self.device)
        dones = torch.BoolTensor([transition[4] for transition in batch]).to(self.device)

        # Compute advantages using GAE
        with torch.no_grad():
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze(-1)

            # Compute targets
            targets = rewards + self.gamma * next_values * (~dones)

            # Compute advantages
            _, values = self.actor_critic(states)
            values = values.squeeze(-1)
            advantages = targets - values

        # Get log probabilities and entropy
        log_probs, entropy, values = self.actor_critic.evaluate(states, actions)
        values = values.squeeze(-1)

        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, targets.detach())
        entropy_loss = -self.entropy_coef * entropy.mean()

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return total_loss.item()

    def train_policy(self, env, episodes: int = 1000):
        """Train the policy using environment interactions"""
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action = self.select_action(state)

                # Execute action in environment
                next_state, reward, done, _ = env.step(action)

                # Store transition
                self.store_transition(state, action, reward, next_state, done)

                # Update policy
                loss = self.update()

                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)

            if episode % 100 == 0:
                avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        return episode_rewards
```

### Humanoid-Specific Reward Design

The reward function is crucial for humanoid robot learning, especially for balance and locomotion:

```python
class HumanoidRewardDesigner:
    """Design reward functions for humanoid robot tasks"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.balance_weight = 1.0
        self.velocity_weight = 0.8
        self.energy_weight = 0.3
        self.smoothness_weight = 0.2

    def compute_locomotion_reward(self, robot_state: Dict,
                                target_velocity: float = 1.0) -> float:
        """Compute reward for locomotion tasks"""
        reward = 0.0

        # Balance reward - keep center of mass stable
        com_pos = robot_state.get('com_position', np.array([0, 0, 0.8]))
        com_vel = robot_state.get('com_velocity', np.array([0, 0, 0]))

        # Penalize large deviations from nominal CoM height
        height_error = abs(com_pos[2] - self.robot_config.get('nominal_com_height', 0.8))
        balance_reward = max(0, 1.0 - height_error)  # Higher reward for stable height

        # Penalize CoM velocity in Z direction
        z_velocity_penalty = max(0, abs(com_vel[2]) - 0.1)
        balance_reward -= z_velocity_penalty * 0.5

        reward += balance_reward * self.balance_weight

        # Velocity reward - encourage forward movement
        current_velocity = np.linalg.norm(com_vel[:2])
        velocity_reward = abs(current_velocity - target_velocity) * -0.5  # Encourage target velocity
        velocity_reward += max(0, current_velocity) * 0.1  # Small bonus for any forward movement
        reward += velocity_reward * self.velocity_weight

        # Energy efficiency reward - penalize excessive joint movements
        joint_velocities = robot_state.get('joint_velocities', np.zeros(28))  # Example: 28 DOF
        energy_penalty = np.mean(np.abs(joint_velocities)) * 0.01
        reward -= energy_penalty * self.energy_weight

        # Smoothness reward - penalize jerky movements
        joint_accelerations = robot_state.get('joint_accelerations', np.zeros(28))
        smoothness_penalty = np.mean(np.abs(joint_accelerations)) * 0.005
        reward -= smoothness_penalty * self.smoothness_weight

        # Bonus for maintaining balance over time
        if robot_state.get('is_balanced', True):
            reward += 0.1

        return reward

    def compute_manipulation_reward(self, robot_state: Dict,
                                  target_object_pos: np.ndarray,
                                  hand_pos: np.ndarray) -> float:
        """Compute reward for manipulation tasks"""
        reward = 0.0

        # Distance to target reward
        dist_to_target = np.linalg.norm(hand_pos - target_object_pos)
        distance_reward = max(0, 1.0 - dist_to_target)  # Higher reward when closer
        reward += distance_reward

        # Bonus for successful grasp
        if robot_state.get('is_grasping', False):
            reward += 5.0  # Large bonus for successful grasp

        # Balance maintenance during manipulation
        com_pos = robot_state.get('com_position', np.array([0, 0, 0.8]))
        height_error = abs(com_pos[2] - self.robot_config.get('nominal_com_height', 0.8))
        balance_maintenance = max(0, 1.0 - height_error * 2)  # Stronger penalty during manipulation
        reward += balance_maintenance * 0.5

        # Penalize excessive joint torques
        joint_torques = robot_state.get('joint_torques', np.zeros(28))
        torque_penalty = np.mean(np.abs(joint_torques)) * 0.001
        reward -= torque_penalty

        return reward

    def compute_social_interaction_reward(self, robot_state: Dict,
                                        human_feedback: Dict) -> float:
        """Compute reward for social interaction tasks"""
        reward = 0.0

        # Positive feedback reward
        if human_feedback.get('positive', False):
            reward += 2.0

        # Engagement reward
        if human_feedback.get('engaged', False):
            reward += 1.0

        # Appropriate distance maintenance
        distance_to_human = robot_state.get('distance_to_human', 2.0)
        comfortable_distance = robot_state.get('comfortable_distance', 1.0)
        distance_error = abs(distance_to_human - comfortable_distance)
        distance_reward = max(0, 1.0 - distance_error)
        reward += distance_reward

        # Appropriate behavior reward
        if robot_state.get('behavior_appropriate', False):
            reward += 1.5

        return reward

    def compute_adaptive_reward(self, robot_state: Dict, task_progress: float,
                              success_threshold: float = 0.8) -> float:
        """Compute adaptive reward that changes based on task progress"""
        base_reward = self.compute_locomotion_reward(robot_state)

        # Scale reward based on task progress to encourage steady improvement
        progress_factor = 1.0 + (task_progress / success_threshold)
        adaptive_reward = base_reward * progress_factor

        # Add exploration bonus for trying new strategies
        exploration_bonus = robot_state.get('exploration_factor', 0.0) * 0.1
        adaptive_reward += exploration_bonus

        return adaptive_reward
```

## Imitation Learning for Skill Acquisition

### Behavioral Cloning and GAIL

Imitation learning allows humanoid robots to learn skills by observing human demonstrations:

```python
class HumanoidImitationLearner:
    """Imitation learning system for humanoid robots"""

    def __init__(self, robot_config: Dict, state_dim: int, action_dim: int):
        self.robot_config = robot_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Expert policy network (for GAIL discriminator)
        self.expert_policy = HumanoidActorCritic(state_dim, action_dim).to(self.device)
        self.student_policy = HumanoidActorCritic(state_dim, action_dim).to(self.device)

        # Discriminator for GAIL
        self.discriminator = self._build_discriminator(state_dim, action_dim).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.student_policy.parameters(), lr=3e-4)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=3e-4)

        # Experience buffers
        self.expert_buffer = deque(maxlen=50000)
        self.student_buffer = deque(maxlen=50000)

        # Training parameters
        self.gail_weight = 0.8
        self.bc_weight = 0.2

    def _build_discriminator(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build discriminator network for GAIL"""
        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim + action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()  # Output probability of expert vs. student
                )

            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                return self.network(x)

        return Discriminator()

    def add_expert_demonstration(self, states: List[np.ndarray],
                               actions: List[np.ndarray]):
        """Add expert demonstration to buffer"""
        for state, action in zip(states, actions):
            self.expert_buffer.append((state, action))

    def behavioral_cloning_update(self) -> float:
        """Update policy using behavioral cloning (supervised learning)"""
        if len(self.expert_buffer) < 32:
            return 0.0

        # Sample batch from expert demonstrations
        batch = random.sample(self.expert_buffer, 32)
        states = torch.FloatTensor([s[0] for s in batch]).to(self.device)
        expert_actions = torch.FloatTensor([s[1] for s in batch]).to(self.device)

        # Get student actions
        student_action_means, _ = self.student_policy(states)

        # Compute behavioral cloning loss (MSE between expert and student actions)
        bc_loss = F.mse_loss(student_action_means, expert_actions)

        # Update student policy
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()

        return bc_loss.item()

    def gail_update(self) -> Tuple[float, float]:
        """Update policy using GAIL (Generative Adversarial Imitation Learning)"""
        if len(self.expert_buffer) < 32 or len(self.student_buffer) < 32:
            return 0.0, 0.0

        # Sample expert and student transitions
        expert_batch = random.sample(self.expert_buffer, 32)
        student_batch = random.sample(self.student_buffer, 32)

        expert_states = torch.FloatTensor([s[0] for s in expert_batch]).to(self.device)
        expert_actions = torch.FloatTensor([s[1] for s in expert_batch]).to(self.device)
        student_states = torch.FloatTensor([s[0] for s in student_batch]).to(self.device)
        student_actions = torch.FloatTensor([s[1] for s in student_batch]).to(self.device)

        # Update discriminator
        self.discriminator_optimizer.zero_grad()

        # Discriminator loss: expert = 1, student = 0
        expert_logits = self.discriminator(expert_states, expert_actions)
        student_logits = self.discriminator(student_states, student_actions)

        discriminator_loss = -(
            torch.log(expert_logits + 1e-8).mean() +
            torch.log(1 - student_logits + 1e-8).mean()
        )

        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Update policy using adversarial loss
        self.policy_optimizer.zero_grad()

        # Get current policy actions
        current_actions, _, _ = self.student_policy.get_action(student_states)

        # Compute adversarial loss (student wants to fool discriminator)
        adversarial_logits = self.discriminator(student_states, current_actions)
        adversarial_loss = -torch.log(adversarial_logits + 1e-8).mean()

        adversarial_loss.backward()
        self.policy_optimizer.step()

        return discriminator_loss.item(), adversarial_loss.item()

    def train_imitation(self, expert_demonstrations: List[Tuple[List, List]],
                       epochs: int = 1000):
        """Train using expert demonstrations"""
        for epoch in range(epochs):
            # Add demonstrations to buffer
            for states, actions in expert_demonstrations:
                self.add_expert_demonstration(states, actions)

            # Update using both BC and GAIL
            bc_loss = self.behavioral_cloning_update()
            disc_loss, adv_loss = self.gail_update()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: BC Loss: {bc_loss:.4f}, "
                      f"Disc Loss: {disc_loss:.4f}, Adv Loss: {adv_loss:.4f}")

    def generate_reward_from_discriminator(self, state: np.ndarray,
                                         action: np.ndarray) -> float:
        """Generate reward using discriminator confidence"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            discriminator_output = self.discriminator(state_tensor, action_tensor)
            # Use log-probability as reward (higher for expert-like behavior)
            reward = torch.log(discriminator_output + 1e-8).item()

        return reward

class HumanoidDemonstrationCollector:
    """Collect human demonstrations for imitation learning"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.demonstrations = []
        self.current_demonstration = {'states': [], 'actions': [], 'timestamps': []}

    def start_demonstration(self):
        """Start collecting a new demonstration"""
        self.current_demonstration = {'states': [], 'actions': [], 'timestamps': []}

    def record_step(self, state: np.ndarray, action: np.ndarray):
        """Record a single step of demonstration"""
        self.current_demonstration['states'].append(state.copy())
        self.current_demonstration['actions'].append(action.copy())
        self.current_demonstration['timestamps'].append(time.time())

    def end_demonstration(self, task_label: str = ""):
        """End current demonstration and save it"""
        self.current_demonstration['task_label'] = task_label
        self.current_demonstration['duration'] = (
            self.current_demonstration['timestamps'][-1] -
            self.current_demonstration['timestamps'][0]
        )

        self.demonstrations.append(self.current_demonstration.copy())
        print(f"Recorded demonstration: {task_label}, "
              f"Steps: {len(self.current_demonstration['states'])}, "
              f"Duration: {self.current_demonstration['duration']:.2f}s")

    def preprocess_demonstrations(self) -> List[Tuple[List, List]]:
        """Preprocess demonstrations for training"""
        processed_demos = []

        for demo in self.demonstrations:
            states = demo['states']
            actions = demo['actions']

            # Apply preprocessing: normalization, filtering, etc.
            processed_states = self._normalize_states(states)
            processed_actions = self._normalize_actions(actions)

            processed_demos.append((processed_states, processed_actions))

        return processed_demos

    def _normalize_states(self, states: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize state values"""
        if not states:
            return []

        states_array = np.array(states)
        # Normalize each dimension separately
        normalized = (states_array - states_array.mean(axis=0)) / (states_array.std(axis=0) + 1e-8)
        return [row for row in normalized]

    def _normalize_actions(self, actions: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize action values to [-1, 1] range"""
        if not actions:
            return []

        actions_array = np.array(actions)
        # Clamp to reasonable ranges and normalize
        clamped = np.clip(actions_array, -10.0, 10.0)  # Example: clamp to [-10, 10]
        normalized = np.tanh(clamped / 5.0)  # Normalize to [-1, 1]
        return [row for row in normalized]
```

## Adaptation and Transfer Learning

### Meta-Learning for Rapid Adaptation

Meta-learning enables humanoid robots to quickly adapt to new tasks or environments:

```python
class HumanoidMetaLearner:
    """Meta-learning system for rapid adaptation in humanoid robots"""

    def __init__(self, robot_config: Dict, state_dim: int, action_dim: int):
        self.robot_config = robot_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Meta-learner network (MAML-style)
        self.meta_policy = HumanoidActorCritic(state_dim, action_dim).to(self.device)
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=1e-3)

        # Task encoder network
        self.task_encoder = self._build_task_encoder().to(self.device)
        self.task_optimizer = optim.Adam(self.task_encoder.parameters(), lr=1e-3)

        # Memory for storing task-specific parameters
        self.task_memory = {}

        # Training parameters
        self.meta_lr = 0.01  # Inner loop learning rate
        self.meta_batch_size = 10
        self.adaptation_steps = 5

    def _build_task_encoder(self) -> nn.Module:
        """Build network to encode task information"""
        class TaskEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(self.state_dim + self.action_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),  # Task embedding
                )

            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                return self.encoder(x)

        return TaskEncoder()

    def sample_tasks(self, num_tasks: int) -> List[Dict]:
        """Sample different tasks for meta-training"""
        tasks = []

        for i in range(num_tasks):
            task_type = random.choice(['walk', 'balance', 'manipulate', 'navigate'])

            if task_type == 'walk':
                task = {
                    'type': 'locomotion',
                    'terrain_type': random.choice(['flat', 'sloped', 'uneven']),
                    'target_velocity': random.uniform(0.5, 2.0),
                    'obstacles': random.randint(0, 5)
                }
            elif task_type == 'balance':
                task = {
                    'type': 'balance',
                    'disturbance_type': random.choice(['push', 'sloped', 'narrow']),
                    'disturbance_magnitude': random.uniform(0.1, 0.5)
                }
            elif task_type == 'manipulate':
                task = {
                    'type': 'manipulation',
                    'object_type': random.choice(['light', 'heavy', 'fragile']),
                    'target_position': np.random.uniform(-1, 1, 3),
                    'precision_required': random.choice([True, False])
                }
            else:  # navigate
                task = {
                    'type': 'navigation',
                    'environment_complexity': random.uniform(0.1, 1.0),
                    'social_context': random.choice(['none', 'single', 'crowd']),
                    'dynamic_obstacles': random.randint(0, 3)
                }

            tasks.append(task)

        return tasks

    def inner_loop_update(self, task_env, policy_params: Dict,
                         adaptation_data: List[Tuple]) -> Dict:
        """Perform inner loop adaptation for a specific task"""
        # Create temporary policy with current parameters
        temp_policy = HumanoidActorCritic(self.state_dim, self.action_dim).to(self.device)
        temp_policy.load_state_dict(policy_params)

        # Adapt using adaptation data
        optimizer = optim.Adam(temp_policy.parameters(), lr=self.meta_lr)

        for _ in range(self.adaptation_steps):
            if adaptation_data:
                # Sample data for adaptation
                batch = random.sample(adaptation_data, min(len(adaptation_data), 32))

                states = torch.FloatTensor([d[0] for d in batch]).to(self.device)
                actions = torch.FloatTensor([d[1] for d in batch]).to(self.device)
                rewards = torch.FloatTensor([d[2] for d in batch]).to(self.device)

                # Compute loss and update
                log_probs, entropy, values = temp_policy.evaluate(states, actions)
                advantages = rewards - values.squeeze(-1)
                loss = -(log_probs * advantages.detach()).mean() - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return temp_policy.state_dict()

    def meta_update(self, tasks: List[Dict], num_episodes: int = 10):
        """Perform meta-learning update"""
        meta_loss = 0.0

        for task in tasks:
            # Sample adaptation and evaluation data for this task
            adaptation_data = self._collect_adaptation_data(task, num_episodes // 2)
            evaluation_data = self._collect_adaptation_data(task, num_episodes // 2)

            # Get current policy parameters
            current_params = self.meta_policy.state_dict()

            # Adapt to task (inner loop)
            adapted_params = self.inner_loop_update(
                task, current_params, adaptation_data
            )

            # Evaluate adapted policy (outer loop)
            temp_policy = HumanoidActorCritic(self.state_dim, self.action_dim).to(self.device)
            temp_policy.load_state_dict(adapted_params)

            # Compute evaluation loss
            eval_loss = self._compute_evaluation_loss(temp_policy, evaluation_data)
            meta_loss += eval_loss

        # Update meta-policy
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item() / len(tasks)

    def _collect_adaptation_data(self, task: Dict, num_episodes: int) -> List[Tuple]:
        """Collect adaptation data for a specific task"""
        # This would interact with the environment to collect data
        # For now, return simulated data
        data = []

        for _ in range(num_episodes):
            # Simulate episode collection
            episode_data = [
                (np.random.randn(self.state_dim),
                 np.random.randn(self.action_dim),
                 np.random.randn())  # state, action, reward
                for _ in range(100)  # 100 steps per episode
            ]
            data.extend(episode_data)

        return data

    def _compute_evaluation_loss(self, policy: HumanoidActorCritic,
                               evaluation_data: List[Tuple]) -> torch.Tensor:
        """Compute evaluation loss for adapted policy"""
        if not evaluation_data:
            return torch.tensor(0.0)

        states = torch.FloatTensor([d[0] for d in evaluation_data]).to(self.device)
        actions = torch.FloatTensor([d[1] for d in evaluation_data]).to(self.device)
        rewards = torch.FloatTensor([d[2] for d in evaluation_data]).to(self.device)

        log_probs, entropy, values = policy.evaluate(states, actions)
        advantages = rewards - values.squeeze(-1)
        loss = -(log_probs * advantages.detach()).mean() - 0.01 * entropy.mean()

        return loss

    def adapt_to_new_task(self, new_task: Dict, initial_data: List[Tuple]) -> Dict:
        """Adapt to a new task using meta-learning"""
        # Get meta-policy parameters
        meta_params = self.meta_policy.state_dict()

        # Adapt using initial data
        adapted_params = self.inner_loop_update(
            new_task, meta_params, initial_data
        )

        # Return adapted policy parameters
        return adapted_params

    def train_meta_policy(self, num_iterations: int = 1000):
        """Train the meta-policy"""
        for iteration in range(num_iterations):
            # Sample tasks for this iteration
            tasks = self.sample_tasks(self.meta_batch_size)

            # Perform meta-update
            loss = self.meta_update(tasks)

            if iteration % 100 == 0:
                print(f"Meta-training iteration {iteration}, Loss: {loss:.4f}")

class HumanoidTransferLearner:
    """Transfer learning system for humanoid robots"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.skill_library = {}  # Store learned skills
        self.transfer_mappings = {}  # Map between different robot configurations

    def extract_skill_features(self, policy: nn.Module) -> np.ndarray:
        """Extract features representing a learned skill"""
        # Extract key parameters that represent the skill
        features = []

        for param in policy.parameters():
            # Use statistical features of parameters
            features.extend([
                param.mean().item(),
                param.std().item(),
                param.max().item(),
                param.min().item()
            ])

        return np.array(features)

    def store_skill(self, skill_name: str, policy: nn.Module,
                   task_description: str, performance_metrics: Dict):
        """Store a learned skill in the library"""
        skill_features = self.extract_skill_features(policy)

        self.skill_library[skill_name] = {
            'policy': policy.state_dict(),  # Store parameters
            'features': skill_features,
            'task_description': task_description,
            'performance_metrics': performance_metrics,
            'timestamp': time.time()
        }

    def find_similar_skills(self, target_task: str,
                          similarity_threshold: float = 0.7) -> List[str]:
        """Find similar skills for transfer"""
        target_features = self._encode_task_description(target_task)
        similar_skills = []

        for skill_name, skill_data in self.skill_library.items():
            similarity = self._compute_similarity(
                target_features, skill_data['features']
            )

            if similarity > similarity_threshold:
                similar_skills.append((skill_name, similarity))

        # Sort by similarity
        similar_skills.sort(key=lambda x: x[1], reverse=True)
        return [skill[0] for skill in similar_skills]

    def transfer_skill(self, source_skill: str, target_robot_config: Dict) -> nn.Module:
        """Transfer a skill to a different robot configuration"""
        if source_skill not in self.skill_library:
            raise ValueError(f"Skill {source_skill} not found in library")

        source_policy_params = self.skill_library[source_skill]['policy']

        # Create target policy with appropriate architecture
        target_policy = HumanoidActorCritic(
            target_robot_config['state_dim'],
            target_robot_config['action_dim']
        )

        # Map parameters from source to target
        mapped_params = self._map_parameters(
            source_policy_params,
            target_robot_config,
            self.robot_config
        )

        target_policy.load_state_dict(mapped_params)
        return target_policy

    def _encode_task_description(self, task_description: str) -> np.ndarray:
        """Encode task description as features"""
        # Simple encoding based on keywords
        keywords = ['locomotion', 'balance', 'manipulation', 'navigation', 'social']
        features = np.zeros(len(keywords))

        for i, keyword in enumerate(keywords):
            if keyword in task_description.lower():
                features[i] = 1.0

        return features

    def _compute_similarity(self, features1: np.ndarray,
                          features2: np.ndarray) -> float:
        """Compute similarity between feature vectors"""
        # Use cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _map_parameters(self, source_params: Dict,
                       target_config: Dict, source_config: Dict) -> Dict:
        """Map parameters from source to target robot"""
        # This is a simplified parameter mapping
        # In practice, this would involve more sophisticated techniques
        target_params = {}

        for key, value in source_params.items():
            if key in target_params:
                # Direct mapping if keys match
                if target_params[key].shape == value.shape:
                    target_params[key] = value
                else:
                    # Need to adapt shape - this is complex and depends on the layer
                    target_params[key] = self._adapt_parameter_shape(
                        value, target_params[key].shape
                    )
            else:
                # Use the source parameter if target doesn't have it
                target_params[key] = value

        return target_params

    def _adapt_parameter_shape(self, param: torch.Tensor,
                             target_shape: tuple) -> torch.Tensor:
        """Adapt parameter tensor to target shape"""
        if param.shape == target_shape:
            return param

        # For now, return a tensor with target shape filled with source values
        # This is a very simplified approach
        if len(target_shape) == len(param.shape):
            # Same number of dimensions, try to expand or truncate
            result = torch.zeros(target_shape)
            min_sizes = [min(s1, s2) for s1, s2 in zip(param.shape, target_shape)]

            slices = tuple(slice(0, size) for size in min_sizes)
            result[slices] = param[slices]

            return result

        # For different number of dimensions, this requires more complex logic
        return torch.zeros(target_shape)
```

## Curiosity-Driven Exploration

### Intrinsic Motivation Systems

Curiosity-driven exploration helps humanoid robots discover new skills and behaviors:

```python
class CuriosityDrivenExplorer:
    """Curiosity-driven exploration system for humanoid robots"""

    def __init__(self, robot_config: Dict, state_dim: int, action_dim: int):
        self.robot_config = robot_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Forward and inverse models for curiosity
        self.forward_model = self._build_forward_model().to(self.device)
        self.inverse_model = self._build_inverse_model().to(self.device)

        # Optimizers
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=1e-3)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=1e-3)

        # Experience buffer for training models
        self.experience_buffer = deque(maxlen=10000)

        # Exploration parameters
        self.exploration_coeff = 0.1
        self.episode_rewards = []
        self.exploration_memory = {}  # Track visited states

    def _build_forward_model(self) -> nn.Module:
        """Build forward model to predict next state"""
        class ForwardModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(self.state_dim + self.action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.state_dim)
                )

            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                return self.network(x)

        return ForwardModel()

    def _build_inverse_model(self) -> nn.Module:
        """Build inverse model to predict action from state transitions"""
        class InverseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(2 * self.state_dim, 256),  # current + next state
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.action_dim)
                )

            def forward(self, current_state, next_state):
                x = torch.cat([current_state, next_state], dim=-1)
                return self.network(x)

        return InverseModel()

    def compute_intrinsic_reward(self, state: np.ndarray, action: np.ndarray,
                               next_state: np.ndarray) -> float:
        """Compute intrinsic reward based on prediction error"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Predict next state using forward model
        predicted_next_state = self.forward_model(state_tensor, action_tensor)

        # Compute prediction error (intrinsic reward)
        prediction_error = F.mse_loss(predicted_next_state, next_state_tensor, reduction='none')
        intrinsic_reward = prediction_error.mean().item()

        # Add novelty bonus based on state visitation
        state_key = self._hash_state(state)
        visitation_count = self.exploration_memory.get(state_key, 0)
        novelty_bonus = 1.0 / (1.0 + visitation_count)  # Less reward for frequently visited states

        return intrinsic_reward * self.exploration_coeff * novelty_bonus

    def _hash_state(self, state: np.ndarray) -> str:
        """Create a hashable key for state"""
        # Discretize state for counting
        discretized = (state * 10).astype(int)  # 0.1 resolution
        return str(discretized.tolist())

    def update_models(self, state: np.ndarray, action: np.ndarray,
                     next_state: np.ndarray):
        """Update forward and inverse models"""
        # Add to experience buffer
        self.experience_buffer.append((state, action, next_state))

        if len(self.experience_buffer) < 32:
            return

        # Sample batch for training
        batch = random.sample(self.experience_buffer, 32)

        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.FloatTensor([b[1] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b[2] for b in batch]).to(self.device)

        # Update forward model
        self.forward_optimizer.zero_grad()
        predicted_next_states = self.forward_model(states, actions)
        forward_loss = F.mse_loss(predicted_next_states, next_states)
        forward_loss.backward()
        self.forward_optimizer.step()

        # Update inverse model
        self.inverse_optimizer.zero_grad()
        predicted_actions = self.inverse_model(states, next_states)
        inverse_loss = F.mse_loss(predicted_actions, actions)
        inverse_loss.backward()
        self.inverse_optimizer.step()

        # Update exploration memory
        for state in batch:
            state_key = self._hash_state(state[0])
            self.exploration_memory[state_key] = self.exploration_memory.get(state_key, 0) + 1

    def get_exploration_action(self, state: np.ndarray,
                             policy_action: np.ndarray,
                             exploration_level: float = 0.3) -> np.ndarray:
        """Get action that balances policy and exploration"""
        if random.random() < exploration_level:
            # Add exploration to policy action
            exploration_noise = np.random.normal(0, 0.1, size=policy_action.shape)
            exploration_action = policy_action + exploration_noise

            # Clip to valid range
            return np.clip(exploration_action, -1.0, 1.0)
        else:
            return policy_action

class HumanoidCuriositySystem:
    """Integrated curiosity system for humanoid robot exploration"""

    def __init__(self, robot_config: Dict, state_dim: int, action_dim: int):
        self.robot_config = robot_config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Curiosity explorer
        self.curiosity_explorer = CuriosityDrivenExplorer(
            robot_config, state_dim, action_dim
        )

        # Skill discovery system
        self.skill_discoverer = SkillDiscoverer(robot_config)

        # Exploration strategy
        self.exploration_strategy = "intrinsic_motivation"
        self.exploration_phase = True
        self.skill_threshold = 0.8  # Threshold for skill acquisition

    def explore_environment(self, env, steps: int = 10000):
        """Explore environment using curiosity-driven approach"""
        state = env.reset()
        total_reward = 0

        for step in range(steps):
            # Get policy action (random for pure exploration)
            policy_action = np.random.uniform(-1, 1, self.action_dim)

            # Get exploration-enhanced action
            action = self.curiosity_explorer.get_exploration_action(
                state, policy_action
            )

            # Execute action
            next_state, extrinsic_reward, done, info = env.step(action)

            # Compute intrinsic reward
            intrinsic_reward = self.curiosity_explorer.compute_intrinsic_reward(
                state, action, next_state
            )

            # Total reward is sum of extrinsic and intrinsic
            total_step_reward = extrinsic_reward + intrinsic_reward
            total_reward += total_step_reward

            # Update curiosity models
            self.curiosity_explorer.update_models(state, action, next_state)

            # Check for skill opportunities
            skill_opportunity = self.skill_discoverer.detect_skill_opportunity(
                state, action, next_state, intrinsic_reward
            )

            if skill_opportunity:
                self.skill_discoverer.record_skill_attempt(skill_opportunity)

            state = next_state

            if done:
                state = env.reset()

        return total_reward

    def balance_exploration_exploitation(self, current_performance: float,
                                       exploration_budget: float = 0.2) -> bool:
        """Determine whether to explore or exploit based on performance"""
        if current_performance < self.skill_threshold:
            # Need to explore more to improve performance
            return True
        elif random.random() < exploration_budget:
            # Continue exploration with some probability
            return True
        else:
            # Exploit learned skills
            return False

class SkillDiscoverer:
    """Discover and learn new skills during exploration"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.skill_candidates = []
        self.skill_attempts = []
        self.skill_success_threshold = 0.7

    def detect_skill_opportunity(self, state: np.ndarray, action: np.ndarray,
                               next_state: np.ndarray,
                               curiosity_reward: float) -> Dict:
        """Detect potential skill learning opportunities"""
        # Look for patterns in state transitions that might represent skills
        state_change = np.linalg.norm(next_state - state)
        action_magnitude = np.linalg.norm(action)

        # High curiosity reward + significant state change might indicate a skill
        if curiosity_reward > 0.1 and state_change > 0.5 and action_magnitude > 0.3:
            return {
                'state_change': state_change,
                'action_magnitude': action_magnitude,
                'curiosity_reward': curiosity_reward,
                'potential_skill': True
            }

        return {}

    def record_skill_attempt(self, skill_info: Dict):
        """Record a potential skill attempt"""
        self.skill_candidates.append(skill_info)

    def evaluate_skill_candidate(self, candidate: Dict) -> float:
        """Evaluate if a candidate represents a learnable skill"""
        # Score based on various factors
        score = 0.0

        # Higher curiosity reward indicates more novel/interesting behavior
        score += candidate.get('curiosity_reward', 0) * 0.4

        # Significant state change indicates meaningful action
        score += min(candidate.get('state_change', 0), 1.0) * 0.3

        # Moderate action magnitude (not too random)
        action_score = max(0, 1 - abs(candidate.get('action_magnitude', 0) - 0.5))
        score += action_score * 0.3

        return score

    def extract_skills(self) -> List[Dict]:
        """Extract learned skills from exploration data"""
        skills = []

        for candidate in self.skill_candidates:
            score = self.evaluate_skill_candidate(candidate)

            if score > self.skill_success_threshold:
                skills.append({
                    'type': 'discovered_skill',
                    'score': score,
                    'characteristics': candidate
                })

        return skills
```

## Practical Implementation Considerations

### Safe Learning Framework

```python
class SafeLearningFramework:
    """Framework for safe learning in humanoid robots"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.safety_constraints = self._define_safety_constraints()
        self.fallback_behaviors = self._define_fallback_behaviors()
        self.risk_assessment = RiskAssessmentSystem(robot_config)

    def _define_safety_constraints(self) -> Dict:
        """Define safety constraints for learning"""
        return {
            'joint_limits': {
                'position': [-3.14, 3.14],  # Example: +/- pi
                'velocity': [-10.0, 10.0],
                'torque': [-50.0, 50.0]
            },
            'balance_constraints': {
                'com_height_min': 0.3,
                'com_height_max': 1.2,
                'zmp_limits': [-0.15, 0.15],  # meters
                'angular_velocity_max': 2.0   # rad/s
            },
            'collision_constraints': {
                'self_collision': True,
                'environment_collision': True,
                'minimum_distance': 0.1  # meters
            }
        }

    def _define_fallback_behaviors(self) -> Dict:
        """Define safe fallback behaviors"""
        return {
            'balance_recovery': self._balance_recovery_behavior,
            'safe_stop': self._safe_stop_behavior,
            'emergency_shutdown': self._emergency_shutdown_behavior
        }

    def _balance_recovery_behavior(self, robot_state: Dict) -> np.ndarray:
        """Recover balance when in danger"""
        # Simple balance recovery: move CoM over support polygon
        com_pos = robot_state.get('com_position', np.array([0, 0, 0.8]))
        support_polygon_center = robot_state.get('support_polygon_center', np.array([0, 0]))

        # Generate action to move CoM toward support center
        recovery_vector = support_polygon_center - com_pos[:2]
        recovery_action = np.zeros(self.robot_config['action_dim'])
        recovery_action[:2] = np.clip(recovery_vector, -0.1, 0.1)  # Small adjustment

        return recovery_action

    def _safe_stop_behavior(self, robot_state: Dict) -> np.ndarray:
        """Stop robot safely"""
        # Zero out velocity commands
        return np.zeros(self.robot_config['action_dim'])

    def _emergency_shutdown_behavior(self, robot_state: Dict) -> np.ndarray:
        """Emergency shutdown"""
        # This would interface with safety systems
        print("EMERGENCY SHUTDOWN TRIGGERED")
        return np.zeros(self.robot_config['action_dim'])

    def check_safety(self, state: np.ndarray, action: np.ndarray) -> Tuple[bool, str]:
        """Check if action is safe in current state"""
        # Check joint limits
        if not self._check_joint_limits(state, action):
            return False, "Joint limit violation"

        # Check balance constraints
        if not self._check_balance_constraints(state):
            return False, "Balance constraint violation"

        # Check for potential collisions
        if not self._check_collision_risk(state, action):
            return False, "Collision risk detected"

        return True, "Safe"

    def _check_joint_limits(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check if action would violate joint limits"""
        # This would check against actual robot joint limits
        # For simulation, assume action is within reasonable bounds
        return np.all(np.abs(action) <= 2.0)  # Reasonable action bounds

    def _check_balance_constraints(self, state: np.ndarray) -> bool:
        """Check if state violates balance constraints"""
        # Extract CoM information from state
        # This is simplified - actual implementation would extract from full state
        if len(state) >= 3:
            com_height = state[2]  # Assume Z component is CoM height
            min_height = self.safety_constraints['balance_constraints']['com_height_min']
            max_height = self.safety_constraints['balance_constraints']['com_height_max']

            if not (min_height <= com_height <= max_height):
                return False

        return True

    def _check_collision_risk(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check for collision risk"""
        # Simplified collision check
        # In practice, this would use environment mapping and prediction
        return True  # Assume no collision for now

    def safe_learn_step(self, state: np.ndarray, action: np.ndarray,
                       environment_step_func) -> Tuple[np.ndarray, float, bool, Dict]:
        """Perform a safe learning step"""
        # Check if action is safe
        is_safe, safety_msg = self.check_safety(state, action)

        if not is_safe:
            # Use fallback behavior
            fallback_action = self.fallback_behaviors['safe_stop'](state)
            next_state, reward, done, info = environment_step_func(fallback_action)

            # Add safety penalty
            reward -= 10.0  # Penalty for unsafe action
            info['safety_violation'] = safety_msg
        else:
            # Execute normal action
            next_state, reward, done, info = environment_step_func(action)

            # Add safety bonus if maintained
            info['safety_maintained'] = True

        return next_state, reward, done, info

class RiskAssessmentSystem:
    """System for assessing and managing learning risks"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.risk_history = []
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

    def assess_risk(self, state: np.ndarray, action: np.ndarray) -> float:
        """Assess risk level of action in state"""
        risk_score = 0.0

        # Balance risk
        if len(state) >= 6:  # Assume state includes pose and velocity
            com_height = state[2]
            angular_velocity = np.linalg.norm(state[3:6])  # Angular velocity components
            balance_risk = abs(0.8 - com_height) + angular_velocity * 0.1
            risk_score += balance_risk * 0.4

        # Action magnitude risk
        action_magnitude = np.linalg.norm(action)
        action_risk = min(action_magnitude / 2.0, 1.0)  # Normalize to [0,1]
        risk_score += action_risk * 0.3

        # Velocity risk
        if len(state) >= 9:  # Assume state includes linear velocity
            linear_velocity = np.linalg.norm(state[6:9])
            velocity_risk = min(linear_velocity / 1.0, 1.0)  # Normalize
            risk_score += velocity_risk * 0.3

        return min(risk_score, 1.0)  # Clamp to [0,1]

    def adjust_exploration_based_on_risk(self, current_risk: float,
                                       base_exploration: float) -> float:
        """Adjust exploration rate based on current risk"""
        if current_risk > self.risk_thresholds['high']:
            return base_exploration * 0.1  # Very low exploration when high risk
        elif current_risk > self.risk_thresholds['medium']:
            return base_exploration * 0.5  # Reduce exploration for medium risk
        elif current_risk > self.risk_thresholds['low']:
            return base_exploration * 0.8  # Slight reduction for low risk
        else:
            return base_exploration  # Normal exploration when safe

    def record_risk_incident(self, state: np.ndarray, action: np.ndarray,
                           risk_level: str, consequence: str):
        """Record risk incident for learning"""
        incident = {
            'timestamp': time.time(),
            'state': state.copy(),
            'action': action.copy(),
            'risk_level': risk_level,
            'consequence': consequence,
            'assessed_risk': self.assess_risk(state, action)
        }
        self.risk_history.append(incident)

class HumanoidLearningManager:
    """Main manager for learning and adaptation in humanoid robots"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.rl_learner = HumanoidReinforcementLearner(
            robot_config,
            robot_config['state_dim'],
            robot_config['action_dim']
        )
        self.imitation_learner = HumanoidImitationLearner(
            robot_config,
            robot_config['state_dim'],
            robot_config['action_dim']
        )
        self.meta_learner = HumanoidMetaLearner(
            robot_config,
            robot_config['state_dim'],
            robot_config['action_dim']
        )
        self.transfer_learner = HumanoidTransferLearner(robot_config)
        self.curiosity_system = HumanoidCuriositySystem(
            robot_config,
            robot_config['state_dim'],
            robot_config['action_dim']
        )
        self.safety_framework = SafeLearningFramework(robot_config)

        self.learning_modes = {
            'reinforcement': self.rl_learner,
            'imitation': self.imitation_learner,
            'meta': self.meta_learner,
            'transfer': self.transfer_learner,
            'curiosity': self.curiosity_system
        }

    def select_learning_mode(self, task_context: Dict) -> str:
        """Select appropriate learning mode based on context"""
        task_type = task_context.get('type', 'general')
        prior_experience = task_context.get('prior_experience', 0)
        available_demonstrations = task_context.get('demonstrations', 0)

        if available_demonstrations > 0:
            return 'imitation'
        elif prior_experience < 0.1:  # New task
            if task_type in ['locomotion', 'balance']:
                return 'curiosity'  # Explore first
            else:
                return 'meta'  # Use meta-learning
        elif prior_experience < 0.7:  # Learning task
            return 'reinforcement'
        else:  # Mastered task
            return 'transfer'  # Apply to new context

    def learn_task(self, task_description: str, task_context: Dict,
                  training_data: Dict = None) -> Dict:
        """Learn a new task using appropriate learning method"""
        learning_mode = self.select_learning_mode(task_context)

        print(f"Selected learning mode: {learning_mode} for task: {task_description}")

        if learning_mode == 'reinforcement':
            # Use reinforcement learning
            result = self._learn_via_reinforcement(task_description, task_context)
        elif learning_mode == 'imitation':
            # Use imitation learning if demonstrations available
            if training_data and 'demonstrations' in training_data:
                self.imitation_learner.train_imitation(training_data['demonstrations'])
                result = {'success': True, 'method': 'imitation', 'performance': 0.8}
            else:
                result = {'success': False, 'reason': 'No demonstrations available'}
        elif learning_mode == 'meta':
            # Use meta-learning
            self.meta_learner.train_meta_policy()
            result = {'success': True, 'method': 'meta', 'performance': 0.7}
        elif learning_mode == 'curiosity':
            # Use curiosity-driven exploration
            result = self._learn_via_curiosity(task_description, task_context)
        else:
            result = {'success': False, 'reason': 'Unknown learning mode'}

        return result

    def _learn_via_reinforcement(self, task_description: str,
                               task_context: Dict) -> Dict:
        """Learn via reinforcement learning"""
        # This would connect to actual environment
        # For now, return simulated result
        return {
            'success': True,
            'method': 'reinforcement',
            'performance': random.uniform(0.6, 0.9),
            'episodes_trained': 1000
        }

    def _learn_via_curiosity(self, task_description: str,
                           task_context: Dict) -> Dict:
        """Learn via curiosity-driven exploration"""
        # This would use actual environment
        # For now, return simulated result
        return {
            'success': True,
            'method': 'curiosity',
            'discovered_skills': ['basic_locomotion', 'balance_maintenance'],
            'performance': random.uniform(0.5, 0.8)
        }

    def adapt_to_new_condition(self, current_policy: nn.Module,
                             new_condition: Dict) -> nn.Module:
        """Adapt existing policy to new conditions"""
        # Use meta-learning for rapid adaptation
        adaptation_data = self._collect_adaptation_data(new_condition)

        if adaptation_data:
            adapted_policy = self.meta_learner.adapt_to_new_task(
                new_condition, adaptation_data
            )

            # Create new policy with adapted parameters
            new_policy = HumanoidActorCritic(
                self.robot_config['state_dim'],
                self.robot_config['action_dim']
            )
            new_policy.load_state_dict(adapted_policy)

            return new_policy

        return current_policy

    def _collect_adaptation_data(self, condition: Dict) -> List[Tuple]:
        """Collect data for adapting to new condition"""
        # This would interact with environment to collect data
        # For simulation, return empty list
        return []

    def evaluate_learning_progress(self, task_history: List[Dict]) -> Dict:
        """Evaluate learning progress and performance"""
        if not task_history:
            return {'overall_performance': 0.0, 'improvement_rate': 0.0}

        # Calculate overall performance
        performances = [task.get('performance', 0.0) for task in task_history]
        overall_performance = sum(performances) / len(performances)

        # Calculate improvement rate
        if len(performances) > 1:
            improvement_rate = (performances[-1] - performances[0]) / len(performances)
        else:
            improvement_rate = 0.0

        return {
            'overall_performance': overall_performance,
            'improvement_rate': improvement_rate,
            'total_tasks_learned': len(task_history),
            'learning_efficiency': overall_performance * (1 + improvement_rate)
        }
```

## Assessment Questions

1. Explain the challenges of applying reinforcement learning to humanoid robot control. How do balance and locomotion requirements affect the learning process?

2. Design a reward function for a humanoid robot learning to walk on uneven terrain. What factors should be considered?

3. Implement an imitation learning system that can learn manipulation skills from human demonstrations.

4. Create a meta-learning system that allows a humanoid robot to quickly adapt to new tasks with minimal training.

5. Design a curiosity-driven exploration system that helps humanoid robots discover new behaviors safely.

## Practice Exercises

1. **Reinforcement Learning**: Implement a DDPG or PPO algorithm for humanoid locomotion control.

2. **Imitation Learning**: Create a system that learns from motion capture data to reproduce human movements.

3. **Transfer Learning**: Develop a method to transfer walking skills between different humanoid robot models.

4. **Safe Exploration**: Design a learning system that explores safely while avoiding dangerous states.

## Summary

Learning and adaptation are essential capabilities for autonomous humanoid robots. This chapter covered:

- Reinforcement learning algorithms specifically adapted for humanoid control challenges
- Imitation learning systems that enable skill acquisition from human demonstrations
- Meta-learning approaches for rapid adaptation to new tasks and environments
- Curiosity-driven exploration systems that encourage autonomous learning
- Transfer learning techniques for applying learned skills across different robots and tasks
- Safe learning frameworks that ensure robot safety during the learning process
- Risk assessment and management systems for learning in complex environments

The integration of these learning and adaptation mechanisms enables humanoid robots to become more autonomous, capable, and effective in real-world applications, continuously improving their performance through experience.