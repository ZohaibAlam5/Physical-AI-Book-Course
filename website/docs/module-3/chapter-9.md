---
title: "Chapter 9 - Advanced Control Strategies for Humanoid Locomotion"
description: "Advanced control techniques for stable and efficient humanoid locomotion including adaptive control, robust control, and learning-based approaches"
sidebar_label: "Advanced Control Strategies for Humanoid Locomotion"
---

# Advanced Control Strategies for Humanoid Locomotion

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement adaptive control strategies for varying terrain and conditions
- Design robust control systems that handle model uncertainties
- Apply learning-based control approaches for locomotion adaptation
- Create hybrid control systems combining multiple control strategies
- Implement disturbance rejection and recovery mechanisms
- Design control systems that handle actuator limitations and constraints
- Apply optimal control techniques for energy-efficient locomotion

## Introduction

Humanoid locomotion control faces numerous challenges that go beyond simple position or trajectory tracking. Real-world environments present varying terrains, unexpected disturbances, model uncertainties, and changing conditions that require sophisticated control strategies. Advanced control techniques are essential for achieving stable, efficient, and robust humanoid locomotion.

Traditional control approaches often struggle with the complexities of bipedal locomotion, including the need to maintain balance while moving, handling underactuated systems, and dealing with discrete events like foot impacts. Advanced control strategies address these challenges by incorporating adaptation, robustness, learning, and optimization principles.

This chapter explores cutting-edge control techniques specifically designed for humanoid locomotion, including adaptive control for changing conditions, robust control for uncertainty handling, and learning-based approaches for continuous improvement.

## Adaptive Control for Varying Conditions

### Model Reference Adaptive Control (MRAC)

Model Reference Adaptive Control allows humanoid robots to adapt their control parameters based on changing conditions:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class LocomotionState:
    """State representation for humanoid locomotion"""
    com_position: np.ndarray  # Center of mass position [x, y, z]
    com_velocity: np.ndarray  # Center of mass velocity [vx, vy, vz]
    com_acceleration: np.ndarray  # Center of mass acceleration
    joint_positions: np.ndarray  # Joint angles
    joint_velocities: np.ndarray  # Joint velocities
    joint_torques: np.ndarray  # Joint torques
    foot_positions: Dict[str, np.ndarray]  # Position of each foot
    foot_forces: Dict[str, np.ndarray]  # Ground reaction forces
    orientation: np.ndarray  # Robot orientation [roll, pitch, yaw]
    angular_velocity: np.ndarray  # Angular velocity

class ModelReferenceAdaptiveController:
    """Model Reference Adaptive Controller for humanoid locomotion"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.state_dim = robot_config.get('state_dim', 36)  # Example state dimension
        self.action_dim = robot_config.get('action_dim', 28)  # Example action dimension

        # Reference model parameters
        self.reference_model = self._build_reference_model()
        self.adaptive_params = torch.nn.Parameter(
            torch.zeros(self.action_dim, self.state_dim)
        )

        # Adaptation rate
        self.gamma = robot_config.get('adaptation_rate', 0.01)
        self.sigma = robot_config.get('sigma_modification', 0.1)

        # Tracking error and parameter bounds
        self.error_history = []
        self.max_params = robot_config.get('max_adaptive_params', 10.0)

    def _build_reference_model(self) -> nn.Module:
        """Build reference model for desired locomotion behavior"""
        class ReferenceModel(nn.Module):
            def __init__(self, state_dim: int):
                super().__init__()
                self.dynamics = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, state_dim)
                )

            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return self.dynamics(state)

        return ReferenceModel(self.state_dim)

    def compute_control(self, current_state: LocomotionState,
                       reference_state: LocomotionState) -> np.ndarray:
        """Compute adaptive control input"""
        # Convert states to tensors
        current_tensor = torch.FloatTensor(self._state_to_vector(current_state))
        reference_tensor = torch.FloatTensor(self._state_to_vector(reference_state))

        # Get reference model output
        reference_output = self.reference_model(reference_tensor)

        # Compute tracking error
        error = current_tensor - reference_output
        self.error_history.append(error.detach().numpy())

        # Adapt control parameters
        self._adapt_parameters(error, current_tensor)

        # Compute adaptive control
        adaptive_control = torch.matmul(self.adaptive_params, current_tensor.unsqueeze(1)).squeeze()

        return adaptive_control.detach().numpy()

    def _state_to_vector(self, state: LocomotionState) -> np.ndarray:
        """Convert locomotion state to vector representation"""
        # Flatten and concatenate state components
        state_vector = np.concatenate([
            state.com_position,
            state.com_velocity,
            state.com_acceleration,
            state.joint_positions,
            state.joint_velocities,
            state.orientation,
            state.angular_velocity,
            np.array(list(state.foot_forces.get('left', [0, 0, 0]))),
            np.array(list(state.foot_forces.get('right', [0, 0, 0])))
        ])

        return state_vector

    def _adapt_parameters(self, error: torch.Tensor, state: torch.Tensor):
        """Update adaptive parameters using MIT rule with sigma modification"""
        # Compute parameter update using MIT rule
        param_update = -self.gamma * torch.outer(error, state)

        # Apply sigma modification to ensure stability
        error_norm = torch.norm(error)
        if error_norm > self.sigma:
            param_update = param_update * (error_norm - self.sigma) / error_norm

        # Update parameters with bounds
        with torch.no_grad():
            self.adaptive_params += param_update
            # Apply parameter bounds
            self.adaptive_params.data = torch.clamp(
                self.adaptive_params.data, -self.max_params, self.max_params
            )

    def detect_environment_change(self, state_history: List[LocomotionState]) -> bool:
        """Detect significant changes in environment conditions"""
        if len(state_history) < 10:
            return False

        # Analyze recent states for changes in dynamics
        recent_forces = [state.foot_forces for state in state_history[-5:]]
        force_changes = []

        for i in range(1, len(recent_forces)):
            for foot, force in recent_forces[i].items():
                prev_force = recent_forces[i-1].get(foot, np.zeros(3))
                change = np.linalg.norm(np.array(force) - np.array(prev_force))
                force_changes.append(change)

        # If average force change is significant, environment may have changed
        avg_change = np.mean(force_changes) if force_changes else 0
        return avg_change > 50.0  # Threshold for environment change detection

class AdaptiveTerrainController:
    """Adaptive controller for different terrain types"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.terrain_types = {
            'flat': {'friction': 0.8, 'stiffness': 1.0, 'damping': 0.1},
            'sloped': {'friction': 0.7, 'stiffness': 0.9, 'damping': 0.15},
            'uneven': {'friction': 0.6, 'stiffness': 0.8, 'damping': 0.2},
            'slippery': {'friction': 0.3, 'stiffness': 0.7, 'damping': 0.25}
        }

        # Terrain-specific controllers
        self.terrain_controllers = {}
        for terrain_type in self.terrain_types:
            self.terrain_controllers[terrain_type] = self._create_terrain_controller(
                terrain_type, self.terrain_types[terrain_type]
            )

        self.current_terrain = 'flat'
        self.terrain_confidence = 0.0

    def _create_terrain_controller(self, terrain_type: str, terrain_params: Dict):
        """Create controller adapted for specific terrain"""
        return {
            'type': terrain_type,
            'parameters': terrain_params.copy(),
            'gains': self._compute_terrain_gains(terrain_params),
            'adaptation_memory': []
        }

    def _compute_terrain_gains(self, terrain_params: Dict) -> Dict:
        """Compute control gains based on terrain parameters"""
        base_gains = {
            'position': 100.0,  # Base position gain
            'velocity': 10.0,   # Base velocity gain
            'balance': 50.0     # Base balance gain
        }

        # Adjust gains based on terrain properties
        friction_factor = terrain_params['friction'] / 0.8  # Normalize to flat terrain
        stiffness_factor = terrain_params['stiffness']

        adjusted_gains = {
            'position': base_gains['position'] * friction_factor * stiffness_factor,
            'velocity': base_gains['velocity'] * friction_factor,
            'balance': base_gains['balance'] * (1.0 + (1.0 - friction_factor) * 0.5)
        }

        return adjusted_gains

    def identify_terrain(self, sensor_data: Dict) -> Tuple[str, float]:
        """Identify current terrain type based on sensor data"""
        # Analyze ground reaction forces, IMU data, and visual information
        forces = sensor_data.get('ground_forces', {})
        imu_data = sensor_data.get('imu', {})
        visual_data = sensor_data.get('visual', {})

        terrain_scores = {}

        for terrain_type, params in self.terrain_types.items():
            score = self._compute_terrain_score(
                terrain_type, params, forces, imu_data, visual_data
            )
            terrain_scores[terrain_type] = score

        # Select terrain with highest confidence
        best_terrain = max(terrain_scores, key=terrain_scores.get)
        confidence = terrain_scores[best_terrain]

        return best_terrain, confidence

    def _compute_terrain_score(self, terrain_type: str, terrain_params: Dict,
                             forces: Dict, imu: Dict, visual: Dict) -> float:
        """Compute score for a terrain type based on sensor data"""
        score = 0.0

        # Analyze force patterns
        if forces:
            force_variability = np.std(list(forces.values()))
            expected_variability = {
                'flat': 10.0, 'sloped': 20.0, 'uneven': 40.0, 'slippery': 15.0
            }
            variability_match = 1.0 - abs(force_variability - expected_variability[terrain_type]) / 50.0
            score += variability_match * 0.4

        # Analyze IMU data (angular velocity, acceleration)
        if imu:
            angular_vel = np.linalg.norm(imu.get('angular_velocity', [0, 0, 0]))
            linear_acc = np.linalg.norm(imu.get('linear_acceleration', [0, 0, 0]))

            if terrain_type == 'uneven':
                score += min(angular_vel / 2.0, 1.0) * 0.3
            elif terrain_type == 'slippery':
                score += min(linear_acc / 5.0, 1.0) * 0.3

        # Add visual analysis if available
        if visual:
            texture_analysis = visual.get('texture_analysis', {})
            if terrain_type in texture_analysis:
                score += texture_analysis[terrain_type] * 0.3

        return min(score, 1.0)

    def compute_terrain_adaptive_control(self, state: LocomotionState,
                                       sensor_data: Dict) -> np.ndarray:
        """Compute control adapted to current terrain"""
        # Identify current terrain
        terrain_type, confidence = self.identify_terrain(sensor_data)

        # Update current terrain if confidence is high enough
        if confidence > 0.6 and confidence > self.terrain_confidence:
            self.current_terrain = terrain_type
            self.terrain_confidence = confidence

        # Get terrain-specific controller
        controller = self.terrain_controllers[self.current_terrain]

        # Store adaptation data
        controller['adaptation_memory'].append({
            'state': state,
            'terrain_type': terrain_type,
            'confidence': confidence
        })

        # Compute control using terrain-adapted parameters
        control_output = self._compute_terrain_control(state, controller)

        return control_output

    def _compute_terrain_control(self, state: LocomotionState,
                               controller: Dict) -> np.ndarray:
        """Compute control using terrain-specific parameters"""
        gains = controller['gains']
        params = controller['parameters']

        # Simple PD control with terrain-adapted gains
        # In practice, this would be more sophisticated
        control_signal = np.zeros(self.robot_config['action_dim'])

        # Apply position control with terrain-adapted gains
        if hasattr(state, 'joint_positions'):
            desired_positions = self._get_desired_positions(state)
            position_error = desired_positions - state.joint_positions
            control_signal[:len(position_error)] += gains['position'] * position_error

        # Apply balance control
        balance_correction = self._compute_balance_control(state, gains['balance'])
        control_signal[len(position_error):len(position_error)+len(balance_correction)] = balance_correction

        return control_signal

    def _get_desired_positions(self, state: LocomotionState) -> np.ndarray:
        """Get desired joint positions based on locomotion pattern"""
        # This would typically come from a locomotion pattern generator
        # For now, return nominal positions
        return np.zeros(len(state.joint_positions))

    def _compute_balance_control(self, state: LocomotionState, gain: float) -> np.ndarray:
        """Compute balance control based on CoM and ZMP"""
        # Simple balance control based on CoM deviation
        com_error = state.com_position[:2] - self._get_desired_com_position(state)
        balance_control = gain * com_error

        return balance_control

    def _get_desired_com_position(self, state: LocomotionState) -> np.ndarray:
        """Get desired CoM position based on support polygon"""
        # Calculate support polygon center based on foot positions
        left_foot = state.foot_positions.get('left', np.array([0, -0.1, 0]))
        right_foot = state.foot_positions.get('right', np.array([0, 0.1, 0]))

        support_center = (left_foot[:2] + right_foot[:2]) / 2.0
        return support_center
```

### Self-Organizing Maps for Gait Adaptation

```python
class SelfOrganizingGaitController:
    """Self-organizing map based controller for gait adaptation"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.som_size = (10, 10)  # 10x10 SOM grid
        self.input_dim = 12  # Input dimension for terrain features
        self.output_dim = robot_config.get('action_dim', 28)  # Output dimension

        # Initialize SOM weights
        self.weights = np.random.normal(0, 0.1,
                                       (*self.som_size, self.input_dim, self.output_dim))

        # Learning parameters
        self.learning_rate = 0.1
        self.neighborhood_radius = 3.0
        self.decay_factor = 0.99

        # Gait pattern memory
        self.gait_patterns = {}
        self.pattern_weights = np.random.normal(0, 0.1,
                                               (*self.som_size, self.output_dim))

    def get_input_features(self, state: LocomotionState,
                          sensor_data: Dict) -> np.ndarray:
        """Extract input features for SOM"""
        features = np.zeros(self.input_dim)

        # Terrain features
        ground_forces = sensor_data.get('ground_forces', {})
        if ground_forces:
            features[0] = np.mean(list(ground_forces.values()))  # Average force
            features[1] = np.std(list(ground_forces.values()))   # Force variability

        # Balance features
        features[2] = state.com_position[2]  # CoM height
        features[3] = np.linalg.norm(state.com_velocity)  # CoM velocity
        features[4] = abs(state.orientation[1])  # Pitch angle
        features[5] = np.linalg.norm(state.angular_velocity)  # Angular velocity

        # Locomotion features
        features[6] = np.linalg.norm(state.joint_velocities)  # Joint velocity
        features[7] = np.linalg.norm(state.joint_torques)     # Joint torque

        # Step timing features
        features[8] = getattr(state, 'step_time', 0.8)  # Step duration
        features[9] = getattr(state, 'step_length', 0.3)  # Step length
        features[10] = getattr(state, 'step_width', 0.2)  # Step width
        features[11] = getattr(state, 'stance_time', 0.6)  # Stance time

        return features

    def find_bmu(self, input_features: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit in SOM"""
        # Compute distances to all neurons
        distances = np.zeros(self.som_size)

        for i in range(self.som_size[0]):
            for j in range(self.som_size[1]):
                weight_vector = self.weights[i, j].flatten()
                input_flat = input_features.flatten() if input_features.ndim > 1 else input_features
                distances[i, j] = np.linalg.norm(weight_vector - input_flat)

        # Find minimum distance
        bmu_idx = np.unravel_index(np.argmin(distances), self.som_size)
        return bmu_idx

    def update_som(self, input_features: np.ndarray, bmu: Tuple[int, int]):
        """Update SOM weights based on input"""
        bmu_i, bmu_j = bmu

        for i in range(self.som_size[0]):
            for j in range(self.som_size[1]):
                # Calculate distance to BMU
                dist = np.sqrt((i - bmu_i)**2 + (j - bmu_j)**2)

                # Calculate neighborhood function
                if dist <= self.neighborhood_radius:
                    influence = np.exp(-dist**2 / (2 * self.neighborhood_radius**2))

                    # Update weights
                    self.weights[i, j] += (influence * self.learning_rate *
                                         (input_features - self.weights[i, j]))

    def compute_gait_control(self, state: LocomotionState,
                           sensor_data: Dict) -> np.ndarray:
        """Compute gait control using SOM"""
        # Get input features
        features = self.get_input_features(state, sensor_data)

        # Find BMU
        bmu = self.find_bmu(features)

        # Get control from pattern weights
        control_output = self.pattern_weights[bmu[0], bmu_j].copy()

        # Update SOM if in learning mode
        if self.robot_config.get('learning_enabled', False):
            self.update_som(features, bmu)

        # Apply control scaling
        control_output *= self.robot_config.get('control_scaling', 1.0)

        return control_output

    def store_gait_pattern(self, terrain_type: str, pattern: np.ndarray):
        """Store successful gait pattern for specific terrain"""
        if terrain_type not in self.gait_patterns:
            self.gait_patterns[terrain_type] = []

        self.gait_patterns[terrain_type].append(pattern)

    def retrieve_gait_pattern(self, terrain_type: str) -> Optional[np.ndarray]:
        """Retrieve gait pattern for specific terrain"""
        if terrain_type in self.gait_patterns:
            # Return average of stored patterns
            patterns = self.gait_patterns[terrain_type]
            return np.mean(patterns, axis=0)
        return None
```

## Robust Control for Uncertainty Handling

### H-infinity Control

H-infinity control provides robustness against model uncertainties and disturbances:

```python
class HInfinityLocomotionController:
    """H-infinity controller for robust humanoid locomotion"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.state_dim = robot_config.get('state_dim', 36)
        self.input_dim = robot_config.get('action_dim', 28)
        self.disturbance_dim = 6  # External disturbances (forces, torques)

        # H-infinity controller parameters
        self.gamma = robot_config.get('hinf_gamma', 1.0)  # Performance bound
        self.Q = np.eye(self.state_dim) * 10  # State weighting
        self.R = np.eye(self.input_dim) * 0.1  # Control weighting
        self.W1 = np.eye(self.state_dim) * 1.0  # Disturbance weighting

        # Controller gain matrices (to be computed)
        self.K = np.zeros((self.input_dim, self.state_dim))
        self.L = np.zeros((self.state_dim, self.state_dim))

        # Uncertainty bounds
        self.uncertainty_bound = robot_config.get('uncertainty_bound', 0.1)

        # Initialize controller
        self._initialize_controller()

    def _initialize_controller(self):
        """Initialize H-infinity controller matrices"""
        # This is a simplified initialization
        # In practice, this would solve the H-infinity Riccati equations
        try:
            # Solve algebraic Riccati equation approximately
            self.P = self._solve_riccati_equation()
            self.K = self._compute_controller_gain()
        except:
            # Fallback to simple LQR-like gains
            self.K = -np.eye(self.input_dim, self.state_dim) * 0.1

    def _solve_riccati_equation(self) -> np.ndarray:
        """Solve H-infinity Riccati equation (simplified)"""
        # This is a placeholder - in practice, this would use proper numerical methods
        # to solve the H-infinity Riccati equation
        P = np.eye(self.state_dim) * 1.0
        return P

    def _compute_controller_gain(self) -> np.ndarray:
        """Compute H-infinity controller gain"""
        # K = -R^(-1) * B^T * P (simplified)
        # In H-infinity, this is more complex and involves additional terms
        K = -np.eye(self.input_dim, self.state_dim) * 0.1
        return K

    def compute_robust_control(self, state: LocomotionState,
                             estimated_disturbance: np.ndarray = None) -> np.ndarray:
        """Compute robust H-infinity control"""
        # Convert state to vector
        state_vector = self._state_to_control_vector(state)

        # Estimate disturbances if not provided
        if estimated_disturbance is None:
            estimated_disturbance = self._estimate_disturbances(state)

        # Apply robust control law
        control_signal = self._hinf_control_law(state_vector, estimated_disturbance)

        return control_signal

    def _state_to_control_vector(self, state: LocomotionState) -> np.ndarray:
        """Convert locomotion state to control vector"""
        # Extract relevant state components for control
        state_vec = np.concatenate([
            state.com_position,
            state.com_velocity,
            state.orientation,
            state.angular_velocity,
            state.joint_positions,
            state.joint_velocities
        ])

        # Ensure correct dimensionality
        if len(state_vec) > self.state_dim:
            state_vec = state_vec[:self.state_dim]
        elif len(state_vec) < self.state_dim:
            padding = np.zeros(self.state_dim - len(state_vec))
            state_vec = np.concatenate([state_vec, padding])

        return state_vec

    def _estimate_disturbances(self, state: LocomotionState) -> np.ndarray:
        """Estimate external disturbances"""
        # Simple disturbance estimation based on unexpected accelerations
        # In practice, this would use more sophisticated estimation techniques
        estimated_dist = np.zeros(self.disturbance_dim)

        # Estimate based on CoM acceleration deviations
        if hasattr(state, 'com_acceleration'):
            # Threshold for disturbance detection
            if np.linalg.norm(state.com_acceleration) > 2.0:
                estimated_dist[:3] = state.com_acceleration[:3] * 10.0  # Scale factor

        # Estimate based on joint torque deviations
        if hasattr(state, 'joint_torques'):
            nominal_torques = np.zeros(len(state.joint_torques))  # Should come from model
            torque_diff = state.joint_torques - nominal_torques
            if np.linalg.norm(torque_diff) > 50.0:
                estimated_dist[3:] = torque_diff[:3]  # First 3 joints as example

        return estimated_dist

    def _hinf_control_law(self, state_vec: np.ndarray,
                         disturbance: np.ndarray) -> np.ndarray:
        """Apply H-infinity control law"""
        # Basic H-infinity control law: u = -K*x + L*d
        # where x is state, d is disturbance
        control = -np.dot(self.K, state_vec)

        # Add disturbance compensation if available
        if disturbance is not None and len(disturbance) == self.disturbance_dim:
            # Simple disturbance feedforward
            disturbance_gain = np.eye(self.input_dim)[:self.input_dim, :self.disturbance_dim]
            control += np.dot(disturbance_gain[:len(control)], disturbance[:len(control)])

        # Apply saturation limits
        max_control = self.robot_config.get('max_control_input', 100.0)
        control = np.clip(control, -max_control, max_control)

        return control

    def update_robustness(self, performance_metrics: Dict):
        """Update controller robustness based on performance"""
        # Adjust gamma based on disturbance rejection performance
        disturbance_rejection = performance_metrics.get('disturbance_rejection', 0.5)

        if disturbance_rejection < 0.7:
            # Increase robustness by decreasing gamma (more conservative)
            self.gamma *= 0.95
        elif disturbance_rejection > 0.9:
            # Decrease robustness (more aggressive) if performance is good
            self.gamma = min(self.gamma * 1.02, 2.0)  # Upper bound

        # Recompute controller matrices with new gamma
        self._initialize_controller()

class DisturbanceObserverController:
    """Disturbance observer based controller for robust locomotion"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.state_dim = robot_config.get('state_dim', 36)
        self.input_dim = robot_config.get('action_dim', 28)
        self.disturbance_dim = 6

        # Disturbance observer parameters
        self.L_do = np.eye(self.state_dim) * 10  # Observer gain
        self.A_do = -np.eye(self.state_dim) * 5  # Observer dynamics
        self.B_do = np.eye(self.state_dim)       # Input matrix for observer

        # Disturbance estimation
        self.disturbance_estimate = np.zeros(self.disturbance_dim)
        self.observer_state = np.zeros(self.state_dim)

        # Filter parameters
        self.filter_cutoff = robot_config.get('disturbance_filter_cutoff', 10.0)
        self.filter_buffer = []

    def estimate_disturbance(self, state: LocomotionState,
                           control_input: np.ndarray) -> np.ndarray:
        """Estimate external disturbances using disturbance observer"""
        # Convert state to vector
        state_vec = self._state_to_control_vector(state)

        # Predict next state based on model
        predicted_state = self._predict_state(state_vec, control_input)

        # Update observer state
        measurement = state_vec  # Direct measurement
        innovation = measurement - self.observer_state

        # Update disturbance estimate
        self.disturbance_estimate += self.L_do @ innovation * 0.001  # Integration step

        # Update observer dynamics
        self.observer_state = (predicted_state +
                              self.L_do @ innovation +
                              self.disturbance_estimate[:self.state_dim])

        # Apply low-pass filtering to disturbance estimate
        self.filter_buffer.append(self.disturbance_estimate.copy())
        if len(self.filter_buffer) > 10:
            self.filter_buffer.pop(0)

        # Return filtered estimate
        if len(self.filter_buffer) > 0:
            filtered_disturbance = np.mean(self.filter_buffer, axis=0)
        else:
            filtered_disturbance = self.disturbance_estimate

        return filtered_disturbance

    def _predict_state(self, current_state: np.ndarray,
                      control_input: np.ndarray) -> np.ndarray:
        """Predict next state based on system model"""
        # Simplified prediction model
        # In practice, this would use the actual robot dynamics model
        dt = 0.001  # Integration step (1kHz)
        A = -np.eye(len(current_state)) * 2  # Simplified system matrix
        B = np.eye(len(current_state))[:len(current_state), :len(control_input)] if len(control_input) <= len(current_state) else np.eye(len(current_state))

        predicted = current_state + dt * (A @ current_state + B @ control_input[:len(current_state)])
        return predicted

    def compensate_disturbance(self, nominal_control: np.ndarray,
                             estimated_disturbance: np.ndarray) -> np.ndarray:
        """Compensate control for estimated disturbances"""
        # Disturbance compensation
        compensation_gain = 1.0  # Tuning parameter
        disturbance_compensation = -compensation_gain * estimated_disturbance[:len(nominal_control)]

        # Apply compensation
        compensated_control = nominal_control + disturbance_compensation

        # Apply control limits
        max_control = self.robot_config.get('max_control_input', 100.0)
        compensated_control = np.clip(compensated_control, -max_control, max_control)

        return compensated_control

    def reset_observer(self):
        """Reset disturbance observer state"""
        self.disturbance_estimate = np.zeros(self.disturbance_dim)
        self.observer_state = np.zeros(self.state_dim)
        self.filter_buffer = []
```

## Learning-Based Control Approaches

### Reinforcement Learning for Locomotion

```python
class ReinforcementLearningLocomotionController:
    """Reinforcement learning based controller for humanoid locomotion"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.state_dim = robot_config.get('state_dim', 36)
        self.action_dim = robot_config.get('action_dim', 28)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural network for policy
        self.policy_network = self._build_policy_network().to(self.device)
        self.value_network = self._build_value_network().to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=3e-4
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=3e-4
        )

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 100000

        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.95    # GAE parameter
        self.entropy_coef = 0.01
        self.clip_epsilon = 0.2

    def _build_policy_network(self) -> torch.nn.Module:
        """Build policy network for locomotion control"""
        class PolicyNetwork(torch.nn.Module):
            def __init__(self, state_dim: int, action_dim: int):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, action_dim * 2)  # Mean and std for each action
                )

            def forward(self, state: torch.Tensor):
                output = self.network(state)
                action_dim = output.shape[-1] // 2
                mean = output[..., :action_dim]
                log_std = output[..., action_dim:]
                return torch.tanh(mean), torch.clamp(log_std, -20, 2)

        return PolicyNetwork(self.state_dim, self.action_dim)

    def _build_value_network(self) -> torch.nn.Module:
        """Build value network for state evaluation"""
        class ValueNetwork(torch.nn.Module):
            def __init__(self, state_dim: int):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 1)
                )

            def forward(self, state: torch.Tensor):
                return self.network(state)

        return ValueNetwork(self.state_dim)

    def select_action(self, state: np.ndarray,
                     add_exploration: bool = True) -> Tuple[np.ndarray, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, action_log_std = self.policy_network(state_tensor)
            action_std = torch.exp(action_log_std)

            if add_exploration:
                # Sample from distribution
                action_dist = torch.distributions.Normal(action_mean, action_std)
                action = action_dist.rsample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
            else:
                # Use mean action (deterministic)
                action = action_mean
                log_prob = torch.zeros(1)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def compute_value(self, state: np.ndarray) -> float:
        """Compute state value"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_network(state_tensor)
        return value.cpu().numpy()[0, 0]

    def update_policy(self, states: List[np.ndarray], actions: List[np.ndarray],
                     rewards: List[float], next_states: List[np.ndarray],
                     dones: List[bool]) -> Dict:
        """Update policy using collected experiences"""
        if len(states) < 32:  # Need minimum batch size
            return {}

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)

        # Compute advantages using GAE
        with torch.no_grad():
            values = self.value_network(states_tensor).squeeze(-1)
            next_values = self.value_network(next_states_tensor).squeeze(-1)
            targets = rewards_tensor + self.gamma * next_values * (~dones_tensor)
            advantages = targets - values

        # Get current policy
        old_action_mean, old_action_log_std = self.policy_network(states_tensor)
        old_action_std = torch.exp(old_action_log_std)

        # Compute action probabilities
        old_dist = torch.distributions.Normal(old_action_mean, old_action_std)
        old_log_probs = old_dist.log_prob(actions_tensor).sum(dim=-1)

        # Optimize policy (PPO-style)
        for _ in range(10):  # Multiple epochs
            action_mean, action_log_std = self.policy_network(states_tensor)
            action_std = torch.exp(action_log_std)

            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions_tensor).sum(dim=-1)

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Compute surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Add entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            policy_loss -= self.entropy_coef * entropy

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.policy_optimizer.step()

        # Update value network
        for _ in range(10):
            values_pred = self.value_network(states_tensor).squeeze(-1)
            value_loss = torch.nn.functional.mse_loss(values_pred, targets.detach())

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.value_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def store_experience(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        }

        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

class CurriculumLearningController:
    """Curriculum learning for progressive locomotion skill development"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.current_level = 0
        self.performance_thresholds = [0.3, 0.5, 0.7, 0.85]  # Performance thresholds for each level
        self.levels = [
            'balance_still',      # Level 0: Maintain balance while standing
            'simple_stepping',    # Level 1: Simple stepping in place
            'forward_locomotion', # Level 2: Forward walking
            'complex_locomotion'  # Level 3: Complex locomotion with turns, etc.
        ]

        # Level-specific reward functions
        self.level_rewards = {
            'balance_still': self._balance_still_reward,
            'simple_stepping': self._simple_stepping_reward,
            'forward_locomotion': self._forward_locomotion_reward,
            'complex_locomotion': self._complex_locomotion_reward
        }

        # Level-specific constraints
        self.level_constraints = {
            'balance_still': {'max_com_deviation': 0.05, 'max_angular_velocity': 0.1},
            'simple_stepping': {'max_com_deviation': 0.1, 'min_step_count': 1},
            'forward_locomotion': {'min_forward_velocity': 0.2, 'max_energy': 50.0},
            'complex_locomotion': {'min_task_completion': 0.8, 'max_energy': 100.0}
        }

        # Performance tracking
        self.performance_history = {level: [] for level in self.levels}

    def get_current_level(self) -> str:
        """Get current curriculum level"""
        return self.levels[self.current_level]

    def evaluate_performance(self, state: LocomotionState,
                           episode_data: Dict) -> float:
        """Evaluate performance for current level"""
        current_level = self.get_current_level()
        reward_func = self.level_rewards[current_level]

        # Compute performance score
        performance_score = reward_func(state, episode_data)

        # Store in history
        self.performance_history[current_level].append(performance_score)

        # Keep only recent performances
        if len(self.performance_history[current_level]) > 100:
            self.performance_history[current_level] = (
                self.performance_history[current_level][-100:]
            )

        return performance_score

    def _balance_still_reward(self, state: LocomotionState,
                            episode_data: Dict) -> float:
        """Reward function for balance still level"""
        # Reward for maintaining stable CoM position
        com_stability = 1.0 - min(np.linalg.norm(state.com_position[:2]), 1.0)
        orientation_stability = 1.0 - min(abs(state.orientation[1]), 1.0)  # Pitch
        angular_velocity_penalty = max(0, 1.0 - np.linalg.norm(state.angular_velocity))

        reward = (com_stability * 0.4 +
                 orientation_stability * 0.3 +
                 angular_velocity_penalty * 0.3)

        return reward

    def _simple_stepping_reward(self, state: LocomotionState,
                              episode_data: Dict) -> float:
        """Reward function for simple stepping level"""
        # Include stepping behavior in reward
        step_count = episode_data.get('step_count', 0)
        step_reward = min(step_count * 0.1, 0.5)  # Up to 0.5 for steps

        # Balance maintenance
        balance_reward = self._balance_still_reward(state, episode_data) * 0.5

        reward = step_reward + balance_reward
        return reward

    def _forward_locomotion_reward(self, state: LocomotionState,
                                 episode_data: Dict) -> float:
        """Reward function for forward locomotion level"""
        # Forward velocity reward
        forward_velocity = state.com_velocity[0]  # X component
        forward_reward = max(0, forward_velocity) * 0.3

        # Balance maintenance
        balance_reward = self._balance_still_reward(state, episode_data) * 0.4

        # Energy efficiency (negative reward for high energy consumption)
        energy_consumption = episode_data.get('energy_consumed', 0)
        energy_penalty = max(0, energy_consumption / 100.0) * 0.3

        reward = forward_reward + balance_reward - energy_penalty
        return reward

    def _complex_locomotion_reward(self, state: LocomotionState,
                                 episode_data: Dict) -> float:
        """Reward function for complex locomotion level"""
        # Task completion (e.g., reaching target)
        task_completion = episode_data.get('task_completion', 0)
        task_reward = task_completion * 0.4

        # Locomotion quality
        locomotion_quality = self._forward_locomotion_reward(state, episode_data) * 0.3

        # Adaptability to changes
        adaptability_score = episode_data.get('adaptability_score', 0) * 0.3

        reward = task_reward + locomotion_quality + adaptability_score
        return reward

    def should_advance_level(self) -> bool:
        """Check if robot should advance to next level"""
        if self.current_level >= len(self.levels) - 1:
            return False  # Already at highest level

        current_level = self.get_current_level()
        recent_performance = self.performance_history[current_level][-10:]

        if len(recent_performance) < 10:
            return False

        avg_performance = sum(recent_performance) / len(recent_performance)
        threshold = self.performance_thresholds[self.current_level]

        return avg_performance >= threshold

    def advance_level(self):
        """Advance to next curriculum level"""
        if self.should_advance_level():
            self.current_level += 1
            print(f"Advancing to curriculum level: {self.get_current_level()}")
            return True
        return False

    def get_level_constraints(self) -> Dict:
        """Get constraints for current level"""
        current_level = self.get_current_level()
        return self.level_constraints.get(current_level, {})

    def reset_for_level(self, level: str = None):
        """Reset for specific level"""
        if level and level in self.levels:
            self.current_level = self.levels.index(level)
        elif level is None:
            self.current_level = 0  # Reset to beginning

        print(f"Reset to level: {self.get_current_level()}")
```

## Hybrid Control Systems

### Switching Control Strategies

```python
class HybridLocomotionController:
    """Hybrid controller combining multiple control strategies"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config

        # Initialize different control strategies
        self.mrac_controller = ModelReferenceAdaptiveController(robot_config)
        self.hinf_controller = HInfinityLocomotionController(robot_config)
        self.rl_controller = ReinforcementLearningLocomotionController(robot_config)
        self.disturbance_observer = DisturbanceObserverController(robot_config)

        # Controller selector
        self.controller_selector = ControllerSelector(robot_config)

        # Performance monitors
        self.performance_monitors = {
            'mrac': PerformanceMonitor(),
            'hinf': PerformanceMonitor(),
            'rl': PerformanceMonitor()
        }

        # Switching parameters
        self.switching_threshold = 0.1  # Threshold for switching
        self.switching_hysteresis = 0.05  # Hysteresis to prevent chattering

        # Current active controller
        self.active_controller = 'mrac'  # Default to MRAC

    def compute_control(self, state: LocomotionState,
                       sensor_data: Dict) -> np.ndarray:
        """Compute control using hybrid approach"""
        # Monitor current controller performance
        current_performance = self.performance_monitors[self.active_controller].evaluate(state)

        # Check if switching is needed
        if self._should_switch(state, sensor_data):
            self.active_controller = self._select_best_controller(state, sensor_data)

        # Compute control using active controller
        control_output = self._compute_with_active_controller(state, sensor_data)

        # Monitor and store performance
        self.performance_monitors[self.active_controller].update(state, control_output)

        return control_output

    def _should_switch(self, state: LocomotionState,
                      sensor_data: Dict) -> bool:
        """Determine if controller switching is needed"""
        # Check performance degradation
        current_performance = self.performance_monitors[self.active_controller].get_recent_performance()

        if current_performance < self.switching_threshold:
            return True

        # Check for significant environmental changes
        if self._detect_environment_change(sensor_data):
            return True

        # Check for disturbances
        disturbance_level = self._estimate_disturbance_level(state)
        if disturbance_level > 0.5:  # High disturbance
            return True

        return False

    def _select_best_controller(self, state: LocomotionState,
                              sensor_data: Dict) -> str:
        """Select best controller based on current conditions"""
        # Evaluate each controller for current situation
        scores = {}

        # MRAC: Good for changing conditions
        mrac_score = self._evaluate_mrac_suitability(state, sensor_data)
        scores['mrac'] = mrac_score

        # H-infinity: Good for high disturbances
        hinf_score = self._evaluate_hinf_suitability(state, sensor_data)
        scores['hinf'] = hinf_score

        # RL: Good for learned patterns
        rl_score = self._evaluate_rl_suitability(state, sensor_data)
        scores['rl'] = rl_score

        # Select controller with highest score
        best_controller = max(scores, key=scores.get)
        return best_controller

    def _evaluate_mrac_suitability(self, state: LocomotionState,
                                 sensor_data: Dict) -> float:
        """Evaluate suitability of MRAC controller"""
        # MRAC is good when conditions are changing
        environment_change_detected = self.mrac_controller.detect_environment_change([state])
        return 1.0 if environment_change_detected else 0.3

    def _evaluate_hinf_suitability(self, state: LocomotionState,
                                 sensor_data: Dict) -> float:
        """Evaluate suitability of H-infinity controller"""
        # H-infinity is good for high disturbance scenarios
        disturbance_level = self._estimate_disturbance_level(state)
        return min(disturbance_level * 2, 1.0)  # Scale disturbance level

    def _evaluate_rl_suitability(self, state: LocomotionState,
                               sensor_data: Dict) -> float:
        """Evaluate suitability of RL controller"""
        # RL is good when in familiar situations
        familiarity_score = self._estimate_familiarity(state, sensor_data)
        return familiarity_score

    def _estimate_disturbance_level(self, state: LocomotionState) -> float:
        """Estimate current disturbance level"""
        # Estimate based on unexpected accelerations
        expected_com_acc = np.array([0, 0, -9.81])  # Gravity
        actual_com_acc = getattr(state, 'com_acceleration', np.zeros(3))

        disturbance_acc = actual_com_acc - expected_com_acc
        disturbance_magnitude = np.linalg.norm(disturbance_acc)

        # Normalize to [0, 1]
        max_expected_disturbance = 5.0  # m/s^2
        return min(disturbance_magnitude / max_expected_disturbance, 1.0)

    def _estimate_familiarity(self, state: LocomotionState,
                            sensor_data: Dict) -> float:
        """Estimate how familiar the current situation is"""
        # This would use a familiarity estimation system
        # For now, return a simple estimate based on state novelty
        state_vector = self.mrac_controller._state_to_vector(state)

        # Compare with recent states to estimate familiarity
        if not hasattr(self, 'recent_states'):
            self.recent_states = []

        if len(self.recent_states) == 0:
            self.recent_states.append(state_vector)
            return 0.1  # Unfamiliar initially

        # Compute average distance to recent states
        distances = [np.linalg.norm(state_vector - recent) for recent in self.recent_states]
        avg_distance = sum(distances) / len(distances) if distances else 0

        # Convert to familiarity (inverse of novelty)
        familiarity = max(0, 1 - avg_distance / 10.0)  # Normalize

        # Update recent states
        self.recent_states.append(state_vector)
        if len(self.recent_states) > 10:
            self.recent_states.pop(0)

        return familiarity

    def _compute_with_active_controller(self, state: LocomotionState,
                                      sensor_data: Dict) -> np.ndarray:
        """Compute control using active controller"""
        if self.active_controller == 'mrac':
            # MRAC needs reference state
            reference_state = self._generate_reference_state(state)
            return self.mrac_controller.compute_control(state, reference_state)

        elif self.active_controller == 'hinf':
            # H-infinity controller
            estimated_disturbance = self.hinf_controller._estimate_disturbances(state)
            return self.hinf_controller.compute_robust_control(state, estimated_disturbance)

        elif self.active_controller == 'rl':
            # RL controller
            state_vector = self.mrac_controller._state_to_vector(state)
            action, _ = self.rl_controller.select_action(state_vector)
            return action

        else:
            # Fallback to simple PD control
            return self._simple_pd_control(state)

    def _generate_reference_state(self, current_state: LocomotionState) -> LocomotionState:
        """Generate reference state for MRAC"""
        # Create a reference state that represents desired behavior
        reference = LocomotionState(
            com_position=current_state.com_position.copy(),
            com_velocity=np.zeros(3),  # Zero velocity reference
            com_acceleration=np.zeros(3),
            joint_positions=current_state.joint_positions.copy(),
            joint_velocities=np.zeros_like(current_state.joint_velocities),
            joint_torques=np.zeros_like(current_state.joint_torques),
            foot_positions=current_state.foot_positions.copy(),
            foot_forces={foot: np.zeros(3) for foot in current_state.foot_positions.keys()},
            orientation=np.zeros(3),  # Zero orientation error
            angular_velocity=np.zeros(3)
        )

        # Add small adjustments to maintain balance
        reference.com_position[2] = self.robot_config.get('nominal_com_height', 0.8)

        return reference

    def _simple_pd_control(self, state: LocomotionState) -> np.ndarray:
        """Simple PD control as fallback"""
        # Return zero control as simple fallback
        return np.zeros(self.robot_config.get('action_dim', 28))

    def _detect_environment_change(self, sensor_data: Dict) -> bool:
        """Detect environment changes that might require controller switch"""
        # Check for significant changes in ground properties
        forces = sensor_data.get('ground_forces', {})
        if forces:
            force_changes = []
            if hasattr(self, 'prev_forces'):
                for foot in forces:
                    if foot in self.prev_forces:
                        change = np.linalg.norm(
                            np.array(forces[foot]) - np.array(self.prev_forces[foot])
                        )
                        force_changes.append(change)

            avg_change = np.mean(force_changes) if force_changes else 0
            self.prev_forces = forces.copy()

            return avg_change > 50.0  # Threshold for environment change

        return False

class ControllerSelector:
    """Selects appropriate controller based on context"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.context_weights = {
            'disturbance': 0.4,
            'uncertainty': 0.3,
            'familiarity': 0.2,
            'performance': 0.1
        }

    def select_controller(self, context: Dict) -> str:
        """Select controller based on context"""
        scores = {}

        # Score each controller
        for controller in ['mrac', 'hinf', 'rl']:
            scores[controller] = self._score_controller(controller, context)

        # Return controller with highest score
        return max(scores, key=scores.get)

    def _score_controller(self, controller: str, context: Dict) -> float:
        """Score a controller based on context"""
        score = 0.0

        if controller == 'mrac':
            # Good for changing conditions
            score += context.get('uncertainty_level', 0) * self.context_weights['uncertainty']
            score += (1 - context.get('familiarity', 0)) * self.context_weights['familiarity']

        elif controller == 'hinf':
            # Good for high disturbances
            score += context.get('disturbance_level', 0) * self.context_weights['disturbance']
            score += context.get('uncertainty_level', 0) * self.context_weights['uncertainty'] * 0.5

        elif controller == 'rl':
            # Good for familiar situations
            score += context.get('familiarity', 0) * self.context_weights['familiarity']
            score += context.get('performance_stability', 0) * self.context_weights['performance']

        return score

class PerformanceMonitor:
    """Monitor controller performance"""

    def __init__(self):
        self.performance_history = []
        self.max_history = 100

    def evaluate(self, state: LocomotionState) -> float:
        """Evaluate current performance"""
        # Simple performance metric based on balance
        balance_score = self._evaluate_balance(state)
        stability_score = self._evaluate_stability(state)

        performance = 0.6 * balance_score + 0.4 * stability_score
        return performance

    def _evaluate_balance(self, state: LocomotionState) -> float:
        """Evaluate balance performance"""
        # Score based on CoM position relative to support polygon
        com_height = state.com_position[2]
        nominal_height = 0.8  # Nominal CoM height

        height_score = max(0, 1 - abs(com_height - nominal_height) / 0.2)

        # Score based on orientation
        pitch_score = max(0, 1 - abs(state.orientation[1]) / 0.3)  # Max 0.3 rad pitch

        return (height_score + pitch_score) / 2

    def _evaluate_stability(self, state: LocomotionState) -> float:
        """Evaluate stability performance"""
        # Score based on velocity and acceleration
        com_vel_score = max(0, 1 - np.linalg.norm(state.com_velocity) / 1.0)
        angular_vel_score = max(0, 1 - np.linalg.norm(state.angular_velocity) / 1.0)

        return (com_vel_score + angular_vel_score) / 2

    def update(self, state: LocomotionState, control_output: np.ndarray):
        """Update performance history"""
        performance = self.evaluate(state)
        self.performance_history.append(performance)

        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)

    def get_recent_performance(self) -> float:
        """Get average performance over recent period"""
        if not self.performance_history:
            return 0.5  # Default performance

        return sum(self.performance_history[-10:]) / min(10, len(self.performance_history))
```

## Practical Implementation Considerations

### Real-Time Control Implementation

```python
class RealTimeLocomotionController:
    """Real-time implementation of advanced locomotion control"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.control_period = robot_config.get('control_period', 0.001)  # 1kHz
        self.max_computation_time = self.control_period * 0.8  # 80% of period

        # Initialize controllers
        self.hybrid_controller = HybridLocomotionController(robot_config)
        self.safety_controller = SafetyController(robot_config)

        # Timing monitoring
        self.timing_stats = {
            'avg_computation_time': 0,
            'max_computation_time': 0,
            'deadline_misses': 0
        }

        # Control loop state
        self.last_control_time = 0
        self.control_counter = 0

    def compute_real_time_control(self, state: LocomotionState,
                                sensor_data: Dict) -> np.ndarray:
        """Compute control with real-time constraints"""
        import time

        start_time = time.time()

        try:
            # Check timing constraints
            if start_time - self.last_control_time > self.control_period * 1.5:
                # Missed timing deadline, use emergency control
                control_output = self.safety_controller.emergency_control(state)
                self.timing_stats['deadline_misses'] += 1
            else:
                # Normal control computation
                control_output = self.hybrid_controller.compute_control(state, sensor_data)

            # Update timing statistics
            computation_time = time.time() - start_time
            self._update_timing_stats(computation_time)

            # Check if computation exceeded deadline
            if computation_time > self.max_computation_time:
                print(f"WARNING: Control computation exceeded deadline by {computation_time - self.max_computation_time:.4f}s")

            # Apply safety limits
            control_output = self.safety_controller.apply_limits(control_output)

            self.last_control_time = time.time()
            self.control_counter += 1

            return control_output

        except Exception as e:
            print(f"Control computation error: {e}")
            return self.safety_controller.emergency_control(state)

    def _update_timing_stats(self, computation_time: float):
        """Update timing statistics"""
        self.timing_stats['avg_computation_time'] = (
            (self.timing_stats['avg_computation_time'] * self.control_counter + computation_time) /
            (self.control_counter + 1)
        )
        self.timing_stats['max_computation_time'] = max(
            self.timing_stats['max_computation_time'], computation_time
        )

    def get_timing_report(self) -> Dict:
        """Get timing performance report"""
        return self.timing_stats.copy()

class SafetyController:
    """Safety controller for humanoid locomotion"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.safety_thresholds = {
            'max_com_deviation': 0.3,  # meters
            'max_angular_velocity': 2.0,  # rad/s
            'max_joint_velocity': 10.0,  # rad/s
            'min_com_height': 0.3  # meters
        }

    def check_safety(self, state: LocomotionState) -> Tuple[bool, List[str]]:
        """Check if current state is safe"""
        unsafe_conditions = []

        # Check CoM deviation
        com_deviation = np.linalg.norm(state.com_position[:2])
        if com_deviation > self.safety_thresholds['max_com_deviation']:
            unsafe_conditions.append(f"CoM deviation too large: {com_deviation:.3f}m")

        # Check angular velocity
        angular_vel_magnitude = np.linalg.norm(state.angular_velocity)
        if angular_vel_magnitude > self.safety_thresholds['max_angular_velocity']:
            unsafe_conditions.append(f"Angular velocity too high: {angular_vel_magnitude:.3f} rad/s")

        # Check CoM height
        if state.com_position[2] < self.safety_thresholds['min_com_height']:
            unsafe_conditions.append(f"CoM height too low: {state.com_position[2]:.3f}m")

        # Check joint velocities
        max_joint_vel = np.max(np.abs(state.joint_velocities))
        if max_joint_vel > self.safety_thresholds['max_joint_velocity']:
            unsafe_conditions.append(f"Joint velocity limit exceeded: {max_joint_vel:.3f} rad/s")

        is_safe = len(unsafe_conditions) == 0
        return is_safe, unsafe_conditions

    def emergency_control(self, state: LocomotionState) -> np.ndarray:
        """Compute emergency control to recover safe state"""
        control_output = np.zeros(self.robot_config.get('action_dim', 28))

        # Simple recovery: move to nominal pose
        nominal_joints = np.zeros(len(state.joint_positions))  # Should be robot-specific
        joint_recovery = nominal_joints - state.joint_positions
        control_output[:len(joint_recovery)] = joint_recovery * 10.0  # High gain for recovery

        # Balance recovery
        com_error = np.zeros(2)  # Drive CoM to center
        if hasattr(state, 'foot_positions'):
            left_foot = state.foot_positions.get('left', np.array([0, -0.1, 0]))
            right_foot = state.foot_positions.get('right', np.array([0, 0.1, 0]))
            support_center = (left_foot[:2] + right_foot[:2]) / 2.0
            com_error = support_center - state.com_position[:2]

        control_output[len(joint_recovery):len(joint_recovery)+2] = com_error * 50.0

        return control_output

    def apply_limits(self, control_output: np.ndarray) -> np.ndarray:
        """Apply safety limits to control output"""
        max_control = self.robot_config.get('max_control_output', 100.0)
        limited_control = np.clip(control_output, -max_control, max_control)
        return limited_control

class LocomotionOptimizer:
    """Optimization for energy-efficient locomotion"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.energy_weights = {
            'torque': 0.4,
            'velocity': 0.3,
            'acceleration': 0.3
        }

    def optimize_control_for_energy(self, base_control: np.ndarray,
                                  state: LocomotionState,
                                  constraints: Dict = None) -> np.ndarray:
        """Optimize control for energy efficiency"""
        # Simple energy optimization by reducing unnecessary control effort
        optimized_control = base_control.copy()

        # Apply energy-based scaling
        energy_factor = self._compute_energy_factor(state)
        optimized_control *= energy_factor

        # Apply constraints if provided
        if constraints:
            optimized_control = self._apply_optimization_constraints(
                optimized_control, constraints
            )

        return optimized_control

    def _compute_energy_factor(self, state: LocomotionState) -> float:
        """Compute energy scaling factor based on state"""
        # Reduce energy when in stable state
        balance_score = self._evaluate_balance_efficiency(state)
        energy_factor = 0.7 + 0.3 * balance_score  # 0.7 to 1.0 range

        # Increase energy when more control is needed
        if np.linalg.norm(state.com_velocity) > 0.5:
            energy_factor = min(energy_factor * 1.2, 1.0)

        return energy_factor

    def _evaluate_balance_efficiency(self, state: LocomotionState) -> float:
        """Evaluate how efficiently balanced the robot is"""
        # Higher score when CoM is well-positioned over support polygon
        com_pos = state.com_position[:2]

        # Calculate distance to support polygon center
        if hasattr(state, 'foot_positions'):
            left_foot = state.foot_positions.get('left', np.array([0, -0.1]))
            right_foot = state.foot_positions.get('right', np.array([0, 0.1]))
            support_center = (left_foot[:2] + right_foot[:2]) / 2.0
            com_deviation = np.linalg.norm(com_pos - support_center)

            # Score decreases with deviation (max 0.2m deviation)
            efficiency = max(0, 1 - com_deviation / 0.2)
            return efficiency

        return 0.5  # Default efficiency

    def _apply_optimization_constraints(self, control: np.ndarray,
                                      constraints: Dict) -> np.ndarray:
        """Apply optimization constraints"""
        if 'max_torque_rate' in constraints:
            # Limit rate of change of control signals
            if hasattr(self, 'prev_control'):
                rate_limit = constraints['max_torque_rate']
                rate_of_change = control - self.prev_control
                limited_rate = np.clip(rate_of_change, -rate_limit, rate_limit)
                control = self.prev_control + limited_rate

        self.prev_control = control.copy()
        return control
```

## Assessment Questions

1. Explain the principles of Model Reference Adaptive Control (MRAC) and how it applies to humanoid locomotion.

2. Design an H-infinity controller for robust humanoid balance control in the presence of model uncertainties.

3. Implement a reinforcement learning approach for learning locomotion gaits that adapt to different terrains.

4. Create a hybrid control system that switches between different control strategies based on environmental conditions.

5. Design a safety controller that ensures humanoid robot stability under all operating conditions.

## Practice Exercises

1. **Adaptive Control**: Implement an adaptive controller that adjusts its parameters based on changing terrain conditions.

2. **Robust Control**: Create an H-infinity controller that maintains balance despite external disturbances.

3. **Learning Control**: Develop a reinforcement learning system that learns efficient walking patterns.

4. **Hybrid Control**: Build a switching system that selects the best control strategy based on real-time conditions.

## Summary

Advanced control strategies are essential for achieving stable and robust humanoid locomotion. This chapter covered:

- Adaptive control techniques that adjust to changing conditions and environments
- Robust control methods that handle model uncertainties and external disturbances
- Learning-based approaches that improve locomotion through experience
- Hybrid control systems that combine multiple strategies for optimal performance
- Real-time implementation considerations for practical deployment
- Safety and optimization techniques for reliable operation

The integration of these advanced control techniques enables humanoid robots to achieve stable, efficient, and adaptive locomotion in complex real-world environments.