---
title: Chapter 2 - Synthetic Data Generation for Robotics
description: Techniques for generating synthetic data to train and validate robotics AI systems
sidebar_position: 2
---

# Chapter 2: Synthetic Data Generation for Robotics

## Learning Objectives

After completing this chapter, you should be able to:
- Understand the principles of synthetic data generation for robotics
- Implement domain randomization techniques for robust AI training
- Generate photorealistic synthetic datasets using advanced simulation
- Apply synthetic-to-real transfer techniques for deployment
- Validate synthetic data quality and realism

## Introduction

Synthetic data generation has become a cornerstone of modern robotics AI development, particularly for humanoid robots that require extensive training data for perception, control, and decision-making systems. With the computational power of NVIDIA Isaac Sim and advanced rendering capabilities, robotics researchers and engineers can now generate massive, diverse, and perfectly-labeled datasets that would be impossible or prohibitively expensive to collect in the real world.

The key advantage of synthetic data generation is that it provides unlimited access to ground truth information (3D positions, semantic segmentation, depth maps, etc.) while allowing complete control over environmental conditions, lighting, and scene complexity. This chapter explores the techniques and best practices for generating high-quality synthetic data for robotics applications.

## Principles of Synthetic Data Generation

### Why Synthetic Data Matters for Robotics

Physical AI systems, especially those involving humanoid robots, require vast amounts of training data to handle the complexity and variability of real-world environments. Synthetic data generation offers several key advantages:

1. **Ground Truth Availability**: Perfectly accurate labels for training perception systems
2. **Infinite Variation**: Unlimited environmental, lighting, and object variations
3. **Cost Efficiency**: No physical hardware or lab time required
4. **Safety**: Train dangerous behaviors in simulation without risk
5. **Speed**: Generate thousands of samples per minute compared to real-world collection

### The Reality Gap Problem

The main challenge in synthetic data generation is the "reality gap" - the difference between synthetic and real-world data that can cause trained models to fail when deployed on physical robots. This chapter will explore techniques to minimize this gap.

### Synthetic Data Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scene Gen     │    │   Sensor Sim     │    │   Data Label    │
│   (Objects,     │───▶│   (Cameras,      │───▶│   (Semantic,    │
│   Lighting,     │    │   LiDAR, IMU)    │    │   Instance,     │
│   Materials)    │    │                  │    │   Depth, etc.)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Parametric Scene        Realistic Sensor        Structured Dataset
   Generation             Simulation             for ML Training
```

## Domain Randomization

### Concept and Implementation

Domain randomization is a technique that intentionally varies environmental parameters during synthetic data generation to make AI models more robust to real-world variations:

```python
import numpy as np
import random
import cv2
from scipy.spatial.transform import Rotation as R
import os

class DomainRandomizationEngine:
    """Engine for applying domain randomization to synthetic data generation"""

    def __init__(self):
        self.randomization_params = {
            'lighting': {
                'intensity_range': (0.3, 2.0),
                'direction_range': (0, 2*np.pi),
                'color_temperature_range': (3000, 8000)  # Kelvin
            },
            'materials': {
                'albedo_range': (0.1, 1.0),
                'roughness_range': (0.0, 1.0),
                'metallic_range': (0.0, 1.0),
                'texture_scale_range': (0.5, 2.0)
            },
            'camera': {
                'intrinsics': {
                    'focal_length_range': (400, 800),
                    'principal_point_range': (0.1, 0.9),  # As fraction of image dimensions
                    'distortion_range': (0.0, 0.1)
                },
                'extrinsics': {
                    'position_variance': (0.05, 0.05, 0.02),  # meters
                    'orientation_variance': (0.1, 0.1, 0.05)  # radians
                }
            },
            'objects': {
                'position_variance': (0.1, 0.1, 0.05),  # meters
                'scale_variance': (0.8, 1.2),  # Scale factor
                'rotation_variance': (0.2, 0.2, 0.2)  # radians
            },
            'environment': {
                'floor_texture': ['wood', 'tile', 'carpet', 'concrete'],
                'background_objects': ['plants', 'furniture', 'decorations'],
                'clutter_density': (0.0, 0.3)  # Objects per square meter
            }
        }

    def randomize_scene(self, scene_config):
        """Apply domain randomization to scene configuration"""
        randomized_config = scene_config.copy()

        # Randomize lighting
        randomized_config['lighting'] = self.randomize_lighting(scene_config.get('lighting', {}))

        # Randomize materials
        randomized_config['materials'] = self.randomize_materials(scene_config.get('materials', {}))

        # Randomize camera parameters
        if 'camera' in randomized_config:
            randomized_config['camera'] = self.randomize_camera(randomized_config['camera'])

        # Randomize object properties
        if 'objects' in randomized_config:
            randomized_config['objects'] = self.randomize_objects(randomized_config['objects'])

        # Randomize environment
        randomized_config['environment'] = self.randomize_environment(
            scene_config.get('environment', {}))

        return randomized_config

    def randomize_lighting(self, lighting_config):
        """Randomize lighting parameters"""
        randomized_lighting = lighting_config.copy()

        # Randomize intensity
        intensity_factor = random.uniform(
            self.randomization_params['lighting']['intensity_range'][0],
            self.randomization_params['lighting']['intensity_range'][1]
        )
        if 'intensity' in randomized_lighting:
            randomized_lighting['intensity'] *= intensity_factor

        # Randomize direction
        direction_change = random.uniform(
            self.randomization_params['lighting']['direction_range'][0],
            self.randomization_params['lighting']['direction_range'][1]
        )
        if 'direction' in randomized_lighting:
            orig_dir = np.array(randomized_lighting['direction'])
            rotation = R.from_euler('z', direction_change)
            new_dir = rotation.apply(orig_dir)
            randomized_lighting['direction'] = new_dir.tolist()

        # Randomize color temperature
        color_temp = random.uniform(
            self.randomization_params['lighting']['color_temperature_range'][0],
            self.randomization_params['lighting']['color_temperature_range'][1]
        )
        # Convert color temperature to RGB (simplified)
        rgb_color = self.color_temperature_to_rgb(color_temp)
        randomized_lighting['color'] = rgb_color

        return randomized_lighting

    def color_temperature_to_rgb(self, kelvin):
        """Convert color temperature in Kelvin to RGB values"""
        temp = kelvin / 100
        red = 0
        green = 0
        blue = 0

        # Red calculation
        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            red = max(0, min(255, red))

        # Green calculation
        if temp <= 66:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))

        # Blue calculation
        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = max(0, min(255, blue))

        return [red/255, green/255, blue/255, 1.0]

    def randomize_materials(self, materials_config):
        """Randomize material properties"""
        randomized_materials = {}

        for material_name, material_props in materials_config.items():
            new_props = material_props.copy()

            # Randomize albedo (base color)
            if 'albedo' in new_props:
                albedo_factor = random.uniform(
                    self.randomization_params['materials']['albedo_range'][0],
                    self.randomization_params['materials']['albedo_range'][1]
                )
                # Apply factor to each color channel
                new_albedo = [c * albedo_factor for c in new_props['albedo']]
                new_props['albedo'] = [max(0, min(1, c)) for c in new_albedo]

            # Randomize roughness
            if 'roughness' in new_props:
                roughness_factor = random.uniform(
                    self.randomization_params['materials']['roughness_range'][0],
                    self.randomization_params['materials']['roughness_range'][1]
                )
                new_props['roughness'] = max(0, min(1, roughness_factor))

            # Randomize metallic
            if 'metallic' in new_props:
                metallic_factor = random.uniform(
                    self.randomization_params['materials']['metallic_range'][0],
                    self.randomization_params['materials']['metallic_range'][1]
                )
                new_props['metallic'] = max(0, min(1, metallic_factor))

            # Randomize texture scale
            if 'texture_scale' in new_props:
                scale_factor = random.uniform(
                    self.randomization_params['materials']['texture_scale_range'][0],
                    self.randomization_params['materials']['texture_scale_range'][1]
                )
                new_props['texture_scale'] = scale_factor

            randomized_materials[material_name] = new_props

        return randomized_materials

    def randomize_camera(self, camera_config):
        """Randomize camera parameters"""
        randomized_camera = camera_config.copy()

        # Randomize intrinsics
        if 'intrinsics' in randomized_camera:
            intrinsics = randomized_camera['intrinsics']

            # Focal length
            focal_factor = random.uniform(
                self.randomization_params['camera']['intrinsics']['focal_length_range'][0],
                self.randomization_params['camera']['intrinsics']['focal_length_range'][1]
            )
            if 'fx' in intrinsics:
                intrinsics['fx'] = focal_factor
            if 'fy' in intrinsics:
                intrinsics['fy'] = focal_factor

            # Principal point (as fraction of image dimensions)
            pp_x_factor = random.uniform(
                self.randomization_params['camera']['intrinsics']['principal_point_range'][0],
                self.randomization_params['camera']['intrinsics']['principal_point_range'][1]
            )
            pp_y_factor = random.uniform(
                self.randomization_params['camera']['intrinsics']['principal_point_range'][0],
                self.randomization_params['camera']['intrinsics']['principal_point_range'][1]
            )
            if 'cx' in intrinsics:
                intrinsics['cx'] = pp_x_factor * camera_config.get('width', 640)
            if 'cy' in intrinsics:
                intrinsics['cy'] = pp_y_factor * camera_config.get('height', 480)

            # Distortion parameters
            if 'distortion' in intrinsics:
                for i in range(len(intrinsics['distortion'])):
                    distortion_var = random.uniform(
                        -self.randomization_params['camera']['intrinsics']['distortion_range'][1],
                        self.randomization_params['camera']['intrinsics']['distortion_range'][1]
                    )
                    intrinsics['distortion'][i] += distortion_var
                    intrinsics['distortion'][i] = max(-1, min(1, intrinsics['distortion'][i]))

        # Randomize extrinsics
        if 'position' in randomized_camera:
            pos_variance = self.randomization_params['camera']['extrinsics']['position_variance']
            randomized_camera['position'] = [
                pos + random.uniform(-var, var)
                for pos, var in zip(randomized_camera['position'], pos_variance)
            ]

        if 'orientation' in randomized_camera:
            rot_variance = self.randomization_params['camera']['extrinsics']['orientation_variance']
            original_rot = R.from_euler('xyz', randomized_camera['orientation'])

            # Add random rotation
            random_rot = R.from_euler('xyz', [
                random.uniform(-var, var) for var in rot_variance
            ])

            new_rot = random_rot * original_rot
            randomized_camera['orientation'] = new_rot.as_euler('xyz').tolist()

        return randomized_camera

    def randomize_objects(self, objects_config):
        """Randomize object properties in the scene"""
        randomized_objects = []

        for obj in objects_config:
            new_obj = obj.copy()

            # Randomize position
            pos_variance = self.randomization_params['objects']['position_variance']
            new_obj['position'] = [
                pos + random.uniform(-var, var)
                for pos, var in zip(new_obj['position'], pos_variance)
            ]

            # Randomize scale
            scale_factor = random.uniform(
                self.randomization_params['objects']['scale_variance'][0],
                self.randomization_params['objects']['scale_variance'][1]
            )
            if 'scale' in new_obj:
                if isinstance(new_obj['scale'], (list, tuple)):
                    new_obj['scale'] = [s * scale_factor for s in new_obj['scale']]
                else:
                    new_obj['scale'] = new_obj['scale'] * scale_factor

            # Randomize rotation
            rot_variance = self.randomization_params['objects']['rotation_variance']
            new_obj['rotation'] = [
                rot + random.uniform(-var, var)
                for rot, var in zip(new_obj['rotation'], rot_variance)
            ]

            randomized_objects.append(new_obj)

        return randomized_objects

    def randomize_environment(self, env_config):
        """Randomize environment properties"""
        randomized_env = env_config.copy()

        # Randomize floor texture
        floor_textures = self.randomization_params['environment']['floor_texture']
        randomized_env['floor_texture'] = random.choice(floor_textures)

        # Add random background objects
        bg_objects = self.randomization_params['environment']['background_objects']
        clutter_density = random.uniform(
            self.randomization_params['environment']['clutter_density'][0],
            self.randomization_params['environment']['clutter_density'][1]
        )

        # Calculate number of background objects based on area
        area = randomized_env.get('room_size', [5, 5])[0] * randomized_env.get('room_size', [5, 5])[1]
        n_objects = int(area * clutter_density)

        randomized_env['background_objects'] = []
        for _ in range(n_objects):
            obj_type = random.choice(bg_objects)
            randomized_env['background_objects'].append({
                'type': obj_type,
                'position': [
                    random.uniform(-area/2, area/2),
                    random.uniform(-area/2, area/2),
                    0  # Ground level
                ],
                'scale': random.uniform(0.5, 1.5)
            })

        return randomized_env

    def generate_randomized_dataset(self, base_scene, n_samples, output_dir):
        """Generate a dataset with domain randomization applied"""
        os.makedirs(output_dir, exist_ok=True)

        dataset_info = {
            'n_samples': n_samples,
            'randomization_applied': True,
            'base_scene': base_scene,
            'generated_samples': []
        }

        for i in range(n_samples):
            # Randomize the scene
            randomized_scene = self.randomize_scene(base_scene)

            # Generate sample
            sample_data = self.render_sample(randomized_scene)

            # Save sample
            sample_filename = f"sample_{i:06d}.png"
            sample_path = os.path.join(output_dir, sample_filename)
            cv2.imwrite(sample_path, sample_data['image'])

            # Save labels
            labels_filename = f"sample_{i:06d}_labels.json"
            labels_path = os.path.join(output_dir, labels_filename)
            with open(labels_path, 'w') as f:
                import json
                json.dump(sample_data['labels'], f)

            # Record sample info
            dataset_info['generated_samples'].append({
                'filename': sample_filename,
                'labels_file': labels_filename,
                'randomization_params': randomized_scene
            })

            if i % 100 == 0:
                print(f"Generated {i}/{n_samples} samples...")

        # Save dataset info
        info_path = os.path.join(output_dir, "dataset_info.json")
        with open(info_path, 'w') as f:
            import json
            json.dump(dataset_info, f, indent=2)

        return dataset_info

    def render_sample(self, scene_config):
        """Render a sample from scene configuration (simulated function)"""
        # In practice, this would interface with Isaac Sim's rendering system
        # For now, return simulated data
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Simulated labels (in practice, would come from Isaac Sim's labeling system)
        labels = {
            'semantic_segmentation': np.random.randint(0, 10, (480, 640), dtype=np.uint8),
            'instance_masks': [{'id': j, 'mask': np.random.randint(0, 2, (480, 640), dtype=np.uint8)} for j in range(5)],
            'object_poses': [{'id': j, 'position': [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 2)], 'rotation': [0, 0, random.uniform(0, 2*np.pi)]} for j in range(5)],
            'depth_map': np.random.uniform(0.1, 10.0, (480, 640)).astype(np.float32)
        }

        return {
            'image': image,
            'labels': labels
        }
```

### Advanced Randomization Techniques

```python
class AdvancedDomainRandomization:
    """Advanced domain randomization techniques for improved robustness"""

    def __init__(self):
        # Advanced randomization parameters
        self.advanced_params = {
            'atmospheric': {
                'fog_density_range': (0.0, 0.05),
                'haze_amount_range': (0.0, 0.2),
                'rain_amount_range': (0.0, 0.3),
                'snow_amount_range': (0.0, 0.3)
            },
            'dynamic_elements': {
                'moving_objects': (0, 3),  # Number of moving objects
                'light_flicker_range': (0.0, 0.1),  # Intensity flicker
                'camera_shake_range': (0.0, 0.01)   # Position shake
            },
            'sensor_noise': {
                'gaussian_noise_range': (0.001, 0.02),
                'salt_pepper_noise_range': (0.0, 0.01),
                'motion_blur_range': (0.0, 0.1)
            }
        }

    def randomize_atmospheric_conditions(self, scene_config):
        """Randomize atmospheric conditions"""
        # In Isaac Sim, this would modify the rendering environment
        # For simulation purposes:
        atmospheric_effects = {
            'fog_density': random.uniform(
                self.advanced_params['atmospheric']['fog_density_range'][0],
                self.advanced_params['atmospheric']['fog_density_range'][1]),
            'haze_amount': random.uniform(
                self.advanced_params['atmospheric']['haze_amount_range'][0],
                self.advanced_params['atmospheric']['haze_amount_range'][1]),
            'weather': random.choices(
                ['clear', 'cloudy', 'rainy', 'snowy'],
                weights=[0.6, 0.3, 0.05, 0.05])[0]
        }

        scene_config['atmospheric_effects'] = atmospheric_effects
        return scene_config

    def randomize_dynamic_elements(self, scene_config):
        """Add dynamic elements to the scene"""
        # Add moving objects that create realistic motion blur and occlusion
        n_moving_objects = random.randint(
            self.advanced_params['dynamic_elements']['moving_objects'][0],
            self.advanced_params['dynamic_elements']['moving_objects'][1])

        moving_objects = []
        for _ in range(n_moving_objects):
            moving_objects.append({
                'type': random.choice(['box', 'cylinder', 'sphere']),
                'initial_position': [
                    random.uniform(-2, 2),  # x
                    random.uniform(-2, 2),  # y
                    random.uniform(0.5, 2)   # z
                ],
                'velocity': [
                    random.uniform(-0.5, 0.5),  # dx/dt
                    random.uniform(-0.5, 0.5),  # dy/dt
                    0  # dz/dt (for now, no vertical motion)
                ],
                'motion_type': random.choice(['linear', 'circular', 'oscillating']),
                'motion_params': self.generate_motion_parameters()
            })

        scene_config['moving_objects'] = moving_objects
        return scene_config

    def generate_motion_parameters(self):
        """Generate motion parameters for dynamic objects"""
        motion_type = random.choice(['linear', 'circular', 'oscillating'])

        if motion_type == 'linear':
            return {
                'direction': [random.uniform(-1, 1), random.uniform(-1, 1), 0],
                'speed': random.uniform(0.1, 0.5),
                'duration': random.uniform(1, 5)
            }
        elif motion_type == 'circular':
            return {
                'center': [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1.5)],
                'radius': random.uniform(0.2, 0.8),
                'angular_velocity': random.uniform(-2, 2),
                'duration': random.uniform(2, 10)
            }
        else:  # oscillating
            return {
                'amplitude': [random.uniform(0.1, 0.3), random.uniform(0.1, 0.3), 0],
                'frequency': random.uniform(0.5, 2.0),
                'phase': random.uniform(0, 2*np.pi),
                'duration': random.uniform(1, 5)
            }

    def apply_sensor_noise_model(self, rendered_image, scene_config):
        """Apply realistic sensor noise model to rendered image"""
        noisy_image = rendered_image.astype(np.float32)

        # Apply different types of noise based on scene conditions
        if 'sensor_noise' in scene_config:
            params = scene_config['sensor_noise']

            # Gaussian noise
            gaussian_sigma = random.uniform(
                params['gaussian_noise_range'][0],
                params['gaussian_noise_range'][1])
            gaussian_noise = np.random.normal(0, gaussian_sigma, rendered_image.shape)
            noisy_image += gaussian_noise

            # Salt and pepper noise
            salt_vs_pepper = 0.5
            amount = random.uniform(
                params['salt_pepper_noise_range'][0],
                params['salt_pepper_noise_range'][1])

            if amount > 0:
                # Salt mode
                num_salt = np.ceil(amount * rendered_image.size * salt_vs_pepper)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in rendered_image.shape]
                noisy_image[coords] = 1

                # Pepper mode
                num_pepper = np.ceil(amount * rendered_image.size * (1. - salt_vs_pepper))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in rendered_image.shape]
                noisy_image[coords] = 0

            # Motion blur (simulated)
            motion_blur_strength = random.uniform(
                params['motion_blur_range'][0],
                params['motion_blur_range'][1])

            if motion_blur_strength > 0:
                # Apply simple motion blur
                kernel_size = int(motion_blur_strength * 10) + 1
                if kernel_size > 1:
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[int(kernel_size/2), :] = np.ones(kernel_size) / kernel_size
                    noisy_image = cv2.filter2D(noisy_image, -1, kernel)

        # Clip values to valid range
        noisy_image = np.clip(noisy_image, 0, 255)

        return noisy_image.astype(np.uint8)

    def temporal_consistency_randomization(self, scene_config, previous_frame_config):
        """Maintain temporal consistency while applying randomization"""
        # For video sequences, maintain some consistency between frames
        # while still introducing variation

        consistent_config = scene_config.copy()

        # Maintain object positions with small variations
        if 'objects' in consistent_config and 'objects' in previous_frame_config:
            for i, obj in enumerate(consistent_config['objects']):
                if i < len(previous_frame_config['objects']):
                    prev_obj = previous_frame_config['objects'][i]

                    # Apply small random walk to position
                    pos_variance = self.randomization_params['objects']['position_variance']
                    obj['position'] = [
                        prev_pos + random.uniform(-var*0.3, var*0.3)  # Smaller variance for consistency
                        for prev_pos, var in zip(prev_obj['position'], pos_variance)
                    ]

        # Maintain lighting with gradual changes
        if 'lighting' in consistent_config and 'lighting' in previous_frame_config:
            prev_lighting = previous_frame_config['lighting']
            current_lighting = consistent_config['lighting']

            # Gradually change lighting parameters
            for param in ['intensity', 'direction', 'color']:
                if param in current_lighting and param in prev_lighting:
                    # Blend previous and new values for smooth transition
                    blend_factor = random.uniform(0.7, 0.9)  # Mostly previous, slightly new
                    if isinstance(current_lighting[param], (list, tuple)):
                        consistent_config['lighting'][param] = [
                            blend_factor * prev + (1-blend_factor) * curr
                            for prev, curr in zip(prev_lighting[param], current_lighting[param])
                        ]
                    else:
                        consistent_config['lighting'][param] = (
                            blend_factor * prev_lighting[param] +
                            (1-blend_factor) * current_lighting[param]
                        )

        return consistent_config
```

## Photorealistic Rendering

### RTX-Based Rendering for Robotics

NVIDIA Isaac Sim's RTX rendering capabilities enable photorealistic synthetic data generation:

```python
class RTXRenderingEngine:
    """RTX-based rendering engine for photorealistic robotics data"""

    def __init__(self, scene_config):
        self.scene_config = scene_config
        self.render_settings = {
            'renderer': 'rtx',
            'resolution': (1920, 1080),
            'samples_per_pixel': 16,
            'max_bounces': 8,
            'denoise': True,
            'ray_tracing': True,
            'global_illumination': True,
            'subsurface_scattering': True,
            'motion_blur': True,
            'depth_of_field': True
        }

    def setup_photorealistic_scene(self):
        """Setup scene with photorealistic materials and lighting"""
        # Configure advanced materials
        self.setup_advanced_materials()

        # Configure realistic lighting
        self.setup_realistic_lighting()

        # Configure camera with realistic parameters
        self.setup_realistic_camera()

        # Enable advanced rendering features
        self.enable_advanced_rendering_features()

    def setup_advanced_materials(self):
        """Setup physically-based materials for photorealistic rendering"""
        # Define realistic material properties based on real-world measurements
        self.material_library = {
            'human_skin': {
                'albedo': [0.8, 0.6, 0.5],  # Skin color
                'roughness': 0.7,
                'metallic': 0.0,
                'specular': 0.5,
                'subsurface': 0.3,  # For skin subsurface scattering
                'subsurface_color': [0.9, 0.5, 0.3],
                'normal_map': 'textures/human_skin_normals.png',
                'displacement_map': 'textures/human_skin_displacement.png'
            },
            'robot_metal': {
                'albedo': [0.7, 0.75, 0.8],  # Silver/gray metal
                'roughness': 0.2,
                'metallic': 0.9,
                'specular': 1.0,
                'subsurface': 0.0,
                'normal_map': 'textures/brushed_metal_normals.png',
                'displacement_map': 'textures/metal_displacement.png'
            },
            'robot_plastic': {
                'albedo': [0.8, 0.8, 0.9],  # White plastic
                'roughness': 0.4,
                'metallic': 0.0,
                'specular': 0.5,
                'subsurface': 0.1,
                'ior': 1.49,  # Index of refraction for acrylic plastic
                'normal_map': 'textures/plastic_normals.png'
            },
            'floor_wood': {
                'albedo': [0.6, 0.4, 0.2],  # Wood color
                'roughness': 0.6,
                'metallic': 0.0,
                'specular': 0.3,
                'subsurface': 0.0,
                'anisotropic': 0.5,  # Wood grain anisotropy
                'normal_map': 'textures/wood_normals.png',
                'displacement_map': 'textures/wood_displacement.png',
                'diffuse_map': 'textures/wood_diffuse.png'
            },
            'floor_tile': {
                'albedo': [0.8, 0.8, 0.8],  # White tile
                'roughness': 0.3,
                'metallic': 0.0,
                'specular': 0.5,
                'subsurface': 0.0,
                'normal_map': 'textures/tile_normals.png',
                'diffuse_map': 'textures/tile_diffuse.png'
            }
        }

    def setup_realistic_lighting(self):
        """Setup realistic lighting conditions"""
        # Multiple light sources for realistic illumination
        self.lighting_setup = {
            'main_light': {
                'type': 'distant',
                'color': self.color_temperature_to_rgb(5500),  # Daylight (5500K)
                'intensity': 3.0,
                'direction': [-0.5, -0.5, -1.0],  # From upper left
                'angular_diameter': 0.5  # Size of light source (for soft shadows)
            },
            'fill_light': {
                'type': 'distant',
                'color': self.color_temperature_to_rgb(4000),  # Warmer fill (4000K)
                'intensity': 1.0,
                'direction': [0.3, 0.3, -0.5],  # From lower right
                'angular_diameter': 1.0
            },
            'rim_light': {
                'type': 'distant',
                'color': self.color_temperature_to_rgb(6500),  # Cool rim light (6500K)
                'intensity': 0.5,
                'direction': [0.8, -0.2, -0.3],  # From behind and to the side
                'angular_diameter': 0.8
            },
            'environment_light': {
                'type': 'dome',
                'texture': 'textures/hdri_outdoor_01.hdr',
                'intensity': 1.0,
                'exposure': 0.0
            }
        }

    def color_temperature_to_rgb(self, kelvin):
        """Convert color temperature to RGB (more accurate implementation)"""
        temp = kelvin / 100
        red = 0
        green = 0
        blue = 0

        # Red calculation
        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            red = max(0, min(255, red))

        # Green calculation
        if temp <= 66:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))

        # Blue calculation
        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = max(0, min(255, blue))

        return [red/255, green/255, blue/255]

    def setup_realistic_camera(self):
        """Setup camera with realistic parameters"""
        self.camera_config = {
            'intrinsics': {
                'fx': 600,  # Focal length in pixels
                'fy': 600,
                'cx': 320,  # Principal point
                'cy': 240,
                'k1': -0.1,  # Radial distortion
                'k2': 0.02,
                'p1': 0.001,  # Tangential distortion
                'p2': -0.001,
                'k3': 0.0
            },
            'exposure': {
                'aperture': 2.8,  # f-stop
                'shutter_speed': 1.0/60,  # 60fps
                'iso': 100,
                'exposure_compensation': 0.0
            },
            'sensor': {
                'width': 36.0,  # mm
                'height': 24.0,  # mm
                'pixel_size': 0.0058  # mm per pixel (APS-C sensor)
            },
            'lens': {
                'focal_length': 35.0,  # mm
                'focus_distance': 2.0,  # meters
                'f_stop': 2.8,
                'distortion_model': 'brown_conrady'
            }
        }

    def enable_advanced_rendering_features(self):
        """Enable advanced RTX rendering features"""
        # Configure RTX-specific features
        self.rtx_features = {
            'ray_tracing': True,
            'global_illumination': True,
            'denoising': {
                'enable': True,
                'type': 'optix',  # Use OptiX denoiser
                'strength': 0.8
            },
            'motion_blur': {
                'enable': True,
                'samples': 8,
                'shutter_angle': 180  # degrees
            },
            'depth_of_field': {
                'enable': True,
                'focal_distance': 2.0,
                'f_stop': 2.8,
                'blur_samples': 16
            },
            'chromatic_aberration': {
                'enable': True,
                'strength': 0.01
            },
            'vignette': {
                'enable': True,
                'strength': 0.2
            }
        }

    def render_photorealistic_frame(self, scene_state, sensor_data=None):
        """Render photorealistic frame with ground truth labels"""
        # In Isaac Sim, this would use the RTX renderer
        # For simulation, we'll create a photorealistic-looking image

        # Create base image with realistic lighting
        width, height = self.render_settings['resolution']
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Simulate realistic camera response
        image = self.simulate_camera_response(image, scene_state)

        # Apply realistic sensor effects
        if sensor_data:
            image = self.apply_sensor_effects(image, sensor_data)

        # Generate ground truth labels
        labels = self.generate_ground_truth_labels(scene_state)

        return {
            'image': image,
            'labels': labels,
            'sensor_data': sensor_data or {}
        }

    def simulate_camera_response(self, image, scene_state):
        """Simulate realistic camera sensor response"""
        height, width = image.shape[:2]

        # Add realistic lighting gradients
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Simulate vignetting (darkening at corners)
        center_x, center_y = width/2, height/2
        radii = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_radius = np.sqrt(center_x**2 + center_y**2)
        vignette_factor = 1.0 - 0.3 * (radii / max_radius)**2  # 30% darkening at corners
        vignette_factor = np.expand_dims(vignette_factor, axis=2)  # Broadcast to RGB channels

        # Apply vignetting
        image = (image.astype(np.float32) * vignette_factor).astype(np.uint8)

        # Add realistic noise pattern
        noise = np.random.normal(0, 2, image.shape).astype(np.float32)

        # Add photon noise (signal-dependent)
        photon_noise = np.random.poisson(image.astype(np.float32) / 255 * 100) / 100 * 255
        photon_noise = photon_noise - np.mean(photon_noise, axis=(0,1), keepdims=True)  # Center around 0

        image = np.clip(image.astype(np.float32) + noise + photon_noise, 0, 255).astype(np.uint8)

        return image

    def apply_sensor_effects(self, image, sensor_data):
        """Apply realistic sensor effects to image"""
        # Apply motion blur based on robot movement
        if 'velocity' in sensor_data:
            vel_magnitude = np.linalg.norm(sensor_data['velocity'])
            if vel_magnitude > 0.1:  # Only apply if moving significantly
                blur_strength = min(vel_magnitude * 0.1, 5)  # Cap at 5 pixels
                kernel_size = int(blur_strength) + 1
                if kernel_size > 1:
                    # Create motion blur kernel in direction of movement
                    angle = np.arctan2(sensor_data['velocity'][1], sensor_data['velocity'][0])
                    kernel = np.zeros((kernel_size, kernel_size))
                    cos_a, sin_a = np.cos(angle), np.sin(angle)

                    # Create directional blur
                    for i in range(kernel_size):
                        x = int(i * cos_a)
                        y = int(i * sin_a)
                        if 0 <= x < kernel_size and 0 <= y < kernel_size:
                            kernel[y, x] = 1.0 / kernel_size

                    image = cv2.filter2D(image, -1, kernel)

        return image

    def generate_ground_truth_labels(self, scene_state):
        """Generate comprehensive ground truth labels for training"""
        labels = {
            'semantic_segmentation': self.generate_semantic_segmentation(scene_state),
            'instance_segmentation': self.generate_instance_segmentation(scene_state),
            'depth_map': self.generate_depth_map(scene_state),
            'normal_map': self.generate_normal_map(scene_state),
            'object_poses': self.generate_object_poses(scene_state),
            'camera_intrinsics': self.camera_config['intrinsics'],
            'camera_extrinsics': self.get_camera_pose(scene_state)
        }

        return labels

    def generate_semantic_segmentation(self, scene_state):
        """Generate semantic segmentation labels"""
        height, width = self.render_settings['resolution']
        segmentation = np.zeros((height, width), dtype=np.uint8)

        # For each object in scene, create semantic label
        for i, obj in enumerate(scene_state.get('objects', [])):
            # This would use Isaac Sim's semantic labeling capabilities
            # For simulation, we'll create a simple segmentation
            segmentation = self.draw_object_segmentation(segmentation, obj, i+1)

        return segmentation

    def generate_instance_segmentation(self, scene_state):
        """Generate instance segmentation labels"""
        height, width = self.render_settings['resolution']
        instances = []

        for i, obj in enumerate(scene_state.get('objects', [])):
            # Create binary mask for each instance
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = self.draw_object_mask(mask, obj)

            instance_info = {
                'id': i+1,
                'class_id': obj.get('class_id', 1),
                'mask': mask,
                'bbox': self.calculate_bounding_box(mask),
                'centroid': self.calculate_centroid(mask)
            }
            instances.append(instance_info)

        return instances

    def generate_depth_map(self, scene_state):
        """Generate depth map with realistic accuracy"""
        height, width = self.render_settings['resolution']
        depth_map = np.zeros((height, width), dtype=np.float32)

        # Calculate depth from camera position to each object
        camera_pos = scene_state.get('camera_position', np.array([0, 0, 1.5]))

        for obj in scene_state.get('objects', []):
            obj_pos = np.array(obj['position'])
            distance = np.linalg.norm(obj_pos - camera_pos)

            # Add realistic depth noise
            noise = np.random.normal(0, 0.01 * distance)  # 1% of distance as noise
            depth_map = self.draw_depth_values(depth_map, obj, distance + noise)

        return depth_map

    def draw_object_segmentation(self, segmentation, obj, label_id):
        """Draw object segmentation in segmentation map (simulated)"""
        # In practice, this would use Isaac Sim's rendering pipeline
        # For simulation, we'll just return the segmentation unchanged
        return segmentation

    def draw_object_mask(self, mask, obj):
        """Draw object mask (simulated)"""
        # In practice, this would use Isaac Sim's instance labeling
        # For simulation, we'll just return the mask unchanged
        return mask

    def calculate_bounding_box(self, mask):
        """Calculate bounding box from mask (simulated)"""
        # Find non-zero pixels
        y_coords, x_coords = np.nonzero(mask)
        if len(y_coords) > 0 and len(x_coords) > 0:
            return [int(np.min(x_coords)), int(np.min(y_coords)),
                   int(np.max(x_coords)), int(np.max(y_coords))]
        else:
            return [0, 0, 0, 0]

    def calculate_centroid(self, mask):
        """Calculate centroid from mask (simulated)"""
        y_coords, x_coords = np.nonzero(mask)
        if len(y_coords) > 0 and len(x_coords) > 0:
            return [int(np.mean(x_coords)), int(np.mean(y_coords))]
        else:
            return [0, 0]

    def draw_depth_values(self, depth_map, obj, distance):
        """Draw depth values for object (simulated)"""
        # In practice, this would use Isaac Sim's depth rendering
        # For simulation, we'll just return the depth map unchanged
        return depth_map

    def get_camera_pose(self, scene_state):
        """Get camera extrinsic parameters"""
        # Return camera position and orientation in world frame
        return scene_state.get('camera_pose', {
            'position': [0, 0, 1.5],
            'orientation': [0, 0, 0, 1]  # Quaternion
        })

    def generate_normal_map(self, scene_state):
        """Generate surface normal map for each pixel"""
        height, width = self.render_settings['resolution']
        normal_map = np.zeros((height, width, 3), dtype=np.float32)

        # Calculate surface normals based on object geometry
        # This would use Isaac Sim's normal rendering capabilities
        for obj in scene_state.get('objects', []):
            normal_map = self.draw_surface_normals(normal_map, obj)

        return normal_map

    def generate_object_poses(self, scene_state):
        """Generate 6-DoF poses for all objects"""
        object_poses = []

        for obj in scene_state.get('objects', []):
            pose_info = {
                'id': obj.get('id', len(object_poses)),
                'position': obj.get('position', [0, 0, 0]),
                'orientation': obj.get('orientation', [0, 0, 0, 1]),  # Quaternion
                'class_name': obj.get('class_name', 'unknown'),
                'visibility': obj.get('visibility', 1.0)  # How much of object is visible
            }
            object_poses.append(pose_info)

        return object_poses
```

## Data Labeling and Annotation

### Automated Label Generation

Isaac Sim provides powerful automated labeling capabilities:

```python
class SyntheticDataLabeler:
    """Automated data labeling for synthetic robotics datasets"""

    def __init__(self):
        self.label_types = [
            'semantic_segmentation',
            'instance_segmentation',
            'depth_map',
            'normal_map',
            'object_poses',
            'keypoints',
            'bounding_boxes',
            '3d_bounding_boxes'
        ]

        # Class mapping for robotics objects
        self.class_mapping = {
            'humanoid_robot': 1,
            'humanoid_torso': 2,
            'humanoid_head': 3,
            'humanoid_arm': 4,
            'humanoid_leg': 5,
            'humanoid_hand': 6,
            'humanoid_foot': 7,
            'table': 8,
            'chair': 9,
            'box': 10,
            'cylinder': 11,
            'sphere': 12,
            'floor': 13,
            'wall': 14,
            'obstacle': 15
        }

        # Color palette for semantic segmentation
        self.color_palette = self.generate_color_palette(len(self.class_mapping))

    def generate_color_palette(self, n_classes):
        """Generate distinct colors for semantic segmentation"""
        # Generate distinct RGB colors for each class
        colors = []
        for i in range(n_classes):
            hue = i / n_classes
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.2      # Vary brightness slightly

            # Convert HSV to RGB
            rgb = self.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)

        return colors

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB"""
        if s == 0.0:
            return [v, v, v]

        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        i = i % 6
        if i == 0:
            return [v, t, p]
        elif i == 1:
            return [q, v, p]
        elif i == 2:
            return [p, v, t]
        elif i == 3:
            return [p, q, v]
        elif i == 4:
            return [t, p, v]
        else:
            return [v, p, q]

    def generate_semantic_labels(self, scene_config):
        """Generate semantic segmentation labels"""
        # In Isaac Sim, this uses the semantic labeling system
        # For simulation, we'll create labels based on object properties

        width, height = scene_config.get('resolution', (640, 480))
        semantic_map = np.zeros((height, width), dtype=np.uint8)

        # For each object, determine which pixels it occupies
        for obj in scene_config.get('objects', []):
            class_id = self.class_mapping.get(obj.get('class_name', 'unknown'), 0)
            if class_id > 0:
                # Calculate object's projection onto image plane
                projected_pixels = self.project_object_to_image(obj, scene_config)

                # Assign class ID to those pixels
                for px, py in projected_pixels:
                    if 0 <= px < width and 0 <= py < height:
                        semantic_map[py, px] = class_id

        return semantic_map

    def generate_instance_labels(self, scene_config):
        """Generate instance segmentation labels"""
        width, height = scene_config.get('resolution', (640, 480))

        # Create instance masks for each object
        instances = []
        for i, obj in enumerate(scene_config.get('objects', [])):
            # Create binary mask for this instance
            mask = np.zeros((height, width), dtype=np.uint8)

            # Project object to image and create mask
            projected_pixels = self.project_object_to_image(obj, scene_config)
            for px, py in projected_pixels:
                if 0 <= px < width and 0 <= py < height:
                    mask[py, px] = 1

            instance_info = {
                'id': i + 1,  # Instance ID (1-indexed)
                'class_id': self.class_mapping.get(obj.get('class_name', 'unknown'), 0),
                'mask': mask,
                'bbox': self.calculate_bounding_box(mask),
                'centroid': self.calculate_centroid(mask),
                'area': np.sum(mask)
            }

            instances.append(instance_info)

        return instances

    def generate_depth_labels(self, scene_config):
        """Generate depth map labels"""
        width, height = scene_config.get('resolution', (640, 480))
        depth_map = np.zeros((height, width), dtype=np.float32)

        # Calculate depth for each pixel based on camera and object positions
        camera_pos = scene_config.get('camera_position', np.array([0, 0, 1.5]))

        for obj in scene_config.get('objects', []):
            obj_pos = np.array(obj.get('position', [0, 0, 0]))
            distance = np.linalg.norm(obj_pos - camera_pos)

            # Project object to image and assign depth
            projected_pixels = self.project_object_to_image(obj, scene_config)
            for px, py in projected_pixels:
                if 0 <= px < width and 0 <= py < height:
                    # Only assign depth if closer than existing (for occlusion handling)
                    if depth_map[py, px] == 0 or distance < depth_map[py, px]:
                        depth_map[py, px] = distance

        return depth_map

    def generate_keypoint_labels(self, humanoid_config):
        """Generate keypoint labels for humanoid robot"""
        # Define keypoint positions for humanoid robot
        keypoints = {
            'torso_center': humanoid_config.get('torso_position', [0, 0, 0.8]),
            'head_center': humanoid_config.get('head_position', [0, 0, 1.1]),
            'left_shoulder': humanoid_config.get('left_shoulder_position', [-0.15, 0.1, 0.9]),
            'right_shoulder': humanoid_config.get('right_shoulder_position', [0.15, 0.1, 0.9]),
            'left_elbow': humanoid_config.get('left_elbow_position', [-0.3, 0.1, 0.7]),
            'right_elbow': humanoid_config.get('right_elbow_position', [0.3, 0.1, 0.7]),
            'left_wrist': humanoid_config.get('left_wrist_position', [-0.45, 0.1, 0.5]),
            'right_wrist': humanoid_config.get('right_wrist_position', [0.45, 0.1, 0.5]),
            'left_hip': humanoid_config.get('left_hip_position', [-0.05, -0.08, 0.2]),
            'right_hip': humanoid_config.get('right_hip_position', [0.05, -0.08, 0.2]),
            'left_knee': humanoid_config.get('left_knee_position', [-0.05, -0.08, -0.2]),
            'right_knee': humanoid_config.get('right_knee_position', [0.05, -0.08, -0.2]),
            'left_ankle': humanoid_config.get('left_ankle_position', [-0.05, -0.08, -0.6]),
            'right_ankle': humanoid_config.get('right_ankle_position', [0.05, -0.08, -0.6]),
            'left_foot_center': humanoid_config.get('left_foot_position', [-0.05, -0.13, -0.65]),
            'right_foot_center': humanoid_config.get('right_foot_position', [0.05, 0.05, -0.65])
        }

        # Project 3D keypoints to 2D image coordinates
        projected_keypoints = {}
        for name, pos_3d in keypoints.items():
            pos_2d = self.project_3d_to_2d(pos_3d, humanoid_config.get('camera_params', {}))
            projected_keypoints[name] = pos_2d

        return {
            'keypoints_3d': keypoints,
            'keypoints_2d': projected_keypoints,
            'visibility': self.calculate_keypoint_visibility(projected_keypoints)
        }

    def project_3d_to_2d(self, point_3d, camera_params):
        """Project 3D point to 2D image coordinates"""
        # Simplified perspective projection
        fx = camera_params.get('fx', 600)
        fy = camera_params.get('fy', 600)
        cx = camera_params.get('cx', 320)
        cy = camera_params.get('cy', 240)

        x, y, z = point_3d

        # Perspective projection
        if z != 0:
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
        else:
            u, v = cx, cy  # Default to center if z is 0

        return [u, v]

    def calculate_keypoint_visibility(self, projected_keypoints):
        """Calculate visibility of keypoints in image"""
        visibility = {}
        width, height = (640, 480)  # Default image size

        for name, (u, v) in projected_keypoints.items():
            # Keypoint is visible if within image bounds
            is_visible = (0 <= u < width and 0 <= v < height)
            visibility[name] = is_visible

        return visibility

    def generate_bounding_box_labels(self, scene_config):
        """Generate bounding box labels for objects"""
        bounding_boxes = []

        for obj in scene_config.get('objects', []):
            # Get object dimensions and position
            dimensions = obj.get('dimensions', [1, 1, 1])
            position = obj.get('position', [0, 0, 0])

            # Calculate 3D bounding box in world coordinates
            half_dims = np.array(dimensions) / 2
            corners_3d = np.array([
                [position[0] - half_dims[0], position[1] - half_dims[1], position[2] - half_dims[2]],  # rear-bottom-left
                [position[0] + half_dims[0], position[1] - half_dims[1], position[2] - half_dims[2]],  # rear-bottom-right
                [position[0] + half_dims[0], position[1] + half_dims[1], position[2] - half_dims[2]],  # front-bottom-right
                [position[0] - half_dims[0], position[1] + half_dims[1], position[2] - half_dims[2]],  # front-bottom-left
                [position[0] - half_dims[0], position[1] - half_dims[1], position[2] + half_dims[2]],  # rear-top-left
                [position[0] + half_dims[0], position[1] - half_dims[1], position[2] + half_dims[2]],  # rear-top-right
                [position[0] + half_dims[0], position[1] + half_dims[1], position[2] + half_dims[2]],  # front-top-right
                [position[0] - half_dims[0], position[1] + half_dims[1], position[2] + half_dims[2]]   # front-top-left
            ])

            # Project 3D corners to 2D image coordinates
            camera_params = scene_config.get('camera_params', {})
            corners_2d = []
            for corner in corners_3d:
                u, v = self.project_3d_to_2d(corner, camera_params)
                corners_2d.append([u, v])

            # Calculate 2D bounding box from projected corners
            u_coords = [corner[0] for corner in corners_2d]
            v_coords = [corner[1] for corner in corners_2d]

            min_u, max_u = min(u_coords), max(u_coords)
            min_v, max_v = min(v_coords), max(v_coords)

            # Create bounding box annotation
            bbox_info = {
                'class_name': obj.get('class_name', 'unknown'),
                'class_id': self.class_mapping.get(obj.get('class_name', 'unknown'), 0),
                'bbox_2d': [min_u, min_v, max_u, max_v],
                'bbox_3d': {
                    'center': position,
                    'dimensions': dimensions
                },
                'corners_2d': corners_2d,
                'corners_3d': corners_3d.tolist()
            }

            bounding_boxes.append(bbox_info)

        return bounding_boxes

    def project_object_to_image(self, obj, scene_config):
        """Project object to image pixels (simplified)"""
        # This would use Isaac Sim's rendering pipeline in practice
        # For simulation, we'll return a few random pixels
        width, height = scene_config.get('resolution', (640, 480))

        # Generate some pixels based on object position and size
        center_x, center_y = self.project_3d_to_2d(obj.get('position', [0, 0, 0]),
                                                  scene_config.get('camera_params', {}))

        # Generate pixels in a region around the projected center
        pixels = []
        obj_size = obj.get('scale', 1.0) * 10  # Scale to pixel area

        for _ in range(int(obj_size * obj_size)):
            px = int(center_x + random.uniform(-obj_size, obj_size))
            py = int(center_y + random.uniform(-obj_size, obj_size))
            pixels.append((px, py))

        return pixels

    def create_dataset_annotations(self, rendered_samples, output_dir):
        """Create comprehensive dataset annotations in standard format"""
        import json

        # Create COCO-style annotations for robotics dataset
        coco_annotations = {
            "info": {
                "description": "Synthetic Robotics Dataset for Humanoid Perception",
                "version": "1.0",
                "year": 2025,
                "contributor": "Physical AI Robotics Lab",
                "date_created": "2025-12-17"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Creative Commons Attribution 4.0 International License",
                    "url": "http://creativecommons.org/licenses/by/4.0/"
                }
            ],
            "categories": [
                {"id": cat_id, "name": cat_name, "supercategory": "robotics"}
                for cat_name, cat_id in self.class_mapping.items()
            ],
            "images": [],
            "annotations": []
        }

        annotation_id = 1

        for i, sample in enumerate(rendered_samples):
            # Add image info
            image_info = {
                "id": i,
                "width": sample['image'].shape[1],
                "height": sample['image'].shape[0],
                "file_name": f"image_{i:06d}.png",
                "license": 1,
                "date_captured": "2025-12-17"
            }
            coco_annotations["images"].append(image_info)

            # Add annotations for each object in the image
            for instance in sample['labels']['instance_segmentation']:
                bbox = instance['bbox']
                area = instance['area']

                annotation_info = {
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": instance['class_id'],
                    "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],  # COCO format: [x, y, width, height]
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": self.encode_mask_as_polygon(instance['mask'])
                }

                coco_annotations["annotations"].append(annotation_info)
                annotation_id += 1

        # Save annotations
        annotations_path = os.path.join(output_dir, "annotations.json")
        with open(annotations_path, 'w') as f:
            json.dump(coco_annotations, f, indent=2)

        return annotations_path

    def encode_mask_as_polygon(self, mask):
        """Encode mask as polygon for COCO format (simplified)"""
        # In practice, would use proper mask encoding
        # For simulation, return a simple bounding polygon
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Flatten contour points
            polygon = largest_contour.flatten().tolist()
            return [polygon]
        else:
            return [[]]

    def validate_label_quality(self, labels, scene_config):
        """Validate quality of generated labels"""
        validation_results = {
            'semantic_consistency': True,
            'instance_separation': True,
            'depth_accuracy': True,
            'keypoint_visibility': True,
            'overall_quality_score': 0.0
        }

        # Check semantic segmentation consistency
        semantic_map = labels.get('semantic_segmentation')
        if semantic_map is not None:
            unique_labels = np.unique(semantic_map)
            # Check that all labels are in class mapping
            for label in unique_labels:
                if label != 0 and label not in self.class_mapping.values():
                    validation_results['semantic_consistency'] = False
                    break

        # Check instance segmentation separation
        instances = labels.get('instance_segmentation', [])
        if len(instances) > 1:
            # Check that instances don't overlap significantly
            for i in range(len(instances)):
                for j in range(i+1, len(instances)):
                    overlap = np.sum(instances[i]['mask'] & instances[j]['mask'])
                    total_area = np.sum(instances[i]['mask']) + np.sum(instances[j]['mask'])
                    if overlap > 0.1 * total_area:  # More than 10% overlap
                        validation_results['instance_separation'] = False

        # Check depth accuracy (should be positive and reasonable)
        depth_map = labels.get('depth_map')
        if depth_map is not None:
            if np.any(depth_map < 0) or np.any(depth_map > 100):  # Unreasonable depths
                validation_results['depth_accuracy'] = False

        # Calculate overall quality score
        quality_factors = [
            validation_results['semantic_consistency'],
            validation_results['instance_separation'],
            validation_results['depth_accuracy'],
            validation_results['keypoint_visibility']
        ]
        validation_results['overall_quality_score'] = sum(quality_factors) / len(quality_factors)

        return validation_results
```

## Synthetic-to-Real Transfer

### Bridging the Reality Gap

```python
class SyntheticToRealTransfer:
    """Techniques for bridging the synthetic-to-real gap in robotics"""

    def __init__(self):
        self.transfer_techniques = {
            'domain_randomization': {
                'enabled': True,
                'intensity': 'high',
                'validation_required': True
            },
            'simulated_sensor_noise': {
                'enabled': True,
                'models': ['gaussian', 'poisson', 'salt_pepper']
            },
            'texture_randomization': {
                'enabled': True,
                'types': ['albedo', 'roughness', 'normal']
            },
            'lighting_augmentation': {
                'enabled': True,
                'effects': ['shadows', 'glare', 'color_variations']
            },
            'dynamic_occlusion': {
                'enabled': True,
                'types': ['partial', 'full', 'temporal']
            }
        }

        # Style transfer network parameters (conceptual)
        self.style_transfer_params = {
            'vgg_layers': [1, 6, 11, 16, 21],  # VGG feature layers
            'content_weight': 0.025,
            'style_weight': 1.0,
            'tv_weight': 0.001,  # Total variation for smoothness
            'learning_rate': 0.01
        }

    def apply_photo_realistic_augmentation(self, synthetic_image, augmentation_type='mixed'):
        """Apply photo-realistic augmentations to synthetic images"""
        augmented_image = synthetic_image.copy().astype(np.float32)

        if augmentation_type in ['mixed', 'color']:
            # Apply color jittering to simulate different lighting conditions
            augmented_image = self.apply_color_jittering(augmented_image)

        if augmentation_type in ['mixed', 'noise']:
            # Apply realistic sensor noise
            augmented_image = self.apply_realistic_noise(augmented_image)

        if augmentation_type in ['mixed', 'blur']:
            # Apply motion blur and defocus blur
            augmented_image = self.apply_blur_augmentation(augmented_image)

        if augmentation_type in ['mixed', 'occlusion']:
            # Apply partial occlusions with realistic textures
            augmented_image = self.apply_realistic_occlusions(augmented_image)

        # Ensure values are in valid range
        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

        return augmented_image

    def apply_color_jittering(self, image):
        """Apply realistic color variations"""
        # Randomly adjust brightness
        brightness_factor = random.uniform(0.8, 1.2)
        image = image * brightness_factor

        # Randomly adjust contrast
        contrast_factor = random.uniform(0.9, 1.1)
        image = 127 + (image - 127) * contrast_factor

        # Randomly adjust saturation
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation_factor = random.uniform(0.8, 1.2)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return image

    def apply_realistic_noise(self, image):
        """Apply realistic sensor noise patterns"""
        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, 1, image.shape).astype(np.float32)
        image += gaussian_noise * 0.5

        # Add Poisson noise (photon noise)
        poisson_noise = np.random.poisson(image / 255 * 100) / 100 * 255 - image
        image += poisson_noise * 0.3

        # Add salt and pepper noise occasionally
        if random.random() < 0.1:  # 10% chance
            salt_vs_pepper = 0.5
            amount = random.uniform(0.001, 0.005)

            # Salt mode
            num_salt = np.ceil(amount * image.size * salt_vs_pepper)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - salt_vs_pepper))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0

        return image

    def apply_blur_augmentation(self, image):
        """Apply realistic blurring effects"""
        height, width = image.shape[:2]

        # Motion blur in random direction
        if random.random() < 0.3:  # 30% chance
            blur_strength = random.uniform(1, 3)
            kernel_size = int(blur_strength) + 1
            angle = random.uniform(0, 2*np.pi)

            kernel = np.zeros((kernel_size, kernel_size))
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            for i in range(kernel_size):
                x = int(i * cos_a + kernel_size/2)
                y = int(i * sin_a + kernel_size/2)
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1.0 / kernel_size

            image = cv2.filter2D(image, -1, kernel)

        # Defocus blur occasionally
        if random.random() < 0.2:  # 20% chance
            blur_radius = random.uniform(0.5, 2.0)
            kernel_size = int(blur_radius * 2) + 1
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2

            for i in range(kernel_size):
                for j in range(kernel_size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    if dist <= blur_radius:
                        kernel[i, j] = 1.0

            kernel = kernel / np.sum(kernel)
            image = cv2.filter2D(image, -1, kernel)

        return image

    def apply_realistic_occlusions(self, image):
        """Apply realistic partial occlusions"""
        height, width = image.shape[:2]

        # Add random occlusions with realistic textures
        n_occlusions = random.randint(0, 2)

        for _ in range(n_occlusions):
            # Create occlusion patch
            patch_size = random.randint(20, 100)
            start_x = random.randint(0, width - patch_size)
            start_y = random.randint(0, height - patch_size)

            # Generate realistic occlusion texture (like dust, scratches, or partial objects)
            occlusion_texture = self.generate_realistic_occlusion_texture(patch_size)

            # Apply occlusion with some transparency
            alpha = random.uniform(0.3, 0.7)
            image[start_y:start_y+patch_size, start_x:start_x+patch_size] = (
                alpha * image[start_y:start_y+patch_size, start_x:start_x+patch_size] +
                (1 - alpha) * occlusion_texture
            )

        return image

    def generate_realistic_occlusion_texture(self, size):
        """Generate realistic occlusion texture"""
        # Create texture that looks like environmental elements
        texture = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

        # Add some realistic patterns (like dust, scratches, or environmental elements)
        for i in range(size):
            for j in range(size):
                if random.random() < 0.05:  # 5% of pixels
                    # Add noise-like patterns
                    noise_val = random.randint(0, 50)
                    texture[i, j] = [noise_val, noise_val, noise_val]

        return texture

    def implement_domain_adaptation_network(self, synthetic_data, real_data_samples=None):
        """Implement domain adaptation to reduce synthetic-to-real gap"""
        # In practice, this would use deep learning techniques
        # For now, we'll implement a simplified domain adaptation approach

        if real_data_samples is not None:
            # Use real data statistics to adapt synthetic data
            real_mean = np.mean(real_data_samples, axis=(0, 1, 2))
            real_std = np.std(real_data_samples, axis=(0, 1, 2))

            # Adapt synthetic data to match real data statistics
            synthetic_mean = np.mean(synthetic_data, axis=(0, 1, 2))
            synthetic_std = np.std(synthetic_data, axis=(0, 1, 2))

            # Normalize synthetic data to 0-1 range, then scale to real data statistics
            adapted_data = (synthetic_data - synthetic_mean) / (synthetic_std + 1e-8)
            adapted_data = adapted_data * real_std + real_mean

            # Clip to valid range
            adapted_data = np.clip(adapted_data, 0, 255)

            return adapted_data.astype(np.uint8)
        else:
            # If no real data available, use photo-realistic augmentations
            return self.apply_photo_realistic_augmentation(synthetic_data, 'mixed')

    def generate_training_validation_splits(self, dataset_path, validation_ratio=0.2):
        """Generate training and validation splits for synthetic dataset"""
        import os
        import shutil
        from sklearn.model_selection import train_test_split

        # Get all image files in dataset
        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        # Split into training and validation
        train_files, val_files = train_test_split(
            image_files, test_size=validation_ratio, random_state=42)

        # Create output directories
        train_dir = os.path.join(dataset_path, 'train')
        val_dir = os.path.join(dataset_path, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Copy files to respective directories
        for file in train_files:
            dest_path = os.path.join(train_dir, os.path.basename(file))
            shutil.copy2(file, dest_path)

        for file in val_files:
            dest_path = os.path.join(val_dir, os.path.basename(file))
            shutil.copy2(file, dest_path)

        # Also split label files if they exist
        self.split_label_files(dataset_path, train_files, val_files)

        return {
            'train_count': len(train_files),
            'val_count': len(val_files),
            'train_dir': train_dir,
            'val_dir': val_dir
        }

    def split_label_files(self, dataset_path, train_files, val_files):
        """Split corresponding label files"""
        # This would handle splitting of annotation files
        # For now, we'll just indicate that this functionality exists
        pass

    def implement_curriculum_learning(self, difficulty_levels=None):
        """Implement curriculum learning for synthetic-to-real transfer"""
        if difficulty_levels is None:
            difficulty_levels = [
                'simple_backgrounds',      # Level 1: Simple, clean backgrounds
                'moderate_clutter',        # Level 2: Some environmental clutter
                'complex_environments',    # Level 3: Complex, realistic environments
                'dynamic_conditions',      # Level 4: Moving elements, lighting changes
                'realistic_scenarios'      # Level 5: Near-real conditions
            ]

        curriculum = {}
        for i, level in enumerate(difficulty_levels):
            curriculum[f'level_{i+1}'] = {
                'name': level,
                'parameters': self.get_curriculum_parameters(level),
                'sample_count': 1000 * (i + 1),  # Increase sample count with difficulty
                'training_epochs': 10 * (i + 1)   # Increase epochs with difficulty
            }

        return curriculum

    def get_curriculum_parameters(self, level_name):
        """Get parameters for a specific curriculum level"""
        params = {
            'simple_backgrounds': {
                'domain_randomization': 'low',
                'texture_variation': 'minimal',
                'lighting_changes': 'fixed',
                'occlusions': 'none',
                'sensor_noise': 'minimal'
            },
            'moderate_clutter': {
                'domain_randomization': 'medium',
                'texture_variation': 'moderate',
                'lighting_changes': 'variable',
                'occlusions': 'partial',
                'sensor_noise': 'realistic'
            },
            'complex_environments': {
                'domain_randomization': 'high',
                'texture_variation': 'high',
                'lighting_changes': 'dynamic',
                'occlusions': 'variable',
                'sensor_noise': 'variable'
            },
            'dynamic_conditions': {
                'domain_randomization': 'high',
                'texture_variation': 'high',
                'lighting_changes': 'changing',
                'occlusions': 'dynamic',
                'sensor_noise': 'variable',
                'motion_blur': 'present',
                'temporal_consistency': 'required'
            },
            'realistic_scenarios': {
                'domain_randomization': 'very_high',
                'texture_variation': 'very_high',
                'lighting_changes': 'photorealistic',
                'occlusions': 'realistic',
                'sensor_noise': 'sensor_accurate',
                'motion_blur': 'accurate',
                'temporal_consistency': 'photorealistic'
            }
        }

        return params.get(level_name, params['simple_backgrounds'])

    def validate_synthetic_to_real_transfer(self, synthetic_model, real_data_evaluation):
        """Validate effectiveness of synthetic-to-real transfer"""
        # This would involve evaluating a model trained on synthetic data
        # with real-world data to measure transfer effectiveness

        validation_results = {
            'accuracy_drop': 0.0,  # How much accuracy drops from synthetic to real
            'transfer_success_rate': 0.0,
            'domain_gap_metrics': {},
            'improvement_recommendations': []
        }

        # Simulate validation process
        if real_data_evaluation:
            # Compare synthetic and real performance
            synth_accuracy = 0.95  # Simulated synthetic accuracy
            real_accuracy = 0.75   # Simulated real accuracy after transfer

            validation_results['accuracy_drop'] = synth_accuracy - real_accuracy
            validation_results['transfer_success_rate'] = real_accuracy / synth_accuracy

            # Calculate domain gap metrics
            validation_results['domain_gap_metrics'] = {
                'feature_distribution_distance': 0.2,  # Lower is better
                'classification_performance': real_accuracy,
                'generalization_score': 0.8  # 0-1 scale
            }

            # Generate recommendations based on validation
            if validation_results['accuracy_drop'] > 0.2:
                validation_results['improvement_recommendations'].append(
                    "Significant accuracy drop detected. Increase domain randomization and add more realistic sensor noise.")
            if validation_results['transfer_success_rate'] < 0.8:
                validation_results['improvement_recommendations'].append(
                    "Poor transfer success rate. Consider implementing domain adaptation techniques.")

        return validation_results

    def implement_adversarial_domain_randomization(self):
        """Implement adversarial techniques for domain randomization"""
        # This would use adversarial networks to find worst-case domain parameters
        # For now, we'll implement a simplified version

        adversarial_params = {
            'worst_case_lighting': self.find_worst_case_lighting(),
            'challenging_textures': self.find_challenging_textures(),
            'difficult_viewpoints': self.find_difficult_viewpoints(),
            'problematic_sensor_conditions': self.find_problematic_sensor_conditions()
        }

        return adversarial_params

    def find_worst_case_lighting(self):
        """Find lighting conditions that make perception most difficult"""
        # This would use adversarial techniques to find challenging lighting
        # For simulation, return challenging lighting parameters
        return {
            'backlighting': True,
            'strong_glare': True,
            'low_contrast': True,
            'color_cast': 'random'
        }

    def find_challenging_textures(self):
        """Find textures that make perception difficult"""
        return {
            'reflective_surfaces': True,
            'transparent_objects': True,
            'camouflage_patterns': True,
            'low_contrast_textures': True
        }

    def find_difficult_viewpoints(self):
        """Find viewpoints that make perception difficult"""
        return {
            'occluded_views': True,
            'extreme_angles': True,
            'close_proximity': True,
            'long_range': True
        }

    def find_problematic_sensor_conditions(self):
        """Find sensor conditions that make perception difficult"""
        return {
            'motion_blur': True,
            'high_noise': True,
            'low_light': True,
            'saturation': True
        }
```

## Performance Optimization

### Efficient Data Generation Pipelines

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading

class EfficientDataGenerationPipeline:
    """Efficient pipeline for synthetic data generation"""

    def __init__(self, n_processes=None, batch_size=32):
        if n_processes is None:
            n_processes = mp.cpu_count() - 2  # Leave some cores free
        self.n_processes = n_processes
        self.batch_size = batch_size

        # Pre-allocate arrays to avoid memory allocation during generation
        self.pre_allocated = {
            'images': np.zeros((batch_size, 480, 640, 3), dtype=np.uint8),
            'labels': [{}] * batch_size,
            'scene_configs': [{}] * batch_size
        }

        # Queues for pipeline stages
        self.config_queue = queue.Queue(maxsize=100)
        self.render_queue = queue.Queue(maxsize=100)
        self.label_queue = queue.Queue(maxsize=100)

        # Statistics for performance monitoring
        self.stats = {
            'total_samples': 0,
            'generation_rate': 0,
            'render_time_avg': 0,
            'label_time_avg': 0,
            'memory_usage': 0
        }

    def setup_parallel_pipeline(self):
        """Setup parallel processing pipeline for data generation"""
        # Create pipeline stages
        self.config_generator = self.create_config_generator()
        self.renderer = self.create_renderer()
        self.labeler = self.create_labeler()

        # Start pipeline threads
        self.pipeline_threads = []

        # Configuration generation thread
        config_thread = threading.Thread(target=self.config_generation_worker)
        config_thread.start()
        self.pipeline_threads.append(config_thread)

        # Rendering threads
        self.render_workers = []
        for _ in range(self.n_processes):
            render_thread = threading.Thread(target=self.rendering_worker)
            render_thread.start()
            self.render_workers.append(render_thread)

        # Labeling threads
        self.label_workers = []
        for _ in range(self.n_processes):
            label_thread = threading.Thread(target=self.labeling_worker)
            label_thread.start()
            self.label_workers.append(label_thread)

    def create_config_generator(self):
        """Create configuration generator with domain randomization"""
        dr_engine = DomainRandomizationEngine()
        return dr_engine

    def create_renderer(self):
        """Create rendering engine"""
        rtx_engine = RTXRenderingEngine({})
        return rtx_engine

    def create_labeler(self):
        """Create labeling engine"""
        labeler = SyntheticDataLabeler()
        return labeler

    def config_generation_worker(self):
        """Worker thread for generating scene configurations"""
        while True:
            try:
                # Generate randomized scene configuration
                base_config = self.get_base_scene_config()
                randomized_config = self.config_generator.randomize_scene(base_config)

                # Add to render queue
                self.config_queue.put(randomized_config)

                # Update statistics
                self.stats['total_samples'] += 1

            except Exception as e:
                print(f"Configuration generation error: {e}")
                continue

    def rendering_worker(self):
        """Worker thread for rendering images"""
        while True:
            try:
                # Get configuration from queue
                scene_config = self.config_queue.get(timeout=1.0)

                # Render image
                start_time = time.time()
                rendered_result = self.renderer.render_photorealistic_frame(scene_config)
                render_time = time.time() - start_time

                # Add to labeling queue
                self.render_queue.put(rendered_result)

                # Update statistics
                self.stats['render_time_avg'] = (
                    self.stats['render_time_avg'] * 0.9 + render_time * 0.1)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Rendering error: {e}")
                continue

    def labeling_worker(self):
        """Worker thread for generating labels"""
        while True:
            try:
                # Get rendered result from queue
                rendered_result = self.render_queue.get(timeout=1.0)

                # Generate labels
                start_time = time.time()
                labels = self.labeler.generate_ground_truth_labels(rendered_result['scene_state'])
                label_time = time.time() - start_time

                # Complete result
                complete_result = {
                    'image': rendered_result['image'],
                    'labels': labels,
                    'generation_metadata': rendered_result.get('metadata', {})
                }

                # Add to output queue
                self.label_queue.put(complete_result)

                # Update statistics
                self.stats['label_time_avg'] = (
                    self.stats['label_time_avg'] * 0.9 + label_time * 0.1)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Labeling error: {e}")
                continue

    def get_base_scene_config(self):
        """Get base scene configuration"""
        return {
            'robot_position': [0, 0, 0],
            'objects': [
                {'name': 'table', 'position': [1, 0, 0], 'class_name': 'table'},
                {'name': 'box', 'position': [1.2, 0.2, 0.8], 'class_name': 'box'}
            ],
            'lighting': {
                'main_light': {'intensity': 3.0, 'direction': [-0.5, -0.5, -1.0]},
                'fill_light': {'intensity': 1.0, 'direction': [0.3, 0.3, -0.5]}
            },
            'camera': {
                'position': [0, -2, 1.5],
                'orientation': [0, 0, 0, 1],
                'resolution': (640, 480)
            }
        }

    def generate_dataset_batch(self, n_samples, output_dir):
        """Generate a batch of synthetic data samples"""
        os.makedirs(output_dir, exist_ok=True)

        samples_generated = 0
        batch_results = []

        while samples_generated < n_samples:
            try:
                # Get completed sample from queue
                complete_sample = self.label_queue.get(timeout=0.1)

                # Save image
                image_path = os.path.join(output_dir, f"image_{samples_generated:06d}.png")
                cv2.imwrite(image_path, complete_sample['image'])

                # Save labels
                labels_path = os.path.join(output_dir, f"labels_{samples_generated:06d}.json")
                with open(labels_path, 'w') as f:
                    import json
                    json.dump(complete_sample['labels'], f)

                batch_results.append({
                    'image_path': image_path,
                    'labels_path': labels_path,
                    'metadata': complete_sample['generation_metadata']
                })

                samples_generated += 1

                if samples_generated % 100 == 0:
                    print(f"Generated {samples_generated}/{n_samples} samples...")

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break

        return batch_results

    def optimize_rendering_performance(self):
        """Optimize rendering performance for faster data generation"""
        optimization_strategies = {
            'multi_gpu_rendering': self.configure_multi_gpu_rendering(),
            'level_of_detail': self.implement_lod_system(),
            'batch_rendering': self.enable_batch_rendering(),
            'texture_streaming': self.configure_texture_streaming(),
            'dynamic_loading': self.implement_dynamic_loading()
        }

        return optimization_strategies

    def configure_multi_gpu_rendering(self):
        """Configure multi-GPU rendering if available"""
        # Check for multiple GPUs and configure rendering
        try:
            import torch
            if torch.cuda.device_count() > 1:
                return {
                    'enabled': True,
                    'device_count': torch.cuda.device_count(),
                    'strategy': 'distributed_rendering'
                }
            else:
                return {
                    'enabled': False,
                    'device_count': 1,
                    'strategy': 'single_gpu'
                }
        except ImportError:
            return {
                'enabled': False,
                'device_count': 1,
                'strategy': 'cpu_rendering'
            }

    def implement_lod_system(self):
        """Implement Level of Detail system for performance"""
        # Create different quality levels for objects based on distance from camera
        lod_config = {
            'high_detail_distance': 1.0,    # meters
            'medium_detail_distance': 3.0,  # meters
            'low_detail_distance': 10.0,    # meters
            'objects': {
                'humanoid_robot': {
                    'lod_distances': [1.0, 3.0, 10.0],
                    'mesh_complexity': ['high', 'medium', 'low'],
                    'texture_resolution': ['high', 'medium', 'low']
                },
                'environment_objects': {
                    'lod_distances': [2.0, 5.0, 15.0],
                    'mesh_complexity': ['high', 'medium', 'low'],
                    'texture_resolution': ['high', 'medium', 'low']
                }
            }
        }

        return lod_config

    def enable_batch_rendering(self):
        """Enable batch rendering for multiple samples at once"""
        # Configure renderer to process multiple samples in a single batch
        batch_config = {
            'batch_size': 8,  # Render 8 samples per batch
            'shared_geometry': True,  # Share static geometry between samples
            'variable_parameters': ['lighting', 'camera_position', 'object_positions'],
            'memory_optimization': True
        }

        return batch_config

    def generate_performance_report(self):
        """Generate performance report for data generation pipeline"""
        report = {
            'samples_per_second': self.stats['total_samples'] / self.get_runtime(),
            'average_render_time_ms': self.stats['render_time_avg'] * 1000,
            'average_label_time_ms': self.stats['label_time_avg'] * 1000,
            'total_samples_generated': self.stats['total_samples'],
            'memory_usage_mb': self.stats['memory_usage'],
            'pipeline_efficiency': self.calculate_pipeline_efficiency()
        }

        return report

    def calculate_pipeline_efficiency(self):
        """Calculate pipeline efficiency metrics"""
        # Calculate how efficiently the pipeline is utilizing resources
        # This would involve measuring queue depths, worker utilization, etc.
        return 0.85  # Placeholder efficiency value

    def get_runtime(self):
        """Get total runtime of the pipeline"""
        # This would track the actual runtime of the pipeline
        return 3600  # Placeholder (1 hour)
```

## Summary

Synthetic data generation is a critical enabler for Physical AI and humanoid robotics development. The key aspects covered in this chapter include:

1. **Domain Randomization**: Systematically varying environmental parameters to create robust AI models that can generalize to real-world conditions.

2. **Photorealistic Rendering**: Using advanced rendering techniques like RTX ray tracing to create realistic synthetic data that closely matches real sensor data.

3. **Automated Labeling**: Generating perfect ground truth labels including semantic segmentation, instance masks, depth maps, and 6-DoF poses for all objects.

4. **Synthetic-to-Real Transfer**: Techniques to bridge the reality gap and ensure models trained on synthetic data work effectively on real robots.

5. **Performance Optimization**: Efficient pipelines for generating large-scale synthetic datasets quickly and cost-effectively.

The success of synthetic data generation lies in balancing realism with computational efficiency, while ensuring that the generated data captures the essential variations and challenges that real-world environments present. Proper synthetic data generation can significantly accelerate robotics AI development while reducing costs and risks associated with physical data collection.