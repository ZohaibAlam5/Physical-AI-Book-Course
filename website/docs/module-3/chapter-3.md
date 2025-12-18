---
title: Chapter 3 - Photorealistic Simulation and Rendering
description: Advanced techniques for photorealistic simulation and rendering in humanoid robotics
sidebar_position: 3
---

# Chapter 3: Photorealistic Simulation and Rendering

## Learning Objectives

After completing this chapter, you should be able to:
- Implement photorealistic rendering using NVIDIA RTX technology
- Configure advanced lighting and material properties for realistic simulation
- Generate synthetic datasets with photorealistic quality for AI training
- Apply advanced rendering techniques like global illumination and subsurface scattering
- Optimize rendering performance while maintaining visual quality

## Introduction

Photorealistic simulation and rendering are essential for creating high-fidelity synthetic datasets that can effectively bridge the gap between simulation and real-world deployment. For humanoid robots operating in human-centric environments, photorealistic simulation enables the generation of training data that closely matches real sensor inputs, improving the transferability of AI models from simulation to reality.

NVIDIA Isaac Sim leverages RTX technology to provide real-time ray tracing, global illumination, and advanced material simulation that creates truly photorealistic results. This chapter explores how to configure and optimize these advanced rendering capabilities for humanoid robotics applications.

## RTX Rendering Fundamentals

### Ray Tracing and Global Illumination

RTX rendering uses ray tracing to simulate the physical behavior of light, providing realistic shadows, reflections, and global illumination:

```python
import numpy as np
import torch
import os
from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade, UsdLux
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.synthetic_utils import SyntheticDataHelper

class RTXRenderingEngine:
    """Advanced RTX rendering engine for photorealistic humanoid simulation"""

    def __init__(self, stage_path=None):
        self.stage_path = stage_path or "/World"
        self.render_settings = {
            'renderer': 'RayTracedLightMap',  # RTX renderer
            'resolution': (1920, 1080),
            'samples_per_pixel': 16,         # Higher for better quality
            'max_bounces': 8,                # Light bounces
            'enable_denoising': True,        # RTX denoising
            'enable_motion_blur': True,      # Realistic motion blur
            'enable_depth_of_field': True,   # Camera focus effects
            'enable_chromatic_aberration': True,  # Lens effects
            'enable_vignette': True          # Camera corner darkening
        }

        # Advanced material properties
        self.material_library = {
            'human_skin': {
                'albedo': [0.8, 0.6, 0.5],
                'roughness': 0.7,
                'metallic': 0.0,
                'subsurface': 0.3,
                'subsurface_color': [0.9, 0.5, 0.3],
                'specular': 0.5,
                'ior': 1.4,
                'normal_map': 'textures/skin_normals.png',
                'displacement_map': 'textures/skin_displacement.png'
            },
            'robot_metal': {
                'albedo': [0.7, 0.75, 0.8],
                'roughness': 0.2,
                'metallic': 0.9,
                'subsurface': 0.0,
                'specular': 1.0,
                'ior': 1.33,
                'normal_map': 'textures/metal_brushed_normals.png',
                'roughness_map': 'textures/metal_roughness.png'
            },
            'robot_plastic': {
                'albedo': [0.8, 0.8, 0.9],
                'roughness': 0.4,
                'metallic': 0.0,
                'subsurface': 0.1,
                'specular': 0.5,
                'ior': 1.49,  # Acrylic plastic
                'normal_map': 'textures/plastic_normals.png',
                'specular_map': 'textures/plastic_specular.png'
            },
            'floor_wood': {
                'albedo': [0.6, 0.4, 0.2],
                'roughness': 0.6,
                'metallic': 0.0,
                'subsurface': 0.0,
                'specular': 0.3,
                'anisotropic': 0.5,  # Wood grain directionality
                'normal_map': 'textures/wood_normals.png',
                'diffuse_map': 'textures/wood_diffuse.png',
                'specular_map': 'textures/wood_specular.png'
            },
            'floor_tile': {
                'albedo': [0.8, 0.8, 0.8],
                'roughness': 0.3,
                'metallic': 0.0,
                'subsurface': 0.0,
                'specular': 0.5,
                'normal_map': 'textures/tile_normals.png',
                'diffuse_map': 'textures/tile_diffuse.png'
            }
        }

    def configure_rtx_settings(self):
        """Configure RTX-specific rendering settings"""
        # In Isaac Sim, these would be set via the rendering extension
        # For simulation purposes, we'll define the configuration

        rtx_config = {
            'render_products': {
                'color': {
                    'width': self.render_settings['resolution'][0],
                    'height': self.render_settings['resolution'][1],
                    'format': 'rgb8',
                    'samples_per_pixel': self.render_settings['samples_per_pixel'],
                    'max_bounces': self.render_settings['max_bounces']
                },
                'depth': {
                    'format': 'depth',
                    'range': [0.1, 100.0]
                },
                'semantic': {
                    'format': 'semantic_segmentation'
                },
                'instance': {
                    'format': 'instance_segmentation'
                },
                'normals': {
                    'format': 'normals'
                }
            },
            'render_settings': {
                'enable_denoising': self.render_settings['enable_denoising'],
                'denoiser_type': 'optix',  # OptiX denoiser
                'denoiser_model': 'hdr',   # HDR denoising model
                'denoiser_blend': 0.8      # Denoiser strength
            },
            'advanced_effects': {
                'motion_blur': {
                    'enable': self.render_settings['enable_motion_blur'],
                    'samples': 8,
                    'shutter_angle': 180  # degrees
                },
                'depth_of_field': {
                    'enable': self.render_settings['enable_depth_of_field'],
                    'focal_distance': 2.0,
                    'f_stop': 2.8,
                    'blur_samples': 16
                },
                'chromatic_aberration': {
                    'enable': self.render_settings['enable_chromatic_aberration'],
                    'strength': 0.01
                },
                'vignette': {
                    'enable': self.render_settings['enable_vignette'],
                    'strength': 0.2
                }
            }
        }

        return rtx_config

    def setup_advanced_materials(self, stage):
        """Setup advanced physically-based materials"""
        # Create material prims in USD stage
        for material_name, material_props in self.material_library.items():
            material_path = f"/World/Materials/{material_name}"

            # Create material
            material = UsdShade.Material.Define(stage, material_path)

            # Create shader
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.SetTypeName("OmniSurface")

            # Set surface shader
            surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(surface_output)

            # Set material parameters
            self.set_material_parameters(shader, material_props)

    def set_material_parameters(self, shader, material_props):
        """Set advanced material parameters for photorealistic appearance"""
        # Set albedo (base color)
        if 'albedo' in material_props:
            shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*material_props['albedo']))

        # Set roughness
        if 'roughness' in material_props:
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                material_props['roughness'])

        # Set metallic
        if 'metallic' in material_props:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                material_props['metallic'])

        # Set specular
        if 'specular' in material_props:
            shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(
                material_props['specular'])

        # Set subsurface scattering
        if 'subsurface' in material_props:
            shader.CreateInput("subsurface", Sdf.ValueTypeNames.Float).Set(
                material_props['subsurface'])

            if 'subsurface_color' in material_props:
                shader.CreateInput("subsurface_color", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*material_props['subsurface_color']))

        # Set index of refraction
        if 'ior' in material_props:
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(
                material_props['ior'])

        # Set anisotropic properties
        if 'anisotropic' in material_props:
            shader.CreateInput("anisotropic", Sdf.ValueTypeNames.Float).Set(
                material_props['anisotropic'])

        # Set texture maps
        if 'normal_map' in material_props:
            self.add_texture_to_shader(shader, "normal", material_props['normal_map'])

        if 'diffuse_map' in material_props:
            self.add_texture_to_shader(shader, "diffuse", material_props['diffuse_map'])

        if 'specular_map' in material_props:
            self.add_texture_to_shader(shader, "specular", material_props['specular_map'])

        if 'roughness_map' in material_props:
            self.add_texture_to_shader(shader, "roughness", material_props['roughness_map'])

        if 'displacement_map' in material_props:
            self.add_texture_to_shader(shader, "displacement", material_props['displacement_map'])

    def add_texture_to_shader(self, shader, texture_type, texture_path):
        """Add texture map to shader"""
        # Create texture sampler
        sampler_path = shader.GetPath().AppendChild(f"{texture_type}_sampler")
        sampler = UsdShade.Shader.Define(shader.GetStage(), sampler_path)
        sampler.SetTypeName("UsdUVTexture")

        # Set texture file path
        sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)

        # Connect to appropriate shader input
        output = sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # Connect based on texture type
        if texture_type == "normal":
            shader.GetInput("normal").ConnectToSource(output)
        elif texture_type == "diffuse":
            shader.GetInput("diffuse_color").ConnectToSource(output)
        elif texture_type == "specular":
            shader.GetInput("specular_level").ConnectToSource(output)
        elif texture_type == "roughness":
            shader.GetInput("roughness").ConnectToSource(output)

    def configure_advanced_lighting(self, stage):
        """Configure advanced lighting with realistic properties"""
        # Create dome light for environment illumination
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(1.0)
        dome_light.CreateTextureFileAttr("textures/hdri_outdoor_01.hdr")
        dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        # Create key light (main directional light)
        key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key_light.CreateIntensityAttr(3.0)
        key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.9))
        key_light.AddRotateXOp().Set(-45)  # Angle the light
        key_light.AddRotateYOp().Set(30)

        # Create fill light (softer, fills shadows)
        fill_light = UsdLux.DistantLight.Define(stage, "/World/FillLight")
        fill_light.CreateIntensityAttr(1.0)
        fill_light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))
        fill_light.AddRotateXOp().Set(-20)
        fill_light.AddRotateYOp().Set(-60)

        # Create rim light (creates highlights on edges)
        rim_light = UsdLux.DistantLight.Define(stage, "/World/RimLight")
        rim_light.CreateIntensityAttr(0.5)
        rim_light.CreateColorAttr(Gf.Vec3f(0.8, 0.85, 1.0))
        rim_light.AddRotateXOp().Set(20)
        rim_light.AddRotateYOp().Set(120)

        # Configure global illumination settings
        self.configure_global_illumination(stage)

    def configure_global_illumination(self, stage):
        """Configure global illumination for realistic light bouncing"""
        # In Isaac Sim, GI settings are configured via rendering extensions
        # For USD stage, we'll set up appropriate lighting properties

        # Enable multi-bounce lighting
        dome_light = stage.GetPrimAtPath("/World/DomeLight")
        if dome_light.IsValid():
            # Set exposure to control overall brightness
            dome_light.GetAttribute("exposure").Set(0.0)

        # Configure light falloff and area properties for realistic shadows
        key_light = stage.GetPrimAtPath("/World/KeyLight")
        if key_light.IsValid():
            # For distant lights, we can't set area directly, but we can set angular size
            # to affect shadow softness
            key_light.GetAttribute("angle").Set(0.5)  # Angular size affects shadow softness

    def implement_ray_traced_shadows(self):
        """Implement realistic ray-traced shadows"""
        shadow_config = {
            'soft_shadows': True,
            'ray_traced_shadows': True,
            'shadow_map_resolution': 2048,
            'shadow_softness': 0.5,  # Controls penumbra size
            'shadow_bias': 0.001,    # Prevents shadow acne
            'max_shadow_distance': 50.0,  # Maximum distance for shadow calculation
            'shadow_fade_distance': 40.0  # Distance where shadows start fading
        }

        # Ray-traced shadows provide realistic penumbra and umbra
        # Unlike traditional shadow maps, they calculate shadows through ray tracing
        # This results in more accurate shadows with proper softness based on light size

        return shadow_config

    def setup_environment_mapping(self):
        """Setup environment mapping for realistic reflections"""
        env_config = {
            'hdri_environment': 'textures/outdoor_environment.hdr',
            'reflection_probe': {
                'resolution': (512, 512),
                'update_frequency': 'realtime',  # or 'static', 'dynamic'
                'influence_distance': 10.0
            },
            'irradiance_cache': {
                'enabled': True,
                'resolution': 64,
                'max_distance': 50.0
            }
        }

        return env_config

    def implement_subsurface_scattering(self):
        """Implement subsurface scattering for realistic materials"""
        # Subsurface scattering is important for materials like skin, wax, marble
        # It simulates light penetrating the surface and scattering beneath

        sss_config = {
            'materials_with_sss': ['human_skin', 'wax', 'marble', 'milk'],
            'scattering_radius': {
                'red': 1.0,
                'green': 0.5,
                'blue': 0.1
            },
            'scattering_color': [0.9, 0.5, 0.3],  # Color of scattered light
            'thickness_map': 'textures/scattering_thickness.png',
            'scale': 1.0
        }

        return sss_config

    def implement_anisotropic_materials(self):
        """Implement anisotropic materials for brushed metals and wood grains"""
        aniso_config = {
            'anisotropic_materials': ['brushed_metal', 'wood_grain', 'hair', 'fabric'],
            'tangent_direction': 'computed_from_texture',  # or 'explicit', 'from_geometry'
            'anisotropic_direction': [1.0, 0.0, 0.0],  # Default direction
            'anisotropic_rotation': 0.0,  # Rotation of anisotropic direction
            'anisotropic_scale': 1.0
        }

        return aniso_config
```

### Advanced Camera Simulation

```python
class AdvancedCameraSimulator:
    """Advanced camera simulation with realistic optical properties"""

    def __init__(self):
        self.camera_settings = {
            'optical_properties': {
                'focal_length': 35.0,  # mm
                'sensor_width': 36.0,  # mm (full frame)
                'sensor_height': 24.0,  # mm (full frame)
                'aperture': 2.8,       # f-stop
                'focus_distance': 2.0,  # meters
                'shutter_speed': 1.0/60,  # seconds (60fps)
                'iso': 100
            },
            'distortion': {
                'k1': -0.1,    # Radial distortion coefficients
                'k2': 0.02,
                'k3': 0.0,
                'p1': 0.001,   # Tangential distortion
                'p2': -0.001
            },
            'noise_model': {
                'type': 'realistic',  # 'gaussian', 'poisson', 'realistic'
                'photon_noise': True,  # Signal-dependent noise
                'readout_noise': 0.005,  # Fixed noise
                'dark_current_noise': 0.001,  # Thermal noise
                'hot_pixels': 0.001,    # Hot pixel probability
                'dead_pixels': 0.0005   # Dead pixel probability
            },
            'advanced_effects': {
                'chromatic_aberration': {
                    'enabled': True,
                    'strength': 0.01,
                    'falloff': 0.5
                },
                'vignette': {
                    'enabled': True,
                    'strength': 0.2,
                    'falloff': 0.5
                },
                'lens_flare': {
                    'enabled': True,
                    'intensity': 0.1,
                    'quality': 'high'
                },
                'motion_blur': {
                    'enabled': True,
                    'shutter_angle': 180,  # degrees
                    'samples': 8
                }
            }
        }

    def simulate_realistic_camera_optics(self, scene_data):
        """Simulate realistic camera optical effects"""
        # Apply lens distortion
        distorted_image = self.apply_lens_distortion(scene_data['image'])

        # Apply chromatic aberration
        chromatic_image = self.apply_chromatic_aberration(distorted_image)

        # Apply vignette
        vignetted_image = self.apply_vignette(chromatic_image)

        # Apply motion blur if scene is moving
        if 'velocity' in scene_data:
            motion_blurred_image = self.apply_motion_blur(
                vignetted_image, scene_data['velocity'])
        else:
            motion_blurred_image = vignetted_image

        # Add realistic noise
        noisy_image = self.add_realistic_camera_noise(motion_blurred_image)

        return noisy_image

    def apply_lens_distortion(self, image):
        """Apply realistic lens distortion"""
        height, width = image.shape[:2]
        k1, k2, k3 = self.camera_settings['distortion']['k1:k3']
        p1, p2 = self.camera_settings['distortion']['p1:p2']

        # Calculate principal point and focal length
        cx = width / 2
        cy = height / 2
        fx = fy = max(width, height) / 2  # Approximate focal length

        # Normalize coordinates
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        x_norm = (xx - cx) / fx
        y_norm = (yy - cy) / fy

        # Calculate distortion
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2**3

        radial_dist = 1 + k1*r2 + k2*r4 + k3*r6
        tangential_dist_x = 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        tangential_dist_y = p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm

        # Apply distortion
        x_distorted = x_norm * radial_dist + tangential_dist_x
        y_distorted = y_norm * radial_dist + tangential_dist_y

        # Convert back to pixel coordinates
        x_pixels = (x_distorted * fx + cx).astype(np.float32)
        y_pixels = (y_distorted * fy + cy).astype(np.float32)

        # Remap image
        distorted_image = cv2.remap(
            image, x_pixels, y_pixels, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT)

        return distorted_image

    def apply_chromatic_aberration(self, image):
        """Apply chromatic aberration (color fringing at edges)"""
        if not self.camera_settings['advanced_effects']['chromatic_aberration']['enabled']:
            return image

        strength = self.camera_settings['advanced_effects']['chromatic_aberration']['strength']
        height, width = image.shape[:2]

        # Split image into color channels
        if len(image.shape) == 3:
            b, g, r = cv2.split(image.astype(np.float32))
        else:
            return image  # Grayscale, no chromatic aberration

        # Calculate radial distance from center
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        cx, cy = width / 2, height / 2
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2) / max(width, height)

        # Apply different scaling to each channel (red and blue expand/shrink differently)
        # This creates the chromatic separation effect
        scale_red = 1 + strength * r**2
        scale_blue = 1 - strength * r**2

        # Create coordinate grids for remapping
        x_red = ((xx - cx) * scale_red + cx).astype(np.float32)
        y_red = ((yy - cy) * scale_red + cy).astype(np.float32)
        x_blue = ((xx - cx) * scale_blue + cx).astype(np.float32)
        y_blue = ((yy - cy) * scale_blue + cy).astype(np.float32)

        # Remap red and blue channels
        r_remapped = cv2.remap(r, x_red, y_red, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b_remapped = cv2.remap(b, x_blue, y_blue, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Combine channels back
        corrected_image = cv2.merge([b_remapped, g, r_remapped])
        return np.clip(corrected_image, 0, 255).astype(np.uint8)

    def apply_vignette(self, image):
        """Apply vignette (darkening at corners)"""
        if not self.camera_settings['advanced_effects']['vignette']['enabled']:
            return image

        strength = self.camera_settings['advanced_effects']['vignette']['strength']
        height, width = image.shape[:2]

        # Calculate distance from center
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        cx, cy = width / 2, height / 2
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2) / np.sqrt(cx**2 + cy**2)

        # Calculate vignette effect
        vignette = 1 - strength * r**2

        # Apply vignette to each channel
        if len(image.shape) == 3:
            # Apply to each channel
            for i in range(3):
                image[:, :, i] = image[:, :, i] * vignette
        else:
            # Grayscale
            image = image * vignette

        return np.clip(image, 0, 255).astype(np.uint8)

    def apply_motion_blur(self, image, velocity):
        """Apply motion blur based on camera/object velocity"""
        if not self.camera_settings['advanced_effects']['motion_blur']['enabled']:
            return image

        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude < 0.01:  # Below threshold, no blur needed
            return image

        shutter_angle = self.camera_settings['advanced_effects']['motion_blur']['shutter_angle']
        blur_strength = min(velocity_magnitude * shutter_angle / 180.0, 10)  # Cap blur strength

        # Create motion blur kernel
        kernel_size = int(blur_strength) + 1
        if kernel_size <= 1:
            return image

        # Calculate blur direction from velocity
        angle = np.arctan2(velocity[1], velocity[0])  # Y, X components

        # Create directional kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        # Fill the kernel along the motion direction
        for i in range(kernel_size):
            x = int(center + (i - center) * np.cos(angle))
            y = int(center + (i - center) * np.sin(angle))

            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0 / kernel_size

        # Apply blur
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image

    def add_realistic_camera_noise(self, image):
        """Add realistic camera noise including photon noise"""
        noisy_image = image.astype(np.float32)

        # Add photon noise (signal-dependent)
        # Photon noise is proportional to the square root of signal
        photon_noise = np.random.poisson(noisy_image / 255.0 * 100) / 100.0 * 255.0
        photon_noise = photon_noise - np.mean(photon_noise, axis=(0,1), keepdims=True)  # Center around 0

        # Add readout noise (fixed pattern noise)
        readout_noise = np.random.normal(
            0, self.camera_settings['noise_model']['readout_noise'] * 255, image.shape)

        # Add dark current noise (thermal noise)
        dark_noise = np.random.normal(
            0, self.camera_settings['noise_model']['dark_current_noise'] * 255, image.shape)

        # Combine all noise sources
        total_noise = photon_noise + readout_noise + dark_noise

        # Apply to image
        noisy_image = noisy_image + total_noise

        # Add hot/dead pixels occasionally
        if self.camera_settings['noise_model']['hot_pixels'] > 0:
            self.add_hot_dead_pixels(noisy_image)

        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_hot_dead_pixels(self, image):
        """Add hot and dead pixels to simulate sensor defects"""
        height, width = image.shape[:2]

        # Hot pixels (bright spots)
        n_hot_pixels = int(height * width * self.camera_settings['noise_model']['hot_pixels'])
        for _ in range(n_hot_pixels):
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            if len(image.shape) == 3:
                # Make pixel very bright (hot)
                image[y, x, :] = 255
            else:
                image[y, x] = 255

        # Dead pixels (dark spots)
        n_dead_pixels = int(height * width * self.camera_settings['noise_model']['dead_pixels'])
        for _ in range(n_dead_pixels):
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            if len(image.shape) == 3:
                # Make pixel very dark (dead)
                image[y, x, :] = 0
            else:
                image[y, x] = 0

    def simulate_depth_of_field(self, depth_map):
        """Simulate depth of field effect"""
        if not self.camera_settings['advanced_effects']['depth_of_field']['enabled']:
            return np.ones_like(depth_map)

        aperture = self.camera_settings['optical_properties']['aperture']
        focus_distance = self.camera_settings['optical_properties']['focus_distance']

        # Calculate circle of confusion (CoC) for each pixel
        # CoC = (f^2) / (N * d * (d - f))
        # Where f=focal_length, N=aperture, d=focus_distance
        focal_length = self.camera_settings['optical_properties']['focal_length']
        coc = np.abs((focal_length**2) / (aperture * focus_distance * (depth_map - focal_length + 1e-6)))

        # Normalize CoC to 0-1 range
        coc = np.clip(coc, 0, 1)

        return coc  # 0 = in focus, 1 = maximally blurred

    def get_realistic_intrinsics(self):
        """Get realistic camera intrinsic parameters"""
        # Based on the camera's optical properties, calculate realistic intrinsics
        width, height = self.camera_settings['resolution']
        focal_length_mm = self.camera_settings['optical_properties']['focal_length']
        sensor_width_mm = self.camera_settings['optical_properties']['sensor_width']

        # Convert focal length from mm to pixels
        fx = (focal_length_mm / sensor_width_mm) * width
        fy = fx  # Assume square pixels

        # Principal point (assume centered)
        cx = width / 2
        cy = height / 2

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        return intrinsics

    def simulate_lens_flare(self, image, light_sources):
        """Simulate lens flare from bright light sources"""
        if not self.camera_settings['advanced_effects']['lens_flare']['enabled']:
            return image

        if not light_sources:
            return image

        # For each bright light source, add lens flare artifacts
        flared_image = image.copy().astype(np.float32)

        for light_pos, light_intensity in light_sources:
            if light_intensity > 0.5:  # Only significant light sources
                # Calculate flare position based on light position
                flare_x, flare_y = light_pos

                # Add ghost reflections (multiple smaller flares)
                for i in range(3):
                    ghost_x = flare_x + np.random.uniform(-50, 50)
                    ghost_y = flare_y + np.random.uniform(-50, 50)

                    # Add colored ghost flare
                    ghost_size = int(10 / (i + 1))
                    if 0 <= int(ghost_y) < image.shape[0] and 0 <= int(ghost_x) < image.shape[1]:
                        # Create colored flare based on light spectrum
                        flare_colors = [
                            [255, 200, 100],  # Warm yellow
                            [200, 255, 150],  # Cool green
                            [150, 200, 255]   # Cool blue
                        ]

                        color = flare_colors[i % len(flare_colors)]
                        flared_image[int(ghost_y), int(ghost_x)] += np.array(color) * light_intensity * 0.1

        return np.clip(flared_image, 0, 255).astype(np.uint8)
```

## Global Illumination and Light Transport

### Physically-Based Lighting Simulation

```python
class GlobalIlluminationSimulator:
    """Simulate global illumination and complex light transport"""

    def __init__(self):
        self.lighting_config = {
            'direct_illumination': True,
            'indirect_illumination': True,
            'caustics': True,
            'multiple_scattering': True,
            'subsurface_scattering': True
        }

        self.material_properties = {
            'albedo_maps': {},
            'normal_maps': {},
            'roughness_maps': {},
            'metallic_maps': {},
            'transmission_maps': {},
            'volume_scattering_params': {}
        }

    def setup_photon_mapping(self):
        """Setup photon mapping for realistic indirect lighting"""
        photon_config = {
            'caustic_photons': 1000000,      # Number of photons for caustics
            'global_photons': 2000000,       # Number of photons for global illumination
            'volume_photons': 500000,        # Number of photons for volumetric effects
            'photon_radius': 0.05,           # Search radius for photon gathering
            'caustic_radius': 0.02,          # Smaller radius for caustics
            'global_radius': 0.1,            # Larger radius for global illumination
            'volume_radius': 0.03            # Radius for volume scattering
        }

        return photon_config

    def simulate_caustics(self, scene_config):
        """Simulate caustics (focused light patterns through refractive objects)"""
        # Caustics are light patterns created when light is focused by reflective
        # or refractive surfaces (like light through water or glass)

        caustic_effects = {
            'water_caustics': self.generate_water_caustics(scene_config),
            'glass_caustics': self.generate_glass_caustics(scene_config),
            'metallic_reflections': self.generate_metallic_caustics(scene_config)
        }

        return caustic_effects

    def generate_water_caustics(self, scene_config):
        """Generate caustics from water surfaces"""
        # Simulate caustics from water waves/ripples
        if 'water_surfaces' in scene_config:
            water_caustics = []
            for water_obj in scene_config['water_surfaces']:
                # Calculate caustic patterns based on water surface geometry
                caustic_pattern = self.calculate_water_caustic_pattern(
                    water_obj['position'], water_obj['normal'], water_obj['roughness'])
                water_caustics.append(caustic_pattern)
            return water_caustics
        return []

    def calculate_water_caustic_pattern(self, water_pos, water_normal, water_roughness):
        """Calculate caustic pattern from water surface"""
        # This would involve complex ray tracing calculations
        # For simulation, we'll create a representative pattern
        caustic_map = np.zeros((480, 640))  # Resolution of output

        # Create ripple-like caustic pattern
        for i in range(480):
            for j in range(640):
                x, y = j, i  # Pixel coordinates

                # Calculate distance from water surface center
                dist_to_center = np.sqrt((x - 320)**2 + (y - 240)**2)

                # Create caustic rings based on ripples
                ripple_pattern = np.sin(dist_to_center * 0.1 + np.random.random() * 0.5)

                # Apply intensity based on light source and distance
                intensity = max(0, 1 - dist_to_center / 400)  # Fade with distance
                caustic_map[i, j] = intensity * (ripple_pattern + 1) / 2  # Normalize to 0-1

        return caustic_map

    def generate_glass_caustics(self, scene_config):
        """Generate caustics from glass/refractive objects"""
        if 'glass_objects' in scene_config:
            glass_caustics = []
            for glass_obj in scene_config['glass_objects']:
                # Calculate caustic patterns based on glass object geometry
                caustic_pattern = self.calculate_glass_caustic_pattern(
                    glass_obj['geometry'], glass_obj['position'])
                glass_caustics.append(caustic_pattern)
            return glass_caustics
        return []

    def calculate_glass_caustic_pattern(self, glass_geometry, glass_position):
        """Calculate caustic pattern from glass object"""
        # For a simple glass object, create focused light patterns
        caustic_map = np.zeros((480, 640))

        # Calculate based on glass shape and light source positions
        # This is a simplified representation
        glass_x, glass_y = glass_position[0], glass_position[1]

        for i in range(480):
            for j in range(640):
                x, y = j, i
                # Distance from glass center
                dist = np.sqrt((x - glass_x)**2 + (y - glass_y)**2)

                # For glass, create focused patterns
                if 50 < dist < 100:  # Create ring pattern
                    angle = np.arctan2(y - glass_y, x - glass_x)
                    focus_pattern = np.sin(angle * 8)  # 8-fold symmetry for glass
                    caustic_map[i, j] = max(0, focus_pattern * 0.3)

        return caustic_map

    def implement_light_baking(self):
        """Implement light baking for static lighting"""
        # Light baking pre-calculates lighting for static objects
        # This allows for complex global illumination effects at runtime

        baking_config = {
            'lightmap_resolution': 1024,     # Resolution of baked lightmaps
            'static_objects_only': True,     # Only bake for static objects
            'occlusion_culling': True,       # Bake ambient occlusion
            'indirect_lighting': True,       # Bake bounced light
            'light_probes': {
                'enabled': True,
                'spacing': 1.0,              # 1m spacing between probes
                'resolution': 64             # Resolution of each probe
            },
            'irradiance_volumes': {
                'enabled': True,
                'voxel_resolution': 32,      # 32x32x32 voxel grid
                'max_distance': 10.0         # Maximum bake distance
            }
        }

        return baking_config

    def setup_light_probes(self, scene_bounds):
        """Setup light probe grid for dynamic lighting"""
        # Light probes capture lighting information at discrete points
        # This allows for realistic lighting of dynamic objects

        probe_spacing = 2.0  # 2m spacing
        min_x, min_y, min_z = scene_bounds['min']
        max_x, max_y, max_z = scene_bounds['max']

        x_range = np.arange(min_x, max_x, probe_spacing)
        y_range = np.arange(min_y, max_y, probe_spacing)
        z_range = np.arange(min_z, max_z, probe_spacing)

        light_probes = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    probe = {
                        'position': [x, y, z],
                        'irradiance': np.zeros(3),  # RGB irradiance
                        'incoming_directions': np.zeros((64, 3)),  # Hemisphere sampling
                        'coefficients': np.zeros((9, 3))  # SH coefficients
                    }
                    light_probes.append(probe)

        return light_probes

    def calculate_spherical_harmonics(self, light_probes):
        """Calculate spherical harmonics coefficients for light probes"""
        # Spherical harmonics provide efficient representation of lighting
        # This is used for real-time lighting of dynamic objects

        for probe in light_probes:
            # Sample incoming light directions and calculate SH coefficients
            # This is a simplified representation
            for i, direction in enumerate(probe['incoming_directions']):
                # In practice, this would involve ray tracing and lighting calculations
                # For simulation, we'll use a simple lighting model
                probe['coefficients'][:, 0] = np.random.random(9)  # R channel
                probe['coefficients'][:, 1] = np.random.random(9)  # G channel
                probe['coefficients'][:, 2] = np.random.random(9)  # B channel

        return light_probes

    def simulate_atmospheric_scattering(self, scene_config):
        """Simulate atmospheric scattering for outdoor scenes"""
        # Atmospheric scattering creates realistic sky colors, fog, and light attenuation

        atmosphere_config = {
            'rayleigh_scattering': {
                'coefficient': 0.01,         # Rayleigh scattering coefficient
                'wavelength': [0.65, 0.57, 0.475]  # R, G, B wavelengths (micrometers)
            },
            'mie_scattering': {
                'coefficient': 0.005,        # Mie scattering coefficient (for particles)
                'g': 0.8,                    # Asymmetry parameter
                'scale': 0.1                 # Height scale factor
            },
            'absorption': {
                'ozone': 0.0002,             # Ozone absorption
                'water_vapor': 0.002         # Water vapor absorption
            },
            'sky_model': 'preetham',         # Preetham sky model
            'sun_properties': {
                'intensity': 100000,          # Lux
                'angular_radius': 0.00467,   # Radians (0.27 degrees)
                'color_temperature': 6500    # Kelvin
            }
        }

        # Calculate atmospheric effects
        atmospheric_effects = self.compute_atmospheric_effects(
            scene_config, atmosphere_config)

        return atmospheric_effects

    def compute_atmospheric_effects(self, scene_config, atmosphere_config):
        """Compute atmospheric scattering effects"""
        width, height = scene_config.get('resolution', (640, 480))

        # Create sky dome with realistic atmospheric colors
        sky_map = np.zeros((height, width, 3))

        # Calculate sun position and direction
        sun_pos = scene_config.get('sun_position', [0.5, 0.5, 0.8])  # x, y, z in normalized coordinates
        sun_dir = np.array(sun_pos) / np.linalg.norm(sun_pos)

        for i in range(height):
            for j in range(width):
                # Calculate ray direction from camera to this pixel
                # This is a simplified calculation
                pixel_dir = np.array([j - width/2, i - height/2, height/2])
                pixel_dir = pixel_dir / np.linalg.norm(pixel_dir)

                # Calculate atmospheric color based on viewing direction and sun direction
                sky_color = self.calculate_atmospheric_color(
                    pixel_dir, sun_dir, atmosphere_config)
                sky_map[i, j] = sky_color

        return {
            'sky_map': sky_map,
            'scattering_coefficients': atmosphere_config,
            'fog_density': self.calculate_fog_density(scene_config)
        }

    def calculate_atmospheric_color(self, view_dir, sun_dir, atmosphere_config):
        """Calculate atmospheric color based on ray direction and sun position"""
        # Simplified atmospheric scattering calculation
        # In practice, this would use more complex Preetham or Hosek-Wilkie models

        # Angle between view direction and sun direction
        cos_theta = np.dot(view_dir, sun_dir)
        theta = np.arccos(max(-1, min(1, cos_theta)))

        # Rayleigh scattering (shorter wavelengths scattered more)
        rayleigh_factor = atmosphere_config['rayleigh_scattering']['coefficient']
        rayleigh_color = np.array(atmosphere_config['rayleigh_scattering']['wavelength'])**(-4)

        # Mie scattering (larger particles, less wavelength dependent)
        mie_factor = atmosphere_config['mie_scattering']['coefficient']
        mie_phase = (1 - atmosphere_config['mie_scattering']['g']**2) / \
                   (1 + atmosphere_config['mie_scattering']['g']**2 - 2*atmosphere_config['mie_scattering']['g']*np.cos(theta))**1.5

        # Combine effects
        sky_color = rayleigh_factor * rayleigh_color + mie_factor * mie_phase

        # Normalize and return
        sky_color = np.clip(sky_color, 0, 1)
        return sky_color

    def calculate_fog_density(self, scene_config):
        """Calculate fog density based on atmospheric conditions"""
        # Calculate fog based on distance and atmospheric properties
        # This creates realistic depth-based light attenuation

        # In practice, this would be calculated based on scene distance
        # For simulation, return a representative fog map
        width, height = scene_config.get('resolution', (640, 480))
        fog_map = np.zeros((height, width))

        # Create distance-based fog (more fog at distance)
        for i in range(height):
            for j in range(width):
                # Simplified distance from viewer
                dist = np.sqrt((j - width/2)**2 + (i - height/2)**2) / max(width, height)
                fog_map[i, j] = min(1.0, dist * 0.3)  # 30% fog at screen edges

        return fog_map
```

## Material Simulation and Texturing

### Physically-Based Rendering (PBR)

```python
class PBRSimulator:
    """Physically-Based Rendering simulator for realistic materials"""

    def __init__(self):
        self.material_types = {
            'metallic_roughness': {
                'albedo': 'base_color',
                'metallic': 'metallic_map',
                'roughness': 'roughness_map',
                'normal': 'normal_map',
                'ao': 'ambient_occlusion'
            },
            'specular_glossiness': {
                'diffuse': 'diffuse_color',
                'specular': 'specular_map',
                'glossiness': 'glossiness_map',
                'normal': 'normal_map',
                'ao': 'ambient_occlusion'
            },
            'subsurface_scattering': {
                'albedo': 'subsurface_color',
                'subsurface': 'subsurface_amount',
                'thickness': 'thickness_map',
                'normal': 'normal_map',
                'scale': 'subsurface_scale'
            }
        }

    def create_realistic_materials(self, material_type, properties):
        """Create realistic PBR materials"""
        if material_type == 'human_skin':
            return self.create_skin_material(properties)
        elif material_type == 'robot_metal':
            return self.create_metal_material(properties)
        elif material_type == 'robot_plastic':
            return self.create_plastic_material(properties)
        elif material_type == 'floor_wood':
            return self.create_wood_material(properties)
        elif material_type == 'floor_tile':
            return self.create_tile_material(properties)
        else:
            return self.create_default_material(properties)

    def create_skin_material(self, properties):
        """Create realistic human skin material"""
        skin_material = {
            'name': 'human_skin',
            'shader_type': 'subsurface_scattering',
            'base_color': properties.get('albedo', [0.8, 0.6, 0.5]),
            'subsurface_color': [0.9, 0.5, 0.3],
            'subsurface_amount': 0.3,
            'subsurface_scale': 0.1,
            'subsurface_thickness': 0.5,
            'metallic': 0.0,
            'roughness': 0.7,
            'specular': 0.5,
            'normal_map': properties.get('normal_map', 'textures/skin_normals.png'),
            'displacement_map': properties.get('displacement_map', 'textures/skin_displacement.png'),
            'sss_profile': 'skin_profile'
        }

        # Add skin-specific properties
        skin_material.update({
            'melanin': properties.get('melanin', 0.5),  # Skin pigmentation
            'hemoglobin': properties.get('hemoglobin', 0.4),  # Blood content
            'scattering_params': self.calculate_skin_scattering_params(
                skin_material['melanin'], skin_material['hemoglobin'])
        })

        return skin_material

    def calculate_skin_scattering_params(self, melanin, hemoglobin):
        """Calculate realistic skin scattering parameters based on composition"""
        # Based on real skin optical properties research
        # Scattering coefficients for red, green, blue light
        scattering_coeffs = np.array([
            [2.06, 2.16, 2.26],  # Scattering at different wavelengths
            [0.79, 0.85, 0.91],
            [0.30, 0.34, 0.38]
        ])

        # Absorption coefficients for red, green, blue light
        absorption_coeffs = np.array([
            [0.012, 0.024, 0.036],  # Baseline absorption
            [0.008, 0.016, 0.024],
            [0.004, 0.008, 0.012]
        ])

        # Adjust based on melanin and hemoglobin content
        adjusted_absorption = absorption_coeffs * (melanin * 2 + hemoglobin * 1.5)

        return {
            'scattering': scattering_coeffs,
            'absorption': adjusted_absorption,
            'anisotropy': 0.8  # Skin scattering anisotropy factor
        }

    def create_metal_material(self, properties):
        """Create realistic metallic material"""
        metal_material = {
            'name': 'robot_metal',
            'shader_type': 'metallic_roughness',
            'base_color': properties.get('albedo', [0.7, 0.75, 0.8]),
            'metallic': 0.9,
            'roughness': properties.get('roughness', 0.2),
            'specular': 1.0,
            'normal_map': properties.get('normal_map', 'textures/metal_normals.png'),
            'roughness_map': properties.get('roughness_map', 'textures/metal_roughness.png'),
            'metallic_map': properties.get('metallic_map', 'textures/metal_metallic.png'),
            'anisotropic': properties.get('anisotropic', 0.0)
        }

        # Add metal-specific properties
        metal_material.update({
            'reflectance': 0.95,  # High reflectance for metals
            'ior': 1.33,        # Index of refraction for typical metals
            'anisotropic_rotation': properties.get('anisotropic_rotation', 0.0)
        })

        return metal_material

    def create_plastic_material(self, properties):
        """Create realistic plastic material"""
        plastic_material = {
            'name': 'robot_plastic',
            'shader_type': 'metallic_roughness',
            'base_color': properties.get('albedo', [0.8, 0.8, 0.9]),
            'metallic': 0.0,
            'roughness': properties.get('roughness', 0.4),
            'specular': 0.5,
            'ior': 1.49,  # Acrylic plastic IOR
            'normal_map': properties.get('normal_map', 'textures/plastic_normals.png'),
            'specular_map': properties.get('specular_map', 'textures/plastic_specular.png')
        }

        # Add plastic-specific properties
        plastic_material.update({
            'transmission': properties.get('transmission', 0.0),  # For translucent plastics
            'thin_walled': properties.get('thin_walled', False),  # Thin wall transmission
            'subsurface': properties.get('subsurface', 0.1)  # For translucent effects
        })

        return plastic_material

    def create_wood_material(self, properties):
        """Create realistic wood material with grain patterns"""
        wood_material = {
            'name': 'floor_wood',
            'shader_type': 'metallic_roughness',
            'base_color': properties.get('albedo', [0.6, 0.4, 0.2]),
            'metallic': 0.0,
            'roughness': 0.6,
            'specular': 0.3,
            'anisotropic': 0.5,
            'normal_map': properties.get('normal_map', 'textures/wood_normals.png'),
            'diffuse_map': properties.get('diffuse_map', 'textures/wood_diffuse.png'),
            'specular_map': properties.get('specular_map', 'textures/wood_specular.png')
        }

        # Add wood-specific properties
        wood_material.update({
            'grain_direction': properties.get('grain_direction', [1.0, 0.0, 0.0]),
            'ring_pattern': properties.get('ring_pattern', True),
            'annual_ring_frequency': properties.get('annual_ring_freq', 0.1),
            'ray_pattern': properties.get('ray_pattern', True),
            'ray_frequency': properties.get('ray_freq', 0.05)
        })

        return wood_material

    def create_tile_material(self, properties):
        """Create realistic tile material"""
        tile_material = {
            'name': 'floor_tile',
            'shader_type': 'metallic_roughness',
            'base_color': properties.get('albedo', [0.8, 0.8, 0.8]),
            'metallic': 0.0,
            'roughness': 0.3,
            'specular': 0.5,
            'normal_map': properties.get('normal_map', 'textures/tile_normals.png'),
            'diffuse_map': properties.get('diffuse_map', 'textures/tile_diffuse.png'),
            'specular_map': properties.get('specular_map', 'textures/tile_specular.png')
        }

        # Add tile-specific properties
        tile_material.update({
            'grout_lines': properties.get('grout_lines', True),
            'grout_color': properties.get('grout_color', [0.4, 0.4, 0.4]),
            'grout_width': properties.get('grout_width', 0.002),  # 2mm grout
            'tile_pattern': properties.get('tile_pattern', 'rectangular'),
            'tile_size': properties.get('tile_size', [0.3, 0.3])  # 30cm x 30cm
        })

        return tile_material

    def create_default_material(self, properties):
        """Create default material with basic PBR properties"""
        return {
            'name': 'default_material',
            'shader_type': 'metallic_roughness',
            'base_color': properties.get('albedo', [0.5, 0.5, 0.5]),
            'metallic': 0.0,
            'roughness': 0.5,
            'specular': 0.5
        }

    def apply_material_properties_to_mesh(self, mesh, material):
        """Apply material properties to a 3D mesh"""
        # This would interface with the rendering engine to apply materials
        # For simulation, we'll return the mesh with material information

        mesh_with_material = {
            'mesh': mesh,
            'material': material,
            'uv_coordinates': self.generate_uv_coordinates(mesh),
            'texture_coordinates': self.generate_texture_coordinates(mesh),
            'tangent_space': self.calculate_tangent_space(mesh)
        }

        return mesh_with_material

    def generate_uv_coordinates(self, mesh):
        """Generate UV coordinates for texture mapping"""
        # Calculate UV coordinates using different methods based on mesh type
        if mesh['type'] == 'box':
            return self.generate_box_uv_coordinates(mesh)
        elif mesh['type'] == 'cylinder':
            return self.generate_cylinder_uv_coordinates(mesh)
        elif mesh['type'] == 'sphere':
            return self.generate_sphere_uv_coordinates(mesh)
        elif mesh['type'] == 'mesh':
            return self.generate_mesh_uv_coordinates(mesh)
        else:
            # Default to simple planar mapping
            return self.generate_planar_uv_coordinates(mesh)

    def generate_box_uv_coordinates(self, mesh):
        """Generate UV coordinates for box geometry"""
        # Simple box UV mapping
        # Each face gets its own UV space
        vertices = mesh['vertices']
        faces = mesh['faces']

        uv_coords = []
        for face in faces:
            face_vertices = [vertices[i] for i in face]
            # Calculate UV for each vertex in the face
            for vertex in face_vertices:
                # Map vertex position to UV (simplified)
                u = (vertex[0] + 1) / 2  # Normalize to 0-1
                v = (vertex[1] + 1) / 2
                uv_coords.append([u, v])

        return np.array(uv_coords)

    def generate_cylinder_uv_coordinates(self, mesh):
        """Generate UV coordinates for cylinder geometry"""
        vertices = mesh['vertices']

        uv_coords = []
        for vertex in vertices:
            x, y, z = vertex

            # Calculate U from angle around cylinder
            u = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
            # Calculate V from height
            v = (z + 1) / 2  # Assuming cylinder goes from -1 to 1 in z

            uv_coords.append([u, v])

        return np.array(uv_coords)

    def generate_sphere_uv_coordinates(self, mesh):
        """Generate UV coordinates for sphere geometry"""
        vertices = mesh['vertices']

        uv_coords = []
        for vertex in vertices:
            x, y, z = vertex

            # Calculate longitude and latitude
            u = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
            v = np.arcsin(z) / np.pi + 0.5

            uv_coords.append([u, v])

        return np.array(uv_coords)

    def calculate_tangent_space(self, mesh):
        """Calculate tangent space for normal mapping"""
        vertices = mesh['vertices']
        normals = mesh['normals']
        uv_coords = mesh['uv_coordinates']

        tangents = []
        bitangents = []

        for face in mesh['faces']:
            # Calculate tangent and bitangent for each triangle
            v0, v1, v2 = [vertices[i] for i in face[:3]]
            uv0, uv1, uv2 = [uv_coords[i] for i in face[:3]]

            # Calculate edge vectors
            edge1 = v1 - v0
            edge2 = v2 - v0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0

            # Calculate tangent and bitangent
            denom = delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1]
            if abs(denom) < 1e-6:
                tangent = np.array([1, 0, 0])
                bitangent = np.array([0, 1, 0])
            else:
                r = 1.0 / denom
                tangent = (delta_uv2[1] * edge1 - delta_uv1[1] * edge2) * r
                bitangent = (-delta_uv2[0] * edge1 + delta_uv1[0] * edge2) * r

            # Orthogonalize tangent with respect to normal
            tangent = tangent - normals[face[0]] * np.dot(normals[face[0]], tangent)
            tangent = tangent / np.linalg.norm(tangent)

            # Calculate handedness
            tangent_cross = np.cross(normals[face[0]], tangent)
            if np.dot(tangent_cross, bitangent) < 0.0:
                tangent = -tangent

            tangents.extend([tangent] * 3)
            bitangents.extend([bitangent] * 3)

        return {
            'tangents': np.array(tangents),
            'bitangents': np.array(bitangents)
        }

    def implement_texture_streaming(self):
        """Implement texture streaming for large environments"""
        # Texture streaming loads textures on-demand based on visibility
        # This optimizes memory usage for large-scale environments

        texture_streaming_config = {
            'enabled': True,
            'streaming_distance': 50.0,      # Load textures within 50m
            'lod_bias': 0.5,                # Level of detail bias
            'max_texture_memory': 1024,     # MB maximum texture memory
            'preload_radius': 10.0,         # Preload textures within 10m
            'streaming_resolution': 'auto'  # Auto-select resolution based on distance
        }

        return texture_streaming_config

    def setup_material_property_maps(self, material_name, texture_paths):
        """Setup material property maps (albedo, normal, roughness, etc.)"""
        material_maps = {}

        # Load or generate texture maps
        for property_name, texture_path in texture_paths.items():
            if property_name == 'albedo':
                material_maps['diffuse_color'] = self.load_texture(texture_path)
            elif property_name == 'normal':
                material_maps['normal'] = self.load_texture(texture_path)
            elif property_name == 'roughness':
                material_maps['roughness'] = self.load_texture(texture_path)
            elif property_name == 'metallic':
                material_maps['metallic'] = self.load_texture(texture_path)
            elif property_name == 'specular':
                material_maps['specular'] = self.load_texture(texture_path)
            elif property_name == 'ao':  # Ambient Occlusion
                material_maps['ambient_occlusion'] = self.load_texture(texture_path)
            elif property_name == 'displacement':
                material_maps['displacement'] = self.load_texture(texture_path)

        return material_maps

    def load_texture(self, texture_path):
        """Load texture from file path"""
        # In practice, this would load the actual texture file
        # For simulation, return a placeholder
        if os.path.exists(texture_path):
            # Load actual texture
            texture = cv2.imread(texture_path)
            return texture
        else:
            # Generate procedural texture
            return self.generate_procedural_texture(texture_path)

    def generate_procedural_texture(self, texture_path):
        """Generate procedural texture when file not available"""
        # Generate simple procedural texture based on name
        if 'wood' in texture_path.lower():
            return self.generate_procedural_wood_texture()
        elif 'metal' in texture_path.lower():
            return self.generate_procedural_metal_texture()
        elif 'skin' in texture_path.lower():
            return self.generate_procedural_skin_texture()
        elif 'tile' in texture_path.lower():
            return self.generate_procedural_tile_texture()
        else:
            # Default to simple gradient
            height, width = 512, 512
            texture = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    texture[i, j] = [int(255 * i / height), int(255 * j / width), 128]
            return texture

    def generate_procedural_wood_texture(self):
        """Generate procedural wood texture"""
        height, width = 512, 512
        texture = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # Wood grain pattern
                grain = np.sin(j * 0.1) * 0.5 + 0.5
                rings = np.sin(i * 0.05) * 0.3 + 0.3
                noise = np.random.random() * 0.1

                # Wood color variations
                r = int(150 + grain * 50 + rings * 30 + noise * 20)
                g = int(100 + grain * 40 + rings * 20 + noise * 15)
                b = int(50 + grain * 30 + rings * 10 + noise * 10)

                texture[i, j] = [b, g, r]  # OpenCV uses BGR

        return texture

    def generate_procedural_metal_texture(self):
        """Generate procedural metal texture"""
        height, width = 512, 512
        texture = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # Base metal color
                base_r = 200 + np.random.randint(-30, 30)
                base_g = 200 + np.random.randint(-30, 30)
                base_b = 220 + np.random.randint(-30, 30)

                # Add scratches and imperfections
                scratch = np.sin(j * 0.05 + i * 0.02) * 0.2
                imperfection = np.random.random() * 0.1

                r = int(max(0, min(255, base_r + scratch * 50 + imperfection * 30)))
                g = int(max(0, min(255, base_g + scratch * 50 + imperfection * 30)))
                b = int(max(0, min(255, base_b + scratch * 50 + imperfection * 30)))

                texture[i, j] = [b, g, r]

        return texture

    def generate_procedural_skin_texture(self):
        """Generate procedural skin texture"""
        height, width = 512, 512
        texture = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # Base skin tone
                base_r = 220 + np.random.randint(-40, 20)
                base_g = 180 + np.random.randint(-40, 20)
                base_b = 150 + np.random.randint(-40, 20)

                # Add subsurface scattering effect
                subsurface = np.sin((i + j) * 0.03) * 0.1

                # Add pores and skin texture
                pore_pattern = np.random.random() * 0.05

                r = int(max(0, min(255, base_r + subsurface * 30 + pore_pattern * 20)))
                g = int(max(0, min(255, base_g + subsurface * 20 + pore_pattern * 15)))
                b = int(max(0, min(255, base_b + subsurface * 10 + pore_pattern * 10)))

                texture[i, j] = [b, g, r]

        return texture

    def generate_procedural_tile_texture(self):
        """Generate procedural tile texture"""
        height, width = 512, 512
        texture = np.zeros((height, width, 3), dtype=np.uint8)

        tile_size = 64  # pixels per tile
        grout_width = 2  # pixels

        for i in range(height):
            for j in range(width):
                # Calculate which tile this pixel belongs to
                tile_i = i // tile_size
                tile_j = j // tile_size

                # Calculate position within tile
                pos_in_tile_i = i % tile_size
                pos_in_tile_j = j % tile_size

                # Check if this is a grout line
                is_grout = (pos_in_tile_i < grout_width or
                           pos_in_tile_i >= tile_size - grout_width or
                           pos_in_tile_j < grout_width or
                           pos_in_tile_j >= tile_size - grout_width)

                if is_grout:
                    # Grout color (darker)
                    color_val = 100 + np.random.randint(-20, 20)
                else:
                    # Tile color (lighter)
                    color_val = 200 + np.random.randint(-30, 30)

                texture[i, j] = [color_val, color_val, color_val]

        return texture
```

## Performance Optimization for Rendering

### Efficient Rendering Techniques

```python
class RenderingOptimizer:
    """Optimize rendering performance for photorealistic humanoid simulation"""

    def __init__(self):
        self.optimization_settings = {
            'level_of_detail': {
                'enabled': True,
                'distance_thresholds': [1.0, 3.0, 10.0],  # Different LOD at different distances
                'quality_levels': ['high', 'medium', 'low']
            },
            'occlusion_culling': {
                'enabled': True,
                'algorithm': 'hardware',  # hardware or software
                'update_frequency': 60    # Hz
            },
            'frustum_culling': {
                'enabled': True,
                'padding': 0.1  # Extra padding for safety
            },
            'multi_resolution_shading': {
                'enabled': True,
                'shading_rate': 'variable',
                'quality_regions': ['high_detail', 'medium_detail', 'low_detail']
            },
            'adaptive_sampling': {
                'enabled': True,
                'min_samples': 4,
                'max_samples': 64,
                'quality_threshold': 0.01
            },
            'texture_compression': {
                'enabled': True,
                'format': 'BC7',  # Block Compression 7 for high-quality textures
                'quality': 'high'
            }
        }

    def implement_level_of_detail(self, objects):
        """Implement Level of Detail system for performance"""
        lod_objects = []

        for obj in objects:
            # Calculate distance from camera to object
            cam_pos = np.array([0, 0, 1.5])  # Camera position (head of robot)
            obj_pos = np.array(obj['position'])
            distance = np.linalg.norm(cam_pos - obj_pos)

            # Select LOD based on distance
            if distance < self.optimization_settings['level_of_detail']['distance_thresholds'][0]:
                lod_level = 'high'
                quality = 1.0
            elif distance < self.optimization_settings['level_of_detail']['distance_thresholds'][1]:
                lod_level = 'medium'
                quality = 0.5
            else:
                lod_level = 'low'
                quality = 0.2

            # Apply LOD to object
            lod_obj = self.apply_lod_to_object(obj, lod_level)
            lod_objects.append(lod_obj)

        return lod_objects

    def apply_lod_to_object(self, obj, lod_level):
        """Apply Level of Detail to a single object"""
        lod_obj = obj.copy()

        if lod_level == 'high':
            # Keep original quality
            pass
        elif lod_level == 'medium':
            # Reduce mesh complexity
            if 'mesh_resolution' in lod_obj:
                lod_obj['mesh_resolution'] = max(100, lod_obj['mesh_resolution'] // 2)
            # Reduce texture resolution
            if 'material' in lod_obj and 'texture_resolution' in lod_obj['material']:
                lod_obj['material']['texture_resolution'] //= 2
        elif lod_level == 'low':
            # Use simplified proxy geometry
            if 'mesh_resolution' in lod_obj:
                lod_obj['mesh_resolution'] = max(20, lod_obj['mesh_resolution'] // 4)
            # Use lower quality materials
            if 'material' in lod_obj:
                lod_obj['material']['quality'] = 'low'
                if 'texture_resolution' in lod_obj['material']:
                    lod_obj['material']['texture_resolution'] //= 4

        return lod_obj

    def setup_frustum_culling(self, camera_params):
        """Setup view frustum culling for performance"""
        # Define view frustum based on camera parameters
        fov = camera_params.get('fov', 1.396)  # 80 degrees in radians
        aspect_ratio = camera_params.get('aspect_ratio', 1.33)  # 4:3 aspect
        near_clip = camera_params.get('near_clip', 0.1)
        far_clip = camera_params.get('far_clip', 100.0)

        # Calculate frustum planes
        half_fov = fov / 2
        tan_half_fov = np.tan(half_fov)

        frustum_planes = {
            'near': {'distance': near_clip, 'normal': np.array([0, 0, -1])},
            'far': {'distance': far_clip, 'normal': np.array([0, 0, 1])},
            'left': {'distance': 0, 'normal': np.array([1, 0, 0])},
            'right': {'distance': 0, 'normal': np.array([-1, 0, 0])},
            'top': {'distance': 0, 'normal': np.array([0, -1, 0])},
            'bottom': {'distance': 0, 'normal': np.array([0, 1, 0])}
        }

        # Calculate actual plane normals based on camera orientation
        cam_orientation = R.from_quat(camera_params.get('orientation', [0, 0, 0, 1]))

        for plane_name, plane in frustum_planes.items():
            frustum_planes[plane_name]['normal'] = cam_orientation.apply(plane['normal'])

        return frustum_planes

    def setup_occlusion_culling(self):
        """Setup occlusion culling for hidden object removal"""
        # Use hardware occlusion queries or software-based culling
        # This would typically use the graphics API's occlusion culling features

        occlusion_culling_config = {
            'method': 'hardware_queries',  # or 'software_culling'
            'query_frequency': 30,  # Hz
            'conservative_culling': True,  # Don't cull if uncertain
            'cull_small_objects': True,  # Cull objects that are too small to be visible
            'minimum_screen_size': 0.001  # 0.1% of screen
        }

        return occlusion_culling_config

    def implement_adaptive_rendering(self, scene_complexity):
        """Implement adaptive rendering based on scene complexity"""
        # Adjust rendering quality based on scene complexity to maintain performance

        if scene_complexity < 0.3:  # Simple scene
            rendering_config = {
                'quality': 'high',
                'samples_per_pixel': 32,
                'max_bounces': 8,
                'enable_denoising': True,
                'enable_advanced_effects': True
            }
        elif scene_complexity < 0.7:  # Medium complexity
            rendering_config = {
                'quality': 'medium',
                'samples_per_pixel': 16,
                'max_bounces': 6,
                'enable_denoising': True,
                'enable_advanced_effects': True
            }
        else:  # Complex scene
            rendering_config = {
                'quality': 'performance',
                'samples_per_pixel': 8,
                'max_bounces': 4,
                'enable_denoising': False,
                'enable_advanced_effects': False
            }

        return rendering_config

    def optimize_material_rendering(self, materials):
        """Optimize material rendering for performance"""
        optimized_materials = []

        for material in materials:
            optimized_mat = material.copy()

            # Optimize material complexity based on distance/use
            if material.get('complexity', 'high') == 'high':
                # For complex materials, consider simplifying when far away
                optimized_mat['use_simplified_shader'] = False
            elif material.get('complexity', 'high') == 'medium':
                optimized_mat['use_simplified_shader'] = False
            else:  # low complexity
                optimized_mat['use_simplified_shader'] = True

            # Optimize texture streaming
            optimized_mat['streaming_enabled'] = True
            optimized_mat['mipmapping'] = True
            optimized_mat['anisotropic_filtering'] = 4  # 4x anisotropic filtering

            optimized_materials.append(optimized_mat)

        return optimized_materials

    def implement_multi_resolution_shading(self):
        """Implement multi-resolution shading for performance"""
        # Divide screen into regions with different shading quality
        # High detail where eyes are likely to focus, low detail in peripheral areas

        mrs_config = {
            'enabled': True,
            'technique': 'foveated_rendering',  # or 'region_based'
            'quality_zones': {
                'center': {
                    'radius': 0.3,  # 30% of screen radius
                    'shading_rate': 1.0,  # Full resolution
                    'samples_per_pixel': 32
                },
                'middle': {
                    'radius': 0.6,  # 60% of screen radius
                    'shading_rate': 0.5,  # Half resolution
                    'samples_per_pixel': 16
                },
                'outer': {
                    'radius': 1.0,  # Full screen
                    'shading_rate': 0.25,  # Quarter resolution
                    'samples_per_pixel': 8
                }
            },
            'eye_tracking_simulation': True,  # Simulate where user is looking
            'focus_smoothing': True  # Smooth transitions between zones
        }

        return mrs_config

    def setup_texture_compression(self):
        """Setup texture compression for memory optimization"""
        compression_config = {
            'format': 'BC7',  # Best quality block compression
            'quality': 'high',
            'mipmap_generation': True,
            'streaming_compression': True,
            'memory_budget': 512,  # MB for textures
            'compression_threads': 4
        }

        return compression_config

    def monitor_rendering_performance(self):
        """Monitor rendering performance metrics"""
        performance_metrics = {
            'frames_per_second': 0,
            'render_time_ms': 0,
            'gpu_utilization': 0,
            'memory_usage_mb': 0,
            'draw_calls': 0,
            'triangles_rendered': 0,
            'overdraw_factor': 0
        }

        # These would be measured during actual rendering
        # For simulation, return typical values
        return {
            'frames_per_second': 60,
            'render_time_ms': 16.7,
            'gpu_utilization': 0.7,
            'memory_usage_mb': 256,
            'draw_calls': 50,
            'triangles_rendered': 100000,
            'overdraw_factor': 1.5
        }

    def generate_performance_report(self, metrics):
        """Generate rendering performance report"""
        report = {
            'performance_summary': {
                'frame_rate': metrics['frames_per_second'],
                'render_time': metrics['render_time_ms'],
                'gpu_usage': f"{metrics['gpu_utilization']*100:.1f}%",
                'memory_usage': f"{metrics['memory_usage_mb']:.1f}MB"
            },
            'optimization_recommendations': self.analyze_performance_bottlenecks(metrics),
            'quality_performance_tradeoff': self.calculate_quality_performance_ratio(metrics),
            'hardware_requirements': self.estimate_hardware_requirements(metrics)
        }

        return report

    def analyze_performance_bottlenecks(self, metrics):
        """Analyze rendering performance bottlenecks"""
        recommendations = []

        if metrics['frames_per_second'] < 30:  # Target 30fps minimum
            recommendations.append("Frame rate is low - consider reducing rendering quality or complexity")

        if metrics['gpu_utilization'] > 0.9:  # 90% GPU usage
            recommendations.append("High GPU utilization - consider simplifying shaders or reducing samples")

        if metrics['memory_usage_mb'] > 1024:  # 1GB memory usage
            recommendations.append("High memory usage - consider texture compression or streaming")

        if metrics['draw_calls'] > 100:  # High draw call count
            recommendations.append("High draw call count - consider batching similar objects")

        if metrics['overdraw_factor'] > 2.0:  # High overdraw
            recommendations.append("High overdraw - consider depth pre-pass or occlusion culling")

        return recommendations

    def calculate_quality_performance_ratio(self, metrics):
        """Calculate quality vs performance ratio"""
        # This is a simplified metric - in practice would be more complex
        quality_score = 1.0  # Would be based on rendering settings
        performance_score = metrics['frames_per_second'] / 60.0  # Normalize to 60fps

        return quality_score * performance_score

    def estimate_hardware_requirements(self, metrics):
        """Estimate minimum hardware requirements"""
        if metrics['frames_per_second'] >= 60 and metrics['gpu_utilization'] < 0.7:
            return "Minimum: RTX 3060 or equivalent"
        elif metrics['frames_per_second'] >= 30 and metrics['gpu_utilization'] < 0.8:
            return "Minimum: GTX 1660 or equivalent"
        else:
            return "Minimum: RTX 4090 or equivalent for real-time performance"
```

## Advanced Rendering Features

### Specialized Rendering for Robotics

```python
class RoboticsSpecificRenderer:
    """Specialized rendering for robotics applications"""

    def __init__(self):
        self.render_modes = {
            'photorealistic': self.render_photorealistic,
            'line_drawing': self.render_line_drawing,
            'semantic_segmentation': self.render_semantic_segmentation,
            'instance_segmentation': self.render_instance_segmentation,
            'depth_map': self.render_depth_map,
            'normal_map': self.render_normal_map,
            'optic_flow': self.render_optic_flow
        }

        self.semantic_colors = {
            'humanoid_robot': [255, 0, 0],      # Red
            'humanoid_torso': [255, 50, 50],   # Light red
            'humanoid_head': [255, 100, 100],  # Lighter red
            'humanoid_arm': [200, 0, 0],       # Darker red
            'humanoid_leg': [150, 0, 0],       # Even darker red
            'table': [0, 255, 0],              # Green
            'chair': [0, 200, 0],              # Dark green
            'box': [0, 150, 0],                # Darker green
            'floor': [100, 100, 100],          # Gray
            'wall': [150, 150, 150],           # Light gray
            'obstacle': [255, 255, 0],         # Yellow
            'target': [0, 255, 255],           # Cyan
            'person': [255, 0, 255],           # Magenta
            'unknown': [128, 128, 128]         # Medium gray
        }

    def render_semantic_segmentation(self, scene_state):
        """Render semantic segmentation map"""
        width, height = scene_state.get('resolution', (640, 480))
        semantic_map = np.zeros((height, width, 3), dtype=np.uint8)

        # For each object in scene, render with semantic color
        for obj in scene_state.get('objects', []):
            obj_class = obj.get('class', 'unknown')
            color = self.semantic_colors.get(obj_class, self.semantic_colors['unknown'])

            # Project object to image and fill pixels
            projected_pixels = self.project_object_to_image_pixels(obj, scene_state)
            for px, py in projected_pixels:
                if 0 <= px < width and 0 <= py < height:
                    semantic_map[py, px] = color

        return semantic_map

    def render_instance_segmentation(self, scene_state):
        """Render instance segmentation map with unique colors per instance"""
        width, height = scene_state.get('resolution', (640, 480))
        instance_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate unique color for each instance
        for i, obj in enumerate(scene_state.get('objects', [])):
            # Generate unique color based on instance ID
            hue = i / len(scene_state.get('objects', [1]))  # Distribute hues
            color = self.hsv_to_rgb(hue, 0.8, 0.9)

            # Project object to image and fill pixels
            projected_pixels = self.project_object_to_image_pixels(obj, scene_state)
            for px, py in projected_pixels:
                if 0 <= px < width and 0 <= py < height:
                    instance_map[py, px] = color

        return instance_map

    def render_depth_map(self, scene_state):
        """Render depth map showing distance to each pixel"""
        width, height = scene_state.get('resolution', (640, 480))
        depth_map = np.full((height, width), np.inf, dtype=np.float32)

        # Calculate depth for each object
        camera_pos = scene_state.get('camera_position', [0, 0, 1.5])

        for obj in scene_state.get('objects', []):
            obj_pos = obj.get('position', [0, 0, 0])
            distance = np.linalg.norm(np.array(obj_pos) - np.array(camera_pos))

            # Project object to image and assign depth
            projected_pixels = self.project_object_to_image_pixels(obj, scene_state)
            for px, py in projected_pixels:
                if 0 <= px < width and 0 <= py < height:
                    # Only assign if closer than existing depth (for occlusion)
                    if distance < depth_map[py, px]:
                        depth_map[py, px] = distance

        # Convert to 8-bit for visualization (normalize to 0-255)
        if np.any(np.isfinite(depth_map)):
            depth_map = np.clip(depth_map, 0, 10)  # Clip to 0-10m range
            depth_map = (depth_map / 10 * 255).astype(np.uint8)
        else:
            depth_map = np.zeros((height, width), dtype=np.uint8)

        return depth_map

    def render_normal_map(self, scene_state):
        """Render normal map showing surface orientations"""
        width, height = scene_state.get('resolution', (640, 480))
        normal_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate surface normals for each pixel
        # This is a simplified approach - in practice would use actual surface geometry
        for obj in scene_state.get('objects', []):
            # Calculate object's surface normals based on its geometry
            obj_normals = self.calculate_object_normals(obj)

            # Project normals to image pixels
            projected_pixels = self.project_object_to_image_pixels(obj, scene_state)
            for i, (px, py) in enumerate(projected_pixels):
                if 0 <= px < width and 0 <= py < height:
                    # Convert normal vector to RGB (normalize from [-1,1] to [0,255])
                    if i < len(obj_normals):
                        normal = obj_normals[i]
                        rgb_normal = ((normal + 1) / 2 * 255).astype(np.uint8)
                        normal_map[py, px] = rgb_normal

        return normal_map

    def render_optic_flow(self, scene_state, previous_scene_state):
        """Render optic flow showing motion between frames"""
        width, height = scene_state.get('resolution', (640, 480))
        optic_flow = np.zeros((height, width, 2), dtype=np.float32)

        # Calculate motion for each pixel based on object movement
        # This compares current and previous object positions
        current_objects = scene_state.get('objects', [])
        prev_objects = previous_scene_state.get('objects', [])

        # Map object positions between frames to calculate motion
        for i, curr_obj in enumerate(current_objects):
            if i < len(prev_objects):
                prev_obj = prev_objects[i]

                # Calculate object motion
                curr_pos = np.array(curr_obj.get('position', [0, 0, 0]))
                prev_pos = np.array(prev_obj.get('position', [0, 0, 0]))
                motion = curr_pos - prev_pos

                # Project motion to image space
                projected_pixels = self.project_object_to_image_pixels(curr_obj, scene_state)
                for px, py in projected_pixels:
                    if 0 <= px < width and 0 <= py < height:
                        # Store 2D motion vector (x, y components only)
                        optic_flow[py, px] = motion[:2] * 100  # Scale for visualization

        # Convert to RGB representation
        flow_rgb = self.flow_to_rgb(optic_flow)
        return flow_rgb

    def project_object_to_image_pixels(self, obj, scene_state):
        """Project object to image pixels"""
        # This would use the camera's projection matrix in practice
        # For simulation, return a set of pixels based on object position and size

        # Get object position and size
        obj_pos = np.array(obj.get('position', [0, 0, 0]))
        obj_size = obj.get('size', [0.1, 0.1, 0.1])

        # Simple perspective projection
        camera_pos = np.array(scene_state.get('camera_position', [0, 0, 1.5]))
        camera_fov = scene_state.get('camera_fov', 1.396)  # 80 degrees in radians

        # Calculate projected size based on distance
        distance = np.linalg.norm(obj_pos - camera_pos)
        if distance > 0:
            projected_size = min(100, int((obj_size[0] * 300) / distance))  # Simplified projection
        else:
            projected_size = 50

        # Calculate image position
        # For simplicity, assume camera looks along -Z axis
        if obj_pos[2] > 0:  # Object is in front of camera
            # Simplified projection to image coordinates
            img_x = int((obj_pos[0] / distance) * 300 + 320)  # 320 = center of 640px image
            img_y = int((obj_pos[1] / distance) * 300 + 240)  # 240 = center of 480px image

            # Generate pixels around the projected position
            pixels = []
            for dx in range(-projected_size//2, projected_size//2):
                for dy in range(-projected_size//2, projected_size//2):
                    pixels.append((img_x + dx, img_y + dy))

            return pixels
        else:
            return []  # Object is behind camera

    def calculate_object_normals(self, obj):
        """Calculate surface normals for object"""
        # For simulation, return a simple normal based on object type
        obj_type = obj.get('type', 'box')

        if obj_type == 'sphere':
            # For sphere, normal points outward from center
            center = np.array(obj.get('position', [0, 0, 0]))
            radius = obj.get('radius', 0.1)
            # Return normals for a simple sphere representation
            return [np.array([1, 0, 0])] * 10  # Simplified
        elif obj_type == 'cylinder':
            # For cylinder, return normals for curved surface and caps
            return [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])] * 3  # Simplified
        else:  # box or other
            # Return normals for 6 faces of a box
            return [
                np.array([1, 0, 0]), np.array([-1, 0, 0]),  # X faces
                np.array([0, 1, 0]), np.array([0, -1, 0]),  # Y faces
                np.array([0, 0, 1]), np.array([0, 0, -1])   # Z faces
            ]

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color"""
        if s == 0.0:
            return [int(v * 255), int(v * 255), int(v * 255)]

        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i = i % 6

        if i == 0:
            return [int(v * 255), int(t * 255), int(p * 255)]
        elif i == 1:
            return [int(q * 255), int(v * 255), int(p * 255)]
        elif i == 2:
            return [int(p * 255), int(v * 255), int(t * 255)]
        elif i == 3:
            return [int(p * 255), int(q * 255), int(v * 255)]
        elif i == 4:
            return [int(t * 255), int(p * 255), int(v * 255)]
        else:
            return [int(v * 255), int(p * 255), int(q * 255)]

    def flow_to_rgb(self, flow):
        """Convert flow field to RGB representation"""
        # Calculate magnitude and angle of flow vectors
        mag = np.linalg.norm(flow, axis=2)
        angle = np.arctan2(flow[:,:,1], flow[:,:,0])

        # Normalize magnitude
        max_mag = np.max(mag)
        if max_mag > 0:
            mag = mag / max_mag

        # Convert to HSV then to RGB
        hsv = np.zeros((*flow.shape[:2], 3))
        hsv[:,:,0] = (angle + np.pi) / (2 * np.pi)  # Hue: direction
        hsv[:,:,1] = 1.0  # Saturation: full
        hsv[:,:,2] = mag  # Value: magnitude

        # Convert HSV to RGB
        rgb = self.hsv_to_rgb_vectorized(hsv)
        return rgb.astype(np.uint8)

    def hsv_to_rgb_vectorized(self, hsv):
        """Vectorized HSV to RGB conversion"""
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        i = (h * 6).astype(int)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        i = i % 6
        rgb = np.zeros_like(hsv)

        # Create masks for each sector
        mask = i == 0
        rgb[mask, 0], rgb[mask, 1], rgb[mask, 2] = v[mask], t[mask], p[mask]

        mask = i == 1
        rgb[mask, 0], rgb[mask, 1], rgb[mask, 2] = q[mask], v[mask], p[mask]

        mask = i == 2
        rgb[mask, 0], rgb[mask, 1], rgb[mask, 2] = p[mask], v[mask], t[mask]

        mask = i == 3
        rgb[mask, 0], rgb[mask, 1], rgb[mask, 2] = p[mask], q[mask], v[mask]

        mask = i == 4
        rgb[mask, 0], rgb[mask, 1], rgb[mask, 2] = t[mask], p[mask], v[mask]

        mask = i == 5
        rgb[mask, 0], rgb[mask, 1], rgb[mask, 2] = v[mask], p[mask], q[mask]

        return rgb * 255

    def render_line_drawing(self, scene_state):
        """Render line drawing style for visualization"""
        # Convert 3D objects to 2D line drawings
        # This is useful for debugging and understanding robot structure

        width, height = scene_state.get('resolution', (640, 480))
        line_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Set background to white
        line_image.fill(255)

        # For each object, draw its outline
        for obj in scene_state.get('objects', []):
            # Draw object outline in black
            obj_outline = self.calculate_object_outline(obj, scene_state)
            for px, py in obj_outline:
                if 0 <= px < width and 0 <= py < height:
                    line_image[py, px] = [0, 0, 0]  # Black lines

        return line_image

    def calculate_object_outline(self, obj, scene_state):
        """Calculate 2D outline of 3D object"""
        # For simulation, return a simple rectangular outline
        # In practice, would project 3D mesh edges to 2D

        obj_pos = obj.get('position', [0, 0, 0])
        obj_size = obj.get('size', [0.1, 0.1, 0.1])

        # Simple projection to 2D
        camera_pos = np.array(scene_state.get('camera_position', [0, 0, 1.5]))
        distance = np.linalg.norm(np.array(obj_pos) - camera_pos)

        if distance > 0:
            scale = min(100, int((obj_size[0] + obj_size[1]) * 150 / distance))
        else:
            scale = 50

        center_x = int((obj_pos[0] / distance) * 300 + 320)
        center_y = int((obj_pos[1] / distance) * 300 + 240)

        # Create rectangular outline
        outline = []
        for dx in [-scale//2, scale//2]:
            for dy in range(-scale//2, scale//2 + 1):
                outline.append((center_x + dx, center_y + dy))
        for dy in [-scale//2, scale//2]:
            for dx in range(-scale//2, scale//2 + 1):
                outline.append((center_x + dx, center_y + dy))

        return outline
```

## Validation and Quality Assurance

### Rendering Quality Validation

```python
class RenderingQualityValidator:
    """Validate rendering quality and realism"""

    def __init__(self):
        self.quality_metrics = {
            'photorealism_score': 0.0,
            'physical_accuracy': 0.0,
            'performance_metrics': {},
            'visual_artifacts': [],
            'lighting_consistency': True
        }

    def validate_rendering_quality(self, rendered_image, real_image=None):
        """Validate rendering quality against real-world images"""
        quality_report = {
            'structural_similarity': self.calculate_ssim(rendered_image, real_image) if real_image is not None else 0.5,
            'perceptual_quality': self.estimate_perceptual_quality(rendered_image),
            'physical_consistency': self.check_physical_consistency(rendered_image),
            'artifact_detection': self.detect_visual_artifacts(rendered_image),
            'lighting_validation': self.validate_lighting(rendered_image)
        }

        # Calculate overall quality score
        quality_score = (
            0.3 * quality_report['structural_similarity'] +
            0.25 * quality_report['perceptual_quality'] +
            0.2 * quality_report['physical_consistency'] +
            0.15 * quality_report['lighting_validation'] +
            0.1 * (1.0 - len(quality_report['artifact_detection']) / 10)  # Penalize artifacts
        )

        quality_report['overall_quality_score'] = quality_score

        return quality_report

    def calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index"""
        # Simplified SSIM calculation
        # In practice, would use scikit-image or similar library
        if img1.shape != img2.shape:
            return 0.0

        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2

        # Calculate mean and variance
        mu1 = np.mean(img1_gray)
        mu2 = np.mean(img2_gray)
        sigma1_sq = np.var(img1_gray)
        sigma2_sq = np.var(img2_gray)
        sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))

        # SSIM constants
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim = numerator / denominator if denominator != 0 else 0.0
        return max(0.0, min(1.0, ssim))

    def estimate_perceptual_quality(self, image):
        """Estimate perceptual quality of rendered image"""
        # Analyze various perceptual quality metrics
        metrics = {
            'sharpness': self.calculate_sharpness(image),
            'contrast': self.calculate_contrast(image),
            'color_accuracy': self.estimate_color_accuracy(image),
            'detail_preservation': self.estimate_detail_preservation(image)
        }

        # Combine metrics into overall score
        quality_score = (
            0.3 * metrics['sharpness'] +
            0.2 * metrics['contrast'] +
            0.3 * metrics['color_accuracy'] +
            0.2 * metrics['detail_preservation']
        )

        return quality_score

    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate Laplacian variance (measure of sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 scale (typical values range from 0 to several hundred)
        normalized_sharpness = min(1.0, laplacian_var / 1000.0)
        return normalized_sharpness

    def calculate_contrast(self, image):
        """Calculate image contrast"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate contrast as standard deviation of pixel intensities
        contrast = np.std(gray) / 128.0  # Normalize to 0-1 scale (max possible std is 128 for 0-255 range)
        return min(1.0, contrast)

    def estimate_color_accuracy(self, image):
        """Estimate color accuracy based on histogram analysis"""
        if len(image.shape) == 3:
            # Calculate colorfulness metric
            rg = np.absolute(image[:,:,2].astype(np.float64) - image[:,:,1].astype(np.float64))
            yb = np.absolute(image[:,:,0].astype(np.float64) -
                           (image[:,:,1].astype(np.float64) + image[:,:,2].astype(np.float64))/2.0)

            mean_rg = np.mean(rg)
            mean_yb = np.mean(yb)
            std_rg = np.std(rg)
            std_yb = np.std(yb)

            colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
            normalized_colorfulness = min(1.0, colorfulness / 100.0)
            return normalized_colorfulness
        else:
            return 0.5  # Grayscale has no color

    def estimate_detail_preservation(self, image):
        """Estimate how well details are preserved in rendering"""
        # Calculate edge density as measure of detail preservation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Use Canny edge detector
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Normalize to 0-1 scale
        return min(1.0, edge_density * 10)  # Multiply by 10 to get reasonable values

    def check_physical_consistency(self, image):
        """Check if rendered image is physically consistent"""
        # Check for physically plausible lighting, shadows, reflections
        physical_checks = [
            self.check_shadow_consistency(image),
            self.check_lighting_direction_consistency(image),
            self.check_material_response_consistency(image)
        ]

        # Return percentage of checks that pass
        return sum(physical_checks) / len(physical_checks) if physical_checks else 1.0

    def check_shadow_consistency(self, image):
        """Check if shadows are consistent with light sources"""
        # In practice, this would analyze shadow shapes, directions, and intensities
        # For simulation, return a reasonable default
        return True  # Placeholder

    def check_lighting_direction_consistency(self, image):
        """Check if lighting directions are consistent across image"""
        # Analyze gradients and lighting patterns for consistency
        return True  # Placeholder

    def check_material_response_consistency(self, image):
        """Check if materials respond consistently to lighting"""
        # Verify that similar materials have similar responses to similar lighting
        return True  # Placeholder

    def detect_visual_artifacts(self, image):
        """Detect visual artifacts in rendered image"""
        artifacts = []

        # Check for aliasing (jagged edges)
        if self.detect_aliasing(image):
            artifacts.append('aliasing')

        # Check for banding (color banding in gradients)
        if self.detect_banding(image):
            artifacts.append('banding')

        # Check for fireflies (bright outlier pixels in ray tracing)
        if self.detect_fireflies(image):
            artifacts.append('fireflies')

        # Check for incorrect reflections
        if self.detect_reflection_artifacts(image):
            artifacts.append('reflection_artifacts')

        return artifacts

    def detect_aliasing(self, image):
        """Detect aliasing artifacts"""
        # Calculate high-frequency content that might indicate aliasing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Use Sobel operator to detect edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Aliasing often manifests as high-frequency edge content
        high_freq_edges = np.sum(edge_magnitude > 200) / edge_magnitude.size
        return high_freq_edges > 0.1  # More than 10% high-frequency edges indicates potential aliasing

    def detect_banding(self, image):
        """Detect color banding artifacts"""
        # Banding appears as distinct color regions in gradients
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate histogram and look for gaps (indicating quantization)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        zero_bins = np.sum(hist == 0)
        banding_score = zero_bins / 256.0

        return banding_score > 0.3  # More than 30% zero bins indicates potential banding

    def detect_fireflies(self, image):
        """Detect firefly artifacts in ray-traced images"""
        if len(image.shape) == 3:
            # Convert to luminance for analysis
            luminance = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
        else:
            luminance = image

        # Fireflies are bright outlier pixels
        mean_lum = np.mean(luminance)
        std_lum = np.std(luminance)

        if std_lum > 0:
            outliers = luminance > (mean_lum + 3 * std_lum)  # 3-sigma rule
            outlier_ratio = np.sum(outliers) / luminance.size
            return outlier_ratio > 0.01  # More than 1% outliers indicates fireflies
        else:
            return False

    def detect_reflection_artifacts(self, image):
        """Detect reflection artifacts"""
        # Check for physically implausible reflections
        # This would involve more complex scene analysis in practice
        return False  # Placeholder

    def validate_lighting(self, image):
        """Validate lighting quality and realism"""
        # Check for proper global illumination, shadow softness, etc.
        lighting_metrics = {
            'global_illumination_present': self.check_global_illumination(image),
            'shadow_softness_appropriate': self.check_shadow_softness(image),
            'specular_highlights_realistic': self.check_specular_highlights(image),
            'color_bleeding_present': self.check_color_bleeding(image)
        }

        # Calculate lighting quality score
        score = sum(light for light in lighting_metrics.values()) / len(lighting_metrics)
        return score

    def check_global_illumination(self, image):
        """Check if global illumination is present"""
        # Look for indirect lighting effects and color bleeding
        return True  # Placeholder

    def check_shadow_softness(self, image):
        """Check if shadows have appropriate softness"""
        # Analyze shadow penumbra for realistic softness
        return True  # Placeholder

    def check_specular_highlights(self, image):
        """Check if specular highlights are realistic"""
        # Verify that highlights follow material properties and lighting
        return True  # Placeholder

    def check_color_bleeding(self, image):
        """Check if color bleeding from surfaces is present"""
        # Look for color transfer between nearby surfaces
        return True  # Placeholder

    def generate_quality_report(self, validation_results):
        """Generate comprehensive quality report"""
        report = {
            'quality_score': validation_results['overall_quality_score'],
            'detailed_metrics': validation_results,
            'recommendations': self.generate_recommendations(validation_results),
            'comparison_with_real': validation_results.get('structural_similarity', 'N/A'),
            'performance_impact': self.estimate_performance_impact(validation_results)
        }

        return report

    def generate_recommendations(self, validation_results):
        """Generate recommendations for improving rendering quality"""
        recommendations = []

        if validation_results['overall_quality_score'] < 0.7:
            recommendations.append("Rendering quality below threshold (0.7). Consider increasing sample count or improving lighting.")

        if 'aliasing' in validation_results['artifact_detection']:
            recommendations.append("Aliasing detected. Enable anti-aliasing or increase sampling rate.")

        if 'banding' in validation_results['artifact_detection']:
            recommendations.append("Color banding detected. Increase bit depth or enable dithering.")

        if validation_results['lighting_validation'] < 0.8:
            recommendations.append("Lighting quality needs improvement. Consider adding global illumination effects.")

        if validation_results['perceptual_quality'] < 0.7:
            recommendations.append("Perceptual quality below expectations. Review texture resolution and material properties.")

        return recommendations

    def estimate_performance_impact(self, validation_results):
        """Estimate performance impact of quality settings"""
        # Higher quality scores typically correlate with lower performance
        quality_to_performance = {
            'high_quality': validation_results['overall_quality_score'] > 0.8,
            'medium_quality': 0.6 <= validation_results['overall_quality_score'] <= 0.8,
            'performance_mode': validation_results['overall_quality_score'] < 0.6
        }

        return quality_to_performance
```

## Summary

Photorealistic simulation and rendering are essential components of Physical AI development for humanoid robots. The key aspects covered in this chapter include:

1. **RTX Rendering Fundamentals**: Understanding ray tracing, global illumination, and advanced material properties for creating realistic visual representations.

2. **Advanced Sensor Simulation**: Implementing realistic camera models with proper optical properties, noise models, and advanced effects like chromatic aberration and motion blur.

3. **Material Simulation**: Creating physically-based materials with realistic properties for different surfaces like human skin, robot metals, plastics, and environmental materials.

4. **Global Illumination**: Implementing advanced lighting techniques including photon mapping, caustics, and atmospheric scattering for realistic environmental simulation.

5. **Performance Optimization**: Balancing visual quality with computational performance through techniques like level-of-detail, occlusion culling, and adaptive rendering.

6. **Quality Validation**: Ensuring rendered images meet quality standards and are suitable for AI training through various validation metrics.

The success of Physical AI systems heavily depends on the quality of simulation rendering, as it directly impacts the effectiveness of perception systems trained on synthetic data. Properly implemented photorealistic rendering enables the creation of synthetic datasets that can effectively bridge the reality gap, allowing AI models to transfer successfully from simulation to real-world applications.