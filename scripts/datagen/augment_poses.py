"""Offline demonstration pose data augmentation script.

Reads UMI-style ``object_poses.json``, applies geometric perturbations
(translation noise, yaw jitter, cross-episode mixing, global transforms),
and outputs an expanded ``object_poses_augmented.json``.

Usage:
    python scripts/datagen/augment_poses.py \
        --input_poses data/YYYYMMDD-taskname/demos/mapping/object_poses.json \
        --output_poses data/YYYYMMDD-taskname/demos/mapping/object_poses_augmented.json \
        --multiplier 5 --mix_episodes --translation_std 0.02 --yaw_std 10
"""

import argparse
import json
import math
import random
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Rodrigues Rotation Formula without OpenCV dependency
# ---------------------------------------------------------------------------
def rodrigues_to_matrix(rvec):
    """Converts a 3D rotation vector into a 3x3 rotation matrix."""
    rvec = np.array(rvec, dtype=float)
    theta = np.linalg.norm(rvec)
    if theta < 1e-10:
        return np.eye(3)
    
    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0]
    ])
    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * np.dot(K, K)
    return R


def matrix_to_rodrigues(R):
    """Converts a 3x3 rotation matrix into a 3D rotation vector."""
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)
    
    if theta < 1e-10:
        return np.zeros(3)
        
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-10:
        # Theta is pi (180 degrees)
        # Eigenvalue problem to find axis
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        axis = eigenvectors[:, np.argmax(eigenvalues)]
        return axis * theta
        
    v = (theta / (2.0 * sin_theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    return v


# ---------------------------------------------------------------------------
# Augmentation Core Logic
# ---------------------------------------------------------------------------
def perturb_translation(tvec, std_dev):
    """Adds random Gaussian noise to x and y coordinates (z remains fixed)."""
    tvec_aug = list(tvec)
    tvec_aug[0] += random.gauss(0.0, std_dev)
    tvec_aug[1] += random.gauss(0.0, std_dev)
    return tvec_aug


def perturb_yaw(rvec, yaw_std_deg):
    """Rotates the object around the world Z-axis by a random angle."""
    angle_rad = math.radians(random.gauss(0.0, yaw_std_deg))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # 3D rotation matrix around Z axis
    R_z = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    R_obj = rodrigues_to_matrix(rvec)
    R_new = np.dot(R_z, R_obj)
    rvec_aug = matrix_to_rodrigues(R_new)
    return [float(x) for x in rvec_aug]


def apply_global_transform(objects, t_offset, theta_deg):
    """Applies a global rigid body transform (Z rotation + translation) to all objects."""
    angle_rad = math.radians(theta_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    R_z = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    transformed_objects = []
    for obj in objects:
        new_obj = obj.copy()
        tvec = np.array(obj["tvec"])
        # Rotate translation vector around anchor origin, then translate
        tvec_new = np.dot(R_z, tvec) + np.array([t_offset[0], t_offset[1], 0.0])
        new_obj["tvec"] = [float(x) for x in tvec_new]
        
        # Rotate orientation vector
        R_obj = rodrigues_to_matrix(obj["rvec"])
        R_new = np.dot(R_z, R_obj)
        rvec_new = matrix_to_rodrigues(R_new)
        new_obj["rvec"] = [float(x) for x in rvec_new]
        transformed_objects.append(new_obj)
        
    return transformed_objects


def mix_episodes(episodes, num_mixed):
    """Combines objects from different episodes to create mixed scenarios."""
    mixed_episodes = []
    
    # Separate objects by name
    knife_pool = [obj for ep in episodes for obj in ep["objects"] if obj["object_name"] == "knife"]
    fork_pool = [obj for ep in episodes for obj in ep["objects"] if obj["object_name"] == "fork"]
    
    if not knife_pool or not fork_pool:
        print("  [Warning] Missing knife or fork pools. Skipping mixing.")
        return episodes
        
    for k in range(num_mixed):
        k_obj = random.choice(knife_pool).copy()
        f_obj = random.choice(fork_pool).copy()
        
        # Plate is usually skipped/fixed, or we copy it from a random episode
        mixed_ep = {
            "video_name": f"augmented_mix_{k}",
            "episode_range": [0, 0],
            "objects": [k_obj, f_obj],
            "status": "full"
        }
        mixed_episodes.append(mixed_ep)
        
    return mixed_episodes


def main():
    parser = argparse.ArgumentParser(description="UMI Poses Dataset Augmentation Utility")
    parser.add_argument("--input_poses", type=str, required=True, help="Path to raw object_poses.json")
    parser.add_argument("--output_poses", type=str, required=True, help="Path to output augmented object_poses.json")
    parser.add_argument("--multiplier", type=int, default=3, help="Generate N augmented copies per original episode.")
    parser.add_argument("--mix_episodes", action="store_true", help="Enable cross-episode object mixing.")
    parser.add_argument("--mix_count", type=int, default=20, help="Number of mixed episodes to generate.")
    parser.add_argument("--translation_std", type=float, default=0.015, help="Standard deviation of translation noise in meters.")
    parser.add_argument("--yaw_std", type=float, default=8.0, help="Standard deviation of yaw angle rotation in degrees.")
    parser.add_argument("--global_shift", action="store_true", help="Enable random global translation & rotation of the setup.")
    args = parser.parse_args()

    input_path = Path(args.input_poses)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")
        
    with open(input_path, "r") as f:
        data = json.load(f)
        
    original_episodes = [ep for ep in data if ep.get("status") == "full"]
    print(f"Loaded {len(original_episodes)} full-status episodes from {input_path.name}")

    augmented_list = []
    
    # 1. Keep original episodes
    augmented_list.extend(original_episodes)

    # 2. Add perturbed duplicates
    for ep_idx, ep in enumerate(original_episodes):
        for m in range(args.multiplier):
            new_ep = ep.copy()
            new_ep["video_name"] = f"{ep['video_name']}_aug_{m}"
            
            aug_objects = []
            
            # Decide global shift parameter
            g_offset = [random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03)] if args.global_shift else [0.0, 0.0]
            g_theta = random.uniform(-10.0, 10.0) if args.global_shift else 0.0
            
            # Apply global shift
            shifted_objects = apply_global_transform(ep["objects"], g_offset, g_theta)
            
            for obj in shifted_objects:
                new_obj = obj.copy()
                # Apply local coordinate perturbations
                new_obj["tvec"] = perturb_translation(obj["tvec"], args.translation_std)
                new_obj["rvec"] = perturb_yaw(obj["rvec"], args.yaw_std)
                aug_objects.append(new_obj)
                
            new_ep["objects"] = aug_objects
            augmented_list.append(new_ep)
            
    # 3. Add mixed episodes
    if args.mix_episodes:
        print(f"Creating {args.mix_count} mixed episodes...")
        mixed_eps = mix_episodes(original_episodes, args.mix_count)
        
        # Decide if mixed episodes also get global shifts / noise
        for m, ep in enumerate(mixed_eps):
            # Decide global shift parameter
            g_offset = [random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03)] if args.global_shift else [0.0, 0.0]
            g_theta = random.uniform(-10.0, 10.0) if args.global_shift else 0.0
            shifted_objects = apply_global_transform(ep["objects"], g_offset, g_theta)
            
            aug_objects = []
            for obj in shifted_objects:
                new_obj = obj.copy()
                new_obj["tvec"] = perturb_translation(obj["tvec"], args.translation_std)
                new_obj["rvec"] = perturb_yaw(obj["rvec"], args.yaw_std)
                aug_objects.append(new_obj)
            ep["objects"] = aug_objects
            augmented_list.append(ep)
            
    # Export augmented database
    output_path = Path(args.output_poses)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(augmented_list, f, indent=4)
        
    print(f"Successfully generated {len(augmented_list)} total episodes (original + augmented).")
    print(f"Exported augmented file to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
