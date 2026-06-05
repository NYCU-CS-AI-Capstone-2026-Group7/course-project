#!/usr/bin/env python3
"""Procedural Pose Generator with PyBullet Kinematics & Collision Filtering.

Generates random spawn poses for the cutlery task, validates them on the CPU
using the PyBullet validator, and exports successful configurations as a UMI-schema
``object_poses.json`` file to be replayed by ``generate.py``.

Usage:
    python scripts/datagen/generate_from_zero.py \
        --num_demos 100 \
        --output data/procedural_spawn/demos/mapping/object_poses.json
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

# Setup paths to import project scripts and packages
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

simulator_src = repo_root / "packages" / "simulator" / "src"
if simulator_src.exists() and str(simulator_src) not in sys.path:
    sys.path.insert(0, str(simulator_src))

try:
    from scripts.datagen.validate_pybullet import PyBulletFrankaValidator
except ImportError:
    raise ImportError(
        "Could not import PyBulletFrankaValidator. Ensure scripts/datagen/validate_pybullet.py exists."
    )


def _euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Roll, pitch, yaw to quaternion (w, x, y, z)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser(
        description="Procedurally generate validated cutlery poses using PyBullet pre-filtering."
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=100,
        help="Number of successfully validated demos to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/procedural_spawn/demos/mapping/object_poses.json",
        help="Path to output object_poses.json file.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run PyBullet with GUI visualization during generation.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=100.0,
        help="Frame rate (FPS) for GUI visualization during generation.",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.273,
        help="Minimum Euclidean distance between fork and knife to prevent overlap (default: 0.273m based on STL AABBs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging of failure reasons in validator.",
    )
    parser.add_argument(
        "--reconnect_interval",
        type=float,
        default=10.0,
        help="PyBullet GUI reconnection interval in seconds to clear rendering cache (default: 10.0s).",
    )
    parser.add_argument(
        "--allow_plate_collision",
        action="store_true",
        help="Allow cutlery (fork/knife) to touch or collide with the plate without failing validation."
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Spawn area constants (matching validate_pybullet.py)
    _TABLE_SURFACE_Z = 0.5
    _CUTLERY_SPAWN_X = (0.25, 0.55)
    _CUTLERY_SPAWN_Y = (-0.60, -0.20)
    _MIN_SPAWN_DIST = args.min_dist
    _FORK_Z = _TABLE_SURFACE_Z + 0.018
    _KNIFE_Z = _TABLE_SURFACE_Z + 0.011

    # Anchor constants used by simulator loader to reconstruct world poses
    # World Position = Anchor Position + R(Anchor Yaw) * Local Position
    # Here anchor_yaw = 0.0, so: World = Anchor + Local -> Local = World - Anchor
    _ANCHOR_X = 0.40
    _ANCHOR_Y = 0.10

    # Per-object yaw offsets in task config:
    # World Yaw = Anchor Yaw + Local Yaw (from rvec) + Per-Object Offset
    # For cutlery: anchor_yaw = 0.0
    _KNIFE_YAW_OFFSET = math.pi
    _FORK_YAW_OFFSET = 2.0 * math.pi

    print("[INFO] Initializing PyBullet Franka Validator...")
    validator = PyBulletFrankaValidator(
        use_gui=args.gui, 
        fps=args.fps, 
        min_dist=args.min_dist, 
        verbose=args.verbose,
        reconnect_interval=args.reconnect_interval,
        allow_plate_collision=args.allow_plate_collision
    )

    episodes = []
    attempts = 0

    print(f"[INFO] Generating {args.num_demos} validated spawn poses...")
    while len(episodes) < args.num_demos:
        attempts += 1

        # Randomize spawn positions in a shared area and ensure no overlap (min distance 0.15m)
        while True:
            f_x = random.uniform(*_CUTLERY_SPAWN_X)
            f_y = random.uniform(*_CUTLERY_SPAWN_Y)
            k_x = random.uniform(*_CUTLERY_SPAWN_X)
            k_y = random.uniform(*_CUTLERY_SPAWN_Y)
            
            # Check Euclidean distance between fork and knife centers
            dist = math.sqrt((f_x - k_x)**2 + (f_y - k_y)**2)
            if dist >= _MIN_SPAWN_DIST:
                break

        f_yaw = random.uniform(-math.pi, math.pi)
        f_quat = _euler_xyz_to_quat_wxyz(0, 0, f_yaw)

        k_yaw = random.uniform(-math.pi, math.pi)
        k_quat = _euler_xyz_to_quat_wxyz(0, 0, k_yaw)

        # 3. Validate kinematic reachability and collisions on CPU
        success = validator.run_validation_for_episode(
            [f_x, f_y, _FORK_Z], f_quat,
            [k_x, k_y, _KNIFE_Z], k_quat
        )

        if success:
            # Reconstruct UMI-style tvec and rvec
            # Local tvec = World Pos - Anchor Pos
            # Use slight vertical offsets (0.018 for fork, 0.011 for knife) to prevent clipping
            tvec_fork = [f_x - _ANCHOR_X, f_y - _ANCHOR_Y, _FORK_Z]
            tvec_knife = [k_x - _ANCHOR_X, k_y - _ANCHOR_Y, _KNIFE_Z]

            # Local rvec = World Yaw - Anchor Yaw - Per-Object Offset
            # Since rvec is a Rodrigues vector around Z-axis, it's [0.0, 0.0, yaw_diff]
            rz_fork = normalize_angle(f_yaw - _FORK_YAW_OFFSET)
            rz_knife = normalize_angle(k_yaw - _KNIFE_YAW_OFFSET)

            episode_entry = {
                "video_name": f"procedural_ep_{len(episodes)}",
                "episode_range": [0, 100],
                "objects": [
                    {
                        "object_name": "fork",
                        "rvec": [0.0, 0.0, rz_fork],
                        "tvec": tvec_fork
                    },
                    {
                        "object_name": "knife",
                        "rvec": [0.0, 0.0, rz_knife],
                        "tvec": tvec_knife
                    }
                ],
                "status": "full"
            }
            episodes.append(episode_entry)
            print(f"  [PASSED] Generated pose {len(episodes)}/{args.num_demos} (attempts: {attempts})")
        else:
            if args.verbose:
                print(f"  [FAILED] Attempt {attempts}: Pose validation failed.")

    validator.close()

    print(f"[INFO] Writing {len(episodes)} validated episodes to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(episodes, f, indent=4)

    print(f"[INFO] Done! Successfully generated procedural poses. Success rate: {args.num_demos/attempts*100:.1f}%")


if __name__ == "__main__":
    main()
