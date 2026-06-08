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
    from scripts.datagen.validate_pybullet import (
        PyBulletFrankaValidator,
        FORK_Z,
        KNIFE_Z,
        normalize_angle,
    )
except ImportError:
    raise ImportError(
        "Could not import PyBulletFrankaValidator. Ensure scripts/datagen/validate_pybullet.py exists."
    )


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
    parser.add_argument(
        "--arm-physics",
        action="store_true",
        help="Enable physical joint-motor control in PyBullet validator."
    )
    parser.add_argument(
        "--self_collision_margin",
        type=float,
        default=-0.01,
        help="Margin for robot self collisions (default: -0.01).",
    )
    parser.add_argument(
        "--robot_obj_collision_margin",
        type=float,
        default=0.005,
        help="Margin for robot-to-object collisions (default: 0.005).",
    )
    parser.add_argument(
        "--obj_obj_collision_margin",
        type=float,
        default=0.005,
        help="Margin for object-to-object collisions (default: 0.005).",
    )
    parser.add_argument(
        "--spawn_margin",
        type=float,
        nargs=4,
        default=[0.12, 0.12, 0.15, 0.12],
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        help="Table inner margin for spawn area: Top Right Bottom Left in meters (default: 0.12 0.12 0.15 0.12).",
    )
    parser.add_argument(
        "--yaw_dist",
        type=str,
        choices=["uniform", "normal"],
        default="uniform",
        help="Yaw distribution type: uniform or normal (default: uniform).",
    )
    parser.add_argument(
        "--yaw_std",
        type=float,
        default=15.0,
        help="Standard deviation for normal yaw distribution in degrees (default: 15.0).",
    )
    parser.add_argument(
        "--fork_mean_yaw",
        type=float,
        default=180.0,
        help="Mean yaw for fork in degrees (default: 180.0, pointing away from arm).",
    )
    parser.add_argument(
        "--knife_mean_yaw",
        type=float,
        default=0.0,
        help="Mean yaw for knife in degrees (default: 0.0, pointing away from arm).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Spawn area constants (imported from validate_pybullet.py)
    _MIN_SPAWN_DIST = args.min_dist

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
        allow_plate_collision=args.allow_plate_collision,
        arm_physics=args.arm_physics,
        self_collision_margin=args.self_collision_margin,
        robot_obj_collision_margin=args.robot_obj_collision_margin,
        obj_obj_collision_margin=args.obj_obj_collision_margin,
        spawn_margin=tuple(args.spawn_margin),
        yaw_dist=args.yaw_dist,
        yaw_std=args.yaw_std,
        fork_mean_yaw=args.fork_mean_yaw,
        knife_mean_yaw=args.knife_mean_yaw,
    )

    episodes = []
    attempts = 0

    print(f"[INFO] Generating {args.num_demos} validated spawn poses...")
    while len(episodes) < args.num_demos:
        attempts += 1

        # 3. Randomize and validate kinematic reachability and collisions on CPU
        success, fork_pos, f_yaw, f_quat, knife_pos, k_yaw, k_quat = validator.run_procedural_test(return_details=True)

        if success:
            f_x, f_y = fork_pos[0], fork_pos[1]
            k_x, k_y = knife_pos[0], knife_pos[1]

            # Reconstruct UMI-style tvec and rvec
            # Local tvec = World Pos - Anchor Pos
            # Use slight vertical offsets (0.018 for fork, 0.011 for knife) to prevent clipping
            tvec_fork = [f_x - _ANCHOR_X, f_y - _ANCHOR_Y, FORK_Z]
            tvec_knife = [k_x - _ANCHOR_X, k_y - _ANCHOR_Y, KNIFE_Z]

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
