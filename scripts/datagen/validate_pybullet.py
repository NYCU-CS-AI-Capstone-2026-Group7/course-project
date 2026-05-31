"""Lightweight CPU kinematic and collision validation script using PyBullet and URDF.

Supports verifying randomly spawned poses OR loading a processed UMI
``object_poses.json`` (Step 3 output) to pre-filter and screen out kinematics
failures and self-collisions on the CPU before running heavy simulations.

Usage:
    # 1. Random procedural path verification
    python scripts/datagen/validate_pybullet.py --num_runs 20 --gui

    # 2. Verify processed UMI demonstration poses from Step 3
    python scripts/datagen/validate_pybullet.py --object_poses data/YYYYMMDD-taskname/demos/mapping/object_poses.json --gui
"""

import argparse
import time
import math
import random
import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError(
        "pybullet is required to run this script. Install it via 'pip install pybullet'"
    )


# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
_Z_SAFE = 0.25
_Z_GRASP = 0.05
_Z_RELEASE = 0.06

_FORK_SPAWN_X = (0.35, 0.48)
_FORK_SPAWN_Y = (-0.28, -0.18)

_KNIFE_SPAWN_X = (0.35, 0.48)
_KNIFE_SPAWN_Y = (-0.62, -0.52)

_PLATE_POS = [0.50, -0.40, 0.05]

_FRANKA_REST_POSE = [0.0, -math.pi/4, 0.0, -3.0*math.pi/4, 0.0, math.pi/2, math.pi/4]


def catmull_rom_spline_1d(P0, P1, P2, P3, t):
    """Interpolates between P1 and P2 using Catmull-Rom spline formula."""
    return 0.5 * (
        (2.0 * P1) +
        (-P0 + P2) * t +
        (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t * t +
        (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t * t * t
    )


def generate_spline_trajectory(c_start, c_end, num_steps):
    """Interpolates linear paths between points for Catmull-Rom endpoints."""
    P0 = c_start
    P1 = c_start
    P2 = c_end
    P3 = c_end
    
    path = []
    t_arr = np.linspace(0, 1, num_steps)
    for t in t_arr:
        pt = catmull_rom_spline_1d(P0, P1, P2, P3, t)
        path.append(pt)
    return np.array(path)


def yaw_from_quat_wxyz(quat_wxyz):
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PyBulletFrankaValidator:
    """Validator using PyBullet physics engine for lightweight CPU simulations."""

    def __init__(self, use_gui=False):
        self.use_gui = use_gui
        if self.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load Table and Robot
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Spawn robot base at (0.35, -0.74, 0.01) to match Isaac Sim config
        self.robot_pos = [0.35, -0.74, 0.01]
        self.robot_quat = p.getQuaternionFromEuler([0, 0, math.pi/2])
        
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            self.robot_pos,
            self.robot_quat,
            useFixedBase=True
        )
        self.ee_index = 11 # panda_hand link index
        
        # Move joints to home rest pose
        for idx, angle in enumerate(_FRANKA_REST_POSE):
            p.resetJointState(self.robot_id, idx, angle)
            
        self.objects = {}
        
    def setup_scene(self, fork_pos, fork_quat, knife_pos, knife_quat):
        """Spawns bounding shapes representing the plate, fork, and knife."""
        # Clean old objects
        for obj_id in self.objects.values():
            p.removeBody(obj_id)
        self.objects.clear()
        
        # 1. Spawn Plate (Flat cylinder cylinder bounding shape)
        plate_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=0.04)
        plate_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=0.04, rgbaColor=[0.9, 0.9, 0.9, 1])
        self.objects["plate"] = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=plate_col,
            baseVisualShapeIndex=plate_visual,
            basePosition=_PLATE_POS,
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 2. Spawn Fork (Box bounding shape)
        fork_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.08, 0.01])
        fork_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.08, 0.01], rgbaColor=[0.1, 0.6, 0.8, 1])
        self.objects["fork"] = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=fork_col,
            baseVisualShapeIndex=fork_visual,
            basePosition=fork_pos,
            baseOrientation=fork_quat
        )

        # 3. Spawn Knife (Box bounding shape)
        knife_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.012, 0.09, 0.01])
        knife_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012, 0.09, 0.01], rgbaColor=[0.9, 0.2, 0.2, 1])
        self.objects["knife"] = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=knife_col,
            baseVisualShapeIndex=knife_visual,
            basePosition=knife_pos,
            baseOrientation=knife_quat
        )
        
    def check_collision(self, ignored_body_id=None):
        """Checks for unintended collisions in the scene."""
        p.performCollisionDetection()
        
        contact_points = p.getContactPoints()
        for contact in contact_points:
            body_a = contact[1]
            body_b = contact[2]
            
            # If collision is with plane/ground, ignore
            if body_a == self.plane_id or body_b == self.plane_id:
                continue
                
            # If collision is between robot and a grabbed object, ignore
            if ignored_body_id is not None:
                if (body_a == self.robot_id and body_b == ignored_body_id) or \
                   (body_b == self.robot_id and body_a == ignored_body_id):
                    continue
                    
            # If robot collided with table objects (unintended contact)
            if body_a == self.robot_id or body_b == self.robot_id:
                if body_a == body_b:
                    continue
                return True
        return False

    def validate_trajectory(self, obj_pos, obj_yaw, target_place_pos, obj_name):
        """Simulates path segments and checks for self-limits and collisions."""
        # Reset joints to rest pose
        for idx, angle in enumerate(_FRANKA_REST_POSE):
            p.resetJointState(self.robot_id, idx, angle)
            
        current_ee_state = p.getLinkState(self.robot_id, self.ee_index)
        current_ee_pos = np.array(current_ee_state[4])
        
        # Build Waypoints
        w_start = current_ee_pos
        w_hover = np.array([obj_pos[0], obj_pos[1], _Z_SAFE])
        w_grasp = np.array([obj_pos[0], obj_pos[1], _Z_GRASP])
        w_lift = np.array([obj_pos[0], obj_pos[1], _Z_SAFE])
        w_transit = np.array([target_place_pos[0], target_place_pos[1], _Z_SAFE])
        w_place = np.array([target_place_pos[0], target_place_pos[1], _Z_RELEASE])
        w_retreat = np.array([target_place_pos[0], target_place_pos[1], _Z_SAFE])
        
        # Segment configuration
        segments = [
            (w_start, w_hover, 40, False),
            (w_hover, w_grasp, 30, False),
            (w_grasp, w_grasp, 10, True),
            (w_grasp, w_lift, 30, True),
            (w_lift, w_transit, 40, True),
            (w_transit, w_place, 30, True),
            (w_place, w_place, 10, False),
            (w_place, w_retreat, 30, False)
        ]
        
        obj_body_id = self.objects[obj_name]
        
        for idx, (p_start, p_end, steps, has_grasped) in enumerate(segments):
            traj = generate_spline_trajectory(p_start, p_end, steps)
            
            for target_pos in traj:
                # Solve Inverse Kinematics command
                joint_angles = p.calculateInverseKinematics(
                    self.robot_id,
                    self.ee_index,
                    targetPosition=list(target_pos),
                    targetOrientation=p.getQuaternionFromEuler([0, math.pi, obj_yaw + math.pi/2])
                )
                
                # Apply joint angles
                for j_idx, angle in enumerate(joint_angles[:7]):
                    p.resetJointState(self.robot_id, j_idx, angle)
                    
                if has_grasped:
                    ee_state = p.getLinkState(self.robot_id, self.ee_index)
                    ee_pos = ee_state[4]
                    ee_quat = ee_state[5]
                    obj_pos_offset = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.05]
                    p.resetBasePositionAndOrientation(obj_body_id, obj_pos_offset, ee_quat)
                    
                # Collision check
                ignored = obj_body_id if has_grasped else None
                collision_detected = self.check_collision(ignored_body_id=ignored)
                
                if collision_detected:
                    return False
                    
                if self.use_gui:
                    p.stepSimulation()
                    time.sleep(0.01)
                    
        return True

    def run_validation_for_episode(self, fork_pos, fork_quat_wxyz, knife_pos, knife_quat_wxyz):
        """Tests kinematics and collisions for a specific cutlery pose setup."""
        # Convert wxyz quaternion to xyzw (PyBullet format)
        w, x, y, z = fork_quat_wxyz
        f_quat_xyzw = [x, y, z, w]
        w, x, y, z = knife_quat_wxyz
        k_quat_xyzw = [x, y, z, w]

        self.setup_scene(fork_pos, f_quat_xyzw, knife_pos, k_quat_xyzw)

        f_yaw = yaw_from_quat_wxyz(fork_quat_wxyz)
        k_yaw = yaw_from_quat_wxyz(knife_quat_wxyz)

        if getattr(self, "fix_fork_yaw", False):
            f_yaw = 0.0
        if getattr(self, "fix_knife_yaw", False):
            k_yaw = 0.0

        # 1. Test Knife (place right)
        k_place_pos = [_PLATE_POS[0] + 0.10, _PLATE_POS[1], _PLATE_POS[2]]
        knife_ok = self.validate_trajectory(knife_pos, k_yaw, k_place_pos, "knife")
        if not knife_ok:
            return False

        # 2. Test Fork (place left)
        f_place_pos = [_PLATE_POS[0] - 0.10, _PLATE_POS[1], _PLATE_POS[2]]
        fork_ok = self.validate_trajectory(fork_pos, f_yaw, f_place_pos, "fork")
        return fork_ok

    def run_procedural_test(self):
        """Randomizes spawns and checks kinematics and collisions."""
        f_x = random.uniform(*_FORK_SPAWN_X)
        f_y = random.uniform(*_FORK_SPAWN_Y)
        f_yaw = random.uniform(-math.pi, math.pi)
        f_quat = p.getQuaternionFromEuler([0, 0, f_yaw])
        f_quat_wxyz = [f_quat[3], f_quat[0], f_quat[1], f_quat[2]] # convert xyzw to wxyz
        
        k_x = random.uniform(*_KNIFE_SPAWN_X)
        k_y = random.uniform(*_KNIFE_SPAWN_Y)
        k_yaw = random.uniform(-math.pi, math.pi)
        k_quat = p.getQuaternionFromEuler([0, 0, k_yaw])
        k_quat_wxyz = [k_quat[3], k_quat[0], k_quat[1], k_quat[2]]
        
        return self.run_validation_for_episode(
            [f_x, f_y, _FORK_Z], f_quat_wxyz,
            [k_x, k_y, _KNIFE_Z], k_quat_wxyz
        )

    def close(self):
        p.disconnect()


def main():
    parser = argparse.ArgumentParser(description="PyBullet Franka Trajectory Validator")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of random validation loops to execute.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization.")
    parser.add_argument(
        "--object_poses",
        type=str,
        default=None,
        help="Path to object_poses.json to validate demonstration episodes from Step 3."
    )
    parser.add_argument("--fix_knife_yaw", action="store_true", help="Force the knife's grasp yaw to be zero.")
    parser.add_argument("--fix_fork_yaw", action="store_true", help="Force the fork's grasp yaw to be zero.")
    args = parser.parse_args()
    
    validator = PyBulletFrankaValidator(use_gui=args.gui)
    validator.fix_knife_yaw = args.fix_knife_yaw
    validator.fix_fork_yaw = args.fix_fork_yaw

    if args.object_poses is not None:
        # Load and validate processed UMI demonstration poses
        from simulator.tasks.cutlery_arrangement.cutlery_arrangement_env_cfg import CutleryArrangementEnvCfg
        from simulator.utils.object_poses_loader import load_episode_poses

        env_cfg = CutleryArrangementEnvCfg()
        print(f"Loading episodes from {args.object_poses}...")
        episodes = load_episode_poses(args.object_poses, env_cfg.object_pose_cfg)
        
        print(f"Verifying {len(episodes)} episodes on CPU...")
        success_count = 0
        failed_episodes = []

        for i, poses in enumerate(episodes):
            fork_pos, fork_quat = poses["fork"]
            knife_pos, knife_quat = poses["knife"]
            
            success = validator.run_validation_for_episode(fork_pos, fork_quat, knife_pos, knife_quat)
            if success:
                success_count += 1
                print(f"  Episode {i}: [PASSED]")
            else:
                failed_episodes.append(i)
                print(f"  Episode {i}: [FAILED] Kinematics failure or Collision detected!")

        print(f"\nDemo File Pre-Filter Validation Summary:")
        print(f"  Total Checked: {len(episodes)}")
        print(f"  Passed: {success_count}/{len(episodes)}")
        if failed_episodes:
            print(f"  Failed Episode Indexes (Should be discarded/re-recorded): {failed_episodes}")
        else:
            print(f"  All episodes passed validation successfully!")
    else:
        # Run procedural random tests
        print(f"Starting {args.num_runs} random validation loops...")
        success_count = 0
        for i in range(args.num_runs):
            try:
                success = validator.run_procedural_test()
                if success:
                    success_count += 1
                    print(f"  Episode {i+1}: [PASSED]")
                else:
                    print(f"  Episode {i+1}: [FAILED]")
            except Exception as e:
                print(f"  Episode {i+1}: [EXCEPTION] {e}")
                
        print(f"\nRandom Spawning Validation Summary: {success_count}/{args.num_runs} passed")
        
    validator.close()


if __name__ == "__main__":
    main()
