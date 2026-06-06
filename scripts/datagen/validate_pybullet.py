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
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

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
# Table Dimensions
_TABLE_SURFACE_Z = 0.5
_TABLE_HEIGHT = 0.0409113
_TABLE_LENGTH = 0.70  # X axis length
_TABLE_WIDTH = 0.65   # Y axis length
_TABLE_CENTER_X = 0.353162
_TABLE_CENTER_Y = -0.351832

_Z_SAFE = _TABLE_SURFACE_Z + 0.20
_Z_GRASP = _TABLE_SURFACE_Z + 0.11
_Z_RELEASE = _TABLE_SURFACE_Z + 0.11

_MIN_SPAWN_DIST = 0.15

_PLATE_POS = [0.50, -0.40, _TABLE_SURFACE_Z + 0.05]

_FORK_Z = _TABLE_SURFACE_Z + 0.018
_KNIFE_Z = _TABLE_SURFACE_Z + 0.011

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


# Publicly exported constants for external scripts (e.g. generate_from_zero.py)
TABLE_SURFACE_Z = _TABLE_SURFACE_Z
FORK_Z = _FORK_Z
KNIFE_Z = _KNIFE_Z


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
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


class PyBulletFrankaValidator:
    """Validator using PyBullet physics engine for lightweight CPU simulations."""

    def __init__(self, use_gui=False, verbose=False, fps=100.0, min_dist=0.273, reconnect_interval=10.0, allow_plate_collision=False, arm_physics=False, self_collision_margin=-0.01, robot_obj_collision_margin=0.005, obj_obj_collision_margin=0.005, spawn_margin=(0.12, 0.12, 0.15, 0.12)):
        self.use_gui = use_gui
        self.verbose = verbose
        self.sleep_time = 1.0 / fps if fps > 0 else 0.0
        self.min_dist = min_dist
        self.self_collision_margin = self_collision_margin
        self.robot_obj_collision_margin = robot_obj_collision_margin
        self.obj_obj_collision_margin = obj_obj_collision_margin
        self.spawn_margin = spawn_margin
        
        # Calculate dynamic cutlery spawn area based on spawn_margin
        self.table_min_x = _TABLE_CENTER_X - (_TABLE_LENGTH / 2)
        self.table_max_x = _TABLE_CENTER_X + (_TABLE_LENGTH / 2)
        self.table_min_y = _TABLE_CENTER_Y - (_TABLE_WIDTH / 2)
        self.table_max_y = _TABLE_CENTER_Y + (_TABLE_WIDTH / 2)
        self.cutlery_spawn_x = (self.table_min_x + self.spawn_margin[3], self.table_max_x - self.spawn_margin[1])
        self.cutlery_spawn_y = (self.table_min_y + self.spawn_margin[2], self.table_max_y - self.spawn_margin[0])

        self.reconnect_interval = reconnect_interval
        self.last_reconnect_time = time.time()
        self.allow_plate_collision = allow_plate_collision
        self.arm_physics = arm_physics
        self.objects = {}
        
        self.initialize_pybullet()

    def initialize_pybullet(self):
        if self.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load Table and Robot
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Spawn Table to match the actual square table dimensions in Isaac Sim
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[_TABLE_LENGTH/2, _TABLE_WIDTH/2, _TABLE_HEIGHT/2])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[_TABLE_LENGTH/2, _TABLE_WIDTH/2, _TABLE_HEIGHT/2], rgbaColor=[0.6, 0.4, 0.2, 1])
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.353162, -0.351832, _TABLE_SURFACE_Z - _TABLE_HEIGHT / 2],
            baseOrientation=[0, 0, 0, 1]
        )
        
        # Spawn robot base at (0.35, -0.74, _TABLE_SURFACE_Z - _TABLE_HEIGHT)
        self.robot_pos = [0.35, -0.74, _TABLE_SURFACE_Z - _TABLE_HEIGHT]
        self.robot_quat = p.getQuaternionFromEuler([0, 0, math.pi/2])
        
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            self.robot_pos,
            self.robot_quat,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        # Dynamically map ee_index (panda_hand) and gripper_center_index (panda_grasptarget)
        self.ee_index = 8
        self.gripper_center_index = 8
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == "panda_hand":
                self.ee_index = i
            elif link_name == "panda_grasptarget":
                self.gripper_center_index = i
        
        # Move joints to home rest pose
        for idx, angle in enumerate(_FRANKA_REST_POSE):
            p.resetJointState(self.robot_id, idx, angle)
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=angle, force=1000)
        p.resetJointState(self.robot_id, 9, 0.04)
        p.resetJointState(self.robot_id, 10, 0.04)
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=1000)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=1000)
            
        # Build parent-child link map for robot to ignore adjacent/sibling collisions
        self.parent_map = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            parent_link = joint_info[16]
            self.parent_map[i] = parent_link
            
        # Fetch joint limits for joints 0-6 (Franka arm joints)
        self.joint_limits = []
        for i in range(7):
            joint_info = p.getJointInfo(self.robot_id, i)
            self.joint_limits.append((joint_info[8], joint_info[9]))
            
        # Fetch joint limits for all active joints (to constrain calculateInverseKinematics)
        self.ik_lower_limits = []
        self.ik_upper_limits = []
        self.ik_joint_ranges = []
        self.ik_rest_poses = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            j_type = joint_info[2]
            if j_type != p.JOINT_FIXED:
                self.ik_lower_limits.append(joint_info[8])
                self.ik_upper_limits.append(joint_info[9])
                self.ik_joint_ranges.append(joint_info[9] - joint_info[8])
                if len(self.ik_rest_poses) < 7:
                    self.ik_rest_poses.append(_FRANKA_REST_POSE[len(self.ik_rest_poses)])
                else:
                    self.ik_rest_poses.append(0.04) # Gripper finger joints rest pose

    def reconnect(self):
        # Get current camera pose in GUI mode before disconnecting to preserve user view
        camera_params = None
        if self.use_gui:
            try:
                cam_info = p.getDebugVisualizerCamera()
                camera_params = {
                    "yaw": cam_info[8],
                    "pitch": cam_info[9],
                    "distance": cam_info[10],
                    "target_pos": cam_info[11]
                }
            except Exception:
                pass

        p.disconnect()
        self.objects.clear()
        self.initialize_pybullet()

        # Restore camera pose in GUI mode
        if self.use_gui and camera_params is not None:
            try:
                p.resetDebugVisualizerCamera(
                    cameraDistance=camera_params["distance"],
                    cameraYaw=camera_params["yaw"],
                    cameraPitch=camera_params["pitch"],
                    cameraTargetPosition=camera_params["target_pos"]
                )
            except Exception:
                pass
        
    def setup_scene(self, fork_pos, fork_quat, knife_pos, knife_quat):
        """Spawns bounding shapes representing the plate, fork, and knife."""
        # Check and perform periodic reconnect to clear rendering cache
        curr_time = time.time()
        if curr_time - self.last_reconnect_time >= self.reconnect_interval:
            if self.verbose:
                print(f"[INFO] Reconnecting PyBullet (interval: {self.reconnect_interval}s) to clear mesh rendering cache...")
            self.reconnect()
            self.last_reconnect_time = curr_time

        # Clean old objects
        for obj_id in self.objects.values():
            p.removeBody(obj_id)
        self.objects.clear()
        
        # Enforce roll = 0 for fork and knife at init
        f_euler = list(p.getEulerFromQuaternion(fork_quat))
        f_euler[0] = 0.0
        fork_quat = p.getQuaternionFromEuler(f_euler)

        k_euler = list(p.getEulerFromQuaternion(knife_quat))
        k_euler[0] = 0.0
        knife_quat = p.getQuaternionFromEuler(k_euler)
        
        # 1. Spawn Plate (Flat cylinder cylinder bounding shape)
        plate_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.036362, height=0.010419)
        plate_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.036362, length=0.010419, rgbaColor=[0.9, 0.9, 0.9, 1])
        self.objects["plate"] = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=plate_col,
            baseVisualShapeIndex=plate_visual,
            basePosition=_PLATE_POS,
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 2. Spawn Fork (STL mesh aligned by -90deg about X-axis, scaled by 100)
        q_align = p.getQuaternionFromEuler([-math.pi/2, 0, math.pi])
        fork_stl = str(_REPO_ROOT / "packages" / "simulator" / "assets" / "scenes" / "dining_room" / "objects" / "Fork" / "fork.stl")
        fork_col = p.createCollisionShape(
            p.GEOM_MESH, 
            fileName=fork_stl, 
            meshScale=[100, 100, 100],
            collisionFrameOrientation=q_align
        )
        fork_visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=fork_stl,
            meshScale=[100, 100, 100],
            visualFrameOrientation=q_align
        )
        # Check if fork overlaps with the plate (threshold 0.12m) to spawn it above the plate
        dist_fork_plate = math.sqrt((fork_pos[0] - _PLATE_POS[0])**2 + (fork_pos[1] - _PLATE_POS[1])**2)
        fork_spawn_z = _PLATE_POS[2] + 0.03 if dist_fork_plate < 0.12 else _FORK_Z

        self.objects["fork"] = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=fork_col,
            baseVisualShapeIndex=fork_visual,
            basePosition=[fork_pos[0], fork_pos[1], fork_spawn_z],
            baseOrientation=fork_quat
        )

        # 3. Spawn Knife (STL mesh aligned by -90deg about X-axis, scaled by 100)
        knife_stl = str(_REPO_ROOT / "packages" / "simulator" / "assets" / "scenes" / "dining_room" / "objects" / "Knife" / "knife.stl")
        knife_col = p.createCollisionShape(
            p.GEOM_MESH, 
            fileName=knife_stl, 
            meshScale=[100, 100, 100],
            collisionFrameOrientation=q_align
        )
        knife_visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=knife_stl,
            meshScale=[100, 100, 100],
            visualFrameOrientation=q_align
        )
        # Check if knife overlaps with the plate (threshold 0.12m) to spawn it above the plate
        dist_knife_plate = math.sqrt((knife_pos[0] - _PLATE_POS[0])**2 + (knife_pos[1] - _PLATE_POS[1])**2)
        knife_spawn_z = _PLATE_POS[2] + 0.03 if dist_knife_plate < 0.12 else _KNIFE_Z

        self.objects["knife"] = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=knife_col,
            baseVisualShapeIndex=knife_visual,
            basePosition=[knife_pos[0], knife_pos[1], knife_spawn_z],
            baseOrientation=knife_quat
        )
        
        # Step simulation to let spawned objects settle flat on the table
        for idx in range(7):
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=_FRANKA_REST_POSE[idx], force=1000)
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=1000)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=1000)
            
        for _ in range(100):
            p.stepSimulation()

        # Check if any object fell off the table during the settling phase
        for name, obj_id in self.objects.items():
            if name in ["fork", "knife"]:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                # If Z height is significantly lower than the table surface, it has fallen
                if pos[2] < _TABLE_SURFACE_Z - 0.01:
                    if self.verbose:
                        print(f"[DEBUG] {name} fell off the table during initialization settling phase! Z: {pos[2]:.4f}")
                    return False
        return True
    def get_topology_distance(self, a, b):
        """Calculates topological distance between two robot links in the kinematic tree."""
        path_a = []
        curr = a
        while curr is not None and curr != -1:
            path_a.append(curr)
            curr = self.parent_map.get(curr)
        path_a.append(-1)
        
        path_b = []
        curr = b
        while curr is not None and curr != -1:
            path_b.append(curr)
            curr = self.parent_map.get(curr)
        path_b.append(-1)
        
        for idx_a, node_a in enumerate(path_a):
            if node_a in path_b:
                idx_b = path_b.index(node_a)
                return idx_a + idx_b
        return 999

    def check_collision(self, ignored_body_id=None):
        """Checks for unintended collisions in the scene using specific distance boundaries."""
        # 1. Robot self-collision
        closest_pts = p.getClosestPoints(self.robot_id, self.robot_id, distance=self.self_collision_margin)
        for pt in closest_pts:
            link_a, link_b = pt[3], pt[4]
            # Exclude links with topological distance <= 2
            topo_dist = self.get_topology_distance(link_a, link_b)
            if topo_dist <= 2:
                continue
            if self.verbose:
                print(f"[DEBUG] Self-collision: link {link_a} and link {link_b} at distance {pt[8]:.4f} (topo_dist={topo_dist})")
            return True

        # 2. Table objects vs each other
        obj_ids = list(self.objects.values())
        for i in range(len(obj_ids)):
            for j in range(i+1, len(obj_ids)):
                body_a, body_b = obj_ids[i], obj_ids[j]
                
                name_a = next((k for k, v in self.objects.items() if v == body_a), str(body_a))
                name_b = next((k for k, v in self.objects.items() if v == body_b), str(body_b))
                
                if self.allow_plate_collision and ("plate" in (name_a, name_b)):
                    continue
                    
                pts = p.getClosestPoints(body_a, body_b, distance=self.obj_obj_collision_margin)
                if pts:
                    if self.verbose:
                        print(f"[DEBUG] Object-object collision: {name_a} and {name_b} at distance {pts[0][8]:.4f}")
                    return True

        # 3. Robot vs Table objects
        for name, obj_id in self.objects.items():
            if obj_id == ignored_body_id:
                continue
            if self.allow_plate_collision and name == "plate":
                continue
                
            pts = p.getClosestPoints(self.robot_id, obj_id, distance=self.robot_obj_collision_margin)
            if pts:
                if self.verbose:
                    print(f"[DEBUG] Robot-object collision: robot and {name} at distance {pts[0][8]:.4f}")
                return True
                
        return False

    def validate_trajectory(self, obj_pos, obj_yaw, target_place_pos, obj_name, target_yaw=0.0):
        """Simulates path segments and checks for self-limits and collisions."""
        # Reset joints to rest pose
        for idx, angle in enumerate(_FRANKA_REST_POSE):
            p.resetJointState(self.robot_id, idx, angle)
            p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.resetJointState(self.robot_id, 9, 0.04)
        p.resetJointState(self.robot_id, 10, 0.04)
            
        current_ee_state = p.getLinkState(self.robot_id, self.ee_index)
        current_ee_pos = np.array(current_ee_state[4])
        
        # Build Waypoints
        w_start = current_ee_pos
        w_hover = np.array([obj_pos[0], obj_pos[1], _Z_SAFE])
        
        # Calculate local frame offset away from the tip (towards the handle)
        # Fork (after 180deg X-axis flip): tip points to -Y in body frame, handle to +Y. Local offset = [0.0, 0.025, 0.0]
        # Knife (after 180deg X-axis flip): tip points to +Y in body frame, handle to -Y. Local offset = [0.0, -0.025, 0.0]
        obj_quat_xyzw = p.getQuaternionFromEuler([0.0, 0.0, obj_yaw])
        local_offset = [0.0, 0.025, 0.0] if obj_name == "fork" else [0.0, -0.025, 0.0]
        
        # Rotate offset to world frame
        world_offset_pos, _ = p.multiplyTransforms([0, 0, 0], obj_quat_xyzw, local_offset, [0, 0, 0, 1])
        grasp_xy = [obj_pos[0] + world_offset_pos[0], obj_pos[1] + world_offset_pos[1]]
            
        w_grasp = np.array([grasp_xy[0], grasp_xy[1], _Z_GRASP])
        w_lift = np.array([obj_pos[0], obj_pos[1], _Z_SAFE])
        w_transit = np.array([target_place_pos[0], target_place_pos[1], _Z_SAFE])
        w_place = np.array([target_place_pos[0], target_place_pos[1], _Z_RELEASE])
        w_retreat = np.array([target_place_pos[0], target_place_pos[1], _Z_SAFE])
        
        # Segment configuration with start and end yaws for interpolation
        segments = [
            (w_start, w_hover, 40, False, obj_yaw, obj_yaw),
            (w_hover, w_grasp, 30, False, obj_yaw, obj_yaw),
            (w_grasp, w_grasp, 15, False, obj_yaw, obj_yaw),  # Wait for arm to fully settle BEFORE fingers close
            (w_grasp, w_grasp, 15, True, obj_yaw, obj_yaw),   # Close fingers and create constraint
            (w_grasp, w_lift, 30, True, obj_yaw, obj_yaw),
            (w_lift, w_transit, 40, True, obj_yaw, target_yaw),
            (w_transit, w_place, 30, True, target_yaw, target_yaw),
            (w_place, w_place, 15, True, target_yaw, target_yaw),  # Wait for arm to settle at place pos
            (w_place, w_place, 15, False, target_yaw, target_yaw), # Open fingers and release constraint
            (w_place, w_retreat, 30, False, target_yaw, target_yaw)
        ]
        
        obj_body_id = self.objects[obj_name]
        grasp_constraint_id = None
        rel_pos, rel_quat = None, None
        
        for p_start, p_end, steps, has_grasped, yaw_s, yaw_e in segments:
            traj = generate_spline_trajectory(p_start, p_end, steps)
            
            # Shortest path angle interpolation for this segment
            diff = (yaw_e - yaw_s + math.pi) % (2.0 * math.pi) - math.pi
            yaw_traj = np.linspace(yaw_s, yaw_s + diff, steps)
            
            for step_idx, (target_pos, target_yaw_val) in enumerate(zip(traj, yaw_traj)):
                # Solve Inverse Kinematics command
                joint_angles = p.calculateInverseKinematics(
                    self.robot_id,
                    self.ee_index,
                    targetPosition=list(target_pos),
                    targetOrientation=p.getQuaternionFromEuler([math.pi, 0, target_yaw_val + math.pi/2]),
                    lowerLimits=self.ik_lower_limits,
                    upperLimits=self.ik_upper_limits,
                    jointRanges=self.ik_joint_ranges,
                    restPoses=self.ik_rest_poses
                )
                

                
                # Apply joint angles
                if self.arm_physics:
                    for j_idx, angle in enumerate(joint_angles[:7]):
                        lower, upper = self.joint_limits[j_idx]
                        clamped_angle = max(lower, min(upper, angle))
                        p.setJointMotorControl2(
                            bodyUniqueId=self.robot_id,
                            jointIndex=j_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=clamped_angle,
                            force=500.0,
                            maxVelocity=5.0
                        )
                    # Control finger joints physically
                    finger_target = 0.015 if has_grasped else 0.04
                    p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=finger_target, force=100.0)
                    p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=finger_target, force=100.0)
                    
                    # Run physics steps to let the arm move to target
                    physics_steps = 20
                    for _ in range(physics_steps):
                        p.stepSimulation()
                        if self.use_gui:
                            time.sleep(1.0 / 240.0) # Match PyBullet's internal 240Hz step for 1x real-time
                else:
                    # Apply joint angles kinematically (clamping to joint limits to match Isaac Sim)
                    for j_idx, angle in enumerate(joint_angles[:7]):
                        lower, upper = self.joint_limits[j_idx]
                        clamped_angle = max(lower, min(upper, angle))
                        p.resetJointState(self.robot_id, j_idx, clamped_angle)
                    
                    # Control finger joints kinematically
                    finger_target = 0.015 if has_grasped else 0.04
                    p.resetJointState(self.robot_id, 9, finger_target)
                    p.resetJointState(self.robot_id, 10, finger_target)
                    
                if has_grasped:
                    if grasp_constraint_id is None:
                        # Grab event! Attach the object at its actual physical relative transform.
                        # We use pure segments to ensure the arm has physically settled BEFORE closing the fingers.
                        # We attach the constraint to the gripper center (panda_grasptarget) to minimize lever arm error.
                        ee_state = p.getLinkState(self.robot_id, self.gripper_center_index)
                        ee_pos, ee_quat = ee_state[4], ee_state[5]
                        obj_pos_now, obj_quat_now = p.getBasePositionAndOrientation(obj_body_id)
                        
                        # Force object orientation to be perfectly flat (roll=0, pitch=0) at grasp
                        o_euler = list(p.getEulerFromQuaternion(obj_quat_now))
                        o_euler[0] = 0.0
                        o_euler[1] = 0.0
                        obj_quat_now = p.getQuaternionFromEuler(o_euler)
                        
                        # Calculate relative transform
                        inv_ee_pos, inv_ee_quat = p.invertTransform(ee_pos, ee_quat)
                        rel_pos, rel_quat = p.multiplyTransforms(inv_ee_pos, inv_ee_quat, obj_pos_now, obj_quat_now)
                        
                        grasp_constraint_id = p.createConstraint(
                            parentBodyUniqueId=self.robot_id,
                            parentLinkIndex=self.gripper_center_index,
                            childBodyUniqueId=obj_body_id,
                            childLinkIndex=-1,
                            jointType=p.JOINT_FIXED,
                            jointAxis=[0, 0, 0],
                            parentFramePosition=rel_pos,
                            childFramePosition=[0, 0, 0],
                            parentFrameOrientation=rel_quat,
                            childFrameOrientation=[0, 0, 0]
                        )
                    
                    if not self.arm_physics:
                        # Manually teleport object to match gripper exactly (kinematic tracking)
                        # This prevents springy constraints from twisting/tilting under high IK step forces.
                        ee_state = p.getLinkState(self.robot_id, self.gripper_center_index)
                        ee_pos, ee_quat = ee_state[4], ee_state[5]
                        target_obj_pos, target_obj_quat = p.multiplyTransforms(ee_pos, ee_quat, rel_pos, rel_quat)
                        
                        # Flatten the object orientation to roll = 0, pitch = 0 to prevent tilt propagation
                        target_e = p.getEulerFromQuaternion(target_obj_quat)
                        target_obj_quat_flat = p.getQuaternionFromEuler([0.0, 0.0, target_e[2]])
                        p.resetBasePositionAndOrientation(obj_body_id, target_obj_pos, target_obj_quat_flat)
                else:
                    if grasp_constraint_id is not None:
                        # Release event! Remove constraint
                        p.removeConstraint(grasp_constraint_id)
                        grasp_constraint_id = None
                    
                # Collision check
                ignored = obj_body_id
                collision_detected = self.check_collision(ignored_body_id=ignored)
                
                if collision_detected:
                    if self.verbose:
                        print(f"[DEBUG] Collision detected in segment starting at {p_start} to {p_end} (has_grasped={has_grasped})")
                    if grasp_constraint_id is not None:
                        p.removeConstraint(grasp_constraint_id)
                    return False
                    
                if self.use_gui:
                    time.sleep(self.sleep_time)
                    
        if grasp_constraint_id is not None:
            p.removeConstraint(grasp_constraint_id)
            
        # Lock final joint positions with motor position control before stepping the settling loop
        final_joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        for j_idx, angle in enumerate(final_joint_angles):
            p.setJointMotorControl2(self.robot_id, j_idx, p.POSITION_CONTROL, targetPosition=angle, force=1000)
            
        # Step simulation for 150 steps to let released object settle to rest under gravity
        for _ in range(150):
            p.stepSimulation()
            if self.use_gui:
                time.sleep(self.sleep_time)
            
            # Check collisions during settling
            if self.check_collision(ignored_body_id=None):
                if self.verbose:
                    print("[DEBUG] Collision detected during settling phase")
                return False
                
        # Stability check: ensure it hasn't flipped over or tilted significantly
        _, quat = p.getBasePositionAndOrientation(obj_body_id)
        matrix = p.getMatrixFromQuaternion(quat)
        local_z_in_world_z = matrix[8] # index 8 is R22 (local Z projection on world Z)
        if local_z_in_world_z < 0.5:
            if self.verbose:
                print(f"[DEBUG] Stability check failed: local_z_in_world_z={local_z_in_world_z:.4f}")
            return False
            
        # Height check: ensure it hasn't fallen off the table
        pos, _ = p.getBasePositionAndOrientation(obj_body_id)
        if pos[2] < _TABLE_SURFACE_Z - 0.01:
            if self.verbose:
                print(f"[DEBUG] Height check failed: pos[2]={pos[2]:.4f} (table height limit={_TABLE_SURFACE_Z - 0.01:.4f})")
            return False
            
        return True

    def run_validation_for_episode(self, fork_pos, fork_quat_wxyz, knife_pos, knife_quat_wxyz):
        """Tests kinematics and collisions for a specific cutlery pose setup."""
        # Convert wxyz quaternion to xyzw (PyBullet format)
        w, x, y, z = fork_quat_wxyz
        f_quat_xyzw = [x, y, z, w]
        w, x, y, z = knife_quat_wxyz
        k_quat_xyzw = [x, y, z, w]

        setup_ok = self.setup_scene(fork_pos, f_quat_xyzw, knife_pos, k_quat_xyzw)
        if not setup_ok:
            return False

        # Get settled positions and orientations after setup phase settling
        fork_settled_pos, fork_settled_quat = p.getBasePositionAndOrientation(self.objects["fork"])
        knife_settled_pos, knife_settled_quat = p.getBasePositionAndOrientation(self.objects["knife"])

        f_quat_wxyz_settled = [fork_settled_quat[3], fork_settled_quat[0], fork_settled_quat[1], fork_settled_quat[2]]
        k_quat_wxyz_settled = [knife_settled_quat[3], knife_settled_quat[0], knife_settled_quat[1], knife_settled_quat[2]]

        f_yaw = yaw_from_quat_wxyz(f_quat_wxyz_settled)
        k_yaw = yaw_from_quat_wxyz(k_quat_wxyz_settled)

        if getattr(self, "fix_fork_yaw", False):
            f_yaw = 0.0
        if getattr(self, "fix_knife_yaw", False):
            k_yaw = 0.0

        # 1. Test Knife (place right)
        k_place_pos = [_PLATE_POS[0] + 0.10, _PLATE_POS[1], _PLATE_POS[2]]
        knife_ok = self.validate_trajectory(knife_settled_pos, k_yaw, k_place_pos, "knife")
        if not knife_ok:
            return False

        # 2. Test Fork (place left)
        f_place_pos = [_PLATE_POS[0] - 0.10, _PLATE_POS[1], _PLATE_POS[2]]
        fork_ok = self.validate_trajectory(fork_settled_pos, f_yaw, f_place_pos, "fork", target_yaw=math.pi)
        return fork_ok

    def run_procedural_test(self, return_details=False):
        """Randomizes spawns and checks kinematics and collisions."""
        # Randomize spawn positions in a shared area and ensure no overlap (min distance)
        while True:
            f_x = random.uniform(*self.cutlery_spawn_x)
            f_y = random.uniform(*self.cutlery_spawn_y)
            k_x = random.uniform(*self.cutlery_spawn_x)
            k_y = random.uniform(*self.cutlery_spawn_y)
            
            # Check Euclidean distance between fork and knife centers
            dist = math.sqrt((f_x - k_x)**2 + (f_y - k_y)**2)
            if dist >= self.min_dist:
                break

        f_yaw = random.uniform(-math.pi, math.pi)
        f_quat = euler_xyz_to_quat_wxyz(0, 0, f_yaw)
        
        k_yaw = random.uniform(-math.pi, math.pi)
        k_quat = euler_xyz_to_quat_wxyz(0, 0, k_yaw)
        
        fork_pos = [f_x, f_y, FORK_Z]
        knife_pos = [k_x, k_y, KNIFE_Z]
        
        success = self.run_validation_for_episode(
            fork_pos, f_quat,
            knife_pos, k_quat
        )
        
        if return_details:
            return success, fork_pos, f_yaw, f_quat, knife_pos, k_yaw, k_quat
        return success

    def close(self):
        p.disconnect()


def main():
    parser = argparse.ArgumentParser(description="PyBullet Franka Trajectory Validator")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of random validation loops to execute.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging of failure reasons.")
    parser.add_argument(
        "--object_poses",
        type=str,
        default=None,
        help="Path to object_poses.json to validate demonstration episodes from Step 3."
    )
    parser.add_argument("--fix_knife_yaw", action="store_true", help="Force the knife's grasp yaw to be zero.")
    parser.add_argument("--fix_fork_yaw", action="store_true", help="Force the fork's grasp yaw to be zero.")
    parser.add_argument("--fps", type=float, default=100.0, help="Playback FPS for GUI visualization (default: 100.0).")
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.273,
        help="Minimum Euclidean distance between fork and knife (default: 0.273m based on STL AABBs).",
    )
    parser.add_argument(
        "--allow_plate_collision",
        action="store_true",
        help="Allow cutlery (fork/knife) to touch or collide with the plate without failing validation."
    )
    parser.add_argument(
        "--reconnect_interval",
        type=float,
        default=10.0,
        help="PyBullet GUI reconnection interval in seconds to clear rendering cache (default: 10.0s).",
    )
    parser.add_argument(
        "--arm-physics",
        action="store_true",
        help="Enable physical joint-motor control instead of kinematic joint resets."
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
    args = parser.parse_args()
    
    validator = PyBulletFrankaValidator(
        use_gui=args.gui, 
        verbose=args.verbose, 
        fps=args.fps, 
        min_dist=args.min_dist,
        reconnect_interval=args.reconnect_interval,
        allow_plate_collision=args.allow_plate_collision,
        arm_physics=args.arm_physics,
        self_collision_margin=args.self_collision_margin,
        robot_obj_collision_margin=args.robot_obj_collision_margin,
        obj_obj_collision_margin=args.obj_obj_collision_margin,
        spawn_margin=tuple(args.spawn_margin)
    )
    validator.fix_knife_yaw = args.fix_knife_yaw
    validator.fix_fork_yaw = args.fix_fork_yaw

    if args.object_poses is not None:
        # Load and validate processed UMI demonstration poses
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        simulator_src = repo_root / "packages" / "simulator" / "src"
        if simulator_src.exists() and str(simulator_src) not in sys.path:
            sys.path.insert(0, str(simulator_src))

        from simulator.utils.object_poses_loader import load_episode_poses, ObjectPoseConfig

        # Define Cutlery Task Constants locally to avoid importing isaaclab
        tag_to_object = {2: "knife", 3: "fork"}
        anchor_tag_id = 0
        anchor_world_pose = (0.40, 0.10, 0.0)
        object_z = _TABLE_SURFACE_Z + 0.05
        object_roll = 0.0
        object_pitch = 0.0
        per_object_yaw_offset = {
            "knife": math.pi,
            "fork": 2.0 * math.pi,
        }
        ignored_object_names = ("plate",)

        object_pose_cfg = ObjectPoseConfig(
            tag_to_object=tag_to_object,
            anchor_tag_id=anchor_tag_id,
            anchor_world_pose=anchor_world_pose,
            object_z=object_z,
            object_roll=object_roll,
            object_pitch=object_pitch,
            per_object_yaw_offset=per_object_yaw_offset,
            use_fixed_yaw=False,
            ignored_object_names=ignored_object_names,
        )

        print(f"Loading episodes from {args.object_poses}...")
        episodes = load_episode_poses(args.object_poses, object_pose_cfg)
        
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
