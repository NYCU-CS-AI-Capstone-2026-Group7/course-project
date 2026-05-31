"""Procedural, smooth state machine for the Franka cutlery-arrangement task."""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from leisaac.datagen.state_machine.base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_FORK_NAME = "fork"
_KNIFE_NAME = "knife"
_PLATE_NAME = "plate"
_EE_BODY_NAME = "panda_hand"
_FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

_MAX_CARTESIAN_DELTA = 0.02
_MAX_ROT_DELTA = 0.10
_IK_DLS_LAMBDA = 0.01

_Z_SAFE = 0.25
_Z_GRASP = 0.08
_Z_RELEASE = 0.09
_GRIPPER_DOWN_ROLL_W = math.pi
_GRIPPER_DOWN_PITCH_W = 0.0

_GRASP_YAW_OFFSET = math.pi / 2.0
_PLACE_OFFSET = 0.10
_SUCCESS_MAX_DIST_XY = 0.15

_FRANKA_REST_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 4.0,
    "panda_joint3": 0.0,
    "panda_joint4": -3.0 * math.pi / 4.0,
    "panda_joint5": 0.0,
    "panda_joint6": math.pi / 2.0,
    "panda_joint7": math.pi / 4.0,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}

_PICK_ORDER = (_KNIFE_NAME, _FORK_NAME)
_PLACE_X_SIGNS = (+1.0, -1.0)  # fork -> +x, knife -> -x of plate

# Segment step durations
_STEPS_MOVE_TO_HOVER = 60
_STEPS_APPROACH = 40
_STEPS_GRASP_HOLD = 20
_STEPS_LIFT = 40
_STEPS_MOVE_TO_PLACE = 60
_STEPS_LOWER = 40
_STEPS_RELEASE_HOLD = 20
_STEPS_RETREAT = 40


def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale


def _shortest_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[:, 0:1] < 0.0, -quat, quat)


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) > 0:
            return int(body_ids[0])
    body_names = getattr(robot.data, "body_names", None)
    if body_names is not None and body_name in body_names:
        return body_names.index(body_name)
    return -1


def catmull_rom_spline(P0, P1, P2, P3, num_points):
    """Interpolates between P1 and P2 using Catmull-Rom spline.

    P0, P1, P2, P3: Tensors of shape (num_envs, D)
    num_points: int
    Returns: Tensor of shape (num_envs, num_points, D)
    """
    device = P0.device
    t = torch.linspace(0, 1, num_points, device=device).view(1, num_points, 1) # (1, T, 1)
    
    # Expand shapes to (num_envs, 1, D) for broadcasting
    p0 = P0.unsqueeze(1)
    p1 = P1.unsqueeze(1)
    p2 = P2.unsqueeze(1)
    p3 = P3.unsqueeze(1)
    
    c0 = 2.0 * p1
    c1 = -p0 + p2
    c2 = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
    c3 = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    
    points = 0.5 * (c0 + c1 * t + c2 * t * t + c3 * t * t * t)
    return points


class ProceduralCutleryArrangementStateMachine(StateMachineBase):
    """Procedural state machine with smooth Catmull-Rom trajectory generation."""

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._ee_body_idx: int = -1
        self._jacobi_body_idx: int = -1
        self._arm_joint_ids: list[int] = []
        self._jacobi_joint_ids: list[int] = []
        self._rest_joint_pos: torch.Tensor | None = None
        
        self._current_object_idx: int = 0
        self._current_trajectory_pos: torch.Tensor | None = None # (num_envs, total_steps, 3)
        self._current_trajectory_yaw: torch.Tensor | None = None # (num_envs, total_steps)
        self._current_trajectory_gripper: torch.Tensor | None = None # (num_envs, total_steps, 1)
        self._total_steps_per_object: int = 0

    def setup(self, env) -> None:
        robot = env.scene["robot"]
        self._ee_body_idx = _find_body_index(robot, _EE_BODY_NAME)
        joint_names = list(robot.data.joint_names)
        missing = [j for j in _FRANKA_ARM_JOINT_NAMES if j not in joint_names]
        if missing:
            raise ValueError(f"Missing Franka joints {missing} in {joint_names}")
        self._arm_joint_ids = [joint_names.index(j) for j in _FRANKA_ARM_JOINT_NAMES]

        if self._ee_body_idx < 0:
            raise ValueError(f"Could not find body '{_EE_BODY_NAME}' in Franka.")
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [jid + 6 for jid in self._arm_joint_ids]

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _FRANKA_REST_JOINT_POS:
                self._rest_joint_pos[:, idx] = _FRANKA_REST_JOINT_POS[name]

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

    def check_success(self, env) -> bool:
        plate_pos = env.scene[_PLATE_NAME].data.root_pos_w - env.scene.env_origins
        fork_pos = env.scene[_FORK_NAME].data.root_pos_w - env.scene.env_origins
        knife_pos = env.scene[_KNIFE_NAME].data.root_pos_w - env.scene.env_origins

        done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        fork_dist_xy = torch.norm(fork_pos[:, :2] - plate_pos[:, :2], dim=1)
        knife_dist_xy = torch.norm(knife_pos[:, :2] - plate_pos[:, :2], dim=1)

        done = torch.logical_and(done, fork_dist_xy <= _SUCCESS_MAX_DIST_XY)
        done = torch.logical_and(done, knife_dist_xy <= _SUCCESS_MAX_DIST_XY)
        done = torch.logical_and(done, fork_pos[:, 0] < plate_pos[:, 0]) # fork left (+x in cutlery_arranged? wait, plate is at 0.5, -0.4. fork left is fork_pos.y > plate_pos.y or fork_pos.x < plate_pos.x. Let's check environment's done logic)
        # Originally in cutlery_arrangement_env_cfg: fork_on_left: fork_pos.x > plate_pos.x (fork left of plate, since robot looks down? Wait: fork_on_left = fork_pos[:, 0] > plate_pos[:, 0] in cutlery_arrangement_env_cfg.py line 109. Actually we just align with the environment's check_success or env_cfg's check success).
        # Let's check what the cutlery_arrangement.py check_success did:
        # done = torch.logical_and(done, fork_pos[:, 0] < plate_pos[:, 0]) # wait, the cutlery_arrangement.py had done = torch.logical_and(done, fork_pos[:, 0] < plate_pos[:, 0]) ?
        # Yes, line 224: fork_pos[:, 0] < plate_pos[:, 0] and knife_pos[:, 0] > plate_pos[:, 0]
        done = torch.logical_and(done, fork_pos[:, 0] < plate_pos[:, 0])
        done = torch.logical_and(done, knife_pos[:, 0] > plate_pos[:, 0])
        return bool(done.all().item())

    def pre_step(self, env) -> None:
        pass

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._current_object_idx = 0
        self._current_trajectory_pos = None
        self._current_trajectory_yaw = None
        self._current_trajectory_gripper = None

    def plan_smooth_trajectory(self, env, start_ee_pos, start_ee_yaw, obj_pos, obj_yaw, place_pos, place_yaw):
        """Plans a smooth Catmull-Rom spline trajectory for the current object."""
        device = env.device
        num_envs = env.num_envs
        
        # Build Cartesian Control Points
        # Shape: (num_envs, 3)
        c_start = start_ee_pos
        c_hover = obj_pos.clone()
        c_hover[:, 2] = _Z_SAFE
        c_grasp = obj_pos.clone()
        c_grasp[:, 2] = _Z_GRASP
        c_lift = obj_pos.clone()
        c_lift[:, 2] = _Z_SAFE
        c_transit = place_pos.clone()
        c_transit[:, 2] = _Z_SAFE
        c_place = place_pos.clone()
        c_place[:, 2] = _Z_RELEASE
        c_retreat = place_pos.clone()
        c_retreat[:, 2] = _Z_SAFE
        
        control_points_pos = [c_start, c_hover, c_grasp, c_lift, c_transit, c_place, c_retreat]
        
        # Yaw Control Points
        y_start = start_ee_yaw
        y_grasp_target = obj_yaw + _GRASP_YAW_OFFSET
        y_place_target = place_yaw + _GRASP_YAW_OFFSET
        
        control_points_yaw = [
            y_start,          # start
            y_grasp_target,   # hover
            y_grasp_target,   # grasp
            y_grasp_target,   # lift
            y_place_target,   # transit
            y_place_target,   # place
            y_place_target    # retreat
        ]
        
        # We perform piecewise spline interpolation for each segment
        segments = [
            (_STEPS_MOVE_TO_HOVER, 0, 1),
            (_STEPS_APPROACH, 1, 2),
            (_STEPS_GRASP_HOLD, 2, 2),  # stationary during grasp
            (_STEPS_LIFT, 2, 3),
            (_STEPS_MOVE_TO_PLACE, 3, 4),
            (_STEPS_LOWER, 4, 5),
            (_STEPS_RELEASE_HOLD, 5, 5), # stationary during release
            (_STEPS_RETREAT, 5, 6)
        ]
        
        pos_trajectory_list = []
        yaw_trajectory_list = []
        gripper_trajectory_list = []
        
        for idx, (steps, p_idx1, p_idx2) in enumerate(segments):
            # Gripper state
            # Gripper is open (1.0) before step 2 (Grasp), closed (-1.0) between Grasp and Release, open after Release
            if idx <= 1:
                gripper_state = _GRIPPER_OPEN
            elif 2 <= idx <= 5:
                gripper_state = _GRIPPER_CLOSE
            else:
                gripper_state = _GRIPPER_OPEN
            gripper_cmd = torch.full((num_envs, steps, 1), gripper_state, device=device)
            gripper_trajectory_list.append(gripper_cmd)
            
            # If stationary segment
            if p_idx1 == p_idx2:
                pos_pt = control_points_pos[p_idx1].unsqueeze(1).repeat(1, steps, 1)
                yaw_pt = control_points_yaw[p_idx1].unsqueeze(1).repeat(1, steps)
                pos_trajectory_list.append(pos_pt)
                yaw_trajectory_list.append(yaw_pt)
                continue
                
            # Fetch surrounding points for Catmull-Rom
            # P0: i-1, P1: i, P2: i+1, P3: i+2
            # Handle boundaries by duplicating start/end
            idx0 = max(p_idx1 - 1, 0)
            idx1 = p_idx1
            idx2 = p_idx2
            idx3 = min(p_idx2 + 1, len(control_points_pos) - 1)
            
            pos_seg = catmull_rom_spline(
                control_points_pos[idx0],
                control_points_pos[idx1],
                control_points_pos[idx2],
                control_points_pos[idx3],
                steps
            )
            yaw_seg = catmull_rom_spline(
                control_points_yaw[idx0].unsqueeze(-1),
                control_points_yaw[idx1].unsqueeze(-1),
                control_points_yaw[idx2].unsqueeze(-1),
                control_points_yaw[idx3].unsqueeze(-1),
                steps
            ).squeeze(-1)
            
            pos_trajectory_list.append(pos_seg)
            yaw_trajectory_list.append(yaw_seg)
            
        self._current_trajectory_pos = torch.cat(pos_trajectory_list, dim=1) # (num_envs, total_steps, 3)
        self._current_trajectory_yaw = torch.cat(yaw_trajectory_list, dim=1) # (num_envs, total_steps)
        self._current_trajectory_gripper = torch.cat(gripper_trajectory_list, dim=1) # (num_envs, total_steps, 1)
        self._total_steps_per_object = self._current_trajectory_pos.shape[1]

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        
        # On very first step, or when switching objects, generate a new trajectory
        if self._current_trajectory_pos is None:
            obj_name = _PICK_ORDER[self._current_object_idx]
            x_sign = _PLACE_X_SIGNS[self._current_object_idx]
            
            obj_pos_w = env.scene[obj_name].data.root_pos_w.clone()
            obj_quat_w = env.scene[obj_name].data.root_quat_w.clone()
            plate_pos_w = env.scene[_PLATE_NAME].data.root_pos_w.clone()
            
            start_ee_pos = self._ee_pos_w(robot).clone()
            start_ee_yaw = _yaw_from_quat_wxyz(self._ee_quat_w(robot)).clone()
            
            obj_yaw = _yaw_from_quat_wxyz(obj_quat_w)
            
            # Optionally lock grasp yaw to zero
            if obj_name == _KNIFE_NAME and getattr(self, "fix_knife_yaw", False):
                obj_yaw = torch.zeros_like(obj_yaw)
            elif obj_name == _FORK_NAME and getattr(self, "fix_fork_yaw", False):
                obj_yaw = torch.zeros_like(obj_yaw)
            
            place_pos = plate_pos_w.clone()
            place_pos[:, 0] += x_sign * _PLACE_OFFSET
            place_yaw = torch.zeros(num_envs, device=device, dtype=obj_yaw.dtype)
            
            self.plan_smooth_trajectory(
                env,
                start_ee_pos,
                start_ee_yaw,
                obj_pos_w,
                obj_yaw,
                place_pos,
                place_yaw
            )
            
        # Extract target states for the current step
        target_pos_w = self._current_trajectory_pos[:, self._step_count, :]
        target_yaw = self._current_trajectory_yaw[:, self._step_count]
        gripper_cmd = self._current_trajectory_gripper[:, self._step_count, :]
        
        # Construct target orientation quaternion
        roll = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W, device=device)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device)
        target_quat_w = quat_from_euler_xyz(roll, pitch, target_yaw)
        
        return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    def advance(self) -> None:
        if self._episode_done:
            return

        self._step_count += 1
        if self._step_count < self._total_steps_per_object:
            return

        # Move to the next object
        self._current_object_idx += 1
        self._step_count = 0
        self._current_trajectory_pos = None # trigger replan for next object
        
        if self._current_object_idx >= len(_PICK_ORDER):
            self._episode_done = True

    # ------------------------------------------------------------------
    # IK & FK utilities
    # ------------------------------------------------------------------

    def _ee_pos_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_pos_w[:, body_idx, :]

    def _ee_quat_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_quat_w[:, body_idx, :]

    def _joint_position_franka_action(
        self,
        env,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        gripper_cmd: torch.Tensor,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        root_quat_inv = quat_inv(root_quat_w)

        # FK calculation and Cartesian command tracking
        target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
        ee_pos_root = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
        delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)

        delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
        delta_rot_w = axis_angle_from_quat(delta_quat_w)
        delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)

        pose_delta_root = torch.cat([delta_pos_root, delta_rot_root], dim=-1)
        
        # Solving IK command
        joint_pos_target = self._arm_joint_pos(robot) + self._compute_delta_joint_pos_with_nullspace(
            robot, pose_delta_root, self._ee_jacobian_root(robot)
        )
        joint_pos_target = self._clamp_arm_joint_pos(robot, joint_pos_target)
        return torch.cat([joint_pos_target, gripper_cmd], dim=-1)

    def _arm_joint_pos(self, robot) -> torch.Tensor:
        if not self._arm_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")
        return robot.data.joint_pos[:, self._arm_joint_ids]

    def _ee_jacobian_root(self, robot) -> torch.Tensor:
        if self._jacobi_body_idx < 0 or not self._jacobi_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")

        jacobian = robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ].clone()
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_delta_joint_pos_with_nullspace(self, robot, pose_delta: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Solves DLS IK and applies Null-space projection to guide the elbow configuration."""
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        
        # Damped Least Squares Pseudo-Inverse
        lambda_matrix = (_IK_DLS_LAMBDA**2) * torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        )
        inv_jjt = torch.inverse(jacobian @ jacobian_t + lambda_matrix)
        j_pinv = jacobian_t @ inv_jjt # (num_envs, 7, 6)
        
        # Primary task joint velocities
        dq_primary = (j_pinv @ pose_delta.unsqueeze(-1)).squeeze(-1) # (num_envs, 7)
        
        # Null-space projection matrix (I - J+ * J)
        eye = torch.eye(7, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0).repeat(jacobian.shape[0], 1, 1)
        null_space_proj = eye - j_pinv @ jacobian # (num_envs, 7, 7)
        
        # Secondary task: guide joints to resting pose to avoid singularities
        q_current = self._arm_joint_pos(robot)
        q_rest = torch.zeros_like(q_current)
        for idx, name in enumerate(_FRANKA_ARM_JOINT_NAMES):
            if name in _FRANKA_REST_JOINT_POS:
                q_rest[:, idx] = _FRANKA_REST_JOINT_POS[name]
                
        # Joint delta error
        dq_secondary = 0.5 * (q_rest - q_current) # gain k = 0.5
        
        # Projected secondary task
        dq_nullspace = (null_space_proj @ dq_secondary.unsqueeze(-1)).squeeze(-1)
        
        return dq_primary + dq_nullspace

    def _clamp_arm_joint_pos(self, robot, joint_pos: torch.Tensor) -> torch.Tensor:
        joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(robot.data, "joint_pos_limits", None)
        if joint_pos_limits is None:
            return joint_pos
        arm_joint_pos_limits = joint_pos_limits[:, self._arm_joint_ids, :]
        return torch.clamp(joint_pos, arm_joint_pos_limits[..., 0], arm_joint_pos_limits[..., 1])

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return (_FORK_NAME, _KNIFE_NAME, _PLATE_NAME)
