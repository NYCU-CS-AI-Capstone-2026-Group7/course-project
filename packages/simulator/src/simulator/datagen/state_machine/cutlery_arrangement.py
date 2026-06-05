"""State machine for the Franka cutlery-arrangement task."""

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

_MAX_CARTESIAN_DELTA = 0.018
_MAX_ROT_DELTA = 0.08
_IK_DLS_LAMBDA = 0.01

_HOVER_Z_OFFSET = 0.15
_GRASP_Z_OFFSET = 0.08
_LIFT_Z_OFFSET = 0.22
_RELEASE_Z_OFFSET = 0.09
_GRIPPER_DOWN_ROLL_W = math.pi
_GRIPPER_DOWN_PITCH_W = 0.0
_GRIPPER_DOWN_YAW_OFFSET_RANGE = (-0.15, 0.15)
# Grasp yaw bias (rad) on top of the object's world yaw, before the random
# jitter. Cutlery items are elongated, so π/2 closes the fingers across the
# short axis. Per-USD orientation correction lives in env_cfg's
# ``per_object_yaw_offset``.
_GRASP_YAW_OFFSET: float = math.pi / 2.0
# Horizontal retreat (m) toward the robot base applied to approach + grasp
# targets. Per-object so each cutlery item can be tuned independently
# (e.g. knife may grab better with no retreat than fork).
_GRASP_RETREAT_PER_OBJECT: dict[str, float] = {
    "fork": 0.025,
    "knife": 0.025,
}

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

# Pick order: fork first (place on +y / left of plate), then knife (place on -y / right)
_PICK_ORDER = (_KNIFE_NAME, _FORK_NAME)
_PLACE_X_SIGNS = (+1.0, -1.0)  # fork → +x of plate, knife → -x of plate

_STEP_SCALE_FACTOR = 1.0
_MAX_STEP_SCALE_FACTOR = 1.5
_MIN_STEP_SCALE_FACTOR = 0.1
_PHASE_DURATIONS_PER_OBJECT = tuple(int(d * _STEP_SCALE_FACTOR) for d in (270, 130, 20, 160, 255, 15, 25, 30))
_PHASES_PER_OBJECT = len(_PHASE_DURATIONS_PER_OBJECT)

_PHASE_NAMES = {
    0: "Move above object (Hover)",
    1: "Approach down to object",
    2: "Close gripper to grasp",
    3: "Lift object upward",
    4: "Move above target position (beside plate)",
    5: "Lower to release",
    6: "Open gripper to release",
    7: "Retreat upward"
}


def _constant_gripper(num_envs: int, device: torch.device, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), value, device=device)


def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale


def _shortest_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[:, 0:1] < 0.0, -quat, quat)


def _retreat_xy_toward(
    target_pos_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    distance: float,
) -> torch.Tensor:
    """Pull ``target_pos_w`` xy toward ``anchor_pos_w`` by ``distance`` metres."""
    out = target_pos_w.clone()
    delta_xy = out[:, :2] - anchor_pos_w[:, :2]
    norm = torch.linalg.norm(delta_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    out[:, :2] -= distance * (delta_xy / norm)
    return out


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Yaw (rotation about world z) from a (w, x, y, z) quaternion."""
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


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class CutleryArrangementStateMachine(StateMachineBase):
    """Scripted Franka policy for arranging cutlery around a plate.

    Picks up the fork and places it on the +y (left) side of the plate,
    then picks up the knife and places it on the -y (right) side.

    Each object goes through 7 phases:

    0. Move above object
    1. Approach down to object
    2. Close gripper to grasp
    3. Lift object upward
    4. Move above target position (beside plate)
    5. Lower and release
    6. Retreat upward

    The action vector is ``[panda_joint1, ..., panda_joint7, gripper]``.
    """

    _ph_durations = _PHASE_DURATIONS_PER_OBJECT
    _ph_timeouts = (
        int(_MAX_STEP_SCALE_FACTOR * _ph_durations[0]),  # Phase 0
        int(_MAX_STEP_SCALE_FACTOR * _ph_durations[1]),  # Phase 1
        _ph_durations[2],                                # Phase 2
        int(_MAX_STEP_SCALE_FACTOR * _ph_durations[3]),  # Phase 3
        int(_MAX_STEP_SCALE_FACTOR * _ph_durations[4]),  # Phase 4
        int(_MAX_STEP_SCALE_FACTOR * _ph_durations[5]),  # Phase 5
        _ph_durations[6],                                # Phase 6
        int(_MAX_STEP_SCALE_FACTOR * _ph_durations[7])   # Phase 7
    )
    MAX_STEPS: int = len(_PICK_ORDER) * sum(_ph_timeouts) + 100
    EPSILON_POS: float = 0.03
    EPSILON_ROT: float = 0.10

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._ee_body_idx: int = -1
        self._jacobi_body_idx: int = -1
        self._arm_joint_ids: list[int] = []
        self._jacobi_joint_ids: list[int] = []
        self._rest_joint_pos: torch.Tensor | None = None
        self._rest_ee_pos_w: torch.Tensor | None = None
        self._initial_ee_pos_w: torch.Tensor | None = None
        self._gripper_down_yaw_w: torch.Tensor | None = None
        self._gripper_down_yaw_offset_w: torch.Tensor | None = None
        self._current_object_idx: int = 0
        self._event: int = 0
        self._events_dt: list[int] = list(_PHASE_DURATIONS_PER_OBJECT) * len(_PICK_ORDER)
        self._last_target_pos_w: torch.Tensor | None = None
        self._last_target_quat_w: torch.Tensor | None = None
        self._last_advance_with_env: bool = False
        self._initial_obj_pos_w: torch.Tensor | None = None
        self._initial_obj_quat_w: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

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
        self._rest_ee_pos_w = self._ee_pos_w(robot).clone()

    def check_success(self, env) -> bool:
        plate_pos = env.scene[_PLATE_NAME].data.root_pos_w - env.scene.env_origins
        fork_pos = env.scene[_FORK_NAME].data.root_pos_w - env.scene.env_origins
        knife_pos = env.scene[_KNIFE_NAME].data.root_pos_w - env.scene.env_origins

        done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        fork_dist_xy = torch.norm(fork_pos[:, :2] - plate_pos[:, :2], dim=1)
        knife_dist_xy = torch.norm(knife_pos[:, :2] - plate_pos[:, :2], dim=1)

        done = torch.logical_and(done, fork_dist_xy <= _SUCCESS_MAX_DIST_XY)
        done = torch.logical_and(done, knife_dist_xy <= _SUCCESS_MAX_DIST_XY)
        done = torch.logical_and(done, fork_pos[:, 0] < plate_pos[:, 0]) # fork left
        done = torch.logical_and(done, knife_pos[:, 0] > plate_pos[:, 0]) # knife right

        return bool(done.all().item())

    def pre_step(self, env) -> None:
        pass

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs

        obj_name = _PICK_ORDER[self._current_object_idx]
        x_sign = _PLACE_X_SIGNS[self._current_object_idx]
        obj_pos_w = env.scene[obj_name].data.root_pos_w.clone()
        obj_quat_w = env.scene[obj_name].data.root_quat_w.clone()
        plate_pos_w = env.scene[_PLATE_NAME].data.root_pos_w.clone()
        robot_root_pos_w = robot.data.root_pos_w.clone()

        if self._initial_obj_pos_w is None or self._initial_obj_pos_w.shape[0] != num_envs:
            self._initial_obj_pos_w = obj_pos_w.clone()
            self._initial_obj_quat_w = obj_quat_w.clone()

        obj_pos_w_ref = self._initial_obj_pos_w
        obj_quat_w_ref = self._initial_obj_quat_w

        place_target_w = plate_pos_w.clone()
        place_target_w[:, 0] += x_sign * _PLACE_OFFSET

        if self._step_count == 0 and self._event == 0:
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        phase_in_cycle = self._event % _PHASES_PER_OBJECT

        target_quat_w = self._gripper_down_quat_w(
            obj_quat_w_ref,
            obj_name,
            num_envs,
            device,
            obj_quat_w_ref.dtype,
            yaw_offset=_GRASP_YAW_OFFSET,
            phase_in_cycle=phase_in_cycle,
        )

        # Calculate local frame offset away from the tip (towards the handle)
        # Fork: tip points to +z, handle points to -z
        # Knife: tip points to -z, handle points to +z
        device = obj_pos_w_ref.device
        num_envs = obj_pos_w_ref.shape[0]
        distance = _GRASP_RETREAT_PER_OBJECT.get(obj_name, 0.0)
        
        local_offset = torch.zeros((num_envs, 3), device=device, dtype=obj_pos_w_ref.dtype)
        if obj_name == _FORK_NAME:
            local_offset[:, 2] = -distance
        elif obj_name == _KNIFE_NAME:
            local_offset[:, 2] = distance
            
        world_offset = quat_apply(obj_quat_w_ref, local_offset)
        
        grasp_anchor_w = obj_pos_w_ref.clone()
        grasp_anchor_w[:, :2] += world_offset[:, :2]

        if phase_in_cycle == 0:
            target_pos_w, gripper_cmd = self._phase_move_above_object(obj_pos_w_ref, num_envs, device)
        elif phase_in_cycle == 1:
            target_pos_w, gripper_cmd = self._phase_approach_object(grasp_anchor_w, num_envs, device)
        elif phase_in_cycle == 2:
            target_pos_w, gripper_cmd = self._phase_grasp(grasp_anchor_w, num_envs, device)
        elif phase_in_cycle == 3:
            target_pos_w, gripper_cmd = self._phase_lift(obj_pos_w_ref, num_envs, device)
        elif phase_in_cycle == 4:
            target_pos_w, gripper_cmd = self._phase_move_above_place(place_target_w, num_envs, device)
        elif phase_in_cycle == 5:
            target_pos_w, gripper_cmd = self._phase_lower_to_release(place_target_w, num_envs, device)
        elif phase_in_cycle == 6:
            target_pos_w, gripper_cmd = self._phase_open_gripper(place_target_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_retreat(place_target_w, num_envs, device)

        self._last_target_pos_w = target_pos_w.clone()
        self._last_target_quat_w = target_quat_w.clone()

        return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_move_above_object(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _HOVER_Z_OFFSET
        if self._initial_ee_pos_w is not None:
            denom = max(self._events_dt[self._event] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            target = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_approach_object(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_grasp(self, obj_pos_w, num_envs, device):
        # Hold the same height as the approach phase so the EE doesn't keep
        # descending while the fingers are closing — that timing race causes
        # empty grasps.
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_move_above_place(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lower_to_release(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _RELEASE_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_open_gripper(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _RELEASE_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_retreat(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def check_arrival(self, env) -> torch.Tensor:
        """Checks if the end-effector has arrived at the target position and orientation.
        
        Returns a boolean tensor of shape (num_envs,) indicating whether each environment's
        end-effector is within the tolerance thresholds.
        """
        if self._last_target_pos_w is None or self._last_target_quat_w is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            
        robot = env.scene["robot"]
        ee_pos = self._ee_pos_w(robot)
        ee_quat = self._ee_quat_w(robot)
        
        # Position error: Euclidean distance
        pos_error = torch.norm(ee_pos - self._last_target_pos_w, dim=-1)
        
        # Rotation error: Angle between quaternions in radians
        dot_product = torch.sum(ee_quat * self._last_target_quat_w, dim=-1).abs()
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        rot_error = 2.0 * torch.acos(dot_product)
        
        epsilon_pos = self.EPSILON_POS
        epsilon_rot = self.EPSILON_ROT
        
        arrived = (pos_error <= epsilon_pos) & (rot_error <= epsilon_rot)
        return arrived

    def _log_timeout_diagnostics(self, env, phase_name: str, max_steps: int) -> None:
        if self._last_target_pos_w is None or self._last_target_quat_w is None:
            return
            
        robot = env.scene["robot"]
        ee_pos = self._ee_pos_w(robot)
        ee_quat = self._ee_quat_w(robot)
        obj_name = _PICK_ORDER[self._current_object_idx]
        
        # Calculate errors for env 0 (assuming single env or reporting the first env)
        target_pos = self._last_target_pos_w[0]
        actual_pos = ee_pos[0]
        target_quat = self._last_target_quat_w[0]
        actual_quat = ee_quat[0]
        
        pos_error = torch.norm(actual_pos - target_pos).item()
        
        dot_product = torch.dot(actual_quat, target_quat).abs().clamp(0.0, 1.0)
        rot_error_rad = (2.0 * torch.acos(dot_product)).item()
        rot_error_deg = math.degrees(rot_error_rad)
        
        epsilon_pos = self.EPSILON_POS
        epsilon_rot = self.EPSILON_ROT
        epsilon_rot_deg = math.degrees(epsilon_rot)
        
        print(f"\033[93m[StateMachine] [TIMEOUT] Event {self._event} ({phase_name}) of object '{obj_name}' timed out at step {self._step_count} (max steps: {max_steps})\033[0m")
        print(f"  -> \033[91mPosition mismatch\033[0m: Error = {pos_error:.4f} m (Tolerance: {epsilon_pos:.4f} m)")
        print(f"     Target: {target_pos.tolist()}")
        print(f"     Actual: {actual_pos.tolist()}")
        print(f"  -> \033[91mRotation mismatch\033[0m: Error = {rot_error_deg:.2f}° / {rot_error_rad:.4f} rad (Tolerance: {epsilon_rot_deg:.2f}° / {epsilon_rot:.4f} rad)")
        print(f"     Target Quat (wxyz): {target_quat.tolist()}")
        print(f"     Actual Quat (wxyz): {actual_quat.tolist()}")

    def advance(self, env=None) -> None:
        if self._episode_done:
            return

        self._step_count += 1
        
        if env is None:
            self._last_advance_with_env = False
            if self._step_count < self._events_dt[self._event]:
                return
            self._event += 1
            self._step_count = 0
        else:
            self._last_advance_with_env = True
            phase_in_cycle = self._event % _PHASES_PER_OBJECT
            default_duration = self._events_dt[self._event]
            obj_name = _PICK_ORDER[self._current_object_idx]
            
            if phase_in_cycle == 2 or phase_in_cycle == 6:
                # Phase 2: Close gripper to grasp
                # Phase 6: Open gripper to release
                # (Both are purely time-based open-loop durations)
                if self._step_count < default_duration:
                    return
                self._event += 1
                self._step_count = 0
            else:
                # Movement phases: 0, 1, 3, 4, 5, 7
                arrived = self.check_arrival(env).all()
                if phase_in_cycle == 5:
                    min_steps = max(5, int(5 * _STEP_SCALE_FACTOR))
                    max_steps = int(35 * _STEP_SCALE_FACTOR * _MAX_STEP_SCALE_FACTOR)
                else:
                    min_steps = max(10, int(_MIN_STEP_SCALE_FACTOR * default_duration))
                    max_steps = int(_MAX_STEP_SCALE_FACTOR * default_duration)
                
                # Check if arrived after minimum steps or if we timed out
                if arrived and self._step_count >= min_steps:
                    self._event += 1
                    self._step_count = 0
                elif self._step_count >= max_steps:
                    phase_name = _PHASE_NAMES.get(phase_in_cycle, f"Phase {phase_in_cycle}")
                    self._log_timeout_diagnostics(env, phase_name, max_steps)
                    self._event += 1
                    self._step_count = 0

        # Post-event transition check
        if self._event >= len(self._events_dt):
            self._episode_done = True
            return

        new_obj_idx = self._event // _PHASES_PER_OBJECT
        if new_obj_idx != self._current_object_idx:
            self._current_object_idx = new_obj_idx
            self._initial_ee_pos_w = None
            self._gripper_down_yaw_w = None
            self._gripper_down_yaw_offset_w = None
            self._initial_obj_pos_w = None
            self._initial_obj_quat_w = None

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._event = 0
        self._current_object_idx = 0
        self._initial_ee_pos_w = None
        self._gripper_down_yaw_w = None
        self._gripper_down_yaw_offset_w = None
        self._last_target_pos_w = None
        self._last_target_quat_w = None
        self._last_advance_with_env = False
        self._initial_obj_pos_w = None
        self._initial_obj_quat_w = None

    # ------------------------------------------------------------------
    # IK / control helpers (same as CupStackingStateMachine)
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

        target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
        ee_pos_root = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
        delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)

        delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
        delta_rot_w = axis_angle_from_quat(delta_quat_w)
        delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)

        pose_delta_root = torch.cat([delta_pos_root, delta_rot_root], dim=-1)
        joint_pos_target = self._arm_joint_pos(robot) + self._compute_delta_joint_pos(
            pose_delta_root, self._ee_jacobian_root(robot)
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

    def _compute_delta_joint_pos(self, pose_delta: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (_IK_DLS_LAMBDA**2) * torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        )
        delta_joint_pos = (
            jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ pose_delta.unsqueeze(-1)
        )
        return delta_joint_pos.squeeze(-1)

    def _clamp_arm_joint_pos(self, robot, joint_pos: torch.Tensor) -> torch.Tensor:
        joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(robot.data, "joint_pos_limits", None)
        if joint_pos_limits is None:
            return joint_pos
        arm_joint_pos_limits = joint_pos_limits[:, self._arm_joint_ids, :]
        return torch.clamp(joint_pos, arm_joint_pos_limits[..., 0], arm_joint_pos_limits[..., 1])

    def _gripper_down_quat_w(
        self,
        obj_quat_w: torch.Tensor,
        obj_name: str,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        yaw_offset: float = 0.0,
        phase_in_cycle: int = 0,
    ) -> torch.Tensor:
        if self._gripper_down_yaw_w is None or self._gripper_down_yaw_w.shape[0] != num_envs:
            base_yaw = _yaw_from_quat_wxyz(obj_quat_w).to(device=device, dtype=dtype) # gripper aligned with the orientation of the object
            self._gripper_down_yaw_offset_w = torch.empty(num_envs, device=device, dtype=dtype).uniform_(
                _GRIPPER_DOWN_YAW_OFFSET_RANGE[0],
                _GRIPPER_DOWN_YAW_OFFSET_RANGE[1],
            )
            #if obj_name == _KNIFE_NAME:
            #    base_yaw = torch.zeros_like(base_yaw) # fixed direction
            self._gripper_down_yaw_w = (
                base_yaw + yaw_offset + self._gripper_down_yaw_offset_w
            ).clone()

        if phase_in_cycle == 4:
            # Gradually interpolate yaw from grasp yaw to place yaw during phase 4
            target_obj_yaw = math.pi if obj_name == _FORK_NAME else 0.0
            place_yaw = target_obj_yaw + yaw_offset + self._gripper_down_yaw_offset_w
            grasp_yaw = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
            
            denom = max(self._events_dt[self._event] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            
            diff = (place_yaw - grasp_yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw = grasp_yaw + alpha * diff
        elif phase_in_cycle > 4:
            target_obj_yaw = math.pi if obj_name == _FORK_NAME else 0.0
            yaw = torch.full((num_envs,), target_obj_yaw + yaw_offset, device=device, dtype=dtype)
            yaw += self._gripper_down_yaw_offset_w
        else:
            yaw = self._gripper_down_yaw_w.to(device=device, dtype=dtype)

        roll = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W, device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return (_FORK_NAME, _KNIFE_NAME, _PLATE_NAME)
