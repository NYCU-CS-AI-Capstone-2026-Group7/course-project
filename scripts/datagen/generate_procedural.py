"""Unified procedural data generation script using randomized spawns and smooth IK control.

Generates cutlery arrangement datasets in Isaac Sim without relying on UMI object_poses.json.
Spawns cutlery randomly in valid zones, applies global frame perturbations, and records trajectories.

Usage:
    python scripts/datagen/generate_procedural.py \
        --task HCIS-CutleryArrangement-SingleArm-v0 \
        --num_envs 1 --device cuda --enable_cameras \
        --num_demos 50 --record --use_lerobot_recorder \
        --lerobot_dataset_repo_id ${HF_USER}/cutlery_procedural
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import signal
import time
import math
import random

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Procedural data generation for LeIsaac tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="HCIS-CutleryArrangement-SingleArm-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--record", action="store_true", help="Whether to enable record function.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/procedural_dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="Whether to resume recording in the existing dataset file.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demonstrations to record.")
parser.add_argument("--quality", action="store_true", help="Whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")
parser.add_argument("--fix_knife_yaw", action="store_true", help="Force the knife's grasp yaw to be zero.")
parser.add_argument("--fix_fork_yaw", action="store_true", help="Force the fork's grasp yaw to be zero.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import simulator.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

from simulator.datagen.state_machine.procedural_cutlery import ProceduralCutleryArrangementStateMachine

# Maps gym task id -> (StateMachineClass, device_type)
TASK_REGISTRY = {
    "HCIS-CutleryArrangement-SingleArm-v0": (ProceduralCutleryArrangementStateMachine, "keyboard"),
}

# Spawning ranges in world frame
_PLATE_X_RANGE = (0.48, 0.52)
_PLATE_Y_RANGE = (-0.42, -0.38)
_PLATE_Z = 0.05

_FORK_X_RANGE = (0.35, 0.48)
_FORK_Y_RANGE = (-0.28, -0.18) # left
_FORK_Z = 0.05

_KNIFE_X_RANGE = (0.35, 0.48)
_KNIFE_Y_RANGE = (-0.62, -0.52) # right
_KNIFE_Z = 0.05


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name):
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None


def _replace_recorder_manager(env, env_cfg, args_cli):
    del env.recorder_manager
    if args_cli.use_lerobot_recorder:
        from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
        from leisaac.enhance.managers.lerobot_recorder_manager import (
            LeRobotRecorderManager,
        )

        dataset_cfg = LeRobotDatasetCfg(
            repo_id=args_cli.lerobot_dataset_repo_id,
            fps=args_cli.lerobot_dataset_fps,
        )
        env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
    else:
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (cy, 0.0, 0.0, sy)


def _apply_random_spawns(env):
    """Procedurally randomizes object positions and writes them to sim."""
    device = env.device
    num_envs = env.num_envs
    
    # Global frame perturbation (平移與旋轉)
    d_x = random.uniform(-0.04, 0.04)
    d_y = random.uniform(-0.04, 0.04)
    d_yaw = random.uniform(-math.radians(15), math.radians(15))
    cos_dy = math.cos(d_yaw)
    sin_dy = math.sin(d_yaw)

    def transform_point(x, y, z):
        # Rotate around world origin, then translate
        rx = cos_dy * x - sin_dy * y + d_x
        ry = sin_dy * x + cos_dy * y + d_y
        return rx, ry, z

    # 1. Plate Pose
    p_x = random.uniform(*_PLATE_X_RANGE)
    p_y = random.uniform(*_PLATE_Y_RANGE)
    p_tx, p_ty, p_tz = transform_point(p_x, p_y, _PLATE_Z)
    p_quat = _yaw_to_quat_wxyz(d_yaw)
    
    # 2. Fork Pose (left)
    f_x = random.uniform(*_FORK_X_RANGE)
    f_y = random.uniform(*_FORK_Y_RANGE)
    f_tx, f_ty, f_tz = transform_point(f_x, f_y, _FORK_Z)
    f_yaw = random.uniform(-math.pi, math.pi) + d_yaw
    f_quat = _yaw_to_quat_wxyz(f_yaw)

    # 3. Knife Pose (right)
    k_x = random.uniform(*_KNIFE_X_RANGE)
    k_y = random.uniform(*_KNIFE_Y_RANGE)
    k_tx, k_ty, k_tz = transform_point(k_x, k_y, _KNIFE_Z)
    k_yaw = random.uniform(-math.pi, math.pi) + d_yaw
    k_quat = _yaw_to_quat_wxyz(k_yaw)

    # Apply to simulation
    poses = {
        "plate": ((p_tx, p_ty, p_tz), p_quat),
        "fork": ((f_tx, f_ty, f_tz), f_quat),
        "knife": ((k_tx, k_ty, k_tz), k_quat),
    }

    for name, (pos, quat) in poses.items():
        obj = env.scene[name]
        pose_tensor = torch.tensor(
            [[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
            device=device,
            dtype=torch.float32,
        ).repeat(num_envs, 1)
        obj.write_root_pose_to_sim(pose_tensor)
        print(f"  [Procedural Spawn] {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) yaw={math.degrees(math.atan2(2.0*(quat[0]*quat[3]), 1.0-2.0*(quat[3]**2))):+6.1f}°")


def _on_episode_done(
    env,
    sm,
    args_cli,
    current_recorded_demo_count,
    start_record_state,
):
    try:
        success = sm.check_success(env)
    except Exception as e:
        print("Success check failed:", e)
        success = False

    print("Procedural Episode success!" if success else "Procedural Episode failed!")

    if start_record_state:
        if args_cli.record:
            print("Stop Recording!!!")
        start_record_state = False

    if args_cli.record and success:
        auto_terminate(env, True)
        current_recorded_demo_count += 1
    else:
        auto_terminate(env, False)

    if (
        args_cli.record
        and env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count
    ):
        current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

    should_break = current_recorded_demo_count >= args_cli.num_demos

    env.reset()
    sm.reset()
    auto_terminate(env, False)
    _apply_random_spawns(env)

    return current_recorded_demo_count, start_record_state, should_break, success


def main():
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task '{task_name}' is not in TASK_REGISTRY. Currently supporting procedural cutlery arrangement.")
    SMClass, device = TASK_REGISTRY[task_name]

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    _configure_env_cfg(env_cfg, args_cli, False, output_dir, output_file_name)

    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # Disable gravity for robot joints
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics
    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if args_cli.record:
        _replace_recorder_manager(env, env_cfg, args_cli)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()

    sm = SMClass()
    sm.fix_knife_yaw = args_cli.fix_knife_yaw
    sm.fix_fork_yaw = args_cli.fix_fork_yaw
    sm.setup(env)
    env.reset()
    sm.reset()

    _apply_random_spawns(env)

    current_recorded_demo_count = 0
    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    cnt = 1
    
    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.device)

                if sm.is_episode_done:
                    (
                        current_recorded_demo_count,
                        start_record_state,
                        should_break,
                        success,
                    ) = _on_episode_done(
                        env,
                        sm,
                        args_cli,
                        current_recorded_demo_count,
                        start_record_state,
                    )
                    if success:
                        print(f"\033[92m[Procedural Generation] {cnt}/{args_cli.num_demos} success.\033[0m")
                        cnt += 1
                    else:
                        print(f"\033[91m[Procedural Generation] Episode failed. Retrying.\033[0m")
                    
                    if should_break:
                        break
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True

                    sm.pre_step(env)
                    actions = sm.get_action(env)
                    
                    # Optional Action/Trajectory perturbation
                    # Add tiny smooth joint command noise
                    if args_cli.record:
                        noise = torch.randn_like(actions[:, :7]) * 0.002
                        actions[:, :7] += noise
                        
                    env.step(actions)
                    sm.advance()

                if rate_limiter:
                    rate_limiter.sleep(env)

            if interrupted:
                break
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
