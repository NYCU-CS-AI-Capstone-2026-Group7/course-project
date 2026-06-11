# make launch-isaaclab-glowsai-l40s
# data generation
python scripts/datagen/generate.py \
    --task HCIS-CutleryArrangement-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --headless \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id fanyi000/cultery_synth \
    --augment_pose_factor 10 \
    --augment_global_xy_jitter 0.01 \
    --augment_local_xy_jitter 0.05 \
    --object_poses data/AI-final-49/object_poses.json

# use some dummy data
python scripts/datagen/generate.py \
    --task HCIS-CutleryArrangement-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --headless \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id fanyi000/cultery_synth \
    --augment_pose_factor 10 \
    --augment_global_xy_jitter 0.01 \
    --augment_local_xy_jitter 0.05 \
    --object_poses data/AI-final-49/object_poses_combined.json

# training
uv sync && source .venv/bin/activate
lerobot-train \
  --dataset.repo_id=${HF_USER}/cultery_synth \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_v1 \
  --job_name=cupstacking_diffusion_v1 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=capstone-manipulation \
  --wandb.entity=<your-wandb-username> \
  --policy.repo_id=${HF_USER}/my_policy

# training exp1
uv sync && source .venv/bin/activate
lerobot-train \
  --dataset.repo_id=${HF_USER}/cultery_synth \
  --policy.type=act \
  --output_dir=<your-output-dir> \
  --job_name=cupstacking_act \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy_act

# evaluation
python scripts/rollout.py \
    --task=eval/cutlery_arrangement_eval.py \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=outputs/my_policy_v2 \
    --policy_action_horizon=8 \
    --device=cuda \
    --headless \
    --enable_cameras \
    --eval_rounds=2 \
    --episode_length_s=60 \
    2>&1 | tee /workspace/aicapstone/eval_results.txt