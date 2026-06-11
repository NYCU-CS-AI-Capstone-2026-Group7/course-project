# make launch-isaaclab-glowsai-l40s
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