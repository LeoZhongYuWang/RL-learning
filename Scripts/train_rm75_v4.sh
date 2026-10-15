#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export RM75_DEBUG=0

.venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75-v0 \
  --exp-name RM75-v4-high-success \
  --total-timesteps 2500000 \
  --num-envs 1 \
  --learning-rate 1e-4 \
  --target-sampling uniform \
  --collision-penalty 45 \
  --initial-action-std 0.30 \
  --final-action-std 0.10 \
  --action-std-hold-fraction 0.20 \
  --eval-interval 200000 \
  --eval-episodes 100 \
  --eval-seed 20000 \
  --eval-max-collision-rate 0.10 \
  --eval-target-success 0.82 \
  --checkpoint-interval 100000 \
  --track
