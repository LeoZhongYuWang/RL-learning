#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export RM75_DEBUG=0

.venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75-v0 \
  --exp-name RM75-point-3cm-limit-v2 \
  --total-timesteps 1500000\
  --num-envs 1 \
  --learning-rate 1e-5 \
  --collision-penalty 45 \
  --initial-action-std 0.15 \
  --final-action-std 0.10 \
  --action-std-hold-fraction 0.20 \
  --eval-interval 50000 \
  --eval-episodes 100 \
  --eval-seed 20000 \
  --eval-max-collision-rate 0.10 \
  --eval-target-success 0.90 \
  --dist-success-thresh 0.03 \
  --checkpoint-interval 100000 \
  --init-from "runs/RM75-v0__RM75-point-3cm-limit-v2__1__1792462716/RM75-point-3cm-limit-v2.best.cleanrl_model" \
  --track
