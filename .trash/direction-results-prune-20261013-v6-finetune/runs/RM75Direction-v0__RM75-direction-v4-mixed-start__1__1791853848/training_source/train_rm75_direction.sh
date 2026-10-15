#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="runs/RM75Direction-v0__RM75-direction-v3-near-goal__1__1791811899/RM75-direction-v3-near-goal.best.cleanrl_model"
# Same reward, mixed training starts; retain the partial near-goal capability.
# --init-from starts a fresh optimizer, step counter and W&B history.
if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Missing validated direction initialization model: $BASE_MODEL" >&2
  exit 1
fi

.venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75Direction-v0 \
  --exp-name RM75-direction-v4-mixed-start \
  --init-from "$BASE_MODEL" \
  --total-timesteps 2000000 \
  --num-envs 1 \
  --learning-rate 5e-5 \
  --target-sampling uniform \
  --collision-penalty 45 \
  --near-start-probability 0.5 \
  --initial-action-std 0.15 \
  --final-action-std 0.06 \
  --action-std-hold-fraction 0.20 \
  --eval-interval 100000 \
  --eval-episodes 100 \
  --eval-seed 20000 \
  --eval-max-collision-rate 0.10 \
  --eval-target-success 0.75 \
  --checkpoint-interval 100000 \
  --track
