#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="runs/RM75Direction-v0__RM75-direction-v6-finetune__1__1791869875/RM75-direction-v6-finetune.best.cleanrl_model"
PREFIX_MODEL="runs/RM75Direction-v0__RM75-direction-v5-near-coordination__1__1791859371/RM75-direction-v5-near-coordination.best.cleanrl_model"
# New run from retained best (97% selection eval, 93% unseen-seed eval).
# Keep V5 reset preparation and all numeric task/training settings.
# Full-start evaluation uses only the learning policy, never the prefix model.
# --init-from starts a fresh optimizer, step counter and W&B history.
if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Missing validated direction initialization model: $BASE_MODEL" >&2
  exit 1
fi
if [[ ! -f "$PREFIX_MODEL" ]]; then
  echo "Missing frozen arrival-start model: $PREFIX_MODEL" >&2
  exit 1
fi

.venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75Direction-v0 \
  --exp-name RM75-direction-v6-clean-finetune \
  --init-from "$BASE_MODEL" \
  --near-start-from "$PREFIX_MODEL" \
  --total-timesteps 500000 \
  --num-envs 1 \
  --learning-rate 1e-5 \
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
