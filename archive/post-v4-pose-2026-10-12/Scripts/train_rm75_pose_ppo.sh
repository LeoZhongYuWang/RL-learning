#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export RM75_DEBUG=0

.venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75Pose-v0 \
  --exp-name RM75-pose-gated-v2 \
  --network-profile pose \
  --total-timesteps 2000000 \
  --num-envs 8 \
  --vector-env-mode async \
  --num-steps 256 \
  --num-minibatches 8 \
  --update-epochs 8 \
  --learning-rate 2e-4 \
  --final-learning-rate 3e-5 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-coef 0.2 \
  --target-kl 0.008 \
  --ent-coef 0.0 \
  --initial-action-std 0.15 \
  --final-action-std 0.05 \
  --action-std-hold-fraction 0.05 \
  --target-mode random \
  --target-direction-mode random_cone \
  --target-tilt-min-deg 0 \
  --target-tilt-max-deg 30 \
  --target-azimuth-min-deg -180 \
  --target-azimuth-max-deg 180 \
  --approach-position-box -0.45 -0.35 -0.18 0.18 \
  --collision-penalty 20 \
  --require-pose-feasible \
  --pose-feasibility-attempts 50 \
  --orient-success-threshold-deg 20 \
  --eval-interval 100000 \
  --eval-episodes 100 \
  --eval-seed 20000 \
  --eval-max-collision-rate 0.01 \
  --eval-target-success 0.90 \
  --checkpoint-interval 100000 \
  --track
