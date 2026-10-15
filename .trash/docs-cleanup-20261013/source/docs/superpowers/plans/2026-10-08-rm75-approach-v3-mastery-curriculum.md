# RM75 Approach V3 Mastery Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the V2 pose-error plateau and train RM75 approach control through success-gated position, orientation, and workspace stages.

**Architecture:** Preserve the 34-observation/7-action policy contract and strict final task `(0.05 m, 15 deg)`. Replace the hard maximum pose potential with a stable smooth maximum plus a small sum term, then let the PPO training loop apply stage-specific target boxes, tilt limits, and active success thresholds. Promotion is based on the recent curriculum success rate rather than elapsed steps; deterministic evaluation remains strict and uses the full final distribution.

**Tech Stack:** Python 3.10, NumPy, MuJoCo, Gymnasium, PyTorch, CleanRL PPO, TensorBoard, unittest.

**Spec:** Current-run diagnosis at 591,872 steps: zero success, zero ever-position success, mean minimum distance near 0.15 m, mean minimum angle near 11 deg, and normalized pose error near 3.14 caused by the hard `max()` reward plateau.

## Global Constraints

- Keep UR5-v0 at 19 observations/6 actions, RM75-v0 at 28/7, and RM75Approach-v0 at 34/7.
- Keep final evaluation success fixed at distance below 0.05 m and angle below 15 deg.
- Do not overwrite or delete the V2 checkpoint; it is the V3 warm-start source.
- Stage promotion requires a complete rolling window and success rate at or above the configured threshold.
- Evaluation environments never receive loose curriculum thresholds.
- Pose-feasibility IK always verifies the strict final thresholds, even during loose training stages.

---

### Task 1: Specify smooth pose error and active-versus-strict success

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: distance, angle error, active curriculum thresholds, and strict reward thresholds.
- Produces: `_normalized_pose_error`, `_is_success`, `_is_strict_success`, and `set_approach_curriculum`.

- [ ] **Step 1: Add failing reward tests**

Verify that improving the non-dominant distance or angle still reduces the smooth pose error, unlike the previous hard maximum.

- [ ] **Step 2: Add failing curriculum-threshold tests**

Set stage 1 to `(0.20 m, 45 deg)` and verify a pose at `(0.15 m, 30 deg)` is a curriculum success but not a strict success.

- [ ] **Step 3: Implement the smooth pose potential**

Use a centered smooth maximum with `beta=2`, blended as `0.8 * smooth_max + 0.1 * (distance_ratio + angle_ratio)` so improving either constraint changes the potential while the worse constraint remains dominant.

- [ ] **Step 4: Implement active curriculum state**

Add environment fields for curriculum stage, active distance threshold, active direction threshold, target box, and tilt limit; keep strict thresholds in `reward_cfg`.

- [ ] **Step 5: Run focused environment tests**

Run: `.venv/bin/python -m unittest tests.test_rm75_env -v`

Expected: all environment tests pass and legacy point-task reward keys remain unchanged.

### Task 2: Replace step-based progression with mastery progression

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: completed-episode curriculum success values.
- Produces: `curriculum_stage_settings`, `should_promote_curriculum`, and `apply_approach_curriculum`.

- [ ] **Step 1: Define three stage settings**

Stage 1 uses tilt `0-15 deg`, success `(0.20 m, 45 deg)`, and box `[-0.42,-0.36,-0.10,0.10]`; stage 2 uses `0-30 deg`, `(0.10 m,30 deg)`, and `[-0.45,-0.35,-0.14,0.14]`; stage 3 uses `0-45 deg`, `(0.05 m,15 deg)`, and `[-0.45,-0.35,-0.18,0.18]`.

- [ ] **Step 2: Add promotion tests**

Verify no promotion before 100 episodes, promotion at a recent-100 success rate of at least 0.70, and no promotion beyond stage 3.

- [ ] **Step 3: Apply stages to training environments only**

Call `set_approach_curriculum` on vectorized training environments. Continue constructing evaluation environments directly from strict final arguments.

- [ ] **Step 4: Promote from completed episodes**

Track a rolling deque, promote only after the threshold is met, clear the stage window after promotion, and log the transition.

- [ ] **Step 5: Run focused PPO tests**

Run: `.venv/bin/python -m unittest tests.test_rm75_ppo_smoke -v`

Expected: all PPO, checkpoint, and evaluation tests pass.

### Task 3: Preserve curriculum state and improve metrics

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: current stage and recent stage-success window.
- Produces: resumable `curriculum_state`, `strict_success` episode info, and curriculum dashboard scalars.

- [ ] **Step 1: Save optional curriculum state in checkpoints**

Store stage and recent success values without breaking older format-version-1 checkpoints.

- [ ] **Step 2: Restore curriculum state on resume**

Default old checkpoints to stage 1 with an empty window; restore new checkpoints exactly.

- [ ] **Step 3: Log strict and curriculum success separately**

Keep `charts/success_rate` as the active-stage learning signal, add `charts/strict_success_rate`, and log active distance, angle, box, and stage.

- [ ] **Step 4: Run checkpoint tests**

Run: `.venv/bin/python -m unittest tests.test_rm75_ppo_smoke -v`

Expected: legacy weight loading and full checkpoint round trips still pass.

### Task 4: Add the V3 warm-start command and verify end-to-end

**Files:**
- Create: `Scripts/train_rm75_approach_v3.sh`
- Modify: `tests/test_rm75_env.py`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: V2 `checkpoint_latest.pt` through `--init-from`.
- Produces: a new V3 run with a reset optimizer, reset learning-rate schedule, mastery curriculum, checkpoints, and W&B tracking.

- [ ] **Step 1: Create the launch script**

Use learning rate `1e-4`, action standard deviation `0.20 -> 0.10`, 2,000,000 steps, recent-100 promotion threshold `0.70`, strict fixed-seed evaluation every 200,000 steps, and the preserved V2 checkpoint as `--init-from`.

- [ ] **Step 2: Validate script syntax and CLI parsing**

Run: `bash -n Scripts/train_rm75_approach_v3.sh` and a 256-step untracked smoke run with the same environment options.

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 4: Verify artifact preservation**

Confirm the V2 checkpoint, UR5-v3, RM75-v3, and RM75-v4-high-success model files still exist.
