# RM75 Stage 2 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Escape the Stage-2 51-56% plateau without weakening its 70% mastery requirement, then preserve a gradual path to the 30 degree / 5 cm / 20 degree final task.

**Architecture:** Keep the PPO network, observation/action interfaces, terminal rewards, and final task unchanged. Fine-tune the latest recoverable policy weights in a new run while reducing the Stage-1-to-2 distribution jump and modestly strengthening persistent position shaping.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, PyTorch PPO, unittest, Bash, Weights & Biases.

**Spec:** User-approved Stage-2 recovery proposal from the 2026-10-09 training review.

## Global Constraints

- Do not resume the old optimizer or old curriculum configuration.
- Initialize weights from the retained `checkpoint_latest.pt` at global step 1,200,128.
- Keep Stage-2 promotion at 70% in both the rolling training window and fixed-seed evaluation.
- Keep the final task at 30 degree maximum tilt, 5 cm distance, and 20 degree direction tolerance.
- Do not start formal training.

---

### Task 1: Specify the recovered curriculum

**Files:**
- Modify: `tests/test_rm75_ppo_smoke.py`
- Modify: `Algorithms/ppo_continuous_action.py`

**Interfaces:**
- Consumes: `curriculum_stage_settings(args, stage)`.
- Produces: stages with tilt limits `15, 15, 20, 25, 30, 30` and progressively expanding position boxes.

- [x] Change the regression expectations so Stage 2 tightens accuracy without expanding tilt or the goal box.
- [x] Run the focused PPO tests and confirm the old schedule fails.
- [x] Update the `Args` defaults to match the reviewed schedule.
- [x] Re-run the focused PPO tests and confirm they pass.

### Task 2: Strengthen persistent position shaping

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: `_normalized_pose_error(dist, angle_error)` and `reward_cfg`.
- Produces: `0.5 * smooth_max + 0.4 * distance_error + 0.1 * direction_error`, with `pose_error_weight=0.03`.

- [x] Add exact reward-configuration and position-priority assertions.
- [x] Run the environment tests and confirm the previous constants fail.
- [x] Update only the approach pose shaping constants; retain success bonus, collision penalty, time penalty, and timeout behavior.
- [x] Re-run the environment tests and confirm they pass.

### Task 3: Build the new Stage-2 fine-tuning launcher

**Files:**
- Modify: `Scripts/train_rm75_approach_final.sh`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `runs/RM75Approach-v0__RM75-approach-v6-4m-position-balanced__1__1791513128/checkpoints/checkpoint_latest.pt` through `--init-from`.
- Produces: a new three-million-step run named `RM75-approach-v7-stage2-recovery` starting at Stage 2.

- [x] Assert the launcher uses `--init-from`, Stage 2, a 0.35 rollback threshold, and the new schedule.
- [x] Configure learning rate `1e-5 -> 2e-6`, action standard deviation `0.08 -> 0.06`, and Stage-5 promotion at 0.60.
- [x] Run `bash -n`, Python compilation, the complete unit-test suite, and a short CUDA smoke run without W&B or model saving.
- [x] Move smoke-test logs out of the formal `runs/` directory.
