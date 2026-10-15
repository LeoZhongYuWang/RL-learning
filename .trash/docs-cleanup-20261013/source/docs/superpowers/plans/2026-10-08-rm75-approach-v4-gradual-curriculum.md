# RM75 Approach V4 Gradual Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the abrupt three-stage RM75 approach curriculum with a recoverable five-stage curriculum and launch a fresh fine-tuning run from the strongest V3 policy weights.

**Architecture:** Keep the final evaluation contract fixed at 5 cm and 15 degrees. Training uses explicit stage specifications, a minimum residence time, a recent-episode mastery gate, and guarded one-stage rollback. Checkpoints persist all curriculum counters so exact continuation remains possible, while the V4 launcher uses `--init-from` to reset optimizer and curriculum state.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, PyTorch PPO, TensorBoard, Bash.

**Spec:** `docs/superpowers/plans/2026-10-08-rm75-approach-v4-gradual-curriculum.md`

## Global Constraints

- Do not start the user's formal training run.
- Preserve all existing UR5, RM75 point-task, V2, and V3 model files.
- Deterministic evaluation always uses the final 45 degree, 5 cm, 15 degree task.
- `--resume-from` must restore curriculum stage and counters exactly.
- V4 must initialize weights from the V3 best model and create a separate run.

---

### Task 1: Define and validate five curriculum stages

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Produces: `curriculum_stage_settings(args: Args, stage: int) -> dict`
- Produces: `should_promote_curriculum(successes, rate, window, stage_steps, minimum_steps) -> bool`
- Produces: `should_demote_curriculum(successes, rate, window, stage_steps, grace_steps) -> bool`

- [x] Add tests for all five stage settings, minimum residence time, promotion, and rollback gating.
- [x] Run the focused PPO tests and confirm the new assertions fail against V3 behavior.
- [x] Add five stage definitions and validation arguments.
- [x] Run the focused PPO tests and confirm they pass.

### Task 2: Implement the recoverable curriculum state machine

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: stage settings and promotion/demotion functions from Task 1.
- Produces checkpoint state `{stage, stage_start_step, recent_successes, rollback_armed}`.

- [x] Test that stage counters are serializable and accepted from older checkpoints without the new keys.
- [x] Track environment steps spent in the current stage.
- [x] Promote only after the minimum step count and mastery window both pass.
- [x] Arm one rollback after promotion and demote when the protected stage collapses below the configured rate.
- [x] Clear stage-local outcome windows on every transition.
- [x] Save and restore the complete state through `checkpoint_latest.pt`.
- [x] Log active stage, stage steps, promotion rate, and strict success rate.

### Task 3: Preserve learning capacity across later stages

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Produces: stage-local action standard-deviation scheduling through `compute_stage_action_std(...)`.

- [x] Add tests showing exploration restarts at the configured initial value after a stage transition.
- [x] Compute action standard deviation from stage-local progress instead of global training progress when curriculum is enabled.
- [x] Keep a fixed learning rate for the V4 launcher so late curriculum stages do not inherit an almost-zero global annealed rate.
- [x] Verify legacy UR5 and non-curriculum behavior remains unchanged.

### Task 4: Add V4 launcher and verify compatibility

**Files:**
- Create: `Scripts/train_rm75_approach_v4.sh`
- Test: `tests/test_rm75_env.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `RM75-approach-v3-mastery-curriculum.best.cleanrl_model` through `--init-from`.

- [x] Add the V4 command with 2,000,000 steps, fixed `5e-5` learning rate, five-stage curriculum, periodic checkpoints, and strict evaluation.
- [x] Validate the script syntax and CLI option parsing.
- [x] Load the V3 best weights into the current 34-observation, 7-action Agent.
- [x] Run Python compilation and the complete RM75 environment/PPO test suites.
- [x] Confirm no formal training process was started and report the copyable launch command.
