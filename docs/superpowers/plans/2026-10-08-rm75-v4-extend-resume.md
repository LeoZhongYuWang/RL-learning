# RM75 V4 Extend-Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the completed V4 run for another two million environment steps without losing PPO optimizer, curriculum stage, exploration, checkpoint history, or W&B continuity.

**Architecture:** Add a runtime-only `--additional-timesteps` option that extends the checkpoint's completed PPO iteration count instead of being overwritten by saved arguments. Preserve exact resume behavior after an interruption and rank approach-task best models by joint pose quality when strict success is tied. Per the user's review, keep the learning rate fixed at `5e-5` in every curriculum stage.

**Tech Stack:** Python 3.10, PyTorch PPO, Gymnasium, TensorBoard, Weights & Biases, Bash.

**Spec:** `docs/superpowers/plans/2026-10-08-rm75-v4-extend-resume.md`

## Global Constraints

- Do not start formal training.
- Resume from the retained V4 `checkpoint_latest.pt` in its existing run directory.
- Keep stage 3, optimizer state, global step, recent curriculum outcomes, and W&B run ID.
- Add 2,000,000 requested steps as complete 2,048-step PPO batches.
- Do not alter reward weights, success thresholds, target sampling, or curriculum stage definitions.

---

### Task 1: Extend a completed checkpoint safely

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Produces: `configure_training_horizon(args: Args, resume_payload: dict | None) -> Args`
- Produces: runtime option `Args.additional_timesteps: int`

- [x] Add tests proving a completed 976-iteration checkpoint gains 976 new iterations for 2,000,000 requested steps.
- [x] Add tests rejecting additional steps without `--resume-from`, negative values, and a request smaller than one PPO batch.
- [x] Implement batch-aligned extension after saved arguments are restored.
- [x] Keep `additional_timesteps` out of `RESUME_CONFIG_FIELDS` so the command-line request cannot be overwritten.
- [x] Print the old global step, effective added steps, and new final target before training begins.

### Task 2: Improve best-model selection for approach control

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Modifies: `should_update_best(candidate, best, max_collision_rate) -> bool`

- [x] Add a test where strict success and collision are tied but lower `mean_min_pose_error` must beat lower final distance.
- [x] Rank approach policies by strict success, collision, minimum joint pose error, position-entry rate, final distance, then direction error.
- [x] Preserve point-task ranking when pose metrics are absent.

### Task 3: Preserve the reviewed fixed learning rate

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: checkpoint values `learning_rate=5e-5` and `anneal_lr=False`.

- [x] Verify the retained checkpoint restores `learning_rate=5e-5` and `anneal_lr=False`.
- [x] Leave learning-rate assignment stage-independent so stages 1-5 all use `5e-5`.
- [x] Document that later adjustment is evidence-driven rather than automatic.

### Task 4: Provide and validate the continuation launcher

**Files:**
- Create: `Scripts/continue_rm75_approach_v4.sh`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: retained V4 `checkpoints/checkpoint_latest.pt`.

- [x] Add a launcher containing only `--resume-from` and `--additional-timesteps 2000000` so saved training configuration remains authoritative.
- [x] Validate shell syntax, Python syntax, CLI help, checkpoint compatibility, and all RM75 tests.
- [x] Confirm the calculated target is 1,952 total PPO iterations and no training process was launched.
