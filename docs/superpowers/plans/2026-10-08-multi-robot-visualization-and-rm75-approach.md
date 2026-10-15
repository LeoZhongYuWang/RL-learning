# Multi-Robot Visualization and RM75 Approach-Direction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore direct visualization of legacy UR5 and RM75 policies, then add a separately registered RM75 task that learns random target positions with a configurable tool approach direction.

**Architecture:** Keep `UR5-v0` and `RM75-v0` behavior and observation sizes unchanged. Detect legacy policy type from checkpoint tensor shapes in the visualization script, while registering `RM75Approach-v0` as an opt-in 34-dimensional mode of `RM75P2PEnv`; the approach mode adds current/target direction observations, gated potential-based orientation reward, joint position-and-direction success, and a MuJoCo direction marker.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo/dm_control MJCF, NumPy, PyTorch, CleanRL PPO, unittest.

**Spec:** User-approved design in the current task: preserve all existing random-point commands and models; use position plus approach direction; ignore start orientation; train the new policy from scratch; report all modifications.

## Global Constraints

- Existing `UR5-v0` stays at 19 observations and 6 actions.
- Existing `RM75-v0` stays at 28 observations and 7 actions.
- Existing UR5-v3, RM75-v3, and RM75-v4 model files are not edited or deleted.
- New `RM75Approach-v0` uses 34 observations and 7 actions.
- Default visualization auto-detects the correct environment from model tensor dimensions.
- Existing random-point visualization commands remain valid without a new required flag.
- Approach success requires both position and direction thresholds.
- The new policy is trained from scratch; compatibility code must not silently load an incompatible model.

---

### Task 1: Model-aware visualization

**Files:**
- Modify: `Scripts/visualize_agent.py`
- Create: `tests/test_visualize_agent.py`

**Interfaces:**
- Consumes: CleanRL state dictionaries or full checkpoints containing `agent_state_dict`.
- Produces: `load_model_state(path)`, `infer_environment_id(state)`, `create_environment(env_id, args, render_mode)`, and `--env-id {auto,UR5-v0,RM75-v0,RM75Approach-v0}`.

- [ ] Write tests asserting `(19, 6) -> UR5-v0`, `(28, 7) -> RM75-v0`, `(34, 7) -> RM75Approach-v0`, plus explicit mismatch errors.
- [ ] Run `python -m unittest tests.test_visualize_agent -v` and verify the new tests fail.
- [ ] Load checkpoint state before environment construction, infer dimensions from `critic.0.weight` and `actor_logstd`, and validate explicit environment selection.
- [ ] Expand newest-model discovery to all `*.cleanrl_model` files and print the resolved environment and model dimensions.
- [ ] Run the visualization unit tests and headless one-episode checks with the actual UR5-v3 and RM75-v4 models.

### Task 2: RM75 approach-direction environment

**Files:**
- Modify: `env_rm75.py`
- Modify: `__init__.py`
- Modify: `tests/test_rm75_env.py`

**Interfaces:**
- Consumes: `approach_task: bool`, `target_direction: tuple[float, float, float]`, and `orient_success_threshold_deg: float`.
- Produces: `RM75Approach-v0`, 34-dimensional observations, `angle_error`, `angle_error_deg`, `direction_success`, orientation reward components, and a movable target direction site.

- [ ] Add failing tests for registration, legacy 28-dimensional compatibility, 34-dimensional approach observations, direction normalization, invalid zero vectors, angle calculation, joint success, and line pose updates.
- [ ] Run the focused RM75 environment tests and confirm failures precede implementation.
- [ ] Add direction-vector helpers and validate the desired vector and threshold in `RM75P2PEnv.__init__`.
- [ ] Add a thin target-direction site to the compiled scene and update its position/quaternion without recompiling on reset.
- [ ] Add current and target direction vectors only in approach mode.
- [ ] Compute angle error and distance-gated potential-based direction reward; require both thresholds only in approach mode.
- [ ] Register `RM75Approach-v0` with `approach_task=True` while leaving the two existing registrations unchanged.
- [ ] Run all RM75 environment tests.

### Task 3: PPO training and evaluation integration

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `--env-id RM75Approach-v0`, `--target-direction DX DY DZ`, and `--orient-success-threshold-deg DEG`.
- Produces: correctly constructed vector environments, deterministic evaluation angle metrics, episodic W&B direction metrics, and normal CleanRL/checkpoint outputs.

- [ ] Add failing tests for approach environment kwargs, vector observation/action dimensions, and evaluation angle metrics.
- [ ] Pass approach-only arguments only to `RM75Approach-v0`.
- [ ] Record `final_angle_error_deg`, `direction_success`, and recent direction-success rate when those values are present.
- [ ] Include mean final angle error and direction success rate in deterministic evaluation and console output.
- [ ] Run PPO smoke tests and a minimal approach PPO subprocess smoke run.

### Task 4: Regression verification and user handoff

**Files:**
- Modify: `Scripts/visualize_agent.py` help text and examples as needed.
- Modify: this plan by checking completed steps only if useful; do not alter model artifacts.

**Interfaces:**
- Consumes: existing model files under `runs/`.
- Produces: verified commands for UR5-v3, RM75-v4, and future RM75 approach models.

- [ ] Run the complete test suite.
- [ ] Run `py_compile` for every modified Python file.
- [ ] Run actual UR5-v3 and RM75-v4 deterministic headless evaluations for one episode each.
- [ ] Verify `git diff` contains no model deletion, no change to old observation dimensions, and no unrelated user-file overwrite.
- [ ] Report every modified file, behavior change, test result, and the exact old/new usage commands.
