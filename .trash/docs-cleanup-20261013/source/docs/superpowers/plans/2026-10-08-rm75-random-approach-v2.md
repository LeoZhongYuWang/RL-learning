# RM75 Random Approach V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed fixed-direction RM75 approach experiment with a seed-controlled random-cone task, joint position-orientation shaping, feasible goal sampling, useful diagnostics, and reproducible training commands while preserving legacy UR5/RM75 tasks.

**Architecture:** Keep the existing 34-observation/7-action `RM75Approach-v0` contract so the visualizer can still infer the environment. Extend only the approach-task configuration and reset path: sample a target direction per episode, reject pose-infeasible position/direction pairs, rotate the visual marker dynamically, and shape reward using the worst normalized position-orientation error. The point-only `RM75-v0` path remains unchanged.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, dm_control MJCF, NumPy, PyTorch/CleanRL PPO, unittest.

**Spec:** User-approved design in the current task: random azimuth and tilt, joint pose reward, staged angle curriculum, preserve prior random-point execution, and delete only the failed previous approach run.

## Global Constraints

- Do not change the 19/6 UR5-v0, 28/7 RM75-v0, or 34/7 RM75Approach-v0 model dimensions.
- Do not delete or modify UR5-v3, RM75-v3, or RM75-v4-high-success results.
- Random direction and feasibility sampling must be controlled by `reset(seed=...)`.
- `RM75-v0` point-reaching reward and full target box must remain unchanged.
- A success requires distance below 0.05 m and direction error below the configured threshold.
- Delete only `RM75Approach-v0__RM75-approach-v1__1__1791439993` and W&B run `hyxntvju`.

---

### Task 1: Define random-cone and joint-pose behavior with tests

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `RM75P2PEnv`, `Args`, and `environment_kwargs`.
- Produces: executable contracts for seeded direction sampling, dynamic marker rotation, pose-progress reward, joint milestones, diagnostics, and CLI forwarding.

- [ ] **Step 1: Add failing environment tests**

Add tests that instantiate `approach_task=True` with `target_direction_mode="random_cone"`, verify identical seeds yield identical unit directions inside the requested cone, verify different seeds vary, and verify marker +Z equals the sampled direction.

- [ ] **Step 2: Add failing reward and diagnostics tests**

Test that reducing only distance while angle is the worse normalized error does not earn pose progress, reducing the worse angle does earn it, milestones require both thresholds, and per-episode minimum/ever-success fields appear in `info`.

- [ ] **Step 3: Add failing PPO argument forwarding tests**

Verify `environment_kwargs` forwards direction mode, cone bounds, approach box, and pose-feasibility settings only to `RM75Approach-v0`.

- [ ] **Step 4: Run the focused tests and confirm failure**

Run: `python -m unittest tests.test_rm75_env tests.test_rm75_ppo_smoke -v`

Expected: new tests fail because the new constructor arguments and pose reward do not exist.

### Task 2: Implement random-cone goals and dynamic visualization

**Files:**
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: constructor fields `target_direction_mode`, tilt/azimuth degree bounds, `approach_position_box`, `require_pose_feasible`, and `pose_feasibility_attempts`.
- Produces: `_sample_target_direction() -> tuple[np.ndarray, float, float]`, `_sample_approach_goal()`, and reset `info` fields `target_tilt_deg`/`target_azimuth_deg`.

- [ ] **Step 1: Validate approach-task configuration**

Accept `fixed` and `random_cone`; validate finite ordered angle ranges; keep the fixed vector behavior as the default for backward compatibility.

- [ ] **Step 2: Sample seeded cone directions**

Use `direction = [sin(theta) cos(phi), sin(theta) sin(phi), -cos(theta)]` with all draws from `self.rng`.

- [ ] **Step 3: Update the marker on every reset**

Write both `model.site_pos` and `model.site_quat` so the yellow cylinder follows the newly sampled direction without recompiling the model.

- [ ] **Step 4: Add pose-feasible rejection sampling**

Use a scratch `MjData` and damped least-squares position-plus-axis IK. Reject candidates that miss the success tolerances or collide; raise a clear error after the configured number of attempts instead of silently training impossible goals.

- [ ] **Step 5: Run the focused environment tests**

Run: `python -m unittest tests.test_rm75_env -v`

Expected: all RM75 environment tests pass.

### Task 3: Replace independent shaping with joint pose shaping

**Files:**
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: current/previous distance and angle error.
- Produces: `_normalized_pose_error`, potential progress based on `max(distance_ratio, angle_ratio)`, one-time joint milestones, and episode diagnostics.

- [ ] **Step 1: Add combined pose potential**

For the approach task compute `pose_error = max(dist / 0.05, angle / angle_threshold)` and reward only reductions in that error. Keep the legacy point-only progress path unchanged.

- [ ] **Step 2: Add joint milestones**

Award once per episode at `(0.20 m, 45 deg)` and `(0.10 m, 30 deg)`; never award a distance-only milestone in the approach task.

- [ ] **Step 3: Add diagnostics**

Track final/minimum distance, final/minimum angle, minimum joint pose error, whether position and direction were ever individually achieved, and whether simultaneous success occurred.

- [ ] **Step 4: Run reward tests**

Run: `python -m unittest tests.test_rm75_env -v`

Expected: all reward and diagnostics tests pass.

### Task 4: Expose training controls and metrics

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `Scripts/visualize_agent.py`
- Modify: `tests/test_rm75_ppo_smoke.py`
- Modify: `tests/test_visualize_agent.py`

**Interfaces:**
- Consumes: new dataclass CLI fields and episode `info` diagnostics.
- Produces: CLI flags, checkpoint-resumable configuration, TensorBoard/W&B metrics, and matching visualization options.

- [ ] **Step 1: Add CLI dataclass fields and validation**

Add direction mode, cone bounds, approach box, feasibility toggle, and maximum sampling attempts. Include them in checkpoint resume configuration.

- [ ] **Step 2: Forward the configuration**

Pass the new fields only for `RM75Approach-v0`; retain all existing environment calls.

- [ ] **Step 3: Log diagnostic metrics**

Add per-episode charts for minimum distance/angle/pose error and ever-position/ever-direction outcomes; add corresponding deterministic evaluation averages.

- [ ] **Step 4: Add visualization flags**

Allow the same fixed or random-cone task configuration during model demonstration so evaluation matches training.

- [ ] **Step 5: Run PPO and visualizer tests**

Run: `python -m unittest tests.test_rm75_ppo_smoke tests.test_visualize_agent -v`

Expected: all tests pass and legacy model inference remains unchanged.

### Task 5: Verify the full project and remove the failed run

**Files:**
- Delete: `runs/RM75Approach-v0__RM75-approach-v1__1__1791439993/`
- Delete: `wandb/run-20261008_141315-hyxntvju/`

**Interfaces:**
- Consumes: completed code and exact failed-run identifiers.
- Produces: clean test results, a short headless smoke run, and no stale local failed-run artifacts.

- [ ] **Step 1: Run syntax and full unit tests**

Run: `python -m compileall env_rm75.py Algorithms/ppo_continuous_action.py Scripts/visualize_agent.py`

Run: `python -m unittest discover -s tests -v`

Expected: compilation and all tests pass.

- [ ] **Step 2: Run a short untracked training smoke test**

Run a small `RM75Approach-v0` random-cone job with saving/tracking disabled and verify PPO completes at least one update.

- [ ] **Step 3: Resolve exact deletion targets**

Confirm the run directory contains only the failed `RM75-approach-v1` outputs and the local W&B directory has run id `hyxntvju` before removal.

- [ ] **Step 4: Delete local and remote failed-run data**

Remove the two exact local directories. If authenticated W&B API access is available, delete remote run `rm75/hyxntvju`; do not touch other runs.

- [ ] **Step 5: Verify preserved results**

Confirm UR5-v3, RM75-v3, and RM75-v4-high-success model files still exist.

