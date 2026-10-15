# RM75 Pure PPO Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean pure-PPO RM75 random-position/random-direction task while restoring the legacy approach environment from 40 observations to its V9-compatible 34-observation contract.

**Architecture:** Preserve all old environment paths and model dimensions, and add `RM75Pose-v0` as an explicit mode of the existing RM75 dynamics so robot construction, seeded sampling, collision handling, and visualization remain shared. Give only the new task a compact normalized observation, keypoint pose reward, five-step success hold, larger pose network, and one reviewed launcher.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, dm_control MJCF, NumPy, PyTorch/CleanRL PPO, unittest, Bash.

**Spec:** `docs/superpowers/plans/2026-10-09-rm75-pure-ppo-pose-spec.md`

## Global Constraints

- Preserve UR5-v0 (19/6), RM75-v0 (28/7), RM75Approach-v0 (34/7), V9 weights, and their legacy 64x64 Tanh network.
- Remove the optional 40-dimensional approach observation path rather than silently mapping it to another environment.
- Keep the new controller pure PPO with normalized 7-dimensional incremental joint actions; do not add IK control or demonstrations.
- Use the fixed full random task distribution with 0-30 degree tilt and -180 to 180 degree azimuth; do not enable curriculum promotion or rollback.
- The user runs formal training; implementation verification is limited to focused tests and one very short no-save/no-track smoke run.

---

### Task 1: Lock observation and pose-task contracts in tests

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify: `tests/test_rm75_ppo_smoke.py`
- Modify: `tests/test_visualize_agent.py`

**Interfaces:**
- Consumes: `RM75P2PEnv`, `RM75Pose-v0`, `Agent`, `environment_kwargs`, visualizer inference helpers.
- Produces: regression contracts for 34-dimensional legacy observation, 20-dimensional pose observation, reward monotonicity, success hold, network topology, and model auto-detection.

- [ ] **Step 1: Remove the obsolete 40-dimensional tests**

Delete the relative-observation environment test and the 34-to-40 weight-expansion test. Change visualizer supported dimensions so `(40, 7)` is rejected.

- [ ] **Step 2: Add the compact-observation test**

Instantiate `RM75Pose-v0`, assert observation/action shapes `(20,)`/`(7,)`, finite normalized values, `horizon == 400`, and `action_scale == 0.05`.

- [ ] **Step 3: Add reward and termination tests**

Assert a smaller keypoint error yields a larger reward with identical actions, action reversal is penalized, and success terminates only after five consecutive in-tolerance steps.

- [ ] **Step 4: Add PPO/network compatibility tests**

Assert `environment_kwargs` selects the pose mode, auto network selection gives 256/128/64 ELU for 20 observations, legacy dimensions keep 64/64 Tanh, and the visualizer maps `(20, 7)` to `RM75Pose-v0`.

- [ ] **Step 5: Run tests to establish the failing baseline**

Run: `.venv/bin/python -m unittest tests.test_rm75_env tests.test_rm75_ppo_smoke tests.test_visualize_agent -v`

Expected: new pose tests fail before implementation; existing 34-dimensional V9 contract continues to pass.

### Task 2: Restore 34-dimensional legacy approach behavior

**Files:**
- Modify: `env_rm75.py`
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `Scripts/visualize_agent.py`

**Interfaces:**
- Consumes: V9 state dictionaries with input shape 34 and action shape 7.
- Produces: one fixed 34-dimensional `RM75Approach-v0` path with exact state-dict loading.

- [ ] **Step 1: Remove the relative observation constructor option**

Delete `relative_pose_observation`, the extra six observation elements, and the 40-dimensional observation-space branch from `RM75P2PEnv`.

- [ ] **Step 2: Remove training and visualization forwarding**

Delete the CLI field, checkpoint configuration field, environment kwarg, 40-dimensional model mapping, and automatic 40-dimensional selection.

- [ ] **Step 3: Restore strict model loading**

Load model state dictionaries exactly; do not synthesize zero-filled observation columns.

- [ ] **Step 4: Verify V9 dimensions and loadability**

Run the visualizer compatibility tests and a headless one-episode V9 load check.

### Task 3: Implement RM75Pose-v0 environment semantics

**Files:**
- Modify: `env_rm75.py`
- Modify: `__init__.py`
- Modify: `Algorithms/ppo_continuous_action.py`

**Interfaces:**
- Consumes: `RM75P2PEnv(..., pose_task=True)` and the existing seeded goal/direction sampler.
- Produces: `_get_pose_observation() -> np.ndarray[20]`, `_keypoint_error(...) -> float`, `_compute_pose_reward(...) -> tuple[float, dict]`, and five-step terminal success.

- [ ] **Step 1: Add explicit pose-task configuration**

Validate that pose mode implies approach mode; set the horizon to 400, action scale to 0.05 rad, success angle to the configured 20 degrees, and required hold to five steps.

- [ ] **Step 2: Build the 20-dimensional normalized observation**

Normalize joint positions by MuJoCo joint ranges, joint velocities by 2.5 rad/s, position error by 0.5 m, and rotation-vector error by pi; clip to their declared observation bounds.

- [ ] **Step 3: Implement unified keypoint error and reward**

Use a 0.15 m axis keypoint and compute `-1.5*e - 0.5*(1-exp(-100*e^2)) - 0.005*sum(a^2) - 0.005*sum((a-prev_a)^2)`, plus 10 on terminal success and minus 10 on collision.

- [ ] **Step 4: Implement held success and diagnostics**

Increment the streak only while both strict tolerances hold, terminate at five, reset the streak otherwise, and expose keypoint error/streak through `info` and reward components.

- [ ] **Step 5: Register and forward RM75Pose-v0**

Register the new ID and pass existing random direction, feasible-target, position box, and orientation threshold arguments without enabling curriculum.

- [ ] **Step 6: Run focused environment tests**

Run: `.venv/bin/python -m unittest tests.test_rm75_env -v`

Expected: legacy and pose environment tests pass.

### Task 4: Add a pose-specific PPO network without breaking old weights

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `Scripts/visualize_agent.py`
- Modify: `tests/test_rm75_ppo_smoke.py`
- Modify: `tests/test_visualize_agent.py`

**Interfaces:**
- Consumes: `network_profile` equal to `auto`, `legacy`, or `pose`.
- Produces: `resolve_network_profile(...)` and `infer_network_profile(state)`; old weights instantiate legacy Agent, new weights instantiate pose Agent.

- [ ] **Step 1: Add profile resolution and validation**

Resolve `auto` to `pose` only for 20-observation spaces and to `legacy` otherwise; reject unsupported names.

- [ ] **Step 2: Add the pose actor and critic**

Use Linear layers 256, 128, and 64 with ELU and retain the existing tanh-squashed Gaussian output/log-probability implementation.

- [ ] **Step 3: Preserve legacy actor and critic exactly**

Leave the 64/64 Tanh parameter names and shapes unchanged so V9 loads without conversion.

- [ ] **Step 4: Infer the profile in visualization**

Recognize the pose state by its `actor_mean.6.weight` output layer and instantiate the matching Agent before strict loading.

- [ ] **Step 5: Run PPO and visualizer tests**

Run: `.venv/bin/python -m unittest tests.test_rm75_ppo_smoke tests.test_visualize_agent -v`

Expected: both old and new architectures pass without partial state loading.

### Task 5: Provide one clean training command and verify it

**Files:**
- Create: `Scripts/train_rm75_pose_ppo.sh`

**Interfaces:**
- Consumes: `RM75Pose-v0` and PPO CLI.
- Produces: a from-scratch 2M-step command with eight async environments, 256 rollout steps, 2048 batch samples, 8 minibatches, 8 update epochs, learning-rate decay `3e-4 -> 3e-5`, KL limit 0.008, and action-std decay `0.30 -> 0.05`.

- [ ] **Step 1: Add the launcher**

Use the full 0-30 degree tilt, full azimuth, feasible-pose sampling, no curriculum, periodic checkpointing, deterministic evaluation, and W&B tracking.

- [ ] **Step 2: Check syntax and focused tests**

Run: `bash -n Scripts/train_rm75_pose_ppo.sh`

Run: `.venv/bin/python -m compileall env_rm75.py Algorithms/ppo_continuous_action.py Scripts/visualize_agent.py`

Run: `.venv/bin/python -m unittest tests.test_rm75_env tests.test_rm75_ppo_smoke tests.test_visualize_agent -v`

- [ ] **Step 3: Run one short untracked smoke update**

Run the new environment for one PPO batch with `--total-timesteps 16 --num-envs 1 --num-steps 16 --num-minibatches 1 --update-epochs 1 --no-track --no-save-model --eval-interval 0 --checkpoint-interval 0 --cuda False`.

Expected: the environment resets, one rollout/update completes, and no model or cloud run is created.

- [ ] **Step 4: Recheck preserved V9 artifact**

Confirm `RM75-approach-v9-strict-terminal.best.cleanrl_model` still has input/action dimensions 34/7 and loads through the unchanged legacy network.
