# RM75 Task 4 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the standalone seven-axis RM75 Gymnasium environment, make reset randomness reproducible, and terminate episodes on table or robot self-collision.

**Architecture:** Compile the fixed MJCF structure once in `__init__`, validate the compiled joint/actuator contract immediately, and change per-episode numeric state directly in `reset`. Keep the PPO-facing action normalized to `[-1, 1]` and map it to bounded joint-reference increments inside the environment. Classify contacts using cached table and robot geom IDs.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, dm_control MJCF, NumPy, unittest.

**Spec:** `docs/superpowers/plans/2026-09-29-rm75-mujoco-migration.md` task 4 plus the user requirements for seed-controlled table friction and RM75 self-collision.

## Global Constraints

- Do not rebuild the fixed MuJoCo model during every reset.
- Joint order is exactly `joint_1` through `joint_7`.
- Actuator order is exactly `joint_1_pos` through `joint_7_pos` and each actuator drives the same-index joint.
- The first environment version supports MuJoCo position actuators only.
- The current PPO receives and emits normalized actions in `[-1, 1]`.
- A reset seed must reproduce both the initial joint noise and table friction.
- Self-collision means contact between two RM75 robot geoms; table collision remains a terminal collision.

---

### Task 1: Lock down the task-4 environment contract

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: `RM75P2PEnv(render_mode=None, seed=0, target_mode="random", target_pos=None)`.
- Produces: normalized action space `(7,)`, observation space `(21,)`, finite reset and one-step results, validated MuJoCo joint/actuator mapping.

- [x] **Step 1: Add failing unittest coverage for reset, normalized actions, compiled name order, one zero-action step, and the default random goal.**
- [x] **Step 2: Run `python -m unittest tests.test_rm75_env -v` and confirm the missing default-goal, seeded-friction, and self-collision contracts fail.**
- [x] **Step 3: Verify the existing `action_scale`, runtime model validation, 50 Hz metadata, and command-line class; add position-only validation and the correct default target mode.**
- [x] **Step 4: Run the focused unittest module and confirm the core contract passes.**

### Task 2: Make table friction reproducible without recompilation

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: `reset(seed: int | None)` and the compiled table geom IDs.
- Produces: the same `model.geom_friction` values for repeated resets with the same seed and potentially different values for different seeds.

- [x] **Step 1: Add a test that compares table friction after seed 123, repeats seed 123, then uses seed 124.**
- [x] **Step 2: Replace compile-time global NumPy randomness with fixed placeholder friction and update `model.geom_friction` from `self.rng` during reset.**
- [x] **Step 3: Run the focused tests and confirm no MJCF recompilation is needed.**

### Task 3: Add RM75 self-collision classification

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify: `env_rm75.py`

**Interfaces:**
- Consumes: compiled geom/body/mesh IDs and MuJoCo contacts.
- Produces: `_check_collision() -> bool`, true for table contact or contact between two RM75 robot geoms.

- [x] **Step 1: Add tests that verify the robot geom set contains the base plus seven links, excludes floor/tables, and classifies robot-robot pairs as collisions.**
- [x] **Step 2: Cache exact table and robot geom IDs after compilation.**
- [x] **Step 3: Extend `_check_collision()` to classify both table and self-collision contacts.**
- [x] **Step 4: Confirm the home/reset pose is not falsely classified as collision.**

### Task 4: Run the complete acceptance pass

**Files:**
- Modify: `docs/superpowers/plans/2026-09-29-rm75-mujoco-migration.md`
- Test: `tests/test_rm75_env.py`
- Test: `tests/test_rm75_model.py`

**Interfaces:**
- Consumes: the completed `RM75P2PEnv`.
- Produces: task-4 implementation record and repeatable verification commands.

- [x] **Step 1: Run Python compilation checks.**
- [x] **Step 2: Run RM75 model and environment unittests.**
- [x] **Step 3: Run a multi-seed reset/step smoke test and a one-second zero-action stability test.**
- [x] **Step 4: Run `git diff --check`.**
- [x] **Step 5: Update the original migration task book with the accepted implementation differences and verification results.**
