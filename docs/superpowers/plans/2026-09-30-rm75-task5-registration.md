# RM75 Task 5 Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register `RM75-v0` through Gymnasium while preserving the existing `UR5-v0` environment.

**Architecture:** Keep registration in the repository root `__init__.py`, import both concrete environment classes, and guard each registration with a registry membership check. Verify both environments through the public `gym.make()` interface and confirm repeated module imports are idempotent.

**Tech Stack:** Python 3.10, Gymnasium, NumPy, unittest, MuJoCo.

**Spec:** `docs/superpowers/plans/2026-09-29-rm75-mujoco-migration.md` task 5.

## Global Constraints

- Keep `UR5-v0` registered and behavior-compatible.
- Register `RM75-v0` with `RM75P2PEnv` and default `target_mode="random"`.
- Do not add PPO integration; that is task 6.
- Registration must be safe when the module is imported more than once.
- Tests use standard-library `unittest`, not unavailable `pytest`.

---

### Task 1: Add failing public-registration coverage

**Files:**
- Modify: `tests/test_rm75_env.py`

**Interfaces:**
- Consumes: repository root registration module and Gymnasium registry.
- Produces: a test that constructs both public environment IDs and runs reset/step.

- [x] **Step 1: Add a unittest that imports the registration module, checks both IDs, reloads it, and calls `gym.make()` for each environment.**
- [x] **Step 2: Run the focused test and confirm `RM75-v0` is absent before implementation.**

### Task 2: Register both environments idempotently

**Files:**
- Modify: `__init__.py`

**Interfaces:**
- Consumes: `UR5P2PEnv`, `RM75P2PEnv`.
- Produces: registry entries `UR5-v0` and `RM75-v0`.

- [x] **Step 1: Import both environment classes and guard both `gym.register()` calls with `if ENV_ID not in gym.registry`.**
- [x] **Step 2: Run the focused registration test.**
- [x] **Step 3: Run all project unittests and Python compilation checks.**
- [x] **Step 4: Update the migration task book with the task-5 result and leave task 6 unchecked.**
