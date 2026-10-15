# RM75 V8 Parallel Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conservative RM75 V8 fine-tuning configuration that restores position learning, bridges curriculum stages smoothly, and supports real four-process environment collection without breaking old single-environment commands.

**Architecture:** Keep the existing observation/action spaces and PPO network unchanged so V7 weights remain loadable. Add a selectable sync/async vector backend, route curriculum updates through Gymnasium's vector method API, and put all experiment-specific curriculum and optimization choices in the V8 launcher. Make only the evidence-backed reward rebalance in the RM75 environment.

**Tech Stack:** Python 3.10, Gymnasium 1.3, MuJoCo, PyTorch PPO, Bash, unittest.

**Spec:** Approved recommendations in the current Codex conversation dated 2026-10-09.

## Global Constraints

- Preserve UR5 and legacy RM75 observation/action dimensions and model loading.
- Do not start the formal multi-million-step training run.
- Use the retained V7 best checkpoint as `--init-from`, not `--resume-from`.
- Keep implementation minimal and avoid unrelated cleanup in the dirty worktree.
- Keep collision terminal, but reduce its training penalty to `40` and allow up to `15%` collision rate when ranking best models.

---

### Task 1: Vector-environment backend

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/Algorithms/ppo_continuous_action.py`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: existing `Args`, `make_env`, and `apply_approach_curriculum`.
- Produces: `Args.vector_env_mode: str` and `make_vector_env(args, run_name)`.

- [x] Add a failing test that constructs two async RM75 approach environments, applies stage settings, resets, and steps them.
- [x] Run the focused test and confirm the missing interface failure.
- [x] Add `sync|async` validation, spawn-based `AsyncVectorEnv` construction, and vector-API curriculum calls.
- [x] Run the focused sync and async tests.

### Task 2: Position-prioritized pose error

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/env_rm75.py`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_env.py`

**Interfaces:**
- Consumes: `_normalized_pose_error(dist, angle_error)` and `reward_cfg`.
- Produces: position/direction weights `0.45/0.05` and `pose_error_weight=0.05`.

- [x] Update the existing reward regression assertions first.
- [x] Run the focused reward tests and confirm the expected failure.
- [x] Apply the minimal reward rebalance without adding new reward terms.
- [x] Run the focused reward tests.

### Task 3: V8 launcher

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/Scripts/train_rm75_approach_final.sh`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: V7 `checkpoint_best.pt`, V8 vector backend, existing six-stage curriculum flags.
- Produces: one reviewed four-environment V8 launch command.

- [x] Update launcher assertions for the V8 source, batch geometry, curriculum bridge, rollback protection, learning rate, and exploration schedule.
- [x] Run the launcher test and confirm the old V7 configuration fails it.
- [x] Configure stages as `12/40, 11/38, 10/35, 8.5/30, 7/25, 5/20`, with gradual workspace/tilt expansion.
- [x] Configure four async environments, `1024` rollout steps, `32` minibatches, three epochs, `5e-6 -> 1e-6` learning rate, and `0.06 -> 0.05` action standard deviation.
- [x] Configure `300000` minimum-stage and rollback-grace steps with rollback threshold `0.25`.
- [x] Configure collision penalty `40` and evaluation collision ceiling `0.15` so success can improve without rewarding impacts.
- [x] Run shell syntax and launcher regression tests.

### Task 4: Verification

**Files:**
- Verify only; no formal training outputs are retained.

**Interfaces:**
- Consumes: all changes above.
- Produces: evidence that old model shapes are unchanged and four subprocess environments can reset, receive curriculum updates, and step.

- [x] Compile modified Python files.
- [x] Run RM75 environment and PPO unit tests.
- [x] Run a short headless async collection smoke test without W&B or saved models.
- [x] Inspect the final diff and report every changed setting, purpose, and the exact user launch command.
