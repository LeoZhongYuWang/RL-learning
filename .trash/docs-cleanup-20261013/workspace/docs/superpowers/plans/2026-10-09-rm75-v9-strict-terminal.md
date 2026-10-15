# RM75 V9 Strict-Terminal Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the curriculum-boundary local optimum so RM75 can continue from Stage 3 toward the final 5 cm / 20 degree pose target, while making async Ctrl+C checkpointing reliable.

**Architecture:** Keep the 34-dimensional observation, 7-dimensional action, network, collision semantics, and six-stage distributions unchanged. Treat active-stage success as a one-time curriculum milestone, terminate only on strict final pose success or collision, use final tolerances for stationary pose shaping, and initialize a new V9 run from the retained V8 latest weights.

**Tech Stack:** Python 3.10, Gymnasium 1.3, MuJoCo, PyTorch PPO, Bash, unittest.

**Spec:** Approved follow-up from the 2026-10-09 V8 training audit in this Codex conversation.

## Global Constraints

- Do not change UR5 or RM75 point-task behavior.
- Do not change model observation/action dimensions.
- Keep collision terminal with penalty `40`; do not manufacture collisions.
- Do not launch formal training.
- Keep V7 and V8 artifacts intact.

---

### Task 1: Separate curriculum progress from terminal success

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/env_rm75.py`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_env.py`

**Interfaces:**
- Consumes: active `_is_success`, final `_is_strict_success`, and existing episode info.
- Produces: `active_success`, `ever_active_success`, one-time `active_success_bonus`, and strict-only approach termination.

- [x] Add regression tests proving active-only success does not terminate, strict success does terminate, and collision remains terminal.
- [x] Add a regression test proving the active milestone bonus is awarded once per episode and resets.
- [x] Implement the minimal state and reward changes.
- [x] Run focused environment tests.

### Task 2: Make pose shaping stationary across stages

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/env_rm75.py`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_env.py`

**Interfaces:**
- Consumes: final `reward_cfg` thresholds.
- Produces: `_normalized_pose_error` independent of active curriculum thresholds.

- [x] Add a test that the same physical pose has the same normalized error before and after a stage change.
- [x] Replace active denominators with final 5 cm and configured final-angle denominators.
- [x] Run focused reward tests.

### Task 3: Preserve evaluation and curriculum semantics

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/Algorithms/ppo_continuous_action.py`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: environment `success=ever_active_success` and `strict_success` info.
- Produces: stage mastery from ever-active episode success and strict completion as a separate metric.

- [x] Extend async/evaluation tests for the new info contract.
- [x] Verify promotion still uses the active-stage success series while best-model ranking still prioritizes strict success.
- [x] Run PPO tests.

### Task 4: Make async Ctrl+C graceful

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/Algorithms/ppo_continuous_action.py`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `make_env` and `make_vector_env`.
- Produces: async workers that ignore `SIGINT`; the parent remains the only shutdown coordinator.

- [x] Add a test that sends SIGINT to async worker PIDs and confirms they remain responsive.
- [x] Install `SIG_IGN` only inside async workers.
- [x] Run focused async tests.

### Task 5: V9 fine-tuning launcher and verification

**Files:**
- Modify: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/Scripts/train_rm75_approach_final.sh`
- Test: `/home/zhongyu/wzy_master/UR5-Reinforcement-Learning/tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: V8 `checkpoint_latest.pt` through `--init-from`.
- Produces: a fresh 1.5M-step Stage-3 V9 run with four async environments.

- [x] Update launcher regression expectations first.
- [x] Configure `5e-6 -> 2e-6`, `0.06 -> 0.05`, Stage-3 start, 4x1024 batch geometry, and unchanged collision constraints.
- [x] Run syntax, unit, model-shape, and short PPO smoke tests.
- [x] Inspect the final diff and provide the exact launch command and monitoring gates.
