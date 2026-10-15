# RM75 Mixed-Start PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans when available; otherwise execute inline and verify each task. Steps use checkbox syntax for tracking.

**Goal:** Teach the missing near-goal position/direction coordination while retaining complete point-to-point practice.

**Architecture:** Add an optional direction-only near-start probability, default zero. Construct non-success near poses through existing diagnostic IK, validate after settling, suppress already-satisfied milestones, and preserve the ordinary reset on failure. Evaluate complete and near-start tasks separately with only complete-task metrics selecting best models.

**Tech Stack:** Python, MuJoCo, Gymnasium, PyTorch, unittest, Bash.

**Spec:** User-approved 50/50 reset mixture from this conversation; original and publishing repositories must not be changed.

## Global Constraints

- Modify only /home/zhongyu/wzy_master/acRM75-RL-source.
- Keep 28-D legacy point and 31-D direction observations, PPO network, reward coefficients, thresholds, action scale, collision behavior, learning rate, discount and exploration configuration unchanged.
- Near start is 8-12 cm from the goal with 25-40 degree direction error; it must be collision-free after warmup and not satisfy strict success.
- IK sets only initial state, never controls agent actions or supplies demonstrations.
- Keep two-million-step scheduling; use 100k-200k new steps as a review window, not a convergence promise.
- Do not run formal training, create cloud test runs, delete saved models, or commit Git changes.

### Task 1: Optional safe seeded near reset

**Files:** env_rm75.py; tests/test_rm75_mixed_start.py.

**Interfaces:** RM75P2PEnv(..., near_start_probability=0.0); _reset_near_goal() -> bool; reset/step info near_start and near_start_requested.

- [x] Add focused tests for validation, deterministic near reset, fallback restoration and no initial milestone payment; run against the old interface to establish failure.
- [x] Validate probability in [0,1] and restrict nonzero use to direction tasks.
- [x] After sampling the same goal as ordinary reset, draw the mixture only when enabled.
- [x] Save original integration state and q_target. Try at most eight seeded candidate poses above/toward the goal with a tilted initial tool direction using solve_direction_ik.
- [x] Restore the true target direction in finally; reject unsuccessful IK and nonfinite q. Reset MjData, latch q_target and settle before checking distance, angle and collision.
- [x] On complete failure, restore original integration state and reference exactly; do not silently report a fallback as near.
- [x] Mark initial joint milestones as claimed only on valid near resets; retain original reward and episode-success trackers.
- [x] Run focused physical reset tests and legacy reward tests.

### Task 2: Training routing, evaluation separation and resume

**Files:** Algorithms/ppo_continuous_action.py; tests/test_rm75_mixed_start.py.

**Interfaces:** Args.near_start_probability; evaluate_policy(..., near_start=False); eval/ and eval_near/ scalar groups.

- [x] Test direction-only kwargs and resume field restoration.
- [x] Test forced probability zero for full evaluation and one for near evaluation even when training uses 0.5.
- [x] Count failed near initialization as unsuccessful, skip full fallback rollouts in near evaluation, and log valid_start_rate.
- [x] Keep deterministic evaluation goals/seeds matched. Only full eval metrics may select .best.cleanrl_model and best_metrics.json.
- [x] Log charts/near_start_rate for actual mixture use; near eval should not be mistaken for full success.
- [x] Run focused routing and model-selection tests.

### Task 3: Entry point, docs and minimum integration check

**Files:** Scripts/train_rm75_direction.sh; README_zh-CN.md.

**Interfaces:** Existing --init-from / --resume-from and new --near-start-probability 0.5.

- [x] Initialize from runs/RM75Direction-v0__RM75-direction-v3-near-goal__1__1791811899/RM75-direction-v3-near-goal.best.cleanrl_model.
- [x] Name new experiment RM75-direction-v4-mixed-start; retain all previous learning and evaluation parameters.
- [x] Explain IK-only initialization, actual mixture/near coverage, separate full vs near acceptance, and checkpoint resume behavior.
- [x] Validate Shell and Python syntax and unchanged legacy/publishing hashes.
- [x] Run one 4096-step, two-env CPU smoke test with no tracking or model writes, real initializer and both evaluation paths; verify scalar groups, finite tensors and checkpoint config roundtrip in /tmp.
- [x] Report exactly what changed and user-run command, with no convergence guarantee.

## Verification record

2026-10-12: 17 direction/legacy point contracts and 6 mixed-start tests passed;
Python compilation, Bash syntax and whitespace checks passed. One 4096-step,
two-environment CPU run used the actual V3 initializer, produced finite PPO
tensors and all four required full/near/coverage/mixture scalar groups. The
checkpoint roundtrip restored near_start_probability=0.5, optimizer and counters.
All smoke artifacts remained in /tmp; no formal run or cloud test was created.
Eight near resets at seeds 20000-20007 met the required bounds without collision;
this small initialization check is not a policy-success or coverage guarantee.
Legacy env/registration/visualizer and publishing env/README hashes were unchanged.
Read-only peer review found no must-fix defect in the reset/evaluation routing.
The PPO network, rewards and control bounds were unchanged; only training starts
and separated diagnostics changed. Final success must still be established by
the user's complete-task evaluation, not by these short integration tests.
