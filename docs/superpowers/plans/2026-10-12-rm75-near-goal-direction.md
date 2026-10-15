# RM75 Near-Goal Direction Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task when available; otherwise execute inline and verify each task. Do not delegate without user authorization.

**Goal:** Preserve learned point-reaching behavior while teaching simultaneous position and fixed approach-direction success.

**Architecture:** Keep the existing 31-D direction observation and PPO architecture. Replace asymmetric angular progress and history-triggered retention with bounded joint-error progress and unconditional distance cost; replace point-only milestones in the direction task with one-time joint milestones.

**Tech Stack:** Python 3.10, MuJoCo, Gymnasium, PyTorch, unittest, Bash.

**Spec:** User-approved near-goal direction task in this conversation: far away prioritize position; near the goal coordinate position and direction; retain the best partially learned policy.

## Global Constraints

- Modify only `/home/zhongyu/wzy_master/acRM75-RL-source`; never modify the publishing repository.
- Preserve legacy RM75-v0 observation size 28 and V4 reward behavior.
- Preserve RM75Direction-v0 observation size 31, 7 actions, 0.02-rad increment limit, fixed downward tool direction, 5-cm/15-degree success held for 5 steps.
- Do not add curricula, networks, or observation dimensions.
- No formal training, cloud mutation, model deletion, or automatic Git commits.

### Task 1: Focused regressions and environment reward

**Files:** Modify env_rm75.py and tests/test_rm75_direction_contract.py.

**Interfaces:** _direction_gate(dist), _compute_direction_progress(prev_angle, angle, prev_dist, dist), _compute_milestone_reward(dist, direction_angle=None), _compute_reward(...).

- [x] Add tests for zero far-direction progress, total progress penalizing distance-only retreat, round-trip joint progress, one-time joint milestones, reset clearing, and history-independent distance cost.
- [x] Run the tests against v2 and confirm the changed expectations fail.
- [x] Define gate g(d)=clip((0.20-d)/0.15,0,1); normalized angle e=angle/pi.
- [x] Keep position progress 50*(d_previous-d), and use direction progress 6*(g_previous*e_previous-g_current*e_current). Since 6/0.15=40<50, distance-only retreat cannot yield positive total progress.
- [x] Use distance cost -0.5*d only for the direction task, retaining -0.03*d for the legacy point task; remove history-triggered position_retention completely.
- [x] Give one-time joint bonuses (distance,angle,bonus): (0.15m,45deg,5), (0.10m,30deg,10), (0.07m,20deg,20). Legacy pure-distance bonuses stay unchanged.
- [x] Keep success/collision/timeout/time terms unchanged. This is heuristic undiscounted difference shaping, not a policy-invariance claim under PPO gamma.
- [x] Run direction and legacy V4 contracts.

### Task 2: Training entry point and documentation

**Files:** Modify Scripts/train_rm75_direction.sh and the current section of README_zh-CN.md.

**Interfaces:** Existing --init-from and 31-D Agent loading; no algorithm changes.

- [x] Set initialization to runs/RM75Direction-v0__RM75-direction-v2-reward-fix__1__1791808412/RM75-direction-v2-reward-fix.best.cleanrl_model.
- [x] Name the fresh experiment RM75-direction-v3-near-goal; retain 2M steps, learning rate 5e-5 with annealing, existing exploration caps, and fixed-seed evaluations.
- [x] Explain rewards, unchanged task scope, init versus resume, and the stage-qualified interpretation of metrics.
- [x] Validate shell syntax and checkpoint dimensions/loadability.
- [x] Run one temporary 4096-step PPO smoke test with tracking/checkpoint/model saving disabled and logs outside the repository.
- [x] Verify unchanged algorithm, visualization, legacy env, and publishing repository hashes.
- [x] Report actual verification, modified file locations, and user-run command; do not promise convergence.


## Actual verification

- New v3 expectations failed against v2 before implementation (four assertions and two missing-interface errors).
- 17 direction and legacy V4 regression tests passed, including the zero-distance retreat edge case.
- Bash syntax validation and Python compilation passed.
- The unmodified PPO entry point loaded the actual v2 best policy, ran 4096 CPU environment steps / two updates, and performed two one-episode fixed-seed evaluations. Returns and all model parameters were finite.
- Test evaluations had zero strict success; this short check is not a convergence experiment.
- Test TensorBoard logs were redirected to /tmp; tracking and all model/checkpoint saves were disabled.
- Algorithm, visualizer, legacy UR5 env, registration, and checked publishing-repository files retained their original hashes.
- env_rm75.py grew by ten lines; no new network or observation dimensions were added.
