# RM75 Direction Reward Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Implement each checked task in order and run the named regression tests before proceeding.

**Goal:** Make the RM75 PPO task learn position and approach direction simultaneously without rewarding retreat from the target.

**Architecture:** Keep the 31-dimensional observation and existing PPO network unchanged. Replace the distance-dependent direction potential difference with gated angular improvement, add a small post-reach position-retention penalty, and separate angle-only, simultaneous-pose, and strict held-success metrics.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, PyTorch, unittest, TensorBoard/W&B.

**Spec:** Current task diagnosis from the 1,001,472-step RM75Direction-v1 checkpoint.

## Global Constraints

- Modify only `/home/zhongyu/wzy_master/acRM75-RL-source`.
- Preserve `RM75-v0` 28-observation compatibility and its saved V4 model.
- Keep `RM75Direction-v0` at 31 observations and 7 actions.
- Do not add a curriculum, a second policy, or new neural-network layers.

---

### Task 1: Correct direction shaping and position retention

**Files:**
- Modify: `env_rm75.py`
- Test: `tests/test_rm75_direction_contract.py`

**Interfaces:**
- Consumes: `_direction_gate(dist: float | None) -> float` and per-step distance/direction errors.
- Produces: `_compute_direction_progress(prev_angle, angle, dist) -> float` and `position_retention` reward component.

- [x] Write regression tests proving unchanged angle earns zero direction reward even when distance changes.
- [x] Write a regression test proving reduced angle earns positive reward near the goal.
- [x] Implement gated angular improvement without differentiating the gate itself.
- [x] Add a small penalty only after the target position has previously been reached and subsequently lost.
- [x] Run the focused direction contract tests.

### Task 2: Separate angle and simultaneous-pose metrics

**Files:**
- Modify: `env_rm75.py`
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `Scripts/visualize_agent.py`
- Test: `tests/test_rm75_direction_contract.py`

**Interfaces:**
- Produces: `angle_reached`, `pose_reached`, `min_direction_angle_at_position_deg`, `angle_success_rate`, and `pose_reached_rate`.

- [x] Track angle-only and simultaneous-pose events independently.
- [x] Track the best direction error observed while distance is below 5 cm.
- [x] Log separate training and fixed-seed evaluation metrics.
- [x] Keep `success_rate` as the strict five-step held success metric.
- [x] Run focused and legacy V4 compatibility tests.

### Task 3: Update the clean training entry point

**Files:**
- Modify: `Scripts/train_rm75_direction.sh`

**Interfaces:**
- Consumes: restored V4 28-D point policy through `--init-from` expansion.
- Produces: a fresh `RM75-direction-v2-reward-fix` run.

- [x] Rename the experiment so it cannot overwrite or resume the flawed v1 run.
- [x] Keep conservative fine-tuning hyperparameters and fixed-seed evaluation.
- [x] Run shell syntax validation and the minimal PPO smoke check.

## Verification and implementation notes

- 13 direction/V4 contract tests passed, including no distance-only direction reward,
  no profitable angular round trip, distinct success events, and reset clearing.
- A temporary 4096-step PPO run completed two updates and fixed-seed evaluation;
  all new TensorBoard metrics were present. No test model or W&B run was saved.
- The legacy V4 visualizer completed two successful random-position episodes.
- The 31-D visualizer reports angle-only and simultaneous-pose rates separately.
- A temporary checkpoint round-trip passed. CUDA RNG tensors are restored on CPU
  as required by the PyTorch API; this test mocked CUDA rather than testing GPU hardware.
- Direction best-model ranking now prefers strict success, simultaneous pose reach,
  then angle error while inside the position range, within the collision limit.
- An episode that never enters the position range contributes 180 degrees to that
  error statistic, preventing low-coverage policies from winning by exclusion.
- The README records the actual v2 command, rewards and metric meanings.
- Necessary tests passed; formal training and convergence assessment remain the user's next step.
