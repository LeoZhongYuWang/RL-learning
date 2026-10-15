# RM75 V5 Best-Checkpoint Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the retained 2,600,960-step RM75 approach-policy weights with correct exploration handling, stage-aware learning-rate decay, achievable later-stage promotion, and resumable best checkpoints.

**Architecture:** Keep the existing PPO and reward definitions, but fix persistence and observability before changing optimization. Start a new run from the retained best agent weights at curriculum stage 3, reset exploration conservatively, decay learning rate within each curriculum stage, and preserve both latest and best full checkpoints.

**Tech Stack:** Python 3.10, PyTorch PPO, Gymnasium, MuJoCo, TensorBoard, Weights & Biases, Bash.

**Spec:** Training audit recorded in the current Codex task on 2026-10-08; retained source weights are `runs/RM75Approach-v0__RM75-approach-v4-gradual-curriculum__1__1791446628/ppo_continuous_action.best.cleanrl_model`.

## Global Constraints

- Do not start formal training.
- Do not modify the RM75 reward weights, collision penalty, task geometry, action scale, or strict 5 cm / 15 degree success definition in this pass.
- Preserve compatibility with UR5-v0, RM75-v0, and existing checkpoints that do not contain the new arguments.
- Use fixed-seed evaluation every 100,000 steps and retain full checkpoint state whenever a new best policy is selected.
- Start V5 from curriculum stage 3 and train for 1,000,000 requested steps.
- Use stage-local learning-rate decay from `2e-5` to `5e-6`; reset the schedule only when the curriculum stage changes.

---

### Task 1: Preserve exact trainable state and best checkpoints

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`
- Test: `tests/checkpoint_roundtrip_worker.py`

**Interfaces:**
- Produces: `checkpoints/checkpoint_best.pt` alongside the agent-only best model.
- Modifies: final model/checkpoint saving so it does not clamp `actor_logstd` before saving.

- [ ] Add a regression test proving final checkpoint saving preserves the current learned `actor_logstd` rather than forcing `final_action_std`.
- [ ] Add a test proving a best evaluation can trigger a full checkpoint save with optimizer, curriculum, global-step, and W&B metadata.
- [ ] Remove the mutating final `set_action_std_ceiling(final_action_std)` call before model/checkpoint serialization.
- [ ] Save `checkpoint_best.pt` whenever deterministic evaluation replaces the best agent-only model.
- [ ] Restore `exp_name` through full-checkpoint resume so output filenames remain stable.

### Task 2: Make curriculum progression stage-aware

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Produces: `curriculum_promotion_threshold(args: Args, stage: int) -> float`.
- Produces: `Args.curriculum_start_stage`, `Args.curriculum_stage3_promotion_success_rate`, and `Args.curriculum_stage4_promotion_success_rate`.

- [ ] Add failing tests for starting an `--init-from` run directly at stage 3 and rejecting stages outside 1-5.
- [ ] Add failing tests proving stages 1-2 use the existing 0.75 threshold, stage 3 uses 0.50, and stage 4 uses 0.45.
- [ ] Initialize fresh/fine-tune curriculum state from `curriculum_start_stage`; full checkpoint resume remains authoritative.
- [ ] Apply the stage-specific threshold in promotion checks and TensorBoard logging.
- [ ] Preserve the existing rollback rules after promotion.

### Task 3: Add stage-local learning-rate decay and actual exploration metrics

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Produces: `compute_stage_learning_rate(initial_lr, final_lr, stage_steps, decay_steps) -> float`.
- Produces: `Agent.effective_action_std_stats() -> tuple[float, float, float]`.
- Produces: `Args.final_learning_rate` and `Args.curriculum_stage_lr_decay_steps`.

- [ ] Add tests proving learning rate starts at `2e-5`, reaches `5e-6` after the configured stage steps, and resets when stage steps reset to zero.
- [ ] Add validation for positive ordered learning-rate bounds and positive decay steps.
- [ ] Use stage-local decay when `approach_curriculum` and `anneal_lr` are enabled; preserve global scheduling for other tasks.
- [ ] Log `effective_action_std_mean`, `effective_action_std_min`, and `effective_action_std_max` from the actual clamped policy distribution.
- [ ] Keep the existing planned `action_std_ceiling` chart for comparison.

### Task 4: Provide the reviewed V5 launcher and validate it

**Files:**
- Create: `Scripts/train_rm75_approach_v5_best_finetune.sh`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: retained V4 best agent-only weights at step 2,600,960.
- Produces: a new V5 run with a fresh optimizer, stage-3 curriculum, controlled exploration, and resumable best checkpoints.

- [ ] Add a launcher validation test that checks the exact source model, stage 3, 1,000,000 requested steps, `2e-5 -> 5e-6` learning-rate schedule, PPO safeguards, exploration settings, and stage-specific promotion thresholds.
- [ ] Create the launcher with `initial_action_std=0.14`, `final_action_std=0.08`, `ent_coef=0.0005`, `update_epochs=3`, `target_kl=0.01`, and 100,000-step evaluation/checkpoint intervals.
- [ ] Run shell syntax validation, Python compilation, CLI help validation, checkpoint/model compatibility checks, and the full test suite.
- [ ] Confirm no formal PPO training process was launched.

