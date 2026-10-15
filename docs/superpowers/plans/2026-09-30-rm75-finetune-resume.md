# RM75 Fine-Tuning and Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RM75-v3 weight-based fine-tuning, hard-region target sampling, and recoverable PPO checkpoints that preserve optimizer and schedule progress.

**Architecture:** Keep legacy `.cleanrl_model` files as evaluation-compatible agent weights. Add a versioned full training checkpoint for interrupted-run resume, while treating `--init-from` as a new fine-tuning run with a fresh optimizer and schedule. Add a deterministic mixed target sampler to the existing RM75 environment without changing observation or action dimensions.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, PyTorch, Tyro, unittest, TensorBoard.

**Spec:** User request in this task: implement current-model fine-tuning and future interrupted-training resume, then document meanings and usage.

## Global Constraints

- Preserve the 28-dimensional observation and 7-dimensional action contracts.
- Preserve RM75-v3 reward values, collision rules, action scale, horizon, and full target box.
- Load the existing `runs/RM75-v3/ppo_continuous_action.cleanrl_model` as legacy agent weights without modifying it.
- `--init-from` starts a new run and optimizer; `--resume-from` continues a versioned full checkpoint.
- Periodic checkpoints must be written atomically and only after a completed PPO iteration.
- The first Ctrl+C requests a checkpointed graceful stop; Ctrl+Z is not part of the workflow.

---

### Task 1: Hard-region mixed target sampling

**Files:**
- Modify: `env_rm75.py`
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_env.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `RM75P2PEnv(target_mode="random")` and the existing full target box.
- Produces: `RM75P2PEnv(..., target_sampling="uniform" | "hard_mixture")`, `_sample_random_goal() -> tuple[np.ndarray, str]`, and `info["target_region"]`.

- [x] **Step 1: Write failing tests for sampler validation, deterministic seeding, legal bounds, and 50/25/25 source proportions.**

```python
env = RM75P2PEnv(target_mode="random", target_sampling="hard_mixture", seed=7)
samples = [env._sample_random_goal() for _ in range(4000)]
self.assertTrue(all(-0.5 <= goal[0] <= -0.35 for goal, _ in samples))
self.assertTrue(all(-0.3 <= goal[1] <= 0.3 for goal, _ in samples))
```

- [x] **Step 2: Run the focused environment tests and confirm they fail because `target_sampling` is not accepted.**

Run: `.venv/bin/python -m unittest tests.test_rm75_env`

- [x] **Step 3: Implement uniform and hard-mixture sampling and pass the option through `make_env`.**

```python
if draw < 0.50:
    region = "uniform"
elif draw < 0.75:
    region = "hard_negative_y"
else:
    region = "hard_far_x"
```

- [x] **Step 4: Run the focused environment and PPO smoke tests.**

Run: `.venv/bin/python -m unittest tests.test_rm75_env tests.test_rm75_ppo_smoke`

### Task 2: Legacy weight initialization and full checkpoint format

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: legacy direct state dicts and versioned checkpoint dictionaries.
- Produces: `load_agent_weights(...)`, `save_training_checkpoint(...)`, `load_training_checkpoint(...)`, `--init-from`, `--resume-from`, and `--checkpoint-interval`.

- [x] **Step 1: Write failing tests showing that legacy weights initialize an Agent and full checkpoints restore Agent, Adam optimizer, global step, completed iteration, and configuration.**

```python
save_training_checkpoint(path, agent, optimizer, global_step=2048,
                         completed_iteration=1, args=args, run_name="test")
metadata = load_training_checkpoint(path, restored_agent, restored_optimizer, "cpu")
self.assertEqual(metadata["global_step"], 2048)
```

- [x] **Step 2: Run the PPO smoke tests and confirm the checkpoint helpers are missing.**

Run: `.venv/bin/python -m unittest tests.test_rm75_ppo_smoke`

- [x] **Step 3: Implement atomic versioned checkpoint save/load and legacy/full agent-weight loading.**

```python
payload = {
    "format_version": 1,
    "agent_state_dict": agent.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step": global_step,
    "completed_iteration": completed_iteration,
    "args": asdict(args),
}
```

- [x] **Step 4: Integrate initialization and resume before the PPO loop; restore schedule position and training-critical arguments for full resume.**

- [x] **Step 5: Run the checkpoint-focused tests.**

Run: `.venv/bin/python -m unittest tests.test_rm75_ppo_smoke`

### Task 3: Periodic saves and graceful interruption

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `save_training_checkpoint` and PPO iteration boundaries.
- Produces: `runs/<run>/checkpoints/checkpoint_latest.pt`, continuous TensorBoard steps on resume, and first-Ctrl+C graceful stop.

- [x] **Step 1: Add schedule-boundary tests for the next checkpoint step.**

```python
self.assertEqual(next_checkpoint_step(0, 100000), 100000)
self.assertEqual(next_checkpoint_step(102400, 100000), 200000)
```

- [x] **Step 2: Save `checkpoint_latest.pt` atomically whenever a completed iteration crosses the interval and at normal completion.**

- [x] **Step 3: Install a first-Ctrl+C handler that stops after the current PPO iteration, saves the checkpoint, and exits without automatic evaluation.**

- [x] **Step 4: Verify focused tests and an 8-step CPU training smoke run.**

Run: `RM75_DEBUG=0 .venv/bin/python Algorithms/ppo_continuous_action.py --env-id RM75-v0 --total-timesteps 8 --num-steps 8 --num-minibatches 1 --update-epochs 1 --no-track --no-save-model --no-cuda`

### Task 4: Documentation and full regression

**Files:**
- Modify: `README_zh-CN.md`
- Modify: `docs/superpowers/plans/2026-09-30-rm75-finetune-resume.md`

**Interfaces:**
- Consumes: all implemented CLI flags and checkpoint semantics.
- Produces: copyable fine-tune, fresh-training, interruption, exact-resume, and visualization commands.

- [x] **Step 1: Document that `--init-from` creates a new fine-tuning schedule while `--resume-from` continues an interrupted planned run.**

- [x] **Step 2: Document the recommended RM75-v3 500k hard-mixture fine-tuning command with `2e-5`, `0.12 -> 0.08`, and fixed evaluation seeds.**

- [x] **Step 3: Run all unit tests, `py_compile`, Gymnasium `check_env`, `git diff --check`, a legacy-init smoke run, and a full-checkpoint load smoke test.**

- [x] **Step 4: Delete only temporary smoke-run artifacts and record verification evidence in this plan.**

## Verification Record

- 26 unit tests passed, including deterministic hard-mixture sampling, legacy weight initialization, full Agent/Adam checkpoint round-trip, schedule boundary, and resume configuration tests.
- `py_compile` passed for the RM75 environment, PPO trainer, visualization script, and isolated checkpoint worker.
- Gymnasium `check_env(..., skip_render_check=True)` passed in `hard_mixture` mode; only the pre-existing infinite observation-bound warning remains.
- An 8-step CPU run loaded `runs/RM75-v3/ppo_continuous_action.cleanrl_model` through `--init-from`, used `hard_mixture`, completed one PPO update, and exited with code 0.
- A separate 8-step run wrote `checkpoint_latest.pt`; a second process loaded it through `--resume-from`, restored `global_step=8` and `completed_iteration=1`, and exited with code 0.
- CLI help exposes `--init-from`, `--resume-from`, `--checkpoint-interval`, and `--target-sampling`.
- All temporary smoke runs created by this implementation were deleted. Existing RM75-v3 and historical formal runs were preserved.
