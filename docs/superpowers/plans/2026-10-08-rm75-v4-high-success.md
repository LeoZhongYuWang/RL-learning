# RM75 V4 High Success Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible RM75 V4 training path that permits a small collision rate, evaluates on fixed seeds, and preserves the best policy by success-first ranking.

**Architecture:** Keep the current PPO and environment structure. Make the RM75 collision penalty configurable, add deterministic evaluation inside the trainer, save a best model beside the final model, and provide one launch script containing the approved V4 parameters.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, PyTorch, TensorBoard, Weights & Biases, unittest

**Spec:** `docs/superpowers/specs/2026-10-08-rm75-v4-high-success.md`

## Global Constraints

- Train from random initialization with `target_sampling=uniform`.
- Use 2,500,000 environment steps and learning rate `1e-4`.
- Use collision penalty 45 while collision still terminates the episode.
- Evaluate 100 deterministic episodes every 200,000 steps using seeds starting at 20000.
- Target success rate is 82% and maximum accepted collision rate is 10%.
- Keep all existing UR5 behavior and model files unchanged.

---

### Task 1: Configurable RM75 collision penalty

**Files:**
- Modify: `env_rm75.py`
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_env.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: `RM75P2PEnv(..., target_sampling: str)` and `Args`.
- Produces: `RM75P2PEnv(..., collision_penalty: float = 75.0)` and `Args.collision_penalty`.

- [x] **Step 1: Write failing tests**

```python
def test_collision_penalty_is_configurable(self):
    env = RM75P2PEnv(collision_penalty=45.0)
    self.assertEqual(env.reward_cfg["collision_penalty"], 45.0)
    env.close()

def test_negative_collision_penalty_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "collision_penalty"):
        RM75P2PEnv(collision_penalty=-1.0)
```

- [x] **Step 2: Run the focused tests and confirm the constructor does not yet accept the parameter**

Run: `.venv/bin/python -m unittest tests.test_rm75_env.RM75EnvironmentTests.test_collision_penalty_is_configurable -v`

Expected: FAIL with an unexpected keyword argument.

- [x] **Step 3: Implement the environment and trainer parameter**

```python
def __init__(..., collision_penalty: float = 75.0):
    if collision_penalty < 0.0:
        raise ValueError("collision_penalty must be non-negative")
    self.reward_cfg["collision_penalty"] = float(collision_penalty)
```

`make_env` passes the argument only for `RM75-v0`, so UR5 construction stays compatible.

- [x] **Step 4: Run the RM75 environment and PPO smoke tests**

Run: `.venv/bin/python -m unittest tests.test_rm75_env tests.test_rm75_ppo_smoke -v`

Expected: PASS.

### Task 2: Fixed-seed evaluation and best-model selection

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Test: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: trained `Agent`, `Args`, Gymnasium `info` fields `success`, `collision`, and `dist`.
- Produces: `evaluate_policy(agent, args, device) -> dict[str, float]` and `should_update_best(metrics, best_metrics, max_collision_rate) -> bool`.

- [x] **Step 1: Write failing ranking tests**

```python
def test_best_eval_prefers_success_before_collision(self):
    best = {"success_rate": 0.80, "collision_rate": 0.00, "mean_final_distance": 0.10}
    candidate = {"success_rate": 0.84, "collision_rate": 0.08, "mean_final_distance": 0.12}
    self.assertTrue(should_update_best(candidate, best, 0.10))

def test_best_eval_rejects_excess_collision(self):
    candidate = {"success_rate": 0.90, "collision_rate": 0.11, "mean_final_distance": 0.05}
    self.assertFalse(should_update_best(candidate, None, 0.10))
```

- [x] **Step 2: Run the ranking tests and confirm the helpers are missing**

Run: `.venv/bin/python -m unittest tests.test_rm75_ppo_smoke.RM75PPOSmokeTests.test_best_eval_prefers_success_before_collision -v`

Expected: FAIL on import.

- [x] **Step 3: Implement deterministic evaluation and ranking**

```python
rank = (
    metrics["success_rate"],
    -metrics["collision_rate"],
    -metrics["mean_final_distance"],
)
```

Evaluation uses `tanh(agent.actor_mean(obs))`, 100 fixed seeds, and a separate environment. The trainer logs `eval/*`, atomically writes `best_metrics.json`, and saves `<exp_name>.best.cleanrl_model` whenever the rank improves within the collision limit.

- [x] **Step 4: Preserve evaluation state during resume**

Add evaluation arguments to `RESUME_CONFIG_FIELDS`; store `best_eval_metrics` in the full checkpoint and restore it on resume.

- [x] **Step 5: Run PPO smoke and checkpoint round-trip tests**

Run: `.venv/bin/python -m unittest tests.test_rm75_ppo_smoke -v`

Expected: PASS.

### Task 3: Reproducible V4 launch command

**Files:**
- Create: `Scripts/train_rm75_v4.sh`
- Modify: `README_zh-CN.md`

**Interfaces:**
- Consumes: the new PPO CLI arguments.
- Produces: one executable command that starts the approved V4 experiment.

- [x] **Step 1: Add the launch script**

```bash
.venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75-v0 \
  --exp-name RM75-v4-high-success \
  --total-timesteps 2500000 \
  --learning-rate 1e-4 \
  --target-sampling uniform \
  --collision-penalty 45 \
  --initial-action-std 0.30 \
  --final-action-std 0.10 \
  --action-std-hold-fraction 0.20 \
  --eval-interval 200000 \
  --eval-episodes 100 \
  --eval-seed 20000 \
  --eval-max-collision-rate 0.10 \
  --eval-target-success 0.82 \
  --checkpoint-interval 100000 \
  --track
```

- [x] **Step 2: Document final, best, and checkpoint files**

Document `.cleanrl_model`, `.best.cleanrl_model`, `best_metrics.json`, and `checkpoints/checkpoint_latest.pt`, including Ctrl+C graceful stop and `--resume-from` recovery.

- [x] **Step 3: Verify CLI parsing without starting a full run**

Run: `.venv/bin/python Algorithms/ppo_continuous_action.py --help`

Expected: all V4 arguments appear.

### Task 4: Full verification and launch

**Files:**
- Verify: `env_rm75.py`
- Verify: `Algorithms/ppo_continuous_action.py`
- Verify: `Scripts/train_rm75_v4.sh`

**Interfaces:**
- Consumes: completed Tasks 1 through 3.
- Produces: a tested and running V4 experiment.

- [x] **Step 1: Run all unit tests**

Run: `RM75_DEBUG=0 .venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

- [x] **Step 2: Run syntax and whitespace checks**

Run: `.venv/bin/python -m py_compile env_rm75.py Algorithms/ppo_continuous_action.py Scripts/visualize_agent.py`

Run: `git diff --check`

Expected: both commands succeed.

- [x] **Step 3: Launch V4 training**

Run: `bash Scripts/train_rm75_v4.sh`

Expected: a new run directory is created, W&B starts, and the first checkpoint and fixed-seed evaluation are scheduled.

## Self-review

- Spec coverage: collision tradeoff, uniform sampling, fixed validation, best model, checkpointing, W&B logging, and UR5 isolation are covered.
- Placeholder scan: no TBD, TODO, or unspecified implementation steps remain.
- Type consistency: evaluation metrics use `dict[str, float]`; checkpoint payload stores the same dictionary as `best_eval_metrics`.
