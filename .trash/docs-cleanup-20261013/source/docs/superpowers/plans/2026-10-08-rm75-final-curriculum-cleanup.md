# RM75 Final Curriculum and Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the strongest RM75 policies, replace the unstable five-stage approach curriculum with a gradual six-stage curriculum, remove verified dead code and obsolete launchers, and produce one canonical training path toward the final 5 cm / 15 degree task.

**Architecture:** Keep the current RM75 environment, PPO network, observation/action contracts, reward function, checkpoint format, and deterministic evaluator. Change only curriculum granularity and transition scheduling: V5 best starts at a new precision-only Stage 4, Stage 5 expands range moderately, and Stage 6 is the final task. Exploration state is preserved across transitions and learning rate becomes globally monotonic.

**Tech Stack:** Python 3.10, PyTorch PPO, Gymnasium, MuJoCo, TensorBoard, Weights & Biases, Bash, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-08-rm75-final-training-audit.md`

## Global Constraints

- Do not start formal PPO training during implementation.
- Do not modify the final 5 cm / 15 degree success definition.
- Do not modify reward weights in this pass.
- Preserve UR5-v0, RM75-v0, RM75Approach-v0, 19/28/34-dimensional observations, and 6/7-dimensional actions.
- Preserve legacy agent-weight loading and full-checkpoint resume.
- Preserve all UR5 runs, `RM75-v3`, the RM75 point-task V4 result, V4 approach fallback, and V5 approach primary source.
- Move failed artifacts to `.trash/` only after explicit approval; never permanently delete them.
- Do not create Git commits unless the user explicitly requests them; the worktree already contains user-owned changes.

---

### Task 1: Lock the retained-model manifest

**Files:**
- Modify: `runs/README_RM75_RESULTS.md`
- Test: read-only path and SHA-256 verification

**Interfaces:**
- Consumes: the fixed-seed audit in the spec.
- Produces: one authoritative list of training sources and artifacts that cleanup must not move.

- [ ] **Step 1: Add the approach-task retention section**

Record these exact primary and fallback artifacts:

```markdown
## RM75 随机位姿任务

- 主训练起点（V5 best）：`RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/RM75-approach-v5-best-finetune.best.cleanrl_model`
- 主最佳断点：`RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/checkpoints/checkpoint_best.pt`
- 方向备用模型（V4 best）：`RM75Approach-v0__RM75-approach-v4-gradual-curriculum__1__1791446628/ppo_continuous_action.best.cleanrl_model`
- 统一100种子结果：V5 pose error 2.0939；V4 pose error 2.1244；两者严格成功率均为0%。
```

- [ ] **Step 2: Verify all protected paths exist**

Run:

```bash
test -f runs/RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/RM75-approach-v5-best-finetune.best.cleanrl_model
test -f runs/RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/checkpoints/checkpoint_best.pt
test -f runs/RM75Approach-v0__RM75-approach-v4-gradual-curriculum__1__1791446628/ppo_continuous_action.best.cleanrl_model
```

Expected: all commands exit 0.

- [ ] **Step 3: Record hashes before cleanup**

Run:

```bash
sha256sum \
  runs/RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/RM75-approach-v5-best-finetune.best.cleanrl_model \
  runs/RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/checkpoints/checkpoint_best.pt \
  runs/RM75Approach-v0__RM75-approach-v4-gradual-curriculum__1__1791446628/ppo_continuous_action.best.cleanrl_model
```

Expected: three hashes are copied into the manifest.

### Task 2: Introduce a six-stage curriculum

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `env_rm75.py`
- Modify: `tests/test_rm75_ppo_smoke.py`
- Modify: `tests/test_rm75_env.py`

**Interfaces:**
- Consumes: `curriculum_stage_settings(args: Args, stage: int) -> dict`.
- Produces: stages 1 through 6, with Stage 6 representing the final task.

- [ ] **Step 1: Write failing PPO stage-profile tests**

Add to `tests/test_rm75_ppo_smoke.py`:

```python
def test_precision_curriculum_has_two_bridges_before_final(self):
    args = Args()
    stage4 = curriculum_stage_settings(args, 4)
    stage5 = curriculum_stage_settings(args, 5)
    stage6 = curriculum_stage_settings(args, 6)

    self.assertEqual(stage4["tilt_max_deg"], 25.0)
    self.assertEqual(stage4["success_distance_thresh"], 0.085)
    self.assertEqual(stage4["success_angle_deg"], 30.0)
    self.assertEqual(stage5["tilt_max_deg"], 35.0)
    self.assertEqual(stage5["success_distance_thresh"], 0.070)
    self.assertEqual(stage5["success_angle_deg"], 22.0)
    self.assertEqual(stage6["tilt_max_deg"], 45.0)
    self.assertEqual(stage6["success_distance_thresh"], 0.05)
    self.assertEqual(stage6["success_angle_deg"], 15.0)
```

Add a promotion-threshold assertion:

```python
self.assertEqual(curriculum_promotion_threshold(args, 4), 0.60)
self.assertEqual(curriculum_promotion_threshold(args, 5), 0.60)
self.assertEqual(curriculum_promotion_threshold(args, 6), 1.0)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
.venv/bin/python -m unittest \
  tests.test_rm75_ppo_smoke.RM75PPOSmokeTests.test_precision_curriculum_has_two_bridges_before_final
```

Expected: fail because Stage 6 and Stage 5 bridge settings do not exist.

- [ ] **Step 3: Add Stage 5 configuration and make Stage 6 final**

In `Args`, add:

```python
curriculum_stage5_tilt_deg: float = 35.0
curriculum_stage5_dist_success_thresh: float = 0.070
curriculum_stage5_orient_success_deg: float = 22.0
curriculum_stage5_position_box: Tuple[float, float, float, float] = (
    -0.45, -0.35, -0.16, 0.16
)
curriculum_stage5_promotion_success_rate: float = 0.60
```

Change Stage 4 defaults to:

```python
curriculum_stage4_tilt_deg = 25.0
curriculum_stage4_dist_success_thresh = 0.085
curriculum_stage4_orient_success_deg = 30.0
curriculum_stage4_position_box = curriculum_stage3_position_box
```

Make `curriculum_stage_settings(args, 5)` return the new bridge and move the current
final-task dictionary to Stage 6. Update `initial_curriculum_stage`, promotion logic,
validation, rollback bounds, and `RM75P2PEnv.set_approach_curriculum` to accept
`{1, 2, 3, 4, 5, 6}`.

- [ ] **Step 4: Add the environment stage-range regression test**

Add to `tests/test_rm75_env.py`:

```python
def test_approach_curriculum_accepts_final_stage_six(self):
    env = RM75P2PEnv(approach_task=True)
    try:
        env.set_approach_curriculum(
            stage=6,
            tilt_max_deg=45.0,
            success_distance_thresh=0.05,
            success_angle_deg=15.0,
            position_box=(-0.45, -0.35, -0.18, 0.18),
        )
        self.assertEqual(env.curriculum_stage, 6)
    finally:
        env.close()
```

- [ ] **Step 5: Run curriculum tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_rm75_ppo_smoke tests.test_rm75_env
```

Expected: all tests pass.

### Task 3: Prevent transition-induced policy regression

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: learned `Agent.actor_logstd`, global PPO iteration, and curriculum stage state.
- Produces: non-increasing learning rate and unchanged learned exploration at stage transitions.

- [ ] **Step 1: Write a failing global learning-rate test**

Replace the stage-reset test with:

```python
def test_learning_rate_is_globally_monotonic(self):
    values = [
        compute_learning_rate(1e-5, 2e-6, step, 2_000_000)
        for step in (0, 300_000, 600_000, 900_000)
    ]
    self.assertEqual(values, sorted(values, reverse=True))
    self.assertGreater(values[1], values[2])
```

- [ ] **Step 2: Replace stage-local LR with global LR**

Rename the pure helper to:

```python
def compute_learning_rate(initial_lr, final_lr, elapsed_steps, total_steps):
    progress = float(np.clip(elapsed_steps / total_steps, 0.0, 1.0))
    return float(initial_lr + progress * (final_lr - initial_lr))
```

In the training loop, use run-global progress and never raise the optimizer's
current learning rate (including after `--resume-from --additional-timesteps`):

```python
scheduled_lr = compute_learning_rate(
    args.learning_rate,
    args.final_learning_rate,
    global_step,
    max(args.total_timesteps, 1),
)
lrnow = min(optimizer.param_groups[0]["lr"], scheduled_lr)
optimizer.param_groups[0]["lr"] = lrnow
```

Remove `curriculum_stage_lr_decay_steps` from `Args`, validation,
`RESUME_CONFIG_FIELDS`, scripts, and tests.

- [ ] **Step 3: Preserve learned exploration on transitions**

Delete this transition block:

```python
if curriculum_transitioned_this_iteration:
    agent.reset_action_std(args.initial_action_std)
```

Also remove `curriculum_transitioned_this_iteration`. Keep the scheduled ceiling;
`set_action_std_ceiling` may clamp excessive variance but must not increase learned
`actor_logstd`.

- [ ] **Step 4: Run PPO tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_rm75_ppo_smoke
```

Expected: all tests pass and checkpoint round-trip still restores optimizer state.

### Task 4: Remove verified dead code

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Modify: `env_rm75.py`
- Modify: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: current call graph confirmed by `rg`.
- Produces: the same public environment/training interfaces with fewer unused symbols.

- [ ] **Step 1: Delete unused PPO symbols**

Remove:

```python
class GoalAugmentWrapper(...): ...
curriculum_stage1_fraction
curriculum_stage2_fraction
```

Remove their commented-out use, resume fields, and obsolete tests/script flags.

- [ ] **Step 2: Delete unused environment symbols**

Remove:

```python
def quat_mul(...): ...
def _orientation_distance_gate(...): ...
```

Change:

```python
def _get_obs(self, q_goal=None):
```

to:

```python
def _get_obs(self):
```

- [ ] **Step 3: Remove unused assignments**

Replace assignments such as:

```python
active_curriculum = apply_approach_curriculum(...)
```

with direct calls:

```python
apply_approach_curriculum(...)
```

- [ ] **Step 4: Verify removed names have no references**

Run:

```bash
rg -n 'GoalAugmentWrapper|curriculum_stage[12]_fraction|quat_mul|_orientation_distance_gate|q_goal|active_curriculum' Algorithms env_rm75.py tests Scripts
```

Expected: no functional references.

### Task 5: Replace historical launchers with one canonical launcher

**Files:**
- Create: `Scripts/train_rm75_approach_final.sh`
- Delete: `Scripts/train_rm75_approach_v2.sh`
- Delete: `Scripts/train_rm75_approach_v3.sh`
- Delete: `Scripts/train_rm75_approach_v4.sh`
- Delete: `Scripts/continue_rm75_approach_v4.sh`
- Delete: `Scripts/train_rm75_approach_v5_best_finetune.sh`
- Delete: `Scripts/train_rm75_approach_v6_strict_finetune.sh`
- Delete: `Scripts/train_rm75_approach_v7_stage4_bridge.sh`
- Modify: `tests/test_rm75_ppo_smoke.py`
- Modify: `README_zh-CN.md`

**Interfaces:**
- Consumes: V5 best weights and the six-stage curriculum.
- Produces: one supported approach-task training command.

- [ ] **Step 1: Write the canonical-launcher test**

Add:

```python
def test_final_launcher_uses_v5_and_six_stage_curriculum(self):
    launcher = (
        Path(__file__).resolve().parents[1]
        / "Scripts"
        / "train_rm75_approach_final.sh"
    )
    content = launcher.read_text(encoding="utf-8")
    self.assertIn("RM75-approach-v5-best-finetune.best.cleanrl_model", content)
    self.assertIn("--curriculum-start-stage 4", content)
    self.assertIn("--curriculum-stage4-promotion-success-rate 0.60", content)
    self.assertIn("--curriculum-stage5-promotion-success-rate 0.60", content)
    self.assertIn("--total-timesteps 2000000", content)
    self.assertIn("--learning-rate 1e-5", content)
    self.assertIn("--final-learning-rate 2e-6", content)
```

Delete launcher tests for V4/V5/V6/V7.

- [ ] **Step 2: Create the launcher**

Use V5 best with:

```text
start stage=4
total timesteps=2,000,000
learning rate=1e-5 -> 2e-6 globally
initial/final action std=0.08 -> 0.05
entropy coefficient=0.0001
promotion window=200
late-stage promotion rate=0.60
minimum stage steps=300,000
rollback grace steps=300,000
evaluation/checkpoint interval=100,000
evaluation episodes=100
```

- [ ] **Step 3: Delete superseded launchers**

Delete only the seven scripts listed in this task. Keep:

- `Scripts/train_rm75_v4.sh` for the point-only RM75 task;
- `Scripts/visualize_agent.py`;
- `Scripts/record_hd.py`.

- [ ] **Step 4: Update the Chinese README**

Document the distinction between:

```text
RM75-v0: random position task
RM75Approach-v0: random position plus approach direction
```

Replace stale reward/parameter examples with the canonical final launcher command and
explain active curriculum success versus strict final success.

- [ ] **Step 5: Validate launcher and documentation**

Run:

```bash
bash -n Scripts/train_rm75_approach_final.sh
.venv/bin/python Algorithms/ppo_continuous_action.py --help
```

Expected: syntax succeeds and all launcher options appear in help.

### Task 6: Regression verification and recoverable artifact cleanup

**Files:**
- Test: all files under `tests/`
- Move after explicit approval: V6/V7 run and local W&B directories listed in the spec

**Interfaces:**
- Consumes: Tasks 1 through 5.
- Produces: a verified codebase and a recoverable cleanup snapshot.

- [ ] **Step 1: Run formatting and syntax checks**

Run:

```bash
git diff --check
.venv/bin/python -m py_compile Algorithms/ppo_continuous_action.py env_rm75.py Scripts/visualize_agent.py
bash -n Scripts/train_rm75_approach_final.sh
```

Expected: all commands exit 0.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all tests pass; no formal PPO process starts.

- [ ] **Step 3: Verify V5 compatibility**

Load the V5 model into a 34-observation/7-action `Agent` and run one deterministic
evaluation episode. Expected: strict state-dict loading succeeds without size mismatch.

- [ ] **Step 4: Present the exact cleanup targets for final approval**

Show the user the four V6/V7 local directories from the spec, their sizes, and their
destination under `.trash/rm75-approach-cleanup-20261008/`. Do not move them before
the user approves this destructive scope.

- [ ] **Step 5: Move approved targets recoverably**

After explicit approval, move only the approved V6/V7 run and W&B directories into the
dated `.trash` folder. Re-run the retained-path checks from Task 1.
