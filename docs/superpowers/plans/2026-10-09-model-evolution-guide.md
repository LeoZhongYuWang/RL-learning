# UR5 to RM75 Reinforcement-Learning Evolution Guide Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-based Chinese guide explaining how the original UR5 project evolved into UR5-V3, RM75-V3, RM75-V4, and the later pose-constrained RM75 policies, and judge which changes are necessary, effective, and reusable.

**Architecture:** Reconstruct the baseline from Git `HEAD`, derive current code changes from the working tree, and read model/run metadata from local TensorBoard and W&B artifacts. Present one consistent comparison framework across environment, reward, constraints, PPO, evaluation, visualization, results, and transferability.

**Tech Stack:** Markdown, Git, Python, Gymnasium, MuJoCo, PyTorch PPO, TensorBoard/W&B event data.

**Spec:** User request on 2026-10-09 to understand the full model and code evolution.

## Global Constraints

- Do not alter training code, models, checkpoints, W&B data, or the active training process.
- Separate verified facts from interpretation.
- Preserve actual run names and distinguish point-reaching from pose-reaching tasks.
- Explain why the code grew from hundreds to thousands of lines without treating line count as quality by itself.
- End with reusable engineering patterns and project-specific choices.

---

### Task 1: Reconstruct the baseline and version timeline

**Files:**
- Read: Git `HEAD` versions of `env.py`, `Algorithms/ppo_continuous_action.py`, and `Scripts/visualize_agent.py`
- Read: `runs/`, `wandb/`, and retained model metadata
- Create: `MODEL_EVOLUTION_GUIDE_zh-CN.md`

**Interfaces:**
- Consumes: Git baseline, local run names, W&B configs, checkpoint metadata.
- Produces: a verified chronological model table with task definitions and retained artifacts.

- [x] Extract original environment, PPO, and visualization contracts from Git.
- [x] Inventory UR5-V3, RM75-V3, RM75-V4, and every retained RM75 approach run.
- [x] Extract comparable success, collision, distance, angle, and curriculum metrics where available.

### Task 2: Compare implementation layers

**Files:**
- Read: `env.py`, `env_rm75.py`, `Algorithms/ppo_continuous_action.py`, `Scripts/visualize_agent.py`
- Create: `MODEL_EVOLUTION_GUIDE_zh-CN.md`

**Interfaces:**
- Consumes: baseline/current diffs and run evidence.
- Produces: environment, reward, constraint, PPO, checkpoint, evaluation, and visualization comparison matrices.

- [x] Explain each material change, its problem statement, mechanism, expected effect, and observed evidence.
- [x] Identify defensive engineering additions separately from learning-design changes.
- [x] Trace major features to current source locations.

### Task 3: Judge necessity and transferability

**Files:**
- Create: `MODEL_EVOLUTION_GUIDE_zh-CN.md`

**Interfaces:**
- Consumes: the comparison matrices and observed results.
- Produces: necessary/optional/experimental judgments plus a reusable workflow for future RL projects.

- [x] Classify changes as universally reusable, robot-specific, task-specific, or experimental.
- [x] Explain which changes pushed the target task forward and which did not yet prove effective.
- [x] Add a practical checklist for future robot-RL environment and PPO projects.
- [x] Validate all local links, headings, run names, and quantitative claims.
