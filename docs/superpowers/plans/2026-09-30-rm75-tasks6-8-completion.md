# RM75 Tasks 6-8 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect PPO to `RM75-v0`, validate the seven-axis environment's stability and target region, benchmark collision performance, and finish the Chinese migration documentation.

**Architecture:** Reuse the dimension-driven PPO network for both robots and add only idempotent RM75 registration/default selection. Express stability and reachability as repeatable unittests, retain the reviewed STL collision model unless the measured benchmark or contact behavior requires proxies, and document the exact conversion/test/train workflow. Keep the temporary `link_7` TCP explicit because no measured tool transform is available.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, MuJoCo, dm_control, NumPy, unittest, TensorBoard.

**Spec:** `docs/superpowers/plans/2026-09-29-rm75-mujoco-migration.md` tasks 6-8 and final verification.

## Global Constraints

- Preserve `UR5-v0` while making `RM75-v0` the migration training default.
- PPO network sizes must come from Gymnasium spaces, never hard-coded 21/7 values.
- Use normalized policy actions in `[-1, 1]` and the environment's `action_scale` mapping.
- Do not invent a real TCP transform; retain and document the temporary `link_7` origin.
- Do not expand URDF joint limits to make unreachable targets pass.
- Keep the STL collision model unless measured performance or contact behavior justifies proxies.
- Tests use standard-library `unittest` because pytest is not installed.
- Do not create Git commits on the user's behalf.

---

### Task 1: Connect PPO to RM75

**Files:**
- Modify: `Algorithms/ppo_continuous_action.py`
- Create: `tests/test_rm75_ppo_smoke.py`

**Interfaces:**
- Consumes: registered `RM75-v0` with observation `(21,)` and action `(7,)`.
- Produces: `Agent` outputs `(batch, 7)` actions and finite PPO statistics; CLI accepts `--env-id RM75-v0`.

- [x] **Step 1: Add a unittest for RM75 vector-environment and Agent tensor shapes.**
- [x] **Step 2: Add idempotent RM75 registration to the PPO module and set `Args.env_id` default to `RM75-v0`.**
- [x] **Step 3: Run the network test and an 8-timestep no-save/no-track CPU training update.**

### Task 2: Validate stability, temporary TCP, and target region

**Files:**
- Modify: `tests/test_rm75_env.py`
- Modify only if evidence requires it: `env_rm75.py`
- Modify only if evidence requires it: `assets/rm75/rm75.xml`

**Interfaces:**
- Consumes: RM75 home pose, `ee_site`, current random target box, position controller.
- Produces: one-second hold drift below `0.05 rad`, collision-free reset poses, and five validated random-box reference targets.

- [x] **Step 1: Add a one-second zero-action stability test.**
- [x] **Step 2: Add a target-region test for four corners plus center: IK error below `0.01 m`, finite in-limit joints, and no table/self collision.**
- [x] **Step 3: Run tests; shrink the box only if measured failures require it.**
- [x] **Step 4: Render the scene offscreen and verify the temporary `ee_site` remains attached to `link_7`; record that real TCP and real joint-direction checks require hardware data.**

### Task 3: Benchmark collision model and finish documentation

**Files:**
- Modify: `README_zh-CN.md`
- Modify: `docs/superpowers/plans/2026-09-29-rm75-mujoco-migration.md`
- Modify only if evidence requires it: `assets/rm75/rm75.xml`

**Interfaces:**
- Consumes: completed RM75 model/environment/PPO path.
- Produces: measured 1000-step timing and reproducible Chinese conversion, validation, visualization, training, and checkpoint-compatibility guidance.

- [x] **Step 1: Benchmark 1000 no-render zero-action environment steps and inspect contacts/stability.**
- [x] **Step 2: Retain STL collision geometry because the benchmark is acceptable and contacts are stable.**
- [x] **Step 3: Add the RM75 workflow and checkpoint incompatibility section to `README_zh-CN.md`.**
- [x] **Step 4: Run all unittests, Gymnasium checker, direct environment smoke test, PPO 8-step update, syntax checks, and diff checks.**
- [x] **Step 5: Mark tasks 6-8 with evidence, leaving real TCP/hardware-only checks explicitly pending external measurements.**

## Completion Evidence

- 17 unittests passed.
- Gymnasium `check_env(..., skip_render_check=True)` passed; only the expected advisory about infinite observation-space bounds was emitted.
- The final 8-step CPU PPO update exited with code 0 and printed `SPS: 907`.
- The full STL model ran 1000 environment steps in `0.212360 s` without collision; maximum zero-action joint drift was `0.034816259 rad`.
- All five target-box reference points solved below 5 mm error, within joint limits and without collision.
- Real TCP translation/orientation and real-controller joint directions remain external hardware measurements, not software-generated values.
