# RM75 姿态任务模型保留与清理记录

## 保留

- `RM75Approach-v0__RM75-approach-v7-stage2-recovery__1__1791515433`
  - 原因：当前训练脚本仍以其 `checkpoint_best.pt` 作为可复现的初始权重。
  - W&B run: `mg2ehp7x`
- `RM75Approach-v0__RM75-approach-v8-position-bridge-4env__1__1791525134`
  - 原因：包含当前最好的 Stage 2 checkpoint 和最新 Stage 3 checkpoint，是后续优化的主要起点。
  - W&B run: `3wsu8mmy`

## 移入本地 .trash

- `RM75-approach-v4-gradual-curriculum`：固定评估成功率 0，平均距离约 19.9 cm。
- `RM75-approach-v5-best-finetune`：固定评估成功率 0，平均距离约 16.1 cm。
- `RM75-approach-final-six-stage`：固定评估成功率 0，平均距离约 16.2 cm。
- `RM75-approach-v6-4m-position-balanced`：只保持 Stage 1；已被 V7/V8 的权重链和更高阶段 checkpoint 替代。

Local W&B directories with run IDs `gkpbfre2`, `k9mac8cr`, `tbw7aefs`, and `u0ki1ig7` are moved to the same trash snapshot. The trash move is locally recoverable.

## 从 W&B 云端删除

- `gkpbfre2` — V4 gradual curriculum
- `k9mac8cr` — V5 best fine-tune
- `qsmo9j8e` — V6 strict fine-tune
- `tbw7aefs` — final six-stage
- `u0ki1ig7` — V6 position-balanced

## 明确不在清理范围

- UR5 运行和模型。
- `RM75-v0__ppo_continuous_action__1__1790740528`。
- `RM75-v0__RM75-v4-high-success__1__1791429044`。
- V7 和 V8 姿态任务运行。

