# RM75 最终位姿任务代码与模型审查

日期：2026-10-08

## 审查范围

- `Algorithms/ppo_continuous_action.py`
- `env_rm75.py`
- `Scripts/train_rm75_approach_*.sh`
- `Scripts/continue_rm75_approach_v4.sh`
- `tests/test_rm75_*.py`
- `runs/RM75Approach-*`
- 本地 W&B 运行目录

本审查不修改训练代码、不删除文件、不启动正式训练。

## 模型结论

V4 与 V5 使用相同的 100 个固定种子重新评估，最终标准均为位置小于
5 cm 且方向误差小于 15 度。

| 模型 | 严格成功率 | 最小距离 | 最小角度 | 最小综合姿态误差 | 最终距离 | 最终角度 |
|---|---:|---:|---:|---:|---:|---:|
| V4 best | 0% | 10.74 cm | 14.29° | 2.1244 | 19.97 cm | 33.08° |
| V5 best | 0% | 10.23 cm | 14.77° | 2.0939 | 16.28 cm | 42.97° |
| V6 best（历史100回合） | 0% | 10.55 cm | 14.36° | 2.1149 | 16.81 cm | 40.83° |
| V7 best（历史100回合） | 0% | 10.71 cm | 14.15° | 2.1182 | 16.88 cm | 40.91° |

### 必须保留

1. 后续训练主起点：
   `runs/RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/RM75-approach-v5-best-finetune.best.cleanrl_model`
2. V5 可恢复最佳断点：
   `runs/RM75Approach-v0__RM75-approach-v5-best-finetune__1__1791462556/checkpoints/checkpoint_best.pt`
3. 方向能力备用模型：
   `runs/RM75Approach-v0__RM75-approach-v4-gradual-curriculum__1__1791446628/ppo_continuous_action.best.cleanrl_model`
4. 用户此前要求保留的全部 UR5 结果、`runs/RM75-v3`，以及位置任务成功率
   84% 的 `runs/RM75-v0__RM75-v4-high-success__1__1791429044`。

### 可恢复清理候选

以下失败实验不作为新训练起点；审核通过后先移动到项目 `.trash/`，不永久删除：

- `runs/RM75Approach-v0__RM75-approach-v6-strict-finetune__1__1791464704`
- `runs/RM75Approach-v0__RM75-approach-v7-stage4-bridge__1__1791465667`
- `wandb/run-20261008_210507-qsmo9j8e`
- `wandb/run-20261008_212113-rwokym5x`

V4、V5 在新模型达到最终严格成功率前不清理。

## 代码问题

### P0：课程阶段一次改变三个难度维度

V5 从 Stage 3 到 Stage 4 同时发生：

- 距离阈值 10 cm -> 7.5 cm；
- 方向阈值 35° -> 25°；
- 最大倾角 25° -> 35°，并扩大位置区域。

Stage 4 只维持约 9.2 万步便回退。V7 直接从 Stage 4 开始，约 48 万步后
掌握率从约 11% 降至约 8%。这说明难度跳变本身不合理，不是单纯训练步数不足。

### P0：阶段切换会重置已学习的探索标准差

课程晋级/回退后执行 `agent.reset_action_std(args.initial_action_std)`。这会把已经
降低到适合精确控制的探索噪声突然增大。V5 进入更精确的 Stage 4 时，探索标准差
被重新提高，容易破坏已经学到的精细定位策略。

### P0：阶段切换会重启学习率计划

当前学习率按 `curriculum_stage_steps` 衰减。进入新阶段后阶段步数归零，学习率恢复
到初始值。在任务变难的同时放大学习率，会增加策略遗忘风险。

### P1：课程只有一个过渡阶段

当前 Stage 4 后直接进入最终 5 cm / 15° / 45° 全范围任务。V6 证明从 V5 直接进入
最终任务后，最小距离从 10.55 cm 恶化到 11.92 cm，连续约 40 万步无严格成功。

### P1：历史启动脚本重复且部分已失效

- V2/V3 脚本引用已不存在的运行目录；
- V4、continue-V4 已被后续策略替代；
- V5/V6/V7 脚本包含大量重复参数，只用于复现实验历史；
- 每新增一次实验就复制一份长脚本，容易出现参数漂移。

### P2：确定无调用的代码

- `GoalAugmentWrapper`：定义后从未实例化；
- `quat_mul`：无调用；
- `_orientation_distance_gate`：无调用；
- `_get_obs(q_goal=None)` 的 `q_goal` 参数：无使用；
- `curriculum_stage1_fraction`、`curriculum_stage2_fraction`：已标记为旧版且不参与课程；
- `active_curriculum` 局部变量：只接收返回值，从未读取。

## 审核后的目标设计

采用六阶段课程，每次只小幅改变精度或任务范围：

| 阶段 | 距离 | 方向 | 最大倾角 | 位置区域 |
|---|---:|---:|---:|---|
| 1 | 15 cm | 45° | 15° | 小 |
| 2 | 12 cm | 40° | 20° | 小幅扩大 |
| 3 | 10 cm | 35° | 25° | 当前 V5 分布 |
| 4 | 8.5 cm | 30° | 25° | 与 Stage 3 相同 |
| 5 | 7 cm | 22° | 35° | 中等范围 |
| 6 | 5 cm | 15° | 45° | 完整最终范围 |

V5 best 从 Stage 4 开始。Stage 4、5 最近 200 回合成功率达到 60% 且至少训练
30 万步才晋级。Stage 6 是最终任务，不再晋级。

阶段切换时：

- 保留策略已经学到的 `actor_logstd`；
- 动作标准差上限可以重开，但不主动放大实际噪声；
- 学习率在整个训练中单调衰减，不因换阶段回升；
- 奖励函数暂不修改，先单独验证课程与优化器稳定性。

## 验收标准

代码验收：

- 全部现有测试通过；
- 新增六阶段、单调学习率、切换不重置探索度的测试；
- UR5-v0、RM75-v0、RM75Approach-v0 的观测/动作维度不变；
- V5 best 仍可通过 `--init-from` 加载；
- 可在 `Ctrl+C` 后从 `checkpoint_latest.pt` 恢复。

训练验收：

- 固定 100 种子严格成功率先达到 60%，最终目标不低于 80%；
- 碰撞率不高于 10%；
- Stage 6 的 `success_rate` 与 `strict_success_rate` 一致；
- 最佳完整断点和最佳可视化模型同时保存。

