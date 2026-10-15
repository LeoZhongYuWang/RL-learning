# acRM75-RL：UR5 / RM75 强化学习仿真

本说明对应 2026-10-13 当前源码和活动 `runs`，不对应已停用的随机方向课程版本。主算法为 PPO；没有真实视觉传感器、夹爪抓取或真机控制。

## 当前模型与任务

| 保留模型 | 环境 | 观测 / 动作 | 实际任务 | 结果口径 |
|---|---|---:|---|---|
| UR5-v3 | UR5-v0 | 19 / 6 | 随机点到达 | 训练末段最近100回合：81%成功、19%碰撞 |
| RM75-v3 | RM75-v0 | 28 / 7 | 随机点到达，含自碰撞 | 训练末段最近100回合：79%成功、0%碰撞 |
| RM75-v4-high-success best | RM75-v0 | 28 / 7 | 优化后的随机点到达 | 固定100种子：84%成功、0%碰撞 |
| RM75-direction-v6-finetune best | RM75Direction-v0 | 31 / 7 | 随机位置＋固定向下切入方向 | 选模种子97%；另200个新种子93%、1%碰撞 |

前两行是训练滚动统计，后两行是确定性评估，不能当作同一组对照实验。方向任务的位置范围也更小。旧模型能加载，不等于当前环境完整复现其历史奖励。

当前方向任务将工具局部 +Z 对齐世界 -Z，绕该轴的旋转自由。成功要求距离 <5 cm、方向误差 <15°，连续保持5个环境步。**它不是在方位角、倾角范围内随机采样的方向任务。** 原随机方向目标尚未完成，历史课程计划不能当作当前启动说明。

## 环境与可视化

项目路径：`/home/zhongyu/wzy_master/acRM75-RL-source`。

```bash
cd /home/zhongyu/wzy_master/acRM75-RL-source
source .venv/bin/activate
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

若激活失败，直接使用 `.venv/bin/python`。换目录后不能只凭提示符 `(.venv)` 判断环境是否正确。

最新方向最佳模型：

```bash
.venv/bin/python Scripts/visualize_agent.py \
  --model-path runs/RM75Direction-v0__RM75-direction-v6-finetune__1__1791869875/RM75-direction-v6-finetune.best.cleanrl_model \
  --episodes 5
```

RM75 点到点最佳模型：

```bash
.venv/bin/python Scripts/visualize_agent.py \
  --model-path runs/RM75-v0__RM75-v4-high-success__1__1791429044/RM75-v4-high-success.best.cleanrl_model \
  --episodes 5
```

两个历史基线：

```bash
.venv/bin/python Scripts/visualize_agent.py \
  --model-path runs/UR5-v3/ppo_continuous_action.cleanrl_model \
  --episodes 5

.venv/bin/python Scripts/visualize_agent.py \
  --model-path runs/RM75-v3/ppo_continuous_action.cleanrl_model \
  --episodes 5
```

无窗口评估：在上述命令加 `--headless`，可把 `--episodes` 改为100，并指定 `--seed 30000`。默认确定性动作；`--stochastic` 才会随机采样动作。建议始终显式指定模型，不依赖“最近修改的文件”选择。

## 训练入口状态：先看这里

`Scripts/train_rm75_v4.sh` 是点任务入口。方向脚本现已修正为使用当前V6-finetune best初始化，并创建 `RM75-direction-v6-clean-finetune` 新实验；不再引用缺失的原V6文件。V5冻结起点权重及所有数值参数保持不变。这个入口不是复现原50万步微调，而是从其best开始新的微调。

当前方向 `checkpoint_latest.pt` 引用的V5权重在最终复查时可找到。该断点已完成原预算，resume不会自动再训练；已有best可视化不需要两个旧模型。

当前best仅替换学生初始化来源，不替换V5起点准备策略。自动重置样本问题仍待行为修复，建议先修复后再正式训练；本次没有启动训练。

- `--init-from`：加载网络权重，开始新实验、新优化器、新步数；适合微调。
- `--resume-from`：恢复网络、优化器、已保存步数和训练配置；用于中断续训，不是自动延长已完成计划。
- `--near-start-from`：冻结的起点准备策略，不是学生初始化模型；仅训练reset使用。
- `.best.cleanrl_model`：评估选出的最佳权重，用于展示或新微调。
- `checkpoint_latest.pt`：最新训练状态，不一定是最佳模型。本次方向断点对应最后88%的版本，不是97%的best。

## 当前实现

```text
PPO Actor → 归一化动作[-1,1]^7 → 0.02 rad关节目标增量
          → q_target累加及限位 → XML位置执行器 → MuJoCo
MuJoCo状态 → env观测/奖励/结束标志 → PPO rollout/GAE/更新
```

RM75物理步长0.002 s，每次环境step执行10个物理步，控制周期0.02 s。XML中的kp/kv形成位置伺服；增量限幅不等于实际速度、加速度或冲击的硬限制。

当前方向训练还使用冻结策略准备近处起点：50%的回合申请，接受比例不一定50%。准备动作不进入学生PPO批次，也不计入学生global_step；评估和可视化没有这个辅助策略。它不是监督模仿学习，但也不能描述成所有回合都从原起点纯探索。

## 图表怎么看

| 指标 | 用途 |
|---|---|
| eval/success_rate | 固定种子完整任务严格成功率，选best的主要指标 |
| eval/collision_rate | 与成功率共同验收，不只看回报 |
| eval/pose_reached_rate | 曾同时满足位置/方向，但不要求连续保持5步 |
| eval/position_success_rate | 曾进入位置范围，不代表方向达标 |
| eval/mean_final_direction_angle_deg | 终止时方向误差，全体回合平均，不是成功回合专属 |
| charts/success_rate | 训练最近100回合，含随机动作和混合起点 |
| eval_near/success_rate | IK近起点诊断，不等于策略实际到达起点的成功率 |
| charts/arrival_prefix_steps | 额外起点准备步数；样本效率比较需要计入 |
| losses/approx_kl、clipfrac | 策略变化程度，不是任务成功指标 |

当前没有 `best_strict_success_rate` 字段。97%是 `eval/success_rate` 在301056步的最佳评估点；对应数据在该运行的 `best_metrics.json`。

## 学习文档与检查结果

1. [代码入门与RM75环境逐函数说明](CODE_GUIDE_zh-CN.md)
2. [当前PPO算法学习指南](PPO_CONTINUOUS_ACTION_GUIDE_zh-CN.md)
3. [当前代码审查和精简建议](CURRENT_CODE_AUDIT_zh-CN.md)
4. [保留模型演变和证据边界](MODEL_EVOLUTION_GUIDE_zh-CN.md)
5. [点任务V4基线对照](RM75_V4_CODE_EVOLUTION_REPORT_zh-CN.md)
6. [模型与结果清单](runs/README_RM75_RESULTS.md)

已完成等价精简：PPO 1604→1452行，RM75环境1266→1255行；删除死代码及未接通的视频/上传扩展，合并日志和保存。观测、网络、奖励、动作和旧权重兼容保持不变。仍待修复：NEXT_STEP重置样本、方向hard_mixture及成功/碰撞语义。旧 `--capture-video`、`--upload-model`、`--hf-entity` 不再属于PPO CLI；可视化脚本不受影响。

现存五份根目录Markdown原文备份位于 `archive/docs-before-audit-20261013/`；原runs清单在复查时已不存在，本次新建。`docs/superpowers/` 为模型相关历史记录；`report_work/` 中旧PDF/DOCX并未重新生成。它们不是当前运行说明，详见各目录索引。

许可及上游归属保留原 `LICENSE`；本次没有更改许可证。
