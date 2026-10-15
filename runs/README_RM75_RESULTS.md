# 当前保留模型与结果

更新：2026-10-13。仅列活动runs中真实存在的任务模型；不推荐已清理的失败随机方向模型。

## 模型清单

| 模型 | 用于展示的权重文件 | 观测/动作 | 任务 |
|---|---|---:|---|
| UR5-v3 | UR5-v3/ppo_continuous_action.cleanrl_model | 19/6 | UR5随机点到达 |
| RM75-v3 | RM75-v3/ppo_continuous_action.cleanrl_model | 28/7 | RM75随机点到达，自碰撞 |
| RM75点任务V4 best | RM75-v0__RM75-v4-high-success__1__1791429044/RM75-v4-high-success.best.cleanrl_model | 28/7 | 当前点到点主模型 |
| RM75方向V6微调best | RM75Direction-v0__RM75-direction-v6-finetune__1__1791869875/RM75-direction-v6-finetune.best.cleanrl_model | 31/7 | 随机位置＋固定向下方向 |

从项目根运行时，以上相对路径都需加 `runs/`。model-path指向文件，不是目录。

## 结果与证据口径

- UR5-v3：1998848步训练末段最近100回合81%成功、19%碰撞，不是本次独立评估。
- RM75-v3：1998848步末段训练窗口79%成功、0%碰撞，不是当前RM75主基线。
- RM75点任务V4：best在2000896步，固定100回合84%成功、0%碰撞、最终平均6.58cm；最终固定评估83%，训练末段88%。
- RM75方向微调：best在301056新增步，固定100回合97%成功、0%碰撞、平均4.51cm/12.34°、242.13步；最终499712步88%成功、7%碰撞。
- 方向best另做30000～30099、40000～40099两组种子验证，每组93%成功；合计200回合93%、1%碰撞、平均4.71cm/14.69°、269.82步。本会话已实际执行，没有上传到原W&B。

方向结果不代表随机倾角/方位角已完成。两类RM75任务目标位置范围不同，不能用成功率直接证明某任务更容易或算法更优。

## 保存文件的意义

`.best.cleanrl_model`是固定评估选出的权重；普通`.cleanrl_model`是最终权重；`checkpoints/checkpoint_latest.pt`是最新网络＋优化器＋计划等训练状态。

best不是latest；当前方向latest对应88%的final，不能拿它精确恢复97%最佳点的Adam状态。两个V3历史基线只保留裸权重与日志。

两个完整计划都已经完成：点任务1220个iteration，方向微调244个iteration。resume默认恢复保存预算，不自动额外增加步骤。

## 当前依赖缺口与清理

旧方向V4/V5/原V6的云端运行已清理，当前方向云端ID为qsg0yrfd；点任务V4为57ie6r8n，V3历史基线有各自本地日志。

当前训练脚本已从最新V6-finetune best初始化新的RM75-direction-v6-clean-finetune实验；V5冻结准备权重保持不变，仅作为训练依赖，不列为现役最佳模型。原微调历史参数仍保留在其checkpoint和W&B，未改写。方向latest checkpoint已完成预算，resume不自动延长。

已有best展示正常；本次修复脚本初始化路径并精简工程代码，没有删模型或修改checkpoint，也没有更换teacher。自动重置样本问题仍待单独修复，建议处理后再正式训练。

## 文档入口

从 [当前README](../README_zh-CN.md) 获取四个模型可复制的可视化命令。指标定义和精简建议见 [当前代码审查](../CURRENT_CODE_AUDIT_zh-CN.md)。
