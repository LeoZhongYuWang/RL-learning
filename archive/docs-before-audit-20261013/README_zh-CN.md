# 基于深度强化学习的 UR5 / RM75 机械臂控制

## 当前保留的切入方向模型（2026-10-13）

V6 已由用户完成约200万学生环境步，完整固定种子评估最佳严格成功率71%、
最终模型68%。当前启动入口已准备小学习率微调，助手尚未启动正式训练。
学习模型使用 V6 最佳权重初始化，不使用最终权重或已完成计划的最新断点：

```text
runs/RM75Direction-v0__RM75-direction-v6-policy-arrival__1__1791865808/RM75-direction-v6-policy-arrival.best.cleanrl_model
```

V6 最佳完整任务评估位于其 `best_metrics.json`：1900544步，严格成功率71%，
曾同时达标72%，曾进入5 cm的比例100%，碰撞率0%，位置范围内最小方向误差
平均12.61°。尚未达到75%的验收阈值。V6 的 W&B ID 为 `d88xg6it`。
冻结的起点准备模型仍使用 V5 最佳权重（401408步、23%），不改成 V6。
V5 的 W&B ID 为 `p57uhgcl`，V4 为 `ammihpnc`；本轮所有现有模型、本地结果
和云端数据均未删除。
V4 历史参数与源码仍在其 `training_config.json` 和 `training_source/` 中。

另外 7 个旧姿态/方向实验及本地 W&B 记录已移入可恢复的
`.trash/pose-prune-20261013/`；云端存在的 3 个对应实验已删除，4 个 ID 未查到。
详细清单及核验状态见该目录 `manifest.json`。云端删除不能保证可以恢复。
历史 V4 参数和源码快照，以及 V5/V6 的最佳权重均不修改。

启动 V6 微调（新实验，不覆盖原V6结果）：

```bash
cd /home/zhongyu/wzy_master/acRM75-RL-source
bash Scripts/train_rm75_direction.sh
```

脚本的 `--init-from` 使用上述 V6 最佳权重，`--near-start-from` 仍使用
`runs/RM75Direction-v0__RM75-direction-v5-near-coordination__1__1791859371/RM75-direction-v5-near-coordination.best.cleanrl_model`。
当前脚本会创建 `RM75-direction-v6-finetune` 新实验，预算50万学生新环境步。
按2048步完整batch执行，实际499712步；起点准备的额外步数仍单独记录。
它继承模型权重，但重新创建优化器、步数和 W&B 实验，不恢复旧训练进度。
旧 V4 位置模型保留不变；V2 和 V3 方向模型已移出活动 `runs`。
微调学习率从 `1e-5` 线性衰减到接近0（原V6从 `5e-5` 衰减）。
其余设置不变，探索标准差上限仍从 `0.15` 降到 `0.06`。
这里的标准差是上限，不会把继承模型的标准差强行增加到 `0.15`。
旧 `RM75-v0` 仍为 28 维观测；`RM75Direction-v0` 为 31 维观测、7 维动作。
方向任务目标位置随机采样，工具局部 +Z 对齐世界 -Z，绕该轴的转角自由。
它要求距离小于 5 cm、方向误差小于 15°，连续保持 5 步。

原V6只改变训练近起点来源。本次微调只改变学习权重初始化、训练预算、
学习率和实验名；V6奖励、网络、观测、动作尺度0.02、碰撞检测、目标分布、
起点申请概率及探索设置均不改变，没有同时实施近处动作缩小。
每个训练回合有 50% 概率申请实际到达起点，其余仍从原起点训练学生策略；
这不是各占 50% 的环境步数。

`Algorithms/ppo_continuous_action.py` 中的 `ArrivalStartWrapper` 仅用于训练：
申请成功时，冻结的 V5 策略无探索噪声地执行到距离 <= 6 cm，再把 qpos、qvel、
ctrl、q_target 和目标原样交给学生。角度沿用实际到达值，不强制变成某个角度。
前半段不进入 PPO rollout，也不计入学生回报、回合长度和 global_step。
它不是模仿学习，没有旧动作监督标签；但训练 reset 确实利用了旧模型辅助，
不是所有回合都由学生从原起点自行探索。
交接时 `env_rm75.py::_finish_reset()` 只清空回合历史并重设奖励基准，
不清零速度、不再 warmup、不重新做 IK。起点已满足的里程碑预先领取，
且不能从位置与方向已经同时达标的状态开始，避免白领成功奖励。
最多执行 512 个准备步；碰撞、已同时达标或未到近处时，恢复原始物理状态
与控制目标，保留同一个目标，转回原起点训练，不重抽容易目标。
单独查看 `charts/arrival_prefix_steps`：它是每个已结束回合的起点准备步数，
包括准备失败的尝试；这些额外操作会增加墙钟耗时。不能仅用学生步数声称
比 V5 更高的总交互样本效率，还应计入准备步数并比较真实耗时。
`charts/near_start_rate` 记录实际接受起点的比例，可能低于申请比例。

兼容入口：不指定 `--near-start-from` 时仍使用 V5 的 IK 近起点，
物理稳定后须距目标 4–6 cm、方向误差 25–35°、无碰撞，
尚不满足严格成功条件（即使位置已在 5 cm 内，方向仍不合格）。
采样半径 4.5–5.5 cm、高度差 3–4 cm，临时 IK 方向倾斜 29–31°；
实际状态经 warmup 后再验收。现有 IK 仅用于构造 reset 状态，不控制 agent 的动作，
不提供示范。最多尝试 8 次；失败时恢复原起点并记录 `near_start=False`。
起点已满足的联合里程碑预先标记为已领取，不能在开局白领奖励。
`near_start_probability` 默认 0，因此旧训练入口和可视化仍从原起点启动。

当前奖励以“远处先靠近，近处协调方向”为目标，不给远处对准单独发奖。
设距离为 `d`（米）、角度误差为 `theta`（弧度）：

```text
g(d) = clip((0.20 - d) / 0.15, 0, 1)
联合误差 C = 50*d + 6*g(d)*theta/pi
进度奖励 = C_previous - C_current
方向状态成本 = -0.05*g(d)*theta/pi
每步奖励 = 进度奖励 - 0.5*d - 0.01 + 方向状态成本 + 一次性里程碑 + 终止奖励
```

20 cm 外方向项为零；20 cm 到 5 cm 线性提高方向的重要性，5 cm 内为完整权重。
门控改变可能让“方向进度”分项暂时为负或正，应与“位置进度”一起看：
`6/0.15=40 < 50`，因此角度不变时，仅后退不会获得正的联合进度。
角度固定时靠近仍有正的联合进度；同位置下减小角度会降低联合误差。
闭合往返的进度差分在不折扣时合计为零；这不是带 PPO 折扣因子的策略不变性保证。

V4 只有方向进度差分：角度停止改善时该项为零。V5 增加持续的小幅状态成本，
使近处停留在错误角度也有代价；20 cm 外成本为零，5 cm 内完整生效。
例如 4 cm / 30° 时，每步方向成本约 -0.00833，不改变 +100 严格成功奖励。
`0.05/0.15 < 0.5`，角度固定时后退不会降低距离与方向状态成本之和，
但这仅是局部奖励检查，不是收敛或完整轨迹的最优策略证明。
实现位于 `env_rm75.py` 的 `reward_cfg`、`step()` 和 `_compute_reward()`；
成本单独记录在 `info['reward_components']['direction_cost']`，不另加 W&B 图表。

删除了“曾到达位置后才触发”的保持惩罚。距离成本现在每一步都适用，
不会因是否曾进入 5 cm 而改变，避免不靠近反而逃避惩罚。
方向任务的纯距离里程碑替换为以下联合里程碑，每项每回合只能领取一次：

| 距离与角度同时满足 | 奖励 |
|---|---:|
| <15 cm 且 <45° | +5 |
| <10 cm 且 <30° | +10 |
| <7 cm 且 <20° | +20 |
| <5 cm 且 <15°，连续保持5步 | +100，并成功终止 |

前三项只是引导，不是课程阶段，也不算严格成功。碰撞仍立即终止并扣45，
未成功且未碰撞的超时扣10；回合上限仍1024步。没有改变 PPO 网络、观测、
动作限幅、关节限制或自碰撞检测。旧位置任务仍使用 `-0.03*d` 和原距离里程碑。

优先查看固定种子 `eval/` 指标：

| 指标 | 含义 |
|---|---|
| `position_success_rate` | 回合中曾进入 5 cm 范围，结束时未必仍在范围内 |
| `angle_success_rate` | 回合中曾达到角度 < 15°，当时可能离目标很远 |
| `pose_reached_rate` | 回合中曾同时满足位置与角度阈值 |
| `success_rate` | 同时满足阈值并连续保持 5 步的严格成功率 |
| `mean_min_direction_angle_at_position_deg` | 进入 5 cm 时的最小角度误差的回合平均；未到达位置的回合按 180°计入 |
| `mean_final_distance` / `mean_final_direction_angle_deg` | 回合结束时的距离与角度误差 |

`eval/` 始终使用原起点、固定种子、无探索噪声，只有这组指标用于选择
`.best.cleanrl_model` 和 `best_metrics.json`。`eval_near/` 使用同一组目标种子，
仍使用 V5 的 IK 近起点作为独立参考，而不是训练用的 V5 到达起点。
两组评估都绕过冻结策略；可视化也只运行学生模型，不能靠前半段辅助提高评估。
`success_rate` 的成功条件仍是 5 cm、15°、保持 5 步。
近处初始化失败计为失败且长度为 0，不滚动评估恢复后的原起点。
先看 `eval_near/valid_start_rate`（成功构造近起点的比例）；若偏低，
近处成功率和平均长度不能单独用来判断策略能力或效率。

`charts/` 中的成功率是最近 100 个混合起点训练回合的统计，包含探索噪声，
不能直接与旧版全原起点训练成功率比较。`charts/near_start_rate` 是其中实际
使用近起点的回合比例，因随机采样和 IK 失败会偏离 50%。每次评估增加一组
近处回合，因此评估耗时会增加。优先比较完整 `eval/` 的位置达标率、
目标附近方向误差、严格成功率与碰撞率，而不是只看 return。
V5 近起点比 V4 更近，不能把两版 `eval_near/` 成功率的上升直接当作能力进步。
完整 `eval/` 的起点、目标种子与成功条件没变，才是跨版本比较与验收依据。
V6 的训练近起点来自 V5 实际到达状态，但学生策略更新后可能走出新的关节状态；
因此仍须保留完整原起点训练和评估，不能保证所有后续到达状态都能泛化。
跨 V5/V6 不直接比较混合训练成功率；完整 `eval/` 和不变的 IK `eval_near/`
才是可比基准，后者不代表实际到达起点的训练成功率。
v2 和 v3 的奖励不同，也不能用 return 绝对值直接比较。
旧 v1 的 `direction_success_rate` 实际记录位置与角度同时达标，不能和新版
纯角度达标率直接比较。最佳模型先按严格成功率、同时达标率选取；这些指标
持平时，优先选择目标位置附近方向误差更小的模型，碰撞率须不超过 10%。

运行中按一次 `Ctrl+C`，等待完成当前 PPO 更新并保存 `checkpoint_latest.pt`。
同一版训练被中断后可用下列命令恢复，替换为实际 checkpoint 路径：

```bash
.venv/bin/python Algorithms/ppo_continuous_action.py \
  --resume-from runs/实际实验目录/checkpoints/checkpoint_latest.pt
```

此次微调使用 `--init-from` 开始新实验：只加载权重，重新创建
优化器、步数、学习率计划与 W&B 实验。上面的 `--resume-from` 命令只用于
当前 V6 代码不变时，恢复同一版 V6 的优化器、计数、参数和随机数状态，
同时恢复 `near_start_from` 路径。保留引用的 V5 最佳权重文件，否则起点准备
无法恢复；冻结策略在创建环境时重新从该文件加载，不会被 PPO 更新。
环境会重新 reset，不是逐帧复现旧仿真轨迹。检查点不包含可自动恢复的
环境源码；V4 的旧奖励源码已留在 `training_source/`，不能在当前环境中
直接 resume V4 并声称原样续训。
不要直接 resume 已完成200万步的V6计划来延长训练：当前代码恢复保存的
total_timesteps，不能通过该命令覆盖总步数自动延长。新微调实验中断后，
才用其自身 checkpoint_latest.pt 续训，恢复的是50万步计划与V5起点模型。
先看前10–20万学生新步：完整严格成功能否稳定超过原V6的71%、
完整任务位置能力是否保住、位置范围内方向误差能否降低，再决定继续。
IK近处评估是额外诊断，不应单凭它低就停止；连续几次完整评估明显低于71%、
且位置范围内方向误差上升时，回退原V6最佳模型，不应仅因预算未用完继续跑。
新实验的 `.best` 只代表本轮最佳，不会自动继承原V6的71%成绩，原文件仍保留。
目标是稳定达到75%以上，并用新种子验证；新增50万步不是收敛保证。
旧位置模型仍可直接可视化：

```bash
.venv/bin/python Scripts/visualize_agent.py \
  --model-path runs/RM75-v0__RM75-v4-high-success__1__1791429044/RM75-v4-high-success.best.cleanrl_model \
  --episodes 5
```

以下为早期项目说明，涉及旧奖励、旧环境 ID 的参数不代表上述当前训练入口。

本次微调只修改启动脚本与本说明，通过Shell语法、实际CLI参数解析及前后参数
差异检查；确认环境、算法、可视化和两份最佳模型未改动，没有重跑物理训练测试。
此前V6修改通过32项实际到达、方向、混合起点和旧位置模型回归测试；
8 个固定种子均得到有效到达起点，另做64个学生步的双环境数据流、有限梯度
反向传播和临时 checkpoint 恢复检查，没有执行 optimizer.step。
临时产物只在 `/tmp`，未启动正式训练、创建正式模型或上传 W&B；
这些检查只验证实现，不代表正式训练已收敛。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/MuJoCo-3.0+-green" />
  <img src="https://img.shields.io/badge/Gymnasium-1.0+-orange" />
  <img src="https://img.shields.io/badge/W%26B-Tracking-yellow?logo=weightsandbiases" />
</p>

**使用 PPO 和 TD3 训练机械臂，使其能够在 MuJoCo 仿真环境中自主到达随机目标位置。项目现同时保留六轴 UR5e 环境和七轴 RM75 环境。**

本项目实现、调优并对比了两种用于连续控制的先进深度强化学习算法：**近端策略优化（PPO）**和**双延迟深度确定性策略梯度（TD3）**。从仿真环境、奖励函数、训练循环到评估流程，均围绕智能机器人研究场景进行构建和迭代。

---

## 仿真环境

<p align="center">
  <img src="docs/mujoco_ur5_setup.png" alt="UR5 MuJoCo 仿真环境" width="720" />
</p>

> MuJoCo 物理仿真器中的 UR5e 机械臂。绿色球体表示智能体需要到达的目标位置，两侧工作台用于模拟真实的桌面操作场景。

---

## 演示

[观看演示视频](https://github.com/niccolasparra/UR5-Reinforcement-Learning/issues/1#issue-4107960304)

> 训练完成的智能体实时控制 UR5 机械臂到达目标位姿。

---

## 项目意义

大多数强化学习教程止步于 CartPole 或 Atari 游戏，而本项目处理的是一个真实的机器人控制问题：使用带有关节限制、重力、碰撞检测和塑形奖励的高精度物理仿真器，控制六自由度机械臂。

项目的核心问题是：**如何仅通过试错，让神经网络学会平滑、无碰撞地运动到任意三维目标位置？**

---

## 技术架构

### 仿真环境（`env.py`）

该项目基于 MuJoCo 构建了一个自定义 Gymnasium 环境，用于模拟具有完整刚体动力学特性的 UR5e 机械臂。

| 组件 | 说明 |
|---|---|
| **观测空间** | 19 维向量：关节位置（6）+ 关节速度（6）+ 末端执行器位置（3）+ 目标位置（3）+ 碰撞标志（1） |
| **动作空间** | 6 维连续动作：各关节每步的变化量限制为 ±0.05 rad |
| **物理仿真** | 每个动作执行 10 个 MuJoCo 子步；使用带重力补偿的 PD 位置控制器（Kp=2000，Kd=400） |
| **回合设置** | 每回合最多 1024 步；末端执行器到目标的距离小于 5 cm 或发生碰撞时终止 |
| **目标采样** | 每次重置时，在机械臂可达工作空间内随机生成目标位置 |

### 奖励函数设计

奖励函数是强化学习系统中最关键的设计之一。本项目经过多次迭代，在**学习速度**、**轨迹平滑度**和**碰撞规避**之间取得平衡：

```text
R(s, a, s') = 进度奖励 + 距离惩罚 + 姿态奖励 + 碰撞惩罚 + 成功奖励 + 时间惩罚
```

| 组成部分 | 公式 | 作用 |
|---|---|---|
| **进度奖励** | `10.0 × (d_prev - d) / max(d, 1e-6)` | 奖励机械臂逐步接近目标 |
| **距离惩罚** | `-0.5 × d` | 持续推动智能体缩短与目标之间的距离 |
| **碰撞惩罚** | 发生接触时为 `-10.0` | 对不安全轨迹施加明确惩罚 |
| **成功奖励** | 当 `d < 0.05 m` 时为 `+100.0` | 提供清晰的终止奖励信号 |
| **时间惩罚** | 每步 `-0.01` | 鼓励更高效地完成任务 |
| **姿态奖励** | `w × (alignment)^2` | 可选：使末端执行器坐标轴与目标方向对齐 |

**关键设计思路：** 使用 `max(d, 1e-6)` 对进度奖励进行归一化，可以形成相对改进信号。智能体越接近目标，相同幅度的小改进获得的奖励比例越高，有助于提高精细定位能力。

### 算法

#### PPO（近端策略优化）——同策略算法

PPO 使用 Actor-Critic 架构，并通过裁剪后的代理目标函数限制策略更新幅度，从而避免灾难性遗忘。

- **Actor**：两层 MLP，每层 64 个单元，使用 Tanh 激活函数，并学习随机高斯策略的对数标准差
- **Critic**：独立的两层 MLP，每层 64 个单元，用于估计状态价值 V(s)
- **主要超参数**：`clip_coef=0.2`、`ent_coef=0.01`、`gae_lambda=0.95`、`num_steps=1024`、`update_epochs=10`
- 所有网络层均使用**正交初始化**，提高训练初期的稳定性

#### TD3（双延迟深度确定性策略梯度）——异策略算法

TD3 是一种确定性 Actor-Critic 算法，通过以下三种机制缓解连续控制中的 Q 值高估问题：

- **Actor**：两层 MLP，每层 256 个单元，使用 ReLU 激活函数，通过 Tanh 输出确定性动作
- **双 Critic**：使用两个独立的 Q 网络，并采用两者中的较小值计算目标，从而降低高估偏差
- **延迟策略更新**：Critic 每更新两次，Actor 更新一次，避免策略因价值估计噪声而发生剧烈振荡
- **目标策略平滑**：向目标动作加入经过裁剪的高斯噪声，使价值函数更加平滑
- **主要超参数**：`buffer_size=1M`、`batch_size=256`、`tau=0.005`、`exploration_noise=0.1`、`policy_noise=0.2`

### 逆运动学模块（`ik.py`、`ik_solver.py`）

项目包含一个自定义逆运动学求解器，用于在每次回合重置时，将机械臂初始化在起点附近。这样可以缩短智能体初始状态与任务起点之间的距离，从而提高样本效率。

---

## 项目结构

```text
├── env.py                         # 自定义 Gymnasium 环境（UR5 + MuJoCo）
├── env_rm75.py                    # 七轴 RM75 Gymnasium 环境
├── __init__.py                    # 注册 UR5-v0 与 RM75-v0
├── ik.py                          # 逆运动学实现
├── ik_solver.py                   # 逆运动学求解工具
│
├── Algorithms/
│   ├── ppo_continuous_action.py   # PPO 实现（基于 CleanRL）
│   └── td3_ur5.py                 # 针对 UR5 调整的 TD3 实现
│
├── Scripts/
│   ├── visualize_agent.py         # 可视化已训练智能体的行为
│   └── record_hd.py               # 录制高清评估视频
│
├── assets/                        # UR5e MuJoCo 模型（MJCF + OBJ 网格）
│   ├── scene.xml                  # 世界场景定义
│   ├── ur5e.xml                   # UR5e 机器人模型
│   └── assets/                    # 三维网格文件（.obj）
├── assets/rm75/                   # RM75 URDF、MJCF 与 STL 网格
├── tools/convert_rm75_urdf.py     # 从 ROS 工作区刷新 RM75 导入产物
├── tests/                         # RM75 模型、环境和 PPO 自动测试
│
├── requirements.txt               # Python 依赖项
└── LICENSE
```

---

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- 推荐使用 GPU 进行训练；也可以使用 CPU，但速度较慢
- macOS 上进行 MuJoCo 渲染时需要 GLFW：`brew install glfw`

### 安装

```bash
git clone https://github.com/niccolasparra/UR5-Reinforcement-Learning.git
cd UR5-Reinforcement-Learning

# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 训练

**训练 PPO：**

```bash
python Algorithms/ppo_continuous_action.py \
    --total-timesteps 1000000 \
    --num-envs 1 \
    --learning-rate 3e-4 \
    --track
```

**训练 TD3：**

```bash
python Algorithms/td3_ur5.py \
    --total-timesteps 500000 \
    --learning-rate 3e-4 \
    --track
```

训练过程中，回合累计奖励、成功率和损失等指标会实时记录到 **Weights & Biases** 和 **TensorBoard**。

### 评估

```bash
# 可视化训练完成的智能体
python Scripts/visualize_agent.py

# 录制评估视频
python Scripts/record_hd.py
```

### 交互式查看环境

```bash
# 随机目标位置
python env.py --interactive --target-mode random

# 使用指定的固定目标位置
python env.py --interactive --target-mode fixed --target-pos -0.4 0.0 0.2905

# 不设置目标位置
python env.py --interactive --target-mode none
```

`--interactive` 用于开启交互式窗口，`--target-mode` 用于选择目标生成方式，两个参数可以同时使用。

---

## RM75 迁移、验证与训练

这一节是 RM75 迁移后的实际操作说明。前面的环境参数和实验结果主要描述原始 UR5e；RM75 应以本节和 `env_rm75.py` 为准。

### 当前 RM75 接口

| 项目 | 当前实现 |
|---|---|
| Gymnasium 环境 ID | `RM75-v0`（随机位置）和 `RM75Approach-v0`（随机位置 + 切入方向）；同时保留 `UR5-v0` |
| 自由度与执行器 | 7 个转动关节、7 个位置执行器 |
| 观测空间 | `RM75-v0` 为 28 维；`RM75Approach-v0` 再加入当前末端方向 3 维和目标方向 3 维，共 34 维 |
| 策略动作 | 7 维归一化动作，每一维都在 `[-1, 1]` |
| 实际关节增量 | `joint_delta = 0.02 × action`，即每个环境步最多改变 `±0.02 rad` |
| 仿真步长 | 物理步长 `0.002 s`，每个环境步 10 个物理子步，即 `0.02 s` |
| 目标区域 | `x ∈ [-0.50, -0.35] m`、`y ∈ [-0.30, 0.30] m`、`z = 0.2905 m` |
| 碰撞终止 | 运动连杆与地面/桌面碰撞，或非相邻连杆发生自碰撞；固定底座与地面的安装接触不算碰撞 |
| 失败终止 | 回合达到 1024 步且未成功、未碰撞时施加 `-25` 超时惩罚 |
| 探索策略 | 动作标准差前 25% 保持 `0.30`，随后线性衰减到 `0.12`；`ent_coef=0.0001` |

`RM75-v0` 只要求末端到达随机位置；`RM75Approach-v0` 同时要求末端到达随机位置并对准随机切入方向。两者都没有加入夹爪闭合、物体接触、抓取保持或物体抬升判据，所以训练成功不等于已经完成随机物体抓取。

### 文件关系与模型更新规则

- `assets/rm75/RM75.urdf`：从 ROS 工作区复制并把 `package://` 网格路径改成本地相对路径的 URDF。
- `assets/rm75/rm75_imported.xml`：MuJoCo 从 URDF 自动导出的参考 MJCF，没有执行器。
- `assets/rm75/rm75.xml`：人工审核后的训练模型，包含执行器、关节限位、力矩限位、临时 TCP 和 home keyframe。
- `assets/rm75/scene.xml`：静态灯光和地面。两张训练桌子由 `env_rm75.py` 加入，以便用 `reset(seed=...)` 控制桌面摩擦力。

上游 ROS URDF 或 STL 改变后，在项目根目录运行：

```bash
.venv/bin/python tools/convert_rm75_urdf.py
```

该命令会刷新本地 `RM75.urdf`、八个 STL 和 `rm75_imported.xml`，但不会覆盖审核后的 `rm75.xml` 或 `scene.xml`。不要直接把自动导出的文件当训练模型；应先比较 `rm75_imported.xml` 与 `rm75.xml`，只把确认后的运动学、惯量或网格变化移植到 `rm75.xml`，并保留执行器、site、关键帧和物理参数。

### TCP 与实机边界

当前 `ee_site` 固定在 `link_7` 原点，是尚未安装真实工具时的临时 TCP。自动测试已经确认它确实属于 `link_7`，但这不能证明它等于真实法兰或夹爪工具中心。取得实机 `link_7 → TCP` 的平移和姿态后，应修改 `assets/rm75/rm75.xml` 中 `ee_site` 的 `pos` 与 `quat`，然后重新运行模型、IK、目标区域和稳定性测试。不要移动目标坐标来补偿错误 TCP。

同样，URDF 能确定仿真关节轴和顺序，但“滑块正方向是否与真实控制器正方向一致”仍必须通过实机低速点动逐轴核对。仿真增益不能直接复制到真实机械臂。

### 自动验证

当前虚拟环境没有安装 pytest，因此使用 Python 标准库的 `unittest`：

```bash
RM75_DEBUG=0 .venv/bin/python -m unittest discover -s tests -v
```

运行 Gymnasium API 检查：

```bash
RM75_DEBUG=0 .venv/bin/python - <<'PY'
from gymnasium.utils.env_checker import check_env
from env_rm75 import RM75P2PEnv

env = RM75P2PEnv(render_mode=None, seed=0, target_mode="random")
check_env(env, skip_render_check=True)
env.close()
print("RM75 check_env: PASS")
PY
```

测试覆盖七轴模型与执行器映射、URDF 限位和力矩范围、归一化动作、执行器跟踪误差、受 seed 控制的桌面摩擦、地面/桌面碰撞、自碰撞、超时惩罚、1 秒零动作保持、目标盒四角与中心 IK，以及 PPO 的 28 输入/7 输出接口。

2026-09-30 在当前机器上保留完整 STL 碰撞网格运行 1000 个无渲染环境步，用时约 `0.212 s`（约 `4709 env steps/s`），碰撞次数为 0，最大关节漂移约 `0.0348 rad`。性能和稳定性足够，因此没有增加近似盒体或胶囊碰撞代理。该速度是本机测量值，不是其他硬件上的保证。

### 可视化 RM75 环境

```bash
RM75_DEBUG=1 .venv/bin/python env_rm75.py \
  --interactive \
  --target-mode random
```

关闭 MuJoCo 窗口即可退出。终端中的第一次 `Ctrl+C` 用于中断前台程序；不要再用 `Ctrl+Z`，因为 `Ctrl+Z` 只是把进程挂起到后台，并没有正常结束它。

### PPO 冒烟测试与正式训练

先做一次 8 步 CPU 冒烟测试，确认完整训练链路可执行：

```bash
RM75_DEBUG=0 .venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75-v0 \
  --total-timesteps 8 \
  --num-steps 8 \
  --num-minibatches 1 \
  --update-epochs 1 \
  --no-track \
  --no-save-model \
  --no-cuda
```

正式训练示例：

```bash
RM75_DEBUG=0 .venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75-v0 \
  --total-timesteps 1000000 \
  --num-envs 1 \
  --learning-rate 1e-4 \
  --track
```

如不使用 W&B，把 `--track` 改成 `--no-track`。模型在训练正常结束且 `save_model=True` 时保存到 `runs/<run_name>/ppo_continuous_action.cleanrl_model`。训练脚本现在同时支持旧模型微调初始化和新格式 checkpoint 断点续训，二者的含义与用法见下一节。

UR5 检查点不能直接加载到 RM75 PPO 智能体：UR5 使用 19 维观测和 6 维动作，当前 RM75 使用 28 维观测和 7 维动作。此前 100 万步 RM75 模型使用旧的 21 维观测，也不能加载到当前 28 维 Agent 中。输入层形状和控制状态定义已经改变，必须重新训练；原模型只适合配合旧版环境进行历史评估。

### 低成功率结果后的 V3 训练修正

V2 在约 108 万步内只有 1 个成功回合。策略一度接近目标，之后又退化；主要问题是成功奖励样本过于稀疏，同时探索标准差从训练开始就下降。当前 V3 做了以下定向修正：

- `dist_weight=0.03`，保留轻量距离约束，避免它在长回合内压过真正的成功信号。
- 只对既未成功也未碰撞的回合超时施加 `-10`；成功或碰撞不会重复计算超时惩罚。
- 每回合首次进入 20 cm 范围奖励 `+5`，首次进入 10 cm 范围奖励 `+15`。奖励只发一次，因此来回跨越阈值不能反复获利；一步跨过两个阈值时累计奖励 `+20`。
- 归一化动作标准差前 25% 训练保持 `0.30`，之后线性下降到 `0.12`，让早期有足够机会采到成功轨迹，同时在后期降低随机动作。
- `ent_coef` 从 `0.001` 降为 `0.0001`，降低策略被持续高随机性拉回的程度。
- 保留累计位置目标 `q_target`，同时把 `q_target-q` 加入观测。相比改成 `q+delta`，这种方式既消除了隐藏控制状态，又保留位置执行器抵抗重力和跟踪未完成目标的能力。

本轮没有修改学习率衰减、1024 步 horizon、checkpoint 保存方式和随机目标课程。

### RM75-v3 微调与断点续训

训练脚本提供两个用途不同的参数：

- `--init-from`：只加载 Agent 网络权重，创建新的优化器、新的学习率进度和新的运行目录。用于在已完成模型上继续优化，也兼容旧的 `.cleanrl_model`。
- `--resume-from`：加载完整 checkpoint，恢复 Agent、Adam 优化器、全局步数、PPO 迭代号、学习率/探索调度参数、随机数状态和 W&B run id。用于恢复一次尚未完成的训练。

现有 `runs/RM75-v3/ppo_continuous_action.cleanrl_model` 是旧的纯权重格式，因此第一次微调必须使用 `--init-from`：

```bash
RM75_DEBUG=0 .venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75-v0 \
  --init-from runs/RM75-v3/ppo_continuous_action.cleanrl_model \
  --target-sampling hard_mixture \
  --total-timesteps 500000 \
  --learning-rate 2e-5 \
  --initial-action-std 0.12 \
  --final-action-std 0.08 \
  --action-std-hold-fraction 0.25 \
  --checkpoint-interval 100000 \
  --track
```

`hard_mixture` 不会缩小原目标范围。每次 reset 的采样来源为：50% 原完整区域、25% 负 Y 困难条带、25% 远 X 困难条带。TensorBoard/W&B 会增加以下分区曲线：

```text
charts/region_success_rate/uniform
charts/region_success_rate/hard_negative_y
charts/region_success_rate/hard_far_x
```

训练期间每跨过 `checkpoint_interval` 个环境步，会在一个完整 PPO 迭代结束后原子更新：

```text
runs/<run_name>/checkpoints/checkpoint_latest.pt
```

安全停止时只按一次 `Ctrl+C`。脚本会完成当前 PPO 迭代、保存 `checkpoint_latest.pt`，然后退出。不要再按 `Ctrl+Z`；`Ctrl+Z` 只是挂起进程，会阻止正常清理和保存流程。

从中断点恢复时只需要 checkpoint 路径，原训练的重要参数会从 checkpoint 自动恢复：

```bash
.venv/bin/python Algorithms/ppo_continuous_action.py \
  --resume-from runs/<run_name>/checkpoints/checkpoint_latest.pt
```

如果原运行使用 CPU，可以在恢复命令后增加 `--no-cuda`。如果训练已经正常完成，而你还想额外训练一段新的步数，应使用新生成的 `.cleanrl_model` 再次执行 `--init-from`，而不是对已经完成计划的 checkpoint 使用 `--resume-from`。

运行目录可以整体改名；恢复时脚本会以你传入的 checkpoint 当前所在目录继续写日志和模型。不要只移动 `checkpoint_latest.pt` 而丢失它所属运行目录中的其他日志文件。

两个文件的用途如下：

| 文件 | 包含内容 | 用途 |
|---|---|---|
| `ppo_continuous_action.cleanrl_model` | Agent 网络权重 | 可视化、评估、开始新的微调 |
| `checkpoints/checkpoint_latest.pt` | Agent、优化器、步数、调度、随机状态、运行信息 | 恢复被中断的同一次训练 |

### RM75-v4 从头训练高成功率版本

V4 延续 V3 的全工作区均匀目标采样和从随机权重开始训练。碰撞仍会终止回合，但碰撞惩罚由 75 降为 45，允许策略在学习初期尝试更直接的路径。训练目标是固定验证集成功率至少 82%，同时碰撞率不超过 10%。

直接运行：

```bash
bash Scripts/train_rm75_v4.sh
```

该命令使用 250 万环境步、`1e-4` 学习率和 `uniform` 目标采样。动作标准差前 20% 保持 0.30，之后下降到 0.10。每 20 万步使用从 20000 开始的 100 个固定种子执行确定性评估，每 10 万步保存断点。

运行目录会生成：

| 文件 | 含义 |
|---|---|
| `RM75-v4-high-success.cleanrl_model` | 正常结束时的最终模型 |
| `RM75-v4-high-success.best.cleanrl_model` | 固定验证集中成功率最高且碰撞率不超过 10% 的模型 |
| `best_metrics.json` | 最佳模型对应的成功率、碰撞率、最终距离、步数和训练步数 |
| `checkpoints/checkpoint_latest.pt` | 可恢复 Agent、优化器、训练进度和最佳验证结果的完整断点 |

最佳模型按成功率优先选择；成功率相同时，先选碰撞率更低的模型，再选平均最终距离更小的模型。因此可以接受少量碰撞换取更高成功率，但不会把碰撞率超过 10% 的模型选为最佳结果。

训练中断时按一次 `Ctrl+C`。恢复命令为：

```bash
.venv/bin/python Algorithms/ppo_continuous_action.py \
  --resume-from runs/<V4运行目录>/checkpoints/checkpoint_latest.pt
```

W&B 和 TensorBoard 中新增 `eval/success_rate`、`eval/collision_rate`、`eval/mean_final_distance`、`eval/mean_episode_length` 和 `eval/meets_acceptance`。比较模型时应优先查看这些固定验证指标，而不是训练中的最近 100 回合曲线。

### RM75 随机位置 + 随机切入方向最终训练

该任务使用 `RM75Approach-v0`。最终严格目标始终是位置误差不超过 `0.05 m`、方向误差不超过 `15°`，随机方向覆盖向下圆锥内 `0°～45°` 倾角和 `-180°～180°` 方位角。训练从保留的 V5 最佳模型开始，不修改奖励权重，而是用更平缓的六阶段课程减少任务突变：

| 阶段 | 最大倾角 | 位置成功阈值 | 方向成功阈值 | 作用 |
|---|---:|---:|---:|---|
| 1 | 15° | 0.15 m | 45° | 小范围入门 |
| 2 | 20° | 0.12 m | 40° | 扩大方向 |
| 3 | 25° | 0.10 m | 35° | V5 已掌握分布 |
| 4 | 25° | 0.085 m | 30° | 只提高精度，不扩范围 |
| 5 | 35° | 0.070 m | 22° | 中等范围桥接 |
| 6 | 45° | 0.050 m | 15° | 最终严格任务 |

Stage 4 和 Stage 5 都要在本阶段至少训练 30 万步，并在最近 200 个完整回合中达到 60% 的本阶段成功率才晋级。课程切换不会重置已经学到的动作标准差；学习率按整次 200 万步训练全局从 `1e-5` 单调衰减到 `2e-6`，不会因晋级重新升高。

正式启动只使用这一条命令：

```bash
bash Scripts/train_rm75_approach_final.sh
```

训练曲线中的 `charts/success_rate` 是当前阶段的宽松成功率，适合判断能否晋级；`charts/strict_success_rate` 和固定评估中的 `eval/success_rate` 才代表最终 `5 cm/15°` 目标。模型选择应优先看固定种子评估，而不是某一小段训练回合的短期波动。

---

## 主要结果

### PPO 与 TD3 成功率对比

<p align="center">
  <img src="docs/success_rate_ppo_vs_td3.png" alt="PPO 与 TD3 成功率对比" width="720" />
</p>

> 使用 Weights & Biases 记录。PPO（蓝色）的成功率超过 85%，且收敛快速、单调；TD3（橙色）收敛速度较慢，但在相同训练周期内逐步提升至约 67%。

| 指标 | PPO | TD3 |
|---|---|---|
| **最高成功率** | **超过 85%** | 约 67% |
| **收敛速度** | 较快——约 4000 个回合时达到 80% | 较慢——约 8000 个回合时达到 67% |
| **训练稳定性** | 高——曲线平滑且基本单调上升 | 中等——上升过程较缓慢 |
| **轨迹质量** | 平滑、保守 | 直接、高效 |
| **样本效率** | 较低（同策略） | 较高（使用异策略经验回放） |

**结论：** 在本任务中，PPO 的表现明显优于 TD3，其成功率超过 85%，并且收敛快速、稳定。TD3 上升较慢是可以预期的，因为异策略算法需要先向经验回放缓冲区中积累数据，并且探索噪声参数可能仍需针对当前动作空间进一步调优。两种算法都能够稳定学习并持续改进，说明环境和奖励函数可以提供有效的学习信号，MDP 建模总体合理。

---

## 技术栈

- **仿真系统**：[MuJoCo](https://mujoco.org/)——机器人研究用物理引擎
- **强化学习框架**：[CleanRL](https://github.com/vwxyzjn/cleanrl)——高质量单文件强化学习算法实现
- **环境 API**：[Gymnasium](https://gymnasium.farama.org/)——标准强化学习环境接口
- **深度学习框架**：[PyTorch](https://pytorch.org/)——神经网络训练
- **实验跟踪**：[Weights & Biases](https://wandb.ai/) + TensorBoard
- **机器人模型**：[UR5e](https://www.universal-robots.com/products/ur5-robot/)——来自 MuJoCo Menagerie

---

## 总结与后续工作

实验结果表明，该系统具有良好的整体架构：PPO 和 TD3 都能从零开始学习有效策略，奖励函数也能够提供足够明确的学习信号，使智能体在数十万步训练中持续进步。PPO 超过 85% 的成功率说明，在没有演示数据或轨迹规划的情况下，六自由度机械臂也能够仅通过强化学习可靠地完成点到点运动任务。

项目仍有进一步提升空间，优先考虑以下三个方向：

1. **奖励归一化。** 当前各奖励分量的尺度不同，例如进度奖励约为 `[-10, 10]`、成功奖励为 `+100`、时间惩罚为 `-0.01`。将它们归一化到接近的范围，或使用滑动奖励标准化层，可能为优化器提供更平滑的损失曲面，并加快 PPO 和 TD3 的收敛。

2. **降低 TD3 的探索噪声。** TD3 收敛较慢，说明 `exploration_noise=0.1` 对于动作限制较小（±0.05 rad/step）的 UR5 环境而言可能过强。训练过程中逐步衰减噪声，或改用参数空间噪声，可能显著提高 TD3 的样本效率并缩小与 PPO 的性能差距。

3. **调整折扣因子 γ。** 两种算法均使用 `γ=0.99`，对应较长的有效规划时域（约 100 步）。对于通常能在更少步数内完成的到达任务，降低 γ（例如 0.95～0.98）可能使价值估计更加稳定，并减少 Critic 预测方差；这一点对依赖 Q 值准确性的 TD3 尤其重要。

除超参数调优外，后续还可以加入**域随机化**（随机改变物体质量、摩擦力和关节阻尼），用于仿真到现实迁移；也可以引入**课程学习**，逐步增加目标难度。

---

## 参考文献

1. Schulman 等，*Proximal Policy Optimization Algorithms*（2017）——[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
2. Fujimoto 等，*Addressing Function Approximation Error in Actor-Critic Methods*（2018）——[arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
3. Huang 等，*CleanRL: High-quality Single-file Implementations*（2022）——[JMLR](https://jmlr.org/papers/v23/21-1342.html)
