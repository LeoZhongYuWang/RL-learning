# RM75 V4 环境与 PPO 演变报告

## 1. 结论先行

本次还原的目标版本是生成以下模型的点到点训练栈：

`runs/RM75-v0__RM75-v4-high-success__1__1791429044/RM75-v4-high-success.best.cleanrl_model`

这不是姿态任务，也不是抓取任务。它学习的是：给定 RM75 当前状态和随机目标点，输出 7 个关节的归一化角度增量，使末端进入目标点 5 cm 范围，同时避免碰撞。

V4 没有改变 PPO 的核心 Actor-Critic 结构，Actor 和 Critic 仍各为两层 64 单元的 `Tanh` MLP。代码从原 PPO 的约 401 行增长到 1314 行，主要增长来自：动作概率分布修复、RM75 环境接入、指标记录、固定种子评估、最佳模型选择、checkpoint/断点续训和安全停止，而不是盲目增加网络规模。

V4 最佳模型的固定种子评估结果为：

- 成功率：0.84
- 碰撞率：0.00
- 平均最终距离：0.0658 m
- 平均回合长度：311.16 step
- 最佳点：global step 2,000,896

训练结束时最近 100 回合成功率为 0.88；最终一次固定种子评估为 0.83。两者统计口径不同，因此不矛盾，模型选择应以固定种子评估为准。

## 2. 还原依据与可信度

本次不是凭记忆把参数“改得像 V4”，而是组合了三个可核对来源：

1. W&B 本地运行 `run-20261008_111046-57ie6r8n` 保存的完整 PPO 源码快照。还原后的 PPO 与该快照 SHA256 完全一致。
2. 当时会话记录里的 `env_rm75.py` 逐行快照，再叠加 V4 开始前唯一的环境补丁：增加 `collision_penalty` 构造参数。
3. V4 运行目录中的 `best_metrics.json`、checkpoint 参数和 `Scripts/train_rm75_v4.sh`。

因此，`env_rm75.py` 和 PPO 是模型实际训练时的代码契约，不是把现在的姿态环境仅仅关闭几个开关。

## 3. 文件关系

```text
Scripts/train_rm75_v4.sh
        │ 传入超参数
        ▼
Algorithms/ppo_continuous_action.py
        │ gym.make("RM75-v0")
        ▼
__init__.py / PPO 内部注册
        │
        ▼
env_rm75.py ──加载──> assets/rm75/scene.xml + rm75.xml + meshes
        │
        ├── 28 维 observation ──> Actor / Critic
        └── 7 维归一化 action <── Actor
```

配套的 `Scripts/visualize_agent.py` 不参与训练，只负责读取权重并进行确定性或随机评估。本次保留了一个小的兼容增强：根据权重维度自动选择 UR5-v0（19/6）或 RM75-v0（28/7），这样还原 V4 后 UR5-v3 仍然可以可视化。

## 4. 从原始 `env.py` 到 RM75 V4 环境

### 4.1 机械臂迁移必须做的修改

| 项目 | 原始 UR5 | RM75 V4 | 修改性质 | 原因 |
|---|---:|---:|---|---|
| 关节/执行器 | 6 / 6 | 7 / 7 | 模型迁移 | RM75 有 7 个受控关节 |
| 动作维度 | 6 | 7 | 模型迁移 | 每个关节一个动作增量 |
| MJCF 场景 | `assets/scene.xml` | `assets/rm75/scene.xml` | 模型迁移 | 几何、关节限位、执行器和 TCP 都不同 |
| 末端 site | 容错查找/临时创建 | 强制 `ee_site` | 代码契约 | TCP 定义错误会让 IK、距离和奖励全部失真 |
| 关节顺序 | 默认相信 XML | 显式验证 `joint_1...joint_7` | 漏洞修复 | 防止动作第 i 维控制错误关节 |
| 执行器映射 | 未严格核验 | 验证执行器名及 joint 映射 | 漏洞修复 | 防止模型能加载但控制含义错位 |

这些修改与 W&B 曲线无关，是机械臂模型更换后的必要适配和启动时自检。

### 4.2 动作契约：从物理小范围动作改成归一化动作

原始环境把动作空间直接定义为每步 `[-0.05, 0.05] rad`。原始 Actor 却从标准差约为 1 的无界高斯采样，随后依赖环境/包装器裁剪。这样 PPO 计算概率时使用的是裁剪前动作，而 MuJoCo 接收的是裁剪后动作，概率模型与真实执行不一致，并且大量动作会挤在边界。

V4 改为：

```python
action_space = [-1, 1]^7
joint_delta = 0.02 * normalized_action
q_target = clip(q_target + joint_delta, joint_limits)
```

这项修改首先是代码审查发现的动作分布漏洞，同时也对应 W&B 中需要观察的 `charts/action_saturation_rate`。V4 结束时该指标为 0，说明策略不是靠长期撞动作边界工作。

`0.02` 的含义是单个环境 step 内每个关节目标角最多改变 0.02 rad，约 1.15°。策略空间仍是安全、统一的 `[-1,1]`，真正的物理幅度由环境缩放控制。

### 4.3 观测从 19 维变成 28 维

原始 UR5 观测：

```text
q(6) + qd(6) + ee(3) + goal(3) + collision(1) = 19
```

RM75 V4 观测：

```text
q(7) + qd(7) + (q_target - q)(7) + ee(3) + goal(3) + collision(1) = 28
```

新增 `q_target - q` 不是为了堆维度，而是消除控制器的部分可观测性。动作累加到伺服目标 `q_target`，真实关节 `q` 经过物理子步才逐渐跟上。如果策略只看到 `q`，无法区分“目标还没追上”与“没有发送目标”。加入跟踪误差后，Actor 和 Critic能看到控制器内部积累状态。

### 4.4 reset、IK 和仿真效率

原始代码每个 episode 重新构建并编译 MJCF。V4 只在构造环境时编译一次，reset 时调用 `mj_resetData`，再用 seed 控制：

- 初始关节噪声；
- IK 初值；
- 随机目标点；
- 桌面摩擦系数。

这项修改主要是工程缺陷修复：场景拓扑并没有每回合变化，反复编译浪费时间，而且容易破坏严格复现。它会直接改善 `charts/SPS`，但机械臂噪声和 IK 仍然每回合执行，不依赖重新编译模型。

reset 中先定义笛卡尔起点，再用带噪声的 `q0` 作为 IK 初值，原因是 IK 必须知道要解到哪里。IK 负责让不同关节噪声下的末端仍从规定起点附近开始；噪声负责增强对初始关节细微变化的鲁棒性。

### 4.5 碰撞检测

原始环境主要判断机器人与桌面接触，未完整处理 RM75 自碰撞。V4 构建机器人碰撞 geom 集合，并区分：

- 机器人与桌面；
- 非底座机器人与地面；
- 非相邻连杆之间的自碰撞；
- 相邻连杆结构接触和底座-地面接触不计为失败。

这是安全语义和模型迁移修复，不是因为某条 loss 曲线才添加。W&B 的 `charts/collision_rate` 用来检验实现后的策略行为，而不能代替碰撞分类本身的代码测试。

## 5. 奖励如何演变

### 5.1 原始奖励的主要问题

原始进度奖励近似为：

```python
progress_scale * (prev_dist - dist) / max(dist, 1e-6)
```

分母使用当前距离会让同样的 1 mm 改善在靠近目标时被无限放大；来回摆动时正负收益也不再严格抵消。这是代码数学性质上的问题，不需要等待 W&B 才能判断。

原始绝对距离惩罚权重为 0.5，逐步累计后可能压过稀疏成功奖励，并诱导策略通过碰撞尽快结束负回报回合。早期训练中出现过低成功、高碰撞或长时间无成功，说明奖励尺度确实需要调整。

### 5.2 V4 点任务奖励

V4 每一步的核心奖励是：

```text
50 × (prev_dist - dist)
- 0.03 × dist
- 0.01
+ 一次性距离里程碑
+ 100（成功）
- collision_penalty（碰撞）
- 10（无成功、无碰撞但超时）
```

里程碑：

- 首次进入 20 cm：+5；
- 首次进入 10 cm：+15；
- 每个里程碑每回合只发一次。

其中 `50 × (prev_dist - dist)` 是势函数差分。向目标靠近获得正奖励，远离得到同等尺度负奖励；若先靠近再退回原距离，两段进度奖励相加约为 0，因此不能靠来回摆动刷进度奖励。

绝对距离惩罚没有完全删除，因为它提供“离目标近比离目标远更好”的持续偏好；权重降到 0.03，是为了不让它压倒进度和终点奖励。最高效到达主要由每步时间惩罚和成功后立即终止保证，而不是单靠大距离惩罚。

环境默认碰撞惩罚仍是 75，但 V4 启动脚本传入 45。原因是前期安全设置过强时，策略更容易学会保守等待；V4 的目标是允许少量风险换取更高成功率，同时用独立评估的碰撞率上限 10% 约束最佳模型。实际 V4 最佳模型碰撞率仍为 0。

## 6. 从原始 PPO 到 V4 PPO

### 6.1 直接运行与注册问题

原 PPO 使用 `import UR5` 触发注册，但仓库并不存在可导入的 `UR5` 包，导致 `ModuleNotFoundError`。V4 根据脚本路径把项目根目录加入 `sys.path`，直接导入 `env.py` 和 `env_rm75.py`，并幂等注册 `UR5-v0`、`RM75-v0`。

这是明确的代码漏洞修复，与训练曲线无关。

### 6.2 Tanh-squashed Gaussian

原 PPO：无界 `Normal(mean, std)`，动作可能远超环境范围。

V4：先采样 raw action，再执行 `tanh(raw_action)`，并加入变量变换后的 log-prob 修正：

```text
log π(a) = log Normal(raw_action) - log(1 - tanh(raw_action)^2)
```

这样 PPO 优化的就是实际传给环境的动作。该修改是动作契约修复，也是成功训练的基础条件。

### 6.3 探索标准差计划

原始 `actor_logstd=0`，即每维标准差初始为 1；V4 默认从 0.30 开始，训练前一段保持探索，再线性降到较小值。V4 正式命令为：

```text
0.30 → 0.10，前 20% 保持 0.30
```

降低初始标准差会减少一开始的大幅随机关节运动和无意义碰撞；保留 20% 平台期避免过早失去探索；后期降低则让轨迹和终点更稳定。W&B 对应查看 `charts/action_std_ceiling`、`losses/entropy`、`charts/action_saturation_rate`、成功率和碰撞率。

### 6.4 PPO 超参数变化

| 参数 | 原始 PPO | V4 正式训练 | 目的/依据 |
|---|---:|---:|---|
| learning rate | 3e-4 | 1e-4，线性衰减 | 减少后期策略来回摆动和性能回退 |
| rollout steps | 1024 | 2048 | 每次更新包含更多完整回合信息 |
| minibatches | 32 | 16 | batch 2048 下 minibatch=128，梯度更稳定 |
| update epochs | 10 | 5 | 减少对同一批数据过度更新 |
| entropy coef | 0.01 | 0.0001 | 避免持续高探索压过精确到点 |
| target KL | None | 0.02 | 单轮策略变化过大时提前停止 epoch |
| total steps | 1,000,000 | 2,500,000 | RM75 V3 在较晚阶段才达到高成功率 |

这些属于“结合训练曲线做的稳定性调参”，重点参考 `old_approx_kl/approx_kl`、`clipfrac`、`entropy`、`value_loss`、`explained_variance`、成功率和碰撞率。不能把单个 loss 越小简单理解成任务越好，最终仍要看固定目标集成功率。

### 6.5 指标记录修复

原代码只累计从开始到当前的总成功率，早期失败会长期拖累曲线；而且 Gymnasium 版本变化后，episode 信息不一定出现在旧的 `final_info` 结构中，曾出现 W&B 没有 `episodic_return` 的情况。

V4 改为：

- 正确读取 Gymnasium 1.3 的 episode info；
- 记录最近 100 回合成功率和碰撞率；
- 记录单回合 return、length、success、collision、timeout、final distance；
- 记录区域成功率、动作饱和率、动作标准差和 PPO losses。

这是 API 兼容性修复加诊断能力建设。没有这些指标，很难判断是奖励、探索、碰撞还是价值网络出了问题。

### 6.6 固定种子评估与最佳模型

训练中的最近 100 回合会因随机目标难度和探索动作而波动，不能只看它选择最终模型。V4 每 200,000 step 在 seeds 20000 起的 100 个固定目标上，用确定性均值动作评估。

最佳模型排序规则：

1. 碰撞率必须不超过 10%；
2. 成功率更高优先；
3. 成功率相同则碰撞率更低优先；
4. 再相同则平均最终距离更小优先。

因此 `.best.cleanrl_model` 比训练结束时的 `.cleanrl_model` 更适合作为展示和后续基线。这个功能直接来自 W&B 中“成功率会先上升、之后回落或波动”的观察。

### 6.7 checkpoint、init-from 与 resume-from

V4 增加完整 checkpoint，保存：Agent、optimizer、global step、iteration、参数、随机数状态、W&B run id 和最佳评估数据。

- `--init-from MODEL`：只加载 Actor/Critic 权重，开始一个新的训练实验；学习率计划、步数和 optimizer 重新开始。适合微调或改变任务。
- `--resume-from checkpoint_latest.pt`：恢复同一次训练，包括 optimizer、进度、随机状态和 W&B run。适合中断后原样继续。

Ctrl+C 会先完成当前 PPO iteration，再保存 checkpoint。不要再用 Ctrl+Z 作为正常退出；Ctrl+Z 只是挂起进程，不保证保存，也可能让 GPU/进程继续占用资源。

## 7. 哪些修改是 W&B 驱动，哪些不是

| 修改 | 主要来源 | W&B 关联指标 |
|---|---|---|
| 6 轴改 7 轴、RM75 XML/TCP、关节映射校验 | 机械臂迁移/代码契约 | 无 |
| 修复 `import UR5` | 明确运行错误 | 无 |
| Tanh 动作与 log-prob 修正 | 代码漏洞 + 动作饱和诊断 | action saturation、collision、success |
| 进度奖励去掉 `/dist` | 奖励数学漏洞 | return、final distance、success |
| reset 不再重编译 | 性能/复现缺陷 | SPS |
| seed 控制摩擦和随机化 | 复现要求 | 固定种子 eval |
| 自碰撞分类 | 安全语义缺失 | collision rate |
| 距离惩罚降权、里程碑、timeout | 早期低成功/长回合 | episodic return/length、timeout、final distance |
| LR、epoch、entropy、std schedule | 曲线稳定性调参 | KL、clipfrac、entropy、value loss、success |
| uniform 目标采样 | 区域能力与整体成功率权衡 | region success rate |
| best model 固定评估 | 训练成功率波动 | eval/success_rate、eval/collision_rate |
| checkpoint/安全 Ctrl+C | 长训练工程需求 | 与曲线无直接关系 |

## 8. 关键 W&B 结果如何解释

本地保留的若干关键结果：

| 运行 | global step | 最近成功率 | 碰撞率 | 最近最终距离 | 说明 |
|---|---:|---:|---:|---:|---|
| 早期 RM75 | 999,424 | 0.14 | 0.14 | 0.2000 | 尚未形成稳定到点策略 |
| 另一早期 RM75 | 1,079,073 | 0.00 | 0.27 | 0.3517 | 探索/奖励失衡，明显退化 |
| RM75 V3 | 1,998,848 | 0.79 | 0.00 | 0.0486 | 点到点能力基本形成 |
| V4 训练末端 | 2,498,560 | 0.88 | 0.00 | 单个末回合 0.2506 | 最近成功率高，但单个最后回合可能超时 |
| V4 固定评估末端 | 2,498,560 | 0.83 | 0.00 | 平均 0.0674 | 可比较的最终评估 |
| V4 best | 2,000,896 | 0.84 | 0.00 | 平均 0.0658 | 最值得保留的 checkpoint |

注意：W&B summary 的 `charts/final_distance` 是最后一个被记录的 episode 值，不是整体平均；不能用它否定同一步的 100 回合 `eval/mean_final_distance`。

最有判断价值的图表优先级：

1. `eval/success_rate`、`eval/collision_rate`、`eval/mean_final_distance`：固定测试集上的最终能力。
2. `charts/success_rate`、`charts/collision_rate`、`charts/timeout`：训练分布最近表现和失败类型。
3. `charts/episodic_length`：成功更快通常会下降；若总是 1024，说明经常超时。
4. `charts/action_saturation_rate`、`charts/action_std_ceiling`、`losses/entropy`：动作是否总撞边界、探索是否过强或过早消失。
5. `losses/approx_kl`、`losses/clipfrac`：单次更新是否过猛。持续过大才需要降学习率/epoch；接近 0 也可能表示几乎没学。
6. `losses/explained_variance`、`losses/value_loss`：Critic 对回报的解释能力。EV 接近 1 通常较好，但成功率低时，高 EV 也可能只是准确预测“总是失败”。
7. `charts/SPS`：仿真吞吐，不代表策略质量。

## 9. 这套修改是否合理、是否可迁移

可直接迁移到其他连续控制项目的部分：

- 归一化动作与环境内物理缩放；
- Tanh-squashed Gaussian 及正确 log-prob；
- 控制器内部目标误差进入观测；
- 固定种子独立评估与 best checkpoint；
- 最近窗口指标而非从头累计指标；
- init-from 与 resume-from 分离；
- seed 控制所有随机源；
- 一次编译、reset 数据；
- 势函数差分奖励和一次性里程碑；
- 先用代码测试确认碰撞语义，再看 W&B 的碰撞率。

不能原样迁移的部分：

- 7 个关节、关节顺序、TCP site、碰撞 geom 集合；
- `0.02 rad` 动作缩放；
- 目标盒、起点和桌面几何；
- 奖励系数 50/0.03/45/100；
- 5 cm 成功阈值。

这些必须按新机器人动力学、控制频率、工作空间和任务容差重新标定。

## 10. V4 的边界与后续影响

恢复后：

- V4 点到点模型可以直接加载和训练；
- UR5-v3 可继续用兼容可视化脚本展示；
- 后来训练的 34 维 approach 模型和 20 维 pose 模型不能在这套 28 维环境中加载；
- 位姿任务代码已备份，不是永久删除；
- V4 仍不包含夹爪、物体、视觉传感器或抓取成功判定。

若以后重新做 pose，正确方法是从这个稳定的点任务基线建立独立环境版本，而不是继续把 pose 逻辑叠回 `RM75-v0` 并改变旧模型的 observation/action contract。

## 11. 使用命令

可视化 V4 best：

```bash
cd /home/zhongyu/wzy_master/UR5-Reinforcement-Learning
.venv/bin/python Scripts/visualize_agent.py \
  --model-path runs/RM75-v0__RM75-v4-high-success__1__1791429044/RM75-v4-high-success.best.cleanrl_model \
  --episodes 5
```

无界面评估 100 回合：

```bash
.venv/bin/python Scripts/visualize_agent.py \
  --model-path runs/RM75-v0__RM75-v4-high-success__1__1791429044/RM75-v4-high-success.best.cleanrl_model \
  --headless \
  --episodes 100 \
  --seed 20000
```

从头复现 V4 参数：

```bash
bash Scripts/train_rm75_v4.sh
```

从尚未完成原计划的 V4 checkpoint 原样恢复：

```bash
.venv/bin/python Algorithms/ppo_continuous_action.py \
  --resume-from runs/RM75-v0__RM75-v4-high-success__1__1791429044/checkpoints/checkpoint_latest.pt
```

注意：当前保留的 V4 checkpoint 已完成原计划的 2.5M step；执行上述命令只会恢复状态并结束，不会增加训练步数。这个精确 V4 版本会从 checkpoint 恢复原 `total_timesteps`，也没有后来添加的 `--additional-timesteps`，所以不能靠命令行覆盖步数来延长一个已完成的 run。若确实要追加训练，应使用 `--init-from ...best.cleanrl_model` 开始一个新的微调 run；如果只是复现或展示 V4，优先从头运行启动脚本或直接使用 best 模型。

## 12. 本次还原后的实际验证

- PPO SHA256 与 W&B 保存的 V4 源码快照一致：`d936f94e...61edf`。
- 13 个必要回归测试全部通过，覆盖 RM75 模型、28/7 契约、seed/reset、碰撞惩罚、V4 权重加载、UR5 动作契约和 best 排序。
- 用 V4 best 模型重新跑 seeds 20000–20099 的 100 回合确定性评估，结果精确复现 `best_metrics.json`：成功率 0.84、碰撞率 0、平均长度 311.2、平均最终距离 0.0658 m。
- 用相同可视化脚本加载 `runs/UR5-v3/ppo_continuous_action.cleanrl_model`，自动识别为 UR5-v0，并完成一个成功回合。
- 还原前的位姿代码保存在 `archive/post-v4-pose-2026-10-12/`，已有模型与 W&B 数据未删除。
