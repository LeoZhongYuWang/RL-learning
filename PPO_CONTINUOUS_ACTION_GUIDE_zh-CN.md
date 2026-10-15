# 当前 PPO 代码学习指南

更新日期：2026-10-13。对应等价精简后的 `Algorithms/ppo_continuous_action.py`（1452行），不是最初只有UR5的脚本，也不是已废弃的34/40维随机方向课程。最初导读原文保存在文档备份中。

## 1. PPO究竟学习什么

Actor和Critic都输入obs，两个网络参数独立，没有共享骨干：

```text
obs → Actor: Linear(D,64)→Tanh→Linear(64,64)→Tanh→Linear(64,A)
    → 高斯均值mu + 可学习logstd → tanh动作
obs → Critic: Linear(D,64)→Tanh→Linear(64,64)→Tanh→Linear(64,1)
    → 状态价值V(s)
```

UR5是D=19/A=6，RM75点任务D=28/A=7，方向任务D=31/A=7。V是预计未来折扣奖励的一个标量，不是Q网络，不是成功概率，也不是直接预测到目标的步数。当前任务没有使用TD3。

Actor训练时采样动作；确定性评估使用 `tanh(actor_mean(obs))`，因此展示时不会用随机探索。Actor输出的是归一化增量指令，物理0.02 rad缩放在环境中。

## 2. 完整训练调用顺序

```text
导入环境、注册 → tyro解析Args → 配置/路径检查
→ 可选恢复配置 → W&B/TensorBoard → 随机种子、device
→ SyncVectorEnv(make_env) → Agent、Adam → init或resume
→ obs/action/logprob/reward/done/value缓冲区
→ rollout → GAE/returns → minibatch PPO更新
→ 指标 → 独立确定性评估选best → checkpoint
→ 完成或Ctrl+C安全停止 → 释放资源
```

一个环境不等于一个回合。`total_timesteps`指环境交互步，不是训练轮次、PPO更新次数或episode数。

## 3. Args如何变为命令行

Python字段使用下划线，tyro命令行通常使用连字符，例如 `near_start_from`→`--near-start-from`。模型路径是权重或断点文件，不是运行目录。

| 参数组 | 主要字段 | 含义 / 当前最佳微调参数 |
|---|---|---|
| 任务 | env_id、target_mode、target_pos | 当前方向用RM75Direction-v0、random位置；不提供随机方向角范围 |
| 采样 | target_sampling | 当前uniform；方向环境hard_mixture有错误，不建议使用 |
| 起点 | near_start_probability、near_start_from | 当前0.5，冻结V5权重；申请比例不是步数比例 |
| 初始化 | init_from | 原微调从原V6 best开始；当前脚本已改为最新V6-finetune best，开始新的实验 |
| 恢复 | resume_from | 版本化完整checkpoint，不能与init_from同时用 |
| 预算 | total_timesteps、num_envs、num_steps | 当前500000、1、2048；实际499712 |
| Adam | learning_rate | 当前1e-5，更新参数步长，不是机器人移动速度 |
| 学习率 | anneal_lr | 当前True，线性趋近0；不开启则保持给定学习率 |
| 折扣 | gamma、gae_lambda | 0.99、0.95，未来奖励权重与优势估计平滑 |
| 更新批次 | num_minibatches、update_epochs | 16、5；每小批128条，最多复用该rollout五遍 |
| 策略稳定 | clip_coef、target_kl、max_grad_norm | 0.2、0.02、0.5；不是环境安全保证 |
| 优势/价值 | norm_adv、clip_vloss、vf_coef | True、True、0.5 |
| 探索 | initial_action_std、final_action_std、action_std_hold_fraction | 上限0.15→0.06，前20%保持；继承std不会被强行升到0.15 |
| 熵 | ent_coef | 0.0001，基础高斯熵正则权重 |
| 环境成本 | collision_penalty | 最新45；碰撞仍终止，不是“允许穿过物体” |
| 评估 | eval_interval、eval_episodes、eval_seed | 100000、100、20000；完整batch结束后评估 |
| 验收 | eval_max_collision_rate、eval_target_success | 0.10、0.75；不是更改单回合5cm/15°成功阈值 |
| 保存 | save_model、checkpoint_interval | True、100000；分别保存best/最终权重及最新训练状态 |
| 实验 | exp_name、seed、track、wandb_project_name、wandb_entity | 实验名、随机种子、同步与项目归属 |
| 设备 | cuda、torch_deterministic | 有CUDA且允许时网络在GPU；MuJoCo仍在CPU |
| 已移除旧功能 | capture_video、upload_model、hf_entity | 未参与当前最佳训练，已从PPO CLI移除；展示仍用visualize_agent |

batch_size=num_envs×num_steps；minibatch_size=batch_size//num_minibatches；num_iterations=total_timesteps//batch_size。预算向下取整，并非少跑了几百个episode。

## 4. 函数和类对应关系

| 部分 / 本次行号 | 职责 | 是否必须用于最新模型链 |
|---|---|---|
| Args，63 | 参数及CLI | 是 |
| GoalAugmentWrapper（已删除） | 旧版给obs追加目标3维 | 无调用，当前goal已经在obs内 |
| environment_kwargs，160 | 区分UR5/RM75参数 | 三任务兼容需要 |
| make_env，177 | 创建env、可选起点包装、统计包装 | 简化接口为make_env(env_id, *, args) |
| log_scalars，198 | 统一写入一组标量图表 | 字段名、数值和global_step保持一致 |
| layer_init，204 | 正交初始化线性层 | 从头创建网络需要 |
| compute_action_std，210 | 探索上限保持再衰减 | 当前训练使用 |
| Agent，242 | Actor/Critic和tanh高斯概率 | 核心 |
| resolve_training_path，345 | 从当前目录或项目根找文件 | 路径与断点需要 |
| _torch_load，356 | 权重加载兼容 | 是 |
| _is_full_training_checkpoint，367 | 区分完整断点和裸权重 | 是 |
| load_agent_weights，376 | init、冻结模型加载；支持28→31列扩展 | 当前微调和点策略转方向需要 |
| ArrivalStartWrapper，439 | 冻结策略准备实际到达状态 | 最新训练使用；评估不调用 |
| NumPy RNG两个函数，486/497 | 可安全序列化随机状态 | checkpoint需要 |
| save/load_training_checkpoint，508/555 | 保存/恢复网络、Adam、配置、计数、全局RNG | 工程需要 |
| next_checkpoint_step，578 | 下一个保存/评估边界 | 是 |
| should_update_best，587 | 碰撞资格过滤，再按任务结果排名 | 保留best需要 |
| evaluate_policy，624 | 单独gym.make，确定性100回合 | 验收核心，不用冻结起点策略 |
| save_best_metrics，721 | 原子保存JSON | best结果可追溯 |
| evaluate_and_maybe_save_best，736 | 记录eval、可选IK近处诊断、保存best | 是；near诊断不是另一个正式任务 |
| RESUME_CONFIG_FIELDS/apply_resume_configuration，788/830 | 恢复保存的训练参数 | 避免无意改任务，但不能自动延长 |
| main，839 | 参数检查、环境、rollout、更新、日志、安全停止 | 可提取小函数提高可读性，不应重写网络 |

## 5. tanh动作概率为什么重要

设u从高斯N(mu,std)采样，动作a=tanh(u)。训练保存a和它的log概率；更新时用atanh(a)还原u。代码加变量变换修正：

```text
log pi(a|s) = sum[log Normal(u|mu,std) - log(1-a²+epsilon)]
```

这样PPO比较的是实际归一化动作分布，不是“高斯动作被另一个ClipAction随意裁剪后执行”。不要为了简化删除atanh或修正项。

`actor_logstd`按关节学习，但受计划上限限制；实现另有0.05下界。当前参数满足0.05≤最终0.06。若用户设置最终上限<0.05，则实际有效std仍不会低于0.05，CLI需要补检查。

日志entropy是tanh前基础高斯熵，不是最终动作空间精确熵。连续分布熵可以为负；不能据此判断训练失效。

## 6. 从rollout到GAE

rollout保存obs、action、旧logprob、reward、done、V。不是经验回放池；PPO通常每轮用新采样数据更新，再采下一批。

```text
delta_t = r_t + gamma*(1-done_next)*V_next - V_t
advantage_t = delta_t + gamma*lambda*(1-done_next)*advantage_next
return_t = advantage_t + V_t
```

return_t是Critic拟合目标；episode return是一个完整回合的奖励和。二者不是同一个字段。

当前代码把termination和truncation合为done。环境显式把1024步未成功定义为超时失败并扣10，因此这里是在实现有限回合失败任务；若以后将time-limit改成仅采集截断，就应给最终状态bootstrap，不能机械照抄。剩余时间并未作为obs输入，也是当前近似。

### 当前必须先修的重置样本问题

安装的Gymnasium为1.3.0，SyncVectorEnv默认NEXT_STEP：episode结束后的下一次step只reset，忽略action，reward=0。当前rollout仍保存该动作/logprob/value并把它计入global_step，形成不是真实执行的训练样本。

本次用单步toy env验证：连续3次向量step实际只执行2次env.step，中间一次是重置。可改成显式重置或正确处理SAME_STEP；改SAME_STEP还必须同步调整final_info/final_obs读取，不能只改一个开关。本次未修改。

## 7. PPO更新的数学对应

```text
ratio = exp(new_logprob-old_logprob)
L_policy = mean(max(-A*ratio, -A*clip(ratio,1-eps,1+eps)))
L_value = 0.5*mean(max(未裁剪V误差平方, 裁剪V误差平方))
L_total = L_policy + vf_coef*L_value - ent_coef*基础高斯entropy
```

每小批先归一化优势（当前开启），再backward、梯度裁剪、Adam.step。目标KL在每个epoch结束检查，目前使用最后一个minibatch的估计，不是整批平均。建议后续日志明确该统计范围。

GPU只计算网络前后向；SyncVectorEnv逐环境CPU步进，num_envs增加不等于CPU并行。网络较小时GPU收益可能有限，应看墙钟速度而非仅看device=cuda。

## 8. init、resume和best的真正区别

| 操作 | 网络 | Adam状态 | 步数/学习率计划 | 环境轨迹 |
|---|---|---|---|---|
| init-from | 加载权重 | 新建 | 新计划，从0开始 | reset后重新采样 |
| resume-from | 加载权重 | 恢复 | 保存计划/计数 | 重新reset，不逐步接回旧物理轨迹 |
| visualize best | 加载权重 | 不需要 | 不更新 | 默认原起点独立评估 |

断点保存Python/NumPy/Torch/CUDA全局随机状态，但没有MuJoCo积分状态、环境rng、rollout、滚动统计，因此不是位级连续复现。保存发生在PPO更新边界，属于训练状态续训。修复CUDA RNG映射需要.cpu()，当前已包含。

resume会覆盖CLI中的total_timesteps和learning_rate等训练参数。已完成的500000计划恢复后不自动增加预算；当前最新断点引用的V5冻结权重可找到。新的启动脚本改从最新best做init，并保留V5；resume不加载保存args里的init源文件。旧断点中的已移除视频/HF参数不会恢复为新CLI功能。

裸best只保存权重，没有当时Adam状态和完整配置；不能拿它当best checkpoint来精确resume。当前目录只有checkpoint_latest，不存在checkpoint_best。

从点策略28维到方向31维时，只扩展Actor/Critic首层输入权重，前三个新方向输入列初始为0，其余兼容参数复制。这保留初始点策略行为，不保证后续不遗忘。

## 9. 当前结果和关键图表

当前方向固定评估序列：82%、85%、97%、94%、88%，最佳301056步。选模100种子可产生选择偏差；另200个新种子评估为93%、1%碰撞，仍不是任意方向验收。

| 图表 | 应如何理解 |
|---|---|
| eval/success_rate | 完整起点、确定性、严格保持成功；主要验收指标 |
| charts/success_rate | 训练最近100回合，随机动作及混合起点，不能替代eval |
| eval/pose_reached_rate | 曾同时达标，未必保持5步 |
| eval/position_success_rate | 曾进入位置范围，非最终仍在范围的比例 |
| eval/angle_success_rate | 曾方向达标，位置可能很远，尤其不能单独当任务成功 |
| eval_near/* | IK近处起点诊断，起点分布不同于训练冻结策略交接 |
| losses/value_loss | 价值拟合误差，尺度受奖励影响；低不证明完成任务 |
| losses/explained_variance | 1-Var(target-pred)/Var(target)，不是成功率 |
| losses/policy_loss | 裁剪代理损失，不要求一直下降或趋零 |
| losses/approx_kl | 当前新旧策略差异估计，过大可能不稳，接近0也可能LR太低 |
| losses/old_approx_kl | 有限样本可为负；别把负值当数学KL为负 |
| losses/clipfrac | ratio超出clip区间的样本比例，不是动作撞限位比例 |
| charts/action_saturation_rate | abs(action)>0.98的比例 |
| charts/SPS | 学生global_step/耗时；包括评估与准备时间，但步数分子没计准备交互 |

所有成功/碰撞率优先同时看；增加学习率、扩大探索或删奖励都不是保证收敛的方法。

## 10. 建议的学习和精简顺序

先读Agent和训练核心，再读checkpoint、评估及日志。已删除GoalAugmentWrapper及HF/重复CleanRL末尾评估，统一标量写入和三处checkpoint调用；未新增模块层次，也未拆分Agent。

必须保留网络层名称及结构，否则state_dict键会变，现有模型无法直接加载。若拆分Agent，实现兼容导出，保证可视化仍能 `from Algorithms.ppo_continuous_action import Agent`。

完整审查及精简验证见 [CURRENT_CODE_AUDIT_zh-CN.md](CURRENT_CODE_AUDIT_zh-CN.md)。本次修改重复工程实现，不修改PPO数学；自动重置行为修复尚未实施。
