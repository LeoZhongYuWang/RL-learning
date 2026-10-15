# 当前项目代码入门与 RM75 环境导读

更新日期：2026-10-13。等价精简后 `env_rm75.py` 1255行、PPO 1452行。以下调用关系对应当前代码，不对应历史随机方向课程；按函数名查找比依赖行号更稳妥。

## 1. 最小心智模型

环境定义“机器人能做什么、看见什么、得到什么奖励”；PPO学习“当前状态下该采取什么动作”。MuJoCo负责物理，不学习策略。网络权重只是一组参数，必须与相同观测含义、关节顺序和控制尺度的环境一起使用。

```text
训练：启动参数 → PPO → gym.make → RM75P2PEnv → MJCF/MuJoCo
展示：visualize_agent → 按权重维度选环境 → Agent → env.step/render
```

`__init__.py` 和PPO都包含注册保护；当前PPO直接导入环境，不依赖不存在的 `UR5` Python包。

| 文件 | 当前职责 | 是否直接参与最新方向训练 |
|---|---|---|
| env_rm75.py | RM75模型、重置、控制、奖励、观测、碰撞 | 是 |
| Algorithms/ppo_continuous_action.py | 网络、rollout、GAE、PPO、起点准备、保存评估 | 是 |
| assets/rm75/scene.xml、rm75.xml、meshes | 七关节模型和位置执行器 | 是 |
| env.py、assets/scene.xml | UR5-v3兼容 | 否，但保留任务需要 |
| Scripts/visualize_agent.py | 加载19/28/31维模型及展示 | 不训练，验收需要 |
| Scripts/train_rm75_v4.sh | 点任务参数入口 | 点任务需要 |
| Scripts/train_rm75_direction.sh | 当前方向best的新微调入口，冻结V5依赖不变 | 路径已核验；先处理自动重置问题再正式训练 |
| ik.py、ik_solver.py | 独立IK工具 | 最新环境未导入，不是控制链的一环 |
| Algorithms/td3_ur5.py | 历史TD3实验 | 否；当前模型不是TD3训练的 |
| Scripts/record_hd.py | 历史视频工具 | 否；仍有旧import UR5，不能当作RM75录制入口 |
| tools/convert_rm75_urdf.py | URDF→MJCF资产转换 | 建模工具，不逐步参与训练 |

## 2. observation、action、reward

当前RM75点任务观测28维，顺序不可随意改：

| 切片 | 内容 | 单位 / 意义 |
|---|---|---|
| 0:7 | q | 7个关节真实角度，rad |
| 7:14 | qd | 真实角速度，rad/s |
| 14:21 | q_target-q | 位置伺服跟踪误差，rad |
| 21:24 | ee | 末端世界位置，m |
| 24:27 | goal | 目标世界位置，m |
| 27 | collision | 当前检测的碰撞标志 |
| 28:31，仅方向任务 | goal_direction-current_direction | 方向向量差，无量纲 |

UR5-v3输入19维：q(6)、qd(6)、ee(3)、goal(3)、collision(1)。RM75-v3/V4是28维；当前方向best是31维。

目标来自环境随机数和MuJoCo真实状态，不是仿真相机识别。真机通常可获得关节反馈，末端位姿可由标定后的运动学计算；视觉目标、接触信息仍需额外实现。不能直接把全部仿真观测视为真实传感器现成输出。

action是7维归一化关节增量指令，不是XYZ移动，也不是直接输出7个绝对角度：

```text
joint_delta = 0.02 * clip(action,-1,1)
q_target = clip(q_target + joint_delta, joint_limits)
data.ctrl = q_target
```

q_target是累积目标，q是真实状态。二者不必瞬间一致，因而跟踪误差有实际价值。

## 3. 一个回合的生命周期

```text
__init__ → 编译固定场景一次
reset → 清MjData → 摩擦/初值噪声 → 起点位置IK → 伺服warmup → 随机目标
      → 可选近起点 → _finish_reset → obs
step → 动作限幅 → 更新q_target → 10个物理子步 → 误差/碰撞
     → 成功保持计数/一次性奖励 → obs,reward,terminated,truncated,info
```

普通起点约 `[0.4,0,0.2905]`。点任务目标x∈[-0.50,-0.35]、y∈[-0.30,0.30]；方向任务x∈[-0.40,-0.35]、y∈[-0.15,0.15]，z均固定0.2905。当前只随机二维位置，不是整个三维工作空间。

模型重置不等于重新编译模型：每回合仍重新计算关节初值、IK和噪声，桌子拓扑固定，仅摩擦数值随机化。

最新训练的近起点由PPO中的 `ArrivalStartWrapper` 使用冻结V5策略准备，非此处IK近起点。冻结策略先执行到目标6 cm以内，再把物理状态交给学生；其动作不进入学生批次。正常评估和可视化只用学生。

## 4. 每个函数做什么

| 函数 / 本次位置 | 输入、实现和返回 | 保留理由 |
|---|---|---|
| __init__，26 | 验证参数、创建动作/观测空间、奖励配置、构建模型 | 核心契约 |
| _build_scene_and_model，173 | 场景深拷贝，加双桌和目标site，XML编译为MjModel/MjData；验证关节/执行器映射 | 必须保留自检；不能为省行数删除 |
| _find_robot_geom_ids，316 | 找基座/七连杆geom，缓存结构相邻允许接触集合 | 自碰撞分类需要 |
| _randomize_table_friction，379 | 用环境rng改变滑动摩擦0.4～0.9 | reset的seed可复现，不重新编译 |
| _update_visualization_sites，392 | 更新红起点、绿目标和固定向下圆柱线 | 展示方向目标需要；不是奖励 |
| _fallback_interactive_viewer，406 | 手动查看器备用循环 | 不参与PPO，可考虑移出环境 |
| interactive_viewer，436 | 手动查看器和关节目标保持 | 不参与训练；当前提示滑块控制，但循环会覆盖ctrl，需澄清 |
| reset，478 | 重置物理数据、初值噪声、位置IK、目标、可选近起点 | 核心 |
| _finish_reset，568 | 清历史、初始化差分奖励基准、预标记已满足里程碑，返回obs/info | 冻结策略交接也调用，不可随意删 |
| _reset_near_goal，623 | 临时倾斜方向做IK，验收近起点，不成功则恢复原状态 | 当前eval_near和兼容IK训练使用，不是随机目标采样 |
| _sample_random_goal，674 | uniform或旧hard_mixture采样，返回goal和区域 | uniform必需；hard_mixture对当前方向范围有错误 |
| _pd_to_ctrl，703 | 把q_des写入七个位置执行器ctrl | 名字带PD，但PD实际在XML执行器内 |
| _step_sim，707 | 每个物理子步保持q_target并mj_step | 核心控制周期 |
| step，713 | 把动作变物理状态，计算奖励、结束、info | 核心 |
| _compute_reward，851 | 汇总进度、状态成本、里程碑、成功/碰撞/超时项 | 奖励权重集中在reward_cfg |
| _compute_milestone_reward，930 | 每项每回合只领一次；区分纯距离和联合距离/角度 | 两类模型兼容必需 |
| _ee_pos，958 | forward后读取site_xpos | 末端位置，重复forward可优化但要验证状态刷新 |
| _ee_approach_direction，964 | 从site_xmat取工具指定局部轴并转为世界方向 | 方向误差需要 |
| _direction_error，975 | 目标方向减当前方向 | 31维输入最后3项 |
| _direction_angle，979 | acos(单位方向点积)，返回rad | 成功判断；显示时再转deg |
| _direction_gate，987 | 20cm外0，5cm内1，中间线性 | 远处先靠近、近处学方向 |
| _compute_direction_progress，995 | 前后门控方向误差差分 | 与位置进度合用，不是角度单项最大化 |
| _get_obs，1024 | 按固定顺序拼接28或31维float32 | 未使用的q_goal形参已删除，观测内容不变 |
| _solve_ik，1049 | 阻尼最小二乘位置IK，限制关节，失败None | 普通reset也用，不能与近处IK一起全删 |
| _axis_alignment_rotvec，1091 | 最短轴对齐旋转向量，处理平行/反向 | 方向IK需要 |
| solve_direction_ik，1109 | 位置＋轴方向阻尼IK，终点检查碰撞 | IK近起点和诊断用，不生成PPO动作 |
| _contact_pair_is_collision，1157 | 桌面、地面、非相邻机器人接触分类 | 自碰撞与安全指标 |
| _check_collision，1183 | forward并扫描当前contact | 不检测整个10子步内的所有瞬时接触 |
| render，1191 | human窗口或rgb_array | 可视化需要，不能因训练无窗口而删除 |
| close，1241 | 释放窗口和离屏Renderer | 资源清理 |
| quat_mul（已删除） | 未在当前RM75链调用 | 删除未调用函数，不影响观测/动作 |

## 5. 奖励为什么这样设计

设d为距离（m），theta为方向误差（rad）：

```text
g(d) = clip((0.20-d)/0.15,0,1)
C = 50*d + 6*g(d)*theta/pi
方向任务每步基础奖励 = C_previous-C_current -0.5*d -0.01 -0.05*g(d)*theta/pi
```

这样远处不因方向对准单独获奖，近处方向才参与；同时保留距离成本，避免只旋转而位置退化。50与6不是物理定律，只是当前经验权重。

| 联合里程碑 | 单回合一次奖励 |
|---|---:|
| d<15 cm 且 theta<45° | +5 |
| d<10 cm 且 theta<30° | +10 |
| d<7 cm 且 theta<20° | +20 |
| d<5 cm 且 theta<15°，连续5步 | +100，并成功终止 |

最新训练碰撞扣45并终止；未成功、未碰撞的超时扣10；每回合上限1024步。点任务保留50倍距离进度、-0.03*d、-0.01及20cm/10cm纯距离里程碑+5/+15，不含方向要求。

差分进度在不折扣闭合往返时总和为0；不能称为严格的gamma势函数塑形证明。里程碑是历史相关奖励，会给不同路径额外收益；是否必要应单独做消融，不凭成功率归因某一项。

## 6. 速度和平滑：有什么、缺什么

MuJoCo dt=0.002 s，10个子步＝环境控制周期0.02 s。位置执行器的kp/kv按关节不同，在rm75.xml中设置，例如前两关节200/20、末三关节40/4。

0.02 rad增量使参考角变化不至于无限大；连续同向满动作对应参考目标约1 rad/s的变化率。但实际qvel可能不同，当前没有显式的真实速度/加速度/jerk硬限幅，也没有相邻动作变化惩罚。必须记录轨迹再谈平滑，不能从成功率推导安全。

## 7. 入门阅读顺序

1. 先看任务规格、维度和单位；知道“方向要求”不是完整姿态，也不是抓取。
2. 看step中action→joint_delta→q_target→ctrl→物理子步，理解控制器状态。
3. 看reset、观测顺序、成功计数，再看reward_cfg和奖励函数。
4. 看Actor/Critic和rollout，再学GAE与PPO更新。
5. 最后看评估、best/断点、近起点包装器；不要从日志长代码开始背。

从零开发建议先验证模型、关节映射和简单控制，再做最小环境、独立评估，最后才调奖励和训练。先建立正确的闭环，比增加课程、阈值和图表更重要。

已执行的精简和验证结果见 [CURRENT_CODE_AUDIT_zh-CN.md](CURRENT_CODE_AUDIT_zh-CN.md)。奖励和PPO数学没有修改；需要改变训练行为的问题仍单独列为待修复。
