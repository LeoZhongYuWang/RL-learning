# UR5 PPO 训练稳定性修复实施方案

> **供执行本方案的智能体使用：** 按任务顺序执行并逐项验证。步骤使用复选框（`- [ ]`）记录完成状态。

**目标：** 修复 UR5 PPO 的动作尺度、回合日志、环境重置效率和过强更新参数，使训练动作与概率计算一致，并能在 W&B 中可靠观察任务学习情况。

**总体结构：** 环境对外暴露 `[-1, 1]` 归一化动作，再映射为每步最大 `0.02 rad` 的关节目标增量；PPO 使用 Tanh 压缩的高斯策略确保采样动作天然有界。MuJoCo 模型只构建一次，每回合只重置 `MjData`、加入关节初值噪声并重新执行 IK。Gymnasium 1.3 的 episode 字段在 rollout 内记录，同时保留回报、成功、碰撞、末端距离和动作饱和率。

**技术栈：** Python 3.10、MuJoCo 3.13、Gymnasium 1.3、PyTorch、标准库 `unittest`。

**依据：** 当前 `env.py`、`Algorithms/ppo_continuous_action.py`，以及用户提供的 50 万步 W&B 曲线。

## 全局约束

- 不修改现有奖励公式和完整随机目标区域，先验证基础控制与 PPO 修复。
- 不加入示范轨迹或课程学习。
- 保留每回合关节初值噪声和 IK 初始化。
- 动作物理含义改变后，旧 PPO 检查点不再继续训练。
- 不覆盖用户其他未提交改动。

---

### 任务 1：修复环境动作和重置

**文件：**

- 修改：`env.py`
- 新建：`tests/test_ur5_training_contract.py`

**接口：**

- 输入：策略动作 `action: np.ndarray`，每维范围 `[-1, 1]`。
- 输出：关节目标增量 `joint_delta = action * 0.02 rad`。
- 重置：复用同一个 `MjModel`，清理 `MjData`，再执行带噪声的 IK。

- [x] 编写测试：动作空间是 `[-1,1]^6`，最大目标增量不超过 `0.02 rad`。
- [x] 编写测试：连续两次 `reset()` 的 `id(env.model)` 不变，且末端仍位于起点 2 cm 内。
- [x] 修改 `__init__()`：增加 `action_scale=0.02`，默认关闭大量调试输出。
- [x] 修改 `reset()`：删除 `_build_scene_and_model()`，改用 `mj_resetData()`；在 IK 前设置 `start_pos`。
- [x] 修改 `step()`：把归一化动作映射为物理关节增量。
- [x] 运行环境契约测试。

### 任务 2：修复 PPO 有界策略、日志和参数

**文件：**

- 修改：`Algorithms/ppo_continuous_action.py`
- 修改：`tests/test_ur5_training_contract.py`

**接口：**

- `Agent.get_action_and_value()` 返回范围严格位于 `(-1,1)` 的 Tanh 压缩动作。
- Gymnasium 1.3 完成回合后记录 return、length、success、collision 和 final distance。

- [x] 编写测试：Agent 动作有界、logprob/entropy/value 有限，使用旧动作重新计算 logprob 时形状一致。
- [x] 删除 `ClipAction`，避免环境动作被二次静默裁剪。
- [x] 初始策略标准差改为 `0.3`，对高斯样本执行 Tanh 压缩，并加入雅可比概率修正。
- [x] 默认参数改为 `learning_rate=1e-4`、`num_steps=2048`、`num_minibatches=16`、`update_epochs=5`、`ent_coef=0.001`、`target_kl=0.02`。
- [x] 检查 episode 日志缩进位于 rollout 内，增加 `action_saturation_rate`。
- [x] 运行策略契约测试。

### 任务 3：完整验证

**文件：**

- 验证：`env.py`
- 验证：`Algorithms/ppo_continuous_action.py`
- 验证：`tests/test_ur5_training_contract.py`

- [x] 运行 Python 编译检查。
- [x] 运行全部新增测试。
- [x] 运行 4096 步无 W&B PPO 冒烟训练，确认至少完成两次更新并保存有限损失。
- [x] 检查 Git 差异，确认没有修改奖励、目标范围和用户其他文件。
- [x] 汇总修改位置、修改原因、行为变化和重新训练命令。
