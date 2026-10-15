# UR5 强化学习项目代码导读

本文档按以下三个角度讲解该项目：

1. 代码运行时究竟怎样逐层调用；
2. `env.py` 中每个函数的职责、输入输出和实现原理；
3. 如果从零开发类似项目，实际工程中更合理的编写顺序是什么。

> 说明：本文描述的是当前仓库中的真实代码。对于尚未接通、重复实现或存在风险的部分，会明确标注，而不会把“文件名看起来有关联”误写成真实调用关系。

---

## 1. 先建立整体认识

这个项目可以分为五层：

```text
第 5 层：运行入口
  Algorithms/ppo_continuous_action.py
  Algorithms/td3_ur5.py
  Scripts/visualize_agent.py
  Scripts/record_hd.py
  env.py 的命令行入口

第 4 层：强化学习算法
  PPO Agent / TD3 Actor / TD3 Critic

第 3 层：Gymnasium 环境接口
  __init__.py 注册 UR5-v0
  env.py 实现 UR5P2PEnv

第 2 层：机器人运动学与控制
  env.py 内置的 _solve_ik、_pd_to_ctrl
  ik.py 的独立 IK 实验
  ik_solver.py 的独立运动学封装

第 1 层：MuJoCo 模型和资源
  assets/scene.xml
    └── include assets/ur5e.xml
          └── assets/assets/*.obj
```

最核心的数据通道是：

```text
神经网络输出 action（6 维）
        ↓
UR5P2PEnv.step(action)
        ↓
累加为目标关节位置 q_target
        ↓
写入 MuJoCo data.ctrl
        ↓
mj_step 推进物理仿真
        ↓
读取 qpos、qvel、末端位置、碰撞
        ↓
生成 observation、reward、terminated、truncated、info
        ↓
训练算法保存经验并更新网络
```

因此，训练算法并不直接控制 MuJoCo。算法只认识 Gymnasium 的 `reset()` 和 `step()`；真正把动作转换成机械臂运动的是 `env.py`。

---

## 2. 文件职责与真实调用关系

| 文件                                    | 职责                                        | 谁调用它                                              | 它调用谁                                  |
| --------------------------------------- | ------------------------------------------- | ----------------------------------------------------- | ----------------------------------------- |
| `__init__.py`                         | 向 Gymnasium 注册`UR5-v0`                 | PPO、TD3、录制脚本原本希望通过`import UR5` 触发     | `env.UR5P2PEnv`                         |
| `env.py`                              | 定义环境、控制器、奖励、碰撞、渲染和简化 IK | Gymnasium、可视化脚本、`ik_solver.py`、也可直接运行 | MuJoCo、dm_control MJCF、XML 模型         |
| `ik.py`                               | 独立的、较完整的阻尼最小二乘 IK 实验        | 目前只在直接运行`python ik.py` 时使用               | MuJoCo、MJCF/XML                          |
| `ik_solver.py`                        | 用类封装正运动学和 IK                       | 目前没有被训练脚本或`env.py` 调用                   | 创建`UR5P2PEnv` 并复用其模型            |
| `Algorithms/ppo_continuous_action.py` | PPO 训练、日志、保存和评估                  | 命令行                                                | Gymnasium 环境、PyTorch、W&B、TensorBoard |
| `Algorithms/td3_ur5.py`               | TD3 训练、经验回放、保存和评估              | 命令行                                                | Gymnasium 环境、PyTorch、CleanRL 工具     |
| `Scripts/visualize_agent.py`          | 加载 PPO 权重并在窗口中运行                 | 命令行                                                | `UR5P2PEnv`、PPO `Agent`              |
| `Scripts/record_hd.py`                | 加载 PPO 权重并将帧编码成视频               | 命令行                                                | Gymnasium、PPO`Agent`、Pillow、MoviePy  |

### 一个容易误解的地方：三套 IK 没有串起来

当前项目中存在三套逆运动学实现：

```text
env.py::_solve_ik
  └── 只被 env.py::reset 使用

ik.py::ik_solve
  └── 只被 ik.py::main 使用

ik_solver.py::UR5Kinematics.inverse_kinematics
  └── 只被 ik_solver.py 自己的示例入口使用
```

也就是说，修改 `ik.py` 或 `ik_solver.py` 不会改变强化学习环境的初始化行为。训练时真正使用的是 `env.py::_solve_ik`。

---

## 3. 三条主要运行调用链

### 3.1 PPO 训练链

```text
运行 ppo_continuous_action.py
  ↓
解析 Args
  ↓
import UR5，期望触发 __init__.py 中的 gym.register
  ↓
make_env() 返回一个延迟创建环境的 thunk
  ↓
SyncVectorEnv 调用 thunk
  ↓
gym.make("UR5-v0")
  ↓
UR5P2PEnv.__init__()
  ├── 定义动作/观测空间
  ├── 读取 scene.xml
  └── _build_scene_and_model()
  ↓
env.reset()
  ├── 重建 MuJoCo 模型
  ├── 设置初始关节状态
  ├── 采样目标
  └── 返回 19 维 observation
  ↓
Agent.get_action_and_value(observation)
  ↓
env.step(action)
  ↓
收集 num_steps 个时间步
  ↓
计算 GAE 和 returns
  ↓
多轮小批量 PPO 更新
  ↓
记录指标、保存模型、执行评估
```

### 3.2 TD3 训练链

```text
运行 td3_ur5.py
  ↓
注册并创建 UR5-v0
  ↓
创建 Actor、两个 Critic 及其目标网络
  ↓
env.reset()
  ↓
前 learning_starts 步使用随机动作
  ↓
之后使用 Actor 动作 + 探索噪声
  ↓
env.step(action)
  ↓
经验写入 ReplayBuffer
  ↓
采样 batch 更新两个 Critic
  ↓
每 policy_frequency 步更新 Actor
  ↓
使用 Polyak 平均软更新目标网络
```

### 3.3 直接交互链

当前 `env.py` 底部的入口是：

```text
python env.py --interactive --target-mode random
  ↓
UR5P2PEnv(render_mode=None, ...)
  ↓
env.reset()
  ↓
env.interactive_viewer()
  ↓
mujoco.viewer.launch_passive(model, data)
  ↓
循环：写控制量 → mj_step → viewer.sync
  ↓
关闭窗口 → env.close()
```

`--interactive` 决定是否打开窗口，`--target-mode` 决定怎样产生目标，两个选项互不冲突。

---

## 4. 理解 `env.py` 前必须知道的 MuJoCo 对象

### `model`：模型结构

`MjModel` 是编译后的、主要描述静态结构的对象，例如：

- 有多少个关节、自由度、执行器、几何体和 site；
- 关节限制；
- 质量、惯量、摩擦；
- 执行器参数；
- 各对象的名称和 ID。

项目通过以下方式生成它：

```python
xml = scene.to_xml_string()
assets = scene.get_assets()
self.model = mj.MjModel.from_xml_string(xml, assets=assets)
```

### `data`：当前仿真状态

`MjData` 保存会随时间变化的状态，例如：

- `qpos`：关节位置；
- `qvel`：关节速度；
- `ctrl`：执行器输入；
- `site_xpos`：site 的世界坐标；
- `site_xmat`：site 的世界旋转矩阵；
- `contact` 和 `ncon`：当前接触信息；
- `qfrc_bias`：重力、科氏力和离心力等偏置项。

### `mj_forward` 与 `mj_step`

```text
mj_forward(model, data)
```

根据当前 `qpos/qvel` 重新计算运动学、动力学派生量和接触，但不让仿真时间向前推进。

```text
mj_step(model, data)
```

根据当前控制量推进一个物理时间步，会真正改变 `qpos/qvel`。

### `site_pos` 与 `site_xpos`

- `model.site_pos[id]` 是 site 相对父 body 的局部位置；世界 body 下的 site 可以直接把它当世界坐标设置。
- `data.site_xpos[id]` 是经过正运动学计算后的世界坐标。

这个项目的 `start_site` 和 `goal_site` 建在 `worldbody` 下，因此可以通过修改 `model.site_pos` 移动标记；`ee_site` 建在腕部 body 下，要从 `data.site_xpos` 读取末端的世界位置。

---

## 5. `env.py` 的数据契约

### 动作空间

```python
self.action_space = Box(low=-0.05, high=0.05, shape=(6,))
```

动作不是直接的力矩，也不是严格意义上的关节速度，而是每个强化学习时间步对“目标关节角”的增量：

```text
q_target(t+1) = q_target(t) + action(t)
```

随后 MuJoCo XML 中的位置伺服执行器负责追踪 `q_target`。因此更准确的名称是“关节位置参考增量”。

### 观测空间

观测是 19 维 `float32` 向量：

| 切片           | 维数 | 含义                       |
| -------------- | ---: | -------------------------- |
| `obs[0:6]`   |    6 | 当前关节角`qpos`         |
| `obs[6:12]`  |    6 | 当前关节速度`qvel`       |
| `obs[12:15]` |    3 | 末端执行器世界坐标         |
| `obs[15:18]` |    3 | 目标世界坐标；无目标时为零 |
| `obs[18]`    |    1 | 是否与桌面碰撞             |

### 一步返回值

`step()` 遵循 Gymnasium 五返回值接口：

```python
observation, reward, terminated, truncated, info
```

- `terminated=True`：任务自然结束，即成功或碰撞；
- `truncated=True`：达到 1024 步时间上限；
- `info`：包含成功、碰撞、距离和各奖励分量，便于调试和统计。

---

## 6. `env.py` 每个定义的含义和实现

### 6.1 `rand_uniform(a, b)`

作用：从 `[a, b)` 均匀采样一个浮点数。

当前只用来随机设置两张桌子的滑动摩擦系数：

```python
friction=[rand_uniform(0.4, 0.9), 0.005, 0.0001]
```

注意：它使用全局 `np.random`，不是环境自己的 `self.rng`，因此 `env.reset(seed=...)` 不能完全控制这部分随机性。在标准工程中应统一使用环境 RNG。

### 6.2 `quat_mul(q1, q2)`

作用：计算两个四元数的 Hamilton 乘积，输入输出采用 `xyzw` 顺序。

它展开实现了：

```text
q = q1 ⊗ q2
```

当前项目没有任何地方调用它。还要注意 MuJoCo 常用四元数排列是 `wxyz`，以后若使用该函数，必须先统一数据约定，否则会产生难以察觉的姿态错误。

### 6.3 `UR5P2PEnv.__init__(...)`

作用：创建环境对象、定义 Gymnasium 接口并加载基础模型。

主要过程：

1. 设置六个关节和六个执行器：`nq=6`、`nu=6`；
2. 保存渲染模式和随机数生成器；
3. 定义动作空间和 19 维观测空间；
4. 初始化 viewer、步数、目标、控制模式等状态；
5. 集中保存奖励参数；
6. 用 `mjcf.from_path()` 读取 `assets/scene.xml`；
7. 调用 `_build_scene_and_model()` 编译第一份 MuJoCo 模型；
8. 保存目标生成模式以及随机目标区域。

重要成员：

| 成员             | 含义                                     |
| ---------------- | ---------------------------------------- |
| `substeps=10`  | 每个 RL 动作执行 10 个 MuJoCo 物理步     |
| `horizon=1024` | 一个回合最多 1024 个 RL 步               |
| `q_target`     | 位置伺服器正在追踪的目标关节角           |
| `prev_dist`    | 上一步末端到目标的距离，用于计算进度奖励 |
| `target_mode`  | `fixed`、`random` 或 `none`        |
| `use_ik_init`  | 是否尝试通过 IK 初始化到起点附近         |
| `reward_cfg`   | 所有奖励系数                             |

工程注意点：`UR5_SCENE_XML` 是相对当前工作目录的路径，而不是相对 `env.py` 的路径。从项目目录之外运行时可能找不到 XML。更稳妥的方式是基于 `Path(__file__)` 构造绝对路径。

### 6.4 `_build_scene_and_model()`

作用：在基础机器人场景上添加项目所需对象，并编译成 MuJoCo 的 `model/data`。

实现可分为五步。

第一步，深拷贝基础 MJCF：

```python
scene = copy.deepcopy(self.base_mjcf)
```

这样每次构建时都从干净模板开始，不会把上一次动态添加的桌子和 site 再添加一遍。

第二步，定位或创建末端执行器 site：

- 优先找模型已有的 `attachment_site`；
- 尝试找到 `wrist_3_link`；
- 创建 `ee_site`，并使其局部位置和姿态与 `attachment_site` 一致；
- 如果模型命名不同，则尝试 `tool0`、`ee_link`、`flange` 等候选名称。

site 是一个没有质量的参考坐标系，方便读取末端位置、姿态和雅可比。

第三步，添加两张桌子：

```text
table_1 中心约为 [ 0.7, 0.07025, 0.07025]
table_2 中心约为 [-0.7, 0.07025, 0.07025]
半尺寸为 [0.4, 0.6, 0.07025]
```

MuJoCo 的 box `size` 表示半尺寸，不是完整长宽高。

第四步，添加起点和目标标记：

- `start_site`：红色；
- `goal_site`：绿色。

第五步，编译并缓存 ID：

- 将 MJCF 转成 XML 字符串；
- 收集 OBJ 等资源；
- 创建 `MjModel` 和 `MjData`；
- 建立几何体名称到 ID、site 名称到 ID 的字典；
- 保存 `ee_sid/start_sid/goal_sid`。

这个方法在构造函数中调用一次，又在每次 `reset()` 时调用一次。重新编译模型开销较大，并且会替换旧的 `model/data`；若旧 viewer 或 renderer 还引用旧模型，就会产生生命周期问题。实际工程中通常只编译一次，reset 时只修改 `data` 和可变参数。

### 6.5 `_fallback_interactive_viewer()`

作用：当 MuJoCo 官方 viewer 不可用时，尝试使用第三方 `mujoco-python-viewer`。

循环中执行：

```text
更新目标标记
→ 保持当前关节目标
→ mj_step
→ viewer.render
```

它捕获 `KeyboardInterrupt` 和普通 Python 异常，并在 `finally` 中关闭 viewer。

局限：原生库产生的 segmentation fault 不是 Python 异常，不能被这里的 `except Exception` 捕获。

### 6.6 `interactive_viewer()`

作用：启动 MuJoCo 官方阻塞式交互窗口，并在窗口存在期间持续推进仿真。

主要逻辑：

1. 导入 `mujoco.viewer`；
2. macOS 上若不是 `mjpython`，切换到备用 viewer；
3. 保存当前关节位置为 `q_hold`；
4. 通过 `launch_passive(model, data)` 启动窗口；
5. 在循环中更新目标、写入控制、推进物理、同步画面。

当前循环每帧都调用：

```python
self._pd_to_ctrl(q_hold)
```

这会反复覆盖右侧 Actuation 面板的滑块值。因此提示文字虽然说可以用滑块移动关节，实际控制逻辑却在持续把关节参考值写回 `q_hold`。

此外，passive viewer 与仿真线程共享 `model/data`。更稳妥的实现应在修改共享数据时使用 viewer 提供的锁。

### 6.7 `reset(seed=None, options=None)`

作用：开始一个新回合，恢复机械臂状态、生成目标并返回初始观测。

实际执行顺序：

```text
必要时重建 self.rng
→ step_i = 0
→ 重建整个 MuJoCo model/data
→ 生成带少量噪声的初始关节角 q0
→ 如果已存在 start_pos，则尝试 IK
→ 写入 qpos/qvel
→ mj_forward
→ 将当前 qpos 锁存为 q_target
→ 预热 10 个物理步
→ 设置固定 start_pos
→ 根据 target_mode 生成 goal_pos
→ 更新可视化 site
→ 保存 prev_dist
→ 必要时渲染
→ 生成 observation 和 info
```

三种目标模式：

- `fixed`：使用命令行 `target_pos`，未提供时采用默认固定点；
- `random`：在 `x∈[-0.5,-0.35]`、`y∈[-0.3,0.3]` 中采样，`z=0.2905`；
- `none`：不设置目标，观测中的目标位置为零。

当前存在一个顺序细节：第一次构造环境时 `start_pos=None`，但 IK 判断发生在设置 `start_pos` 之前，所以第一次 `reset()` 不会运行 IK；第二次及之后才会使用上一次留下的 `start_pos`。标准写法应先确定本回合起点，再调用 IK。

Gymnasium 还要求自定义环境在 reset 开头调用：

```python
super().reset(seed=seed)
```

当前代码没有调用它，因此官方 `check_env` 的 seed 检查不通过。

### 6.8 `_pd_to_ctrl(q_des)`

作用：把期望关节位置转换成执行器控制量。

它设计了三个分支：

```text
position：ctrl = q_des
torque： ctrl = Kp(q_des-q) - Kd*qd + bias
velocity：ctrl = Kp(q_des-q) - Kd*qd
```

默认 `position` 分支与 XML 模型匹配。`ur5e.xml` 中的 `general` actuator 已通过：

```xml
gainprm="2000"
biasprm="0 -2000 -400"
```

配置成类似位置 PD 伺服器，因此 Python 只需把目标角度写入 `data.ctrl`。

但设置环境变量 `UR5_CTRL_MODE=torque` 并不会自动把 XML 执行器改成真正的力矩执行器。因此另外两个分支目前只是接口设想，不是完整可用的控制模式。

### 6.9 `_step_sim(n=1, hold_target=True)`

作用：连续推进若干个 MuJoCo 物理步。

每个子步都会重新写入 `q_target`，然后调用 `mj_step()`。这样位置参考在整个控制周期内保持不变。

`step()` 默认调用 10 个子步。若 XML 使用 MuJoCo 默认的 0.002 秒 timestep，那么一个 RL 步大约对应 0.02 秒仿真时间；应以实际 `model.opt.timestep` 为准。

### 6.10 `step(action)`

这是整个环境最重要的方法，负责完成一次“智能体—环境”交互。

完整流程：

1. 回合步数加一；
2. 将动作裁剪到 `[-0.05, 0.05]`；
3. 把动作累加到 `q_target`；
4. 按关节限制再次裁剪 `q_target`；
5. 更新起点和目标 site；
6. 推进 10 个物理子步；
7. 若是 human 模式则同步画面；
8. 检查机械臂是否接触桌面；
9. 计算末端到目标的距离；
10. 判断是否成功；
11. 调用 `_compute_reward()`；
12. 判断自然终止和时间截断；
13. 生成新观测和调试信息。

终止规则：

```text
terminated = success OR collided
truncated  = step_i >= 1024
```

要特别区分：动作只是改变“目标关节角”，真实 `qpos` 要经过执行器和物理系统响应后才会变化。

### 6.11 `_compute_reward(...)`

作用：将距离变化、当前距离、耗时、姿态、碰撞和成功组合成标量奖励。

当前公式是：

```text
progress = 10 × (prev_dist - dist) / max(dist, 1e-6)
distance = -0.5 × dist
time     = -0.01
collision= -10（发生碰撞时）
success  = +100（成功时）

reward = progress + distance + time + orientation + collision + success
```

各部分意义：

- `progress` 比较前后距离，奖励接近目标、惩罚远离目标；
- `distance` 即使没有进步，也持续鼓励保持接近目标；
- `time` 鼓励尽快完成；
- `collision` 表达安全约束；
- `success` 提供明确的稀疏终点信号。

姿态奖励读取 `site_xmat` 的某一列作为末端局部坐标轴，与世界 `-Z` 做点积。当前 `orient_weight=0`，所以默认关闭。

值得注意：当前姿态项使用 `align ** 2`，这会让 `align=1` 和 `align=-1` 获得相同奖励，也就是同向和反向无法区分。如果目标是“必须朝下”，通常需要截断负值或把 `[-1,1]` 映射到 `[0,1]`。

### 6.12 `_ee_pos()`

作用：返回末端执行器的世界坐标。

它先调用 `mj_forward()` 更新派生量，然后复制：

```python
self.data.site_xpos[self.ee_sid]
```

使用 `.copy()` 很重要，否则返回的 NumPy 视图可能随着下一次 MuJoCo 计算而悄悄变化。

### 6.13 `_get_obs(q_goal=None)`

作用：组装 19 维观测。

它读取：

```text
qpos + qvel + ee_pos + goal_pos + collision_flag
```

最终统一转换为 `float32`，与神经网络常用数据类型一致。

参数 `q_goal` 当前没有被使用，可以删除或真正接入。

性能方面，`_ee_pos()` 和 `_check_collision()` 都会再次执行 `mj_forward()`。而一次 `step()` 中此前已经推进过物理并多次读取状态，因此存在重复计算空间。

### 6.14 `_solve_ik(target_pos, initial_guess)`

作用：用阻尼最小二乘法求一组使末端接近目标位置的关节角，仅约束位置，不约束姿态。

每轮迭代计算位置误差：

```text
e = target_pos - ee_pos
```

再读取位置雅可比 `J`，计算：

```text
dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
```

这是阻尼伪逆的一种写法。阻尼项能缓解奇异位形附近矩阵不可逆或关节更新过大的问题。

随后：

```text
dq 限制到 [-0.2, 0.2]
q ← q + 0.2 × dq
q 按关节上下限裁剪
```

最多迭代 500 次；位置误差小于 1 cm 时返回解，否则返回 `None`。

`jacr` 虽然被分配并传给 `mj_jacSite()`，但没有参与计算，因为这套简化 IK 不控制末端姿态。

### 6.15 `_check_collision()`

作用：判断是否有任何非桌面几何体与两张桌子接触。

实现过程：

1. 通过几何体名称找到两张桌子的 ID；
2. 遍历当前所有接触 `data.contact[0:ncon]`；
3. 若 `geom1` 或 `geom2` 是桌子，并且不是桌子互相接触，则返回 `True`。

当前只把“与桌面接触”视为碰撞：

- 不检查机械臂自碰撞是否需要终止；
- 不检查与地面的接触；
- 不区分允许接触桌面的工具与不允许接触的连杆。

实际机器人任务通常会建立碰撞白名单和黑名单，而不是只按单个几何体集合判断。

### 6.16 `render()`

作用：根据 `render_mode` 提供两种渲染路径。

`rgb_array` 路径：

```text
首次调用时创建 mujoco.Renderer
→ 更新起点/目标 site
→ mj_forward
→ renderer.update_scene(data)
→ 返回 RGB NumPy 数组
```

当前 Renderer 实际尺寸是 `width=1280, height=480`，与注释中的 1280×720 不一致。

`human` 路径：

```text
首次调用时启动官方 passive viewer
→ 失败时尝试第三方 viewer
→ 后续调用只做 sync/render
```

一个关键生命周期问题是：`reset()` 会创建新的 `model/data`，但已存在的 viewer 或 offscreen renderer 没有随之重建。多回合可视化或录制时，渲染器可能仍然引用旧模型。

### 6.17 `close()`

作用：释放图形资源。

它会：

- 关闭并清空 human viewer；
- 关闭并删除 offscreen renderer。

训练和评估应始终在结束时调用 `close()`，更规范的入口还会使用 `try/finally`，确保异常发生时也能清理资源。

### 6.18 `if __name__ == "__main__"` 命令行入口

作用：让 `env.py` 既可作为模块导入，也可以独立运行。

参数：

- `--interactive`：启动阻塞式 viewer；
- `--seed`：随机种子；
- `--target-mode`：`fixed/random/none`；
- `--target-pos x y z`：固定目标坐标。

非交互模式下，入口始终给环境零动作，直到成功、碰撞或超时。它是最简单的 smoke test，不是训练过程。

---

## 7. 其他 Python 文件的实现要点

### 7.1 `__init__.py`

调用 `gym.register()`，把字符串 ID `UR5-v0` 映射到 `env:UR5P2PEnv`。注册完成后，其他代码可以写：

```python
env = gym.make("UR5-v0")
```

而不必直接实例化类。

注册依赖导入副作用：必须先导入这个包，注册语句才会执行。当前训练脚本写的是 `import UR5`，但仓库中没有名为 `UR5` 的可导入模块或已安装包，因此入口检查实际报 `ModuleNotFoundError`。

### 7.2 `ik.py`

这是独立的 IK 实验脚本，功能比 `env.py::_solve_ik` 更完整。

- `_load_model()`：从 XML 构建模型，并确保 `ee_site` 存在；
- `_find_site_id()`：按名称查找 site ID；
- `_fk()`：给定关节角，返回末端位置和旋转矩阵；
- `_r_err_from_R()`：把旋转矩阵误差近似转换成三维旋转误差向量；
- `_build_R_from_z()`：给定期望 z 轴构造正交旋转矩阵；
- `_build_R_with_axis()`：让指定局部轴对齐某个世界方向；
- `ik_solve()`：可同时求位置和姿态，支持两阶段求解；
- `main()`：自动判断末端哪根局部轴更接近世界向下方向，然后运行 IK 并打印 FK 验证结果。

它的 `ik_solve()` 也使用阻尼最小二乘，但可把位置雅可比和旋转雅可比堆成 6 维系统：

```text
J6 = [w_pos J_position]
     [w_rot J_rotation]
```

两阶段求解先只处理位置，再加入姿态约束，通常比从第一步同时满足六维误差更容易收敛。

### 7.3 `ik_solver.py`

该文件提供面向对象的运动学接口：

- `IKSolution`：保存关节解、成功状态、迭代次数、位置误差、姿态误差；
- `UR5Kinematics.__init__()`：创建一个无目标、无渲染的环境，并缓存关节限制和雅可比数组；
- `forward_kinematics(q)`：返回 4×4 齐次变换矩阵；
- `inverse_kinematics(...)`：位置/姿态阻尼最小二乘 IK；
- `_set_configuration(q)`：校验维数并写入 `qpos/qvel`。

它给 IK 增加了趋向关节中位值的偏置：

```text
bias = joint_mid - q
dq ← dq + 0.05 × bias
```

意图是远离关节极限，不过严格的零空间优化通常应把该偏置投影到主任务雅可比的零空间；当前代码是直接相加，可能干扰末端任务。

### 7.4 `Algorithms/ppo_continuous_action.py`

主要定义：

- `Args`：训练配置和命令行参数；
- `GoalAugmentWrapper`：曾用于把目标追加到观测，但环境现在已经包含目标，因此已注释停用；
- `make_env()`：创建环境并包装统计、动作裁剪和可选视频录制；
- `layer_init()`：对线性层做正交初始化；
- `Agent`：包含 Actor 均值网络、可学习的 log standard deviation 和 Critic；
- 主程序：采样 rollout、计算 GAE、更新网络、记录指标和保存模型。

Actor 输出高斯分布：

```text
mean = actor_mean(obs)
std  = exp(actor_logstd)
action ~ Normal(mean, std)
```

Critic 输出状态价值 `V(s)`。PPO 使用新旧动作概率比：

```text
ratio = exp(new_log_prob - old_log_prob)
```

并将它限制在 `1±clip_coef` 附近，避免一次更新把策略推得太远。

### 7.5 `Algorithms/td3_ur5.py`

主要定义：

- `Args`：TD3 超参数；
- `make_env()`：创建环境和回合统计 wrapper；
- `QNetwork`：输入 observation 和 action，输出一个 Q 值；
- `Actor`：输入 observation，经 Tanh 和缩放后输出合法动作；
- 主程序：探索、经验回放、双 Critic 更新、延迟 Actor 更新、目标网络软更新。

TD3 的三个关键机制在代码中的位置：

1. 双 Critic：`qf1`、`qf2`，目标值取二者较小值；
2. 目标策略平滑：目标 Actor 动作加入裁剪噪声；
3. 延迟策略更新：只在 `global_step % policy_frequency == 0` 时更新 Actor。

### 7.6 `Scripts/visualize_agent.py`

流程：

```text
创建 human 环境
→ 创建 dummy vector env，只为确定网络输入输出尺寸
→ 构造 PPO Agent
→ 加载 state_dict
→ reset
→ 网络采样动作
→ env.step
→ 直到回合结束
```

它调用 `get_action_and_value()`，因此评估时仍然从高斯策略采样。若希望可复现的确定性演示，通常直接使用 Actor 的均值输出。

### 7.7 `Scripts/record_hd.py`

`HighQualityVideoWrapper` 包装基础环境：

- `reset()`：按触发规则开始新视频；
- `_start_video_recorder()`：创建帧列表并抓取第一帧；
- `step()`：每步抓帧并在回合结束时保存；
- `_save_video()`：用 MoviePy 编码 H.264 MP4；
- `close()`：保存剩余帧并关闭环境。

当前实现会把整个回合的所有高清帧保存在内存中。Full HD RGB 单帧约 6 MB，1024 帧可能占用约 6 GB 内存。实际工程更适合流式写入 ffmpeg，而不是累计所有帧。

---

## 8. 当前代码中需要谨慎学习的地方

下面不是说项目没有价值，而是区分“展示原理的研究代码”和“可长期维护的工程代码”。

### 已通过实际检查确认的问题

1. `python Algorithms/ppo_continuous_action.py --help` 报 `No module named 'UR5'`；
2. `python Algorithms/td3_ur5.py --help` 同样报错；
3. `python Scripts/visualize_agent.py --help` 同样报错；
4. Gymnasium `check_env` 指出 `reset(seed=...)` 没有调用 `super().reset(seed=seed)`。

前三个问题源于项目目录、包名和导入方式没有统一。标准做法是创建合法的小写 Python 包，例如 `ur5_rl`，提供 `pyproject.toml`，然后用 editable install：

```bash
pip install -e .
```

### 结构和生命周期问题

- 每次 reset 都重新编译 MuJoCo 模型，开销大；
- reset 替换 `model/data`，但 viewer/renderer 可能仍指向旧对象；
- 第一次 reset 不会真正使用 IK 初始化；
- `env.py`、`ik.py`、`ik_solver.py` 重复实现 IK，容易逐渐不一致；
- `render()` 同时负责资源创建和逐帧渲染，职责偏多。

### 控制与任务定义问题

- 默认位置模式有效，但环境变量宣称的 torque/velocity 模式没有同步修改 XML 执行器；
- README 把动作称作速度增量，代码实际上使用目标关节角增量；
- interactive viewer 会持续覆盖 Actuation 滑块；
- 姿态奖励平方后无法区分正向和反向；
- 碰撞检测只覆盖桌面接触；
- 进度奖励除以当前距离，接近目标时可能显著放大。

### 脚本和依赖问题

- 可视化脚本从 `cleanrl.cleanrl.ppo_continuous_action` 导入，而本地算法文件实际在 `Algorithms/`；
- `sys.path.insert()` 插入的是 `Scripts` 自身目录，不是项目根目录；
- 视频脚本使用 Pillow 和 MoviePy，但 `requirements.txt` 未显式列出；
- 视频注释写 1280×720，环境实际渲染为 1280×480，再拉伸会改变宽高比；
- `GoalAugmentWrapper` 已失去用途，是遗留代码。

---

## 9. 实际工程中更合理的开发顺序

调用顺序是“入口 → 算法 → 环境 → MuJoCo”，但编写顺序通常几乎相反。

### 第 1 步：先写任务规格，不急着写神经网络

明确：

- 任务目标是什么；
- action 的物理含义和单位；
- observation 每一维是什么；
- 成功、失败、超时如何定义；
- 控制频率和物理 timestep；
- 奖励每一项的数量级。

建议先形成一页 MDP 设计文档。这个步骤决定后续所有接口。

### 第 2 步：建立和验证 MuJoCo 模型

先只处理 XML：

1. 加载模型；
2. 查看关节方向和限制；
3. 确认质量、碰撞几何、执行器；
4. 确认 `ee_site` 的位置和朝向；
5. 用固定控制量运行 viewer。

这时不应引入 Gymnasium 或强化学习。

### 第 3 步：实现唯一的一套运动学模块

依次实现并测试：

1. 正运动学；
2. 位置雅可比；
3. 位置 IK；
4. 姿态误差；
5. 位姿 IK；
6. 关节限制和奇异点处理。

应让环境和工具脚本共同调用同一个 IK 模块，而不是复制三份。

### 第 4 步：实现并验证底层控制器

先选定一种控制语义：

- 位置参考；
- 速度参考；
- 力矩。

确保 XML actuator 与 Python 控制器严格匹配。用阶跃响应验证关节能否稳定追踪，不要同时开放三个未经测试的模式。

### 第 5 步：实现最小 Gymnasium 环境

第一版只需：

- `__init__`；
- `reset`；
- `step`；
- `close`；
- 明确的 action/observation space。

先用零动作和随机动作运行，并通过 `gymnasium.utils.env_checker.check_env`。

### 第 6 步：加入目标、奖励、终止和碰撞

每增加一个机制都写独立测试，例如：

- 目标采样始终位于指定区域；
- 接近目标时奖励增加；
- 远离目标时奖励减少；
- 成功阈值边界正确；
- 桌面碰撞必然终止；
- seed 能复现目标和随机参数。

### 第 7 步：建立无学习基线

至少准备：

- 零动作；
- 随机动作；
- IK 或简单比例控制策略。

如果 IK 控制都无法完成任务，应先修环境和控制器，而不是开始调 PPO。

### 第 8 步：先接一种强化学习算法

选择 PPO 或 TD3 之一，先保证：

- 数据维度正确；
- 回合统计正确；
- 模型可以保存和恢复；
- 小规模训练的 loss 和 reward 合理；
- 评估与训练使用完全相同的预处理。

第一种算法稳定后再接第二种，避免同时排查算法和环境两套问题。

### 第 9 步：独立实现评估与可视化

训练、评估和渲染应分离：

- 训练默认不创建 viewer；
- 评估可选择确定性或随机策略；
- 视频录制使用离屏 renderer；
- renderer 生命周期与 model/data 生命周期一致；
- 视频采用流式编码控制内存。

### 第 10 步：最后做包结构、配置、测试和文档

一个更工程化的目录可以是：

```text
ur5-rl/
├── pyproject.toml
├── src/ur5_rl/
│   ├── __init__.py
│   ├── envs/
│   │   └── ur5_p2p.py
│   ├── simulation/
│   │   ├── model.py
│   │   └── renderer.py
│   ├── control/
│   │   └── joint_position.py
│   ├── kinematics/
│   │   └── solver.py
│   └── rewards.py
├── train/
│   ├── train_ppo.py
│   └── train_td3.py
├── scripts/
│   ├── evaluate.py
│   └── record_video.py
├── assets/
├── configs/
└── tests/
    ├── test_env.py
    ├── test_kinematics.py
    ├── test_reward.py
    └── test_reproducibility.py
```

---

## 10. 推荐学习顺序

不要从 PPO 数学公式直接开始。按以下顺序阅读，会更容易形成完整心智模型：

1. `assets/scene.xml`：看场景怎样包含机器人模型；
2. `assets/ur5e.xml`：看 body、joint、geom、site 和 actuator；
3. `env.py::__init__` 和 `_build_scene_and_model`：看 XML 怎样变成 Python 对象；
4. `reset`：看一个回合怎样初始化；
5. `step`、`_pd_to_ctrl`、`_step_sim`：看动作怎样变成运动；
6. `_ee_pos`、`_check_collision`、`_get_obs`：看仿真状态怎样变成观测；
7. `_compute_reward`：看任务目标怎样变成学习信号；
8. `_solve_ik`：学习雅可比和阻尼最小二乘；
9. `ppo_continuous_action.py`：理解同策略 rollout 和更新；
10. `td3_ur5.py`：理解经验回放、双 Critic 和目标网络；
11. 可视化和视频脚本：理解模型加载和评估流程。

每读一层，都可以打印以下量帮助建立直觉：

```text
action
q_target
qpos / qvel
ee_pos / goal_pos / dist
reward_components
terminated / truncated
```

只要能沿着这组数据从神经网络输出一路追到 MuJoCo 状态，再追到奖励返回，就真正理解了这个项目的主干。
