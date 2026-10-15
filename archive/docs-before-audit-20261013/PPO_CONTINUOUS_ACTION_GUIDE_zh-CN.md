# `ppo_continuous_action.py` 代码导读

本文档围绕 `Algorithms/ppo_continuous_action.py` 展开，重点解释：

1. PPO 文件与 `env.py`、MuJoCo、Gymnasium 和其他脚本的关系；
2. 文件中每个类、函数和训练阶段的作用；
3. observation、action、reward、value、advantage 等数据怎样流动；
4. PPO 损失函数怎样对应到代码；
5. 当前实现中值得学习的部分，以及需要谨慎对待的问题。

---

## 1. PPO 文件在整个项目中的位置

`env.py` 定义任务，PPO 文件学习任务。

```text
assets/*.xml、*.obj
        ↓ 定义机器人和物理世界
env.py::UR5P2PEnv
        ↓ 提供 reset()/step()
Gymnasium Wrapper + SyncVectorEnv
        ↓ 提供批量 observation/reward
ppo_continuous_action.py::Agent
        ↓ 产生 action、估计 value
PPO rollout 和参数更新
        ↓
runs/.../*.cleanrl_model
        ↓
可视化、评估和视频录制脚本
```

PPO 不直接操作 MuJoCo 的 `qpos`、`qvel` 或 `ctrl`。它只通过 Gymnasium 接口发送动作：

```python
next_obs, reward, terminated, truncated, info = env.step(action)
```

环境内部再完成：

```text
action
→ q_target
→ PD 位置执行器
→ MuJoCo 物理步
→ 新 observation 和 reward
```

依赖方向是单向的：

```text
PPO → env.py → MuJoCo
```

`env.py` 不依赖 PPO，所以同一个环境还可以交给 TD3、随机策略、人工控制或其他算法使用。

---

## 2. 一次完整训练的调用链

```text
运行 Algorithms/ppo_continuous_action.py
  ↓
把项目根目录加入 sys.path
  ↓
导入 UR5P2PEnv 并注册 UR5-v0
  ↓
tyro 根据 Args 解析命令行
  ↓
计算 batch_size、minibatch_size、num_iterations
  ↓
初始化 W&B 和 TensorBoard
  ↓
设置随机种子和 CPU/GPU
  ↓
make_env() 创建一个或多个 UR5P2PEnv
  ↓
创建 Agent 和 Adam 优化器
  ↓
预分配 rollout 缓冲区
  ↓
envs.reset()
  ↓
循环执行：
  ├── Actor 根据 observation 采样 action
  ├── Critic 估计 V(s)
  ├── envs.step(action)
  ├── 保存 obs/action/logprob/reward/done/value
  └── 记录已结束回合的统计信息
  ↓
反向计算 GAE advantage 和 return
  ↓
将 rollout 展平成一个 batch
  ↓
打乱并切分 minibatch
  ↓
计算 PPO policy loss、value loss、entropy
  ↓
反向传播并更新网络
  ↓
记录 loss、KL、成功率和 SPS
  ↓
重复下一轮 rollout
  ↓
保存模型，可选评估和上传
  ↓
关闭环境和日志 Writer
```

---

## 3. 导入、路径和环境注册

### 3.1 项目根目录

文件位于：

```text
UR5-Reinforcement-Learning/Algorithms/ppo_continuous_action.py
```

直接运行脚本时，Python 默认优先搜索 `Algorithms/`，而 `env.py` 位于上一级目录。因此代码计算项目根目录：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

再加入模块搜索路径：

```python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

这样才能执行：

```python
from env import UR5P2PEnv
```

### 3.2 注册 `UR5-v0`

```python
if "UR5-v0" not in gym.registry:
    gym.register(
        id="UR5-v0",
        entry_point=UR5P2PEnv,
        kwargs={
            "render_mode": None,
            "target_mode": "random",
        },
        max_episode_steps=None,
    )
```

注册建立了下面的映射：

```text
字符串 "UR5-v0"
        ↓
Python 类 UR5P2PEnv
```

之后可以统一使用：

```python
env = gym.make("UR5-v0")
```

注册前先检查 `gym.registry`，是为了避免同一个进程重复导入文件时出现重复注册警告。

`max_episode_steps=None` 表示不额外使用 Gymnasium 的 `TimeLimit` 限制。环境自己的：

```python
self.horizon = 1024
```

负责产生 `truncated=True`。

---

## 4. `Args`：训练配置和命令行接口

```python
@dataclass
class Args:
```

`Args` 使用 dataclass 描述全部配置，`tyro` 自动把字段转换成命令行参数。

例如：

```python
total_timesteps: int = 1_000_000
learning_rate: float = 3e-4
```

会自动产生：

```bash
--total-timesteps 1000000
--learning-rate 3e-4
```

### 4.1 实验与设备参数

| 参数                    | 默认值 | 含义                              |
| ----------------------- | -----: | --------------------------------- |
| `exp_name`            | 文件名 | 实验名称                          |
| `seed`                |      1 | Python、NumPy、PyTorch 和环境种子 |
| `torch_deterministic` |   True | 是否要求 cuDNN 使用确定性实现     |
| `cuda`                |   True | 有可用 GPU 时是否使用 CUDA        |

### 4.2 日志和模型参数

| 参数                   |   默认值 | 含义                      |
| ---------------------- | -------: | ------------------------- |
| `track`              |     True | 是否连接 Weights & Biases |
| `wandb_project_name` |  `ur5` | W&B 项目名                |
| `wandb_entity`       |     None | W&B 用户或团队            |
| `capture_video`      |    False | 是否尝试录制训练视频      |
| `save_model`         |     True | 训练结束后是否保存权重    |
| `upload_model`       |    False | 是否上传模型              |
| `hf_entity`          | 空字符串 | Hugging Face 用户或组织   |

布尔参数可以这样关闭：

```bash
--no-track
--no-cuda
--no-save-model
```

### 4.3 PPO 核心参数

| 参数                |     默认值 | 含义                          |
| ------------------- | ---------: | ----------------------------- |
| `env_id`          | `UR5-v0` | Gymnasium 环境 ID             |
| `total_timesteps` |  1,000,000 | 计划收集的总环境步数          |
| `learning_rate`   |       3e-4 | Adam 学习率                   |
| `num_envs`        |          1 | 并行环境数量                  |
| `num_steps`       |       1024 | 每个环境每轮 rollout 的长度   |
| `anneal_lr`       |       True | 是否线性衰减学习率            |
| `gamma`           |       0.99 | 折扣因子                      |
| `gae_lambda`      |       0.95 | GAE 的 λ                     |
| `num_minibatches` |         32 | 每轮更新切成多少小批次        |
| `update_epochs`   |         10 | 同一批 rollout 重复训练多少轮 |
| `norm_adv`        |       True | 是否标准化 advantage          |
| `clip_coef`       |        0.2 | PPO 概率比裁剪范围            |
| `clip_vloss`      |       True | 是否裁剪 Critic 更新          |
| `ent_coef`        |       0.01 | 熵奖励权重                    |
| `vf_coef`         |        0.5 | Value loss 权重               |
| `max_grad_norm`   |        0.5 | 梯度范数上限                  |
| `target_kl`       |       None | KL 超过阈值时提前停止当前更新 |

### 4.4 传给环境的参数

```python
target_mode: str = "random"
target_pos: Optional[Tuple[float, float, float]] = None
```

它们最终传给：

```python
UR5P2PEnv(
    target_mode=args.target_mode,
    target_pos=args.target_pos,
)
```

示例：

```bash
python Algorithms/ppo_continuous_action.py \
    --target-mode fixed \
    --target-pos -0.4 0.0 0.2905
```

### 4.5 运行时计算的参数

```python
args.batch_size = args.num_envs * args.num_steps
args.minibatch_size = args.batch_size // args.num_minibatches
args.num_iterations = args.total_timesteps // args.batch_size
```

默认设置下：

```text
batch_size     = 1 × 1024 = 1024
minibatch_size = 1024 ÷ 32 = 32
num_iterations = 1,000,000 ÷ 1024 向下取整 = 976
```

因此代码实际执行的环境步数是：

```text
976 × 1024 = 999,424
```

而不是精确的 1,000,000。

如果使用：

```bash
--total-timesteps 10000
```

则：

```text
num_iterations = 10000 // 1024 = 9
实际步数 = 9 × 1024 = 9216
```

---

## 5. `GoalAugmentWrapper`

```python
class GoalAugmentWrapper(gym.ObservationWrapper):
```

它的原始用途是把三维目标位置追加到环境观测后面：

```text
原 observation + [goal_x, goal_y, goal_z]
```

### `__init__()`

它把观测空间扩展三维：

```python
low = np.concatenate([orig_space.low, extra_low])
high = np.concatenate([orig_space.high, extra_high])
```

### `observation()`

每次环境产生观测时读取：

```python
goal = self.env.goal_pos
```

并追加到观测：

```python
return np.concatenate([observation, goal])
```

但当前 `env.py::_get_obs()` 已经包含：

```text
q(6) + qd(6) + ee(3) + goal(3) + collision(1) = 19 维
```

因此这个 Wrapper 已被注释：

```python
# env = GoalAugmentWrapper(env)
```

如果重新启用，会把目标重复加入一次，使观测从 19 维变成 22 维。这个类属于代码演化后留下的兼容代码，目前不参与实际训练。

---

## 6. `make_env()`：构造一个训练环境

```python
def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
    gamma,
    *,
    args,
):
```

这个函数不是直接返回环境，而是返回内部函数 `thunk`：

```python
def thunk():
    ...
    return env

return thunk
```

这是因为 `SyncVectorEnv` 需要接收“环境构造函数列表”，以便自己决定何时创建环境。

### 6.1 创建基础环境

```python
env = gym.make(
    env_id,
    render_mode=None,
    target_mode=args.target_mode,
    target_pos=args.target_pos,
)
```

训练默认不渲染，因为渲染会降低采样速度。`gym.make()` 最终调用 `UR5P2PEnv.__init__()`。

### 6.2 `RecordEpisodeStatistics`

```python
env = gym.wrappers.RecordEpisodeStatistics(env)
```

它统计每个回合的：

- 累计奖励 `episode["r"]`；
- 回合长度 `episode["l"]`；
- 回合耗时。

这些数据在回合结束后进入 `infos["final_info"]`。

### 6.3 `ClipAction`

```python
env = gym.wrappers.ClipAction(env)
```

PPO 的高斯策略可以输出任意实数，而环境允许的动作范围只有：

```text
[-0.05, 0.05]
```

这个 Wrapper 会在动作进入环境前裁剪：

```text
action_clipped = clip(action, action_low, action_high)
```

`env.py::step()` 内部又裁剪一次，所以当前有双重保护。

### 6.4 `gamma` 参数

`make_env()` 接收 `gamma`，但函数体没有使用它。这通常是从通用 CleanRL 模板保留下来的接口；某些模板会用它配置奖励归一化 Wrapper。

### 6.5 视频路径的当前问题

代码创建环境时固定使用：

```python
render_mode=None
```

但 `RecordVideo` 一般要求环境以 `rgb_array` 模式创建。当前即使 metadata 声明支持 `rgb_array`，实例本身仍然是 `render_mode=None`。因此 `capture_video=True` 的路径并不完整，正确做法应在第一个环境需要录制时使用：

```python
render_mode="rgb_array"
```

---

## 7. `layer_init()`：神经网络初始化

```python
def layer_init(
    layer,
    std=np.sqrt(2),
    bias_const=0.0,
):
```

它对线性层做正交初始化：

```python
torch.nn.init.orthogonal_(layer.weight, std)
```

并把偏置初始化为常数：

```python
torch.nn.init.constant_(layer.bias, bias_const)
```

正交初始化有助于信号在网络各层传播，常用于 PPO 的 MLP。

不同输出层使用不同 `std`：

- 隐藏层：`sqrt(2)`；
- Critic 输出层：`1.0`；
- Actor 输出层：`0.01`。

Actor 输出层使用较小权重，使训练初期动作均值接近零，避免一开始就产生带明显方向偏置的动作均值。

---

## 8. `Agent`：Actor 和 Critic

```python
class Agent(nn.Module):
```

这个类同时包含策略网络 Actor 和价值网络 Critic。

```text
同一个 observation
       ├── Actor → 动作分布
       └── Critic → 状态价值 V(s)
```

两者网络参数独立，没有共享隐藏层。

### 8.1 Critic 网络

```python
self.critic = nn.Sequential(
    Linear(obs_dim, 64),
    Tanh(),
    Linear(64, 64),
    Tanh(),
    Linear(64, 1),
)
```

对于当前环境：

```text
输入：19 维 observation
隐藏层：64
隐藏层：64
输出：1 个标量 V(s)
```

`V(s)` 表示：

> 从当前状态开始，继续按照当前策略行动，未来大约能获得多少折扣累计奖励。

它不是当前一步 reward，也不是距离预测。

### 8.2 Actor 均值网络

```python
self.actor_mean = nn.Sequential(
    Linear(obs_dim, 64),
    Tanh(),
    Linear(64, 64),
    Tanh(),
    Linear(64, action_dim),
)
```

当前环境下：

```text
输入：19 维 observation
输出：6 维动作均值
```

六维分别对应六个关节的目标角度增量。

### 8.3 `actor_logstd`

```python
self.actor_logstd = nn.Parameter(
    torch.zeros(1, action_dim)
)
```

高斯策略除了均值，还需要标准差：

```text
action ~ Normal(mean, std)
```

代码优化的是标准差的对数：

```text
std = exp(logstd)
```

使用 log standard deviation 的好处是 `exp()` 始终为正，不会产生负标准差。

当前 `actor_logstd` 与状态无关：所有状态共享同一组六个探索尺度，但六个关节可以分别学到不同标准差。

初始值为零，因此：

```text
初始 std = exp(0) = 1
```

而环境动作上限只有 `0.05`。这意味着训练初期绝大多数动作都会被裁剪到边界，是当前实现中非常重要的尺度问题。

### 8.4 `get_value(x)`

```python
def get_value(self, x):
    return self.critic(x)
```

只调用 Critic。它主要在 rollout 结束时计算最后状态的 bootstrap value：

```python
next_value = agent.get_value(next_obs)
```

### 8.5 `get_action_and_value(x, action=None)`

第一步计算动作均值：

```python
action_mean = self.actor_mean(x)
```

把可学习 logstd 扩展到 batch 形状：

```python
action_logstd = self.actor_logstd.expand_as(action_mean)
```

计算标准差并创建正态分布：

```python
action_std = torch.exp(action_logstd)
probs = Normal(action_mean, action_std)
```

如果没有传入动作，就从当前策略采样：

```python
if action is None:
    action = probs.sample()
```

rollout 阶段使用这个路径。

如果传入了动作，就计算该动作在当前新策略下的概率。这用于 PPO 更新阶段重新评估旧动作：

```python
agent.get_action_and_value(
    b_obs[mb_inds],
    b_actions[mb_inds],
)
```

返回四项：

```python
return (
    action,
    log_probability,
    entropy,
    value,
)
```

六个关节的 log probability 被求和：

```python
probs.log_prob(action).sum(1)
```

表示假设各动作维度条件独立时，整个六维动作的联合对数概率。

熵同样按动作维度求和，用于衡量策略随机性。

---

## 9. 命令行解析和批大小计算

```python
args = tyro.cli(Args)
```

它把命令行参数解析成 `Args` 实例。

例如：

```bash
python Algorithms/ppo_continuous_action.py \
    --total-timesteps 10000 \
    --num-envs 1 \
    --learning-rate 3e-4 \
    --no-track
```

之后代码计算：

```text
batch_size = num_envs × num_steps
```

一次 PPO 更新使用一整轮 rollout 数据。

默认：

```text
1 个环境 × 1024 步 = 1024 个 transition
```

然后分成：

```text
32 个 minibatch，每个约 32 条数据
```

`batch_size` 最好能被 `num_minibatches` 整除，否则最后一个 minibatch 大小可能不同。

---

## 10. 日志系统

### 10.1 `run_name`

```python
run_name = (
    f"{env_id}__{exp_name}__{seed}__{timestamp}"
)
```

示例：

```text
UR5-v0__ppo_continuous_action__1__1790235282
```

它用于区分不同实验的日志、模型和视频。

### 10.2 Weights & Biases

当：

```python
args.track is True
```

代码调用：

```python
wandb.init(...)
```

并通过：

```python
sync_tensorboard=True
```

同步 TensorBoard 指标。

如果没有登录或不希望联网，应使用：

```bash
--no-track
```

### 10.3 TensorBoard

```python
writer = SummaryWriter(
    f"runs/{run_name}"
)
```

训练指标写入 `runs/`。查看方式：

```bash
tensorboard --logdir runs
```

---

## 11. 随机种子和计算设备

```python
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
```

分别控制：

- Python 标准随机库；
- NumPy 全局随机数；
- PyTorch 随机数。

设备选择：

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available() and args.cuda
    else "cpu"
)
```

MuJoCo 物理仿真通常仍在 CPU 上执行；CUDA 主要加速 Actor/Critic 网络计算。当前网络很小且只有一个环境，GPU 不一定比 CPU 更快。

---

## 12. `SyncVectorEnv` 与环境批次

```python
envs = gym.vector.SyncVectorEnv(
    [make_env(...) for i in range(args.num_envs)]
)
```

`SyncVectorEnv` 把多个普通环境包装成统一的批量接口。

当 `num_envs=1` 时：

```text
单环境 observation 形状：(19,)
向量环境 observation 形状：(1, 19)

单环境 action 形状：(6,)
向量环境 action 形状：(1, 6)
```

当 `num_envs=4` 时：

```text
observation 形状：(4, 19)
action 形状：(4, 6)
reward 形状：(4,)
done 形状：(4,)
```

这里的 `Sync` 表示四个环境在同一进程中顺序执行，并不是四个 CPU 进程真正并行。

---

## 13. 创建 Agent 和优化器

```python
agent = Agent(envs).to(device)
```

Agent 从环境空间自动获取输入输出尺寸，因此没有把 19 和 6 写死。

优化器：

```python
optimizer = optim.Adam(
    agent.parameters(),
    lr=args.learning_rate,
    eps=1e-5,
)
```

Actor、Critic 和 `actor_logstd` 的参数都由同一个 Adam 更新。

---

## 14. Rollout 缓冲区

PPO 是 on-policy 算法。它先用当前策略收集一批数据，再用这批数据更新若干轮，然后丢弃并重新采样。

代码预分配：

```python
obs
actions
logprobs
rewards
dones
values
```

默认形状为：

| 张量         | 默认形状          | 含义                   |
| ------------ | ----------------- | ---------------------- |
| `obs`      | `(1024, 1, 19)` | 每步观测               |
| `actions`  | `(1024, 1, 6)`  | 每步采样动作           |
| `logprobs` | `(1024, 1)`     | 旧策略动作对数概率     |
| `rewards`  | `(1024, 1)`     | 环境奖励               |
| `dones`    | `(1024, 1)`     | 进入当前状态前是否结束 |
| `values`   | `(1024, 1)`     | Critic 的 V(s)         |

这些张量放在 `device` 上。环境输出先在 CPU NumPy 中产生，再转换为 PyTorch 张量。

PPO 的 rollout buffer 与 TD3 的 replay buffer 不同：

```text
PPO：用完当前 rollout 后覆盖，不长期复用
TD3：经验长期保存在 ReplayBuffer 中并随机重复采样
```

---

## 15. 第一次 `reset()`

```python
next_obs, _ = envs.reset(seed=args.seed)
```

最终进入 `env.py::reset()`：

```text
初始化机械臂
→ 生成目标
→ 获取 q、qd、ee、goal、collision
→ 返回 19 维 observation
```

向量环境将其组成 `(num_envs, 19)`，然后转为张量：

```python
next_obs = torch.as_tensor(
    next_obs,
    dtype=torch.float32,
    device=device,
)
```

`next_done` 初始为零，表示刚开始时没有环境处于终止状态。

---

## 16. 外层 iteration 与学习率衰减

```python
for iteration in range(
    1,
    args.num_iterations + 1,
):
```

每个 iteration 包含：

```text
收集一整个 rollout
→ 计算 advantage/return
→ 做 update_epochs 轮网络更新
```

学习率线性衰减：

```python
frac = 1.0 - (iteration - 1.0) / num_iterations
lrnow = frac * learning_rate
```

训练开始时接近 `3e-4`，随后逐步下降，后期更新更谨慎。

---

## 17. Rollout：Actor 与环境交互

内层循环：

```python
for step in range(args.num_steps):
```

每次循环对应所有向量环境各执行一次 `env.step()`。

### 17.1 保存当前状态

```python
obs[step] = next_obs
dones[step] = next_done
```

变量名 `next_obs` 表示“将要用于当前动作的状态”。执行动作后，它会被下一状态覆盖。

### 17.2 采样动作和估计价值

```python
with torch.no_grad():
    action, logprob, _, value = (
        agent.get_action_and_value(next_obs)
    )
```

采样阶段不需要构建梯度图，因此使用 `torch.no_grad()`，减少内存和计算开销。

保存：

```python
actions[step] = action
logprobs[step] = logprob
values[step] = value.flatten()
```

这里的 `logprob` 是“旧策略”在采样当时给这个动作的概率，后面 PPO 要与更新后的“新策略”概率比较。

### 17.3 执行动作

```python
next_obs, reward, terminations, truncations, infos = (
    envs.step(action.cpu().numpy())
)
```

数据需要从 GPU 张量转换为 CPU NumPy，因为 Gymnasium 和 MuJoCo 使用 NumPy。

进入环境后的调用链：

```text
action `(1,6)`
→ ClipAction
→ UR5P2PEnv.step(action `(6,)`)
→ q_target += action
→ 10 个 MuJoCo 子步
→ reward 和新 observation
→ RecordEpisodeStatistics
→ SyncVectorEnv 组成批次
```

### 17.4 合并结束信号

```python
next_done = np.logical_or(
    terminations,
    truncations,
)
```

- `termination`：成功或碰撞；
- `truncation`：达到 1024 步上限。

PPO 当前把二者都视为 done。

### 17.5 回合统计

向量环境通常会自动 reset 已结束的子环境，并把结束前的信息放在：

```python
infos["final_info"]
```

PPO 从中读取：

```python
info["episode"]["r"]
info["episode"]["l"]
info["success"]
```

分别记录回合奖励、长度和任务成功状态。

---

## 18. 一条 transition 的完整数据流

当前时刻 observation：

```text
sₜ = [q, qd, ee_pos, goal_pos, collision]
```

Actor 产生：

```text
aₜ ~ πθ(a|sₜ)
```

环境执行：

```text
q_target ← q_target + aₜ
```

MuJoCo 运行 10 个物理子步，相当于 0.02 秒仿真时间，然后返回：

```text
rₜ、sₜ₊₁、doneₜ
```

缓冲区保存：

```text
(sₜ, aₜ, logπ_old(aₜ|sₜ), rₜ, doneₜ, V_old(sₜ))
```

PPO 不保存 MuJoCo 的完整模型状态，只保存学习所需的观测和统计量。

---

## 19. GAE：从 reward 得到 advantage

Rollout 收集完成后，先计算最后状态的价值：

```python
next_value = agent.get_value(next_obs)
```

然后从最后一步向前递推。

### 19.1 TD error

```python
delta = (
    reward[t]
    + gamma * next_value * next_nonterminal
    - value[t]
)
```

公式：

```text
δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
```

它衡量：

> 实际得到的“一步奖励 + 下一状态价值”，比 Critic 对当前状态的预测高多少。

如果 `delta > 0`，表示结果比预期好；如果小于零，表示结果比预期差。

终止状态不再 bootstrap：

```text
next_nonterminal = 0
```

于是：

```text
δₜ = rₜ - V(sₜ)
```

### 19.2 GAE 递推

```python
advantages[t] = (
    delta
    + gamma
    * gae_lambda
    * next_nonterminal
    * lastgaelam
)
```

公式：

```text
Aₜ = δₜ + γλδₜ₊₁ + (γλ)²δₜ₊₂ + ...
```

`lambda` 控制偏差与方差折中：

- λ 较小：更依赖一步 TD，方差低但偏差可能大；
- λ 接近 1：考虑更长未来，偏差低但方差可能大。

默认：

```text
gamma = 0.99
lambda = 0.95
```

### 19.3 Return

```python
returns = advantages + values
```

即：

```text
目标回报 = 优势 + 旧价值预测
```

`returns` 用来训练 Critic。

---

## 20. 展平 rollout

原数据包含两个批次维度：

```text
[时间步, 环境编号, ...]
```

代码将前两维合并：

```python
b_obs = obs.reshape((-1, 19))
b_actions = actions.reshape((-1, 6))
```

默认 `num_envs=1` 时：

```text
obs:      (1024, 1, 19) → (1024, 19)
actions:  (1024, 1, 6)  → (1024, 6)
rewards:  (1024, 1)     → (1024,)
```

使用多个环境时，不同环境收集的数据会进入同一个训练 batch。

---

## 21. PPO 更新循环

```python
for epoch in range(args.update_epochs):
    np.random.shuffle(b_inds)

    for start in range(
        0,
        batch_size,
        minibatch_size,
    ):
```

同一批 rollout 默认重复训练 10 个 epoch。每个 epoch 都重新打乱索引，减少数据顺序造成的偏差。

PPO 虽然会重复使用这一批数据，但更新完成后就丢弃它；不能像 TD3 一样长期保留旧策略经验，因为 PPO 是 on-policy 算法。

---

## 22. 新旧策略概率比

更新时，把旧动作重新送入当前 Agent：

```python
_, newlogprob, entropy, newvalue = (
    agent.get_action_and_value(
        b_obs[mb_inds],
        b_actions[mb_inds],
    )
)
```

计算：

```python
logratio = newlogprob - old_logprob
ratio = exp(logratio)
```

即：

```text
ratio = π_new(a|s) / π_old(a|s)
```

含义：

- `ratio = 1`：动作概率没有变化；
- `ratio > 1`：新策略更倾向该动作；
- `ratio < 1`：新策略降低了该动作概率。

---

## 23. Advantage 标准化

```python
mb_advantages = (
    mb_advantages - mean
) / (std + 1e-8)
```

这样每个 minibatch 中 advantage 大致满足：

```text
均值 ≈ 0
标准差 ≈ 1
```

作用是减少不同批次奖励尺度变化对梯度大小的影响。

---

## 24. PPO Policy Loss

未裁剪目标：

```python
pg_loss1 = -advantage * ratio
```

裁剪目标：

```python
pg_loss2 = -advantage * clip(
    ratio,
    1 - clip_coef,
    1 + clip_coef,
)
```

最后取：

```python
pg_loss = max(pg_loss1, pg_loss2).mean()
```

因为优化器执行梯度下降，代码写的是 PPO 最大化目标的负值。

默认：

```text
clip_coef = 0.2
ratio 允许的主要范围约为 [0.8, 1.2]
```

直观含义：

```text
好动作：提高它的概率，但一次不要提高太多
坏动作：降低它的概率，但一次不要降低太多
```

裁剪限制单次策略变化，避免网络因为一批样本发生剧烈更新。

---

## 25. Value Loss

Critic 要拟合：

```text
return = advantage + old_value
```

普通平方误差为：

```python
v_loss_unclipped = (
    newvalue - return
) ** 2
```

启用 `clip_vloss` 时，还构造裁剪后的价值：

```python
v_clipped = old_value + clip(
    newvalue - old_value,
    -clip_coef,
    clip_coef,
)
```

然后取两个误差中较大的一个：

```python
v_loss = 0.5 * max(
    loss_unclipped,
    loss_clipped,
).mean()
```

作用类似策略裁剪：避免 Critic 单次变化过大。

---

## 26. Entropy 与总损失

```python
entropy_loss = entropy.mean()
```

熵越高，动作分布越分散，探索越强。

总损失：

```python
loss = (
    policy_loss
    - ent_coef * entropy
    + vf_coef * value_loss
)
```

默认：

```text
ent_coef = 0.01
vf_coef  = 0.5
```

熵前面是负号，因为优化器最小化 loss，而算法希望增大熵。

---

## 27. 反向传播与梯度裁剪

```python
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(
    agent.parameters(),
    max_grad_norm,
)
optimizer.step()
```

顺序是：

```text
清空旧梯度
→ 计算新梯度
→ 限制整体梯度范数
→ Adam 更新参数
```

梯度裁剪可以避免个别异常 batch 导致参数剧烈变化。

---

## 28. KL、clipfrac 和提前停止

近似 KL：

```python
approx_kl = ((ratio - 1) - logratio).mean()
```

它衡量新旧策略差异。

`clipfrac` 表示有多少样本的 ratio 超出了裁剪区间：

```python
abs(ratio - 1) > clip_coef
```

如果设置：

```bash
--target-kl 0.01
```

并且：

```python
approx_kl > target_kl
```

就提前结束当前 rollout 的剩余 update epoch，防止策略更新过度。

---

## 29. 训练指标怎样理解

### `episodic_return`

一个完整回合的奖励总和。它受回合长度、目标距离、碰撞和成功共同影响，不能单独等同于成功率。

### `episodic_length`

回合执行了多少个 `env.step()`。乘以 `0.02` 可近似换算成当前项目的仿真秒数。

### `episodic_success`

该回合是否以 `success=True` 结束。

### `success_rate`

```python
successful_episodes / finished_episodes
```

当前计算的是从训练开始到现在的累计成功率，不是最近 100 个回合的滑动成功率。因此后期改善会被早期大量失败稀释。

### `policy_loss`

Actor 的 PPO 裁剪目标。数值本身不应像监督学习 loss 那样简单地期待单调下降。

### `value_loss`

Critic 预测与 return 的误差。持续很大可能说明奖励尺度过大、价值网络容量不足或训练不稳定。

### `entropy`

动作随机性。训练过程中通常会下降；过早接近很小可能说明探索不足。

### `explained_variance`

```text
1 - Var(return - value) / Var(return)
```

大致解释：

- 接近 1：Critic 对 return 解释得很好；
- 接近 0：与预测常数差不多；
- 小于 0：比简单常数预测还差。

### `SPS`

Steps Per Second，从训练开始到现在平均每个现实秒处理多少环境 transition。

---

## 30. 保存、评估和模型格式

训练结束后：

```python
torch.save(
    agent.state_dict(),
    model_path,
)
```

保存的是 Agent 参数字典，包括：

- Actor 均值网络参数；
- Critic 参数；
- `actor_logstd`。

没有保存：

- Adam 优化器状态；
- 当前 iteration/global_step；
- 随机数状态；
- 环境配置的完整快照。

因此它适合推理和评估，但不能无损恢复到训练中断点继续训练。

当前环境没有安装 CleanRL evaluation helper 时，会保留模型并跳过自动评估，而不是在训练完成后报错。

---

## 31. 与可视化和录制脚本的关系

训练输出：

```text
runs/<run_name>/ppo_continuous_action.cleanrl_model
```

可视化脚本应该：

1. 创建与训练时观测/动作空间一致的环境；
2. 用环境空间构造相同的 `Agent`；
3. 加载 `state_dict`；
4. 调用 Actor 产生动作；
5. 将动作送入 `env.step()`。

如果 `env.py` 的 observation 维度、action 维度或网络结构发生变化，旧模型将无法加载。

当前 `Scripts/visualize_agent.py` 和 `record_hd.py` 仍使用与本地目录不一致的导入路径，需要在正式使用前统一为本项目的 `Algorithms.ppo_continuous_action`。

评估时还应明确：

```python
agent.get_action_and_value(obs)
```

会随机采样动作；若希望确定性演示，应该使用：

```python
action = agent.actor_mean(obs)
```

并正确处理动作范围。

---

## 32. PPO 与 `env.py` 的字段对应关系

| PPO 侧              | 环境侧                | 关系                    |
| ------------------- | --------------------- | ----------------------- |
| `next_obs`        | `_get_obs()`        | 19 维网络输入           |
| `action`          | `step(action)`      | 六个目标关节角增量      |
| `reward`          | `_compute_reward()` | Actor/Critic 的学习信号 |
| `termination`     | 成功或碰撞            | 停止 bootstrap          |
| `truncation`      | 达到 horizon          | 当前也作为 done 处理    |
| `info["success"]` | `dist < 0.05`       | 成功率统计              |
| `episode["r"]`    | Wrapper 累积 reward   | 回合奖励曲线            |
| `episode["l"]`    | Wrapper 统计 step     | 回合长度曲线            |

环境观测与网络输入的精确对应：

```text
Actor/Critic 输入索引

[0:6]    qpos
[6:12]   qvel
[12:15]  ee position
[15:18]  goal position
[18]     collision flag
```

Actor 输出与环境动作对应：

```text
action[0] → shoulder_pan 目标角增量
action[1] → shoulder_lift 目标角增量
action[2] → elbow 目标角增量
action[3] → wrist_1 目标角增量
action[4] → wrist_2 目标角增量
action[5] → wrist_3 目标角增量
```

---

## 33. 当前实现中最需要注意的问题

### 33.1 初始动作标准差与动作范围严重不匹配

初始：

```text
std = exp(0) = 1.0
```

环境范围：

```text
[-0.05, 0.05]
```

因此大量动作被裁剪为 `±0.05`。而 PPO 保存和计算 log probability 的是裁剪前动作，环境实际执行的是裁剪后动作：

```text
策略认为执行了 0.8
环境实际执行了 0.05
```

这破坏了“动作概率与真实执行动作一致”的假设。

可选改进：

- 初始化 `logstd` 为更小值，例如 `log(0.02)`；
- 使用 Tanh-squashed Gaussian，并按 action scale 映射；
- 让策略输出归一化动作，再由环境统一缩放。

### 33.2 `capture_video` 路径不完整

环境以 `render_mode=None` 创建，却尝试挂载 `RecordVideo`。录制环境应使用 `rgb_array`。

### 33.3 `total_timesteps` 向下取整

训练实际步数是完整 batch 的整数倍，可能少于命令行设置值。

### 33.4 Time-limit truncation 被当作真正终止

代码把：

```text
terminated OR truncated
```

统一视为 done，GAE 不再 bootstrap。对于纯时间上限，理论上通常可以使用结束前状态价值继续 bootstrap，避免 time-limit bias。

### 33.5 成功率是累计值

累计成功率反应缓慢。实际训练更常同时记录最近 N 个回合的滑动成功率。

### 33.6 模型只保存权重

不能完整断点续训，也没有把环境参数与权重绑定保存。

### 33.7 `GoalAugmentWrapper` 和 `gamma` 是遗留接口

它们目前不参与实际逻辑，会增加阅读负担。

### 33.8 `env.py` 的 reset seed 仍需标准化

环境自己更新了 RNG，但没有调用 Gymnasium 推荐的：

```python
super().reset(seed=seed)
```

因此官方环境检查仍会报告 seed 接口问题。

---

## 34. 从零编写类似 PPO 文件的合理顺序

### 第一步：先验证环境

确保：

- `check_env` 通过；
- observation/action shape 正确；
- 随机动作能运行多个回合；
- reward 和 done 正确；
- seed 可复现。

### 第二步：只实现 Agent 前向计算

给一个假 observation，检查：

```text
action shape
logprob shape
entropy shape
value shape
```

并确认动作尺度与环境范围一致。

### 第三步：实现单环境 rollout

先收集几十步，打印：

```text
obs、action、reward、done、value、logprob
```

不做参数更新。

### 第四步：加入 GAE

用很短的人工序列验证递推方向和 done 边界。

### 第五步：加入 PPO update

先使用一个 minibatch、一个 epoch，确认：

- loss 有限；
- 没有 NaN；
- 梯度非零；
- 参数发生变化。

### 第六步：加入向量环境和日志

算法正确后再提高采样吞吐量，并接 TensorBoard/W&B。

### 第七步：加入保存、恢复和独立评估

训练入口不应依赖可视化；评估使用单独脚本和确定性策略。

---

## 35. 推荐阅读和调试顺序

按照下面顺序对照源代码：

1. `Args`：先知道训练参数；
2. `make_env()`：理解环境从哪里来；
3. `Agent.__init__()`：看网络输入输出；
4. `get_action_and_value()`：理解高斯策略；
5. rollout 循环：追踪一条 transition；
6. GAE：理解 reward 怎样变成 advantage；
7. policy loss：理解 Actor 为什么改变；
8. value loss：理解 Critic 怎样拟合 return；
9. logging：判断训练是否正常；
10. save/evaluate：理解模型如何被其他脚本使用。

建议重点打印一次：

```python
print("obs shape:", next_obs.shape)
print("action:", action)
print("logprob:", logprob)
print("value:", value)
print("reward:", reward)
print("done:", next_done)
```

再对应查看环境中的：

```python
print("q_target:", env.q_target)
print("ee_pos:", env._ee_pos())
print("goal_pos:", env.goal_pos)
print("reward components:", info["reward_components"])
```

这样能把“神经网络概率分布”与“机械臂真实运动”连接起来。

---

## 36. 最终心智模型

整个 PPO 训练可以压缩成下面六步：

```text
1. 环境给出状态 s
2. Actor 从 π(a|s) 采样动作 a
3. 环境执行动作，返回 reward 和下一状态
4. Critic 估计状态价值，GAE 判断动作比预期好还是差
5. PPO 提高好动作概率、降低坏动作概率，同时限制更新幅度
6. 重复采样和更新，逐渐形成到达随机目标的控制策略
```

在本项目中，这六步对应的物理意义是：

```text
观察关节和目标
→ 产生六个关节参考增量
→ PD + MuJoCo 推进机械臂
→ 根据距离、碰撞和成功计算奖励
→ PPO 更新 Actor/Critic
→ 逐渐学会从观测反馈中控制 UR5
```

真正理解该文件的关键，不是背诵 PPO 公式，而是能沿着下面这条链追踪每个变量：

```text
env observation
→ Actor mean/std
→ sampled action
→ env q_target
→ MuJoCo motion
→ reward
→ advantage
→ PPO loss
→ updated Actor
```
