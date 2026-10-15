# RM75 MuJoCo 迁移实施方案

> **供执行本方案的智能体使用：** 必须逐项执行并检查本方案；推荐使用 `superpowers:subagent-driven-development`，也可以使用 `superpowers:executing-plans`。所有步骤使用复选框（`- [ ]`）记录完成状态。

**目标：** 在保留现有 UR5 基准环境的前提下，把七轴 RM75-B 机械臂接入 MuJoCo 和 Gymnasium，并让 PPO 能通过 `RM75-v0` 环境完成训练。

**总体结构：** 先复制厂家 URDF 和网格，修正 ROS 专用的 `package://` 网格路径，再由 MuJoCo 生成基础 MJCF。基础转换文件与人工审核后的仿真模型分开保存，避免重新转换时覆盖执行器、末端坐标系和仿真参数。RM75 与 UR5 使用不同的场景和环境类，等模型、控制、Gymnasium 及 PPO 测试全部通过后，再决定是否把 RM75 设为默认训练环境。

**技术栈：** Python 3.10、MuJoCo 3.13、dm_control MJCF、Gymnasium、NumPy、PyTorch、pytest。

**模型依据：** `/home/zhongyu/Realman_Control/Rl_ws/src/harvest_arm_rm/urdf/urdf_rm75/RM75.urdf`、同目录 `meshes/`，以及当前项目的 `env.py` 点到点任务。

## 全局约束

- 原始 RM75 URDF 只作为输入，不在 ROS 工作空间中直接修改。
- 在 RM75 全部测试通过前，不删除或覆盖现有 UR5 环境与资产。
- RM75-B 必须识别为 7 个旋转关节，顺序为 `joint_1` 到 `joint_7`。
- 关节范围和最大力矩使用 URDF 数值，不沿用 UR5 参数。
- 原 URDF 没有独立法兰或工具 TCP；第一版暂时把 `link_7` 原点作为 `ee_site`，取得真实工具变换后只修改该 site 的 `pos` 和 `quat`。
- 第一版使用 MuJoCo 位置执行器，环境动作仍表示每一步增加的目标关节角。
- Gymnasium 接口输出 `float32`，MuJoCo 内部状态使用 `float64`。
- UR5 旧 PPO 权重不能用于 RM75：UR5 是 19 维观测、6 维动作；当前 RM75 是 28 维观测、7 维动作。
- 初始仿真增益：1—2 轴 `kp=200, kv=20`；3—4 轴 `kp=120, kv=12`；5—7 轴 `kp=40, kv=4`。这些不是实机控制参数。
- 原 URDF 与 `RM75-B.csv` 的部分质量和惯量不一致。第一版以 URDF 为准；仿真到实机之前必须确认厂家权威版本。

---

## 文件结构

```text
UR5-Reinforcement-Learning/
├── assets/rm75/
│   ├── RM75.urdf              # 修正为相对网格路径的项目副本
│   ├── rm75_imported.xml      # 可重复生成的基础转换结果
│   ├── rm75.xml               # 人工审核并补齐控制定义的模型
│   ├── scene.xml              # 地面、灯光和 RM75 场景
│   └── meshes/*.STL           # 八个原始网格
├── tools/convert_rm75_urdf.py
├── env_rm75.py
├── __init__.py
├── Algorithms/ppo_continuous_action.py
├── tests/test_rm75_model.py
├── tests/test_rm75_env.py
├── tests/test_rm75_ppo_smoke.py
└── README_zh-CN.md
```

职责：

- `rm75_imported.xml` 只保存自动转换结果，可由脚本重新生成。
- `rm75.xml` 保存审核后的结构、执行器、`ee_site`、初始姿态和仿真参数。
- `scene.xml` 负责世界、地面、灯光并包含 `rm75.xml`。
- `env_rm75.py` 负责任务桌面、目标、奖励、重置、步进、碰撞和渲染。
- PPO 从环境空间动态获得网络维度，不硬编码 28 或 7。

---

### 任务 1：建立可重复的 URDF 转换流程

**文件：**

- 新建：`tools/convert_rm75_urdf.py`
- 新建：`tests/test_rm75_model.py`
- 新建：`assets/rm75/RM75.urdf`
- 新建：`assets/rm75/rm75_imported.xml`
- 新建：`assets/rm75/meshes/base_link.STL` 和 `link_1.STL` 到 `link_7.STL`

**接口：**

- 输入：`/home/zhongyu/Realman_Control/Rl_ws/src/harvest_arm_rm/urdf/urdf_rm75`。
- 输出：包含 7 个关节、0 个执行器的 `assets/rm75/rm75_imported.xml`。

- [x] **步骤 1：先编写会失败的转换测试**

新建 `tests/test_rm75_model.py`：

```python
from pathlib import Path
import mujoco as mj

ROOT = Path(__file__).resolve().parents[1]
IMPORTED_MODEL_PATH = ROOT / "assets" / "rm75" / "rm75_imported.xml"

def test_converted_rm75_model_has_seven_joints():
    model = mj.MjModel.from_xml_path(str(IMPORTED_MODEL_PATH))
    assert (model.nq, model.nv, model.njnt) == (7, 7, 7)
    names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    assert names == [f"joint_{i}" for i in range(1, 8)]
```

- [x] **步骤 2：运行测试并确认因文件不存在而失败**

```bash
.venv/bin/python -m pytest tests/test_rm75_model.py::test_converted_rm75_model_has_seven_joints -v
```

- [x] **步骤 3：实现转换脚本**

新建 `tools/convert_rm75_urdf.py`：

```python
from pathlib import Path
import shutil
import mujoco as mj

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(
    "/home/zhongyu/Realman_Control/Rl_ws/src/"
    "harvest_arm_rm/urdf/urdf_rm75"
)
DEST_DIR = PROJECT_ROOT / "assets" / "rm75"
PACKAGE_PREFIX = "package://harvest_arm_rm/urdf/urdf_rm75/"

def main() -> None:
    source_urdf = SOURCE_DIR / "RM75.urdf"
    source_meshes = SOURCE_DIR / "meshes"
    if not source_urdf.is_file():
        raise FileNotFoundError(source_urdf)
    if not source_meshes.is_dir():
        raise FileNotFoundError(source_meshes)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    destination_meshes = DEST_DIR / "meshes"
    if destination_meshes.exists():
        shutil.rmtree(destination_meshes)
    shutil.copytree(source_meshes, destination_meshes)

    urdf_text = source_urdf.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace(PACKAGE_PREFIX, "")
    local_urdf = DEST_DIR / "RM75.urdf"
    local_urdf.write_text(urdf_text, encoding="utf-8")

    model = mj.MjModel.from_xml_path(str(local_urdf))
    if (model.nq, model.nv, model.njnt) != (7, 7, 7):
        raise RuntimeError(
            f"RM75 dimensions are unexpected: "
            f"nq={model.nq}, nv={model.nv}, njnt={model.njnt}"
        )
    mj.mj_saveLastXML(str(DEST_DIR / "rm75_imported.xml"), model)

if __name__ == "__main__":
    main()
```

- [x] **步骤 4：执行转换并重新运行测试**

```bash
.venv/bin/python tools/convert_rm75_urdf.py
.venv/bin/python -m pytest tests/test_rm75_model.py::test_converted_rm75_model_has_seven_joints -v
```

预期：生成本地 URDF、基础 MJCF 和八个 STL；本地 URDF 不再包含 `package://`；测试通过。

- [x] **步骤 5：完成本任务文件；未代替用户创建 Git 提交**

```bash
git add tools/convert_rm75_urdf.py tests/test_rm75_model.py assets/rm75
git commit -m "feat: import RM75 URDF as MuJoCo model"
```

---

### 任务 2：补齐执行器、末端 site 和初始姿态

**文件：**

- 新建：`assets/rm75/rm75.xml`
- 修改：`tests/test_rm75_model.py`

**接口：**

- 输入：任务 1 的 `rm75_imported.xml`。
- 输出：7 个位置执行器、`ee_site` 和 `home` 关键帧。

- [x] **步骤 1：增加控制契约测试**

追加到 `tests/test_rm75_model.py`：

```python
import numpy as np

MODEL_PATH = ROOT / "assets" / "rm75" / "rm75.xml"
JOINT_LIMITS = np.array([
    [-3.106, 3.106], [-2.2689, 2.2689],
    [-3.106, 3.106], [-2.356, 2.356],
    [-3.106, 3.106], [-2.234, 2.234],
    [-6.28, 6.28],
])
FORCE_LIMITS = np.array([60., 60., 30., 30., 10., 10., 10.])

def test_rm75_control_and_tcp_contract():
    model = mj.MjModel.from_xml_path(str(MODEL_PATH))
    assert model.nu == 7
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "ee_site") >= 0
    names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    assert names == [f"joint_{i}_pos" for i in range(1, 8)]
    np.testing.assert_allclose(model.jnt_range, JOINT_LIMITS)
    np.testing.assert_allclose(model.actuator_ctrlrange, JOINT_LIMITS)
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], FORCE_LIMITS)
    np.testing.assert_allclose(model.actuator_forcerange[:, 0], -FORCE_LIMITS)
```

- [x] **步骤 2：运行测试并确认失败**

```bash
.venv/bin/python -m pytest tests/test_rm75_model.py::test_rm75_control_and_tcp_contract -v
```

- [x] **步骤 3：建立审核版 MJCF**

```bash
cp assets/rm75/rm75_imported.xml assets/rm75/rm75.xml
```

在 `<compiler angle="radian"/>` 后加入：

```xml
<option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
```

转换脚本以后只重写 `rm75_imported.xml`，不得覆盖 `rm75.xml`。

- [x] **步骤 4：在 `link_7` 的 body 内添加临时 TCP**

```xml
<site name="ee_site" pos="0 0 0" size="0.008" rgba="1 0 0 1"/>
```

- [x] **步骤 5：在 `</worldbody>` 后添加位置执行器**

```xml
<actuator>
  <position name="joint_1_pos" joint="joint_1" kp="200" kv="20"
            ctrlrange="-3.106 3.106" forcerange="-60 60"/>
  <position name="joint_2_pos" joint="joint_2" kp="200" kv="20"
            ctrlrange="-2.2689 2.2689" forcerange="-60 60"/>
  <position name="joint_3_pos" joint="joint_3" kp="120" kv="12"
            ctrlrange="-3.106 3.106" forcerange="-30 30"/>
  <position name="joint_4_pos" joint="joint_4" kp="120" kv="12"
            ctrlrange="-2.356 2.356" forcerange="-30 30"/>
  <position name="joint_5_pos" joint="joint_5" kp="40" kv="4"
            ctrlrange="-3.106 3.106" forcerange="-10 10"/>
  <position name="joint_6_pos" joint="joint_6" kp="40" kv="4"
            ctrlrange="-2.234 2.234" forcerange="-10 10"/>
  <position name="joint_7_pos" joint="joint_7" kp="40" kv="4"
            ctrlrange="-6.28 6.28" forcerange="-10 10"/>
</actuator>
```

- [x] **步骤 6：添加初始姿态**

```xml
<keyframe>
  <key name="home"
       qpos="-0.003093 0.447318 0.003770 1.458466 -0.000539 0.552761 0"
       ctrl="-0.003093 0.447318 0.003770 1.458466 -0.000539 0.552761 0"/>
</keyframe>
```

在 `ee_site` 位于 `link_7` 原点的前提下，这组角度把末端放到约 `[0.4, 0.0, 0.2905]`。

- [x] **步骤 7：运行模型测试；未代替用户创建 Git 提交**

```bash
.venv/bin/python -m pytest tests/test_rm75_model.py -v
git add assets/rm75/rm75.xml tests/test_rm75_model.py
git commit -m "feat: add RM75 actuators and TCP site"
```

---

### 任务 3：建立 RM75 场景文件

**文件：**

- 新建：`assets/rm75/scene.xml`
- 修改：`tests/test_rm75_model.py`

**接口：**

- 输入：任务 2 的 `rm75.xml`。
- 输出：可以被 RM75 环境直接加载的完整场景。

- [x] **步骤 1：增加场景加载测试**

追加到 `tests/test_rm75_model.py`：

```python
SCENE_PATH = ROOT / "assets" / "rm75" / "scene.xml"

def test_rm75_scene_loads_with_floor():
    model = mj.MjModel.from_xml_path(str(SCENE_PATH))
    assert model.nq == 7
    assert model.nu == 7
    floor_id = mj.mj_name2id(
        model, mj.mjtObj.mjOBJ_GEOM, "floor"
    )
    assert floor_id >= 0
```

- [x] **步骤 2：运行测试并确认因场景不存在而失败**

```bash
.venv/bin/python -m pytest tests/test_rm75_model.py::test_rm75_scene_loads_with_floor -v
```

- [x] **步骤 3：新建 `assets/rm75/scene.xml`**

```xml
<mujoco model="rm75 scene">
  <include file="rm75.xml"/>
  <statistic center="0 0 0.45" extent="0.9"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1"
               specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
             rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker"
             mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane"
              texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane"
          material="groundplane"/>
  </worldbody>
</mujoco>
```

- [x] **步骤 4：运行模型测试；未代替用户创建 Git 提交**

```bash
.venv/bin/python -m pytest tests/test_rm75_model.py -v
git add assets/rm75/scene.xml tests/test_rm75_model.py
git commit -m "feat: add RM75 MuJoCo scene"
```

#### 任务 1—3 实施记录（2026-09-29）

- 已完成任务 1、2、3；任务 4 及以后尚未开始。
- 当前虚拟环境没有安装 `pytest`，所以 `tests/test_rm75_model.py` 使用标准库
  `unittest` 编写并通过 `.venv/bin/python -m unittest tests.test_rm75_model -v`
  执行；测试契约与任务书一致。
- 转换脚本没有使用 `shutil.rmtree()` 删除整个目标网格目录，而是仅覆盖任务书列出的
  8 个 STL，避免误删以后放入 `assets/rm75/meshes/` 的其他用户资产。
- 测试在任务书基础上增加了：本地 URDF 不含 `package://`、8 个网格存在、
  `ee_site` 属于 `link_7`、`home` 的 `qpos/ctrl` 正确，以及场景至少包含一盏灯。
- `rm75.xml` 和 `scene.xml` 使用补丁创建，没有用复制命令覆盖已有文件；重新运行转换
  脚本后两者 SHA-256 保持不变。
- 按工作区python -m unittest tests.test_rm75_env -v约束未替用户创建 Git 提交；任务书中的提交步骤标记为“文件完成、未提交”。

---

### 任务 4：实现七轴 Gymnasium 环境

**文件：**

- 新建：`env_rm75.py`
- 新建：`tests/test_rm75_env.py`

**接口：**

- 输入：`assets/rm75/scene.xml`、关节 `joint_1` 到 `joint_7`、执行器 `joint_1_pos` 到 `joint_7_pos`、site `ee_site`。
- 输出：`RM75P2PEnv(render_mode=None, seed=0, target_mode="random", target_pos=None)`；观测形状 `(28,)`，动作形状 `(7,)`。

- [x] **步骤 1：编写空间和重置测试**

新建 `tests/test_rm75_env.py`：

```python
import unittpython -m unittest tests.test_rm75_env -vest

import numpy as np

from env_rm75 import RM75P2PEnv


class RM75EnvironmentTests(unittest.TestCase):
    def test_spaces_reset_and_compiled_model_contract(self):
        env = RM75P2PEnv(render_mode=None, seed=7)
        try:
            self.assertEqual(env.nq, 7)
            self.assertEqual(env.nu, 7)
            self.assertEqual(env.action_space.shape, (7,))
            self.assertEqual(env.observation_space.shape, (28,))
            obs, info = env.reset(seed=7)
            self.assertEqual(obs.shape, (28,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(info["q_start"].shape, (7,))
        finally:
            env.close()
```

- [x] **步骤 2：运行测试并确认现有实现缺失任务 4 契约**

```bash
.venv/bin/python -m unittest tests.test_rm75_env -v
python -m unittest tests.test_rm75_env -v```

- [x] **步骤 3：从当前 `env.py` 建立 RM75 环境类**

复制 `env.py` 为 `env_rm75.py`，修改模型路径、类名和自由度：

```python
from pathlib import Path

RM75_SCENE_XML = (
    Path(__file__).resolve().parent / "assets" / "rm75" / "scene.xml"
)

class RM75P2PEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None, seed=0,
                 target_mode="random", target_pos=None):
        super().__init__()
        self.nq = 7
        self.nu = 7
```

空间维度改为动态计算：

```python
self.action_scale = np.full(self.nu, 0.02, dtype=np.float64)
self.action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(self.nu,),
    dtype=np.float32,
)

obs_dim = 2 * self.nq + 3 + 3 + 1
obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
self.observation_space = gym.spaces.Box(
    -obs_high, obs_high, dtype=np.float32
)

self.base_mjcf = mjcf.from_path(str(RM75_SCENE_XML))
self.ee_site_name = "ee_site"
```

环境变量改为：

```python
self.ctrl_mode = os.getenv("RM75_CTRL_MODE", "position")
self.debug = os.getenv("RM75_DEBUG", "0") == "1"
self.use_ik_init = os.getenv("RM75_USE_IK", "1") == "1"
self.orient_axis = os.getenv("RM75_ORIENT_AXIS", "z")
```

- [x] **步骤 4：删除 UR5 专用的末端名称回退逻辑**

在 `_build_scene_and_model()` 中只接受 `ee_site`：

```python
ee_site = scene.find("site", "ee_site")
if ee_site is None:
    raise RuntimeError(
        "RM75 model must define site 'ee_site' on link_7"
    )
self.ee_site_name = "ee_site"
```

删除对 `attachment_site`、`wrist_3_link`、`tool0` 和 `flange` 的搜索。

- [x] **步骤 5：编译后验证关节和执行器顺序**

```python
expected_joints = [f"joint_{i}" for i in range(1, 8)]
actual_joints = [
    mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
    for i in range(self.model.njnt)
]
if actual_joints != expected_joints:
    raise RuntimeError(f"Unexpected RM75 joint order: {actual_joints}")
if self.model.nu != self.nu:
    raise RuntimeError(
        f"Expected {self.nu} RM75 actuators, got {self.model.nu}"
    )
```

- [x] **步骤 6：修正随机种子、起点和七维初始角**

`reset()` 开头改为：

```python
super().reset(seed=seed)
if seed is not None:
self.rng = np.random.default_rng(seed)
self.step_i = 0
mj.mj_resetData(self.model, self.data)
self._randomize_table_friction()

self.start_pos = np.array([0.4, 0.0, 0.2905], dtype=np.float64)
q0 = np.array([
    -0.003093, 0.447318, 0.003770, 1.458466,
    -0.000539, 0.552761, 0.0,
], dtype=np.float64)
q0 += self.rng.normal(0.0, 0.02, size=self.nq)
```

设置 `start_pos` 后再调用：

```python
q_start = self._solve_ik(self.start_pos, q0)
```

删除后部对 `self.start_pos` 的重复赋值。

- [x] **步骤 7：保持归一化的增量关节位置动作语义**

```python
normalized_action = np.clip(
    action,
    self.action_space.low,
    self.action_space.high,
).astype(np.float64)
joint_delta = self.action_scale * normalized_action
self.q_target = self.q_target.astype(np.float64) + joint_delta
self.data.ctrl[:self.nu] = q_des[:self.nu]
```

第一版 MJCF 是位置执行器，因此不启用当前环境中的 `torque` 和 `velocity` 分支。

- [x] **步骤 8：运行空间和重置测试**

```bash
RM75_DEBUG=0 .venv/bin/python -m unittest tests.test_rm75_env -v
```

- [x] **步骤 9：增加单步、seed 摩擦和自碰撞测试**

追加到 `tests/test_rm75_env.py`：

```python
class RM75EnvironmentTests(unittest.TestCase):
    def test_zero_and_normalized_action_steps(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=11,
            target_mode="fixed",
            target_pos=[-0.4, 0.0, 0.2905],
        )
        try:
            env.reset(seed=11)
            result = env.step(np.zeros(7, dtype=np.float32))
            obs, reward, terminated, truncated, info = result
            self.assertEqual(obs.shape, (28,))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertEqual(
                set(info["reward_components"]),
                {
                    "progress", "dist_penalty", "time",
                    "orient", "collision", "success",
                },
            )
        finally:
            env.close()
```

- [x] **步骤 10：运行环境测试并记录结果**

```bash
RM75_DEBUG=0 .venv/bin/python -m unittest tests.test_rm75_env -v
```

#### 任务 4 实施记录（2026-09-30）

- 环境动作保持与现有 PPO 一致的归一化 `[-1, 1]`，再通过
  `action_scale=0.02` 转成七个关节的目标角增量；没有采用会让 PPO
  输出大量饱和的物理动作空间 `[-0.05, 0.05]`。
- 固定场景只在环境构造时编译一次。`reset(seed=...)` 使用
  `mj_resetData()`，并直接修改 `model.geom_friction`，不重新编译模型。
- 桌面滑动摩擦系数使用环境自身的 `self.rng`，相同 reset seed 得到
  相同摩擦、初始关节扰动和随机目标。
- 运行时验证关节顺序、执行器顺序以及执行器到关节的映射。
- 碰撞终止包含机械臂与桌面碰撞，以及 RM75 非相邻连杆之间的
  自碰撞。基座—link_1、link_1—link_2 等相邻关节连接处的结构接触
  被排除，避免 home 姿态误判。
- 当前 MJCF 使用位置执行器，因此环境拒绝 `torque` 和 `velocity`
  控制模式。
- `tests/test_rm75_env.py` 改用标准库 `unittest`，不要求项目额外安装
  `pytest`；包含 reset、单步、动作映射、seed 摩擦、自碰撞分类和真实
  自碰撞姿态测试。
- 本任务没有代替用户创建 Git 提交。

---

### 任务 5：注册 `RM75-v0` 并保留 `UR5-v0`

**文件：**

- 修改：`__init__.py`
- 修改：`tests/test_rm75_env.py`

**接口：**

- 输入：任务 4 的 `RM75P2PEnv`。
- 输出：可以通过 `gym.make("RM75-v0", ...)` 创建的环境。

- [x] **步骤 1：增加注册测试**

追加到 `tests/test_rm75_env.py`：

```python
import importlib
import unittest

import gymnasium as gym

class RM75EnvironmentTests(unittest.TestCase):
    def test_gym_registration_keeps_ur5_and_adds_rm75(self):
        registry_module = importlib.import_module("__init__")
        importlib.reload(registry_module)
        self.assertIn("UR5-v0", gym.registry)
        self.assertIn("RM75-v0", gym.registry)
```

- [x] **步骤 2：运行测试并确认环境尚未注册**

```bash
RM75_DEBUG=0 .venv/bin/python -m unittest \
  tests.test_rm75_env.RM75EnvironmentTests.test_gym_registration_keeps_ur5_and_adds_rm75 -v
```

- [x] **步骤 3：同时注册两个环境**

把 `__init__.py` 更新为：

```python
import gymnasium as gym
from env import UR5P2PEnv
from env_rm75 import RM75P2PEnv

if "UR5-v0" not in gym.registry:
    gym.register(
        id="UR5-v0",
        entry_point=UR5P2PEnv,
        kwargs={"render_mode": None, "target_mode": "random"},
        max_episode_steps=None,
    )

if "RM75-v0" not in gym.registry:
    gym.register(
        id="RM75-v0",
        entry_point=RM75P2PEnv,
        kwargs={"render_mode": None, "target_mode": "random"},
        max_episode_steps=None,
    )
```

- [x] **步骤 4：再次运行注册测试并记录结果**

```bash
RM75_DEBUG=0 .venv/bin/python -m unittest \
  tests.test_rm75_env.RM75EnvironmentTests.test_gym_registration_keeps_ur5_and_adds_rm75 -v
```

#### 任务 5 实施记录（2026-09-30）

- `__init__.py` 同时注册 `UR5-v0` 和 `RM75-v0`，两个注册都先检查
  `gym.registry`，因此重复导入不会覆盖已有环境或产生注册警告。
- 注册测试分别通过 `gym.make()` 创建 UR5 和 RM75；UR5 reset 返回
  `(19,)` 观测，RM75 reset/step 返回 `(28,)` 观测和有限奖励。
- 本任务只完成公共环境注册，没有提前修改 PPO 默认环境；任务 6
  仍保持未完成状态。
- 本任务没有代替用户创建 Git 提交。

---

### 任务 6：把 PPO 连接到 `RM75-v0`

**文件：**

- 修改：`Algorithms/ppo_continuous_action.py`
- 新建：`tests/test_rm75_ppo_smoke.py`

**接口：**

- 输入：观测形状 `(28,)`、动作形状 `(7,)` 的 `RM75-v0`。
- 输出：28 个输入、7 个高斯动作输出的 PPO 智能体。

- [x] **步骤 1：增加 PPO 网络维度测试**

新建 `tests/test_rm75_ppo_smoke.py`：

```python
import gymnasium as gym
import torch
from Algorithms.ppo_continuous_action import Agent, Args, make_env

def test_ppo_builds_for_rm75_spaces():
    args = Args()
    args.env_id = "RM75-v0"
    args.target_mode = "none"
    envs = gym.vector.SyncVectorEnv([
        make_env(
            args.env_id, 0, False, "rm75-test",
            args.gamma, args=args,
        )
    ])
    try:
        agent = Agent(envs)
        obs, _ = envs.reset(seed=5)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action, logprob, entropy, value = (
            agent.get_action_and_value(obs_tensor)
        )
        assert action.shape == (1, 7)
        assert logprob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1, 1)
    finally:
        envs.close()
```

- [x] **步骤 2：运行测试并核对迁移前状态**

```bash
RM75_DEBUG=0 .venv/bin/python -m pytest tests/test_rm75_ppo_smoke.py -v
```

- [x] **步骤 3：在 PPO 入口注册 RM75**

在 `Algorithms/ppo_continuous_action.py` 中导入：

```python
from env import UR5P2PEnv
from env_rm75 import RM75P2PEnv
```

保留 `UR5-v0` 注册并添加：

```python
if "RM75-v0" not in gym.registry:
    gym.register(
        id="RM75-v0",
        entry_point=RM75P2PEnv,
        kwargs={"render_mode": None, "target_mode": "random"},
        max_episode_steps=None,
    )
```

把 `Args` 默认环境修改为：

```python
env_id: str = "RM75-v0"
```

不要在 `Agent` 中写死 28 和 7，继续使用：

```python
envs.single_observation_space.shape
envs.single_action_space.shape
```

- [x] **步骤 4：运行网络测试**

```bash
RM75_DEBUG=0 .venv/bin/python -m pytest tests/test_rm75_ppo_smoke.py -v
```

- [x] **步骤 5：执行 8 个时间步的训练冒烟测试**

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

预期：退出码为 0，损失为有限数值，并输出 `SPS:`。

- [x] **步骤 6：记录本任务（不代替用户创建 Git 提交）**

```bash
git add Algorithms/ppo_continuous_action.py tests/test_rm75_ppo_smoke.py
git commit -m "feat: train PPO on RM75 environment"
```

#### 任务 6 实施记录（2026-09-30）

- `Algorithms/ppo_continuous_action.py` 已具备幂等的 `RM75-v0` 注册，
  `Args.env_id` 默认为 `RM75-v0`，Agent 的输入和输出维度继续由环境空间
  动态计算，没有写死 28 和 7。
- 新增 `tests/test_rm75_ppo_smoke.py`，确认观测 `(28,)`、动作 `(7,)`，
  并检查 action、logprob、entropy、value 的形状、有限值和动作边界。
- 8 步 CPU PPO 更新退出码为 0，输出 `SPS: 515`，证明采样、GAE、
  PPO loss、反向传播和优化器更新链路可执行。
- 本任务没有代替用户创建 Git 提交。

---

### 任务 7：验证运动学、控制稳定性、TCP 和碰撞

**文件：**

- 根据测量结果修改：`assets/rm75/rm75.xml`
- 仅在测试暴露具体问题时修改：`env_rm75.py`
- 修改：`tests/test_rm75_env.py`

**接口：**

- 输入：完整 RM75 环境和七个位置执行器。
- 输出：稳定初始姿态、可用目标区域、正确 TCP 和经过检查的碰撞行为。

- [x] **步骤 1：增加一秒钟保持稳定性测试**

追加到 `tests/test_rm75_env.py`：

```python
def test_rm75_home_hold_is_stable_for_one_second():
    env = RM75P2PEnv(
        render_mode=None, seed=13, target_mode="none"
    )
    try:
        env.reset(seed=13)
        q_start = env.data.qpos[:7].copy()
        for _ in range(50):
            env.step(np.zeros(7, dtype=np.float32))
        q_end = env.data.qpos[:7].copy()
        assert np.max(np.abs(q_end - q_start)) < 0.05
    finally:
        env.close()
```

50 个环境步约等于一秒，因为每个环境步包含 10 个 `0.002 s` 物理子步。

- [x] **步骤 2：运行稳定性测试并按固定规则调参**

```bash
RM75_DEBUG=0 .venv/bin/python -m pytest tests/test_rm75_env.py::test_rm75_home_hold_is_stable_for_one_second -v
```

若某轴振荡，每次只把该轴 `kp` 降低 25%、`kv` 降低 10%，随后重新测试。若近端关节因重力明显下沉，每次把该轴 `kp` 和 `kv` 增加 25%，同时保持 `forcerange` 不超过 URDF effort 上限。

- [x] **步骤 3：运行可视化检查并记录实机检查边界**

为 `env_rm75.py` 增加与 `env.py` 相同的命令行入口，然后运行：

```bash
RM75_DEBUG=1 .venv/bin/python env_rm75.py \
  --interactive \
  --target-mode random
```

逐项确认：

```text
1. 连杆顺序和关节旋转方向与真实 RM75 一致。
2. 初始姿态不穿过地面、桌面或自身。
3. 红色 ee_site 位于预期法兰或工具中心。
4. 起点和目标点标记均可见。
5. 零动作时机械臂没有持续振荡。
6. 七个关节滑块分别驱动正确关节。
```

- [x] **步骤 4：核对 TCP 状态并记录实机标定依赖**

取得 `link_7` 到工具中心的平移 `[x, y, z]` 和 MuJoCo 四元数 `[w, x, y, z]` 后，把临时 site 改成真实数值：

```xml
<site name="ee_site"
      pos="x y z"
      quat="w x y z"
      size="0.008"
      rgba="1 0 0 1"/>
```

重新运行模型、环境、IK 和稳定性测试。不得通过移动目标位置补偿错误 TCP。

- [x] **步骤 5：检查随机目标区域**

分别采样当前目标盒的四角和中心。每个目标只有同时满足以下条件才能保留：

```text
位置误差小于 0.01 m；
所有关节位于 URDF 限位内；
机械臂不与地面、桌面或自身碰撞；
IK 结果不含 NaN 或无穷值。
```

若原 UR5 目标范围存在不可达点，应根据验证结果收缩 RM75 的 `target_random_box`，不得扩大关节限位。

- [x] **步骤 6：记录验证后的参数（不代替用户创建 Git 提交）**

```bash
git add assets/rm75/rm75.xml env_rm75.py tests/test_rm75_env.py
git commit -m "test: validate RM75 control and TCP behavior"
```

#### 任务 7 实施记录（2026-09-30）

- 50 个环境步等于 1 秒仿真时间；零动作保持的最大关节漂移为
  `0.034795243 rad`，无碰撞、无提前终止，低于 `0.05 rad` 门槛，
  因此没有修改执行器增益。
- IK 内部停止误差由 1 cm 收紧为 5 mm，外部目标区域验收门槛仍为
  1 cm。当前目标盒四角与中心的位置误差分别为 `0.004599364`、
  `0.004822495`、`0.004950465`、`0.004077626`、`0.004305936 m`；
  所有关节解均为有限值、在 URDF 限位内且无桌面/自碰撞，因此无需
  收缩 `target_random_box`。
- 离屏渲染成功生成 `(480, 1280, 3)` 图像，机械臂、两张桌子、地面和
  目标标记均正常显示。模型测试确认 `ee_site` 属于 `link_7`。
- 碰撞分类已补齐地面语义：固定底座与地面的安装接触允许，其余运动
  连杆接触地面会被判为碰撞；桌面碰撞和非相邻自碰撞规则保持不变。
- 当前 `ee_site pos="0 0 0"` 仍是临时 TCP。由于没有实机工具变换测量，
  本任务没有编造 `pos/quat`；真实 TCP 和真实控制器关节正方向仍需实机
  低速标定后再修正并重跑全部测试。
- 本任务没有代替用户创建 Git 提交。

---

### 任务 8：优化碰撞模型并补充中文文档

**文件：**

- 根据性能检查修改：`assets/rm75/rm75.xml`
- 修改：`README_zh-CN.md`

**接口：**

- 输入：通过功能验证的 RM75 模型和环境。
- 输出：模型更新规则、验证命令、训练命令和碰撞模型说明。

- [x] **步骤 1：检查 STL 碰撞性能**

先用完整 STL 运行 1000 个无渲染环境步：

```bash
RM75_DEBUG=0 .venv/bin/python - <<'PY'
import time
import numpy as np
from env_rm75 import RM75P2PEnv

env = RM75P2PEnv(render_mode=None, target_mode="none")
env.reset(seed=0)
started = time.perf_counter()
for _ in range(1000):
    env.step(np.zeros(7, dtype=np.float32))
elapsed = time.perf_counter() - started
env.close()
print(f"1000 steps: {elapsed:.3f} s")
PY
```

若接触抖动或速度不能接受，保留 STL geom 作为视觉模型，并增加盒体、圆柱体或胶囊体碰撞代理。视觉 geom 设置：

```xml
contype="0" conaffinity="0" group="2"
```

碰撞代理设置：

```xml
group="3" rgba="0 0 0 0"
```

每增加一个代理体，都要确认它覆盖对应连杆，并且不与相邻连杆在初始姿态中持续穿透。

- [x] **步骤 2：在中文 README 中记录转换规则**

记录命令：

```bash
.venv/bin/python tools/convert_rm75_urdf.py
```

同时说明：该命令只重新生成 `rm75_imported.xml`，不会覆盖审核后的 `rm75.xml`。上游 URDF 改变后，应比较两个 MJCF，再把确认后的运动学和惯量变化移植到 `rm75.xml`，保留执行器、site、关键帧和物理参数。

- [x] **步骤 3：记录验证、可视化和训练命令**

```bash
RM75_DEBUG=0 .venv/bin/python -m unittest discover -s tests -v

.venv/bin/python env_rm75.py \
  --interactive \
  --target-mode random

RM75_DEBUG=0 .venv/bin/python Algorithms/ppo_continuous_action.py \
  --env-id RM75-v0 \
  --total-timesteps 1000000 \
  --num-envs 1 \
  --learning-rate 1e-4 \
  --no-track
```

完成 W&B 登录后，可以把 `--no-track` 换成 `--track`。

- [x] **步骤 4：记录检查点不兼容说明**

```text
UR5 PPO 检查点不能加载到 RM75 PPO 智能体中。
UR5 使用 19 维观测和 6 维动作；当前 RM75 使用 28 维观测和 7 维动作。
RM75 必须从头训练，或另行设计明确的迁移学习方法。
```

- [x] **步骤 5：记录文档与碰撞结论（不代替用户创建 Git 提交）**

```bash
git add README_zh-CN.md assets/rm75/rm75.xml
git commit -m "docs: add RM75 conversion and training guide"
```

#### 任务 8 实施记录（2026-09-30）

- 完整 STL 碰撞模型运行 1000 个无渲染环境步用时 `0.212360 s`，约
  `4709.0 env steps/s`（`47089.7` 个物理子步每秒）；碰撞步数为 0，
  最大关节漂移为 `0.034816259 rad`。
- 当前模型没有出现性能或接触稳定性问题，因此保留准确的 STL 碰撞
  几何，没有增加盒体/圆柱体/胶囊体代理，也没有修改 `rm75.xml`。
- `README_zh-CN.md` 已补充模型文件关系、转换规则、验证/可视化/训练
  命令、临时 TCP 边界、STL 基准结果、正常退出方法和 UR5/RM75
  checkpoint 不兼容说明。
- 转换脚本重新运行成功：`nq=7, nv=7, njnt=7, nu=0`；它只刷新本地
  URDF、STL 和 `rm75_imported.xml`，不会覆盖审核后的训练模型。
- 本任务没有代替用户创建 Git 提交。

---

## 最终验证

运行全部无渲染测试：

```bash
RM75_DEBUG=0 .venv/bin/python -m unittest discover -s tests -v
```

运行 Gymnasium 环境检查：

```bash
RM75_DEBUG=0 .venv/bin/python - <<'PY'
from gymnasium.utils.env_checker import check_env
from env_rm75 import RM75P2PEnv

env = RM75P2PEnv(
    render_mode=None, seed=0, target_mode="random"
)
check_env(env, skip_render_check=True)
env.close()
print("RM75 check_env: PASS")
PY
```

运行短训练：

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

最终必须满足：

```text
model.nq = 7
model.nv = 7
model.nu = 7
action_space.shape = (7,)
observation_space.shape = (28,)
ee_site 位于 link_7 或经过确认的工具 TCP
七个控制范围与 URDF 关节范围一致
七个最大力矩范围与 URDF effort 一致
零动作保持一秒后，各关节偏差小于 0.05 rad
Gymnasium check_env 检查通过
PPO 构建 28 输入、7 输出的策略并完成一次参数更新
```

只有上述检查全部通过，才把 RM75 用于长时间训练。连接真实机械臂之前，还必须单独完成实机限速、急停、碰撞保护、坐标系标定和通信异常处理；仿真执行器增益不得直接作为实机控制参数。

### 最终验证记录（2026-09-30）

- `unittest discover`：17 项全部通过。
- Gymnasium `check_env(..., skip_render_check=True)`：通过；仅提示观测空间使用
  `±inf` 边界，该提示不影响 API 正确性。
- Python 语法检查：通过。
- 最终 8 步 CPU PPO 更新：退出码 0，输出 `SPS: 907`。
- 无头训练进程继续使用项目原有 GLFW 选择；在没有 `DISPLAY` 的终端会出现
  GLFW 警告，但训练本身正常完成。强制切换 EGL 在当前 PyTorch 训练进程中
  会触发退出码 139，因此没有用“消除警告”的改动替换可工作的配置。
- 软件侧任务 1–8 已完成。真实工具 TCP 和真实控制器关节正方向仍需拿到
  实机测量/点动结果后执行，不属于仿真代码能够自行推断的数据。

### 100 万步结果后的 V2 修正（2026-09-30）

- 原训练的 1170 个回合中，746 个超时回合平均仍获得正奖励，最终策略
  表现为低碰撞但大量停在目标外。V2 增加 `timeout_penalty=25.0`，并把
  `dist_weight` 从 `0.02` 提高到 `0.1`。
- PPO 的 `ent_coef` 从 `0.001` 降到 `0.0001`；归一化动作标准差上限
  从 `0.20` 线性衰减到 `0.08`。两次迭代的验证日志分别记录 `0.20`
  和 `0.08`，证明首尾调度生效。
- 保留累计位置目标控制，并把 `q_target-q` 的七维跟踪误差加入观测，
  观测从 21 维变为 28 维。这比每步改成 `q+delta` 更适合当前位置执行器，
  因为它既恢复马尔可夫可观测性，又不会放弃对未完成目标的持续保持。
- 旧 21 维 RM75 checkpoint 与 V2 不兼容，必须重新训练；原 UR5 checkpoint
  的结构加载兼容性已单独验证通过。
- 按用户要求，本轮未修改学习率衰减、1024 步 horizon、最佳 checkpoint
  保存逻辑和目标课程。
- 完整回归结果：18 项 unittest、Gymnasium `check_env`、语法检查均通过；
  8 步 PPO 更新成功。测试产生的临时 run 已删除。

### 低成功率结果后的 V3 修正（2026-09-30）

- V2 训练到约 108 万步时只有 1 个成功回合；中期最终距离曾下降，随后碰撞率和
  最终距离重新上升，说明策略并非完全没有学到接近方向，而是缺少足够成功样本，
  且探索从训练开始就持续收缩。
- 将 `timeout_penalty` 从 `25.0` 降为 `10.0`，将 `dist_weight` 从 `0.1`
  降为 `0.03`，避免长回合的累计负奖励压过稀有成功信号。
- 增加一次性距离里程碑：每回合首次进入 20 cm 奖励 `+5`、首次进入
  10 cm 奖励 `+15`。同一里程碑不会重复奖励，因此来回摆动不能刷分。
- 归一化动作标准差改为：前 25% 训练保持 `0.30`，之后线性下降到 `0.12`；
  `ent_coef` 保持 `0.0001`。
- 继续保留 28 维观测、`action_scale=0.02`、学习率退火、1024 步 horizon、
  最终模型保存方式和随机目标范围。
