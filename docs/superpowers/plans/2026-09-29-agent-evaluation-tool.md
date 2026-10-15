# UR5 PPO 模型评估工具实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复训练模型可视化脚本，并提供可复现的随机目标批量评估结果。

**Architecture:** `Scripts/visualize_agent.py` 直接导入项目根目录中的环境和 PPO `Agent`，以确定性策略作为默认评估方式。默认打开 MuJoCo 人工查看窗口；`--headless` 模式关闭渲染并汇总成功、碰撞、超时、距离、回报和步数。

**Tech Stack:** Python 3.10、MuJoCo、Gymnasium、PyTorch、标准库 argparse。

**Spec:** 用户需要加载训练完成的 `.cleanrl_model`，观察随机目标轨迹并判断能否完成任务。

## Global Constraints

- 不修改训练得到的模型文件。
- 评估默认使用 `tanh(actor_mean)`，避免随机探索影响真实性能。
- 保留可选随机采样模式，用于评估训练时策略分布。
- 明确区分末端随机点到达与真实夹爪抓取。
- 相对模型路径从当前目录或项目根目录解析；未指定时自动选择最新模型。

---

### Task 1: 修复模型加载与可视化

**Files:**
- Modify: `Scripts/visualize_agent.py`

**Interfaces:**
- Consumes: `UR5P2PEnv`、`Algorithms.ppo_continuous_action.Agent`、`.cleanrl_model`。
- Produces: `main()` 命令行入口，以及每回合和汇总评估结果。

- [x] 删除不存在的 `UR5` 和 `cleanrl.cleanrl` 导入。
- [x] 将项目根目录加入 `sys.path`，直接创建环境和 Agent。
- [x] 未传 `--model-path` 时选择 `runs/` 下最新模型。
- [x] 默认使用确定性动作，增加 `--stochastic` 可选随机动作。
- [x] 增加 `--headless` 批量评估模式和统计汇总。
- [x] 人工查看模式按仿真步长限速，便于观察运动。

### Task 2: 验证工具与最终模型

**Files:**
- Verify: `Scripts/visualize_agent.py`
- Verify: `runs/UR5-v0__ppo_continuous_action__1__1790669799/ppo_continuous_action.cleanrl_model`

**Interfaces:**
- Consumes: Task 1 的命令行工具。
- Produces: 编译、帮助命令、短批量评估与模型性能报告。

- [x] 运行 Python 编译检查。
- [x] 运行 `--help`，确认不再出现 `No module named 'UR5'`。
- [x] 运行10回合无界面随机目标评估。
- [x] 运行至少100回合无界面随机目标评估，报告成功率和碰撞率。
- [x] 运行现有训练契约测试，确认环境与 Agent 行为未受影响。
- [x] 检查差异格式并记录可复制命令。
