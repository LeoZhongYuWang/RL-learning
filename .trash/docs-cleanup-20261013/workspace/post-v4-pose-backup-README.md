# V4 还原前的位姿任务代码备份

该目录保存 2026-10-12 将活动代码恢复到 RM75 V4 点到点版本之前的位姿任务实现。文件按原仓库相对路径保存。

## SHA256

```text
b987affdcef2c99ba5cc01c2b832e837703c35faa6ec956d38c230e08104ac1d  env_rm75.py
f01a70e79edfcb9aaeb1e720026905299bcea7f3f5f47f16cdf3a74677820f1e  Algorithms/ppo_continuous_action.py
d403436d0609a08e2c462f6f6985475263f7e076c6cc421aea00d09455725f33  Scripts/visualize_agent.py
1583ad5a7b5b0f7d12fa53d548ac55051b359edad24e1d3c4fc1a8c03540dbab  __init__.py
7ab914405a90f387a37a96c05a7507554e803eb20594b89b197de3d9b53931fc  Scripts/train_rm75_pose_ppo.sh
e3ad9459d5bc643e4ae1b75896c6dc462923e25ce534862d08990a4b6f760ce3  tests/test_rm75_env.py
394c8549135790e4a4f7d8bd01e53baeadcb391bab3f354f23634b0a29019b6c  tests/test_rm75_ppo_smoke.py
3d3164a7d0ba5717bd858ca58dffd781e10da41a0c934a45937a6b823998a029  tests/test_visualize_agent.py
```

## 说明

- 这些文件对应后续随机切入方向/pose 课程训练代码，不与 V4 的 28 维 observation 契约兼容。
- 本次只把活动训练栈恢复到 V4；已有 pose 模型和 W&B 数据没有删除。
- 若以后恢复 pose，应整体恢复环境、PPO、注册、可视化、启动脚本和测试，不能只替换单个文件。
