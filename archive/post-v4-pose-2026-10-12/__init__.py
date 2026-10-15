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

if "RM75Approach-v0" not in gym.registry:
    gym.register(
        id="RM75Approach-v0",
        entry_point=RM75P2PEnv,
        kwargs={
            "render_mode": None,
            "target_mode": "random",
            "approach_task": True,
        },
        max_episode_steps=None,
    )

if "RM75Pose-v0" not in gym.registry:
    gym.register(
        id="RM75Pose-v0",
        entry_point=RM75P2PEnv,
        kwargs={
            "render_mode": None,
            "target_mode": "random",
            "approach_task": True,
            "pose_task": True,
            "target_direction_mode": "random_cone",
            "target_tilt_max_deg": 30.0,
            "require_pose_feasible": True,
            "pose_feasibility_attempts": 50,
            "orient_success_threshold_deg": 20.0,
        },
        max_episode_steps=None,
    )
