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

if "RM75Direction-v0" not in gym.registry:
    gym.register(
        id="RM75Direction-v0",
        entry_point=RM75P2PEnv,
        kwargs={
            "render_mode": None,
            "target_mode": "random",
            "require_direction": True,
        },
        max_episode_steps=None,
    )
