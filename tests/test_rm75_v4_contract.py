"""Minimal regression tests for the restored RM75 V4 point-task stack."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RM75_DEBUG", "0")

import __init__ as local_registration  # noqa: F401, E402
from Algorithms.ppo_continuous_action import (  # noqa: E402
    Agent,
    Args,
    environment_kwargs,
    should_update_best,
)
from env_rm75 import RM75P2PEnv  # noqa: E402


MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "RM75-v0__RM75-v4-high-success__1__1791429044"
    / "RM75-v4-high-success.best.cleanrl_model"
)


class RM75V4ContractTests(unittest.TestCase):
    def test_registration_and_spaces_match_saved_policy(self):
        self.assertIn("UR5-v0", gym.registry)
        self.assertIn("RM75-v0", gym.registry)
        env = RM75P2PEnv(target_mode="none")
        try:
            self.assertEqual(env.observation_space.shape, (28,))
            self.assertEqual(env.action_space.shape, (7,))
            self.assertTrue(np.all(env.action_space.low == -1.0))
            self.assertTrue(np.all(env.action_space.high == 1.0))
        finally:
            env.close()

    def test_collision_penalty_is_configurable_and_validated(self):
        env = RM75P2PEnv(
            target_mode="none",
            collision_penalty=45.0,
        )
        try:
            self.assertEqual(env.reward_cfg["collision_penalty"], 45.0)
        finally:
            env.close()
        with self.assertRaisesRegex(ValueError, "collision_penalty"):
            RM75P2PEnv(target_mode="none", collision_penalty=-1.0)

    def test_reset_reuses_compiled_model_and_is_seeded(self):
        env = RM75P2PEnv(seed=17, target_mode="random")
        try:
            model_id = id(env.model)
            obs_a, _ = env.reset(seed=17)
            goal_a = env.goal_pos.copy()
            friction_a = env.table_friction.copy()
            obs_b, _ = env.reset(seed=17)
            self.assertEqual(id(env.model), model_id)
            np.testing.assert_allclose(env.goal_pos, goal_a)
            np.testing.assert_allclose(env.table_friction, friction_a)
            np.testing.assert_allclose(obs_b, obs_a)
        finally:
            env.close()

    def test_v4_model_dimensions_and_loading(self):
        self.assertTrue(MODEL_PATH.is_file(), MODEL_PATH)
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.assertEqual(tuple(state["critic.0.weight"].shape), (64, 28))
        self.assertEqual(tuple(state["actor_logstd"].shape), (1, 7))
        env = RM75P2PEnv(target_mode="none")
        try:
            agent = Agent(env)
            agent.load_state_dict(state)
        finally:
            env.close()

    def test_rm75_only_kwargs_do_not_leak_into_ur5(self):
        args = Args()
        args.collision_penalty = 45.0
        rm75_kwargs = environment_kwargs("RM75-v0", args)
        ur5_kwargs = environment_kwargs("UR5-v0", args)
        self.assertEqual(rm75_kwargs["collision_penalty"], 45.0)
        self.assertNotIn("collision_penalty", ur5_kwargs)
        self.assertNotIn("target_sampling", ur5_kwargs)

    def test_best_model_ranking_is_success_first_with_safety_limit(self):
        best = {
            "success_rate": 0.80,
            "collision_rate": 0.00,
            "mean_final_distance": 0.10,
        }
        better = {
            "success_rate": 0.84,
            "collision_rate": 0.08,
            "mean_final_distance": 0.12,
        }
        unsafe = {
            "success_rate": 0.90,
            "collision_rate": 0.11,
            "mean_final_distance": 0.05,
        }
        self.assertTrue(should_update_best(better, best, 0.10))
        self.assertFalse(should_update_best(unsafe, best, 0.10))


if __name__ == "__main__":
    unittest.main()
