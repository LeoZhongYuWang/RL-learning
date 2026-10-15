"""Necessary reset/routing regressions for mixed-start RM75 direction PPO."""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RM75_DEBUG", "0")
import Algorithms.ppo_continuous_action as ppo
from env_rm75 import RM75P2PEnv


class FakeEvalEnv:
    def __init__(self, near):
        self.near = near
        self.unwrapped = self
        self.horizon = 1
        self.steps = 0

    def reset(self, *, seed):
        return np.zeros(31, dtype=np.float32), {
            "near_start": self.near, "dist": 0.10,
            "direction_angle_deg": 30.0,
        }

    def step(self, action):
        self.steps += 1
        return np.zeros(31, dtype=np.float32), 0.0, True, False, {
            "success": True, "collision": False, "dist": 0.04,
            "direction_angle_deg": 10.0, "position_reached": True,
            "angle_reached": True, "pose_reached": True,
            "min_direction_angle_at_position_deg": 10.0,
        }

    def close(self):
        pass


class MixedStartTests(unittest.TestCase):
    def test_probability_validation_kwargs_and_resume(self):
        for probability in (-0.1, 1.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "near_start_probability"):
                RM75P2PEnv(require_direction=True, near_start_probability=probability)
        with self.assertRaisesRegex(ValueError, "direction"):
            RM75P2PEnv(near_start_probability=0.5)
        args = ppo.Args(near_start_probability=0.5)
        self.assertEqual(ppo.environment_kwargs("RM75Direction-v0", args)["near_start_probability"], 0.5)
        self.assertNotIn("near_start_probability", ppo.environment_kwargs("RM75-v0", args))
        self.assertNotIn("near_start_probability", ppo.environment_kwargs("UR5-v0", args))
        ppo.apply_resume_configuration(args, {"args": {"near_start_probability": 0.25}})
        self.assertEqual(args.near_start_probability, 0.25)

    def test_near_reset_is_seeded_valid_and_does_not_pay_initial_milestones(self):
        env = RM75P2PEnv(require_direction=True, near_start_probability=1.0)
        try:
            obs, info = env.reset(seed=7)
            self.assertTrue(info["near_start"])
            self.assertEqual(obs.shape, (31,))
            self.assertGreaterEqual(info["dist"], 0.04)
            self.assertLessEqual(info["dist"], 0.06)
            self.assertGreaterEqual(info["direction_angle_deg"], 25.0)
            self.assertLessEqual(info["direction_angle_deg"], 35.0)
            self.assertFalse(env._check_collision())
            self.assertEqual(env.direction_hold_steps, 0)
            self.assertFalse(env.position_reached)
            self.assertEqual(env._compute_milestone_reward(info["dist"], info["direction_angle"]), 0.0)
            np.testing.assert_allclose(env.goal_direction, [0.0, 0.0, -1.0])
            np.testing.assert_allclose(env.q_target, env.data.ctrl[:7])
            repeated, repeated_info = env.reset(seed=7)
            np.testing.assert_allclose(repeated, obs)
            self.assertEqual(repeated_info["near_start"], info["near_start"])
        finally:
            env.close()

    def test_failed_ik_restores_original_start_exactly(self):
        ordinary = RM75P2PEnv(require_direction=True)
        near = RM75P2PEnv(require_direction=True, near_start_probability=1.0)
        try:
            expected, _ = ordinary.reset(seed=8)
            with mock.patch.object(near, "solve_direction_ik", return_value=None):
                actual, info = near.reset(seed=8)
            self.assertTrue(info["near_start_requested"])
            self.assertFalse(info["near_start"])
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_allclose(near.goal_pos, ordinary.goal_pos)
            np.testing.assert_allclose(near.start_pos, ordinary.start_pos)
            np.testing.assert_allclose(near.q_target, ordinary.q_target)
            np.testing.assert_allclose(near.data.ctrl, ordinary.data.ctrl)
            self.assertAlmostEqual(near.data.time, ordinary.data.time)
            np.testing.assert_allclose(near.goal_direction, [0.0, 0.0, -1.0])
        finally:
            ordinary.close()
            near.close()

    def test_full_and_near_evaluation_force_distinct_reset_distributions(self):
        args = ppo.Args(env_id="RM75Direction-v0", near_start_probability=0.5, eval_episodes=2)
        agent = mock.Mock(training=True)
        agent.actor_mean.return_value = torch.zeros((1, 7))
        calls = []
        def make_env(env_id, **kwargs):
            calls.append(kwargs["near_start_probability"])
            return FakeEvalEnv(near=kwargs["near_start_probability"] == 1.0)
        with mock.patch.object(ppo.gym, "make", side_effect=make_env):
            full = ppo.evaluate_policy(agent, args, "cpu")
            near = ppo.evaluate_policy(agent, args, "cpu", near_start=True)
        self.assertEqual(calls, [0.0, 1.0])
        self.assertEqual(full["success_rate"], 1.0)
        self.assertEqual(near["success_rate"], 1.0)
        self.assertEqual(near["valid_start_rate"], 1.0)

    def test_failed_near_start_is_not_evaluated_as_full_success(self):
        args = ppo.Args(env_id="RM75Direction-v0", near_start_probability=0.5, eval_episodes=1)
        env = FakeEvalEnv(near=False)
        agent = mock.Mock(training=True)
        agent.actor_mean.return_value = torch.zeros((1, 7))
        with mock.patch.object(ppo.gym, "make", return_value=env):
            metrics = ppo.evaluate_policy(agent, args, "cpu", near_start=True)
        self.assertEqual(env.steps, 0)
        self.assertEqual(metrics["valid_start_rate"], 0.0)
        self.assertEqual(metrics["success_rate"], 0.0)
        self.assertEqual(metrics["pose_reached_rate"], 0.0)
        self.assertTrue(np.isfinite(metrics["mean_final_distance"]))

    def test_near_success_cannot_select_best_full_task_model(self):
        args = ppo.Args(env_id="RM75Direction-v0", near_start_probability=0.5, save_model=False)
        base = {"success_rate": 0.0, "collision_rate": 0.0, "mean_final_distance": 0.10,
                "mean_final_direction_angle_deg": 40.0, "mean_episode_length": 1024,
                "pose_reached_rate": 0.0, "position_success_rate": 0.95,
                "mean_min_direction_angle_at_position_deg": 35.0}
        near = dict(base, success_rate=0.90, valid_start_rate=1.0)
        best = dict(base, success_rate=0.10)
        writer = mock.Mock()
        with mock.patch.object(ppo, "evaluate_policy", side_effect=[base, near]):
            metrics, updated = ppo.evaluate_and_maybe_save_best(
                mock.Mock(), args, "cpu", writer, global_step=2048,
                best_metrics=best, best_model_path="unused", best_metrics_path="unused",
            )
        self.assertFalse(updated)
        self.assertEqual(metrics["success_rate"], 0.10)
        calls = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
        self.assertEqual(calls["eval/success_rate"], 0.0)
        self.assertEqual(calls["eval_near/success_rate"], 0.90)


if __name__ == "__main__":
    unittest.main()
