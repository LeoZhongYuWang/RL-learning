"""Focused regressions for training-only, frozen-policy arrival starts."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import mujoco as mj
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RM75_DEBUG", "0")
import Algorithms.ppo_continuous_action as ppo
from env_rm75 import RM75P2PEnv

MODEL = ROOT / "runs/RM75Direction-v0__RM75-direction-v5-near-coordination__1__1791859371/RM75-direction-v5-near-coordination.best.cleanrl_model"
torch.set_num_threads(1)


class PolicyArrivalTests(unittest.TestCase):
    def wrapper(self, probability=1.0):
        return ppo.ArrivalStartWrapper(
            RM75P2PEnv(require_direction=True, collision_penalty=45),
            MODEL, probability,
        )

    def test_exact_real_arrival_state_and_seeded_repeat(self):
        manual = RM75P2PEnv(require_direction=True, collision_penalty=45)
        wrapped = self.wrapper()
        try:
            obs, _ = manual.reset(seed=20000)
            teacher = ppo.Agent(manual)
            ppo.load_agent_weights(MODEL, teacher, "cpu")
            for prefix in range(1, 513):
                with torch.no_grad():
                    action = torch.tanh(teacher.actor_mean(torch.as_tensor(obs[None])))[0].numpy()
                obs, _, terminated, truncated, info = manual.step(action)
                self.assertFalse(terminated or truncated)
                if info["dist"] <= 0.06:
                    break
            actual, reset = wrapped.reset(seed=20000)
            self.assertTrue(reset["near_start"])
            self.assertTrue(reset["near_start_requested"])
            self.assertEqual(reset["arrival_prefix_steps"], prefix)
            np.testing.assert_allclose(actual, obs)
            for attr in ("qpos", "qvel", "ctrl"):
                np.testing.assert_allclose(getattr(wrapped.unwrapped.data, attr), getattr(manual.data, attr))
            np.testing.assert_allclose(wrapped.unwrapped.q_target, manual.q_target)
            np.testing.assert_allclose(wrapped.unwrapped.goal_pos, manual.goal_pos)
            self.assertGreater(np.linalg.norm(wrapped.unwrapped.data.qvel), 0)
            self.assertEqual(wrapped.unwrapped.step_i, 0)
            self.assertEqual(wrapped.unwrapped.direction_hold_steps, 0)
            self.assertFalse(wrapped.unwrapped.position_reached)
            self.assertFalse(wrapped.unwrapped.direction_reached)
            self.assertEqual(wrapped.unwrapped._compute_milestone_reward(reset["dist"], reset["direction_angle"]), 0)
            repeated, repeated_info = wrapped.reset(seed=20000)
            np.testing.assert_allclose(repeated, actual)
            self.assertEqual(repeated_info["arrival_prefix_steps"], prefix)
            self.assertTrue(all(not p.requires_grad for p in wrapped.teacher.parameters()))
        finally:
            manual.close()
            wrapped.close()

    def test_zero_probability_preserves_original_reset(self):
        ordinary = RM75P2PEnv(require_direction=True)
        wrapped = self.wrapper(probability=0.0)
        try:
            expected, original = ordinary.reset(seed=7)
            actual, info = wrapped.reset(seed=7)
            np.testing.assert_allclose(expected, actual)
            np.testing.assert_allclose(original["q_start"], info["q_start"])
            self.assertFalse(info["near_start_requested"])
            self.assertEqual(info["arrival_prefix_steps"], 0)
        finally:
            ordinary.close()
            wrapped.close()

    def test_failed_prefix_restores_exact_original_state_not_a_new_goal(self):
        ordinary = RM75P2PEnv(require_direction=True, collision_penalty=45)
        wrapped = self.wrapper()
        try:
            expected, _ = ordinary.reset(seed=8)
            with mock.patch.object(wrapped, "prefix_limit", 2):
                actual, info = wrapped.reset(seed=8)
            self.assertFalse(info["near_start"])
            self.assertTrue(info["near_start_requested"])
            self.assertEqual(info["arrival_prefix_steps"], 2)
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_allclose(wrapped.unwrapped.q_target, ordinary.q_target)
            np.testing.assert_allclose(wrapped.unwrapped.goal_pos, ordinary.goal_pos)
            spec = mj.mjtState.mjSTATE_INTEGRATION
            left = np.empty(mj.mj_stateSize(ordinary.model, spec))
            right = left.copy()
            mj.mj_getState(ordinary.model, ordinary.data, left, spec)
            mj.mj_getState(wrapped.unwrapped.model, wrapped.unwrapped.data, right, spec)
            np.testing.assert_allclose(left, right)
        finally:
            ordinary.close()
            wrapped.close()

    def test_collision_or_already_qualified_pose_cannot_be_a_start(self):
        for terminal_info in (
            {"collision": True, "dist": .04, "position_success": True, "direction_success": False},
            {"collision": False, "dist": .04, "position_success": True, "direction_success": True},
        ):
            wrapped = self.wrapper()
            try:
                with mock.patch.object(wrapped.env, "step", return_value=(
                    np.zeros(31, dtype=np.float32), 100., False, False, terminal_info,
                )):
                    _, info = wrapped.reset(seed=20000)
                self.assertFalse(info["near_start"])
                self.assertEqual(info["arrival_prefix_steps"], 1)
                self.assertGreater(info["dist"], .2)
                self.assertEqual(wrapped.unwrapped.direction_hold_steps, 0)
            finally:
                wrapped.close()

    def test_factory_statistics_exclude_prefix_and_resume_restores_source(self):
        args = ppo.Args(env_id="RM75Direction-v0", near_start_probability=1.,
                        near_start_from=str(MODEL))
        env = ppo.make_env(args.env_id, args=args)()
        try:
            self.assertIsInstance(env, gym.wrappers.RecordEpisodeStatistics)
            self.assertIsInstance(env.env, ppo.ArrivalStartWrapper)
            _, info = env.reset(seed=20000)
            self.assertTrue(info["near_start"])
            self.assertEqual(env.episode_lengths, 0)
            self.assertEqual(env.episode_returns, 0)
            _, reward, _, _, info = env.step(np.zeros(7, dtype=np.float32))
            self.assertEqual(env.episode_lengths, 1)
            self.assertAlmostEqual(env.episode_returns, reward)
            self.assertGreater(info["arrival_prefix_steps"], 0)
            self.assertNotIn("near_start_from", ppo.environment_kwargs(args.env_id, args))
            args.near_start_from = None
            ppo.apply_resume_configuration(args, {"args": {"near_start_from": str(MODEL)}})
            self.assertEqual(args.near_start_from, str(MODEL))
        finally:
            env.close()

    def test_evaluation_bypasses_training_teacher(self):
        args = ppo.Args(env_id="RM75Direction-v0", near_start_probability=.5,
                        near_start_from="not-needed-for-evaluation", eval_episodes=1)
        fake = mock.Mock()
        fake.unwrapped = fake
        fake.horizon = 1
        fake.reset.return_value = (np.zeros(31, dtype=np.float32),
                                   {"near_start": True, "dist": .06, "direction_angle_deg": 20})
        fake.step.return_value = (np.zeros(31, dtype=np.float32), 0., True, False,
                                  {"success": True, "dist": .04, "direction_angle_deg": 10})
        agent = mock.Mock(training=True)
        agent.actor_mean.return_value = torch.zeros(1, 7)
        with mock.patch.object(ppo.gym, "make", return_value=fake) as make:
            ppo.evaluate_policy(agent, args, "cpu")
            self.assertEqual(make.call_args.kwargs["near_start_probability"], 0.)
            self.assertNotIn("near_start_from", make.call_args.kwargs)
            ppo.evaluate_policy(agent, args, "cpu", near_start=True)
            self.assertEqual(make.call_args.kwargs["near_start_probability"], 1.)


if __name__ == "__main__":
    unittest.main()
