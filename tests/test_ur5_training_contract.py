import unittest

import numpy as np
import torch

from Algorithms.ppo_continuous_action import Agent
from env import UR5P2PEnv


class UR5TrainingContractTests(unittest.TestCase):
    def test_normalized_action_maps_to_bounded_joint_delta(self):
        env = UR5P2PEnv(
            render_mode=None,
            seed=3,
            target_mode="none",
        )
        try:
            env.reset(seed=3)
            np.testing.assert_allclose(env.action_space.low, -1.0)
            np.testing.assert_allclose(env.action_space.high, 1.0)

            q_target_before = env.q_target.copy()
            env.step(np.ones(env.nu, dtype=np.float32))
            requested_delta = env.q_target - q_target_before

            self.assertTrue(
                np.all(requested_delta <= env.action_scale + 1e-12)
            )
            self.assertTrue(
                np.all(requested_delta >= -env.action_scale - 1e-12)
            )
        finally:
            env.close()

    def test_reset_reuses_model_and_runs_noisy_ik_near_start(self):
        env = UR5P2PEnv(
            render_mode=None,
            seed=7,
            target_mode="none",
        )
        try:
            model_id = id(env.model)
            observations = []
            joint_starts = []

            for seed in (7, 8):
                obs, info = env.reset(seed=seed)
                observations.append(obs)
                joint_starts.append(info["q_start"])
                self.assertEqual(id(env.model), model_id)
                self.assertLess(
                    np.linalg.norm(env._ee_pos() - env.start_pos),
                    0.02,
                )

            self.assertFalse(
                np.allclose(joint_starts[0], joint_starts[1])
            )
            self.assertTrue(
                all(item.dtype == np.float32 for item in observations)
            )
        finally:
            env.close()

    def test_squashed_gaussian_agent_returns_bounded_actions(self):
        env = UR5P2PEnv(
            render_mode=None,
            seed=11,
            target_mode="none",
        )
        try:
            agent = Agent(env)
            obs, _ = env.reset(seed=11)
            obs_batch = torch.as_tensor(
                obs[None, :],
                dtype=torch.float32,
            )

            action, logprob, entropy, value = (
                agent.get_action_and_value(obs_batch)
            )
            self.assertEqual(action.shape, (1, env.nu))
            self.assertTrue(torch.all(action > -1.0))
            self.assertTrue(torch.all(action < 1.0))
            self.assertTrue(torch.isfinite(logprob).all())
            self.assertTrue(torch.isfinite(entropy).all())
            self.assertTrue(torch.isfinite(value).all())

            _, replay_logprob, replay_entropy, _ = (
                agent.get_action_and_value(obs_batch, action)
            )
            self.assertEqual(replay_logprob.shape, logprob.shape)
            self.assertEqual(replay_entropy.shape, entropy.shape)
            self.assertTrue(torch.isfinite(replay_logprob).all())
            self.assertTrue(torch.isfinite(replay_entropy).all())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
