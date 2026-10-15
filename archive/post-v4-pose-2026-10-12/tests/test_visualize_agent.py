import unittest

import gymnasium as gym
import numpy as np
import torch

from Scripts.visualize_agent import (
    infer_environment_id,
    infer_network_profile,
    model_dimensions,
    validate_model_environment,
)


def fake_state(observation_dim, action_dim, *, pose_network=False):
    state = {
        "actor_logstd": torch.zeros((1, action_dim)),
        "critic.0.weight": torch.zeros((64, observation_dim)),
        "actor_mean.0.weight": torch.zeros((64, observation_dim)),
        "actor_mean.4.weight": torch.zeros((action_dim, 64)),
    }
    if pose_network:
        state = {
            "actor_logstd": torch.zeros((1, action_dim)),
            "critic.0.weight": torch.zeros((256, observation_dim)),
            "actor_mean.0.weight": torch.zeros((256, observation_dim)),
            "actor_mean.6.weight": torch.zeros((action_dim, 64)),
        }
    return state


class DummyEnv:
    def __init__(self, observation_dim, action_dim):
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )


class VisualizeAgentCompatibilityTests(unittest.TestCase):
    def test_model_dimensions_are_read_from_state_dict(self):
        self.assertEqual(model_dimensions(fake_state(19, 6)), (19, 6))

    def test_environment_is_inferred_for_all_supported_models(self):
        self.assertEqual(
            infer_environment_id(fake_state(19, 6)),
            "UR5-v0",
        )
        self.assertEqual(
            infer_environment_id(fake_state(28, 7)),
            "RM75-v0",
        )
        self.assertEqual(
            infer_environment_id(fake_state(34, 7)),
            "RM75Approach-v0",
        )
        self.assertEqual(
            infer_environment_id(fake_state(20, 7, pose_network=True)),
            "RM75Pose-v0",
        )

    def test_network_profile_is_inferred_from_saved_layers(self):
        self.assertEqual(
            infer_network_profile(fake_state(34, 7)),
            "legacy",
        )
        self.assertEqual(
            infer_network_profile(fake_state(20, 7, pose_network=True)),
            "pose",
        )

    def test_unknown_model_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model dimensions"):
            infer_environment_id(fake_state(40, 7))

    def test_explicit_environment_mismatch_has_clear_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "model expects 19 observations and 6 actions",
        ):
            validate_model_environment(
                fake_state(19, 6),
                DummyEnv(28, 7),
                "RM75-v0",
            )

    def test_matching_environment_is_accepted(self):
        validate_model_environment(
            fake_state(28, 7),
            DummyEnv(28, 7),
            "RM75-v0",
        )


if __name__ == "__main__":
    unittest.main()
