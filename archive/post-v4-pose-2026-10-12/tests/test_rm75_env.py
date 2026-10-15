import importlib
from collections import Counter
import unittest
from unittest import mock

import gymnasium as gym
import mujoco as mj
import numpy as np

from env_rm75 import RM75P2PEnv


EXPECTED_JOINTS = [f"joint_{index}" for index in range(1, 8)]
EXPECTED_ACTUATORS = [
    f"joint_{index}_pos"
    for index in range(1, 8)
]


def object_names(model, object_type, count):
    return [
        mj.mj_id2name(model, object_type, index)
        for index in range(count)
    ]


class RM75EnvironmentTests(unittest.TestCase):
    def test_collision_penalty_is_configurable(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            collision_penalty=45.0,
        )
        try:
            self.assertEqual(
                env.reward_cfg["collision_penalty"],
                45.0,
            )
        finally:
            env.close()

    def test_negative_collision_penalty_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "collision_penalty",
        ):
            RM75P2PEnv(
                render_mode=None,
                target_mode="none",
                collision_penalty=-1.0,
            )

    def test_gym_registration_keeps_ur5_and_adds_rm75(self):
        registry_module = importlib.import_module("__init__")
        importlib.reload(registry_module)

        self.assertIn("UR5-v0", gym.registry)
        self.assertIn("RM75-v0", gym.registry)
        self.assertIn("RM75Approach-v0", gym.registry)
        self.assertIn("RM75Pose-v0", gym.registry)

        ur5_env = gym.make(
            "UR5-v0",
            render_mode=None,
            target_mode="none",
        )
        rm75_env = gym.make(
            "RM75-v0",
            render_mode=None,
            target_mode="none",
        )
        try:
            ur5_obs, _ = ur5_env.reset(seed=3)
            rm75_obs, _ = rm75_env.reset(seed=3)
            self.assertEqual(ur5_obs.shape, (19,))
            self.assertEqual(rm75_obs.shape, (28,))

            rm75_result = rm75_env.step(
                np.zeros(7, dtype=np.float32)
            )
            self.assertEqual(rm75_result[0].shape, (28,))
            self.assertTrue(np.isfinite(rm75_result[1]))
        finally:
            ur5_env.close()
            rm75_env.close()

    def test_approach_environment_has_separate_observation_contract(self):
        registry_module = importlib.import_module("__init__")
        importlib.reload(registry_module)
        point_env = gym.make(
            "RM75-v0",
            render_mode=None,
            target_mode="random",
        )
        approach_env = gym.make(
            "RM75Approach-v0",
            render_mode=None,
            target_mode="random",
            target_direction=(0.0, 0.0, -2.0),
        )
        try:
            point_obs, _ = point_env.reset(seed=31)
            approach_obs, info = approach_env.reset(seed=31)

            self.assertEqual(point_obs.shape, (28,))
            self.assertEqual(approach_obs.shape, (34,))
            self.assertEqual(point_env.action_space.shape, (7,))
            self.assertEqual(approach_env.action_space.shape, (7,))
            np.testing.assert_allclose(
                approach_env.unwrapped.target_direction,
                [0.0, 0.0, -1.0],
            )
            np.testing.assert_allclose(
                approach_obs[30:33],
                [0.0, 0.0, -1.0],
            )
            self.assertIn("angle_error_deg", info)
            self.assertIn("target_direction", info)
        finally:
            point_env.close()
            approach_env.close()

    def test_pose_environment_uses_compact_normalized_contract(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="fixed",
            target_pos=(0.4, 0.0, 0.2905),
            approach_task=True,
            pose_task=True,
            orient_success_threshold_deg=20.0,
        )
        try:
            observation, _ = env.reset(seed=31)
            self.assertEqual(observation.shape, (20,))
            self.assertEqual(env.action_space.shape, (7,))
            self.assertEqual(env.horizon, 600)
            np.testing.assert_allclose(env.action_scale, 0.025)
            self.assertTrue(np.isfinite(observation).all())
            self.assertTrue(env.observation_space.contains(observation))
            next_observation, reward, _, _, info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (20,))
            self.assertTrue(np.isfinite(next_observation).all())
            self.assertTrue(np.isfinite(reward))
            self.assertIn("keypoint_error", info)
        finally:
            env.close()

    def test_pose_reward_prioritizes_progress_and_penalizes_reversal(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
            pose_task=True,
        )
        try:
            zero = np.zeros(7, dtype=np.float64)
            forward = np.full(7, 0.5, dtype=np.float64)
            progress_reward, _ = env._compute_pose_reward(
                dist=0.15,
                prev_dist=0.20,
                angle_error=np.deg2rad(20.0),
                prev_angle_error=np.deg2rad(20.0),
                action=zero,
                previous_action=zero,
                collided=False,
                success=False,
            )
            stationary_reward, _ = env._compute_pose_reward(
                dist=0.20,
                prev_dist=0.20,
                angle_error=np.deg2rad(20.0),
                prev_angle_error=np.deg2rad(20.0),
                action=zero,
                previous_action=zero,
                collided=False,
                success=False,
            )
            smooth_reward, _ = env._compute_pose_reward(
                dist=0.10,
                prev_dist=0.10,
                angle_error=np.deg2rad(20.0),
                prev_angle_error=np.deg2rad(20.0),
                action=forward,
                previous_action=forward,
                collided=False,
                success=False,
            )
            reversal_reward, _ = env._compute_pose_reward(
                dist=0.10,
                prev_dist=0.10,
                angle_error=np.deg2rad(20.0),
                prev_angle_error=np.deg2rad(20.0),
                action=-forward,
                previous_action=forward,
                collided=False,
                success=False,
            )

            self.assertGreater(progress_reward, stationary_reward)
            self.assertGreater(smooth_reward, reversal_reward)
        finally:
            env.close()

    def test_pose_orientation_progress_is_weighted_more_near_goal(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
            pose_task=True,
        )
        try:
            zero = np.zeros(7, dtype=np.float64)
            common = {
                "angle_error": np.deg2rad(20.0),
                "prev_angle_error": np.deg2rad(25.0),
                "action": zero,
                "previous_action": zero,
                "collided": False,
                "success": False,
            }
            _, far_components = env._compute_pose_reward(
                dist=0.35,
                prev_dist=0.35,
                **common,
            )
            _, near_components = env._compute_pose_reward(
                dist=0.08,
                prev_dist=0.08,
                **common,
            )
            self.assertGreater(
                near_components["orientation_progress"],
                far_components["orientation_progress"],
            )
        finally:
            env.close()

    def test_pose_collision_is_worse_than_waiting_for_full_horizon(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
            pose_task=True,
        )
        try:
            zero = np.zeros(7, dtype=np.float64)
            common = {
                "dist": 0.30,
                "prev_dist": 0.30,
                "angle_error": np.deg2rad(20.0),
                "prev_angle_error": np.deg2rad(20.0),
                "action": zero,
                "previous_action": zero,
                "success": False,
            }
            waiting_reward, _ = env._compute_pose_reward(
                collided=False,
                **common,
            )
            collision_reward, _ = env._compute_pose_reward(
                collided=True,
                **common,
            )
            self.assertLess(collision_reward, env.horizon * waiting_reward)
        finally:
            env.close()

    def test_pose_success_requires_five_consecutive_steps(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
            pose_task=True,
        )
        try:
            for _ in range(4):
                self.assertFalse(env._update_pose_success_streak(True))
            self.assertTrue(env._update_pose_success_streak(True))
            self.assertFalse(env._update_pose_success_streak(False))
            self.assertEqual(env.pose_success_streak, 0)
        finally:
            env.close()

    def test_approach_environment_rejects_invalid_direction(self):
        with self.assertRaisesRegex(ValueError, "target_direction"):
            RM75P2PEnv(
                render_mode=None,
                approach_task=True,
                target_direction=(0.0, 0.0, 0.0),
            )

    def test_approach_angle_and_success_require_both_conditions(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=37,
            target_mode="none",
            approach_task=True,
            target_direction=(0.0, 0.0, -1.0),
            orient_success_threshold_deg=15.0,
        )
        try:
            self.assertAlmostEqual(
                env._direction_angle_error(
                    np.array([0.0, 0.0, -1.0])
                ),
                0.0,
            )
            self.assertAlmostEqual(
                np.degrees(
                    env._direction_angle_error(
                        np.array([0.0, 0.0, 1.0])
                    )
                ),
                180.0,
            )
            self.assertTrue(env._is_success(0.04, np.deg2rad(10.0)))
            self.assertFalse(env._is_success(0.06, np.deg2rad(10.0)))
            self.assertFalse(env._is_success(0.04, np.deg2rad(20.0)))
        finally:
            env.close()

    def test_approach_direction_marker_follows_random_goal(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=41,
            target_mode="random",
            approach_task=True,
            target_direction=(0.0, 0.0, -1.0),
        )
        try:
            env.reset(seed=41)
            marker_position = env.model.site_pos[
                env.goal_direction_sid
            ]
            expected_position = (
                env.goal_pos
                - 0.5
                * env.goal_direction_length
                * env.target_direction
            )
            np.testing.assert_allclose(
                marker_position,
                expected_position,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                env.model.site_rgba[env.goal_direction_sid],
                [1.0, 0.8, 0.1, 0.9],
                atol=1e-8,
            )
            mj.mj_forward(env.model, env.data)
            marker_axis = env.data.site_xmat[
                env.goal_direction_sid
            ].reshape(3, 3)[:, 2]
            np.testing.assert_allclose(
                marker_axis,
                env.target_direction,
                atol=1e-8,
            )
        finally:
            env.close()

    def test_random_cone_direction_is_seeded_and_updates_marker(self):
        kwargs = {
            "render_mode": None,
            "target_mode": "fixed",
            "target_pos": (-0.4, 0.0, 0.2905),
            "approach_task": True,
            "target_direction_mode": "random_cone",
            "target_tilt_min_deg": 5.0,
            "target_tilt_max_deg": 20.0,
            "target_azimuth_min_deg": -180.0,
            "target_azimuth_max_deg": 180.0,
            "require_pose_feasible": False,
        }
        first = RM75P2PEnv(seed=1, **kwargs)
        second = RM75P2PEnv(seed=99, **kwargs)
        try:
            _, first_info = first.reset(seed=123)
            _, second_info = second.reset(seed=123)
            np.testing.assert_allclose(
                first.target_direction,
                second.target_direction,
            )
            self.assertAlmostEqual(
                np.linalg.norm(first.target_direction),
                1.0,
            )
            self.assertTrue(
                5.0 <= first_info["target_tilt_deg"] <= 20.0
            )
            self.assertTrue(
                -180.0
                <= first_info["target_azimuth_deg"]
                <= 180.0
            )

            mj.mj_forward(first.model, first.data)
            marker_axis = first.data.site_xmat[
                first.goal_direction_sid
            ].reshape(3, 3)[:, 2]
            np.testing.assert_allclose(
                marker_axis,
                first.target_direction,
                atol=1e-8,
            )

            first.reset(seed=124)
            self.assertFalse(
                np.allclose(
                    first.target_direction,
                    second.target_direction,
                )
            )
        finally:
            first.close()
            second.close()

    def test_pose_feasible_random_cone_reset_succeeds(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=5,
            target_mode="random",
            approach_task=True,
            pose_task=True,
            target_direction_mode="random_cone",
            target_tilt_min_deg=0.0,
            target_tilt_max_deg=30.0,
            approach_position_box=(-0.45, -0.35, -0.18, 0.18),
            require_pose_feasible=True,
            pose_feasibility_attempts=50,
        )
        try:
            for seed in range(5):
                obs, info = env.reset(seed=seed)
                self.assertEqual(obs.shape, (20,))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(-0.45 <= env.goal_pos[0] <= -0.35)
                self.assertTrue(-0.18 <= env.goal_pos[1] <= 0.18)
                self.assertTrue(0.0 <= info["target_tilt_deg"] <= 30.0)

            _, _, _, _, step_info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            for field in (
                "min_distance",
                "min_angle_error_deg",
                "min_pose_error",
                "ever_position_success",
                "ever_direction_success",
                "simultaneous_success",
            ):
                self.assertIn(field, step_info)
        finally:
            env.close()

    def test_approach_reward_favors_reducing_direction_error(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
        )
        try:
            improved_reward, improved_components = env._compute_reward(
                dist=0.04,
                prev_dist=0.04,
                collided=False,
                success=False,
                angle_error=np.deg2rad(10.0),
                prev_angle_error=np.deg2rad(40.0),
            )
            worsened_reward, worsened_components = env._compute_reward(
                dist=0.04,
                prev_dist=0.04,
                collided=False,
                success=False,
                angle_error=np.deg2rad(40.0),
                prev_angle_error=np.deg2rad(10.0),
            )

            self.assertGreater(improved_components["pose_progress"], 0.0)
            self.assertLess(worsened_components["pose_progress"], 0.0)
            self.assertGreater(improved_reward, worsened_reward)
        finally:
            env.close()

    def test_smooth_pose_progress_rewards_both_constraints(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
        )
        try:
            distance_only_reward, distance_only_components = (
                env._compute_reward(
                    dist=0.10,
                    prev_dist=0.12,
                    collided=False,
                    success=False,
                    angle_error=np.deg2rad(60.0),
                    prev_angle_error=np.deg2rad(60.0),
                )
            )
            angle_reward, angle_components = env._compute_reward(
                dist=0.10,
                prev_dist=0.10,
                collided=False,
                success=False,
                angle_error=np.deg2rad(45.0),
                prev_angle_error=np.deg2rad(60.0),
            )

            self.assertGreater(
                distance_only_components["pose_progress"],
                0.0,
            )
            self.assertGreater(angle_components["pose_progress"], 0.0)
            self.assertTrue(np.isfinite(distance_only_reward))
            self.assertTrue(np.isfinite(angle_reward))
        finally:
            env.close()

    def test_approach_pose_progress_prioritizes_position(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
        )
        try:
            self.assertAlmostEqual(
                env.reward_cfg["pose_error_weight"],
                0.05,
            )
            position_reward, _ = env._compute_reward(
                dist=0.09,
                prev_dist=0.10,
                collided=False,
                success=False,
                angle_error=np.deg2rad(30.0),
                prev_angle_error=np.deg2rad(30.0),
            )
            direction_reward, _ = env._compute_reward(
                dist=0.10,
                prev_dist=0.10,
                collided=False,
                success=False,
                angle_error=np.deg2rad(27.0),
                prev_angle_error=np.deg2rad(30.0),
            )

            self.assertGreater(position_reward, direction_reward)
        finally:
            env.close()

    def test_approach_curriculum_uses_loose_training_but_strict_metrics(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
        )
        try:
            env.set_approach_curriculum(
                stage=1,
                tilt_max_deg=15.0,
                success_distance_thresh=0.20,
                success_angle_deg=45.0,
                position_box=(-0.42, -0.36, -0.10, 0.10),
            )
            self.assertEqual(env.curriculum_stage, 1)
            self.assertAlmostEqual(env.active_dist_success_thresh, 0.20)
            self.assertAlmostEqual(env.active_orient_success_thresh_deg, 45.0)
            np.testing.assert_allclose(
                env.approach_position_box,
                [-0.42, -0.36, -0.10, 0.10],
            )
            self.assertTrue(env._is_success(0.15, np.deg2rad(30.0)))
            self.assertFalse(
                env._is_strict_success(0.15, np.deg2rad(30.0))
            )
            self.assertTrue(
                env._is_strict_success(0.04, np.deg2rad(10.0))
            )
        finally:
            env.close()

    def test_active_pose_success_does_not_end_approach_episode(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
            orient_success_threshold_deg=20.0,
        )
        try:
            env.reset(seed=41)
            env.set_approach_curriculum(
                stage=3,
                tilt_max_deg=15.0,
                success_distance_thresh=0.10,
                success_angle_deg=35.0,
                position_box=(-0.42, -0.36, -0.10, 0.10),
            )
            env.goal_pos = np.zeros(3, dtype=np.float64)
            env.prev_dist = 0.09
            env.prev_angle_error = np.deg2rad(32.0)
            env.min_distance = env.prev_dist
            env.min_angle_error = env.prev_angle_error
            env.min_pose_error = env._normalized_pose_error(
                env.prev_dist,
                env.prev_angle_error,
            )

            with (
                mock.patch.object(env, "_step_sim"),
                mock.patch.object(env, "_check_collision", return_value=False),
                mock.patch.object(
                    env,
                    "_ee_pos",
                    return_value=np.array([0.08, 0.0, 0.0]),
                ),
                mock.patch.object(
                    env,
                    "_direction_angle_error",
                    return_value=np.deg2rad(30.0),
                ),
            ):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(7, dtype=np.float32)
                )

            self.assertTrue(info["active_success"])
            self.assertTrue(info["ever_active_success"])
            self.assertTrue(info["success"])
            self.assertFalse(info["strict_success"])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(
                info["reward_components"]["active_success"],
                0.0,
            )
        finally:
            env.close()

    def test_strict_pose_success_ends_approach_episode(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
            orient_success_threshold_deg=20.0,
        )
        try:
            env.reset(seed=42)
            env.set_approach_curriculum(
                stage=3,
                tilt_max_deg=15.0,
                success_distance_thresh=0.10,
                success_angle_deg=35.0,
                position_box=(-0.42, -0.36, -0.10, 0.10),
            )
            env.goal_pos = np.zeros(3, dtype=np.float64)
            env.prev_dist = 0.06
            env.prev_angle_error = np.deg2rad(22.0)

            with (
                mock.patch.object(env, "_step_sim"),
                mock.patch.object(env, "_check_collision", return_value=False),
                mock.patch.object(
                    env,
                    "_ee_pos",
                    return_value=np.array([0.04, 0.0, 0.0]),
                ),
                mock.patch.object(
                    env,
                    "_direction_angle_error",
                    return_value=np.deg2rad(10.0),
                ),
            ):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(7, dtype=np.float32)
                )

            self.assertTrue(info["active_success"])
            self.assertTrue(info["strict_success"])
            self.assertTrue(info["success"])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(
                info["reward_components"]["success"],
                env.reward_cfg["success_bonus"],
            )
        finally:
            env.close()

    def test_active_pose_bonus_is_once_per_episode(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
        )
        try:
            env.reset(seed=43)
            env.set_approach_curriculum(
                stage=3,
                tilt_max_deg=15.0,
                success_distance_thresh=0.10,
                success_angle_deg=35.0,
                position_box=(-0.42, -0.36, -0.10, 0.10),
            )
            self.assertEqual(env._claim_active_success_bonus(True), 0.0)
            self.assertEqual(env._claim_active_success_bonus(True), 0.0)
            env.reset(seed=43)
            env.set_approach_curriculum(
                stage=3,
                tilt_max_deg=15.0,
                success_distance_thresh=0.10,
                success_angle_deg=35.0,
                position_box=(-0.42, -0.36, -0.10, 0.10),
            )
            self.assertEqual(env._claim_active_success_bonus(True), 0.0)
        finally:
            env.close()

    def test_pose_error_uses_final_thresholds_across_curriculum(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
            orient_success_threshold_deg=20.0,
        )
        try:
            before = env._normalized_pose_error(
                0.10,
                np.deg2rad(30.0),
            )
            env.set_approach_curriculum(
                stage=1,
                tilt_max_deg=15.0,
                success_distance_thresh=0.20,
                success_angle_deg=60.0,
                position_box=(-0.42, -0.36, -0.10, 0.10),
            )
            after = env._normalized_pose_error(
                0.10,
                np.deg2rad(30.0),
            )
            self.assertAlmostEqual(before, after)
        finally:
            env.close()

    def test_approach_curriculum_accepts_final_stage_six(self):
        env = RM75P2PEnv(approach_task=True)
        try:
            env.set_approach_curriculum(
                stage=6,
                tilt_max_deg=45.0,
                success_distance_thresh=0.05,
                success_angle_deg=15.0,
                position_box=(-0.45, -0.35, -0.18, 0.18),
            )
            self.assertEqual(env.curriculum_stage, 6)
        finally:
            env.close()

    def test_approach_milestones_require_distance_and_direction(self):
        env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            approach_task=True,
        )
        try:
            env.reset(seed=19)
            self.assertEqual(
                env._compute_milestone_reward(
                    0.19,
                    np.deg2rad(60.0),
                ),
                0.0,
            )
            self.assertEqual(
                env._compute_milestone_reward(
                    0.19,
                    np.deg2rad(40.0),
                ),
                2.0,
            )
            self.assertEqual(
                env._compute_milestone_reward(
                    0.09,
                    np.deg2rad(35.0),
                ),
                0.0,
            )
            self.assertEqual(
                env._compute_milestone_reward(
                    0.09,
                    np.deg2rad(25.0),
                ),
                0.0,
            )
            self.assertEqual(
                env._compute_milestone_reward(
                    0.08,
                    np.deg2rad(20.0),
                ),
                0.0,
            )
            self.assertEqual(
                env._compute_milestone_reward(
                    0.07,
                    np.deg2rad(20.0),
                ),
                5.0,
            )
        finally:
            env.close()

    def test_spaces_reset_and_compiled_model_contract(self):
        env = RM75P2PEnv(render_mode=None, seed=7)
        try:
            self.assertEqual(env.nq, 7)
            self.assertEqual(env.nu, 7)
            self.assertEqual(env.action_space.shape, (7,))
            self.assertEqual(env.observation_space.shape, (28,))
            np.testing.assert_allclose(env.action_space.low, -1.0)
            np.testing.assert_allclose(env.action_space.high, 1.0)
            np.testing.assert_allclose(env.action_scale, 0.02)

            self.assertEqual(
                object_names(
                    env.model,
                    mj.mjtObj.mjOBJ_JOINT,
                    env.model.njnt,
                ),
                EXPECTED_JOINTS,
            )
            self.assertEqual(
                object_names(
                    env.model,
                    mj.mjtObj.mjOBJ_ACTUATOR,
                    env.model.nu,
                ),
                EXPECTED_ACTUATORS,
            )
            actuator_joints = [
                mj.mj_id2name(
                    env.model,
                    mj.mjtObj.mjOBJ_JOINT,
                    int(joint_id),
                )
                for joint_id in env.model.actuator_trnid[:, 0]
            ]
            self.assertEqual(actuator_joints, EXPECTED_JOINTS)

            obs, info = env.reset(seed=7)
            self.assertEqual(obs.shape, (28,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(info["q_start"].shape, (7,))
            np.testing.assert_allclose(
                obs[14:21],
                env.q_target - env.data.qpos[:env.nq],
                atol=1e-6,
            )
            self.assertIsNotNone(env.goal_pos)
            self.assertEqual(env.target_mode, "random")
            self.assertFalse(env._check_collision())
            self.assertEqual(info["target_region"], "uniform")
        finally:
            env.close()

    def test_hard_mixture_target_sampling_is_seeded_and_in_bounds(self):
        first = RM75P2PEnv(
            render_mode=None,
            seed=29,
            target_mode="random",
            target_sampling="hard_mixture",
        )
        second = RM75P2PEnv(
            render_mode=None,
            seed=29,
            target_mode="random",
            target_sampling="hard_mixture",
        )
        try:
            first_samples = [first._sample_random_goal() for _ in range(4000)]
            second_samples = [second._sample_random_goal() for _ in range(20)]

            for (first_goal, first_region), (second_goal, second_region) in zip(
                first_samples[:20],
                second_samples,
            ):
                np.testing.assert_allclose(first_goal, second_goal)
                self.assertEqual(first_region, second_region)

            counts = Counter(region for _, region in first_samples)
            self.assertTrue(1800 <= counts["uniform"] <= 2200)
            self.assertTrue(800 <= counts["hard_negative_y"] <= 1200)
            self.assertTrue(800 <= counts["hard_far_x"] <= 1200)

            for goal, region in first_samples:
                self.assertTrue(-0.5 <= goal[0] <= -0.35)
                self.assertTrue(-0.3 <= goal[1] <= 0.3)
                self.assertAlmostEqual(goal[2], 0.2905)
                if region == "hard_negative_y":
                    self.assertTrue(-0.3 <= goal[1] <= -0.15)
                elif region == "hard_far_x":
                    self.assertTrue(-0.5 <= goal[0] <= -0.45)
        finally:
            first.close()
            second.close()

    def test_rejects_unknown_target_sampling_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "target_sampling",
        ):
            RM75P2PEnv(
                render_mode=None,
                target_sampling="unsupported",
            )

    def test_zero_and_normalized_action_steps(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=11,
            target_mode="fixed",
            target_pos=[-0.4, 0.0, 0.2905],
        )
        try:
            env.reset(seed=11)
            q_target_before = env.q_target.copy()

            result = env.step(np.zeros(7, dtype=np.float32))
            obs, reward, terminated, truncated, info = result
            self.assertEqual(obs.shape, (28,))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            np.testing.assert_allclose(env.q_target, q_target_before)
            self.assertEqual(
                set(info["reward_components"]),
                {
                    "progress",
                    "dist_penalty",
                    "time",
                    "milestone",
                    "orient",
                    "collision",
                    "success",
                    "timeout",
                },
            )

            previous_target = env.q_target.copy()
            env.step(np.ones(7, dtype=np.float32))
            expected_target = np.clip(
                previous_target + env.action_scale,
                env.model.jnt_range[:env.nq, 0],
                env.model.jnt_range[:env.nq, 1],
            )
            np.testing.assert_allclose(env.q_target, expected_target)
        finally:
            env.close()

    def test_timeout_penalty_is_applied_only_to_unsuccessful_truncation(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=17,
            target_mode="none",
        )
        try:
            timeout_reward, timeout_components = env._compute_reward(
                dist=0.2,
                prev_dist=0.2,
                collided=False,
                success=False,
                timed_out=True,
            )
            active_reward, active_components = env._compute_reward(
                dist=0.2,
                prev_dist=0.2,
                collided=False,
                success=False,
                timed_out=False,
            )

            self.assertAlmostEqual(
                timeout_components["dist_penalty"],
                -0.006,
            )
            self.assertAlmostEqual(
                timeout_components["timeout"],
                -10.0,
            )
            self.assertAlmostEqual(active_components["timeout"], 0.0)
            self.assertAlmostEqual(
                timeout_reward,
                active_reward - 10.0,
            )

            env.reset(seed=17)
            env.step_i = env.horizon - 1
            _, step_reward, terminated, truncated, info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(info["timeout"])
            self.assertAlmostEqual(
                info["reward_components"]["timeout"],
                -10.0,
            )
            self.assertTrue(np.isfinite(step_reward))
        finally:
            env.close()

    def test_distance_milestones_are_paid_once_per_episode(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=19,
            target_mode="none",
        )
        try:
            env.reset(seed=19)
            self.assertEqual(
                env._compute_distance_milestone_reward(0.25),
                0.0,
            )
            self.assertEqual(
                env._compute_distance_milestone_reward(0.19),
                5.0,
            )
            self.assertEqual(
                env._compute_distance_milestone_reward(0.15),
                0.0,
            )
            self.assertEqual(
                env._compute_distance_milestone_reward(0.09),
                15.0,
            )
            self.assertEqual(
                env._compute_distance_milestone_reward(0.08),
                0.0,
            )

            env.reset(seed=20)
            self.assertEqual(
                env._compute_distance_milestone_reward(0.09),
                20.0,
            )
            self.assertEqual(
                env._compute_distance_milestone_reward(None),
                0.0,
            )
        finally:
            env.close()

    def test_reset_seed_controls_table_friction(self):
        env = RM75P2PEnv(render_mode=None, seed=0)
        try:
            compiled_model = env.model
            simulation_data = env.data
            env.reset(seed=123)
            friction_123_first = env.table_friction.copy()

            env.reset(seed=123)
            friction_123_second = env.table_friction.copy()

            env.reset(seed=124)
            friction_124 = env.table_friction.copy()

            np.testing.assert_allclose(
                friction_123_first,
                friction_123_second,
            )
            self.assertFalse(
                np.allclose(friction_123_first, friction_124)
            )
            self.assertTrue(
                np.all(
                    (friction_123_first[:, 0] >= 0.4)
                    & (friction_123_first[:, 0] <= 0.9)
                )
            )
            self.assertIs(env.model, compiled_model)
            self.assertIs(env.data, simulation_data)
        finally:
            env.close()

    def test_rejects_non_position_control_mode(self):
        with mock.patch.dict(
            "os.environ",
            {"RM75_CTRL_MODE": "torque"},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "supports only position actuators",
            ):
                RM75P2PEnv(render_mode=None)

    def test_self_collision_geom_classification(self):
        env = RM75P2PEnv(render_mode=None, seed=5)
        try:
            env.reset(seed=5)
            self.assertEqual(len(env.robot_geom_ids), 8)
            self.assertTrue(
                env.robot_geom_ids.isdisjoint(env.table_geom_ids)
            )

            robot_ids = sorted(env.robot_geom_ids)
            table_ids = sorted(env.table_geom_ids)
            floor_id = env.geom_name2id["floor"]
            adjacent_pair = next(
                iter(env.allowed_adjacent_robot_geom_pairs)
            )

            self.assertTrue(
                env._contact_pair_is_collision(
                    robot_ids[0],
                    robot_ids[-1],
                )
            )
            self.assertTrue(
                env._contact_pair_is_collision(
                    robot_ids[0],
                    table_ids[0],
                )
            )
            self.assertFalse(
                env._contact_pair_is_collision(*adjacent_pair)
            )
            self.assertFalse(
                env._contact_pair_is_collision(
                    table_ids[0],
                    table_ids[1],
                )
            )
            self.assertFalse(
                env._contact_pair_is_collision(
                    robot_ids[0],
                    floor_id,
                )
            )
            moving_link_id = next(
                geom_id
                for geom_id in robot_ids
                if geom_id not in env.base_geom_ids
            )
            self.assertTrue(
                env._contact_pair_is_collision(
                    moving_link_id,
                    floor_id,
                )
            )
        finally:
            env.close()

    def test_known_non_adjacent_self_collision_pose(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=2026,
            target_mode="none",
        )
        try:
            self_collision_q = np.array(
                [
                    -0.3828206331,
                    2.250511096,
                    -0.0005235360808,
                    2.185547915,
                    2.344775866,
                    -1.808870573,
                    4.678767042,
                ],
                dtype=np.float64,
            )
            env.data.qpos[:env.nq] = self_collision_q
            env.data.qvel[:] = 0.0
            mj.mj_forward(env.model, env.data)

            self_contacts = [
                contact
                for contact in env.data.contact
                if (
                    int(contact.geom1) in env.robot_geom_ids
                    and int(contact.geom2) in env.robot_geom_ids
                    and env._contact_pair_is_collision(
                        contact.geom1,
                        contact.geom2,
                    )
                )
            ]
            self.assertGreater(len(self_contacts), 0)
            self.assertTrue(env._check_collision())
        finally:
            env.close()

    def test_home_hold_is_stable_for_one_second(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=13,
            target_mode="none",
        )
        try:
            env.reset(seed=13)
            q_start = env.data.qpos[:env.nq].copy()
            collision_steps = 0

            for _ in range(50):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(7, dtype=np.float32)
                )
                collision_steps += int(info["collision"])
                self.assertFalse(terminated)
                self.assertFalse(truncated)

            q_end = env.data.qpos[:env.nq].copy()
            max_joint_drift = np.max(np.abs(q_end - q_start))
            self.assertLess(max_joint_drift, 0.05)
            self.assertEqual(collision_steps, 0)
        finally:
            env.close()

    def test_random_target_box_reference_points_are_reachable(self):
        env = RM75P2PEnv(
            render_mode=None,
            seed=13,
            target_mode="none",
        )
        try:
            xmin, xmax, ymin, ymax = env.target_random_box.tolist()
            z = 0.2905
            targets = {
                "xmin_ymin": np.array([xmin, ymin, z]),
                "xmin_ymax": np.array([xmin, ymax, z]),
                "xmax_ymin": np.array([xmax, ymin, z]),
                "xmax_ymax": np.array([xmax, ymax, z]),
                "center": np.array([
                    (xmin + xmax) / 2.0,
                    (ymin + ymax) / 2.0,
                    z,
                ]),
            }
            joint_limits = env.model.jnt_range[:env.nq]

            for name, target in targets.items():
                with self.subTest(target=name):
                    env.reset(seed=13)
                    initial_guess = env.data.qpos[:env.nq].copy()
                    solution = env._solve_ik(target, initial_guess)

                    self.assertIsNotNone(solution)
                    self.assertTrue(np.isfinite(solution).all())
                    self.assertTrue(
                        np.all(solution >= joint_limits[:, 0])
                    )
                    self.assertTrue(
                        np.all(solution <= joint_limits[:, 1])
                    )

                    env.data.qpos[:env.nq] = solution
                    env.data.qvel[:] = 0.0
                    mj.mj_forward(env.model, env.data)

                    position_error = np.linalg.norm(
                        env._ee_pos() - target
                    )
                    self.assertLess(position_error, 0.01)
                    self.assertFalse(env._check_collision())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
