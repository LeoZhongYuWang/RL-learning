"""Focused regression tests for the compact RM75 direction task."""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RM75_DEBUG", "0")

from Algorithms.ppo_continuous_action import should_update_best  # noqa: E402
from env_rm75 import RM75P2PEnv  # noqa: E402


class RM75DirectionContractTests(unittest.TestCase):
    def setUp(self):
        self.env = RM75P2PEnv(
            render_mode=None,
            target_mode="none",
            require_direction=True,
        )

    def tearDown(self):
        self.env.close()

    def test_direction_task_keeps_compact_observation_contract(self):
        observation, _ = self.env.reset(seed=7)
        self.assertEqual(observation.shape, (31,))
        self.assertEqual(self.env.action_space.shape, (7,))

    def test_far_direction_changes_have_no_progress_reward(self):
        for previous, current in ((90.0, 10.0), (10.0, 90.0)):
            reward = self.env._compute_direction_progress(
                math.radians(previous), math.radians(current), 0.30, 0.35,
            )
            self.assertAlmostEqual(reward, 0.0)

    def test_joint_progress_rewards_approach_not_distance_only_retreat(self):
        angle = math.radians(60.0)
        toward_goal = self.env._compute_direction_progress(
            prev_angle=angle,
            direction_angle=angle,
            prev_dist=0.35,
            dist=0.08,
        )
        away_from_goal = self.env._compute_direction_progress(
            prev_angle=angle,
            direction_angle=angle,
            prev_dist=0.08,
            dist=0.35,
        )
        scale = self.env.reward_cfg["progress_scale"]
        self.assertGreater(scale * (0.35 - 0.08) + toward_goal, 0.0)
        self.assertLess(scale * (0.08 - 0.35) + away_from_goal, 0.0)
        # Even an opposite tool axis must not make distance-only retreat pay.
        for dist in np.linspace(0.04, 0.22, 20):
            direction = self.env._compute_direction_progress(
                math.pi, math.pi, dist, dist + 0.01,
            )
            self.assertLess(-scale * 0.01 + direction, 0.0)

    def test_near_goal_improvement_and_reversal_cannot_profit(self):
        worse = math.radians(60.0)
        better = math.radians(30.0)
        improvement = self.env._compute_direction_progress(
            prev_angle=worse,
            direction_angle=better,
            prev_dist=0.08,
            dist=0.35,
        )
        reversal = self.env._compute_direction_progress(
            prev_angle=better,
            direction_angle=worse,
            prev_dist=0.35,
            dist=0.08,
        )
        self.assertAlmostEqual(improvement + reversal, 0.0)

    def test_improve_near_then_worsen_far_cannot_profit(self):
        worse = math.radians(60.0)
        better = math.radians(30.0)
        states = [(0.04, worse), (0.04, better), (0.60, better),
                  (0.60, worse), (0.04, worse)]
        total = sum(
            self.env._compute_direction_progress(a0, a1, d0, d1)
            for (d0, a0), (d1, a1) in zip(states, states[1:])
        )
        self.assertAlmostEqual(total, 0.0)

    def test_distance_cost_is_independent_of_previous_position_success(self):
        rewards = []
        for reached in (False, True):
            self.env.position_reached = reached
            reward, components = self.env._compute_reward(
                dist=0.20, prev_dist=0.20, collided=False, success=False,
            )
            self.assertAlmostEqual(components["dist_penalty"], -0.10)
            self.assertNotIn("position_retention", components)
            rewards.append(reward)
        self.assertAlmostEqual(rewards[0], rewards[1])

    def test_retreat_from_zero_distance_still_loses_position_progress(self):
        angle = math.radians(120.0)
        direction = self.env._compute_direction_progress(
            angle, angle, 0.0, 0.06,
        )
        _, components = self.env._compute_reward(
            dist=0.06, prev_dist=0.0, collided=False, success=False,
            direction_progress=direction,
        )
        self.assertLess(
            components["progress"] + components["direction_progress"], 0.0,
        )

    def test_angular_state_cost_penalizes_lingering_only_near_goal(self):
        def reward_at(distance, angle):
            return self.env._compute_reward(
                dist=distance, prev_dist=distance, collided=False,
                success=False, direction_angle=angle,
            )
        correct, correct_parts = reward_at(0.04, 0.0)
        wrong, wrong_parts = reward_at(0.04, math.pi / 6)
        self.assertAlmostEqual(correct_parts["direction_cost"], 0.0)
        self.assertAlmostEqual(wrong_parts["direction_cost"], -0.05 / 6)
        self.assertAlmostEqual(wrong - correct, -0.05 / 6)
        _, far_parts = reward_at(0.25, math.pi)
        self.assertEqual(far_parts["direction_cost"], 0.0)
        _, unavailable = self.env._compute_reward(
            dist=None, prev_dist=None, collided=False, success=False,
            direction_angle=None,
        )
        self.assertEqual(unavailable["direction_cost"], 0.0)

    def test_gated_state_cost_does_not_make_fixed_angle_retreat_cheaper(self):
        # Worst-case angle: throughout the gate, distance cost grows faster
        # than the angular cost can fall. This is not a convergence proof.
        for distance in np.linspace(0.0, 0.25, 51):
            costs = []
            for d in (distance, distance + 0.001):
                _, parts = self.env._compute_reward(
                    dist=d, prev_dist=d, collided=False, success=False,
                    direction_angle=math.pi,
                )
                costs.append(-parts["dist_penalty"] - parts["direction_cost"])
            self.assertGreater(costs[1], costs[0])

    def test_step_passes_current_angle_into_reward(self):
        self.env.reset(seed=7)
        self.env.goal_pos = np.zeros(3)
        with (
            mock.patch.object(self.env, "_step_sim"),
            mock.patch.object(self.env, "_check_collision", return_value=False),
            mock.patch.object(self.env, "_ee_pos", return_value=np.array([0.04, 0.0, 0.0])),
            mock.patch.object(self.env, "_direction_angle", return_value=math.pi / 6),
        ):
            _, _, _, _, info = self.env.step(np.zeros(7, dtype=np.float32))
        self.assertAlmostEqual(info["reward_components"]["direction_cost"], -0.05 / 6)

    def test_joint_milestones_require_both_errors_and_pay_once(self):
        milestone = self.env._compute_milestone_reward
        self.assertEqual(milestone(0.04, math.radians(80.0)), 0.0)
        self.assertEqual(milestone(0.30, math.radians(10.0)), 0.0)
        self.assertEqual(milestone(0.14, math.radians(40.0)), 5.0)
        self.assertEqual(milestone(0.14, math.radians(40.0)), 0.0)
        self.assertEqual(milestone(0.09, math.radians(25.0)), 10.0)
        self.assertEqual(milestone(0.06, math.radians(18.0)), 20.0)
        self.assertEqual(milestone(0.04, math.radians(10.0)), 0.0)
        self.env.reset(seed=7)
        self.assertEqual(milestone(0.04, math.radians(10.0)), 35.0)

    def test_legacy_point_rewards_and_milestones_are_unchanged(self):
        legacy = RM75P2PEnv(target_mode="none")
        try:
            self.assertEqual(legacy.reward_cfg["dist_weight"], 0.03)
            _, parts = legacy._compute_reward(
                dist=0.04, prev_dist=0.04, collided=False, success=False,
                direction_angle=math.pi,
            )
            self.assertEqual(parts["direction_cost"], 0.0)
            self.assertEqual(legacy._compute_milestone_reward(0.19), 5.0)
            self.assertEqual(legacy._compute_milestone_reward(0.09), 15.0)
            self.assertEqual(legacy._compute_milestone_reward(0.09), 0.0)
            self.assertEqual(
                legacy._compute_direction_progress(1.0, 0.0, 0.10, 0.05),
                0.0,
            )
        finally:
            legacy.close()

    def test_angle_position_and_strict_success_are_distinct(self):
        self.env.reset(seed=7)
        self.env.goal_pos = np.zeros(3)
        zero_action = np.zeros(7, dtype=np.float32)
        with (
            mock.patch.object(self.env, "_step_sim"),
            mock.patch.object(self.env, "_check_collision", return_value=False),
            mock.patch.object(self.env, "_ee_pos", return_value=np.array([0.30, 0.0, 0.0])) as position,
            mock.patch.object(self.env, "_direction_angle", return_value=math.radians(10.0)) as angle,
        ):
            _, _, terminated, _, info = self.env.step(zero_action)
            self.assertTrue(info["angle_reached"])
            self.assertFalse(info["pose_reached"])
            self.assertFalse(terminated)
            self.assertEqual(info["min_direction_angle_at_position_deg"], 180.0)

            position.return_value = np.array([0.04, 0.0, 0.0])
            angle.return_value = math.radians(40.0)
            _, _, _, _, info = self.env.step(zero_action)
            self.assertTrue(info["position_reached"])
            self.assertFalse(info["pose_reached"])
            self.assertAlmostEqual(info["min_direction_angle_at_position_deg"], 40.0)

            angle.return_value = math.radians(10.0)
            for hold_step in range(1, 6):
                _, _, terminated, _, info = self.env.step(zero_action)
                self.assertTrue(info["pose_reached"])
                self.assertEqual(info["direction_hold_steps"], hold_step)
                self.assertEqual(terminated, hold_step == 5)
            self.assertTrue(info["success"])

        self.env.reset(seed=7)
        self.assertFalse(self.env.position_reached)
        self.assertFalse(self.env.angle_reached)
        self.assertFalse(self.env.direction_reached)
        self.assertEqual(self.env.direction_hold_steps, 0)
        self.assertEqual(self.env.min_direction_angle_at_position_deg, 180.0)

    def test_best_pose_model_uses_pose_quality_before_final_distance(self):
        best = {
            "success_rate": 0.0,
            "pose_reached_rate": 0.0,
            "collision_rate": 0.0,
            "position_success_rate": 0.90,
            "mean_min_direction_angle_at_position_deg": 50.0,
            "mean_final_distance": 0.10,
            "mean_final_direction_angle_deg": 80.0,
        }
        better_pose = {
            "success_rate": 0.0,
            "pose_reached_rate": 0.0,
            "collision_rate": 0.06,
            "position_success_rate": 0.85,
            "mean_min_direction_angle_at_position_deg": 30.0,
            "mean_final_distance": 0.15,
            "mean_final_direction_angle_deg": 70.0,
        }
        self.assertTrue(should_update_best(better_pose, best, 0.10))


if __name__ == "__main__":
    unittest.main()
