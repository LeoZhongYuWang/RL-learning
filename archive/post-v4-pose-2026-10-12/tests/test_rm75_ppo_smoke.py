import os
import signal
import tempfile
import subprocess
import sys
import time
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from Algorithms.ppo_continuous_action import (
    Agent,
    Args,
    accumulate_reward_components,
    apply_resume_configuration,
    apply_approach_curriculum,
    configure_training_horizon,
    compute_stage_action_std,
    compute_learning_rate,
    curriculum_promotion_threshold,
    curriculum_stage_settings,
    initial_curriculum_stage,
    initial_curriculum_rollback_armed,
    should_demote_curriculum,
    should_promote_curriculum,
    compute_action_std,
    environment_kwargs,
    evaluate_policy,
    load_agent_weights,
    make_vector_env,
    make_env,
    next_checkpoint_step,
    normalize_cuda_rng_states,
    resolve_network_profile,
    save_agent_weights,
    should_update_best,
)


class DummyRM75Spaces:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(28,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(7,),
        dtype=np.float32,
    )


class RM75PPOSmokeTests(unittest.TestCase):
    def test_reward_component_logging_isolated_from_rollout_buffers(self):
        infos = {
            "reward_components": {
                "pose_progress": np.array([1.0, 2.0]),
                "_pose_progress": np.array([True, False]),
                "time": np.array([-0.01, -0.01]),
                "_time": np.array([True, True]),
            }
        }
        component_sums = {}
        component_counts = {}

        accumulate_reward_components(
            infos,
            2,
            component_sums,
            component_counts,
        )

        self.assertEqual(component_sums["pose_progress"], 1.0)
        self.assertEqual(component_counts["pose_progress"], 1)
        self.assertAlmostEqual(component_sums["time"], -0.02)
        self.assertEqual(component_counts["time"], 2)

    def test_approach_curriculum_uses_mastery_stage_settings(self):
        args = Args()
        args.target_tilt_max_deg = 30.0
        args.orient_success_threshold_deg = 20.0
        stage1 = curriculum_stage_settings(args, 1)
        stage2 = curriculum_stage_settings(args, 2)
        stage3 = curriculum_stage_settings(args, 3)
        stage4 = curriculum_stage_settings(args, 4)
        stage5 = curriculum_stage_settings(args, 5)
        stage6 = curriculum_stage_settings(args, 6)

        self.assertEqual(stage1["tilt_max_deg"], 15.0)
        self.assertEqual(stage1["success_distance_thresh"], 0.12)
        self.assertEqual(stage1["success_angle_deg"], 40.0)
        self.assertEqual(stage2["tilt_max_deg"], 15.0)
        self.assertEqual(stage2["success_distance_thresh"], 0.10)
        self.assertEqual(stage2["success_angle_deg"], 35.0)
        self.assertEqual(stage3["tilt_max_deg"], 20.0)
        self.assertEqual(stage3["success_distance_thresh"], 0.085)
        self.assertEqual(stage3["success_angle_deg"], 30.0)
        self.assertEqual(stage4["tilt_max_deg"], 25.0)
        self.assertEqual(stage4["success_distance_thresh"], 0.070)
        self.assertEqual(stage4["success_angle_deg"], 25.0)
        self.assertEqual(stage5["tilt_max_deg"], 30.0)
        self.assertEqual(stage5["success_distance_thresh"], 0.060)
        self.assertEqual(stage5["success_angle_deg"], 22.0)
        self.assertEqual(stage6["tilt_max_deg"], 30.0)
        self.assertEqual(stage6["success_distance_thresh"], 0.05)
        self.assertEqual(stage6["success_angle_deg"], 20.0)

        self.assertFalse(initial_curriculum_rollback_armed(1, None))
        self.assertTrue(initial_curriculum_rollback_armed(4, None))
        self.assertFalse(initial_curriculum_rollback_armed(
            4, {"rollback_armed": False}
        ))

        successes = [1.0] * 150 + [0.0] * 50
        self.assertFalse(should_promote_curriculum(
            successes, 0.75, 200, 99999, 100000
        ))
        self.assertTrue(should_promote_curriculum(
            successes, 0.75, 200, 100000, 100000
        ))
        self.assertFalse(should_promote_curriculum(
            [1.0] * 149 + [0.0] * 51,
            0.75,
            200,
            100000,
            100000,
        ))
        self.assertFalse(should_demote_curriculum(
            [0.0] * 100, 0.10, 100, 49999, 50000
        ))
        self.assertTrue(should_demote_curriculum(
            [0.0] * 90 + [1.0] * 10,
            0.10,
            100,
            50000,
            50000,
        ))

        args.env_id = "RM75Approach-v0"
        args.target_direction_mode = "random_cone"
        envs = gym.vector.SyncVectorEnv([
            make_env(args.env_id, 0, False, "curriculum-test", args.gamma, args=args)
        ])
        try:
            applied = apply_approach_curriculum(envs, args, stage=1)
            self.assertEqual(applied["stage"], 1)
            env = envs.envs[0].unwrapped
            self.assertAlmostEqual(env.active_dist_success_thresh, 0.12)
            self.assertAlmostEqual(env.target_tilt_max_deg, 15.0)
        finally:
            envs.close()

    def test_async_vector_env_applies_curriculum_and_steps(self):
        args = Args()
        args.env_id = "RM75Approach-v0"
        args.num_envs = 2
        args.vector_env_mode = "async"
        args.target_direction_mode = "random_cone"
        envs = make_vector_env(args, "async-curriculum-test")
        try:
            for process in envs.processes:
                os.kill(process.pid, signal.SIGINT)
            time.sleep(0.1)
            applied = apply_approach_curriculum(envs, args, stage=2)
            self.assertEqual(applied["stage"], 2)
            observations, _ = envs.reset(seed=17)
            self.assertEqual(observations.shape, (2, 34))
            observations, rewards, terminated, truncated, _ = envs.step(
                np.zeros((2, 7), dtype=np.float32)
            )
            self.assertEqual(observations.shape, (2, 34))
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
        finally:
            envs.close()

    def test_default_args_and_agent_match_rm75_spaces(self):
        args = Args()
        args.target_mode = "none"

        self.assertEqual(args.env_id, "RM75-v0")
        self.assertAlmostEqual(args.ent_coef, 0.0001)
        self.assertAlmostEqual(args.initial_action_std, 0.30)
        self.assertAlmostEqual(args.final_action_std, 0.12)
        self.assertAlmostEqual(args.action_std_hold_fraction, 0.25)
        self.assertEqual(args.target_sampling, "uniform")
        self.assertAlmostEqual(args.collision_penalty, 75.0)
        self.assertEqual(args.eval_interval, 0)

        envs = gym.vector.SyncVectorEnv([
            make_env(
                args.env_id,
                0,
                False,
                "rm75-test",
                args.gamma,
                args=args,
            )
        ])
        try:
            self.assertEqual(envs.single_observation_space.shape, (28,))
            self.assertEqual(envs.single_action_space.shape, (7,))
            self.assertEqual(
                envs.envs[0].unwrapped.target_sampling,
                "uniform",
            )
            self.assertAlmostEqual(
                envs.envs[0].unwrapped.reward_cfg[
                    "collision_penalty"
                ],
                75.0,
            )

            agent = Agent(
                envs,
                initial_action_std=args.initial_action_std,
            )
            observations, _ = envs.reset(seed=5)
            observation_tensor = torch.as_tensor(
                observations,
                dtype=torch.float32,
            )
            action, logprob, entropy, value = (
                agent.get_action_and_value(observation_tensor)
            )

            self.assertEqual(action.shape, (1, 7))
            self.assertEqual(logprob.shape, (1,))
            self.assertEqual(entropy.shape, (1,))
            self.assertEqual(value.shape, (1, 1))
            self.assertTrue(torch.isfinite(action).all())
            self.assertTrue(torch.isfinite(logprob).all())
            self.assertTrue(torch.isfinite(entropy).all())
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(np.all(action.detach().numpy() <= 1.0))
            self.assertTrue(np.all(action.detach().numpy() >= -1.0))

            agent.set_action_std_ceiling(args.final_action_std)
            effective_logstd = agent._effective_action_logstd(
                torch.zeros((1, 7), dtype=torch.float32)
            )
            expected_ceiling = np.log(args.final_action_std)
            self.assertTrue(
                torch.all(
                    effective_logstd
                    <= expected_ceiling + 1e-6
                )
            )
            agent.reset_action_std(0.18)
            reset_logstd = agent._effective_action_logstd(
                torch.zeros((1, 7), dtype=torch.float32)
            )
            torch.testing.assert_close(
                reset_logstd,
                torch.full_like(reset_logstd, np.log(0.18)),
            )
        finally:
            envs.close()

    def test_action_std_holds_then_decays_to_final_value(self):
        initial = 0.30
        final = 0.12
        hold_fraction = 0.25
        num_iterations = 9

        self.assertAlmostEqual(
            compute_action_std(
                initial,
                final,
                iteration=1,
                num_iterations=num_iterations,
                hold_fraction=hold_fraction,
            ),
            initial,
        )
        self.assertAlmostEqual(
            compute_action_std(
                initial,
                final,
                iteration=6,
                num_iterations=num_iterations,
                hold_fraction=hold_fraction,
            ),
            0.21,
        )
        self.assertAlmostEqual(
            compute_action_std(
                initial,
                final,
                iteration=3,
                num_iterations=num_iterations,
                hold_fraction=hold_fraction,
            ),
            initial,
        )
        self.assertAlmostEqual(
            compute_action_std(
                initial,
                final,
                iteration=9,
                num_iterations=num_iterations,
                hold_fraction=hold_fraction,
            ),
            final,
        )

    def test_curriculum_action_std_restarts_for_each_stage(self):
        self.assertAlmostEqual(
            compute_stage_action_std(0.18, 0.10, 0, 300000, 0.25),
            0.18,
        )
        self.assertAlmostEqual(
            compute_stage_action_std(
                0.18, 0.10, 75000, 300000, 0.25
            ),
            0.18,
        )
        self.assertAlmostEqual(
            compute_stage_action_std(
                0.18, 0.10, 300000, 300000, 0.25
            ),
            0.10,
        )

    def test_stage_transition_does_not_raise_learned_action_std(self):
        agent = Agent(DummyRM75Spaces(), initial_action_std=0.08)
        agent.set_action_std_ceiling(0.055)
        agent.set_action_std_ceiling(0.08)

        mean_std, min_std, max_std = agent.effective_action_std_stats()
        self.assertAlmostEqual(mean_std, 0.055, places=6)
        self.assertAlmostEqual(min_std, 0.055, places=6)
        self.assertAlmostEqual(max_std, 0.055, places=6)

    def test_learning_rate_is_globally_monotonic(self):
        values = [
            compute_learning_rate(1e-5, 2e-6, step, 2_000_000)
            for step in (0, 300_000, 600_000, 900_000)
        ]

        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[1], values[2])

    def test_curriculum_uses_stage_specific_promotion_thresholds(self):
        args = Args()
        args.curriculum_promotion_success_rate = 0.75
        args.curriculum_stage3_promotion_success_rate = 0.50
        args.curriculum_stage4_promotion_success_rate = 0.60
        args.curriculum_stage5_promotion_success_rate = 0.60

        self.assertAlmostEqual(curriculum_promotion_threshold(args, 1), 0.75)
        self.assertAlmostEqual(curriculum_promotion_threshold(args, 2), 0.75)
        self.assertAlmostEqual(curriculum_promotion_threshold(args, 3), 0.50)
        self.assertAlmostEqual(curriculum_promotion_threshold(args, 4), 0.60)
        self.assertAlmostEqual(curriculum_promotion_threshold(args, 5), 0.60)
        self.assertAlmostEqual(curriculum_promotion_threshold(args, 6), 1.0)

    def test_finetune_can_start_from_curriculum_stage_three(self):
        args = Args()
        args.curriculum_start_stage = 3

        self.assertEqual(initial_curriculum_stage(args, None), 3)
        self.assertEqual(
            initial_curriculum_stage(args, {"stage": 4}),
            4,
        )
        args.curriculum_start_stage = 6
        self.assertEqual(initial_curriculum_stage(args, None), 6)
        args.curriculum_start_stage = 7
        with self.assertRaisesRegex(ValueError, "within \\[1, 6\\]"):
            initial_curriculum_stage(args, None)

    def test_effective_action_std_stats_report_actual_clamped_values(self):
        agent = Agent(DummyRM75Spaces(), initial_action_std=0.14)
        with torch.no_grad():
            values = torch.tensor(
                [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
            )
            agent.actor_logstd.copy_(values.log().reshape(1, -1))
        agent.set_action_std_ceiling(0.12)

        mean_std, min_std, max_std = agent.effective_action_std_stats()

        self.assertAlmostEqual(min_std, 0.08, places=6)
        self.assertAlmostEqual(max_std, 0.12, places=6)
        self.assertAlmostEqual(
            mean_std,
            float(np.mean([0.08, 0.09, 0.10, 0.11, 0.12, 0.12, 0.12])),
            places=6,
        )

    def test_agent_weight_export_does_not_mutate_exploration(self):
        agent = Agent(DummyRM75Spaces(), initial_action_std=0.14)
        with torch.no_grad():
            agent.actor_logstd.fill_(float(np.log(0.123)))
        before = agent.actor_logstd.detach().clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "agent.cleanrl_model"
            save_agent_weights(model_path, agent)
            saved = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )

        torch.testing.assert_close(agent.actor_logstd, before)
        torch.testing.assert_close(saved["actor_logstd"], before)

    def test_approach_environment_kwargs_and_spaces(self):
        args = Args()
        args.env_id = "RM75Approach-v0"
        args.target_mode = "random"
        args.target_direction = (0.5, 0.0, -0.8660254)
        args.target_direction_mode = "random_cone"
        args.target_tilt_min_deg = 2.0
        args.target_tilt_max_deg = 15.0
        args.target_azimuth_min_deg = -90.0
        args.target_azimuth_max_deg = 90.0
        args.approach_position_box = (-0.45, -0.35, -0.18, 0.18)
        args.require_pose_feasible = False
        args.pose_feasibility_attempts = 12
        args.orient_success_threshold_deg = 12.0

        kwargs = environment_kwargs(
            args.env_id,
            args,
            render_mode=None,
        )
        self.assertEqual(kwargs["target_direction"], args.target_direction)
        self.assertEqual(kwargs["target_direction_mode"], "random_cone")
        self.assertEqual(kwargs["target_tilt_min_deg"], 2.0)
        self.assertEqual(kwargs["target_tilt_max_deg"], 15.0)
        self.assertEqual(kwargs["target_azimuth_min_deg"], -90.0)
        self.assertEqual(kwargs["target_azimuth_max_deg"], 90.0)
        self.assertEqual(
            kwargs["approach_position_box"],
            (-0.45, -0.35, -0.18, 0.18),
        )
        self.assertFalse(kwargs["require_pose_feasible"])
        self.assertEqual(kwargs["pose_feasibility_attempts"], 12)
        self.assertEqual(kwargs["orient_success_threshold_deg"], 12.0)

        envs = gym.vector.SyncVectorEnv([
            make_env(
                args.env_id,
                0,
                False,
                "rm75-approach-test",
                args.gamma,
                args=args,
            )
        ])
        try:
            self.assertEqual(envs.single_observation_space.shape, (34,))
            self.assertEqual(envs.single_action_space.shape, (7,))
            observation, _ = envs.reset(seed=43)
            self.assertEqual(observation.shape, (1, 34))
            agent = Agent(envs)
            action = torch.tanh(
                agent.actor_mean(torch.as_tensor(observation))
            )
            self.assertEqual(action.shape, (1, 7))
        finally:
            envs.close()

    def test_legacy_weights_can_initialize_a_new_finetune_run(self):
        spaces = DummyRM75Spaces()
        source = Agent(spaces, initial_action_std=0.12)
        expected = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
        }
        restored = Agent(spaces, initial_action_std=0.12)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "legacy.cleanrl_model"
            torch.save(source.state_dict(), model_path)
            source_kind = load_agent_weights(
                model_path,
                restored,
                device="cpu",
            )

        self.assertEqual(source_kind, "legacy_weights")
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[key])

    def test_pose_environment_kwargs_and_network_profile(self):
        args = Args()
        args.env_id = "RM75Pose-v0"
        args.target_direction_mode = "random_cone"
        args.target_tilt_max_deg = 30.0
        args.orient_success_threshold_deg = 20.0

        kwargs = environment_kwargs(
            args.env_id,
            args,
            render_mode=None,
        )
        self.assertTrue(kwargs["approach_task"])
        self.assertTrue(kwargs["pose_task"])
        self.assertEqual(kwargs["target_tilt_max_deg"], 30.0)

        class PoseSpaces:
            observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20,),
                dtype=np.float32,
            )
            action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32,
            )

        self.assertEqual(resolve_network_profile("auto", 20), "pose")
        self.assertEqual(resolve_network_profile("auto", 34), "legacy")
        agent = Agent(PoseSpaces(), network_profile="pose")
        self.assertEqual(agent.actor_mean[0].in_features, 20)
        self.assertEqual(agent.actor_mean[0].out_features, 256)
        self.assertEqual(agent.actor_mean[2].out_features, 128)
        self.assertEqual(agent.actor_mean[4].out_features, 64)
        self.assertIsInstance(agent.actor_mean[1], torch.nn.ELU)
        self.assertEqual(agent.actor_mean[6].out_features, 7)

    def test_full_checkpoint_restores_training_state(self):
        worker = Path(__file__).with_name(
            "checkpoint_roundtrip_worker.py"
        )
        worker_env = os.environ.copy()
        worker_env.pop("MUJOCO_GL", None)
        result = subprocess.run(
            [sys.executable, str(worker)],
            cwd=worker.parents[1],
            env=worker_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("checkpoint round-trip: OK", result.stdout)

    def test_next_checkpoint_step_uses_the_next_strict_boundary(self):
        self.assertEqual(next_checkpoint_step(0, 100000), 100000)
        self.assertEqual(next_checkpoint_step(99999, 100000), 100000)
        self.assertEqual(next_checkpoint_step(100000, 100000), 200000)
        self.assertEqual(next_checkpoint_step(102400, 100000), 200000)
        self.assertIsNone(next_checkpoint_step(102400, 0))

    def test_best_eval_prefers_success_before_collision(self):
        best = {
            "success_rate": 0.80,
            "collision_rate": 0.00,
            "mean_final_distance": 0.10,
        }
        candidate = {
            "success_rate": 0.84,
            "collision_rate": 0.08,
            "mean_final_distance": 0.12,
        }

        self.assertTrue(
            should_update_best(
                candidate,
                best,
                max_collision_rate=0.10,
            )
        )

    def test_best_eval_rejects_excess_collision(self):
        candidate = {
            "success_rate": 0.90,
            "collision_rate": 0.11,
            "mean_final_distance": 0.05,
        }

        self.assertFalse(
            should_update_best(
                candidate,
                None,
                max_collision_rate=0.10,
            )
        )

    def test_best_eval_breaks_success_ties_by_safety_then_distance(self):
        best = {
            "success_rate": 0.84,
            "collision_rate": 0.06,
            "mean_final_distance": 0.11,
        }
        safer = {
            "success_rate": 0.84,
            "collision_rate": 0.04,
            "mean_final_distance": 0.13,
        }
        closer = {
            "success_rate": 0.84,
            "collision_rate": 0.06,
            "mean_final_distance": 0.09,
        }

        self.assertTrue(
            should_update_best(safer, best, 0.10)
        )
        self.assertTrue(
            should_update_best(closer, best, 0.10)
        )

    def test_best_approach_eval_uses_angle_after_distance_tie(self):
        best = {
            "success_rate": 0.70,
            "collision_rate": 0.05,
            "mean_final_distance": 0.06,
            "mean_final_angle_error_deg": 14.0,
        }
        better_angle = {
            "success_rate": 0.70,
            "collision_rate": 0.05,
            "mean_final_distance": 0.06,
            "mean_final_angle_error_deg": 9.0,
        }

        self.assertTrue(
            should_update_best(better_angle, best, 0.10)
        )

    def test_best_approach_eval_prefers_lower_joint_pose_error(self):
        best = {
            "success_rate": 0.20,
            "collision_rate": 0.04,
            "mean_min_pose_error": 0.18,
            "ever_position_success_rate": 0.55,
            "mean_final_distance": 0.05,
            "mean_final_angle_error_deg": 12.0,
        }
        better_pose = {
            "success_rate": 0.20,
            "collision_rate": 0.04,
            "mean_min_pose_error": 0.12,
            "ever_position_success_rate": 0.50,
            "mean_final_distance": 0.08,
            "mean_final_angle_error_deg": 15.0,
        }

        self.assertTrue(
            should_update_best(better_pose, best, 0.10)
        )

    def test_unmastered_harder_stage_does_not_replace_mastered_best(self):
        mastered_stage1 = {
            "curriculum_stage": 1,
            "stage_mastered": True,
            "strict_success_rate": 0.0,
            "success_rate": 0.80,
            "collision_rate": 0.0,
            "mean_final_distance": 0.10,
        }
        unmastered_stage2 = {
            "curriculum_stage": 2,
            "stage_mastered": False,
            "strict_success_rate": 0.0,
            "success_rate": 0.20,
            "collision_rate": 0.0,
            "mean_final_distance": 0.12,
        }
        mastered_stage2 = dict(
            unmastered_stage2,
            stage_mastered=True,
            success_rate=0.72,
        )

        self.assertFalse(should_update_best(
            unmastered_stage2, mastered_stage1, 0.10
        ))
        self.assertTrue(should_update_best(
            mastered_stage2, mastered_stage1, 0.10
        ))

    def test_deterministic_evaluation_reports_metrics(self):
        args = Args()
        args.target_mode = "fixed"
        args.target_pos = (0.4, 0.0, 0.2905)
        args.eval_episodes = 1
        args.eval_seed = 91
        args.collision_penalty = 45.0
        envs = gym.vector.SyncVectorEnv([
            make_env(
                args.env_id,
                0,
                False,
                "rm75-eval-test",
                args.gamma,
                args=args,
            )
        ])
        try:
            agent = Agent(envs)
            metrics = evaluate_policy(
                agent,
                args,
                torch.device("cpu"),
            )
        finally:
            envs.close()

        self.assertEqual(
            set(metrics),
            {
                "success_rate",
                "success_rate_standard_error",
                "collision_rate",
                "timeout_rate",
                "mean_final_distance",
                "mean_episode_length",
            },
        )
        self.assertTrue(0.0 <= metrics["success_rate"] <= 1.0)
        self.assertTrue(0.0 <= metrics["collision_rate"] <= 1.0)
        self.assertTrue(np.isfinite(metrics["mean_final_distance"]))
        self.assertGreater(metrics["mean_episode_length"], 0.0)

    def test_approach_evaluation_reports_direction_metrics(self):
        args = Args()
        args.env_id = "RM75Approach-v0"
        args.target_mode = "fixed"
        args.target_pos = (0.4, 0.0, 0.2905)
        args.target_direction = (0.0, 0.0, -1.0)
        args.orient_success_threshold_deg = 15.0
        args.eval_episodes = 1
        args.eval_seed = 97
        envs = gym.vector.SyncVectorEnv([
            make_env(
                args.env_id,
                0,
                False,
                "rm75-approach-eval-test",
                args.gamma,
                args=args,
            )
        ])
        try:
            agent = Agent(envs)
            metrics = evaluate_policy(
                agent,
                args,
                torch.device("cpu"),
                curriculum_stage=1,
            )
        finally:
            envs.close()

        self.assertIn("direction_success_rate", metrics)
        self.assertIn("strict_success_rate", metrics)
        self.assertIn("strict_success_rate_standard_error", metrics)
        self.assertEqual(metrics["curriculum_stage"], 1)
        self.assertIn("mean_final_angle_error_deg", metrics)
        self.assertTrue(
            0.0 <= metrics["direction_success_rate"] <= 1.0
        )
        self.assertTrue(
            np.isfinite(metrics["mean_final_angle_error_deg"])
        )

    def test_resume_configuration_restores_training_but_not_device(self):
        args = Args()
        args.cuda = False
        payload = {
            "args": {
                "exp_name": "saved-experiment-name",
                "learning_rate": 2e-5,
                "target_sampling": "hard_mixture",
                "total_timesteps": 500000,
                "track": False,
                "cuda": True,
                "collision_penalty": 45.0,
                "eval_interval": 200000,
                "network_profile": "pose",
            }
        }

        apply_resume_configuration(args, payload)

        self.assertEqual(args.exp_name, "saved-experiment-name")
        self.assertAlmostEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.target_sampling, "hard_mixture")
        self.assertEqual(args.total_timesteps, 500000)
        self.assertFalse(args.track)
        self.assertFalse(args.cuda)
        self.assertAlmostEqual(args.collision_penalty, 45.0)
        self.assertEqual(args.eval_interval, 200000)
        self.assertEqual(args.network_profile, "pose")

    def test_additional_timesteps_extend_completed_checkpoint_by_batches(self):
        args = Args()
        args.additional_timesteps = 2_000_000
        payload = {"completed_iteration": 976}

        configure_training_horizon(args, payload)

        self.assertEqual(args.batch_size, 2048)
        self.assertEqual(args.minibatch_size, 128)
        self.assertEqual(args.num_iterations, 1952)
        self.assertEqual(args.total_timesteps, 3_997_696)

    def test_additional_timesteps_require_resume_checkpoint(self):
        args = Args()
        args.additional_timesteps = 2_000_000

        with self.assertRaisesRegex(ValueError, "requires --resume-from"):
            configure_training_horizon(args, None)

    def test_additional_timesteps_must_be_non_negative(self):
        args = Args()
        args.additional_timesteps = -1

        with self.assertRaisesRegex(ValueError, "non-negative"):
            configure_training_horizon(args, {"completed_iteration": 976})

    def test_additional_timesteps_require_one_complete_batch(self):
        args = Args()
        args.additional_timesteps = 2047

        with self.assertRaisesRegex(ValueError, "complete PPO batch"):
            configure_training_horizon(args, {"completed_iteration": 976})

    def test_cuda_rng_states_are_cpu_byte_tensors_for_restore(self):
        states = normalize_cuda_rng_states(
            [torch.tensor([0, 255, 256], dtype=torch.int64)]
        )

        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].device.type, "cpu")
        self.assertEqual(states[0].dtype, torch.uint8)
        self.assertTrue(states[0].is_contiguous())
        torch.testing.assert_close(
            states[0],
            torch.tensor([0, 255, 0], dtype=torch.uint8),
        )

if __name__ == "__main__":
    unittest.main()
