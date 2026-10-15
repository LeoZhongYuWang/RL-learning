#!/usr/bin/env python3
"""Evaluate or visualize a trained UR5 or RM75 policy.

Examples:
    # Open the MuJoCo viewer with the newest saved model.
    python Scripts/visualize_agent.py

    # Evaluate 100 random goals without rendering.
    python Scripts/visualize_agent.py --headless --episodes 100

    # Evaluate a specific model with stochastic training-time actions.
    python Scripts/visualize_agent.py \
        --model-path runs/RUN_NAME/ppo_continuous_action.cleanrl_model \
        --headless \
        --episodes 100 \
        --stochastic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithms.ppo_continuous_action import Agent
from env import UR5P2PEnv
from env_rm75 import RM75P2PEnv


SUPPORTED_MODEL_ENVS = {
    (19, 6): "UR5-v0",
    (28, 7): "RM75-v0",
    (34, 7): "RM75Approach-v0",
    (20, 7): "RM75Pose-v0",
}


def latest_model_path() -> Path:
    """Return the most recently modified saved PPO model."""
    candidates = list(
        (PROJECT_ROOT / "runs").glob("**/*.cleanrl_model")
    )
    if not candidates:
        raise FileNotFoundError(
            "No .cleanrl_model file was found under runs/."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_model_path(model_path: str | None) -> Path:
    """Resolve an explicit model path or select the newest saved model."""
    if model_path is None:
        return latest_model_path()

    candidate = Path(model_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.is_file():
        return project_candidate.resolve()

    raise FileNotFoundError(f"Model file does not exist: {model_path}")


def load_model_state(model_path: Path) -> dict[str, torch.Tensor]:
    """Load an Agent state dict from weights or a full checkpoint."""
    try:
        payload = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")

    state = (
        payload["agent_state_dict"]
        if isinstance(payload, dict) and "agent_state_dict" in payload
        else payload
    )
    if not isinstance(state, dict):
        raise ValueError(
            f"Model file does not contain a state dictionary: {model_path}"
        )
    return state


def model_dimensions(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Return the observation and action dimensions encoded by a policy."""
    try:
        observation_dim = int(state["critic.0.weight"].shape[1])
        action_dim = int(state["actor_logstd"].shape[-1])
    except (KeyError, IndexError, AttributeError) as exc:
        raise ValueError(
            "Cannot determine model dimensions from critic.0.weight and "
            "actor_logstd."
        ) from exc
    return observation_dim, action_dim


def infer_environment_id(state: dict[str, torch.Tensor]) -> str:
    """Map a supported policy architecture to its matching environment."""
    dimensions = model_dimensions(state)
    try:
        return SUPPORTED_MODEL_ENVS[dimensions]
    except KeyError as exc:
        supported = ", ".join(
            f"{obs} observations/{actions} actions ({env_id})"
            for (obs, actions), env_id in SUPPORTED_MODEL_ENVS.items()
        )
        raise ValueError(
            "Unsupported model dimensions: "
            f"{dimensions[0]} observations and {dimensions[1]} actions. "
            f"Supported models: {supported}."
        ) from exc


def infer_network_profile(state: dict[str, torch.Tensor]) -> str:
    """Infer the legacy or pose MLP from its saved output-layer key."""
    if "actor_mean.6.weight" in state:
        return "pose"
    if "actor_mean.4.weight" in state:
        return "legacy"
    raise ValueError("Cannot infer network profile from saved actor layers.")


def validate_model_environment(
    state: dict[str, torch.Tensor],
    env,
    env_id: str,
) -> None:
    """Reject a model/environment mismatch before PyTorch loading."""
    model_observations, model_actions = model_dimensions(state)
    env_observations = int(np.prod(env.observation_space.shape))
    env_actions = int(np.prod(env.action_space.shape))
    if (model_observations, model_actions) != (
        env_observations,
        env_actions,
    ):
        raise ValueError(
            f"The model expects {model_observations} observations and "
            f"{model_actions} actions, but {env_id} provides "
            f"{env_observations} observations and {env_actions} actions. "
            "Use --env-id auto or select the matching environment."
        )


def create_environment(env_id: str, args, render_mode):
    """Construct the selected robot/task environment."""
    common_kwargs = {
        "render_mode": render_mode,
        "seed": args.seed,
        "target_mode": args.target_mode,
        "target_pos": (
            tuple(args.target_pos)
            if args.target_pos is not None
            else None
        ),
    }
    if env_id == "UR5-v0":
        return UR5P2PEnv(**common_kwargs)
    if env_id == "RM75-v0":
        return RM75P2PEnv(**common_kwargs)
    if env_id in {"RM75Approach-v0", "RM75Pose-v0"}:
        return RM75P2PEnv(
            **common_kwargs,
            approach_task=True,
            pose_task=env_id == "RM75Pose-v0",
            target_direction=tuple(args.target_direction),
            target_direction_mode=args.target_direction_mode,
            target_tilt_min_deg=args.target_tilt_min_deg,
            target_tilt_max_deg=args.target_tilt_max_deg,
            target_azimuth_min_deg=args.target_azimuth_min_deg,
            target_azimuth_max_deg=args.target_azimuth_max_deg,
            approach_position_box=(
                tuple(args.approach_position_box)
                if args.approach_position_box is not None
                else None
            ),
            require_pose_feasible=args.require_pose_feasible,
            pose_feasibility_attempts=args.pose_feasibility_attempts,
            orient_success_threshold_deg=(
                args.orient_success_threshold_deg
            ),
        )
    raise ValueError(f"Unsupported environment: {env_id}")


def load_agent(state: dict[str, torch.Tensor], env, env_id: str) -> Agent:
    """Create an Agent with matching spaces and load saved weights."""
    validate_model_environment(state, env, env_id)
    agent = Agent(
        env,
        network_profile=infer_network_profile(state),
    ).to(torch.device("cpu"))

    agent.load_state_dict(state)
    agent.eval()
    return agent


def select_action(
    agent: Agent,
    observation: np.ndarray,
    *,
    stochastic: bool,
) -> np.ndarray:
    """Return either a deterministic policy action or a sampled action."""
    obs_tensor = torch.as_tensor(
        observation[None, :],
        dtype=torch.float32,
    )
    with torch.no_grad():
        if stochastic:
            action = agent.get_action_and_value(obs_tensor)[0]
        else:
            action = torch.tanh(agent.actor_mean(obs_tensor))
    return action[0].cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize or batch-evaluate a trained UR5 or RM75 policy."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to a .cleanrl_model file. When omitted, the newest model "
            "under runs/ is used."
        ),
    )
    parser.add_argument(
        "--env-id",
        choices=("auto", *SUPPORTED_MODEL_ENVS.values()),
        default="auto",
        help=(
            "Environment used for evaluation. The default reads the model "
            "dimensions and selects the matching UR5 or RM75 task."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="First evaluation seed; each episode adds one.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("random", "fixed"),
        default="random",
        help="How evaluation goals are selected.",
    )
    parser.add_argument(
        "--target-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed target position; requires --target-mode fixed.",
    )
    parser.add_argument(
        "--target-direction",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -1.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "Desired tool approach direction for RM75 approach/pose tasks. "
            "The vector is normalized by the environment."
        ),
    )
    parser.add_argument(
        "--target-direction-mode",
        choices=("fixed", "random_cone"),
        default="random_cone",
        help=(
            "Use one fixed vector or sample a new downward-cone direction "
            "for every RM75 approach/pose episode."
        ),
    )
    parser.add_argument(
        "--target-tilt-min-deg",
        type=float,
        default=0.0,
        help="Minimum random-cone tilt from world -Z.",
    )
    parser.add_argument(
        "--target-tilt-max-deg",
        type=float,
        default=30.0,
        help="Maximum random-cone tilt from world -Z.",
    )
    parser.add_argument(
        "--target-azimuth-min-deg",
        type=float,
        default=-180.0,
        help="Minimum random-cone azimuth around world -Z.",
    )
    parser.add_argument(
        "--target-azimuth-max-deg",
        type=float,
        default=180.0,
        help="Maximum random-cone azimuth around world -Z.",
    )
    parser.add_argument(
        "--approach-position-box",
        type=float,
        nargs=4,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help=(
            "Approach-task random target box. Default: "
            "-0.45 -0.35 -0.18 0.18."
        ),
    )
    parser.add_argument(
        "--require-pose-feasible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject sampled position/direction pairs that fail pose IK.",
    )
    parser.add_argument(
        "--pose-feasibility-attempts",
        type=int,
        default=20,
        help="Maximum approach-pose samples checked per reset.",
    )
    parser.add_argument(
        "--orient-success-threshold-deg",
        type=float,
        default=20.0,
        help="Direction-error threshold in degrees for RM75 pose tasks.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the viewer and run a faster statistical evaluation.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Sample from the training-time action distribution. The default "
            "uses deterministic mean actions."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print distance every N steps; use 0 to disable.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace rendered episodes at simulated real time.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.target_mode == "fixed" and args.target_pos is None:
        parser.error("--target-mode fixed requires --target-pos X Y Z")
    if args.orient_success_threshold_deg <= 0.0:
        parser.error("--orient-success-threshold-deg must be positive")
    if not (
        0.0
        <= args.target_tilt_min_deg
        <= args.target_tilt_max_deg
        <= 90.0
    ):
        parser.error(
            "tilt bounds must satisfy 0 <= min <= max <= 90"
        )
    if args.target_azimuth_min_deg > args.target_azimuth_max_deg:
        parser.error("azimuth min must not exceed azimuth max")
    if args.pose_feasibility_attempts <= 0:
        parser.error("--pose-feasibility-attempts must be positive")
    return args


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    state = load_model_state(model_path)
    inferred_env_id = infer_environment_id(state)
    env_id = inferred_env_id if args.env_id == "auto" else args.env_id
    render_mode = None if args.headless else "human"
    env = create_environment(env_id, args, render_mode)
    agent = load_agent(state, env, env_id)

    action_mode = "stochastic" if args.stochastic else "deterministic"
    print(f"Model: {model_path}")
    print(f"Environment: {env_id}")
    print(
        "Model dimensions: "
        f"{model_dimensions(state)[0]} observations, "
        f"{model_dimensions(state)[1]} actions"
    )
    print(f"Action mode: {action_mode}")
    print(f"Target mode: {args.target_mode}")

    results: list[dict[str, float | int | bool | str]] = []
    simulation_step_seconds = float(
        env.model.opt.timestep * env.substeps
    )

    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            initial_distance = float(env.prev_dist)
            total_reward = 0.0
            info: dict = {}

            if not args.headless:
                print(
                    f"\nEpisode {episode + 1}/{args.episodes}: "
                    f"goal={np.round(env.goal_pos, 4)}, "
                    f"initial_dist={initial_distance:.4f}"
                    + (
                        ", target_tilt="
                        f"{float(reset_info['target_tilt_deg']):.2f}deg, "
                        "target_azimuth="
                        f"{float(reset_info['target_azimuth_deg']):.2f}deg"
                        if "target_tilt_deg" in reset_info
                        else ""
                    )
                )

            for step in range(1, env.horizon + 1):
                wall_start = time.perf_counter()
                action = select_action(
                    agent,
                    observation,
                    stochastic=args.stochastic,
                )
                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )
                total_reward += float(reward)

                if (
                    not args.headless
                    and args.progress_every
                    and step % args.progress_every == 0
                ):
                    print(
                        f"  step={step}, dist={float(info['dist']):.4f}, "
                        + (
                            "angle="
                            f"{float(info['angle_error_deg']):.2f} deg, "
                            if "angle_error_deg" in info
                            else ""
                        )
                        + f"reward={float(reward):.3f}"
                    )

                if not args.headless and args.realtime:
                    elapsed = time.perf_counter() - wall_start
                    time.sleep(max(0.0, simulation_step_seconds - elapsed))

                if terminated or truncated:
                    break

            strict_success = bool(
                info.get("strict_success", info.get("success", False))
            )
            active_success = bool(
                info.get("ever_active_success", strict_success)
            )
            success = strict_success
            collision = bool(info.get("collision", False))
            timeout = bool(
                info.get("timeout", not success and not collision)
            )
            reason = (
                "success"
                if success
                else "collision"
                if collision
                else "timeout"
            )
            final_distance = float(info.get("dist", np.nan))
            final_angle_error_deg = float(
                info.get("angle_error_deg", np.nan)
            )
            results.append(
                {
                    "success": success,
                    "active_success": active_success,
                    "collision": collision,
                    "timeout": timeout,
                    "reason": reason,
                    "steps": step,
                    "initial_distance": initial_distance,
                    "final_distance": final_distance,
                    "final_angle_error_deg": final_angle_error_deg,
                    "return": total_reward,
                }
            )
            print(
                f"Episode {episode + 1:03d}: {reason:9s} | "
                f"steps={step:4d} | final_dist={final_distance:.4f} | "
                + (
                    f"angle={final_angle_error_deg:.2f} deg | "
                    if np.isfinite(final_angle_error_deg)
                    else ""
                )
                + f"return={total_reward:.2f}"
            )
    finally:
        env.close()

    success_rate = np.mean([item["success"] for item in results])
    active_success_rate = np.mean(
        [item["active_success"] for item in results]
    )
    collision_rate = np.mean([item["collision"] for item in results])
    timeout_rate = np.mean([item["timeout"] for item in results])
    mean_steps = np.mean([item["steps"] for item in results])
    mean_initial_distance = np.mean(
        [item["initial_distance"] for item in results]
    )
    mean_final_distance = np.mean(
        [item["final_distance"] for item in results]
    )
    mean_return = np.mean([item["return"] for item in results])
    final_angles = np.asarray(
        [item["final_angle_error_deg"] for item in results],
        dtype=np.float64,
    )

    print("\nEvaluation summary")
    print(f"  episodes:             {len(results)}")
    print(f"  strict_success_rate:  {success_rate:.3f}")
    print(f"  active_success_rate:  {active_success_rate:.3f}")
    print(f"  collision_rate:       {collision_rate:.3f}")
    print(f"  timeout_rate:         {timeout_rate:.3f}")
    print(f"  mean_steps:           {mean_steps:.1f}")
    print(f"  mean_initial_distance:{mean_initial_distance:9.4f}")
    print(f"  mean_final_distance:  {mean_final_distance:.4f}")
    if np.isfinite(final_angles).any():
        print(
            "  mean_final_angle_deg: "
            f"{float(np.nanmean(final_angles)):.2f}"
        )
    print(f"  mean_return:          {mean_return:.2f}")
    print(
        "\nTask scope: this environment evaluates end-effector reaching"
        + (
            " with a required approach direction."
            if env_id in {"RM75Approach-v0", "RM75Pose-v0"}
            else "."
        )
        + " It does not simulate object grasping or gripper closure."
    )


if __name__ == "__main__":
    main()
