#!/usr/bin/env python3
"""Evaluate UR5/RM75 point policies or the RM75 approach-direction policy.

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

# ==================== 1. 依赖导入与全局配置 ====================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithms.ppo_continuous_action import Agent
from env import UR5P2PEnv
from env_rm75 import RM75P2PEnv


SUPPORTED_MODEL_ENVS = {
    (19, 6): "UR5-v0",
    (21, 7): "RM75-v0",
    (28, 7): "RM75-v0",
    (31, 7): "RM75Direction-v0",
}


# ==================== 2. 命令行参数与输入检查 ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize or batch-evaluate a trained UR5 or RM75 random-point "
            "reaching policy."
        )
    )
    parser.add_argument(
        "--env-id",
        choices=("auto", "UR5-v0", "RM75-v0", "RM75Direction-v0"),
        default="auto",
        help=(
            "Evaluation environment. The default infers it from the saved "
            "observation/action dimensions."
        ),
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
        "--dist-success-thresh",
        type=float,
        default=0.05,
        help="Success distance threshold for RM75 point-reaching.",
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
    if not np.isfinite(args.dist_success_thresh) or args.dist_success_thresh <= 0.0:
        parser.error("--dist-success-thresh must be finite and positive")
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.target_mode == "fixed" and args.target_pos is None:
        parser.error("--target-mode fixed requires --target-pos X Y Z")
    return args


# ==================== 3. 模型路径、权重读取与维度识别 ====================

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
    """Read observation and action dimensions from a saved point policy."""
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
    """Map supported saved dimensions to the matching point environment."""
    dimensions = model_dimensions(state)
    try:
        return SUPPORTED_MODEL_ENVS[dimensions]
    except KeyError as exc:
        raise ValueError(
            "Unsupported model dimensions: "
            f"{dimensions[0]} observations and {dimensions[1]} actions. "
            "Supported models are UR5-v0 (19/6) and RM75-v0 (21/7 or legacy 28/7). "
            "Direction models use RM75Direction-v0 (31/7)."
        ) from exc


# ==================== 4. 环境构建与历史28维观测兼容 ====================

class LegacyRM75ObservationEnv(RM75P2PEnv):
    """仅供可视化：恢复历史点任务的28维观测。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 * self.nq + 3 + 3 + 1,),
            dtype=np.float32,
        )

    def _get_obs(self):
        observation = super()._get_obs()

        if self.q_target is None:
            tracking_error = np.zeros(
                self.nq,
                dtype=np.float64,
            )
        else:
            tracking_error = (
                self.q_target - self.data.qpos[:self.nq]
            )

        # 当前21维：q、qd、TCP位置、目标位置、碰撞。
        # 历史28维：在qd之后插入7维真实追踪误差。
        split = 2 * self.nq
        return np.concatenate(
            [
                observation[:split],
                tracking_error,
                observation[split:],
            ]
        ).astype(np.float32)


def create_environment(
    env_id: str,
    args,
    render_mode,
    observation_dim: int,
):
    """Construct the point or direction environment matching saved inputs."""
    kwargs = {
        "render_mode": render_mode,
        "seed": args.seed,
        "target_mode": args.target_mode,
        "target_pos": tuple(args.target_pos) if args.target_pos is not None else None,
    }
    if env_id == "UR5-v0":
        return UR5P2PEnv(**kwargs)
    if env_id == "RM75Direction-v0":
        if observation_dim != 31:
            raise ValueError("RM75Direction-v0 requires 31 observations")
        return RM75P2PEnv(
            **kwargs,
            require_direction=True,
            dist_success_thresh=args.dist_success_thresh,
        )
    if env_id == "RM75-v0":
        if observation_dim == 28:
            env_class = LegacyRM75ObservationEnv
        elif observation_dim == 21:
            env_class = RM75P2PEnv
        else:
            raise ValueError("RM75-v0 requires 21 or 28 observations")
        return env_class(**kwargs, dist_success_thresh=args.dist_success_thresh)
    raise ValueError(f"Unsupported environment: {env_id}")


# ==================== 5. 策略网络加载与动作推理 ====================

def load_agent(state: dict[str, torch.Tensor], env, env_id: str) -> Agent:
    """Create a V4 Agent and reject model/environment mismatches clearly."""
    model_shape = model_dimensions(state)
    env_shape = (
        int(np.prod(env.observation_space.shape)),
        int(np.prod(env.action_space.shape)),
    )
    if model_shape != env_shape:
        raise ValueError(
            f"The model expects {model_shape[0]} observations and "
            f"{model_shape[1]} actions, but {env_id} provides "
            f"{env_shape[0]} observations and {env_shape[1]} actions."
        )

    agent = Agent(env).to(torch.device("cpu"))
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


# ==================== 6. 评估主流程与结果统计 ====================

def main() -> None:
    # ---------- 6.1 参数解析、模型加载与环境初始化 ----------
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    state = load_model_state(model_path)
    observation_dim, action_dim = model_dimensions(state)

    env_id = (
        infer_environment_id(state)
        if args.env_id == "auto"
        else args.env_id
    )

    render_mode = None if args.headless else "human"
    env = create_environment(
        env_id,
        args,
        render_mode,
        observation_dim=observation_dim,
    )
    agent = load_agent(state, env, env_id)

    action_mode = (
        "stochastic"
        if args.stochastic
        else "deterministic"
    )
    print(f"Model: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Observation dimensions: {observation_dim}")
    print(f"Action dimensions: {action_dim}")
    print(f"Action mode: {action_mode}")
    print(f"Target mode: {args.target_mode}")

    if env_id in {"RM75-v0", "RM75Direction-v0"}:
        print(
            "Success distance threshold: "
            f"{env.reward_cfg['dist_success_thresh']:.3f} m"
        )
        if observation_dim == 28:
            print(
                "Legacy 28-D observation compatibility enabled. "
                "Physics and reset logic use the current environment."
            )

    results: list[dict[str, float | int | bool | str]] = []
    simulation_step_seconds = float(
        env.model.opt.timestep * env.substeps
    )

    # ---------- 6.2 回合重置、策略执行与实时诊断 ----------
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            initial_distance = float(env.prev_dist)
            min_distance = initial_distance
            previous_tcp = env.data.site_xpos[env.ee_sid].copy()
            total_reward = 0.0
            info: dict = {}

            if not args.headless:
                print(
                    f"\nEpisode {episode + 1}/{args.episodes}: "
                    f"goal={np.round(env.goal_pos, 4)}, "
                    f"initial_dist={initial_distance:.4f}"
                )

            for step in range(1, env.horizon + 1):
                wall_start = time.perf_counter()
                action = select_action(
                    agent,
                    observation,
                    stochastic=args.stochastic,
                )
                previous_q = env.data.qpos[:env.nq].copy()
                previous_q_ref = env.q_target[:env.nq].copy()
                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                min_distance = min(min_distance, float(info["dist"]))
                reference_delta = (
                    env.q_target[:env.nq] - previous_q_ref
                )
                joint_delta = (
                    env.data.qpos[:env.nq] - previous_q
                )
                current_tcp = env.data.site_xpos[env.ee_sid].copy()
                tcp_speed = float(
                    np.linalg.norm(current_tcp - previous_tcp)
                    / simulation_step_seconds
                )
                previous_tcp = current_tcp
                total_reward += float(reward)

                if (
                    args.progress_every
                    and step % args.progress_every == 0
                ):
                    q = env.data.qpos[:env.nq]
                    q_ref = env.q_target[:env.nq]
                    tracking_error = float(np.max(np.abs(q_ref - q)))
                    lag_joint = int(np.argmax(np.abs(q_ref - q)))

                    joint_ranges = env.model.jnt_range[:env.nq]
                    limited = env.model.jnt_limited[:env.nq].astype(bool)
                    near_limit = np.flatnonzero(
                        limited & (
                            (np.abs(q_ref - joint_ranges[:, 0]) < 0.01)
                            | (np.abs(q_ref - joint_ranges[:, 1]) < 0.01)
                        )
                    ) + 1

                    print(
                        f"  step={step}, dist={float(info['dist']):.4f}, "
                        f"reward={float(reward):.3f}, "
                        f"tcp_speed={tcp_speed:.4f}m/s, "
                        f"track_err={tracking_error:.4f}rad, "
                        f"dqref_max={np.max(np.abs(reference_delta)):.4f}rad, "
                        f"lag_joint={lag_joint + 1}, "
                        f"dqref_j={reference_delta[lag_joint]:+.4f}rad, "
                        f"dq_j={joint_delta[lag_joint]:+.4f}rad, "
                        f"qref_limits={near_limit.tolist()}"
                    )

                if not args.headless and args.realtime:
                    elapsed = time.perf_counter() - wall_start
                    time.sleep(max(0.0, simulation_step_seconds - elapsed))

                if terminated or truncated:
                    break

            # ---------- 6.3 单回合结果记录与打印 ----------
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            timeout = not success and not collision
            reason = (
                "success"
                if success
                else "collision"
                if collision
                else "timeout"
            )
            final_distance = float(info.get("dist", np.nan))
            final_direction_angle = info.get("direction_angle_deg", np.nan)
            final_direction_angle = (
                np.nan
                if final_direction_angle is None
                else float(final_direction_angle)
            )
            results.append(
                {
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "reason": reason,
                    "steps": step,
                    "initial_distance": initial_distance,
                    "final_distance": final_distance,
                    "min_distance": min_distance,
                    "final_direction_angle_deg": final_direction_angle,
                    "position_success": bool(
                        info.get(
                            "position_reached",
                            info.get("position_success", success),
                        )
                    ),
                    "angle_success": bool(info.get("angle_reached", False)),
                    "pose_reached": bool(info.get("pose_reached", False)),
                    "min_direction_angle_at_position_deg": float(
                        info.get("min_direction_angle_at_position_deg", 180.0)
                    ),
                    "return": total_reward,
                }
            )
            print(
                f"Episode {episode + 1:03d}: {reason:9s} | "
                f"seed={args.seed + episode} | "
                f"goal={np.round(env.goal_pos, 4)} | "
                f"steps={step:4d} | final_dist={final_distance:.4f} | "
                f"min_dist={min_distance:.4f} | "
                f"entered_5cm={min_distance < 0.05} | "
                f"direction={final_direction_angle:6.2f} deg | "
                f"return={total_reward:.2f}"
            )
    finally:
        # ---------- 6.4 关闭环境与可视化资源 ----------
        env.close()

    # ---------- 6.5 全部回合的汇总统计与任务说明 ----------
    success_rate = np.mean([item["success"] for item in results])
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
    position_success_rate = np.mean(
        [item["position_success"] for item in results]
    )
    angle_success_rate = np.mean(
        [item["angle_success"] for item in results]
    )
    pose_reached_rate = np.mean([item["pose_reached"] for item in results])
    mean_min_angle_at_position = np.mean([
        item["min_direction_angle_at_position_deg"] for item in results
    ])
    direction_angles = np.asarray(
        [item["final_direction_angle_deg"] for item in results],
        dtype=np.float64,
    )
    mean_direction_angle = (
        float(np.nanmean(direction_angles))
        if not np.all(np.isnan(direction_angles))
        else np.nan
    )

    print("\nEvaluation summary")
    print(f"  episodes:             {len(results)}")
    print(f"  success_rate:         {success_rate:.3f}")
    if env_id == "RM75Direction-v0":
        print(f"  position_success_rate:{position_success_rate:9.3f}")
        print(f"  angle_success_rate:   {angle_success_rate:.3f}")
        print(f"  pose_reached_rate:    {pose_reached_rate:.3f}")
        print(f"  mean_min_angle_at_position_deg: {mean_min_angle_at_position:.2f}")
        print("  (180 deg represents an episode that never reached position.)")
    print(f"  collision_rate:       {collision_rate:.3f}")
    print(f"  timeout_rate:         {timeout_rate:.3f}")
    print(f"  mean_steps:           {mean_steps:.1f}")
    print(f"  mean_initial_distance:{mean_initial_distance:9.4f}")
    print(f"  mean_final_distance:  {mean_final_distance:.4f}")
    if env_id == "RM75Direction-v0":
        print(f"  mean_direction_deg:   {mean_direction_angle:.2f}")
    print(f"  mean_return:          {mean_return:.2f}")
    entered_5cm_rate = np.mean([
        item["min_distance"] < 0.05 for item in results
    ])
    mean_min_distance = np.mean([
        item["min_distance"] for item in results
    ])
    print(f"  entered_5cm_rate:     {entered_5cm_rate:.3f}")
    print(f"  mean_min_distance:    {mean_min_distance:.4f}")
    if env_id == "RM75Direction-v0":
        print(
            "\nTask scope: target position plus a fixed tool-axis approach "
            "direction, with rotation around that axis left free."
        )
    else:
        print("\nTask scope: end-effector point reaching only.")
    print("Object grasping and gripper closure are not simulated.")


# ==================== 7. 脚本入口 ====================

if __name__ == "__main__":
    main()
