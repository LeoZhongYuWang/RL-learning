# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

# ==================== 1. 依赖导入与环境注册 ====================

import json
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
# import mujoco as mj
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Allow this file to be run directly from Algorithms/ while importing the
# environment module located in the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import UR5P2PEnv
from env_rm75 import RM75P2PEnv

# Register the local environment without relying on a non-existent `UR5`
# package.  The guard keeps repeated imports safe in the same Python process.
if "UR5-v0" not in gym.registry:
    gym.register(
        id="UR5-v0",
        entry_point=UR5P2PEnv,
        kwargs={"render_mode": None, "target_mode": "random"},
        max_episode_steps=None,
    )

if "RM75-v0" not in gym.registry:
    gym.register(
        id="RM75-v0",
        entry_point=RM75P2PEnv,
        kwargs={"render_mode": None, "target_mode": "random"},
        max_episode_steps=None,
    )

# 停用方向任务：注册入口保留在环境注册区，不执行。
# if "RM75Direction-v0" not in gym.registry:
#     gym.register(
#         id="RM75Direction-v0",
#         entry_point=RM75P2PEnv,
#         kwargs={
#             "render_mode": None,
#             "target_mode": "random",
#             "require_direction": True,
#         },
#         max_episode_steps=None,
#     )

# ==================== 2. 训练参数与命令行配置 ====================

@dataclass
class Args:
    # 2.1 实验命名、设备、日志与模型管理
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rm75"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    init_from: Optional[str] = None
    """agent weights used to start a new fine-tuning run"""
    resume_from: Optional[str] = None
    """full checkpoint used to continue an interrupted run"""
    checkpoint_interval: int = 100000
    """environment steps between atomic full checkpoints; 0 disables periodic saves"""
    # 2.2 PPO采样、优化与探索参数
    # Algorithm specific arguments
    env_id: str = "RM75-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0001
    """coefficient of the entropy"""
    initial_action_std: float = 0.30
    """initial ceiling for each normalized action standard deviation"""
    final_action_std: float = 0.12
    """final ceiling for each normalized action standard deviation"""
    action_std_hold_fraction: float = 0.25
    """fraction of training that keeps the initial action standard deviation"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""

    # 2.3 目标与环境参数；方向近处起点参数继续保持注释
    target_mode: str = "random"
    """goal sampling mode passed to UR5 environment"""
    target_pos: Optional[Tuple[float, float, float]] = None
    """explicit goal position when target_mode=='fixed'"""
    dist_success_thresh: float = 0.05
    """RM75 point-reaching success distance in meters"""
    collision_penalty: float = 75.0
    """RM75 collision reward penalty; collision still terminates the episode"""
    # near_start_probability: float = 0.0
    # """direction-task fraction requesting near-goal training starts; 0 preserves full starts"""
    # near_start_from: Optional[str] = None
    # """frozen policy preparing arrival starts during training only; None uses IK starts"""
    # 2.4 固定种子评估与验收参数
    eval_interval: int = 0
    """environment steps between deterministic evaluations; 0 disables them"""
    eval_episodes: int = 100
    """number of fixed-seed episodes in each deterministic evaluation"""
    eval_seed: int = 20000
    """first fixed seed used by deterministic evaluation"""
    eval_max_collision_rate: float = 0.10
    """largest collision rate eligible for best-model selection"""
    eval_target_success: float = 0.82
    """success-rate target reported by deterministic evaluation"""

    # 2.5 运行时计算的批次与迭代数量
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# ==================== 3. 环境创建与通用辅助 ====================

def environment_kwargs(env_id, args: Args, *, render_mode=None):
    """Build environment kwargs without passing RM75-only fields to UR5."""
    kwargs = {
        "render_mode": render_mode,
        "target_mode": args.target_mode,
        "target_pos": args.target_pos,
    }
    # if env_id in {"RM75-v0", "RM75Direction-v0"}:
    if env_id == "RM75-v0":
        kwargs.update({
            "collision_penalty": args.collision_penalty,
            "dist_success_thresh": args.dist_success_thresh,
        })
    # if env_id == "RM75Direction-v0":
    #     kwargs["near_start_probability"] = args.near_start_probability
    return kwargs


def make_env(env_id, *, args: Args):
    def thunk():
        kwargs = environment_kwargs(env_id, args, render_mode=None)
        # arrival_starts = (
        #     env_id == "RM75Direction-v0" and args.near_start_from is not None
        #     and args.near_start_probability > 0.0
        # )
        # if arrival_starts:
        #     kwargs["near_start_probability"] = 0.0  # Prefix starts from the original pose.
        env = gym.make(
            env_id,
            **kwargs,
        )
        # if arrival_starts:
        #     env = ArrivalStartWrapper(env, args.near_start_from, args.near_start_probability)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# 停用方向任务：近处起点包装器属于环境构造，保留注释、不执行。
# class ArrivalStartWrapper(gym.Wrapper):
#     """Training-only reset preparation; prefix actions never enter PPO batches."""
#     prefix_limit = 512

#     def __init__(self, env, model_path, probability):
#         super().__init__(env)
#         if not self.unwrapped.require_direction or self.unwrapped.near_start_probability != 0:
#             raise ValueError("arrival starts require a direction environment with IK near starts disabled")
#         if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
#             raise ValueError("arrival-start probability must be in [0, 1]")
#         self.probability = float(probability)
#         self.teacher = Agent(env)
#         load_agent_weights(model_path, self.teacher, "cpu")
#         self.teacher.eval().requires_grad_(False)

#     def reset(self, *, seed=None, options=None):
#         obs, info = self.env.reset(seed=seed, options=options)
#         base = self.unwrapped
#         if self.probability == 0.0 or base.rng.random() >= self.probability:
#             return obs, info
#         base.near_start_requested = True
#         spec = mj.mjtState.mjSTATE_INTEGRATION
#         saved = np.empty(mj.mj_stateSize(base.model, spec))
#         mj.mj_getState(base.model, base.data, saved, spec)
#         q_target = base.q_target.copy()
#         for _ in range(min(self.prefix_limit, base.horizon)):
#             with torch.no_grad():
#                 action = torch.tanh(self.teacher.actor_mean(
#                     torch.as_tensor(obs[None], dtype=torch.float32),
#                 ))[0].numpy()
#             obs, _, terminated, truncated, info = self.env.step(action)
#             base.arrival_prefix_steps += 1
#             qualified = info["position_success"] and info["direction_success"]
#             if terminated or truncated or info["collision"] or qualified:
#                 break  # Do not start at a collision or an already-qualified pose.
#             if info["dist"] <= 0.06:
#                 base.near_start = True
#                 base.start_pos = base._ee_pos()
#                 base._update_visualization_sites()
#                 return base._finish_reset()
#         # Preserve the original target and physical state, not an easier redraw.
#         mj.mj_setState(base.model, base.data, saved, spec)
#         base.q_target = q_target
#         mj.mj_forward(base.model, base.data)
#         return base._finish_reset()


def log_scalars(writer, metrics, global_step, *, prefix="charts"):
    """Write scalar groups without changing metric names or step semantics."""
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", float(value), global_step)


# ==================== 4. Actor-Critic与动作分布 ====================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_action_std(
    initial_action_std,
    final_action_std,
    iteration,
    num_iterations,
    hold_fraction,
):
    """Hold exploration, then linearly decay it to the final value."""
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")
    if iteration < 1 or iteration > num_iterations:
        raise ValueError("iteration must be within [1, num_iterations]")
    if not 0.0 <= hold_fraction < 1.0:
        raise ValueError("hold_fraction must be within [0, 1)")

    training_progress = (iteration - 1.0) / max(
        num_iterations - 1,
        1,
    )
    if training_progress <= hold_fraction:
        return float(initial_action_std)

    decay_progress = (
        (training_progress - hold_fraction)
        / (1.0 - hold_fraction)
    )
    return float(
        initial_action_std
        + decay_progress * (final_action_std - initial_action_std)
    )


class Agent(nn.Module):
    def __init__(self, envs, initial_action_std=0.30):
        super().__init__()
        if initial_action_std <= 0.0:
            raise ValueError("initial_action_std must be positive")
        observation_space = getattr(
            envs,
            "single_observation_space",
            envs.observation_space,
        )
        action_space = getattr(
            envs,
            "single_action_space",
            envs.action_space,
        )
        action_dim = int(np.prod(action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # The environment accepts normalized actions in [-1, 1]. Start with a
        # moderate standard deviation and squash every Gaussian sample with
        # tanh so the action used for PPO is exactly the action sent to MuJoCo.
        self.actor_logstd = nn.Parameter(
            torch.full(
                (1, action_dim),
                float(np.log(initial_action_std)),
            )
        )
        self.register_buffer(
            "action_logstd_ceiling",
            torch.tensor(
                [[float(np.log(initial_action_std))]],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def set_action_std_ceiling(self, action_std):
        """Set the maximum exploration standard deviation for all joints."""
        if not np.isfinite(action_std) or action_std <= 0.0:
            raise ValueError("action_std must be finite and positive")
        logstd_ceiling = float(np.log(action_std))
        with torch.no_grad():
            self.action_logstd_ceiling.fill_(logstd_ceiling)
            self.actor_logstd.clamp_(max=logstd_ceiling)

    def _effective_action_logstd(self, action_mean):
        learned_logstd = torch.minimum(
            self.actor_logstd,
            self.action_logstd_ceiling,
        )
        learned_logstd = torch.clamp(
            learned_logstd,
            min=float(np.log(0.05)),
        )
        return learned_logstd.expand_as(action_mean)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        # The parameter may learn a smaller variance, but the scheduled
        # ceiling prevents exploration from re-expanding late in training.
        action_logstd = self._effective_action_logstd(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            raw_action = probs.rsample()
            action = torch.tanh(raw_action)
        else:
            # atanh is the inverse of tanh. Clamping only protects the
            # logarithm/inverse from exact floating-point boundary values.
            action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            raw_action = torch.atanh(action)

        # Change-of-variables correction for a tanh-squashed Gaussian.
        logprob = probs.log_prob(raw_action) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        logprob = logprob.sum(1)

        # The base-Gaussian entropy is a stable exploration diagnostic. The
        # policy likelihood above still uses the exact squashed log-probability.
        entropy = probs.entropy().sum(1)
        return action, logprob, entropy, self.critic(x)


# ==================== 5. 模型加载与断点续训 ====================

# 5.1 checkpoint格式、路径与文件读取
CHECKPOINT_FORMAT_VERSION = 1


def resolve_training_path(path_value):
    """Resolve a model/checkpoint path from cwd or the project root."""
    candidate = Path(path_value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.is_file():
        return project_candidate.resolve()
    raise FileNotFoundError(f"Training file does not exist: {path_value}")


def _torch_load(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def _is_full_training_checkpoint(payload):
    return (
        isinstance(payload, dict)
        and payload.get("format_version") == CHECKPOINT_FORMAT_VERSION
        and "agent_state_dict" in payload
        and "optimizer_state_dict" in payload
    )


# 5.2 init-from：加载权重；历史方向观测扩维分支仍保留注释
def load_agent_weights(path, agent, device):
    """Load legacy weights or only the Agent part of a full checkpoint."""
    payload = _torch_load(resolve_training_path(path), device)
    if _is_full_training_checkpoint(payload):
        state_dict = payload["agent_state_dict"]
        source_kind = "full_checkpoint_weights"
    elif isinstance(payload, dict) and "agent_state_dict" in payload:
        state_dict = payload["agent_state_dict"]
        source_kind = "wrapped_weights"
    else:
        state_dict = payload
        source_kind = "legacy_weights"
    target_state = agent.state_dict()
    source_obs_dim = int(state_dict["actor_mean.0.weight"].shape[1])
    target_obs_dim = int(target_state["actor_mean.0.weight"].shape[1])
    if source_obs_dim == target_obs_dim:
        agent.load_state_dict(state_dict, strict=True)
    # elif source_obs_dim == 28 and target_obs_dim == 31:
    #     # Preserve the validated 28-D point-reaching policy. The three new
    #     # direction-error inputs start with zero influence and are learned by
    #     # fine-tuning; every other compatible parameter is copied exactly.
    #     expanded_state = {}
    #     expandable_keys = {
    #         "actor_mean.0.weight",
    #         "critic.0.weight",
    #     }
    #     for key, target_value in target_state.items():
    #         if key not in state_dict:
    #             raise ValueError(f"initialization is missing parameter {key}")
    #         source_value = state_dict[key]
    #         if key in expandable_keys:
    #             if (
    #                 source_value.shape[0] != target_value.shape[0]
    #                 or source_value.shape[1] != 28
    #                 or target_value.shape[1] != 31
    #             ):
    #                 raise ValueError(
    #                     f"unsupported input-layer shapes for {key}: "
    #                     f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
    #                 )
    #             expanded = torch.zeros_like(target_value)
    #             expanded[:, :28] = source_value.to(
    #                 device=expanded.device,
    #                 dtype=expanded.dtype,
    #             )
    #             expanded_state[key] = expanded
    #         elif source_value.shape == target_value.shape:
    #             expanded_state[key] = source_value
    #         else:
    #             raise ValueError(
    #                 f"unsupported parameter shape for {key}: "
    #                 f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
    #             )
    #     agent.load_state_dict(expanded_state, strict=True)
    #     source_kind += "_expanded_28_to_31"
    else:
        raise ValueError(
            "unsupported observation-size transfer: "
            f"{source_obs_dim} -> {target_obs_dim}"
        )
    return source_kind


# 5.3 随机状态的序列化与恢复
def _capture_numpy_random_state():
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": torch.as_tensor(keys.astype(np.int64)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_random_state(state):
    keys = state["keys"].cpu().numpy().astype(np.uint32)
    np.random.set_state((
        state["name"],
        keys,
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    ))


# 5.4 完整训练checkpoint的保存、恢复与定期保存边界
def save_training_checkpoint(
    path,
    agent,
    optimizer,
    *,
    global_step,
    completed_iteration,
    args,
    run_name,
    wandb_run_id=None,
    best_eval_metrics=None,
):
    """Atomically save the state needed to continue a PPO run."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )
    agent_device = next(agent.parameters()).device
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "completed_iteration": int(completed_iteration),
        "args": asdict(args),
        "run_name": str(run_name),
        "wandb_run_id": wandb_run_id,
        "best_eval_metrics": best_eval_metrics,
        "python_random_state": random.getstate(),
        "numpy_random_state": _capture_numpy_random_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all()
            if agent_device.type == "cuda"
            else None
        ),
    }
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return checkpoint_path


def load_training_checkpoint(path, agent, optimizer, device):
    """Restore Agent, optimizer, counters, configuration, and RNG state."""
    checkpoint_path = resolve_training_path(path)
    payload = _torch_load(checkpoint_path, device)
    if not _is_full_training_checkpoint(payload):
        raise ValueError(
            "--resume-from requires a versioned full training checkpoint; "
            "use --init-from for legacy .cleanrl_model weights"
        )

    agent.load_state_dict(payload["agent_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_random_state"])
    _restore_numpy_random_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    cuda_state = payload.get("cuda_random_state")
    if cuda_state is not None and torch.cuda.is_available():
        # map_location may move checkpoint RNG bytes to CUDA, but this API
        # requires CPU ByteTensors even when restoring CUDA generators.
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return payload


def next_checkpoint_step(global_step, checkpoint_interval):
    """Return the next strict checkpoint boundary, or None if disabled."""
    if checkpoint_interval <= 0:
        return None
    return (
        (int(global_step) // int(checkpoint_interval)) + 1
    ) * int(checkpoint_interval)


# 5.5 resume-from：从checkpoint恢复训练配置
RESUME_CONFIG_FIELDS = (
    "seed",
    "torch_deterministic",
    "track",
    "wandb_project_name",
    "wandb_entity",
    "save_model",
    "checkpoint_interval",
    "env_id",
    "total_timesteps",
    "learning_rate",
    "num_envs",
    "num_steps",
    "anneal_lr",
    "gamma",
    "gae_lambda",
    "num_minibatches",
    "update_epochs",
    "norm_adv",
    "clip_coef",
    "clip_vloss",
    "ent_coef",
    "initial_action_std",
    "final_action_std",
    "action_std_hold_fraction",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
    "target_mode",
    "target_pos",
    "collision_penalty",
    "dist_success_thresh",
    # "near_start_probability",
    # "near_start_from",
    "eval_interval",
    "eval_episodes",
    "eval_seed",
    "eval_max_collision_rate",
    "eval_target_success",
)


def apply_resume_configuration(args, checkpoint_payload):
    """Restore optimization-critical arguments while keeping runtime choices."""
    saved_args = checkpoint_payload["args"]
    for field_name in RESUME_CONFIG_FIELDS:
        if field_name in saved_args:
            setattr(args, field_name, saved_args[field_name])
    return args


# ==================== 6. 策略评估与最佳模型选择 ====================

def should_update_best(
    metrics,
    best_metrics,
    max_collision_rate,
):
    """Prefer held success; use simultaneous pose quality for direction tasks."""
    if metrics["collision_rate"] > max_collision_rate:
        return False
    if best_metrics is None:
        return True
    # if "pose_reached_rate" in metrics:
    #     def direction_rank(item):
    #         return (
    #             item["success_rate"],
    #             item.get("pose_reached_rate", 0.0),
    #             -item.get("mean_min_direction_angle_at_position_deg", 180.0),
    #             -item["collision_rate"],
    #             item.get("position_success_rate", 0.0),
    #             -item["mean_final_distance"],
    #         )
    #     return direction_rank(metrics) > direction_rank(best_metrics)
    candidate_rank = (
        metrics["success_rate"],
        -metrics["collision_rate"],
        -metrics["mean_final_distance"],
        # -metrics.get("mean_final_direction_angle_deg", 0.0),
    )
    best_rank = (
        best_metrics["success_rate"],
        -best_metrics["collision_rate"],
        -best_metrics["mean_final_distance"],
        # -best_metrics.get("mean_final_direction_angle_deg", 0.0),
    )
    return candidate_rank > best_rank


@torch.no_grad()
# def evaluate_policy(agent, args, device, *, near_start=False):
def evaluate_policy(agent, args, device):
    """Always evaluate full starts, never the training-only arrival wrapper."""
    kwargs = environment_kwargs(args.env_id, args, render_mode=None)
    # if args.env_id == "RM75Direction-v0":
    #     kwargs["near_start_probability"] = 1.0 if near_start else 0.0
    eval_env = gym.make(
        args.env_id,
        **kwargs,
    )
    successes = []
    collisions = []
    final_distances = []
    min_distances = []
    # final_direction_angles = []
    position_successes = []
    # angle_successes = []
    # pose_reached = []
    # min_angles_at_position = []
    episode_lengths = []
    # valid_starts = []
    was_training = agent.training
    agent.eval()
    try:
        horizon = int(getattr(eval_env.unwrapped, "horizon", 1024))
        for episode in range(args.eval_episodes):
            observation, reset_info = eval_env.reset(
                seed=args.eval_seed + episode
            )
            # valid_start = not near_start or bool(reset_info.get("near_start", False))
            # valid_starts.append(float(valid_start))
            info = reset_info
            min_distance = float(reset_info.get("dist", np.inf))
            step = 0
            for step in range(1, horizon + 1):
                # if not valid_start:
                #     # A failed near reset is a failed sample, not full-task success.
                #     step = 0
                #     break
                observation_tensor = torch.as_tensor(
                    observation[None, :],
                    dtype=torch.float32,
                    device=device,
                )
                action = torch.tanh(
                    agent.actor_mean(observation_tensor)
                )[0].cpu().numpy()
                observation, _, terminated, truncated, info = (
                    eval_env.step(action)
                )
                if info.get("dist") is not None:
                    min_distance = min(min_distance, float(info["dist"]))
                if terminated or truncated:
                    break
            successes.append(float(bool(info.get("success", False))))
            collisions.append(float(bool(info.get("collision", False))))
            final_distances.append(float(info.get("dist", np.nan)))
            min_distances.append(min_distance)
            # direction_angle = info.get("direction_angle_deg", np.nan)
            # final_direction_angles.append(
            #     np.nan if direction_angle is None else float(direction_angle)
            # )
            position_successes.append(float(bool(
                info.get(
                    "position_reached",
                    info.get("position_success", info.get("success", False)),
                )
            )))
            # angle_successes.append(float(bool(info.get("angle_reached", False))))
            # pose_reached.append(float(bool(info.get("pose_reached", False))))
            # min_angles_at_position.append(float(
            #     info.get("min_direction_angle_at_position_deg", 180.0)
            # ))
            episode_lengths.append(float(step))
    finally:
        eval_env.close()
        agent.train(was_training)

    metrics = {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_final_distance": float(np.mean(final_distances)),
        # "mean_final_direction_angle_deg": float(
        #     np.nanmean(final_direction_angles)
        #     if not np.all(np.isnan(final_direction_angles))
        #     else 0.0
        # ),
        "position_success_rate": float(np.mean(position_successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }
    if args.env_id == "RM75-v0" and args.target_mode != "none":
        entered_5cm = np.asarray(min_distances) < 0.05
        failed = np.asarray(successes) == 0.0
        metrics.update({
            "entered_5cm_rate": float(np.mean(entered_5cm)),
            "near_failure_rate": float(np.mean(entered_5cm & failed)),
            "never_entered_5cm_rate": float(np.mean(~entered_5cm & failed)),
            "mean_min_distance": float(np.mean(min_distances)),
        })
    # if args.env_id == "RM75Direction-v0":
    #     metrics.update({
    #         "angle_success_rate": float(np.mean(angle_successes)),
    #         "pose_reached_rate": float(np.mean(pose_reached)),
    #         "mean_min_direction_angle_at_position_deg": float(
    #             np.mean(min_angles_at_position)
    #         ),
    #     })
    # if near_start:
    #     metrics["valid_start_rate"] = float(np.mean(valid_starts))
    return metrics


def save_best_metrics(path, metrics):
    """Atomically save human-readable best-model evaluation metadata."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metrics_path.with_suffix(
        metrics_path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_path, metrics_path)
    return metrics_path


def evaluate_and_maybe_save_best(
    agent,
    args,
    device,
    writer,
    *,
    global_step,
    best_metrics,
    best_model_path,
    best_metrics_path,
):
    """Log fixed-seed evaluation and persist a better eligible model."""
    metrics = evaluate_policy(agent, args, device)
    metrics["global_step"] = int(global_step)
    metrics["meets_acceptance"] = bool(
        metrics["success_rate"] >= args.eval_target_success
        and metrics["collision_rate"]
        <= args.eval_max_collision_rate
    )
    log_scalars(writer, metrics, global_step, prefix="eval")

    print(
        "Fixed-seed evaluation: "
        f"success={metrics['success_rate']:.3f}, "
        f"collision={metrics['collision_rate']:.3f}, "
        f"final_dist={metrics['mean_final_distance']:.4f}, "
        # f"direction_deg={metrics['mean_final_direction_angle_deg']:.2f}, "
        f"steps={metrics['mean_episode_length']:.1f}, "
        f"acceptance={metrics['meets_acceptance']}"
    )
    # if args.env_id == "RM75Direction-v0" and args.near_start_probability > 0.0:
    #     near_metrics = evaluate_policy(agent, args, device, near_start=True)
    #     log_scalars(writer, near_metrics, global_step, prefix="eval_near")
    #     print(
    #         f"Near-start evaluation: success={near_metrics['success_rate']:.3f}, "
    #         f"pose_reached={near_metrics['pose_reached_rate']:.3f}, "
    #         f"valid_start={near_metrics['valid_start_rate']:.3f}"
    #     )
    if not should_update_best(
        metrics,
        best_metrics,
        args.eval_max_collision_rate,
    ):
        return best_metrics, False

    if args.save_model:
        torch.save(agent.state_dict(), best_model_path)
        save_best_metrics(best_metrics_path, metrics)
        print(f"new best model saved to {best_model_path}")
    return metrics, True


# ==================== 7. 训练主程序 ====================

if __name__ == "__main__":
    # ---------- 7.1 参数解析、校验与续训配置读取 ----------
    args = tyro.cli(Args)
    if args.init_from is not None and args.resume_from is not None:
        raise ValueError(
            "--init-from and --resume-from are mutually exclusive"
        )
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if args.collision_penalty < 0.0:
        raise ValueError("collision_penalty must be non-negative")
    # if not 0.0 <= args.near_start_probability <= 1.0:
    #     raise ValueError("near_start_probability must be within [0, 1]")
    # if args.near_start_probability > 0.0 and args.env_id != "RM75Direction-v0":
    #     raise ValueError("near starts require RM75Direction-v0")
    if args.eval_interval < 0:
        raise ValueError("eval_interval must be non-negative")
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if not 0.0 <= args.eval_max_collision_rate <= 1.0:
        raise ValueError(
            "eval_max_collision_rate must be within [0, 1]"
        )
    if not 0.0 <= args.eval_target_success <= 1.0:
        raise ValueError("eval_target_success must be within [0, 1]")

    resume_preview = None
    resume_path = None
    if args.resume_from is not None:
        resume_path = resolve_training_path(args.resume_from)
        resume_preview = _torch_load(resume_path, "cpu")
        if not _is_full_training_checkpoint(resume_preview):
            raise ValueError(
                "--resume-from requires checkpoint_latest.pt; "
                "use --init-from for legacy .cleanrl_model weights"
            )
        apply_resume_configuration(args, resume_preview)

    # if args.near_start_from is not None:
    #     if args.env_id != "RM75Direction-v0" or args.near_start_probability <= 0:
    #         raise ValueError("--near-start-from requires RM75Direction-v0 and a positive near-start probability")
    #     if args.target_mode == "none":
    #         raise ValueError("arrival starts require a target")
    #     args.near_start_from = str(resolve_training_path(args.near_start_from))

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if not (
        args.initial_action_std >= args.final_action_std > 0.0
    ):
        raise ValueError(
            "action std schedule requires "
            "initial_action_std >= final_action_std > 0"
        )
    if not 0.0 <= args.action_std_hold_fraction < 1.0:
        raise ValueError(
            "action_std_hold_fraction must be within [0, 1)"
        )
    if args.num_iterations < 1:
        raise ValueError(
            "total_timesteps must contain at least one complete PPO batch"
        )

    # ---------- 7.2 运行目录与日志记录器 ----------
    if resume_preview is None:
        run_name = (
            f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        )
        run_directory = PROJECT_ROOT / "runs" / run_name
        resume_global_step = 0
    else:
        if resume_path.parent.name == "checkpoints":
            run_directory = resume_path.parent.parent
            run_name = run_directory.name
        else:
            run_name = resume_preview["run_name"]
            run_directory = PROJECT_ROOT / "runs" / run_name
        resume_global_step = int(resume_preview["global_step"])

    wandb_run = None
    if args.track:
        import wandb

        wandb_kwargs = {}
        if resume_preview is not None and resume_preview.get("wandb_run_id"):
            wandb_kwargs.update({
                "id": resume_preview["wandb_run_id"],
                "resume": "allow",
            })
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            **wandb_kwargs,
        )
    writer = SummaryWriter(
        str(run_directory),
        purge_step=(resume_global_step if resume_preview is not None else None),
    )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ---------- 7.3 随机种子与训练设备 ----------
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---------- 7.4 环境、Agent与优化器创建 ----------
    # env setup
    envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args=args) for _ in range(args.num_envs)],
    autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        envs,
        initial_action_std=args.initial_action_std,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ---------- 7.5 初始化权重或恢复完整训练状态 ----------
    global_step = 0
    completed_iteration = 0
    start_iteration = 1
    if args.init_from is not None:
        source_kind = load_agent_weights(
            args.init_from,
            agent,
            device,
        )
        print(
            f"Fine-tune initialization loaded ({source_kind}): "
            f"{resolve_training_path(args.init_from)}"
        )
    elif resume_path is not None:
        resume_payload = load_training_checkpoint(
            resume_path,
            agent,
            optimizer,
            device,
        )
        global_step = int(resume_payload["global_step"])
        completed_iteration = int(
            resume_payload["completed_iteration"]
        )
        start_iteration = completed_iteration + 1
        if completed_iteration > args.num_iterations:
            raise ValueError(
                "checkpoint iteration exceeds the saved training plan"
            )
        print(
            f"Resuming run {run_name} from global_step={global_step}, "
            f"next_iteration={start_iteration}/{args.num_iterations}"
        )

    # ---------- 7.6 rollout缓存、初始观测与统计状态 ----------
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    session_start_step = global_step
    reset_seed = args.seed + completed_iteration
    next_obs, _ = envs.reset(seed=reset_seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs).to(device)

    finished_episodes = 0
    recent_successes = deque(maxlen=100)
    recent_collisions = deque(maxlen=100)
    recent_position_successes = deque(maxlen=100)
    # recent_angle_successes = deque(maxlen=100)
    # recent_pose_reached = deque(maxlen=100)
    # recent_near_starts = deque(maxlen=100)
    recent_region_successes = {
        "uniform": deque(maxlen=100),
        "fixed": deque(maxlen=100),
    }

    latest_checkpoint_path = (
        run_directory / "checkpoints" / "checkpoint_latest.pt"
    )
    best_model_path = (
        run_directory / f"{args.exp_name}.best.cleanrl_model"
    )
    best_metrics_path = run_directory / "best_metrics.json"
    best_eval_metrics = (
        resume_preview.get("best_eval_metrics")
        if resume_preview is not None
        else None
    )
    next_save_step = next_checkpoint_step(
        global_step,
        args.checkpoint_interval,
    )
    next_eval_step = next_checkpoint_step(
        global_step,
        args.eval_interval,
    )
    last_eval_step = None
    stop_requested = {"value": False}

    # ---------- 7.7 安全停止与checkpoint保存回调 ----------
    def save_latest_checkpoint():
        return save_training_checkpoint(
            latest_checkpoint_path, agent, optimizer,
            global_step=global_step,
            completed_iteration=completed_iteration,
            args=args, run_name=run_name,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
            best_eval_metrics=best_eval_metrics,
        )

    def request_graceful_stop(signum, frame):
        del signum, frame
        if stop_requested["value"]:
            return
        stop_requested["value"] = True
        print(
            "\nCtrl+C received: finishing the current PPO iteration, "
            "then saving checkpoint_latest.pt."
        )

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_graceful_stop)
    interrupted = False

    # ---------- 7.8 微调前的固定种子基线评估 ----------
    # 微调前评估初始化模型，将其作为本轮选模基线。
    if args.init_from is not None and args.eval_interval > 0:
        print("Evaluating initialization before PPO updates...")

        best_eval_metrics, _ = evaluate_and_maybe_save_best(
            agent,
            args,
            device,
            writer,
            global_step=global_step,
            best_metrics=None,
            best_model_path=best_model_path,
            best_metrics_path=best_metrics_path,
        )
        last_eval_step = global_step

    # ---------- 7.9 PPO训练循环 ----------
    for iteration in range(start_iteration, args.num_iterations + 1):
        # 7.9.1 本轮探索上限与学习率调度
        scheduled_action_std = compute_action_std(
            initial_action_std=args.initial_action_std,
            final_action_std=args.final_action_std,
            iteration=iteration,
            num_iterations=args.num_iterations,
            hold_fraction=args.action_std_hold_fraction,
        )
        agent.set_action_std_ceiling(scheduled_action_std)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 7.9.2 交互采样：观测、动作、旧概率、旧Value与奖励
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            finished_envs= np.logical_or(terminations, truncations)
            if np.any(finished_envs):
                next_obs, _ = envs.reset(
                    options={"reset_mask": finished_envs}
                )
            next_done = finished_envs
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done.astype(np.float32), dtype=torch.float32, device=device)

            # 7.9.3 已完成回合的指标统计；方向指标继续保持注释
            # Gymnasium 1.3.x puts completed episode data directly in infos.
            if "episode" in infos:
                episode_mask = infos.get(
                    "_episode",
                    np.ones(args.num_envs, dtype=bool),
                )

                for env_idx in np.flatnonzero(episode_mask):
                    episode_return = float(
                        infos["episode"]["r"][env_idx]
                    )
                    episode_length = int(
                        infos["episode"]["l"][env_idx]
                    )

                    success = float(
                        infos.get(
                            "success",
                            np.zeros(args.num_envs),
                        )[env_idx]
                    )

                    collision = float(
                        infos.get(
                            "collision",
                            np.zeros(args.num_envs),
                        )[env_idx]
                    )

                    timeout = float(
                        infos.get(
                            "timeout",
                            np.zeros(args.num_envs),
                        )[env_idx]
                    )

                    final_dist = float(
                        infos.get(
                            "dist",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    position_success = float(
                        infos.get(
                            "position_reached",
                            infos.get(
                                "position_success",
                                infos.get(
                                    "success",
                                    np.zeros(args.num_envs),
                                ),
                            ),
                        )[env_idx]
                    )
                    # angle_success = float(
                    #     infos.get("angle_reached", np.zeros(args.num_envs))[env_idx]
                    # )
                    # pose_reached = float(
                    #     infos.get("pose_reached", np.zeros(args.num_envs))[env_idx]
                    # )
                    # direction_angle_deg = infos.get(
                    #     "direction_angle_deg",
                    #     np.full(args.num_envs, np.nan),
                    # )[env_idx]
                    # direction_angle_deg = (
                    #     np.nan
                    #     if direction_angle_deg is None
                    #     else float(direction_angle_deg)
                    # )
                    target_region = str(
                        infos.get(
                            "target_region",
                            np.full(
                                args.num_envs,
                                "unknown",
                                dtype=object,
                            ),
                        )[env_idx]
                    )

                    finished_episodes += 1
                    recent_successes.append(success)
                    recent_collisions.append(collision)
                    recent_position_successes.append(position_success)
                    # recent_angle_successes.append(angle_success)
                    # recent_pose_reached.append(pose_reached)
                    # recent_near_starts.append(float(
                    #     infos.get("near_start", np.zeros(args.num_envs))[env_idx]
                    # ))
                    if target_region in recent_region_successes:
                        recent_region_successes[target_region].append(
                            success
                        )

                    if finished_episodes % 10 == 0 or success:
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={episode_return:.3f}, "
                            f"episodic_length={episode_length}, "
                            f"success={bool(success)}, "
                            f"collision={bool(collision)}, "
                            f"timeout={bool(timeout)}, "
                            f"target_region={target_region}, "
                            f"final_dist={final_dist:.4f}, "
                            # f"direction_deg={direction_angle_deg:.2f}"
                        )

                    episode_metrics = {
                        "episodic_return": episode_return,
                        "episodic_length": episode_length,
                        "episodic_success": success,
                        "collision": collision,
                        "timeout": timeout,
                        "final_distance": final_dist,
                        "position_success": position_success,
                    }
                    # if np.isfinite(direction_angle_deg):
                    #     episode_metrics["final_direction_angle_deg"] = direction_angle_deg
                    # if args.env_id == "RM75Direction-v0":
                    #     episode_metrics.update({
                    #         "arrival_prefix_steps": infos.get("arrival_prefix_steps", np.zeros(args.num_envs))[env_idx],
                    #         "angle_success": angle_success,
                    #         "pose_reached": pose_reached,
                    #         "min_direction_angle_at_position_deg": infos["min_direction_angle_at_position_deg"][env_idx],
                    #     })
                    log_scalars(writer, episode_metrics, global_step)

        # 7.9.4 GAE优势与Critic回报目标；不构建梯度图
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 7.9.5 将时间维与环境维展平，形成训练批次
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 7.9.6 小批次Actor/Critic损失、反向传播与参数更新
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 7.9.7 损失、学习率、探索、吞吐量与最近回合指标
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log_scalars(writer, {
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
        }, global_step, prefix="losses")
        action_saturation_rate = float(
            (actions.abs() > 0.98).float().mean().item()
        )
        session_steps = global_step - session_start_step
        sps = int(session_steps / max(time.time() - start_time, 1e-6))
        print("SPS:", sps)
        log_scalars(writer, {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "action_std_ceiling": scheduled_action_std,
            "action_saturation_rate": action_saturation_rate,
            "SPS": sps,
        }, global_step)
        if recent_successes:
            success_rate = float(np.mean(recent_successes))
            collision_rate = float(np.mean(recent_collisions))
            print(
                "Recent-100 success/collision rate: "
                f"{success_rate:.3f}/{collision_rate:.3f}"
            )
            recent_metrics = {
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "position_success_rate": np.mean(recent_position_successes),
            }
            # if args.env_id == "RM75Direction-v0":
            #     recent_metrics.update({
            #         "near_start_rate": np.mean(recent_near_starts),
            #         "angle_success_rate": np.mean(recent_angle_successes),
            #         "pose_reached_rate": np.mean(recent_pose_reached),
            #     })
            log_scalars(writer, recent_metrics, global_step)

        for region, outcomes in recent_region_successes.items():
            if outcomes:
                writer.add_scalar(
                    f"charts/region_success_rate/{region}",
                    float(np.mean(outcomes)),
                    global_step,
                )

        # 7.9.8 固定种子评估与最佳模型更新
        completed_iteration = iteration
        if (
            not stop_requested["value"]
            and next_eval_step is not None
            and global_step >= next_eval_step
        ):
            best_eval_metrics, _ = evaluate_and_maybe_save_best(
                agent,
                args,
                device,
                writer,
                global_step=global_step,
                best_metrics=best_eval_metrics,
                best_model_path=best_model_path,
                best_metrics_path=best_metrics_path,
            )
            last_eval_step = global_step
            next_eval_step = next_checkpoint_step(
                global_step,
                args.eval_interval,
            )

        # 7.9.9 定期保存可续训的完整checkpoint
        if (
            args.save_model
            and next_save_step is not None
            and global_step >= next_save_step
        ):
            save_latest_checkpoint()
            print(f"checkpoint saved to {latest_checkpoint_path}")
            next_save_step = next_checkpoint_step(
                global_step,
                args.checkpoint_interval,
            )

        # 7.9.10 完成本轮后响应安全停止
        if stop_requested["value"]:
            interrupted = True
            if args.save_model:
                save_latest_checkpoint()
                print(
                    "graceful-stop checkpoint saved to "
                    f"{latest_checkpoint_path}"
                )
            break

    # ---------- 7.10 最终评估、模型保存与资源关闭 ----------
    signal.signal(signal.SIGINT, previous_sigint_handler)

    if (
        not interrupted
        and args.eval_interval > 0
        and last_eval_step != global_step
    ):
        best_eval_metrics, _ = evaluate_and_maybe_save_best(
            agent,
            args,
            device,
            writer,
            global_step=global_step,
            best_metrics=best_eval_metrics,
            best_model_path=best_model_path,
            best_metrics_path=best_metrics_path,
        )
        last_eval_step = global_step

    if interrupted:
        resume_device_flag = " --no-cuda" if not args.cuda else ""
        print(
            "Training stopped safely. Resume with:\n"
            ".venv/bin/python Algorithms/ppo_continuous_action.py "
            f"--resume-from {latest_checkpoint_path}"
            f"{resume_device_flag}"
        )
    elif args.save_model:
        # Persist the final effective exploration scale through the existing
        # actor_logstd parameter without adding a new checkpoint key.
        agent.set_action_std_ceiling(args.final_action_std)
        model_path = run_directory / f"{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        save_latest_checkpoint()
        print(f"final checkpoint saved to {latest_checkpoint_path}")

    envs.close()
    writer.close()
