# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
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

if "RM75Approach-v0" not in gym.registry:
    gym.register(
        id="RM75Approach-v0",
        entry_point=RM75P2PEnv,
        kwargs={
            "render_mode": None,
            "target_mode": "random",
            "approach_task": True,
        },
        max_episode_steps=None,
    )

if "RM75Pose-v0" not in gym.registry:
    gym.register(
        id="RM75Pose-v0",
        entry_point=RM75P2PEnv,
        kwargs={
            "render_mode": None,
            "target_mode": "random",
            "approach_task": True,
            "pose_task": True,
            "target_direction_mode": "random_cone",
            "target_tilt_max_deg": 30.0,
            "require_pose_feasible": True,
            "pose_feasibility_attempts": 50,
            "orient_success_threshold_deg": 20.0,
        },
        max_episode_steps=None,
    )

@dataclass
class Args:
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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    init_from: Optional[str] = None
    """agent weights used to start a new fine-tuning run"""
    resume_from: Optional[str] = None
    """full checkpoint used to continue an interrupted run"""
    additional_timesteps: int = 0
    """extra environment steps to add to a full checkpoint, in complete PPO batches"""
    checkpoint_interval: int = 100000
    """environment steps between atomic full checkpoints; 0 disables periodic saves"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "RM75-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    final_learning_rate: float = 0.0
    """final learning rate used when anneal_lr is enabled"""
    num_envs: int = 1
    """the number of parallel game environments"""
    vector_env_mode: str = "sync"
    """vector backend: sync runs serially; async uses spawned worker processes"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
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
    network_profile: str = "auto"
    """policy network: auto, legacy, or pose"""

    target_mode: str = "random"
    """goal sampling mode passed to UR5 environment"""
    target_pos: Optional[Tuple[float, float, float]] = None
    """explicit goal position when target_mode=='fixed'"""
    target_sampling: str = "uniform"
    """random-goal sampler: uniform or hard_mixture"""
    collision_penalty: float = 75.0
    """RM75 collision reward penalty; collision still terminates the episode"""
    target_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    """desired tool +Z direction for RM75Approach-v0"""
    target_direction_mode: str = "fixed"
    """RM75 approach direction mode: fixed or random_cone"""
    target_tilt_min_deg: float = 0.0
    """minimum downward-cone tilt sampled at each reset"""
    target_tilt_max_deg: float = 15.0
    """maximum downward-cone tilt sampled at each reset"""
    target_azimuth_min_deg: float = -180.0
    """minimum azimuth sampled at each reset"""
    target_azimuth_max_deg: float = 180.0
    """maximum azimuth sampled at each reset"""
    approach_position_box: Optional[
        Tuple[float, float, float, float]
    ] = None
    """optional xmin xmax ymin ymax box for the approach task"""
    require_pose_feasible: bool = False
    """reject approach position/direction pairs that fail pose IK"""
    pose_feasibility_attempts: int = 20
    """maximum sampled pairs checked per reset"""
    approach_curriculum: bool = False
    """promote random-cone approach stages only after demonstrated mastery"""
    curriculum_start_stage: int = 1
    """initial approach curriculum stage for a fresh or init-from run"""
    curriculum_stage1_tilt_deg: float = 15.0
    """stage-1 maximum sampled approach tilt"""
    curriculum_stage2_tilt_deg: float = 15.0
    """stage-2 maximum sampled approach tilt"""
    curriculum_stage3_tilt_deg: float = 20.0
    """stage-3 maximum sampled approach tilt"""
    curriculum_stage4_tilt_deg: float = 25.0
    """stage-4 maximum sampled approach tilt"""
    curriculum_stage5_tilt_deg: float = 30.0
    """stage-5 maximum sampled approach tilt"""
    curriculum_mastery_window: int = 200
    """completed active-stage episodes considered for promotion"""
    curriculum_promotion_success_rate: float = 0.75
    """stage-1/2 active success rate required to promote"""
    curriculum_stage3_promotion_success_rate: float = 0.50
    """stage-3 active success rate required to promote"""
    curriculum_stage4_promotion_success_rate: float = 0.60
    """stage-4 active success rate required to promote"""
    curriculum_stage5_promotion_success_rate: float = 0.60
    """stage-5 active success rate required to promote"""
    curriculum_min_stage_steps: int = 100000
    """minimum environment steps spent in a stage before promotion"""
    curriculum_rollback_window: int = 100
    """completed episodes considered for post-promotion rollback"""
    curriculum_rollback_success_rate: float = 0.10
    """active-stage rate at or below which a protected stage rolls back"""
    curriculum_rollback_grace_steps: int = 50000
    """minimum stage steps before rollback can be triggered"""
    curriculum_stage_action_std_decay_steps: int = 300000
    """stage-local steps over which action exploration decays"""
    curriculum_stage1_dist_success_thresh: float = 0.12
    """stage-1 active position tolerance in metres"""
    curriculum_stage1_orient_success_deg: float = 40.0
    """stage-1 active direction tolerance in degrees"""
    curriculum_stage2_dist_success_thresh: float = 0.10
    """stage-2 active position tolerance in metres"""
    curriculum_stage2_orient_success_deg: float = 35.0
    """stage-2 active direction tolerance in degrees"""
    curriculum_stage3_dist_success_thresh: float = 0.085
    """stage-3 active position tolerance in metres"""
    curriculum_stage3_orient_success_deg: float = 30.0
    """stage-3 active direction tolerance in degrees"""
    curriculum_stage4_dist_success_thresh: float = 0.070
    """stage-4 active position tolerance in metres"""
    curriculum_stage4_orient_success_deg: float = 25.0
    """stage-4 active direction tolerance in degrees"""
    curriculum_stage5_dist_success_thresh: float = 0.060
    """stage-5 active position tolerance in metres"""
    curriculum_stage5_orient_success_deg: float = 22.0
    """stage-5 active direction tolerance in degrees"""
    curriculum_stage1_position_box: Tuple[float, float, float, float] = (
        -0.42, -0.36, -0.10, 0.10
    )
    """stage-1 xmin xmax ymin ymax goal sampling box"""
    curriculum_stage2_position_box: Tuple[float, float, float, float] = (
        -0.42, -0.36, -0.10, 0.10
    )
    """stage-2 xmin xmax ymin ymax goal sampling box"""
    curriculum_stage3_position_box: Tuple[float, float, float, float] = (
        -0.43, -0.36, -0.12, 0.12
    )
    """stage-3 xmin xmax ymin ymax goal sampling box"""
    curriculum_stage4_position_box: Tuple[float, float, float, float] = (
        -0.44, -0.35, -0.14, 0.14
    )
    """stage-4 xmin xmax ymin ymax goal sampling box"""
    curriculum_stage5_position_box: Tuple[float, float, float, float] = (
        -0.45, -0.35, -0.16, 0.16
    )
    """stage-5 xmin xmax ymin ymax goal sampling box"""
    orient_success_threshold_deg: float = 15.0
    """largest direction error counted as approach-task success"""
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def environment_kwargs(env_id, args: Args, *, render_mode=None):
    """Build environment kwargs without passing RM75-only fields to UR5."""
    kwargs = {
        "render_mode": render_mode,
        "target_mode": args.target_mode,
        "target_pos": args.target_pos,
    }
    if env_id in {"RM75-v0", "RM75Approach-v0", "RM75Pose-v0"}:
        kwargs.update({
            "target_sampling": args.target_sampling,
            "collision_penalty": args.collision_penalty,
        })
    if env_id in {"RM75Approach-v0", "RM75Pose-v0"}:
        kwargs.update({
            "approach_task": True,
            "pose_task": env_id == "RM75Pose-v0",
            "target_direction": args.target_direction,
            "target_direction_mode": args.target_direction_mode,
            "target_tilt_min_deg": args.target_tilt_min_deg,
            "target_tilt_max_deg": args.target_tilt_max_deg,
            "target_azimuth_min_deg": args.target_azimuth_min_deg,
            "target_azimuth_max_deg": args.target_azimuth_max_deg,
            "approach_position_box": args.approach_position_box,
            "require_pose_feasible": args.require_pose_feasible,
            "pose_feasibility_attempts": (
                args.pose_feasibility_attempts
            ),
            "orient_success_threshold_deg": (
                args.orient_success_threshold_deg
            ),
        })
    return kwargs


def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
    gamma,
    *,
    args: Args,
):
    def thunk():
        env = gym.make(
            env_id,
            **environment_kwargs(env_id, args, render_mode=None),
        )
        if capture_video and idx == 0 and "rgb_array" in getattr(env, "metadata", {}).get("render_modes", []):
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def _sigint_ignoring_async_worker(*worker_args):
    """Let only the parent process coordinate graceful interruption."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from gymnasium.vector.async_vector_env import _async_worker

    return _async_worker(*worker_args)


def make_vector_env(args: Args, run_name: str):
    """Create the selected vector backend without changing environment APIs."""
    env_fns = [
        make_env(
            args.env_id,
            index,
            args.capture_video,
            run_name,
            args.gamma,
            args=args,
        )
        for index in range(args.num_envs)
    ]
    if args.vector_env_mode == "async":
        # Spawn avoids forking a parent process after CUDA has initialized.
        return gym.vector.AsyncVectorEnv(
            env_fns,
            context="spawn",
            worker=_sigint_ignoring_async_worker,
        )
    if args.vector_env_mode == "sync":
        return gym.vector.SyncVectorEnv(env_fns)
    raise ValueError("vector_env_mode must be sync or async")


def accumulate_reward_components(
    infos,
    num_envs,
    component_sums,
    component_counts,
):
    """Accumulate vector reward diagnostics outside the PPO buffer scope."""
    components = infos.get("reward_components", {})
    for name, component_values in components.items():
        if name.startswith("_"):
            continue
        mask = np.asarray(
            components.get(f"_{name}", np.ones(num_envs)),
            dtype=bool,
        )
        valid_values = np.asarray(
            component_values,
            dtype=np.float64,
        )[mask]
        component_sums[name] = (
            component_sums.get(name, 0.0)
            + float(valid_values.sum())
        )
        component_counts[name] = (
            component_counts.get(name, 0)
            + int(valid_values.size)
        )


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


def compute_stage_action_std(
    initial_action_std,
    final_action_std,
    stage_steps,
    decay_steps,
    hold_fraction,
):
    """Hold and decay exploration using steps since the last stage change."""
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive")
    if stage_steps < 0:
        raise ValueError("stage_steps must be non-negative")
    if not 0.0 <= hold_fraction < 1.0:
        raise ValueError("hold_fraction must be within [0, 1)")
    if not initial_action_std >= final_action_std > 0.0:
        raise ValueError(
            "action std requires initial >= final > 0"
        )

    progress = float(np.clip(stage_steps / decay_steps, 0.0, 1.0))
    if progress <= hold_fraction:
        return float(initial_action_std)
    decay_progress = (
        (progress - hold_fraction) / (1.0 - hold_fraction)
    )
    return float(
        initial_action_std
        + decay_progress * (final_action_std - initial_action_std)
    )


def compute_learning_rate(
    initial_learning_rate,
    final_learning_rate,
    elapsed_steps,
    total_steps,
):
    """Linearly decay learning rate over the complete training horizon."""
    if total_steps <= 0:
        raise ValueError("learning-rate total_steps must be positive")
    if elapsed_steps < 0:
        raise ValueError("learning-rate elapsed_steps must be non-negative")
    if not (
        initial_learning_rate > 0.0
        and 0.0 <= final_learning_rate <= initial_learning_rate
    ):
        raise ValueError(
            "learning rates require initial > 0 and 0 <= final <= initial"
        )
    progress = float(np.clip(elapsed_steps / total_steps, 0.0, 1.0))
    return float(
        initial_learning_rate
        + progress * (final_learning_rate - initial_learning_rate)
    )


def curriculum_promotion_threshold(args: Args, stage: int) -> float:
    """Return the mastery threshold appropriate for one curriculum stage."""
    if stage in {1, 2}:
        return float(args.curriculum_promotion_success_rate)
    if stage == 3:
        return float(args.curriculum_stage3_promotion_success_rate)
    if stage == 4:
        return float(args.curriculum_stage4_promotion_success_rate)
    if stage == 5:
        return float(args.curriculum_stage5_promotion_success_rate)
    if stage == 6:
        return 1.0
    raise ValueError("curriculum stage must be within [1, 6]")


def initial_curriculum_stage(args: Args, saved_state=None) -> int:
    """Choose a reviewed start stage unless an exact resume overrides it."""
    stage = (
        saved_state.get("stage", args.curriculum_start_stage)
        if saved_state is not None
        else args.curriculum_start_stage
    )
    stage = int(stage)
    if stage not in {1, 2, 3, 4, 5, 6}:
        raise ValueError("curriculum_start_stage must be within [1, 6]")
    return stage


def initial_curriculum_rollback_armed(
    stage: int,
    saved_state=None,
) -> bool:
    """Arm fresh high-stage starts; exact resumes restore saved protection."""
    if saved_state is not None:
        return bool(saved_state.get("rollback_armed", False))
    return int(stage) > 1


def curriculum_stage_settings(args: Args, stage: int) -> dict:
    """Return one self-contained active distribution for an approach stage."""
    if stage == 1:
        return {
            "stage": 1,
            "tilt_max_deg": float(args.curriculum_stage1_tilt_deg),
            "success_distance_thresh": float(
                args.curriculum_stage1_dist_success_thresh
            ),
            "success_angle_deg": float(
                args.curriculum_stage1_orient_success_deg
            ),
            "position_box": tuple(args.curriculum_stage1_position_box),
        }
    if stage == 2:
        return {
            "stage": 2,
            "tilt_max_deg": float(args.curriculum_stage2_tilt_deg),
            "success_distance_thresh": float(
                args.curriculum_stage2_dist_success_thresh
            ),
            "success_angle_deg": float(
                args.curriculum_stage2_orient_success_deg
            ),
            "position_box": tuple(args.curriculum_stage2_position_box),
        }
    if stage == 3:
        return {
            "stage": 3,
            "tilt_max_deg": float(args.curriculum_stage3_tilt_deg),
            "success_distance_thresh": float(
                args.curriculum_stage3_dist_success_thresh
            ),
            "success_angle_deg": float(
                args.curriculum_stage3_orient_success_deg
            ),
            "position_box": tuple(args.curriculum_stage3_position_box),
        }
    if stage == 4:
        return {
            "stage": 4,
            "tilt_max_deg": float(args.curriculum_stage4_tilt_deg),
            "success_distance_thresh": float(
                args.curriculum_stage4_dist_success_thresh
            ),
            "success_angle_deg": float(
                args.curriculum_stage4_orient_success_deg
            ),
            "position_box": tuple(args.curriculum_stage4_position_box),
        }
    if stage == 5:
        return {
            "stage": 5,
            "tilt_max_deg": float(args.curriculum_stage5_tilt_deg),
            "success_distance_thresh": float(
                args.curriculum_stage5_dist_success_thresh
            ),
            "success_angle_deg": float(
                args.curriculum_stage5_orient_success_deg
            ),
            "position_box": tuple(args.curriculum_stage5_position_box),
        }
    if stage == 6:
        return {
            "stage": 6,
            "tilt_max_deg": float(args.target_tilt_max_deg),
            "success_distance_thresh": 0.05,
            "success_angle_deg": float(args.orient_success_threshold_deg),
            "position_box": (
                tuple(args.approach_position_box)
                if args.approach_position_box is not None
                else (-0.45, -0.35, -0.18, 0.18)
            ),
        }
    raise ValueError("curriculum stage must be within [1, 6]")


def should_promote_curriculum(
    recent_successes,
    promotion_success_rate: float,
    mastery_window: int,
    stage_steps: int,
    minimum_stage_steps: int,
) -> bool:
    """Require both minimum stage residence and a full mastery window."""
    if (
        stage_steps < minimum_stage_steps
        or len(recent_successes) < mastery_window
    ):
        return False
    return bool(
        float(np.mean(list(recent_successes)[-mastery_window:]))
        >= promotion_success_rate
    )


def should_demote_curriculum(
    recent_successes,
    rollback_success_rate: float,
    rollback_window: int,
    stage_steps: int,
    rollback_grace_steps: int,
) -> bool:
    """Detect a sustained collapse soon after entering a harder stage."""
    if (
        stage_steps < rollback_grace_steps
        or len(recent_successes) < rollback_window
    ):
        return False
    return bool(
        float(np.mean(list(recent_successes)[-rollback_window:]))
        <= rollback_success_rate
    )


def apply_approach_curriculum(envs, args: Args, *, stage: int) -> dict:
    """Update training envs; evaluation envs retain the final task settings."""
    settings = curriculum_stage_settings(args, stage)
    # ``call`` works for both SyncVectorEnv and AsyncVectorEnv and therefore
    # keeps curriculum state inside each worker process.
    if args.env_id == "RM75Approach-v0":
        envs.call("set_approach_curriculum", **settings)
    return settings


def resolve_network_profile(profile, observation_dim):
    """Resolve one network topology without changing legacy weights."""
    normalized = str(profile).lower().strip()
    if normalized == "auto":
        return "pose" if int(observation_dim) == 20 else "legacy"
    if normalized not in {"legacy", "pose"}:
        raise ValueError("network_profile must be auto, legacy, or pose")
    return normalized


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        initial_action_std=0.30,
        network_profile="auto",
    ):
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
        observation_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.network_profile = resolve_network_profile(
            network_profile,
            observation_dim,
        )
        if self.network_profile == "pose":
            self.critic = nn.Sequential(
                layer_init(nn.Linear(observation_dim, 256)),
                nn.ELU(),
                layer_init(nn.Linear(256, 128)),
                nn.ELU(),
                layer_init(nn.Linear(128, 64)),
                nn.ELU(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(observation_dim, 256)),
                nn.ELU(),
                layer_init(nn.Linear(256, 128)),
                nn.ELU(),
                layer_init(nn.Linear(128, 64)),
                nn.ELU(),
                layer_init(nn.Linear(64, action_dim), std=0.01),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(observation_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(observation_dim, 64)),
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

    def reset_action_std(self, action_std):
        """Reset learned exploration and its ceiling to one safe value."""
        if not np.isfinite(action_std) or action_std <= 0.0:
            raise ValueError("action_std must be finite and positive")
        logstd = float(np.log(action_std))
        with torch.no_grad():
            self.action_logstd_ceiling.fill_(logstd)
            self.actor_logstd.fill_(logstd)

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

    def effective_action_std_stats(self):
        """Return mean/min/max standard deviation actually used by PPO."""
        effective_logstd = torch.minimum(
            self.actor_logstd,
            self.action_logstd_ceiling,
        )
        effective_std = torch.exp(torch.clamp(
            effective_logstd,
            min=float(np.log(0.05)),
        ))
        return (
            float(effective_std.mean().detach().cpu().item()),
            float(effective_std.min().detach().cpu().item()),
            float(effective_std.max().detach().cpu().item()),
        )

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


def load_agent_weights(path, agent, device):
    """Load an exact legacy weight set or Agent part of a checkpoint."""
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
    agent.load_state_dict(state_dict, strict=True)
    return source_kind


def save_agent_weights(path, agent):
    """Export agent weights without mutating learned exploration state."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    return model_path


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


def normalize_cuda_rng_states(states):
    """Convert saved CUDA RNG states to CPU-contiguous uint8 tensors.

    Checkpoints loaded with ``map_location='cuda'`` contain CUDA tensors,
    while ``torch.cuda.set_rng_state_all`` expects CPU ``ByteTensor`` values.
    Normalizing here keeps old GPU-created checkpoints portable across PyTorch
    versions and CPU/GPU map locations.
    """
    if states is None:
        return None
    if isinstance(states, torch.Tensor):
        states = [states]
    normalized = []
    for state in states:
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        normalized.append(
            state.detach().to(
                device="cpu",
                dtype=torch.uint8,
            ).contiguous()
        )
    return normalized


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
    curriculum_state=None,
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
        "curriculum_state": curriculum_state,
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
    cuda_state = normalize_cuda_rng_states(
        payload.get("cuda_random_state")
    )
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return payload


def next_checkpoint_step(global_step, checkpoint_interval):
    """Return the next strict checkpoint boundary, or None if disabled."""
    if checkpoint_interval <= 0:
        return None
    return (
        (int(global_step) // int(checkpoint_interval)) + 1
    ) * int(checkpoint_interval)


def should_update_best(
    metrics,
    best_metrics,
    max_collision_rate,
):
    """Rank point and approach policies using task-relevant quality metrics."""
    if metrics["collision_rate"] > max_collision_rate:
        return False
    if best_metrics is None:
        return True

    def rank(evaluation):
        stage_mastered = int(evaluation.get("stage_mastered", True))
        mastered_stage = (
            int(evaluation.get("curriculum_stage", 0))
            if stage_mastered
            else 0
        )
        common = (
            stage_mastered,
            mastered_stage,
            evaluation.get(
                "strict_success_rate",
                evaluation["success_rate"],
            ),
            evaluation["success_rate"],
            -evaluation["collision_rate"],
        )
        if "mean_min_pose_error" in metrics:
            return common + (
                -evaluation.get("mean_min_pose_error", np.inf),
                evaluation.get("ever_position_success_rate", 0.0),
                -evaluation["mean_final_distance"],
                -evaluation.get("mean_final_angle_error_deg", np.inf),
            )
        return common + (
            -evaluation["mean_final_distance"],
            -evaluation.get("mean_final_angle_error_deg", np.inf),
        )

    candidate_rank = rank(metrics)
    best_rank = rank(best_metrics)
    return candidate_rank > best_rank


@torch.no_grad()
def evaluate_policy(agent, args, device, curriculum_stage=None):
    """Evaluate deterministic actions on fixed seeds and one active stage."""
    eval_env = gym.make(
        args.env_id,
        **environment_kwargs(args.env_id, args, render_mode=None),
    )
    if curriculum_stage is not None:
        eval_env.unwrapped.set_approach_curriculum(
            **curriculum_stage_settings(args, curriculum_stage)
        )
    successes = []
    strict_successes = []
    collisions = []
    timeouts = []
    final_distances = []
    episode_lengths = []
    direction_successes = []
    final_angle_errors_deg = []
    min_distances = []
    min_angle_errors_deg = []
    min_pose_errors = []
    ever_position_successes = []
    ever_direction_successes = []
    was_training = agent.training
    agent.eval()
    try:
        horizon = int(getattr(eval_env.unwrapped, "horizon", 1024))
        for episode in range(args.eval_episodes):
            observation, _ = eval_env.reset(
                seed=args.eval_seed + episode
            )
            info = {}
            step = 0
            for step in range(1, horizon + 1):
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
                if terminated or truncated:
                    break
            successes.append(float(bool(info.get(
                "ever_active_success",
                info.get("success", False),
            ))))
            if "strict_success" in info:
                strict_successes.append(
                    float(bool(info["strict_success"]))
                )
            collisions.append(float(bool(info.get("collision", False))))
            timeouts.append(float(bool(info.get("timeout", False))))
            final_distances.append(float(info.get("dist", np.nan)))
            episode_lengths.append(float(step))
            if "direction_success" in info:
                direction_successes.append(
                    float(bool(info["direction_success"]))
                )
            if "angle_error_deg" in info:
                final_angle_errors_deg.append(
                    float(info["angle_error_deg"])
                )
                min_distances.append(float(info["min_distance"]))
                min_angle_errors_deg.append(
                    float(info["min_angle_error_deg"])
                )
                min_pose_errors.append(float(info["min_pose_error"]))
                ever_position_successes.append(
                    float(bool(info["ever_position_success"]))
                )
                ever_direction_successes.append(
                    float(bool(info["ever_direction_success"]))
                )
    finally:
        eval_env.close()
        agent.train(was_training)

    metrics = {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }
    metrics["success_rate_standard_error"] = float(np.sqrt(
        metrics["success_rate"]
        * (1.0 - metrics["success_rate"])
        / max(len(successes), 1)
    ))
    if direction_successes:
        metrics["strict_success_rate"] = float(
            np.mean(strict_successes)
        )
        metrics["strict_success_rate_standard_error"] = float(
            np.sqrt(
                metrics["strict_success_rate"]
                * (1.0 - metrics["strict_success_rate"])
                / max(len(strict_successes), 1)
            )
        )
        if curriculum_stage is not None:
            metrics["curriculum_stage"] = int(curriculum_stage)
        metrics["direction_success_rate"] = float(
            np.mean(direction_successes)
        )
        metrics["mean_final_angle_error_deg"] = float(
            np.mean(final_angle_errors_deg)
        )
        metrics["mean_min_distance"] = float(np.mean(min_distances))
        metrics["mean_min_angle_error_deg"] = float(
            np.mean(min_angle_errors_deg)
        )
        metrics["mean_min_pose_error"] = float(
            np.mean(min_pose_errors)
        )
        metrics["ever_position_success_rate"] = float(
            np.mean(ever_position_successes)
        )
        metrics["ever_direction_success_rate"] = float(
            np.mean(ever_direction_successes)
        )
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
    curriculum_stage=None,
):
    """Log fixed-seed evaluation and persist a better eligible model."""
    metrics = evaluate_policy(
        agent,
        args,
        device,
        curriculum_stage=curriculum_stage,
    )
    metrics["global_step"] = int(global_step)
    if curriculum_stage is not None:
        stage_mastery_target = (
            args.eval_target_success
            if curriculum_stage == 6
            else curriculum_promotion_threshold(
                args, curriculum_stage
            )
        )
        metrics["stage_mastery_target"] = float(
            stage_mastery_target
        )
        metrics["stage_mastered"] = bool(
            metrics["success_rate"] >= stage_mastery_target
        )
    acceptance_success_rate = metrics.get(
        "strict_success_rate",
        metrics["success_rate"],
    )
    final_stage = (
        "strict_success_rate" not in metrics
        or curriculum_stage is None
        or curriculum_stage == 6
    )
    metrics["meets_acceptance"] = bool(
        final_stage
        and acceptance_success_rate >= args.eval_target_success
        and metrics["collision_rate"]
        <= args.eval_max_collision_rate
    )
    for metric_name, value in metrics.items():
        writer.add_scalar(
            f"eval/{metric_name}",
            float(value),
            global_step,
        )

    print(
        "Fixed-seed evaluation: "
        f"success={metrics['success_rate']:.3f}, "
        f"collision={metrics['collision_rate']:.3f}, "
        f"final_dist={metrics['mean_final_distance']:.4f}, "
        f"steps={metrics['mean_episode_length']:.1f}, "
        + (
            "angle="
            f"{metrics['mean_final_angle_error_deg']:.2f}deg, "
            f"direction_success={metrics['direction_success_rate']:.3f}, "
            f"strict_success={metrics['strict_success_rate']:.3f}, "
            if "mean_final_angle_error_deg" in metrics
            else ""
        )
        + f"acceptance={metrics['meets_acceptance']}"
    )
    if not should_update_best(
        metrics,
        best_metrics,
        args.eval_max_collision_rate,
    ):
        return best_metrics, False, metrics

    if args.save_model:
        save_agent_weights(best_model_path, agent)
        save_best_metrics(best_metrics_path, metrics)
        print(f"new best model saved to {best_model_path}")
    return metrics, True, metrics


RESUME_CONFIG_FIELDS = (
    "exp_name",
    "seed",
    "torch_deterministic",
    "track",
    "wandb_project_name",
    "wandb_entity",
    "capture_video",
    "save_model",
    "checkpoint_interval",
    "env_id",
    "total_timesteps",
    "learning_rate",
    "final_learning_rate",
    "num_envs",
    "vector_env_mode",
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
    "network_profile",
    "target_mode",
    "target_pos",
    "target_sampling",
    "collision_penalty",
    "target_direction",
    "target_direction_mode",
    "target_tilt_min_deg",
    "target_tilt_max_deg",
    "target_azimuth_min_deg",
    "target_azimuth_max_deg",
    "approach_position_box",
    "require_pose_feasible",
    "pose_feasibility_attempts",
    "approach_curriculum",
    "curriculum_start_stage",
    "curriculum_stage1_tilt_deg",
    "curriculum_stage2_tilt_deg",
    "curriculum_stage3_tilt_deg",
    "curriculum_stage4_tilt_deg",
    "curriculum_stage5_tilt_deg",
    "curriculum_mastery_window",
    "curriculum_promotion_success_rate",
    "curriculum_stage3_promotion_success_rate",
    "curriculum_stage4_promotion_success_rate",
    "curriculum_stage5_promotion_success_rate",
    "curriculum_min_stage_steps",
    "curriculum_rollback_window",
    "curriculum_rollback_success_rate",
    "curriculum_rollback_grace_steps",
    "curriculum_stage_action_std_decay_steps",
    "curriculum_stage1_dist_success_thresh",
    "curriculum_stage1_orient_success_deg",
    "curriculum_stage2_dist_success_thresh",
    "curriculum_stage2_orient_success_deg",
    "curriculum_stage3_dist_success_thresh",
    "curriculum_stage3_orient_success_deg",
    "curriculum_stage4_dist_success_thresh",
    "curriculum_stage4_orient_success_deg",
    "curriculum_stage5_dist_success_thresh",
    "curriculum_stage5_orient_success_deg",
    "curriculum_stage1_position_box",
    "curriculum_stage2_position_box",
    "curriculum_stage3_position_box",
    "curriculum_stage4_position_box",
    "curriculum_stage5_position_box",
    "orient_success_threshold_deg",
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


def configure_training_horizon(args, resume_payload=None):
    """Compute PPO batch sizes and optionally extend a resumed checkpoint."""
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    if args.additional_timesteps < 0:
        raise ValueError("additional_timesteps must be non-negative")
    if args.additional_timesteps > 0:
        if resume_payload is None:
            raise ValueError(
                "--additional-timesteps requires --resume-from"
            )
        additional_iterations = (
            int(args.additional_timesteps) // args.batch_size
        )
        if additional_iterations < 1:
            raise ValueError(
                "additional_timesteps must contain at least one complete "
                "PPO batch"
            )
        if "completed_iteration" not in resume_payload:
            raise ValueError(
                "resume checkpoint is missing completed_iteration"
            )
        completed_iteration = int(
            resume_payload["completed_iteration"]
        )
        args.num_iterations = completed_iteration + additional_iterations
        args.total_timesteps = args.num_iterations * args.batch_size
    else:
        args.num_iterations = args.total_timesteps // args.batch_size

    if args.num_iterations < 1:
        raise ValueError(
            "total_timesteps must contain at least one complete PPO batch"
        )
    return args


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.init_from is not None and args.resume_from is not None:
        raise ValueError(
            "--init-from and --resume-from are mutually exclusive"
        )
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if args.vector_env_mode not in {"sync", "async"}:
        raise ValueError("vector_env_mode must be sync or async")
    if args.collision_penalty < 0.0:
        raise ValueError("collision_penalty must be non-negative")
    if not 0.0 < args.orient_success_threshold_deg <= 180.0:
        raise ValueError(
            "orient_success_threshold_deg must be within (0, 180]"
        )
    if args.target_direction_mode not in {"fixed", "random_cone"}:
        raise ValueError(
            "target_direction_mode must be fixed or random_cone"
        )
    if args.network_profile not in {"auto", "legacy", "pose"}:
        raise ValueError(
            "network_profile must be auto, legacy, or pose"
        )
    if not (
        0.0
        <= args.target_tilt_min_deg
        <= args.target_tilt_max_deg
        <= 90.0
    ):
        raise ValueError(
            "target tilt bounds must satisfy 0 <= min <= max <= 90"
        )
    if args.target_azimuth_min_deg > args.target_azimuth_max_deg:
        raise ValueError(
            "target azimuth min must not exceed target azimuth max"
        )
    if args.pose_feasibility_attempts <= 0:
        raise ValueError("pose_feasibility_attempts must be positive")
    if args.approach_curriculum and not (
        args.target_tilt_min_deg
        <= args.curriculum_stage1_tilt_deg
        <= args.curriculum_stage2_tilt_deg
        <= args.curriculum_stage3_tilt_deg
        <= args.curriculum_stage4_tilt_deg
        <= args.curriculum_stage5_tilt_deg
        <= args.target_tilt_max_deg
    ):
        raise ValueError(
            "curriculum tilt limits must increase within the target range"
        )
    if args.curriculum_mastery_window <= 0:
        raise ValueError("curriculum_mastery_window must be positive")
    if not (
        args.learning_rate > 0.0
        and 0.0 <= args.final_learning_rate <= args.learning_rate
    ):
        raise ValueError(
            "learning rates must satisfy 0 <= final <= initial"
        )
    initial_curriculum_stage(args)
    for name, rate in (
        ("curriculum_promotion_success_rate", args.curriculum_promotion_success_rate),
        (
            "curriculum_stage3_promotion_success_rate",
            args.curriculum_stage3_promotion_success_rate,
        ),
        (
            "curriculum_stage4_promotion_success_rate",
            args.curriculum_stage4_promotion_success_rate,
        ),
        (
            "curriculum_stage5_promotion_success_rate",
            args.curriculum_stage5_promotion_success_rate,
        ),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if args.curriculum_min_stage_steps <= 0:
        raise ValueError("curriculum_min_stage_steps must be positive")
    if args.curriculum_rollback_window <= 0:
        raise ValueError("curriculum_rollback_window must be positive")
    if not 0.0 <= args.curriculum_rollback_success_rate <= 1.0:
        raise ValueError(
            "curriculum_rollback_success_rate must be within [0, 1]"
        )
    if args.curriculum_rollback_grace_steps < 0:
        raise ValueError(
            "curriculum_rollback_grace_steps must be non-negative"
        )
    if args.curriculum_stage_action_std_decay_steps <= 0:
        raise ValueError(
            "curriculum_stage_action_std_decay_steps must be positive"
        )
    if args.approach_curriculum:
        distance_thresholds = (
            args.curriculum_stage1_dist_success_thresh,
            args.curriculum_stage2_dist_success_thresh,
            args.curriculum_stage3_dist_success_thresh,
            args.curriculum_stage4_dist_success_thresh,
            args.curriculum_stage5_dist_success_thresh,
            0.05,
        )
        if any(value <= 0.0 for value in distance_thresholds):
            raise ValueError(
                "curriculum distance success thresholds must be positive"
            )
        if any(
            earlier < later
            for earlier, later in zip(
                distance_thresholds, distance_thresholds[1:]
            )
        ):
            raise ValueError(
                "curriculum distance thresholds must not loosen by stage"
            )
        direction_thresholds = (
            args.curriculum_stage1_orient_success_deg,
            args.curriculum_stage2_orient_success_deg,
            args.curriculum_stage3_orient_success_deg,
            args.curriculum_stage4_orient_success_deg,
            args.curriculum_stage5_orient_success_deg,
            args.orient_success_threshold_deg,
        )
        if any(
            not 0.0 < value <= 180.0
            for value in direction_thresholds
        ):
            raise ValueError(
                "curriculum direction success thresholds must be within (0, 180]"
            )
        if any(
            earlier < later
            for earlier, later in zip(
                direction_thresholds, direction_thresholds[1:]
            )
        ):
            raise ValueError(
                "curriculum direction thresholds must not loosen by stage"
            )
        for name, box in (
            ("curriculum_stage1_position_box", args.curriculum_stage1_position_box),
            ("curriculum_stage2_position_box", args.curriculum_stage2_position_box),
            ("curriculum_stage3_position_box", args.curriculum_stage3_position_box),
            ("curriculum_stage4_position_box", args.curriculum_stage4_position_box),
            ("curriculum_stage5_position_box", args.curriculum_stage5_position_box),
        ):
            values = np.asarray(box, dtype=np.float64).reshape(-1)
            if (
                values.shape != (4,)
                or not np.isfinite(values).all()
                or values[0] > values[1]
                or values[2] > values[3]
            ):
                raise ValueError(
                    f"{name} must contain ordered finite xmin xmax ymin ymax"
                )
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

    configure_training_horizon(args, resume_preview)
    if args.additional_timesteps > 0:
        completed_iteration = int(
            resume_preview["completed_iteration"]
        )
        effective_added_steps = (
            args.num_iterations - completed_iteration
        ) * args.batch_size
        print(
            "Extending checkpoint training: "
            f"old_global_step={int(resume_preview['global_step'])}, "
            f"effective_added_steps={effective_added_steps}, "
            f"new_final_global_step={args.total_timesteps}, "
            f"total_iterations={args.num_iterations}"
        )
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Training device: {device}")

    # env setup
    envs = make_vector_env(args, run_name)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        envs,
        initial_action_std=args.initial_action_std,
        network_profile=args.network_profile,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    global_step = 0
    completed_iteration = 0
    start_iteration = 1
    saved_curriculum_state = None
    if args.init_from is not None:
        source_kind = load_agent_weights(
            args.init_from,
            agent,
            device,
        )
        # A new fine-tuning run deliberately resets exploration while keeping
        # all learned network weights. Exact --resume-from runs do not do this.
        agent.reset_action_std(args.initial_action_std)
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
        saved_curriculum_state = resume_payload.get("curriculum_state")
        if completed_iteration > args.num_iterations:
            raise ValueError(
                "checkpoint iteration exceeds the saved training plan"
            )
        print(
            f"Resuming run {run_name} from global_step={global_step}, "
            f"next_iteration={start_iteration}/{args.num_iterations}"
        )

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
    curriculum_enabled = bool(
        args.approach_curriculum
        and args.env_id == "RM75Approach-v0"
        and args.target_direction_mode == "random_cone"
    )
    curriculum_stage = (
        initial_curriculum_stage(args, saved_curriculum_state)
        if curriculum_enabled
        else 6
    )
    curriculum_stage_start_step = global_step
    curriculum_rollback_armed = initial_curriculum_rollback_armed(
        curriculum_stage,
        saved_curriculum_state,
    )
    curriculum_recent_successes = deque(
        maxlen=args.curriculum_mastery_window
    )
    latest_curriculum_eval_stage = None
    latest_curriculum_eval_success_rate = None
    if curriculum_enabled and saved_curriculum_state is not None:
        curriculum_stage_start_step = int(
            saved_curriculum_state.get("stage_start_step", global_step)
        )
        saved_successes = saved_curriculum_state.get(
            "recent_successes", []
        )
        curriculum_recent_successes.extend(
            float(value) for value in saved_successes
        )
    apply_approach_curriculum(
        envs, args, stage=curriculum_stage
    )
    reset_seed = args.seed + completed_iteration
    next_obs, _ = envs.reset(seed=reset_seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs).to(device)

    finished_episodes = 0
    recent_successes = deque(maxlen=100)
    recent_strict_successes = deque(maxlen=100)
    recent_collisions = deque(maxlen=100)
    recent_direction_successes = deque(maxlen=100)
    recent_region_successes = {
        "uniform": deque(maxlen=100),
        "hard_negative_y": deque(maxlen=100),
        "hard_far_x": deque(maxlen=100),
        "fixed": deque(maxlen=100),
    }

    latest_checkpoint_path = (
        run_directory / "checkpoints" / "checkpoint_latest.pt"
    )
    best_checkpoint_path = (
        run_directory / "checkpoints" / "checkpoint_best.pt"
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

    def current_curriculum_state():
        """Return enough state to resume a mastery-gated curriculum exactly."""
        if not curriculum_enabled:
            return None
        return {
            "stage": int(curriculum_stage),
            "stage_start_step": int(curriculum_stage_start_step),
            "recent_successes": list(curriculum_recent_successes),
            "rollback_armed": bool(curriculum_rollback_armed),
        }

    def save_best_checkpoint():
        save_training_checkpoint(
            best_checkpoint_path,
            agent,
            optimizer,
            global_step=global_step,
            completed_iteration=completed_iteration,
            args=args,
            run_name=run_name,
            wandb_run_id=(
                wandb_run.id if wandb_run is not None else None
            ),
            best_eval_metrics=best_eval_metrics,
            curriculum_state=current_curriculum_state(),
        )
        print(f"best checkpoint saved to {best_checkpoint_path}")

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

    for iteration in range(start_iteration, args.num_iterations + 1):
        reward_component_sums = {}
        reward_component_counts = {}
        apply_approach_curriculum(
            envs, args, stage=curriculum_stage
        )
        curriculum_stage_steps = max(
            0, global_step - curriculum_stage_start_step
        )
        if curriculum_enabled:
            scheduled_action_std = compute_stage_action_std(
                initial_action_std=args.initial_action_std,
                final_action_std=args.final_action_std,
                stage_steps=curriculum_stage_steps,
                decay_steps=(
                    args.curriculum_stage_action_std_decay_steps
                ),
                hold_fraction=args.action_std_hold_fraction,
            )
        else:
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
            scheduled_lr = compute_learning_rate(
                args.learning_rate,
                args.final_learning_rate,
                global_step,
                max(args.total_timesteps, 1),
            )
            lrnow = min(
                optimizer.param_groups[0]["lr"],
                scheduled_lr,
            )
            optimizer.param_groups[0]["lr"] = lrnow

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
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done.astype(np.float32), dtype=torch.float32, device=device)

            accumulate_reward_components(
                infos,
                args.num_envs,
                reward_component_sums,
                reward_component_counts,
            )

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
                            "ever_active_success",
                            infos.get(
                                "success",
                                np.zeros(args.num_envs),
                            ),
                        )[env_idx]
                    )
                    strict_success = float(
                        infos.get(
                            "strict_success",
                            np.full(args.num_envs, success),
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
                    final_angle_error_deg = float(
                        infos.get(
                            "angle_error_deg",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    direction_success = float(
                        infos.get(
                            "direction_success",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    min_distance = float(
                        infos.get(
                            "min_distance",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    min_angle_error_deg = float(
                        infos.get(
                            "min_angle_error_deg",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    min_pose_error = float(
                        infos.get(
                            "min_pose_error",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    ever_position_success = float(
                        infos.get(
                            "ever_position_success",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    ever_direction_success = float(
                        infos.get(
                            "ever_direction_success",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
                    target_tilt_deg = float(
                        infos.get(
                            "target_tilt_deg",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )
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
                    recent_strict_successes.append(strict_success)
                    if curriculum_enabled:
                        curriculum_recent_successes.append(success)
                    recent_collisions.append(collision)
                    if np.isfinite(direction_success):
                        recent_direction_successes.append(
                            direction_success
                        )
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
                            f"final_dist={final_dist:.4f}"
                            + (
                                ", final_angle="
                                f"{final_angle_error_deg:.2f}deg, "
                                "direction_success="
                                f"{bool(direction_success)}, "
                                f"min_dist={min_distance:.4f}, "
                                "min_angle="
                                f"{min_angle_error_deg:.2f}deg, "
                                f"tilt={target_tilt_deg:.2f}deg"
                                if np.isfinite(final_angle_error_deg)
                                else ""
                            )
                        )

                    writer.add_scalar(
                        "charts/episodic_return",
                        episode_return,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        episode_length,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_success",
                        success,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/strict_success",
                        strict_success,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/collision",
                        collision,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/timeout",
                        timeout,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/final_distance",
                        final_dist,
                        global_step,
                    )
                    if np.isfinite(final_angle_error_deg):
                        writer.add_scalar(
                            "charts/final_angle_error_deg",
                            final_angle_error_deg,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/direction_success",
                            direction_success,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/min_distance",
                            min_distance,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/min_angle_error_deg",
                            min_angle_error_deg,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/min_pose_error",
                            min_pose_error,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/ever_position_success",
                            ever_position_success,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/ever_direction_success",
                            ever_direction_success,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/target_tilt_deg",
                            target_tilt_deg,
                            global_step,
                        )

                    if curriculum_enabled:
                        stage_steps = max(
                            0,
                            global_step - curriculum_stage_start_step,
                        )
                        if (
                            curriculum_rollback_armed
                            and curriculum_stage > 1
                            and should_demote_curriculum(
                                curriculum_recent_successes,
                                args.curriculum_rollback_success_rate,
                                args.curriculum_rollback_window,
                                stage_steps,
                                args.curriculum_rollback_grace_steps,
                            )
                        ):
                            previous_stage = curriculum_stage
                            curriculum_stage -= 1
                            curriculum_stage_start_step = global_step
                            curriculum_rollback_armed = (
                                curriculum_stage > 1
                            )
                            latest_curriculum_eval_stage = None
                            latest_curriculum_eval_success_rate = None
                            curriculum_recent_successes.clear()
                            recent_successes.clear()
                            apply_approach_curriculum(
                                envs, args, stage=curriculum_stage
                            )
                            print(
                                "Curriculum rolled back: "
                                f"stage {previous_stage} -> "
                                f"{curriculum_stage}; recent success "
                                f"<= {args.curriculum_rollback_success_rate:.2f}."
                            )
                        elif (
                            curriculum_stage < 6
                            and (
                                args.eval_interval <= 0
                                or (
                                    latest_curriculum_eval_stage
                                    == curriculum_stage
                                    and latest_curriculum_eval_success_rate
                                    >= curriculum_promotion_threshold(
                                        args, curriculum_stage
                                    )
                                )
                            )
                            and should_promote_curriculum(
                                curriculum_recent_successes,
                                curriculum_promotion_threshold(
                                    args, curriculum_stage
                                ),
                                args.curriculum_mastery_window,
                                stage_steps,
                                args.curriculum_min_stage_steps,
                            )
                        ):
                            previous_stage = curriculum_stage
                            curriculum_stage += 1
                            curriculum_stage_start_step = global_step
                            curriculum_rollback_armed = True
                            curriculum_recent_successes.clear()
                            latest_curriculum_eval_stage = None
                            latest_curriculum_eval_success_rate = None
                            # Active success rates from two different stages
                            # must not be averaged into the same chart window.
                            recent_successes.clear()
                            apply_approach_curriculum(
                                envs, args, stage=curriculum_stage
                            )
                            print(
                                "Curriculum promoted: "
                                f"stage {previous_stage} -> "
                                f"{curriculum_stage} after at least "
                                f"{args.curriculum_min_stage_steps} steps "
                                f"and {args.curriculum_mastery_window} "
                                "mastery episodes."
                            )

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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        for name, total in reward_component_sums.items():
            writer.add_scalar(
                f"reward/{name}",
                total / max(reward_component_counts[name], 1),
                global_step,
            )
        if args.env_id == "RM75Approach-v0":
            logged_curriculum = curriculum_stage_settings(
                args, curriculum_stage
            )
            writer.add_scalar(
                "charts/curriculum_tilt_max_deg",
                logged_curriculum["tilt_max_deg"],
                global_step,
            )
            writer.add_scalar(
                "charts/curriculum_stage",
                curriculum_stage,
                global_step,
            )
            writer.add_scalar(
                "charts/curriculum_success_distance_thresh",
                logged_curriculum["success_distance_thresh"],
                global_step,
            )
            writer.add_scalar(
                "charts/curriculum_success_angle_deg",
                logged_curriculum["success_angle_deg"],
                global_step,
            )
            writer.add_scalar(
                "charts/curriculum_stage_steps",
                max(0, global_step - curriculum_stage_start_step),
                global_step,
            )
            writer.add_scalar(
                "charts/curriculum_promotion_threshold",
                curriculum_promotion_threshold(args, curriculum_stage),
                global_step,
            )
            if curriculum_recent_successes:
                writer.add_scalar(
                    "charts/curriculum_mastery_rate",
                    float(np.mean(curriculum_recent_successes)),
                    global_step,
                )
        writer.add_scalar(
            "charts/action_std_ceiling",
            scheduled_action_std,
            global_step,
        )
        std_mean, std_min, std_max = agent.effective_action_std_stats()
        writer.add_scalar("charts/action_std_mean", std_mean, global_step)
        writer.add_scalar("charts/action_std_min", std_min, global_step)
        writer.add_scalar("charts/action_std_max", std_max, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        action_saturation_rate = float(
            (actions.abs() > 0.98).float().mean().item()
        )
        writer.add_scalar(
            "charts/action_saturation_rate",
            action_saturation_rate,
            global_step,
        )
        session_steps = global_step - session_start_step
        sps = int(session_steps / max(time.time() - start_time, 1e-6))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)
        if recent_successes:
            success_rate = float(np.mean(recent_successes))
            collision_rate = float(np.mean(recent_collisions))
            print(
                "Recent-100 success/collision rate: "
                f"{success_rate:.3f}/{collision_rate:.3f}"
            )
            writer.add_scalar(
                "charts/success_rate",
                success_rate,
                global_step,
            )
            writer.add_scalar(
                "charts/collision_rate",
                collision_rate,
                global_step,
            )
            if recent_strict_successes:
                strict_success_rate = float(
                    np.mean(recent_strict_successes)
                )
                writer.add_scalar(
                    "charts/strict_success_rate",
                    strict_success_rate,
                    global_step,
                )
            if recent_direction_successes:
                direction_success_rate = float(
                    np.mean(recent_direction_successes)
                )
                print(
                    "Recent-100 direction success rate: "
                    f"{direction_success_rate:.3f}"
                )
                writer.add_scalar(
                    "charts/direction_success_rate",
                    direction_success_rate,
                    global_step,
                )

        for region, outcomes in recent_region_successes.items():
            if outcomes:
                writer.add_scalar(
                    f"charts/region_success_rate/{region}",
                    float(np.mean(outcomes)),
                    global_step,
                )

        completed_iteration = iteration
        if (
            not stop_requested["value"]
            and next_eval_step is not None
            and global_step >= next_eval_step
        ):
            (
                best_eval_metrics,
                best_updated,
                current_eval_metrics,
            ) = evaluate_and_maybe_save_best(
                agent,
                args,
                device,
                writer,
                global_step=global_step,
                best_metrics=best_eval_metrics,
                best_model_path=best_model_path,
                best_metrics_path=best_metrics_path,
                curriculum_stage=(
                    curriculum_stage if curriculum_enabled else None
                ),
            )
            if curriculum_enabled:
                latest_curriculum_eval_stage = int(
                    current_eval_metrics["curriculum_stage"]
                )
                latest_curriculum_eval_success_rate = float(
                    current_eval_metrics["success_rate"]
                )
            if best_updated and args.save_model:
                save_best_checkpoint()
            last_eval_step = global_step
            next_eval_step = next_checkpoint_step(
                global_step,
                args.eval_interval,
            )

        if (
            args.save_model
            and next_save_step is not None
            and global_step >= next_save_step
        ):
            save_training_checkpoint(
                latest_checkpoint_path,
                agent,
                optimizer,
                global_step=global_step,
                completed_iteration=completed_iteration,
                args=args,
                run_name=run_name,
                wandb_run_id=(
                    wandb_run.id if wandb_run is not None else None
                ),
                best_eval_metrics=best_eval_metrics,
                curriculum_state=current_curriculum_state(),
            )
            print(f"checkpoint saved to {latest_checkpoint_path}")
            next_save_step = next_checkpoint_step(
                global_step,
                args.checkpoint_interval,
            )

        if stop_requested["value"]:
            interrupted = True
            if args.save_model:
                save_training_checkpoint(
                    latest_checkpoint_path,
                    agent,
                    optimizer,
                    global_step=global_step,
                    completed_iteration=completed_iteration,
                    args=args,
                    run_name=run_name,
                    wandb_run_id=(
                        wandb_run.id if wandb_run is not None else None
                    ),
                    best_eval_metrics=best_eval_metrics,
                    curriculum_state=current_curriculum_state(),
                )
                print(
                    "graceful-stop checkpoint saved to "
                    f"{latest_checkpoint_path}"
                )
            break

    signal.signal(signal.SIGINT, previous_sigint_handler)

    if (
        not interrupted
        and args.eval_interval > 0
        and last_eval_step != global_step
    ):
        (
            best_eval_metrics,
            best_updated,
            current_eval_metrics,
        ) = evaluate_and_maybe_save_best(
            agent,
            args,
            device,
            writer,
            global_step=global_step,
            best_metrics=best_eval_metrics,
            best_model_path=best_model_path,
            best_metrics_path=best_metrics_path,
            curriculum_stage=(
                curriculum_stage if curriculum_enabled else None
            ),
        )
        if curriculum_enabled:
            latest_curriculum_eval_stage = int(
                current_eval_metrics["curriculum_stage"]
            )
            latest_curriculum_eval_success_rate = float(
                current_eval_metrics["success_rate"]
            )
        if best_updated and args.save_model:
            save_best_checkpoint()
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
        model_path = run_directory / f"{args.exp_name}.cleanrl_model"
        save_agent_weights(model_path, agent)
        print(f"model saved to {model_path}")
        save_training_checkpoint(
            latest_checkpoint_path,
            agent,
            optimizer,
            global_step=global_step,
            completed_iteration=completed_iteration,
            args=args,
            run_name=run_name,
            wandb_run_id=(
                wandb_run.id if wandb_run is not None else None
            ),
            best_eval_metrics=best_eval_metrics,
            curriculum_state=current_curriculum_state(),
        )
        print(f"final checkpoint saved to {latest_checkpoint_path}")
        try:
            from cleanrl_utils.evals.ppo_eval import evaluate
        except ModuleNotFoundError:
            evaluate = None
            print(
                "CleanRL evaluation helper is not installed; "
                "the model was saved successfully and automatic evaluation was skipped."
            )

        episodic_returns = []
        if evaluate is not None:
            episodic_returns = evaluate(
                model_path,
                lambda env_id, idx, capture_video, run_name, gamma: make_env(
                    env_id, idx, capture_video, run_name, gamma, args=args
                ),
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=Agent,
                device=device,
                capture_video=False,
                gamma=args.gamma,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            if evaluate is None:
                raise RuntimeError("Model upload requires the CleanRL evaluation helpers.")
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
