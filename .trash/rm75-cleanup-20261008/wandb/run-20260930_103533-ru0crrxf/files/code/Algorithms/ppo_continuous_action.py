# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
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
    num_envs: int = 1
    """the number of parallel game environments"""
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
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""

    target_mode: str = "random"
    """goal sampling mode passed to UR5 environment"""
    target_pos: Optional[Tuple[float, float, float]] = None
    """explicit goal position when target_mode=='fixed'"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class GoalAugmentWrapper(gym.ObservationWrapper):
    """Append the current goal position to the observation vector."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Box), "Only Box observation spaces are supported"
        orig_space: gym.spaces.Box = self.observation_space
        extra_low = np.full(3, -np.inf, dtype=np.float32)
        extra_high = np.full(3, np.inf, dtype=np.float32)
        low = np.concatenate([orig_space.low.astype(np.float32), extra_low])
        high = np.concatenate([orig_space.high.astype(np.float32), extra_high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        goal = getattr(self.env, "goal_pos", None)
        if goal is None:
            goal = np.zeros(3, dtype=np.float32)
        else:
            goal = np.asarray(goal, dtype=np.float32)
        return np.concatenate([observation.astype(np.float32), goal], dtype=np.float32)
def make_env(env_id, idx, capture_video, run_name, gamma, *, args: Args):
    def thunk():
        env = gym.make(
            env_id,
            render_mode=None,
            target_mode=args.target_mode,
            target_pos=args.target_pos,
        )
        if capture_video and idx == 0 and "rgb_array" in getattr(env, "metadata", {}).get("render_modes", []):
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # NOTE: GoalAugmentWrapper removed because env.py already includes goal in observation
        # env = GoalAugmentWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
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
        initial_action_std = 0.3
        self.actor_logstd = nn.Parameter(
            torch.full(
                (1, action_dim),
                float(np.log(initial_action_std)),
            )
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        # Bound the learned exploration scale to avoid entropy re-expanding
        # into nearly random saturated commands late in training.
        action_logstd = torch.clamp(
            self.actor_logstd,
            min=-3.0,
            max=-0.3,
        ).expand_as(action_mean)
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args=args) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs).to(device)

    finished_episodes = 0
    recent_successes = deque(maxlen=100)
    recent_collisions = deque(maxlen=100)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
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

                    final_dist = float(
                        infos.get(
                            "dist",
                            np.full(args.num_envs, np.nan),
                        )[env_idx]
                    )

                    finished_episodes += 1
                    recent_successes.append(success)
                    recent_collisions.append(collision)

                    if finished_episodes % 10 == 0 or success:
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={episode_return:.3f}, "
                            f"episodic_length={episode_length}, "
                            f"success={bool(success)}, "
                            f"collision={bool(collision)}, "
                            f"final_dist={final_dist:.4f}"
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
                        "charts/collision",
                        collision,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/final_distance",
                        final_dist,
                        global_step,
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
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

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
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
