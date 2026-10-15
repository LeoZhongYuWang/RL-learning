"""Isolated checkpoint round-trip used by the unittest suite.

MuJoCo's EGL teardown and PyTorch serialization can conflict when dozens of
MuJoCo models have already been constructed in the same test process. Running
this round-trip in a fresh process verifies the production path without that
unrelated native-library teardown interaction.
"""

import tempfile
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithms.ppo_continuous_action import (
    Agent,
    Args,
    load_training_checkpoint,
    save_training_checkpoint,
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


def main():
    args = Args()
    args.target_mode = "none"
    args.total_timesteps = 4096
    args.batch_size = 2048
    args.minibatch_size = 128
    args.num_iterations = 2

    spaces = DummyRM75Spaces()
    source = Agent(spaces, initial_action_std=0.12)
    source_optimizer = torch.optim.Adam(
        source.parameters(),
        lr=args.learning_rate,
    )
    first_parameter = next(source.parameters())
    source_optimizer.state[first_parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(first_parameter),
        "exp_avg_sq": torch.zeros_like(first_parameter),
    }

    restored = Agent(spaces, initial_action_std=0.12)
    restored_optimizer = torch.optim.Adam(
        restored.parameters(),
        lr=args.learning_rate,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoint_latest.pt"
        save_training_checkpoint(
            checkpoint_path,
            source,
            source_optimizer,
            global_step=2048,
            completed_iteration=1,
            args=args,
            run_name="checkpoint-test",
            wandb_run_id="wandb-test-id",
            curriculum_state={
                "stage": 3,
                "stage_start_step": 1024,
                "recent_successes": [1.0, 0.0],
                "rollback_armed": True,
            },
        )
        metadata = load_training_checkpoint(
            checkpoint_path,
            restored,
            restored_optimizer,
            device="cpu",
        )
        assert not checkpoint_path.with_suffix(".pt.tmp").exists()

    assert metadata["format_version"] == 1
    assert metadata["global_step"] == 2048
    assert metadata["completed_iteration"] == 1
    assert metadata["run_name"] == "checkpoint-test"
    assert metadata["wandb_run_id"] == "wandb-test-id"
    assert metadata["args"]["total_timesteps"] == 4096
    assert metadata["curriculum_state"] == {
        "stage": 3,
        "stage_start_step": 1024,
        "recent_successes": [1.0, 0.0],
        "rollback_armed": True,
    }
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[key])
    assert len(restored_optimizer.state) == len(source_optimizer.state)
    print("checkpoint round-trip: OK")


if __name__ == "__main__":
    main()
