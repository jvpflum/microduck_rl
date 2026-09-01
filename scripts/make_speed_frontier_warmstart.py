"""Create a phase-ready frontier seed without modifying its source checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def prepare_checkpoint(source: Path, destination: Path, std: float, lr: float) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_iter = int(checkpoint["iter"])
    expected_widths = {"actor_state_dict": 61, "critic_state_dict": 78}
    for state_name, expected_width in expected_widths.items():
        state = checkpoint[state_name]
        first_layer = state["mlp.0.weight"]
        if first_layer.shape[1] != expected_width:
            raise ValueError(
                f"{state_name} width is {first_layer.shape[1]}, expected {expected_width}"
            )

        # body_command is the final six observation channels.  It was always
        # zero for the source skate policy, leaving random unused input weights
        # and near-zero normalizer variance.  Neutralize both so phase features
        # initially have exactly zero influence but immediately receive useful
        # gradients during frontier PPO.
        first_layer[:, -6:] = 0.0
        state["obs_normalizer._mean"][:, -6:] = 0.0
        state["obs_normalizer._var"][:, -6:] = 1.0
        state["obs_normalizer._std"][:, -6:] = 1.0

    checkpoint["actor_state_dict"]["distribution.std_param"].fill_(std)
    checkpoint["iter"] = 0
    optimizer = checkpoint["optimizer_state_dict"]
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = lr
        group["initial_lr"] = lr

    checkpoint["infos"] = {
        "lineage": {
            "source": str(source),
            "source_iter": source_iter,
            "source_world_speed_mps": 1.09913684,
            "source_wheel_frictionloss": 0.003,
            "optimizer_moments_reset": True,
            "phase_input_columns_zeroed": True,
            "exploration_std": std,
            "learning_rate": lr,
        }
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--std", type=float, default=0.08)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    args = parser.parse_args()
    prepare_checkpoint(args.source, args.destination, args.std, args.learning_rate)


if __name__ == "__main__":
    main()
