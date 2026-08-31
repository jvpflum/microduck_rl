"""Forward-speed specialist for the passive-wheel MicroDuck roller model.

The legacy roller task treats ``cmd_x`` as throttle and mainly rewards wheel
spin.  That is useful for discovering a skating gait, but it cannot distinguish
fast chassis motion from spinning/slipping wheels.  Sprint instead treats
``cmd_x`` as a target forward speed and makes measured base-velocity tracking
the dominant task reward.  The robot, observations, actuator model, domain
randomization, and deployable 61D actor interface remain unchanged.
"""

import dataclasses
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def forward_speed_progress(
    env: "ManagerBasedRlEnv", command_name: str = "twist"
) -> torch.Tensor:
    """Dense 0..1 progress toward commanded chassis speed.

    Unlike a narrow Gaussian, this retains a constant useful gradient from a
    standstill.  It uses measured body velocity, not wheel spin, and saturates
    at target speed so overspeed cannot farm additional reward.
    """
    command = env.command_manager.get_command(command_name)[:, 0]
    speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    target = torch.clamp(command, min=0.05)
    return torch.clamp(torch.clamp(speed, min=0.0) / target, max=1.0)


def lateral_speed_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Squared body-frame lateral speed; zero is straight-line skating."""
    lateral_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 1]
    return torch.nan_to_num(lateral_speed.square(), nan=0.0)


def make_microduck_velocity_sprint_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the straight-line Sprint-v3 roller environment."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    command = cfg.commands["twist"]
    # Stage 1 protects and extends Pollen's established gait.  Low-speed and
    # stop samples are deliberately excluded: Pollen already stops cleanly,
    # while those samples dominated Sprint-v1 and pulled PPO into standing.
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.30, 0.60)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)

    # The primary objective measures body motion.  Wheel spin remains only as a
    # small discovery aid, so wheel slip cannot dominate the learned solution.
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=8.0,
        params={"command_name": "twist", "std": 0.35},
    )
    cfg.rewards["forward_speed_progress"] = RewardTermCfg(
        func=forward_speed_progress,
        weight=6.0,
        params={"command_name": "twist"},
    )
    cfg.rewards["wheel_speed"].weight = 5.0
    cfg.rewards["wheel_speed"].params["vel_scale"] = 0.50
    cfg.rewards["coasting"] = RewardTermCfg(
        func=microduck_mdp.coasting_reward,
        weight=1.0,
        params={"command_name": "twist", "vel_std": 0.25, "stillness_std": 5.0},
    )

    # Sprint-v2 proved that the Pollen gait can exceed 0.50 m/s, but its late
    # checkpoints bought speed with 17 degree peak tilt and accumulated yaw.
    # Make those failure modes expensive while retaining the same dense speed
    # gradient.  A narrower upright Gaussian has useful discrimination in the
    # 10--17 degree band, and the angle-based heading reward remains corrective.
    cfg.rewards["upright"].weight = 5.0
    cfg.rewards["upright"].params["std"] = 0.20
    cfg.rewards["body_ang_vel"].weight = -0.08
    cfg.rewards["heading_hold"].weight = 4.0
    cfg.rewards["heading_hold"].params["std"] = 0.20
    cfg.rewards["gait_symmetry"].weight = -2.0
    cfg.rewards["lateral_speed"] = RewardTermCfg(
        func=lateral_speed_penalty,
        weight=-2.0,
    )
    # A small lean remains useful for acceleration, but no longer pays the
    # policy to live near the stability limit.
    cfg.rewards["forward_lean"].weight = 0.25
    cfg.rewards["forward_lean"].params.update({"target_pitch": 0.06, "std": 0.10})

    # Negative commands no longer mean brake.  Stopping is learned from the
    # zero-command fraction through the same measured velocity objective.
    cfg.rewards.pop("braking", None)

    return cfg


MicroduckSprintRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    algorithm=dataclasses.replace(
        MicroduckRollersRlCfg.algorithm,
        learning_rate=2.0e-4,
        desired_kl=0.003,
        entropy_coef=0.005,
        num_learning_epochs=2,
        clip_param=0.10,
    ),
    experiment_name="velocity_sprint",
    run_name="velocity_sprint",
    save_interval=5,
)
