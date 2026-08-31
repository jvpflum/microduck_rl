"""Straight-line retention stage for the discovered passive-roller speed gait.

This stage follows speed discovery.  It keeps the small reward surface and
nominal roller physics, but changes useful progress from body-forward velocity
alone to fixed world +X progress.  A closed-loop race-line correction is sent
through the existing yaw-command observation, so the 61D actor contract stays
unchanged and the policy can actually observe which way to correct.
"""

from __future__ import annotations

import dataclasses

import torch

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_discovery_env_cfg import (
    MicroduckSpeedDiscoveryRlCfg,
    SPEED_DISCOVERY_CAP_MPS,
    make_microduck_speed_discovery_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_race_env_cfg import (
    race_heading_error_squared,
    race_lane_error_squared,
)


def world_forward_velocity(env, safety_cap_mps: float = SPEED_DISCOVERY_CAP_MPS):
    speed = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
    return torch.nan_to_num(
        speed, nan=0.0, posinf=safety_cap_mps, neginf=-safety_cap_mps
    ).clamp(-safety_cap_mps, safety_cap_mps)


def world_forward_velocity_squared(
    env, safety_cap_mps: float = SPEED_DISCOVERY_CAP_MPS
):
    speed = world_forward_velocity(env, safety_cap_mps=safety_cap_mps)
    return speed * torch.abs(speed)


def world_lateral_velocity_squared(env):
    lateral = env.scene["robot"].data.root_link_lin_vel_w[:, 1]
    return torch.nan_to_num(lateral, nan=0.0).square()


def make_microduck_speed_straightening_env_cfg(play: bool = False):
    """Keep the fast skate gait while teaching it to follow a world-X lane."""
    cfg = make_microduck_speed_discovery_env_cfg(play=play)

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.80, 0.80)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.18, 0.18)
    command_args = dict(vars(command))
    command_args.pop("rel_turn_in_place_envs", None)
    cfg.commands["twist"] = microduck_mdp.RaceLineVelocityCommandCfg(
        **command_args,
        yaw_kp=0.55,
        lateral_kp=0.10,
        yaw_kd=0.08,
        max_correction=0.18,
    )

    # Body-forward terms preserve the discovered skate cadence.  World-frame
    # terms are stronger: circling no longer earns progress, and reversing in
    # world X is explicitly negative.  Direction costs start modestly so this
    # stage cannot solve straightness simply by stopping.
    cfg.rewards["forward_velocity_mps"].weight = 2.0
    cfg.rewards["forward_velocity_squared"].weight = 0.25
    cfg.rewards["speed_target_progress"].weight = 0.5
    cfg.rewards["world_forward_velocity_mps"] = RewardTermCfg(
        func=world_forward_velocity,
        weight=5.0,
        params={"safety_cap_mps": SPEED_DISCOVERY_CAP_MPS},
    )
    cfg.rewards["world_forward_velocity_squared"] = RewardTermCfg(
        func=world_forward_velocity_squared,
        weight=0.75,
        params={"safety_cap_mps": SPEED_DISCOVERY_CAP_MPS},
    )
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=2.0,
        params={"std": 0.35},
    )
    cfg.rewards["lane_error"] = RewardTermCfg(
        func=race_lane_error_squared,
        weight=-1.0,
        params={"lane_half_width_m": 1.0},
    )
    cfg.rewards["world_lateral_velocity"] = RewardTermCfg(
        func=world_lateral_velocity_squared,
        weight=-1.0,
    )
    cfg.rewards["heading_error"] = RewardTermCfg(
        func=race_heading_error_squared,
        weight=-1.0,
        params={"maximum_yaw_rad": 0.785398},
    )
    return cfg


MicroduckSpeedStraighteningRlCfg = dataclasses.replace(
    MicroduckSpeedDiscoveryRlCfg,
    experiment_name="microduck_speed_straightening",
    run_name="microduck_speed_straightening",
    save_interval=20,
)
