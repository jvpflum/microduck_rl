"""Race5 refinement: retain the discovered skate speed and restore control.

Unlike unconstrained discovery, this stage uses the exact wheel-bearing drag
from the official Race5 evaluator.  Most environments still request full race
effort, while smaller buckets practice cruise, stopping, and left/right turns.
The reward is mode-gated so a stop command cannot earn the speed objective and
a turn command is not punished for leaving the straight race heading.
"""

from __future__ import annotations

import dataclasses

import torch

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_discovery_env_cfg import (
    SPEED_DISCOVERY_CAP_MPS,
)
from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
    world_forward_velocity,
)
from mjlab_microduck.tasks.microduck_velocity_race_env_cfg import (
    race_heading_error_squared,
    race_lane_error_squared,
)


class SpeedRetentionRaceLineCommand(microduck_mdp.RaceLineVelocityCommand):
    """Sample deployment-relevant discrete modes with race effort dominant."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._manual_turn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        sample = torch.rand(len(env_ids), device=self.device)
        stop = sample < 0.15
        cruise = (sample >= 0.15) & (sample < 0.25)
        left = (sample >= 0.25) & (sample < 0.325)
        right = (sample >= 0.325) & (sample < 0.40)
        race = sample >= 0.40
        self.vel_command_b[env_ids[stop], 0] = 0.0
        self.vel_command_b[env_ids[cruise], 0] = 0.30
        self.vel_command_b[env_ids[left | right], 0] = 0.20
        self.vel_command_b[env_ids[race], 0] = 0.80
        self._manual_turn[env_ids] = left | right
        self.vel_command_b[env_ids[left], 2] = 0.30
        self.vel_command_b[env_ids[right], 2] = -0.30

    def _update_command(self) -> None:
        manual_yaw = self.vel_command_b[:, 2].clone()
        super()._update_command()
        self.vel_command_b[self._manual_turn, 2] = manual_yaw[self._manual_turn]


@dataclasses.dataclass(kw_only=True)
class SpeedRetentionRaceLineCommandCfg(microduck_mdp.RaceLineVelocityCommandCfg):
    def build(self, env):
        return SpeedRetentionRaceLineCommand(self, env)


def _commands(env):
    return env.command_manager.get_command("twist")


def race_mode(env):
    return (_commands(env)[:, 0] >= 0.60).to(dtype=torch.float32)


def race_body_forward_velocity(env):
    return race_mode(env) * microduck_mdp.speed_discovery_forward_velocity(
        env, safety_cap_mps=SPEED_DISCOVERY_CAP_MPS
    )


def race_world_forward_velocity(env):
    return race_mode(env) * world_forward_velocity(
        env, safety_cap_mps=SPEED_DISCOVERY_CAP_MPS
    )


def race_world_forward_velocity_squared(env):
    speed = world_forward_velocity(env, safety_cap_mps=SPEED_DISCOVERY_CAP_MPS)
    return race_mode(env) * speed * torch.abs(speed)


def race_lane_error(env):
    return race_mode(env) * race_lane_error_squared(env, lane_half_width_m=1.0)


def race_heading_error(env):
    return race_mode(env) * race_heading_error_squared(
        env, maximum_yaw_rad=0.785398
    )


def race_lateral_velocity_squared(env):
    lateral = torch.nan_to_num(env.scene["robot"].data.root_link_lin_vel_w[:, 1], nan=0.0)
    return race_mode(env) * lateral.square()


def cruise_error_squared(env):
    cmd = _commands(env)
    active = ((cmd[:, 0] >= 0.10) & (cmd[:, 0] <= 0.40)).to(torch.float32)
    speed = torch.nan_to_num(env.scene["robot"].data.root_link_lin_vel_b[:, 0], nan=0.0)
    return active * torch.square(speed - cmd[:, 0])


def stop_speed_squared(env):
    cmd = _commands(env)
    active = (torch.abs(cmd[:, 0]) < 0.05).to(torch.float32)
    speed = torch.nan_to_num(
        env.scene["robot"].data.root_link_lin_vel_w[:, :2], nan=0.0
    ).norm(dim=-1)
    # A Gaussian at 0.1 m/s underflows at race speed and supplies no braking
    # gradient.  Quadratic speed remains informative throughout the stop.
    return active * speed.square()


def turn_tracking(env, std: float = 0.20):
    cmd = _commands(env)
    active = (torch.abs(cmd[:, 2]) >= 0.25).to(torch.float32)
    yaw_rate = torch.nan_to_num(env.scene["robot"].data.root_link_ang_vel_w[:, 2], nan=0.0)
    return active * torch.exp(-torch.square((yaw_rate - cmd[:, 2]) / std))


def upright_state(env):
    quat = env.scene["robot"].data.root_link_quat_w
    return torch.nan_to_num(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), nan=0.0)


def make_microduck_speed_retention_env_cfg(play: bool = False):
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.episode_length_s = 24.0

    # Match the evaluator instead of learning on frictionless bearings.
    cfg.events["official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (0.003, 0.003),
        },
    )

    old = cfg.commands["twist"]
    command_args = dict(vars(old))
    for extra in ("yaw_kp", "lateral_kp", "yaw_kd", "max_correction"):
        command_args.pop(extra, None)
    command_args["rel_standing_envs"] = 0.0
    command_args["ranges"].lin_vel_x = (0.0, 0.80)
    command_args["ranges"].ang_vel_z = (-0.30, 0.30)
    cfg.commands["twist"] = SpeedRetentionRaceLineCommandCfg(
        **command_args,
        yaw_kp=0.55,
        lateral_kp=0.10,
        yaw_kd=0.08,
        max_correction=0.18,
    )

    # Replace the unconditional discovery terms with mode-gated objectives.
    for name in (
        "forward_velocity_mps", "forward_velocity_squared", "speed_target_progress",
        "world_forward_velocity_mps", "world_forward_velocity_squared", "heading_hold",
        "lane_error", "world_lateral_velocity", "heading_error",
    ):
        cfg.rewards.pop(name, None)
    cfg.rewards.update(
        {
            "race_body_speed": RewardTermCfg(func=race_body_forward_velocity, weight=2.0),
            "race_world_speed": RewardTermCfg(func=race_world_forward_velocity, weight=5.0),
            "race_world_speed_squared": RewardTermCfg(
                func=race_world_forward_velocity_squared, weight=0.75
            ),
            "race_lane": RewardTermCfg(func=race_lane_error, weight=-3.0),
            "race_heading": RewardTermCfg(func=race_heading_error, weight=-2.0),
            "race_lateral_speed": RewardTermCfg(
                func=race_lateral_velocity_squared, weight=-3.0
            ),
            "cruise_error": RewardTermCfg(func=cruise_error_squared, weight=-3.0),
            "stop_speed": RewardTermCfg(func=stop_speed_squared, weight=-6.0),
            "turn_tracking": RewardTermCfg(func=turn_tracking, weight=2.0),
            "upright_state": RewardTermCfg(func=upright_state, weight=2.0),
        }
    )
    cfg.curriculum.clear()
    return cfg


MicroduckSpeedRetentionRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=5.0e-6,
        desired_kl=2.5e-4,
        clip_param=0.03,
        num_learning_epochs=1,
    ),
    experiment_name="microduck_speed_retention",
    run_name="microduck_speed_retention",
    save_interval=20,
)
