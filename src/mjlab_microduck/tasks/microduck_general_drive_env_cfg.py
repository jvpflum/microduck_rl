"""Official-friction multi-skill continuation of the fast roller donor.

The task deliberately samples stand/stop, low-speed tracking, turning, mixed
driving, and high-speed straight effort in one rollout population. High-speed
reward is gated by low lateral velocity and low yaw rate, so PPO cannot improve
its score by circling or by sliding sideways. The fixed external evaluation
battery remains the promotion authority.
"""

from __future__ import annotations

import dataclasses
import os
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_discovery_env_cfg import (
    MicroduckSpeedDiscoveryRlCfg,
    make_microduck_speed_discovery_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _twist(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.command_manager.get_command("twist")


def command_progress(env: "ManagerBasedRlEnv", safety_cap_mps: float = 2.5) -> torch.Tensor:
    """Body-forward progress for every positive drive command."""
    command = _twist(env)
    moving = command[:, 0] > 0.05
    speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    speed = torch.nan_to_num(speed, nan=0.0).clamp(-safety_cap_mps, safety_cap_mps)
    return moving * speed


def low_speed_tracking(
    env: "ManagerBasedRlEnv", linear_std_mps: float = 0.18, lateral_std_mps: float = 0.12
) -> torch.Tensor:
    """Track controllable low/mid commands without capping the speed frontier."""
    command = _twist(env)
    velocity = env.scene["robot"].data.root_link_lin_vel_b
    low = (command[:, 0] > 0.05) & (command[:, 0] < 0.55)
    cost = (
        ((velocity[:, 0] - command[:, 0]) / linear_std_mps).square()
        + (velocity[:, 1] / lateral_std_mps).square()
    )
    return low * torch.exp(-torch.nan_to_num(cost, nan=100.0, posinf=100.0))


def yaw_rate_tracking(env: "ManagerBasedRlEnv", std_rad_s: float = 0.12) -> torch.Tensor:
    """Follow turn commands and actively stop yawing when the stick is released."""
    command = _twist(env)
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    cost = ((yaw_rate - command[:, 2]) / std_rad_s).square()
    return torch.exp(-torch.nan_to_num(cost, nan=100.0, posinf=100.0))


def yaw_rate_error_cost(
    env: "ManagerBasedRlEnv", safety_cap_rad_s: float = 3.0
) -> torch.Tensor:
    """Dense turn/heading error that still has gradient far from the target."""
    command = _twist(env)
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    error = torch.nan_to_num(
        yaw_rate - command[:, 2], nan=0.0, posinf=safety_cap_rad_s,
        neginf=-safety_cap_rad_s,
    ).clamp(-safety_cap_rad_s, safety_cap_rad_s)
    return error.square()


def stand_still(
    env: "ManagerBasedRlEnv", linear_std_mps: float = 0.05, angular_std_rad_s: float = 0.10
) -> torch.Tensor:
    """Reward an actual stationary idle after either a reset or command transition."""
    command = _twist(env)
    standing = torch.linalg.vector_norm(command[:, :3], dim=1) < 0.01
    linear = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    angular = env.scene["robot"].data.root_link_ang_vel_b
    cost = (
        linear.square().sum(dim=1) / (linear_std_mps * linear_std_mps)
        + angular.square().sum(dim=1) / (angular_std_rad_s * angular_std_rad_s)
    )
    return standing * torch.exp(-torch.nan_to_num(cost, nan=100.0, posinf=100.0))


def stand_motion_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = _twist(env)
    standing = torch.linalg.vector_norm(command[:, :3], dim=1) < 0.01
    linear = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return standing * (linear.square().sum(dim=1) + 0.25 * yaw_rate.square())


def stand_action_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Suppress the donor's self-propelling gait whenever the command is zero."""
    command = _twist(env)
    standing = torch.linalg.vector_norm(command[:, :3], dim=1) < 0.01
    actions = env.action_manager.action
    return standing * actions.square().mean(dim=1)


def high_straight_usable_speed(
    env: "ManagerBasedRlEnv",
    reference_speed_mps: float = 1.0,
    safety_cap_mps: float = 3.0,
    lateral_std_mps: float = 0.12,
    yaw_rate_std_rad_s: float = 0.10,
) -> torch.Tensor:
    """Quadratic speed frontier that pays only for a genuinely straight gait."""
    command = _twist(env)
    high_straight = (command[:, 0] >= 0.55) & (torch.abs(command[:, 2]) <= 0.05)
    linear = env.scene["robot"].data.root_link_lin_vel_b
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    speed = torch.nan_to_num(linear[:, 0], nan=0.0).clamp(0.0, safety_cap_mps)
    quality_cost = (
        (linear[:, 1] / lateral_std_mps).square()
        + (yaw_rate / yaw_rate_std_rad_s).square()
    )
    quality = torch.exp(-torch.nan_to_num(quality_cost, nan=100.0, posinf=100.0))
    return high_straight * (speed / reference_speed_mps).square() * quality


def moving_lateral_velocity_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    command = _twist(env)
    moving = command[:, 0] > 0.05
    lateral = env.scene["robot"].data.root_link_lin_vel_b[:, 1]
    return moving * torch.nan_to_num(lateral.square(), nan=0.0)


def forward_world_progress(
    env: "ManagerBasedRlEnv", safety_cap_mps: float = 2.5
) -> torch.Tensor:
    """Reward actual world-X progress in dedicated straight-line environments.

    Body-frame forward speed cannot distinguish a straight run from a wide arc.
    The forward-only command population starts at world heading zero, so world-X
    velocity gives PPO a dense signal for the behavior used by the native speed
    test without applying that constraint to turning environments.
    """
    term = env.command_manager.get_term("twist")
    active = term.is_forward_env & ~term.is_standing_env
    speed = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
    speed = torch.nan_to_num(speed, nan=0.0).clamp(-safety_cap_mps, safety_cap_mps)
    return active * speed


def forward_path_error_cost(
    env: "ManagerBasedRlEnv",
    lateral_cap_m: float = 3.0,
    heading_cap_rad: float = 1.5707963267948966,
) -> torch.Tensor:
    """Penalize accumulated lane and heading error for straight-line episodes."""
    term = env.command_manager.get_term("twist")
    active = term.is_forward_env & ~term.is_standing_env
    robot = env.scene["robot"].data
    # Vectorized scenes are tiled across a world-space grid.  Measure lane
    # deviation from each environment's own origin, not from global y=0.
    lateral_from_lane = robot.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    lateral = torch.nan_to_num(
        lateral_from_lane, nan=0.0,
        posinf=lateral_cap_m, neginf=-lateral_cap_m,
    ).clamp(-lateral_cap_m, lateral_cap_m)
    heading = torch.nan_to_num(
        robot.heading_w, nan=0.0,
        posinf=heading_cap_rad, neginf=-heading_cap_rad,
    ).clamp(-heading_cap_rad, heading_cap_rad)
    return active * (lateral.square() + heading.square())


def forward_world_line_speed(
    env: "ManagerBasedRlEnv",
    safety_cap_mps: float = 2.5,
    lateral_std_m: float = 0.30,
    heading_std_rad: float = 0.15,
) -> torch.Tensor:
    """Pay only for forward speed that remains on the original world-space lane.

    This positive, bounded-quality objective cannot be improved by stopping or
    terminating an episode, unlike a large accumulated path-error penalty.
    """
    term = env.command_manager.get_term("twist")
    active = term.is_forward_env & ~term.is_standing_env
    robot = env.scene["robot"].data
    speed = torch.nan_to_num(
        robot.root_link_lin_vel_w[:, 0], nan=0.0
    ).clamp(0.0, safety_cap_mps)
    lateral = torch.nan_to_num(
        robot.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1], nan=0.0
    )
    heading = torch.nan_to_num(robot.heading_w, nan=0.0)
    quality_cost = (
        (lateral / lateral_std_m).square()
        + (heading / heading_std_rad).square()
    )
    quality = torch.exp(-torch.nan_to_num(quality_cost, nan=100.0, posinf=100.0))
    return active * speed * quality


def make_microduck_general_drive_env_cfg(play: bool = False):
    cfg = make_microduck_speed_discovery_env_cfg(play=play)
    roller_cfg = make_microduck_velocity_rollers_env_cfg(play=play)
    cfg.episode_length_s = float(os.environ.get("DUCKLAB_GENERAL_EPISODE_S", "20.0"))

    reset_pose = cfg.events["reset_base"].params["pose_range"]
    reset_pose["y"] = (0.0, 0.0)
    reset_pose["yaw"] = (0.0, 0.0)
    cfg.events["general_drive_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (
                float(os.environ.get("DUCKLAB_GENERAL_FRICTION", "0.003")),
                float(os.environ.get("DUCKLAB_GENERAL_FRICTION", "0.003")),
            ),
        },
    )

    command = cfg.commands["twist"]
    command.resampling_time_range = (
        float(os.environ.get("DUCKLAB_GENERAL_COMMAND_MIN_S", "3.0")),
        float(os.environ.get("DUCKLAB_GENERAL_COMMAND_MAX_S", "6.0")),
    )
    command.rel_standing_envs = float(os.environ.get("DUCKLAB_GENERAL_STAND_FRACTION", "0.25"))
    command.rel_forward_envs = float(os.environ.get("DUCKLAB_GENERAL_STRAIGHT_FRACTION", "0.35"))
    command.rel_turn_in_place_envs = float(os.environ.get("DUCKLAB_GENERAL_TURN_FRACTION", "0.20"))
    command.ranges.lin_vel_x = (
        float(os.environ.get("DUCKLAB_GENERAL_LIN_MIN", "0.10")),
        float(os.environ.get("DUCKLAB_GENERAL_LIN_MAX", "0.80")),
    )
    command.ranges.lin_vel_y = (0.0, 0.0)
    yaw_limit = float(os.environ.get("DUCKLAB_GENERAL_YAW_LIMIT", "0.30"))
    command.ranges.ang_vel_z = (-yaw_limit, yaw_limit)
    if os.environ.get("DUCKLAB_GENERAL_RACELINE", "0").lower() in {
        "1", "true", "yes",
    }:
        race_line_fields = {
            field.name for field in dataclasses.fields(
                microduck_mdp.RaceLineVelocityCommandCfg
            )
        }
        race_line_kwargs = {
            name: value for name, value in vars(command).items()
            if name in race_line_fields
            and name not in {"yaw_kp", "lateral_kp", "yaw_kd", "max_correction"}
        }
        cfg.commands["twist"] = microduck_mdp.RaceLineVelocityCommandCfg(
            **race_line_kwargs,
            yaw_kp=float(os.environ.get("DUCKLAB_GENERAL_LINE_YAW_KP", "0.55")),
            lateral_kp=float(
                os.environ.get("DUCKLAB_GENERAL_LINE_LATERAL_KP", "0.10")
            ),
            yaw_kd=float(os.environ.get("DUCKLAB_GENERAL_LINE_YAW_KD", "0.08")),
            max_correction=float(
                os.environ.get("DUCKLAB_GENERAL_LINE_MAX_CORRECTION", "0.18")
            ),
        )

    retained = {
        name: deepcopy(roller_cfg.rewards[name])
        for name in (
            "pose",
            "upright",
            "body_ang_vel",
            "action_rate_l2",
            "self_collisions",
            "action_over_limit",
            "joint_torques_l2",
            "neck_action_rate_l2",
            "neck_joint_pos_l2",
        )
    }
    alive = deepcopy(cfg.rewards["alive"])
    fall = deepcopy(cfg.rewards["fall"])
    cfg.rewards.clear()
    cfg.rewards.update(retained)
    cfg.rewards["alive"] = alive
    cfg.rewards["fall"] = fall
    cfg.rewards["pose"].weight = 2.5
    cfg.rewards["upright"].weight = 6.0
    cfg.rewards["body_ang_vel"].weight = -0.08
    cfg.rewards["action_rate_l2"].weight = -0.35
    cfg.rewards["joint_torques_l2"].weight = -8.0e-4
    cfg.rewards["command_progress"] = RewardTermCfg(func=command_progress, weight=3.0)
    cfg.rewards["low_speed_tracking"] = RewardTermCfg(func=low_speed_tracking, weight=6.0)
    cfg.rewards["yaw_rate_tracking"] = RewardTermCfg(func=yaw_rate_tracking, weight=5.0)
    cfg.rewards["yaw_rate_error"] = RewardTermCfg(
        func=yaw_rate_error_cost,
        weight=float(os.environ.get("DUCKLAB_GENERAL_YAW_ERROR_WEIGHT", "-2.0")),
    )
    cfg.rewards["stand_still"] = RewardTermCfg(func=stand_still, weight=8.0)
    cfg.rewards["stand_motion"] = RewardTermCfg(func=stand_motion_cost, weight=-12.0)
    cfg.rewards["stand_action"] = RewardTermCfg(
        func=stand_action_cost,
        weight=float(os.environ.get("DUCKLAB_GENERAL_STAND_ACTION_WEIGHT", "-4.0")),
    )
    cfg.rewards["high_straight_usable_speed"] = RewardTermCfg(
        func=high_straight_usable_speed,
        weight=float(os.environ.get("DUCKLAB_GENERAL_HIGH_SPEED_WEIGHT", "10.0")),
    )
    cfg.rewards["moving_lateral_velocity"] = RewardTermCfg(
        func=moving_lateral_velocity_cost,
        weight=-8.0,
    )
    cfg.rewards["forward_world_progress"] = RewardTermCfg(
        func=forward_world_progress,
        weight=float(os.environ.get("DUCKLAB_GENERAL_WORLD_PROGRESS_WEIGHT", "0.0")),
    )
    cfg.rewards["forward_path_error"] = RewardTermCfg(
        func=forward_path_error_cost,
        weight=float(os.environ.get("DUCKLAB_GENERAL_PATH_ERROR_WEIGHT", "0.0")),
    )
    cfg.rewards["forward_world_line_speed"] = RewardTermCfg(
        func=forward_world_line_speed,
        weight=float(os.environ.get("DUCKLAB_GENERAL_WORLD_LINE_WEIGHT", "0.0")),
    )
    cfg.curriculum.clear()
    return cfg


MicroduckGeneralDriveRlCfg = dataclasses.replace(
    MicroduckSpeedDiscoveryRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedDiscoveryRlCfg.algorithm,
        class_name="mjlab_microduck.algorithms.donor_anchored_ppo:DonorAnchoredPPO",
        learning_rate=float(os.environ.get("DUCKLAB_GENERAL_LR", "2e-6")),
        schedule="fixed",
        desired_kl=float(os.environ.get("DUCKLAB_GENERAL_KL", "1e-4")),
        clip_param=float(os.environ.get("DUCKLAB_GENERAL_CLIP", "0.02")),
        entropy_coef=float(os.environ.get("DUCKLAB_GENERAL_ENTROPY", "0.002")),
        num_learning_epochs=int(os.environ.get("DUCKLAB_GENERAL_EPOCHS", "2")),
        num_mini_batches=int(os.environ.get("DUCKLAB_GENERAL_MINI_BATCHES", "8")),
        symmetry_cfg=(
            {
                **SYMMETRY_CFG,
                "use_data_augmentation": True,
                "mirror_loss_coeff": float(
                    os.environ.get("DUCKLAB_GENERAL_SYMMETRY_COEFF", "1.0")
                ),
            }
            if os.environ.get("DUCKLAB_GENERAL_SYMMETRY", "0").lower()
            in {"1", "true", "yes"}
            else None
        ),
    ),
    experiment_name="microduck_general_drive",
    run_name="microduck_general_drive",
    save_interval=10,
    num_steps_per_env=48,
    max_iterations=8_000,
)
