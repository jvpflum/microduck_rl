"""Race-v1: fixed-distance, straight-line roller racing for MicroDuck.

Race-v1 is deliberately separate from Sprint. Sprint tracks a requested speed;
Race rewards finishing a five-metre lane as quickly as possible while remaining
upright, straight, and deployable. Candidate policies compete in evaluation
heats, not through a non-stationary cross-environment PPO reward.
"""

import dataclasses
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.microduck_velocity_sprint_env_cfg import (
    MicroduckSprintRlCfg,
    make_microduck_velocity_sprint_env_cfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


RACE_DISTANCE_M = 5.0
RACE_SPEED_REFERENCE_MPS = 0.55
RACE_SPEED_SAFETY_CAP_MPS = 1.20


def race_speed_squared(
    env: "ManagerBasedRlEnv",
    reference_speed: float = RACE_SPEED_REFERENCE_MPS,
    safety_cap: float = RACE_SPEED_SAFETY_CAP_MPS,
) -> torch.Tensor:
    """Reward forward world-speed quadratically, without Sprint's target cap.

    For a fixed-distance episode, integrating linear velocity gives every
    finisher roughly the same return. Squared velocity instead gives a faster
    finisher more return over the same distance. The safety cap prevents an
    unstable simulator launch from becoming an unbounded reward exploit.
    """
    speed = torch.clamp(
        env.scene["robot"].data.root_link_lin_vel_w[:, 0],
        min=-safety_cap,
        max=safety_cap,
    )
    normalized = speed / reference_speed
    # Preserve direction. A circular policy previously farmed positive half-
    # laps because negative world-X velocity was clipped to zero before the
    # square. Signed quadratic return makes a loop average near zero while a
    # genuinely straight sprint keeps the full speed incentive.
    return torch.nan_to_num(normalized * torch.abs(normalized), nan=0.0)


def race_forward_progress_rate(
    env: "ManagerBasedRlEnv",
    reference_speed: float = RACE_SPEED_REFERENCE_MPS,
    safety_cap: float = RACE_SPEED_SAFETY_CAP_MPS,
) -> torch.Tensor:
    """Dense signed world-X progress rate toward the fixed finish line."""
    speed = torch.clamp(
        env.scene["robot"].data.root_link_lin_vel_w[:, 0],
        min=-safety_cap,
        max=safety_cap,
    )
    return torch.nan_to_num(speed / reference_speed, nan=0.0)


def race_speed_target_progress(
    env: "ManagerBasedRlEnv",
    target_speed_mps: float,
    safety_cap: float = 4.4704,
) -> torch.Tensor:
    """Dense 0..1 progress toward a measured world-forward speed milestone."""
    speed = torch.clamp(
        env.scene["robot"].data.root_link_lin_vel_w[:, 0],
        min=0.0,
        max=safety_cap,
    )
    target = max(float(target_speed_mps), 0.05)
    return torch.nan_to_num(torch.clamp(speed / target, max=1.0), nan=0.0)


def race_lane_error_squared(
    env: "ManagerBasedRlEnv", lane_half_width_m: float = 0.50
) -> torch.Tensor:
    """Bounded, normalized lane penalty that cannot swamp every reward."""
    robot_y = env.scene["robot"].data.root_link_pos_w[:, 1]
    lane_y = env.scene.env_origins[:, 1]
    normalized = torch.clamp(
        torch.abs(robot_y - lane_y) / lane_half_width_m, max=1.0
    )
    return torch.nan_to_num(normalized.square(), nan=0.0, posinf=1.0)


def race_elapsed_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Constant per-step cost: reaching the finish sooner costs less."""
    return torch.ones(env.num_envs, device=env.device)


def race_finished(
    env: "ManagerBasedRlEnv", distance_m: float = RACE_DISTANCE_M
) -> torch.Tensor:
    """True once the trunk crosses the finish line in world +X."""
    robot_x = env.scene["robot"].data.root_link_pos_w[:, 0]
    start_x = env.scene.env_origins[:, 0]
    return torch.nan_to_num(robot_x - start_x, nan=-1.0) >= distance_m


def race_out_of_lane(
    env: "ManagerBasedRlEnv", lane_half_width_m: float = 0.50
) -> torch.Tensor:
    robot_y = env.scene["robot"].data.root_link_pos_w[:, 1]
    lane_y = env.scene.env_origins[:, 1]
    return torch.abs(robot_y - lane_y) >= lane_half_width_m


def race_heading_departure(
    env: "ManagerBasedRlEnv", maximum_yaw_rad: float = 0.785398
) -> torch.Tensor:
    quat = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.abs(yaw) >= maximum_yaw_rad


def race_heading_error_squared(
    env: "ManagerBasedRlEnv", maximum_yaw_rad: float = 0.523599
) -> torch.Tensor:
    """Smooth 0..1 heading cost with useful gradient before termination."""
    quat = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    normalized = torch.clamp(torch.abs(yaw) / maximum_yaw_rad, max=1.0)
    return torch.nan_to_num(normalized.square(), nan=1.0, posinf=1.0)


def race_invalid_heat(
    env: "ManagerBasedRlEnv",
    lane_half_width_m: float = 0.40,
    maximum_yaw_rad: float = 0.523599,
) -> torch.Tensor:
    """Charge an explicit terminal cost for leaving the legal race corridor."""
    invalid = race_out_of_lane(env, lane_half_width_m=lane_half_width_m)
    invalid |= race_heading_departure(env, maximum_yaw_rad=maximum_yaw_rad)
    return invalid.to(dtype=torch.float32)


def race_finish_bonus(
    env: "ManagerBasedRlEnv", distance_m: float = RACE_DISTANCE_M
) -> torch.Tensor:
    return race_finished(env, distance_m=distance_m).to(dtype=torch.float32)


def make_microduck_velocity_race_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Race-v1 five-metre straight-line roller environment."""
    cfg = make_microduck_velocity_sprint_env_cfg(play=play)
    cfg.episode_length_s = 14.0

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    # The command remains in the 61D deployable observation, but Race-v1 is not
    # target tracking. It indicates full race effort over a narrow curriculum.
    command.ranges.lin_vel_x = (0.55, 0.80)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    # A race lane is world +X. Unlike general velocity training, every heat
    # starts facing the same finish line.
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

    for name in ("track_linear_velocity", "forward_speed_progress", "coasting"):
        cfg.rewards.pop(name, None)

    cfg.rewards["race_speed_squared"] = RewardTermCfg(
        func=race_speed_squared,
        weight=10.0,
        params={
            "reference_speed": RACE_SPEED_REFERENCE_MPS,
            "safety_cap": RACE_SPEED_SAFETY_CAP_MPS,
        },
    )
    cfg.rewards["race_finish"] = RewardTermCfg(
        func=race_finish_bonus,
        weight=50.0,
        params={"distance_m": RACE_DISTANCE_M},
    )
    cfg.rewards["race_elapsed"] = RewardTermCfg(
        func=race_elapsed_cost,
        weight=-0.5,
    )
    cfg.rewards["race_lane_error"] = RewardTermCfg(
        func=race_lane_error_squared,
        weight=-8.0,
        params={"lane_half_width_m": 0.50},
    )

    # Preserve Sprint-v3's gait while tightening the failure modes that become
    # dangerous as speed rises. Wheel speed is only a weak discovery aid;
    # chassis progress, not passive-wheel spin, decides the race.
    cfg.rewards["wheel_speed"].weight = 2.0
    cfg.rewards["upright"].weight = 6.0
    cfg.rewards["upright"].params["std"] = 0.18
    cfg.rewards["heading_hold"].weight = 5.0
    cfg.rewards["lateral_speed"].weight = -3.0
    cfg.rewards["body_ang_vel"].weight = -0.10
    cfg.rewards["action_rate_l2"].weight = -1.2
    cfg.rewards["joint_torques_l2"].weight = -1.5e-3

    cfg.terminations["race_finished"] = TerminationTermCfg(
        func=race_finished,
        params={"distance_m": RACE_DISTANCE_M},
        time_out=False,
    )
    # The inherited 1-radian fall termination remains active. Make the body
    # target explicit so later robot variants cannot silently change the gate.
    if "fell_over" in cfg.terminations:
        cfg.terminations["fell_over"].params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=("trunk_base",)
        )

    return cfg


MicroduckRaceRlCfg = dataclasses.replace(
    MicroduckSprintRlCfg,
    experiment_name="velocity_race",
    run_name="velocity_race",
    save_interval=5,
)
