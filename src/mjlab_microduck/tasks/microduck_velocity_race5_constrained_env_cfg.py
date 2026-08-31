"""V16: conservative V11 fine-tuning for faster *usable* skate racing.

This is deliberately not another raw-speed task.  It pays increasing
world-forward speed only while the duck stays centred, aligned with the course,
and avoids cross-track slide.  The same quantities remain independently
measured by the deployment evaluator; this in-training envelope merely makes
the PPO gradient agree with the A-to-B objective.
"""

import dataclasses
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks.microduck_velocity_race5_env_cfg import (
    RACE5_STRETCH_MPS,
    MicroduckRace5RlCfg,
    make_microduck_velocity_race5_env_cfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _race_line_quality(
    env: "ManagerBasedRlEnv",
    lane_std_m: float,
    heading_std_rad: float,
    lateral_speed_std_mps: float,
) -> torch.Tensor:
    """Return a smooth 0..1 racing-line quality factor in the world frame."""
    asset = env.scene["robot"]
    lateral_error = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    lateral_speed = asset.data.root_link_lin_vel_w[:, 1]
    quat = asset.data.root_link_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cost = (
        (lateral_error / lane_std_m).square()
        + (yaw / heading_std_rad).square()
        + (lateral_speed / lateral_speed_std_mps).square()
    )
    return torch.exp(-torch.nan_to_num(cost, nan=100.0, posinf=100.0))


def race_usable_speed_squared(
    env: "ManagerBasedRlEnv",
    reference_speed_mps: float = 0.55,
    safety_cap_mps: float = RACE5_STRETCH_MPS,
    lane_std_m: float = 0.20,
    heading_std_rad: float = 0.14,
    lateral_speed_std_mps: float = 0.18,
) -> torch.Tensor:
    """Quadratic speed return gated by a smooth, physically useful race line."""
    speed = torch.clamp(
        env.scene["robot"].data.root_link_lin_vel_w[:, 0],
        min=0.0,
        max=safety_cap_mps,
    )
    normalized = speed / reference_speed_mps
    quality = _race_line_quality(
        env, lane_std_m, heading_std_rad, lateral_speed_std_mps
    )
    return torch.nan_to_num(normalized.square() * quality, nan=0.0)


def race_usable_launch_speed(
    env: "ManagerBasedRlEnv",
    reference_speed_mps: float = 0.55,
    window_s: float = 2.0,
    safety_cap_mps: float = RACE5_STRETCH_MPS,
    lane_std_m: float = 0.20,
    heading_std_rad: float = 0.14,
    lateral_speed_std_mps: float = 0.18,
) -> torch.Tensor:
    """Pay early acceleration only when it builds a controllable race line."""
    speed = torch.clamp(
        env.scene["robot"].data.root_link_lin_vel_w[:, 0],
        min=0.0,
        max=safety_cap_mps,
    )
    age_s = env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)
    early = torch.clamp(1.0 - age_s / max(window_s, 0.05), min=0.0)
    quality = _race_line_quality(
        env, lane_std_m, heading_std_rad, lateral_speed_std_mps
    )
    return torch.nan_to_num((speed / reference_speed_mps) * quality * early, nan=0.0)


def make_microduck_velocity_race5_constrained_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the V16 V11-preserving drag-race fine-tuning environment."""
    cfg = make_microduck_velocity_race5_env_cfg(play=play)

    # V15's independent raw-speed terms can pay a launch that later needs a
    # large line correction.  Keep their useful dense gradient, but make the
    # dominant incremental incentive depend on usable world-forward velocity.
    cfg.rewards["race_speed_squared"].weight = 2.0
    cfg.rewards["race_forward_progress"].weight = 8.0
    cfg.rewards["race_launch_speed"].weight = 3.0
    cfg.rewards["race_usable_speed"] = RewardTermCfg(
        func=race_usable_speed_squared,
        weight=14.0,
    )
    cfg.rewards["race_usable_launch"] = RewardTermCfg(
        func=race_usable_launch_speed,
        weight=8.0,
    )

    # Time and a legal finish remain terminal priorities; no 5 mph hard gate
    # exists.  Improvement is selected outside PPO by the fixed V11 battery.
    cfg.rewards["race_finish"].weight = 650.0
    cfg.rewards["race_elapsed"].weight = -1.2
    cfg.rewards["race_lane_error"].weight = -18.0
    cfg.rewards["race_world_lateral_speed"].weight = -10.0
    cfg.rewards["race_heading_error"].weight = -18.0
    cfg.rewards["heading_hold"].weight = 14.0

    return cfg


MicroduckRace5ConstrainedRlCfg = dataclasses.replace(
    MicroduckRace5RlCfg,
    algorithm=dataclasses.replace(
        MicroduckRace5RlCfg.algorithm,
        # A true fine-tune: explore enough to improve the gait, but remain in
        # a tighter trust region around V11 than V15 did.
        learning_rate=2.0e-6,
        desired_kl=1.0e-4,
        clip_param=0.01,
        num_learning_epochs=1,
    ),
    experiment_name="velocity_race5_constrained",
    run_name="velocity_race5_constrained",
    save_interval=5,
)
