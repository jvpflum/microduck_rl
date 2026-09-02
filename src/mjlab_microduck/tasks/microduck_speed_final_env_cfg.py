"""V23: exact-V59, official-friction final speed-branch optimization.

Earlier V19/V21/V22 experiments initialized a different actor and asked a
teacher loss to reconstruct V59 while PPO changed it.  The deployment replay
showed that this loses the valuable high-speed gait.  V23 instead imports the
deployed V59 actor and observation normalizer exactly into a resumable
checkpoint, starts PPO at zero teacher error, and searches locally around that
policy.

This remains a replaceable high-command specialist.  The qualified V66 router
continues to own idle, cruise, turning, and braking.  Training therefore uses
only the straight race-effort command and concentrates its gradient budget on
speed, launch, line quality, and upright survival.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from copy import deepcopy

import torch

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.actuator import FrictionDRBamActuatorCfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_discovery_env_cfg import (
    SPEED_DISCOVERY_CAP_MPS,
)
from mjlab_microduck.tasks.microduck_speed_frontier_env_cfg import (
    OFFICIAL_WHEEL_FRICTION,
    WHEEL_RADIUS_M,
    XL330_CURRENT_LIMIT_A,
    frontier_final_speed,
    frontier_lateral_speed_squared,
    frontier_speed_gain,
    frontier_world_speed,
    frontier_world_speed_squared,
    reset_frontier_state,
)
from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
)


FIVE_MPH_MPS = 5.0 * 0.44704
EXACT_V59_CHECKPOINT = (
    "/home/juice/projects/microduck-lab/upstream/microduck_rl/"
    "protected_checkpoints/duckwing_v23/v59-exact-import.pt"
)


def _course_state(env):
    asset = env.scene["robot"]
    speed = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 0],
        nan=0.0,
        posinf=SPEED_DISCOVERY_CAP_MPS,
        neginf=-SPEED_DISCOVERY_CAP_MPS,
    ).clamp(-SPEED_DISCOVERY_CAP_MPS, SPEED_DISCOVERY_CAP_MPS)
    lateral_error = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    lateral_speed = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 1], nan=0.0)
    quat = asset.data.root_link_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    up_z = torch.clamp(1.0 - 2.0 * (x.square() + y.square()), -1.0, 1.0)
    tilt = torch.acos(up_z)
    return speed, lateral_error, lateral_speed, yaw, tilt


def final_usable_speed(
    env,
    target_speed_mps: float = FIVE_MPH_MPS,
    lane_std_m: float = 0.24,
    lateral_speed_std_mps: float = 0.22,
    heading_std_rad: float = 0.20,
    tilt_std_rad: float = 0.32,
):
    """Quadratic speed paid only while the chassis is usable and stable."""
    speed, lateral_error, lateral_speed, yaw, tilt = _course_state(env)
    forward = torch.clamp(speed, min=0.0)
    quality_cost = (
        (lateral_error / lane_std_m).square()
        + (lateral_speed / lateral_speed_std_mps).square()
        + (yaw / heading_std_rad).square()
        + (tilt / tilt_std_rad).square()
    )
    quality = torch.exp(-torch.nan_to_num(quality_cost, nan=100.0, posinf=100.0))
    return torch.nan_to_num((forward / target_speed_mps).square() * quality, nan=0.0)


def final_usable_launch(
    env,
    target_speed_mps: float = FIVE_MPH_MPS,
    window_s: float = 1.5,
):
    """Dense launch pressure with the same straight/upright legality gate."""
    speed, lateral_error, lateral_speed, yaw, tilt = _course_state(env)
    age_s = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
    early = torch.clamp(1.0 - age_s / window_s, min=0.0)
    quality_cost = (
        (lateral_error / 0.20).square()
        + (lateral_speed / 0.20).square()
        + (yaw / 0.16).square()
        + (tilt / 0.34).square()
    )
    quality = torch.exp(-torch.nan_to_num(quality_cost, nan=100.0, posinf=100.0))
    return early * torch.clamp(speed, min=0.0) / target_speed_mps * quality


def final_tilt_squared(env, reference_rad: float = 0.30):
    """Small standalone pressure against a high-speed fall trajectory."""
    *_, tilt = _course_state(env)
    return torch.nan_to_num((tilt / reference_rad).square(), nan=100.0)


@dataclass
class ExactV59FinalPpoCfg(RslRlPpoAlgorithmCfg):
    teacher_checkpoint: str = EXACT_V59_CHECKPOINT
    # Start at zero imitation error and allow a bounded departure from V59.
    teacher_loss_coef: float = 0.40
    teacher_loss_decay: float = 0.9995
    teacher_loss_floor: float = 0.08
    probe_loss_share: float = 0.0
    speed_command_threshold: float = 0.5
    smooth_turn_start: float = 0.08
    smooth_turn_end: float = 0.25
    command_x_index: int = 48
    command_y_index: int = 49
    command_yaw_index: int = 50


def make_microduck_speed_final_env_cfg(play: bool = False):
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.episode_length_s = 20.0

    # Exact deployment actuator ceiling and fixed nominal delay.  The final
    # candidate is selected under this same 1.75 A / 0.003-friction contract.
    robot = deepcopy(MICRODUCK_WALK_ROLLERS_ROBOT_CFG)
    robot.articulation.actuators = (
        FrictionDRBamActuatorCfg(
            motor_name="xl330",
            model="m6",
            target_names_expr=(r"^(?!passive_).*",),
            kp_fw=200.0,
            vin=7.4,
            vin_range=None,
            vin_drop_gain_range=None,
            vin_min=None,
            max_current=XL330_CURRENT_LIMIT_A,
            delay_min_lag=3,
            delay_max_lag=3,
        ),
    )
    cfg.scene.entities["robot"] = robot
    cfg.events["final_official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^passive_.*wheel$",)
            ),
            "operation": "abs",
            "ranges": (OFFICIAL_WHEEL_FRICTION, OFFICIAL_WHEEL_FRICTION),
        },
    )
    # Reverse curriculum: retain plenty of rest launches, while exposing the
    # actor to 0.7--2.6 m/s states it otherwise almost never visits on-policy.
    cfg.events["reset_final_speed_state"] = EventTermCfg(
        func=reset_frontier_state,
        mode="reset",
        params={
            "frequency_range_hz": (2.5, 4.5),
            "bootstrap_speed_range_mps": (0.70, 2.60),
            "bootstrap_fraction_stages": (
                (0, 0.40),
                (1_500 * 24, 0.30),
                (3_500 * 24, 0.22),
            ),
            "wheel_radius_m": WHEEL_RADIUS_M,
        },
    )

    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.80, 0.80)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.15, 0.15)
    command.yaw_kp = 0.70
    command.lateral_kp = 0.14
    command.yaw_kd = 0.07
    command.max_correction = 0.15

    # Keep the discovery surface small.  Speed and speed retention dominate;
    # the multiplicative usable term makes tilt/drift reduction valuable only
    # in conjunction with motion, avoiding a stand-still optimum.
    retained = {
        name: deepcopy(cfg.rewards[name])
        for name in ("alive", "fall", "self_collisions", "action_over_limit", "action_rate_l2")
    }
    cfg.rewards.clear()
    cfg.rewards["world_speed"] = RewardTermCfg(func=frontier_world_speed, weight=5.0)
    cfg.rewards["world_speed_squared"] = RewardTermCfg(
        func=frontier_world_speed_squared, weight=2.0
    )
    cfg.rewards["speed_gain"] = RewardTermCfg(func=frontier_speed_gain, weight=5.0)
    cfg.rewards["final_speed"] = RewardTermCfg(func=frontier_final_speed, weight=4.0)
    cfg.rewards["usable_speed"] = RewardTermCfg(func=final_usable_speed, weight=30.0)
    cfg.rewards["usable_launch"] = RewardTermCfg(func=final_usable_launch, weight=18.0)
    cfg.rewards["lateral_speed"] = RewardTermCfg(
        func=frontier_lateral_speed_squared, weight=-0.50
    )
    cfg.rewards["tilt_cost"] = RewardTermCfg(func=final_tilt_squared, weight=-0.20)
    for name, weight in (
        ("alive", 0.25),
        ("fall", -1000.0),
        ("self_collisions", -0.08),
        ("action_over_limit", -0.12),
        ("action_rate_l2", -0.04),
    ):
        term = retained[name]
        term.weight = weight
        cfg.rewards[name] = term
    cfg.curriculum.clear()
    return cfg


_distribution = dict(MicroduckSpeedStraighteningRlCfg.actor.distribution_cfg)
_distribution["init_std"] = 0.06

MicroduckSpeedFinalRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    actor=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.actor,
        distribution_cfg=_distribution,
    ),
    algorithm=ExactV59FinalPpoCfg(
        class_name="mjlab_microduck.teacher_guided_ppo.TeacherGuidedPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        entropy_coef=0.0005,
        learning_rate=7.5e-7,
        desired_kl=5.0e-5,
        clip_param=0.008,
        num_learning_epochs=2,
        num_mini_batches=4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        max_grad_norm=0.75,
    ),
    experiment_name="microduck_speed_final",
    run_name="microduck_speed_final",
    save_interval=5,
    num_steps_per_env=24,
    max_iterations=5_000,
)
