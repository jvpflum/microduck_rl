"""Official-friction, state-feedback residual frontier for MicroDuck Race5.

This stage starts exactly from the preserved V26 actor and trains only a bounded
neural correction.  Short episodes split between rest and physically consistent
rolling entries expose acceleration and high-speed balance without allowing the
policy to depend on an artificial launch.  The dominant reward is usable world-X
speed: speed is valuable only while lane, heading, and lateral motion stay sane.
"""

from __future__ import annotations

import dataclasses
import math
import os

import torch

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
    world_forward_velocity,
)
from mjlab_microduck.tasks.microduck_velocity_race5_constrained_env_cfg import (
    race_usable_launch_speed,
    race_usable_speed_squared,
)


def speed_gated_upright(
    env,
    reference_speed_mps: float = 0.60,
    upright_std_rad: float = 0.20,
    safety_cap_mps: float = 7.5,
):
    """Reward upright travel without making standing still an attractive fix."""
    speed = world_forward_velocity(env, safety_cap_mps=safety_cap_mps)
    speed_gate = torch.clamp(speed / reference_speed_mps, 0.0, 1.0)
    upright = microduck_mdp.body_upright_gaussian(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        std=upright_std_rad,
    )
    return speed_gate * upright


def speed_gated_body_angular_velocity(
    env,
    reference_speed_mps: float = 0.60,
    safety_cap_mps: float = 7.5,
):
    """Penalize roll/pitch wobble only while making forward progress."""
    speed = world_forward_velocity(env, safety_cap_mps=safety_cap_mps)
    speed_gate = torch.clamp(speed / reference_speed_mps, 0.0, 1.0)
    angular_velocity = env.scene["robot"].data.root_link_ang_vel_w[:, :2]
    angular_cost = torch.nan_to_num(angular_velocity, nan=0.0).square().sum(dim=1)
    return speed_gate * angular_cost


def speed_gated_lateral_excess(
    env,
    reference_speed_mps: float = 0.60,
    max_lateral_speed_mps: float = 0.05,
    safety_cap_mps: float = 7.5,
):
    """Charge only lateral motion beyond the Race5 qualification boundary."""
    if max_lateral_speed_mps <= 0.0:
        raise ValueError("max_lateral_speed_mps must be positive")
    speed = world_forward_velocity(env, safety_cap_mps=safety_cap_mps)
    speed_gate = torch.clamp(speed / reference_speed_mps, 0.0, 1.0)
    lateral_speed = torch.nan_to_num(
        env.scene["robot"].data.root_link_lin_vel_w[:, 1], nan=0.0
    ).abs()
    relative_excess = torch.relu(
        lateral_speed - max_lateral_speed_mps
    ) / max_lateral_speed_mps
    return speed_gate * relative_excess


def speed_gated_tilt_excess(
    env,
    reference_speed_mps: float = 0.60,
    max_tilt_deg: float = 16.0,
    safety_cap_mps: float = 7.5,
):
    """Charge tilt only above a guardrail, while forward speed remains useful."""
    max_tilt_rad = math.radians(max_tilt_deg)
    if max_tilt_rad <= 0.0:
        raise ValueError("max_tilt_deg must be positive")
    speed = world_forward_velocity(env, safety_cap_mps=safety_cap_mps)
    speed_gate = torch.clamp(speed / reference_speed_mps, 0.0, 1.0)
    quat = env.scene["robot"].data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    tilt = torch.acos(cos_tilt.clamp(-1.0, 1.0))
    relative_excess = torch.relu(tilt - max_tilt_rad) / max_tilt_rad
    return speed_gate * relative_excess


def make_microduck_speed_residual_frontier_env_cfg(play: bool = False):
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.episode_length_s = float(
        os.environ.get("DUCKLAB_RESIDUAL_EPISODE_LENGTH_S", "5.0")
    )
    hip_pitch_gain = float(
        os.environ.get("DUCKLAB_RESIDUAL_HIP_PITCH_GAIN", "1.0")
    )
    if not 0.5 <= hip_pitch_gain <= 1.5:
        raise ValueError("DUCKLAB_RESIDUAL_HIP_PITCH_GAIN must be in [0.5, 1.5]")
    cfg.actions["joint_pos"].scale = {
        r"^(left_hip_pitch|right_hip_pitch)$": hip_pitch_gain,
        r"^(left_ankle|right_ankle)$": 1.06,
    }

    # Discovery must rank policy changes against one repeatable hardware model.
    # Battery, voltage-drop, and command-delay randomization are deferred until
    # a candidate passes the nominal speed/line/survival gate.
    robot_cfg = cfg.scene.entities["robot"]
    nominal_actuators = tuple(
        dataclasses.replace(
            actuator,
            vin_range=(7.4, 7.4),
            vin_drop_gain_range=(0.10, 0.10),
            delay_min_lag=4,
            delay_max_lag=4,
        )
        for actuator in robot_cfg.articulation.actuators
    )
    cfg.scene.entities["robot"] = dataclasses.replace(
        robot_cfg,
        articulation=dataclasses.replace(
            robot_cfg.articulation,
            actuators=nominal_actuators,
        ),
    )

    cfg.events["official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^passive_.*wheel$",)
            ),
            "operation": "abs",
            "ranges": (0.003, 0.003),
        },
    )
    # Half of resets stay untouched and must accelerate from rest.  The other
    # half sample coasting states with wheel omega=v/r so PPO learns recovery and
    # propulsion at speeds that ordinary exploratory launches rarely reach.
    rolling_fraction = float(
        os.environ.get("DUCKLAB_RESIDUAL_ROLLING_FRACTION", "0.50")
    )
    if not 0.0 <= rolling_fraction <= 1.0:
        raise ValueError("DUCKLAB_RESIDUAL_ROLLING_FRACTION must be in [0, 1]")
    rolling_min_mps = float(
        os.environ.get("DUCKLAB_RESIDUAL_ROLLING_MIN_MPS", "0.50")
    )
    rolling_max_mps = float(
        os.environ.get("DUCKLAB_RESIDUAL_ROLLING_MAX_MPS", "1.20")
    )
    if rolling_min_mps < 0.0 or rolling_max_mps <= rolling_min_mps:
        raise ValueError(
            "DUCKLAB_RESIDUAL_ROLLING_MIN_MPS/MAX_MPS must define a "
            "non-negative increasing range"
        )
    if rolling_max_mps > 7.5:
        raise ValueError("DUCKLAB_RESIDUAL_ROLLING_MAX_MPS must not exceed 7.5")
    if rolling_fraction > 0.0:
        cfg.events["rolling_state_exposure"] = EventTermCfg(
            func=microduck_mdp.reset_with_forward_velocity,
            mode="reset",
            params={
                "velocity_range": (rolling_min_mps, rolling_max_mps),
                "fraction_stages": [
                    {"step": 0, "fraction": rolling_fraction}
                ],
            },
        )

    reset_pose = cfg.events["reset_base"].params["pose_range"]
    reset_pose["y"] = (-0.03, 0.03)
    reset_pose["yaw"] = (-0.035, 0.035)

    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.80, 0.80)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.18, 0.18)
    # Match the deployed Race5 evaluator exactly.  A policy trained with more
    # steering authority can appear straight in PPO and then drift in the app.
    command.yaw_kp = float(
        os.environ.get("DUCKLAB_RESIDUAL_LINE_YAW_KP", "0.55")
    )
    command.lateral_kp = float(
        os.environ.get("DUCKLAB_RESIDUAL_LINE_LATERAL_KP", "0.10")
    )
    command.yaw_kd = float(
        os.environ.get("DUCKLAB_RESIDUAL_LINE_YAW_KD", "0.08")
    )
    command.max_correction = float(
        os.environ.get("DUCKLAB_RESIDUAL_LINE_MAX_WZ", "0.18")
    )

    cfg.curriculum.clear()
    cfg.rewards["speed_target_progress"].params["target_speed_mps"] = 2.2352
    cfg.rewards["forward_velocity_mps"].weight = 1.0
    cfg.rewards["forward_velocity_squared"].weight = 0.10
    cfg.rewards["world_forward_velocity_mps"].weight = 5.0
    cfg.rewards["world_forward_velocity_squared"].weight = 1.0
    cfg.rewards["heading_hold"].weight = float(
        os.environ.get("DUCKLAB_RESIDUAL_HEADING_HOLD_WEIGHT", "1.0")
    )
    cfg.rewards["lane_error"].weight = float(
        os.environ.get("DUCKLAB_RESIDUAL_LANE_WEIGHT", "-1.0")
    )
    cfg.rewards["world_lateral_velocity"].weight = float(
        os.environ.get("DUCKLAB_RESIDUAL_LATERAL_WEIGHT", "-1.5")
    )
    cfg.rewards["heading_error"].weight = float(
        os.environ.get("DUCKLAB_RESIDUAL_HEADING_ERROR_WEIGHT", "-1.5")
    )
    cfg.rewards["usable_speed"] = RewardTermCfg(
        func=race_usable_speed_squared,
        weight=float(
            os.environ.get("DUCKLAB_RESIDUAL_USABLE_SPEED_WEIGHT", "12.0")
        ),
        params={
            "reference_speed_mps": 0.60,
            "safety_cap_mps": 7.5,
            "lane_std_m": float(
                os.environ.get("DUCKLAB_RESIDUAL_LANE_STD_M", "0.30")
            ),
            "heading_std_rad": float(
                os.environ.get("DUCKLAB_RESIDUAL_HEADING_STD_RAD", "0.18")
            ),
            "lateral_speed_std_mps": float(
                os.environ.get(
                    "DUCKLAB_RESIDUAL_LATERAL_SPEED_STD_MPS", "0.20"
                )
            ),
        },
    )
    cfg.rewards["usable_launch"] = RewardTermCfg(
        func=race_usable_launch_speed,
        weight=float(
            os.environ.get("DUCKLAB_RESIDUAL_LAUNCH_WEIGHT", "5.0")
        ),
        params={
            "reference_speed_mps": 0.60,
            "window_s": 2.0,
            "safety_cap_mps": 7.5,
            "lane_std_m": float(
                os.environ.get("DUCKLAB_RESIDUAL_LANE_STD_M", "0.30")
            ),
            "heading_std_rad": float(
                os.environ.get("DUCKLAB_RESIDUAL_HEADING_STD_RAD", "0.18")
            ),
            "lateral_speed_std_mps": float(
                os.environ.get(
                    "DUCKLAB_RESIDUAL_LATERAL_SPEED_STD_MPS", "0.20"
                )
            ),
        },
    )
    cfg.rewards["upright_at_speed"] = RewardTermCfg(
        func=speed_gated_upright,
        weight=float(
            os.environ.get("DUCKLAB_RESIDUAL_UPRIGHT_AT_SPEED_WEIGHT", "0.0")
        ),
        params={
            "reference_speed_mps": 0.60,
            "upright_std_rad": float(
                os.environ.get("DUCKLAB_RESIDUAL_UPRIGHT_STD_RAD", "0.20")
            ),
            "safety_cap_mps": 7.5,
        },
    )
    cfg.rewards["angular_velocity_at_speed"] = RewardTermCfg(
        func=speed_gated_body_angular_velocity,
        weight=float(
            os.environ.get("DUCKLAB_RESIDUAL_ANGULAR_VELOCITY_WEIGHT", "0.0")
        ),
        params={
            "reference_speed_mps": 0.60,
            "safety_cap_mps": 7.5,
        },
    )
    cfg.rewards["lateral_guardrail_at_speed"] = RewardTermCfg(
        func=speed_gated_lateral_excess,
        weight=float(
            os.environ.get("DUCKLAB_RESIDUAL_LATERAL_GUARDRAIL_WEIGHT", "0.0")
        ),
        params={
            "reference_speed_mps": 0.60,
            "max_lateral_speed_mps": float(
                os.environ.get(
                    "DUCKLAB_RESIDUAL_MAX_LATERAL_SPEED_MPS", "0.05"
                )
            ),
            "safety_cap_mps": 7.5,
        },
    )
    cfg.rewards["tilt_guardrail_at_speed"] = RewardTermCfg(
        func=speed_gated_tilt_excess,
        weight=float(
            os.environ.get("DUCKLAB_RESIDUAL_TILT_GUARDRAIL_WEIGHT", "0.0")
        ),
        params={
            "reference_speed_mps": 0.60,
            "max_tilt_deg": float(
                os.environ.get("DUCKLAB_RESIDUAL_MAX_TILT_DEG", "16.0")
            ),
            "safety_cap_mps": 7.5,
        },
    )
    cfg.rewards["fall"].weight = float(
        os.environ.get("DUCKLAB_RESIDUAL_FALL_WEIGHT", "-750.0")
    )
    cfg.rewards["alive"].weight = 0.50
    return cfg


MicroduckSpeedResidualFrontierRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    actor=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.actor,
        class_name=(
            "mjlab_microduck.algorithms.residual_frontier_ppo:ResidualMLPModel"
        ),
    ),
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        class_name="mjlab_microduck.algorithms.residual_frontier_ppo:ResidualPPO",
        learning_rate=float(os.environ.get("DUCKLAB_RESIDUAL_LR", "1e-4")),
        schedule="fixed",
        desired_kl=0.01,
        clip_param=0.10,
        entropy_coef=0.001,
        num_learning_epochs=4,
        num_mini_batches=8,
        symmetry_cfg=None,
    ),
    experiment_name="microduck_speed_residual_frontier",
    run_name="microduck_speed_residual_frontier",
    save_interval=10,
    num_steps_per_env=24,
    max_iterations=2_000,
)
