"""Unconstrained forward-speed discovery for the passive-wheel MicroDuck.

This task is deliberately separate from the deployable roller, Sprint, and
Race5 recipes.  It always uses MicroDuck's passive-roller model and skating
checkpoint; only the optimization pattern is adapted from HannesVonEssen's
running-policy work: flat nominal physics, actual chassis progress instead of
command tracking, weak action regularization, no pose imitation, no steering
objective, and no robustness disturbances until a fast skate gait exists.

The first phase is simulation research, not a hardware-ready policy.  A later
continuation must restore actuator/domain randomization and qualify control,
impacts, current, temperature, braking, and steering before deployment.
"""

from __future__ import annotations

import dataclasses
import os
from copy import deepcopy

from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, MetricsTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)


MPH_PER_MPS = 2.2369362921
SPEED_DISCOVERY_CAP_MPS = 7.5
SPEED_DISCOVERY_STAGES = (
    {
        "target_speed_mps": 2.5,
        "advance_mean_speed_mps": 2.0,
        "advance_survival_fraction": 0.80,
    },
    {
        "target_speed_mps": 3.5,
        "advance_mean_speed_mps": 2.8,
        "advance_survival_fraction": 0.80,
    },
    {
        "target_speed_mps": 4.5,
        "advance_mean_speed_mps": 3.6,
        "advance_survival_fraction": 0.82,
    },
    {
        "target_speed_mps": 5.5,
        "advance_mean_speed_mps": 4.4,
        "advance_survival_fraction": 0.85,
    },
    {
        "target_speed_mps": 6.7,
        # Final-stage thresholds are logged as the 15 mph qualification target;
        # there is no stage beyond this one.
        "advance_mean_speed_mps": 6.0,
        "advance_survival_fraction": 0.85,
    },
)


def make_microduck_speed_discovery_env_cfg(play: bool = False):
    """Create the nominal, speed-only roller discovery environment."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)
    cfg.episode_length_s = 20.0

    # Identical clean launch on a flat, effectively unbounded plane.  No random
    # yaw, initial tilt, position offset, or injected base velocity.
    reset_pose = cfg.events["reset_base"].params["pose_range"]
    reset_pose.update(
        {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.1385, 0.1385),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )
    cfg.events["reset_base"].params["velocity_range"] = {}

    # Discovery physics is nominal.  Retain only reset bookkeeping and BAM's
    # required per-world field expansion.  Robustness is a later continuation.
    essential_events = {
        "reset_base",
        "reset_robot_joints",
        "expand_bam_friction_fields",
        "reset_action_history",
    }
    for name in list(cfg.events):
        if name not in essential_events:
            cfg.events.pop(name)
    fixed_friction = os.environ.get("DUCKLAB_SPEED_DISCOVERY_FRICTION")
    if fixed_friction is not None:
        friction = float(fixed_friction)
        cfg.events["speed_discovery_wheel_friction"] = EventTermCfg(
            func=dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(r"^passive_.*wheel",)
                ),
                "operation": "abs",
                "ranges": (friction, friction),
            },
        )

    # Remove observation noise, sensor delay, encoder bias, and IMU mounting DR
    # while keeping the exact deployable 61D observation layout.
    actor_terms = cfg.observations["actor"].terms
    actor_terms["base_ang_vel"].func = mdp.base_ang_vel
    actor_terms["base_ang_vel"].params = {}
    actor_terms["projected_gravity"].func = mdp.projected_gravity
    actor_terms["projected_gravity"].params = {}
    for term in actor_terms.values():
        term.noise = None
        term.delay_min_lag = 0
        term.delay_max_lag = 0
        term.delay_update_period = 0
    actor_terms["joint_pos"].params["biased"] = False

    # Hannes's public recipe uses measured body-forward progress rather than
    # a Gaussian command tracker.  Linear speed is dominant; signed quadratic
    # pressure makes a 4 m/s solution more valuable than two 2 m/s intervals
    # while penalizing reverse motion.  Neither term saturates at the command.
    retained = {
        name: deepcopy(cfg.rewards[name])
        for name in ("self_collisions", "action_over_limit", "action_rate_l2")
    }
    cfg.rewards.clear()
    cfg.rewards["forward_velocity_mps"] = RewardTermCfg(
        func=microduck_mdp.speed_discovery_forward_velocity,
        weight=5.0,
        params={"safety_cap_mps": SPEED_DISCOVERY_CAP_MPS},
    )
    cfg.rewards["forward_velocity_squared"] = RewardTermCfg(
        func=microduck_mdp.speed_discovery_forward_velocity_squared,
        weight=0.75,
        params={"safety_cap_mps": SPEED_DISCOVERY_CAP_MPS},
    )
    cfg.rewards["speed_target_progress"] = RewardTermCfg(
        func=microduck_mdp.speed_discovery_target_progress,
        weight=1.0,
        params={"target_speed_mps": SPEED_DISCOVERY_STAGES[0]["target_speed_mps"]},
    )
    cfg.rewards["alive"] = RewardTermCfg(
        func=microduck_mdp.speed_discovery_alive,
        weight=0.25,
    )
    cfg.rewards["fall"] = RewardTermCfg(
        func=microduck_mdp.speed_discovery_fall,
        weight=-500.0,
    )
    retained["action_rate_l2"].weight = -0.10
    retained["self_collisions"].weight = -0.10
    retained["action_over_limit"].weight = -0.05
    cfg.rewards.update(retained)

    # Falling and NaNs terminate; lane, heading, posture, gait form, energy,
    # slip, torque, and smoothness beyond Hannes's weak -0.1 action rate do not.
    for name in list(cfg.terminations):
        if name not in {"time_out", "fell_over", "nan_state"}:
            cfg.terminations.pop(name)

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.03
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (0.80, 0.80)
    command.ranges.lin_vel_y = (-0.02, 0.02)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(
        **vars(command), rel_turn_in_place_envs=0.0
    )
    # The reference policy explicitly zero-pads the unused head/body commands.
    for name in ("head_pose", "body_pose"):
        if name in cfg.commands:
            cfg.commands[name].ranges = tuple(
                (0.0, 0.0) for _ in cfg.commands[name].ranges
            )
            cfg.commands[name].zero_command_prob = 1.0

    cfg.curriculum.clear()
    cfg.curriculum["speed_stage"] = CurriculumTermCfg(
        func=microduck_mdp.speed_discovery_performance_curriculum,
        params={
            "command_name": "twist",
            "target_reward_name": "speed_target_progress",
            "stages": [dict(stage) for stage in SPEED_DISCOVERY_STAGES],
            "min_attempts": 4096,
            "required_windows": 2,
            "effort_command": 0.80,
        },
    )

    # Metrics have no reward weight or episode-length normalization.  The peak
    # and survival terms use the terminal value so logs expose real episode
    # outcomes rather than a reward proxy.
    cfg.metrics["forward_velocity_mps"] = MetricsTermCfg(
        func=microduck_mdp.speed_discovery_mean_forward_velocity,
    )
    cfg.metrics["forward_velocity_mph"] = MetricsTermCfg(
        func=microduck_mdp.speed_discovery_mean_forward_velocity_mph,
    )
    cfg.metrics["world_forward_velocity_mps"] = MetricsTermCfg(
        func=microduck_mdp.speed_discovery_mean_world_forward_velocity,
    )
    cfg.metrics["peak_forward_velocity_mps"] = MetricsTermCfg(
        func=microduck_mdp.speed_discovery_peak_forward_velocity,
        reduce="last",
    )
    cfg.metrics["survival_fraction"] = MetricsTermCfg(
        func=microduck_mdp.speed_discovery_survival_fraction,
        reduce="last",
    )
    return cfg


MicroduckSpeedDiscoveryRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    algorithm=dataclasses.replace(
        MicroduckRollersRlCfg.algorithm,
        # Continue gently from the V11-derived skate gait.  Spark pilots showed
        # that 3e-4 and high entropy destroy the deterministic policy within a
        # handful of updates, while 3e-5 with bounded exploration improves it.
        learning_rate=3.0e-5,
        desired_kl=0.01,
        entropy_coef=0.001,
        clip_param=0.10,
        num_learning_epochs=5,
        num_mini_batches=4,
        symmetry_cfg=None,
    ),
    experiment_name="microduck_speed_discovery",
    run_name="microduck_speed_discovery",
    save_interval=25,
    num_steps_per_env=24,
    max_iterations=6_000,
)
