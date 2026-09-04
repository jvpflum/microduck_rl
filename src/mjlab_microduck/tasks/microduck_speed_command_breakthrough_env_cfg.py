"""Official-friction speed continuation with staged command exposure.

The friction-transfer run preserved the discovered skate at Race5's 0.003
wheel friction, but the actor still observed the historical 0.8 full-effort
token.  The published fast running policy was trained through a 2.2 m/s
command.  This task keeps actual world-X velocity as the dominant objective
and exposes 0.8 -> 2.2 commands only after completed episodes demonstrate
survival and retained forward progress.

This remains a speed-discovery stage.  Mild lane, lateral, and heading costs
prevent circular body-speed reward hacking; braking and agility are restored
only after a faster official-physics checkpoint exists.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
)


COMMAND_BREAKTHROUGH_STAGES = (
    {"effort_command": 0.80, "target_speed_mps": 1.20, "wheel_friction": 0.003,
     "advance_mean_speed_mps": 0.66, "advance_survival_fraction": 0.98},
    {"effort_command": 1.10, "target_speed_mps": 1.45, "wheel_friction": 0.003,
     "advance_mean_speed_mps": 0.68, "advance_survival_fraction": 0.98},
    {"effort_command": 1.40, "target_speed_mps": 1.70, "wheel_friction": 0.003,
     "advance_mean_speed_mps": 0.70, "advance_survival_fraction": 0.98},
    {"effort_command": 1.70, "target_speed_mps": 1.95, "wheel_friction": 0.003,
     "advance_mean_speed_mps": 0.72, "advance_survival_fraction": 0.98},
    {"effort_command": 2.00, "target_speed_mps": 2.15, "wheel_friction": 0.003,
     "advance_mean_speed_mps": 0.74, "advance_survival_fraction": 0.98},
    {"effort_command": 2.20, "target_speed_mps": 2.2352, "wheel_friction": 0.003,
     "advance_mean_speed_mps": 0.76, "advance_survival_fraction": 0.98},
)


def make_microduck_speed_command_breakthrough_env_cfg(play: bool = False):
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.events["official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (0.003, 0.003),
        },
    )

    # World-forward rewards remain dominant.  Direction costs are deliberately
    # mild during breakthrough discovery so the policy cannot earn more by
    # stopping, while gross circling or sideways motion is still unprofitable.
    cfg.rewards["heading_hold"].weight = 0.75
    cfg.rewards["lane_error"].weight = -0.40
    cfg.rewards["world_lateral_velocity"].weight = -0.40
    cfg.rewards["heading_error"].weight = -0.40

    cfg.curriculum.clear()
    cfg.curriculum["command_breakthrough"] = CurriculumTermCfg(
        func=microduck_mdp.speed_discovery_performance_curriculum,
        params={
            "command_name": "twist",
            "target_reward_name": "speed_target_progress",
            "stages": [dict(stage) for stage in COMMAND_BREAKTHROUGH_STAGES],
            "min_attempts": 1024,
            "required_windows": 1,
            "effort_command": 0.80,
            "friction_event_name": "official_wheel_friction",
        },
    )
    return cfg


MicroduckSpeedCommandBreakthroughRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=5.0e-6,
        desired_kl=2.5e-4,
        clip_param=0.05,
        num_learning_epochs=2,
    ),
    experiment_name="microduck_speed_command_breakthrough",
    run_name="microduck_speed_command_breakthrough",
    # About every 1-2 minutes on the Spark at 1,024-4,096 environments: frequent
    # enough to catch a gait transition without producing thousands of files.
    save_interval=100,
)
