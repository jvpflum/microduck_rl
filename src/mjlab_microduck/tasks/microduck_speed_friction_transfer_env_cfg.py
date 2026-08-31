"""Performance-gated transfer of the discovered fast skate to Race5 drag.

Jumping directly from frictionless discovery to the official 0.003 wheel
friction cut verified top speed from roughly five to three mph.  This stage
keeps the full-effort, straight-line objective and raises bearing drag only
after a large completed-episode window demonstrates speed and survival.  Low
speed cruise/brake/turn retention remains a later stage so it cannot dilute
the transfer objective before the official-speed gait exists.
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


FRICTION_TRANSFER_STAGES = (
    {"target_speed_mps": 2.2352, "wheel_friction": 0.0005,
     "advance_mean_speed_mps": 0.95, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.2352, "wheel_friction": 0.0010,
     "advance_mean_speed_mps": 0.90, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.2352, "wheel_friction": 0.0015,
     "advance_mean_speed_mps": 0.85, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.2352, "wheel_friction": 0.0020,
     "advance_mean_speed_mps": 0.80, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.2352, "wheel_friction": 0.0025,
     "advance_mean_speed_mps": 0.75, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.2352, "wheel_friction": 0.0030,
     "advance_mean_speed_mps": 0.70, "advance_survival_fraction": 0.90},
)


def make_microduck_speed_friction_transfer_env_cfg(play: bool = False):
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.events["transfer_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (0.0005, 0.0005),
        },
    )
    # Keep actual world-X velocity dominant. Slightly strengthen the two early
    # direction signals to prevent the friction adaptation from rediscovering
    # the old curved body-speed solution.
    cfg.rewards["lane_error"].weight = -1.5
    cfg.rewards["world_lateral_velocity"].weight = -1.5
    cfg.rewards["heading_error"].weight = -1.5
    cfg.curriculum.clear()
    cfg.curriculum["friction_transfer"] = CurriculumTermCfg(
        func=microduck_mdp.speed_discovery_performance_curriculum,
        params={
            "command_name": "twist",
            "target_reward_name": "speed_target_progress",
            "stages": [dict(stage) for stage in FRICTION_TRANSFER_STAGES],
            "min_attempts": 1024,
            "required_windows": 1,
            "effort_command": 0.80,
            "friction_event_name": "transfer_wheel_friction",
        },
    )
    return cfg


MicroduckSpeedFrictionTransferRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=1.0e-5,
        desired_kl=5.0e-4,
        clip_param=0.05,
        num_learning_epochs=2,
    ),
    experiment_name="microduck_speed_friction_transfer",
    run_name="microduck_speed_friction_transfer",
    save_interval=20,
)
