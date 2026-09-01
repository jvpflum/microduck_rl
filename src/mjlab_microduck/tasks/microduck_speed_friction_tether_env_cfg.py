"""Preservation-first transfer of the 5.4 mph skate gait to official drag.

Unlike the earlier direct adaptation, this task does not ask PPO to rediscover
speed after a full friction jump.  It tethers the known fast policy to a series
of fixed bearing-drag bands and only raises drag after the current band keeps
real world-X velocity.  PPO updates and exploration are deliberately small so
the transferred actor is changed only when the new physics demands it.
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


# The first gate is intentionally below the scout's measured 1.86 m/s
# controller score, while the final gate is above the current usable
# official-friction baseline.  A stage cannot advance on iteration count.
FRICTION_TETHER_STAGES = (
    {"target_speed_mps": 2.25, "wheel_friction": 0.0000,
     "advance_mean_speed_mps": 1.65, "advance_survival_fraction": 0.98},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0005,
     "advance_mean_speed_mps": 1.50, "advance_survival_fraction": 0.98},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0010,
     "advance_mean_speed_mps": 1.38, "advance_survival_fraction": 0.98},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0015,
     "advance_mean_speed_mps": 1.28, "advance_survival_fraction": 0.98},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0020,
     "advance_mean_speed_mps": 1.20, "advance_survival_fraction": 0.98},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0025,
     "advance_mean_speed_mps": 1.14, "advance_survival_fraction": 0.98},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0030,
     "advance_mean_speed_mps": 1.10, "advance_survival_fraction": 0.99},
)


def make_microduck_speed_friction_tether_env_cfg(play: bool = False):
    """Create a low-plasticity, performance-gated skate-preservation task."""
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.episode_length_s = 20.0
    cfg.events["tether_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (0.0, 0.0),
        },
    )

    # Faster world-X remains more valuable than a slow perfectly-centred cruise,
    # but line terms are strong enough to rule out the drift exploit seen in the
    # previous 4k official-adaptation run.
    cfg.rewards["world_forward_velocity_mps"].weight = 6.5
    cfg.rewards["world_forward_velocity_squared"].weight = 1.25
    cfg.rewards["heading_hold"].weight = 2.5
    cfg.rewards["lane_error"].weight = -1.2
    cfg.rewards["world_lateral_velocity"].weight = -1.2
    cfg.rewards["heading_error"].weight = -1.2

    cfg.curriculum.clear()
    cfg.curriculum["friction_tether"] = CurriculumTermCfg(
        func=microduck_mdp.speed_discovery_performance_curriculum,
        params={
            "command_name": "twist",
            "target_reward_name": "speed_target_progress",
            "stages": [dict(stage) for stage in FRICTION_TETHER_STAGES],
            "min_attempts": 4096,
            "required_windows": 2,
            "effort_command": 0.80,
            "friction_event_name": "tether_wheel_friction",
        },
    )
    return cfg


MicroduckSpeedFrictionTetherRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=7.5e-7,
        desired_kl=1.0e-4,
        clip_param=0.01,
        num_learning_epochs=2,
    ),
    experiment_name="microduck_speed_friction_tether",
    run_name="microduck_speed_friction_tether",
    save_interval=20,
)
