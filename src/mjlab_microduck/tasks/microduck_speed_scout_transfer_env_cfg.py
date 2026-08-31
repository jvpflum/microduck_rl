"""Transfer the 5.41 mph frictionless speed scout into autonomous Race5 drag.

This is deliberately a speed-gait transfer, not the all-around Race5 task.
It begins at the scout's zero bearing-drag physics and only adds drag after
completed episodes prove both forward world-X progress and survival.  The
RaceLine command gives the policy an automatic yaw correction signal from
heading and cross-track error; this replaces an operator holding left without
giving the policy an external action or hiding a controller in the score.
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


# Each transition requires real completed-episode speed and survival.  The
# thresholds ease only because bearing drag rises; iteration count alone never
# advances the physics.
SPEED_SCOUT_TRANSFER_STAGES = (
    {"target_speed_mps": 2.25, "wheel_friction": 0.0000,
     "advance_mean_speed_mps": 1.60, "advance_survival_fraction": 0.95},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0005,
     "advance_mean_speed_mps": 1.40, "advance_survival_fraction": 0.93},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0010,
     "advance_mean_speed_mps": 1.25, "advance_survival_fraction": 0.92},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0015,
     "advance_mean_speed_mps": 1.10, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0020,
     "advance_mean_speed_mps": 1.00, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0025,
     "advance_mean_speed_mps": 0.90, "advance_survival_fraction": 0.90},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0030,
     "advance_mean_speed_mps": 0.80, "advance_survival_fraction": 0.90},
)


def make_microduck_speed_scout_transfer_env_cfg(play: bool = False):
    """Full-speed, autonomous-line-hold transfer from the 5.41 mph scout."""
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.events["scout_transfer_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (0.0, 0.0),
        },
    )
    # Preserve the scout's acceleration incentive.  These costs only reject
    # obvious divergence; they are intentionally far weaker than V16's gated
    # usable-speed objective.
    cfg.rewards["lane_error"].weight = -0.75
    cfg.rewards["world_lateral_velocity"].weight = -0.75
    cfg.rewards["heading_error"].weight = -0.75
    cfg.curriculum.clear()
    cfg.curriculum["speed_scout_friction_transfer"] = CurriculumTermCfg(
        func=microduck_mdp.speed_discovery_performance_curriculum,
        params={
            "command_name": "twist",
            "target_reward_name": "speed_target_progress",
            "stages": [dict(stage) for stage in SPEED_SCOUT_TRANSFER_STAGES],
            "min_attempts": 4096,
            "required_windows": 2,
            "effort_command": 0.80,
            "friction_event_name": "scout_transfer_wheel_friction",
        },
    )
    return cfg


MicroduckSpeedScoutTransferRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=5.0e-6,
        desired_kl=2.5e-4,
        clip_param=0.03,
        num_learning_epochs=2,
    ),
    experiment_name="microduck_speed_scout_transfer",
    run_name="microduck_speed_scout_transfer",
    save_interval=10,
)
