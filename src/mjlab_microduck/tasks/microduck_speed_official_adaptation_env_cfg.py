"""Official-friction speed adaptation for the preserved 5.41 mph skate gait.

This task intentionally does *not* ramp from zero friction.  Every rollout is
near the Race5 bearing-drag profile, so PPO cannot keep a frictionless-only
solution and call it progress.  World-X speed stays the primary objective;
the small line costs merely reject the obvious steer-by-drifting exploit.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
)


def make_microduck_speed_official_adaptation_env_cfg(play: bool = False):
    """Robust speed training around official Race5 roller friction."""
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.episode_length_s = 20.0
    cfg.events["official_band_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            # Realistic variation around the benchmark value; 0.003 remains
            # the deterministic evaluation point.
            "ranges": (0.0025, 0.0035),
        },
    )
    command = cfg.commands["twist"]
    command.yaw_kp = 0.80
    command.lateral_kp = 0.16
    command.yaw_kd = 0.10
    command.max_correction = 0.22

    # Do not let a stable slow cruise dominate.  The original speed-discovery
    # terms remain continuous (not command-capped); world progress carries the
    # largest weight.  Direction costs are deliberately modest.
    cfg.rewards["forward_velocity_mps"].weight = 3.0
    cfg.rewards["forward_velocity_squared"].weight = 0.75
    cfg.rewards["speed_target_progress"].weight = 0.75
    cfg.rewards["world_forward_velocity_mps"].weight = 7.0
    cfg.rewards["world_forward_velocity_squared"].weight = 1.50
    cfg.rewards["heading_hold"].weight = 1.5
    cfg.rewards["lane_error"].weight = -0.60
    cfg.rewards["world_lateral_velocity"].weight = -0.60
    cfg.rewards["heading_error"].weight = -0.60
    cfg.curriculum.clear()
    return cfg


MicroduckSpeedOfficialAdaptationRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=2.0e-6,
        desired_kl=1.0e-4,
        clip_param=0.015,
        num_learning_epochs=2,
    ),
    experiment_name="microduck_speed_official_adaptation",
    run_name="microduck_speed_official_adaptation",
    save_interval=10,
)
