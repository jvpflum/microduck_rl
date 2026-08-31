"""Recover peak speed without discarding i6159's line-hold gains."""

from __future__ import annotations

import dataclasses

from mjlab_microduck.tasks.microduck_speed_retention_env_cfg import (
    MicroduckSpeedRetentionRlCfg,
    make_microduck_speed_retention_env_cfg,
)


def make_microduck_speed_retention_boost_env_cfg(play: bool = False):
    cfg = make_microduck_speed_retention_env_cfg(play=play)
    cfg.rewards["race_body_speed"].weight = 3.0
    cfg.rewards["race_world_speed"].weight = 8.0
    cfg.rewards["race_world_speed_squared"].weight = 1.50
    cfg.rewards["race_lane"].weight = -4.0
    cfg.rewards["race_heading"].weight = -3.0
    cfg.rewards["race_lateral_speed"].weight = -4.0
    # Idle is a first-class deployment behavior: zero command must settle,
    # not creep into a skating gait before the operator requests launch.
    cfg.rewards["stop_speed"].weight = -20.0
    cfg.rewards["cruise_error"].weight = -5.0
    return cfg


MicroduckSpeedRetentionBoostRlCfg = dataclasses.replace(
    MicroduckSpeedRetentionRlCfg,
    experiment_name="microduck_speed_retention_boost",
    run_name="microduck_speed_retention_boost",
    save_interval=50,
)
