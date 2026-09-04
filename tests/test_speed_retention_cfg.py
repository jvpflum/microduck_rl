from mjlab_microduck.tasks.microduck_speed_retention_env_cfg import (
    MicroduckSpeedRetentionRlCfg,
    SpeedRetentionRaceLineCommandCfg,
    make_microduck_speed_retention_env_cfg,
)


def test_retention_matches_official_wheel_friction_and_modes():
    cfg = make_microduck_speed_retention_env_cfg()
    assert cfg.events["official_wheel_friction"].params["ranges"] == (0.003, 0.003)
    assert isinstance(cfg.commands["twist"], SpeedRetentionRaceLineCommandCfg)
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.8)
    assert cfg.commands["twist"].ranges.ang_vel_z == (-0.3, 0.3)


def test_retention_has_gated_speed_and_control_rewards():
    cfg = make_microduck_speed_retention_env_cfg()
    assert cfg.rewards["race_world_speed"].weight == 5.0
    assert cfg.rewards["race_world_speed_squared"].weight == 0.75
    assert cfg.rewards["cruise_error"].weight == -3.0
    assert cfg.rewards["stop_speed"].weight == -6.0
    assert cfg.rewards["turn_tracking"].weight == 2.0
    assert cfg.rewards["upright_state"].weight == 2.0
    assert MicroduckSpeedRetentionRlCfg.experiment_name == "microduck_speed_retention"
