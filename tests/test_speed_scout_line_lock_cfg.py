from mjlab_microduck.tasks.microduck_speed_scout_line_lock_env_cfg import (
    MicroduckSpeedScoutLineLockRlCfg,
    make_microduck_speed_scout_line_lock_env_cfg,
)


def test_line_lock_keeps_calibrated_drag_without_curriculum(monkeypatch):
    monkeypatch.setenv("DUCKLAB_LINE_LOCK_FRICTION", "0.001")
    cfg = make_microduck_speed_scout_line_lock_env_cfg()
    assert cfg.curriculum == {}
    event = cfg.events["line_lock_wheel_friction"]
    assert event.params["ranges"] == (0.001, 0.001)
    command = cfg.commands["twist"]
    assert command.yaw_kp == 0.90
    assert command.lateral_kp == 0.20
    assert command.yaw_kd == 0.12
    assert command.max_correction == 0.30


def test_line_lock_prioritizes_speed_retention_with_gated_updates():
    cfg = make_microduck_speed_scout_line_lock_env_cfg()
    assert cfg.rewards["world_forward_velocity_mps"].weight == 8.0
    assert cfg.rewards["world_forward_velocity_squared"].weight == 1.5
    assert cfg.rewards["heading_hold"].weight == 2.5
    assert cfg.rewards["lane_error"].weight == -1.5
    assert cfg.rewards["world_lateral_velocity"].weight == -1.5
    assert cfg.rewards["heading_error"].weight == -1.5
    assert cfg.rewards["usable_speed"].weight == 2.0
    assert MicroduckSpeedScoutLineLockRlCfg.algorithm.learning_rate == 5.0e-7
    assert MicroduckSpeedScoutLineLockRlCfg.algorithm.desired_kl == 5.0e-5
    assert MicroduckSpeedScoutLineLockRlCfg.algorithm.clip_param == 0.01
    assert MicroduckSpeedScoutLineLockRlCfg.algorithm.num_learning_epochs == 1
