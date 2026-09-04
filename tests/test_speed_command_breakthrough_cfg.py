from mjlab_microduck.tasks.microduck_speed_command_breakthrough_env_cfg import (
    COMMAND_BREAKTHROUGH_STAGES,
    MicroduckSpeedCommandBreakthroughRlCfg,
    make_microduck_speed_command_breakthrough_env_cfg,
)


def test_breakthrough_stages_expose_command_gradually_at_official_friction():
    cfg = make_microduck_speed_command_breakthrough_env_cfg()
    assert [stage["effort_command"] for stage in COMMAND_BREAKTHROUGH_STAGES] == [
        0.8, 1.1, 1.4, 1.7, 2.0, 2.2
    ]
    assert [stage["target_speed_mps"] for stage in COMMAND_BREAKTHROUGH_STAGES] == [
        1.2, 1.45, 1.7, 1.95, 2.15, 2.2352
    ]
    assert cfg.events["official_wheel_friction"].params["ranges"] == (0.003, 0.003)
    curriculum = cfg.curriculum["command_breakthrough"]
    assert curriculum.params["min_attempts"] == 1024
    assert curriculum.params["required_windows"] == 1
    assert curriculum.params["friction_event_name"] == "official_wheel_friction"


def test_breakthrough_keeps_speed_dominant_with_mild_direction_costs():
    cfg = make_microduck_speed_command_breakthrough_env_cfg()
    assert cfg.rewards["world_forward_velocity_mps"].weight == 5.0
    assert cfg.rewards["world_forward_velocity_squared"].weight == 0.75
    assert cfg.rewards["heading_hold"].weight == 0.75
    assert cfg.rewards["lane_error"].weight == -0.4
    assert cfg.rewards["world_lateral_velocity"].weight == -0.4
    assert cfg.rewards["heading_error"].weight == -0.4
    assert MicroduckSpeedCommandBreakthroughRlCfg.experiment_name == (
        "microduck_speed_command_breakthrough"
    )
