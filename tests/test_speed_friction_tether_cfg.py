from mjlab_microduck.tasks.microduck_speed_friction_tether_env_cfg import (
    FRICTION_TETHER_STAGES,
    MicroduckSpeedFrictionTetherRlCfg,
    make_microduck_speed_friction_tether_env_cfg,
)


def test_tether_raises_drag_only_after_retained_speed_and_survival():
    cfg = make_microduck_speed_friction_tether_env_cfg()
    assert [stage["wheel_friction"] for stage in FRICTION_TETHER_STAGES] == [
        0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003,
    ]
    assert [stage["advance_mean_speed_mps"] for stage in FRICTION_TETHER_STAGES] == [
        1.65, 1.50, 1.38, 1.28, 1.20, 1.14, 1.10,
    ]
    assert cfg.events["tether_wheel_friction"].params["ranges"] == (0.0, 0.0)
    curriculum = cfg.curriculum["friction_tether"]
    assert curriculum.params["min_attempts"] == 4096
    assert curriculum.params["required_windows"] == 2


def test_tether_is_low_plasticity_but_speed_and_line_aware():
    cfg = make_microduck_speed_friction_tether_env_cfg()
    assert cfg.rewards["world_forward_velocity_mps"].weight == 6.5
    assert cfg.rewards["world_forward_velocity_squared"].weight == 1.25
    assert cfg.rewards["lane_error"].weight == -1.2
    assert cfg.rewards["world_lateral_velocity"].weight == -1.2
    assert cfg.rewards["heading_error"].weight == -1.2
    assert MicroduckSpeedFrictionTetherRlCfg.algorithm.learning_rate == 7.5e-7
    assert MicroduckSpeedFrictionTetherRlCfg.experiment_name == "microduck_speed_friction_tether"
