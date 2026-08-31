from mjlab_microduck.tasks.microduck_speed_friction_transfer_env_cfg import (
    FRICTION_TRANSFER_STAGES,
    MicroduckSpeedFrictionTransferRlCfg,
    make_microduck_speed_friction_transfer_env_cfg,
)


def test_transfer_stages_raise_drag_without_changing_effort_command():
    cfg = make_microduck_speed_friction_transfer_env_cfg()
    assert [stage["wheel_friction"] for stage in FRICTION_TRANSFER_STAGES] == [
        0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003
    ]
    assert [stage["advance_mean_speed_mps"] for stage in FRICTION_TRANSFER_STAGES] == [
        0.95, 0.90, 0.85, 0.80, 0.75, 0.70
    ]
    assert cfg.events["transfer_wheel_friction"].params["ranges"] == (0.0005, 0.0005)
    curriculum = cfg.curriculum["friction_transfer"]
    assert curriculum.params["effort_command"] == 0.8
    assert curriculum.params["friction_event_name"] == "transfer_wheel_friction"
    assert curriculum.params["min_attempts"] == 1024


def test_transfer_keeps_world_speed_and_line_objective():
    cfg = make_microduck_speed_friction_transfer_env_cfg()
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.8, 0.8)
    assert cfg.rewards["world_forward_velocity_mps"].weight == 5.0
    assert cfg.rewards["world_forward_velocity_squared"].weight == 0.75
    assert cfg.rewards["lane_error"].weight == -1.5
    assert cfg.rewards["world_lateral_velocity"].weight == -1.5
    assert cfg.rewards["heading_error"].weight == -1.5
    assert MicroduckSpeedFrictionTransferRlCfg.experiment_name == (
        "microduck_speed_friction_transfer"
    )
