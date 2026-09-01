from mjlab_microduck.tasks.microduck_velocity_race5_frontier_env_cfg import (
    CONTROL_TEACHER_CHECKPOINT,
    OFFICIAL_WHEEL_FRICTIONLOSS,
    SPEED_TEACHER_ONNX,
    MicroduckRace5FrontierRlCfg,
    make_microduck_velocity_race5_frontier_env_cfg,
)


def test_frontier_trains_on_the_exact_scored_physics_and_controller():
    cfg = make_microduck_velocity_race5_frontier_env_cfg()
    command = cfg.commands["twist"]

    assert cfg.events["randomize_wheel_friction"].params["ranges"] == (
        OFFICIAL_WHEEL_FRICTIONLOSS,
        OFFICIAL_WHEEL_FRICTIONLOSS,
    )
    assert "wheel_friction" not in cfg.curriculum
    assert command.yaw_kp == 0.55
    assert command.lateral_kp == 0.25
    assert command.yaw_kd == 0.05
    assert command.max_correction == 0.10


def test_frontier_uses_both_hybrid_teachers_and_command_replay():
    algorithm = MicroduckRace5FrontierRlCfg.algorithm

    assert algorithm.class_name.endswith("TeacherGuidedPPO")
    assert algorithm.teacher_checkpoint == CONTROL_TEACHER_CHECKPOINT
    assert algorithm.speed_teacher_onnx == SPEED_TEACHER_ONNX
    assert algorithm.speed_command_threshold == 0.5
    assert algorithm.smooth_turn_start == 0.02
    assert algorithm.smooth_turn_end == 0.12
    assert algorithm.probe_loss_share > 0.0
    assert algorithm.teacher_loss_floor > 0.0
    assert algorithm.learning_rate == 1.0e-6


def test_frontier_pushes_speed_and_launch_inside_the_usable_envelope():
    cfg = make_microduck_velocity_race5_frontier_env_cfg()

    assert cfg.rewards["race_usable_speed"].weight > cfg.rewards["race_speed_squared"].weight
    assert cfg.rewards["race_usable_launch"].weight > cfg.rewards["race_launch_speed"].weight
    assert cfg.rewards["race_finish"].weight == 800.0
    assert cfg.rewards["race_elapsed"].weight == -1.5
