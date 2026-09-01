from mjlab_microduck.tasks.microduck_velocity_race5_fusion_env_cfg import (
    CONTROL_TEACHER_CHECKPOINT,
    SPEED_TEACHER_ONNX,
    MicroduckRace5FusionRlCfg,
    make_microduck_velocity_race5_fusion_env_cfg,
)


def test_fusion_matches_the_scored_physics_and_controller():
    cfg = make_microduck_velocity_race5_fusion_env_cfg()
    command = cfg.commands["twist"]

    assert cfg.events["randomize_wheel_friction"].params["ranges"] == (0.003, 0.003)
    assert "wheel_friction" not in cfg.curriculum
    assert command.yaw_kp == 0.55
    assert command.lateral_kp == 0.10
    assert command.yaw_kd == 0.0422
    assert command.max_correction == 0.18


def test_fusion_uses_v57b_teacher_without_sacrificing_yaw_probes():
    algorithm = MicroduckRace5FusionRlCfg.algorithm

    assert algorithm.class_name.endswith("TeacherGuidedPPO")
    assert algorithm.teacher_checkpoint == CONTROL_TEACHER_CHECKPOINT
    assert algorithm.speed_teacher_onnx == SPEED_TEACHER_ONNX
    assert algorithm.speed_command_threshold == 0.5
    assert algorithm.smooth_turn_start == 0.08
    assert algorithm.smooth_turn_end == 0.25
    assert 0.0 < algorithm.probe_loss_share < 0.5
    assert algorithm.teacher_loss_floor >= 0.35


def test_fusion_targets_the_measured_v61_launch_gap():
    cfg = make_microduck_velocity_race5_fusion_env_cfg()

    assert cfg.rewards["race_usable_launch"].weight >= cfg.rewards["race_usable_speed"].weight
    assert cfg.rewards["race_launch_speed"].weight >= cfg.rewards["race_forward_progress"].weight
    assert cfg.rewards["race_finish"].weight == 800.0
