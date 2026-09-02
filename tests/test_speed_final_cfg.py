from mjlab_microduck.tasks.microduck_speed_final_env_cfg import (
    EXACT_V59_CHECKPOINT,
    FIVE_MPH_MPS,
    OFFICIAL_WHEEL_FRICTION,
    MicroduckSpeedFinalRlCfg,
    make_microduck_speed_final_env_cfg,
)


def test_final_task_uses_exact_official_operating_point():
    cfg = make_microduck_speed_final_env_cfg()
    command = cfg.commands["twist"]

    assert cfg.events["final_official_wheel_friction"].params["ranges"] == (
        OFFICIAL_WHEEL_FRICTION,
        OFFICIAL_WHEEL_FRICTION,
    )
    assert command.ranges.lin_vel_x == (0.80, 0.80)
    assert command.yaw_kp == 0.70
    assert command.lateral_kp == 0.14
    assert command.yaw_kd == 0.07
    assert command.max_correction == 0.15


def test_final_task_has_high_speed_reverse_curriculum():
    cfg = make_microduck_speed_final_env_cfg()
    params = cfg.events["reset_final_speed_state"].params

    assert params["bootstrap_speed_range_mps"][1] > FIVE_MPH_MPS
    assert params["bootstrap_fraction_stages"][0][1] == 0.40
    assert params["bootstrap_fraction_stages"][-1][1] > 0.0


def test_final_task_starts_at_exact_v59_and_optimizes_usable_speed():
    cfg = make_microduck_speed_final_env_cfg()
    algorithm = MicroduckSpeedFinalRlCfg.algorithm

    assert algorithm.class_name.endswith("TeacherGuidedPPO")
    assert algorithm.teacher_checkpoint == EXACT_V59_CHECKPOINT
    assert algorithm.teacher_loss_floor > 0.0
    assert algorithm.probe_loss_share == 0.0
    assert cfg.rewards["usable_speed"].weight > cfg.rewards["world_speed"].weight
    assert cfg.rewards["fall"].weight < 0.0
    assert cfg.rewards["tilt_cost"].weight < 0.0
    assert cfg.rewards["action_rate_l2"].weight < 0.0
