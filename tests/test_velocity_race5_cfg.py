from mjlab_microduck.tasks.microduck_velocity_race5_env_cfg import (
    MicroduckRace5RlCfg,
    RACE5_TRAIN_DISTANCE_M,
    make_microduck_velocity_race5_env_cfg,
    race_launch_speed_progress,
    race_world_lateral_speed_squared,
)


def test_race5_clean_sprint_stage_starts_on_a_fixed_line():
    cfg = make_microduck_velocity_race5_env_cfg()
    reset_pose = cfg.events["reset_base"].params["pose_range"]

    assert RACE5_TRAIN_DISTANCE_M == 30.48
    assert cfg.episode_length_s == 60.0
    assert cfg.rewards["race_finish"].params["distance_m"] == 30.48
    assert cfg.terminations["race_finished"].params["distance_m"] == 30.48
    assert reset_pose["x"] == (0.0, 0.0)
    assert reset_pose["y"] == (0.0, 0.0)
    assert reset_pose["yaw"] == (0.0, 0.0)


def test_race5_uses_the_same_measured_line_controller_as_evaluation():
    cfg = make_microduck_velocity_race5_env_cfg()
    command = cfg.commands["twist"]

    assert command.__class__.__name__ == "RaceLineVelocityCommandCfg"
    assert command.yaw_kp == 0.55
    assert command.lateral_kp == 0.04
    assert command.yaw_kd == 0.08
    assert command.max_correction == 0.08
    assert command.ranges.lin_vel_x == (0.80, 0.80)


def test_race5_penalizes_world_cross_track_velocity_and_allows_recovery():
    cfg = make_microduck_velocity_race5_env_cfg()

    assert cfg.rewards["race_world_lateral_speed"].func is race_world_lateral_speed_squared
    assert cfg.rewards["race_world_lateral_speed"].weight < 0.0
    assert cfg.rewards["race_lane_error"].weight < cfg.rewards["lateral_speed"].weight
    assert cfg.terminations["race_out_of_lane"].params["lane_half_width_m"] == 0.30
    assert cfg.terminations["race_heading_departure"].params["maximum_yaw_rad"] == 0.436332


def test_race5_v14_uses_a_static_objective_and_champion_smoothness():
    cfg = make_microduck_velocity_race5_env_cfg()

    assert "action_rate_weight" not in cfg.curriculum
    assert "race_measured_speed_target" not in cfg.curriculum
    assert "race_speed_target" not in cfg.rewards
    assert "race_speed_weight" not in cfg.curriculum
    assert cfg.rewards["action_rate_l2"].weight == -1.2
    assert cfg.rewards["race_launch_speed"].func is race_launch_speed_progress
    assert cfg.rewards["race_speed_squared"].weight == 3.5
    assert cfg.rewards["race_lane_error"].weight == -45.0
    assert cfg.rewards["race_world_lateral_speed"].weight == -28.0
    assert cfg.rewards["race_heading_error"].weight == -32.0
    assert cfg.rewards["forward_lean"].weight == 0.6
    assert cfg.rewards["forward_lean"].params["target_pitch"] == 0.10
    assert cfg.rewards["glide"].weight == 1.5
    assert cfg.rewards["glide"].params["normalize_joint_count"] is True


def test_race5_uses_small_symmetric_ppo_updates():
    symmetry = MicroduckRace5RlCfg.algorithm.symmetry_cfg

    assert MicroduckRace5RlCfg.num_steps_per_env == 48
    assert MicroduckRace5RlCfg.algorithm.learning_rate == 5.0e-6
    assert MicroduckRace5RlCfg.algorithm.desired_kl == 2.5e-4
    assert MicroduckRace5RlCfg.algorithm.clip_param == 0.02
    assert MicroduckRace5RlCfg.algorithm.num_learning_epochs == 1
    assert symmetry is not None
    assert symmetry["use_data_augmentation"] is True
    assert symmetry["use_mirror_loss"] is True
    assert symmetry["mirror_loss_coeff"] == 2.0
