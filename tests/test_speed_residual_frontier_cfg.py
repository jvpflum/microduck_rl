from mjlab_microduck.tasks.microduck_speed_residual_frontier_env_cfg import (
    MicroduckSpeedResidualFrontierRlCfg,
    make_microduck_speed_residual_frontier_env_cfg,
)


def test_residual_frontier_uses_official_drag_and_mixed_starts():
    cfg = make_microduck_speed_residual_frontier_env_cfg()
    friction = cfg.events["official_wheel_friction"]
    exposure = cfg.events["rolling_state_exposure"]

    assert cfg.episode_length_s == 5.0
    assert cfg.actions["joint_pos"].scale == {
        r"^(left_hip_pitch|right_hip_pitch)$": 1.0,
        r"^(left_ankle|right_ankle)$": 1.06
    }
    assert friction.params["ranges"] == (0.003, 0.003)
    assert friction.params["asset_cfg"].joint_names == (r"^passive_.*wheel$",)
    assert exposure.params["velocity_range"] == (0.50, 1.20)
    assert exposure.params["fraction_stages"] == [{"step": 0, "fraction": 0.50}]
    command = cfg.commands["twist"]
    assert command.yaw_kp == 0.55
    assert command.lateral_kp == 0.10
    assert command.yaw_kd == 0.08
    assert command.max_correction == 0.18
    assert cfg.curriculum == {}
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        assert actuator.vin_range == (7.4, 7.4)
        assert actuator.vin_drop_gain_range == (0.10, 0.10)
        assert actuator.delay_min_lag == 4
        assert actuator.delay_max_lag == 4
    assert cfg.rewards["usable_speed"].weight > cfg.rewards["world_forward_velocity_mps"].weight
    assert cfg.rewards["fall"].weight == -750.0


def test_residual_frontier_supports_deployment_matched_hip_pitch_gain(monkeypatch):
    monkeypatch.setenv("DUCKLAB_RESIDUAL_HIP_PITCH_GAIN", "1.014")

    cfg = make_microduck_speed_residual_frontier_env_cfg()

    assert cfg.actions["joint_pos"].scale == {
        r"^(left_hip_pitch|right_hip_pitch)$": 1.014,
        r"^(left_ankle|right_ankle)$": 1.06,
    }


def test_residual_frontier_supports_high_speed_rolling_exposure(monkeypatch):
    monkeypatch.setenv("DUCKLAB_RESIDUAL_ROLLING_FRACTION", "0.50")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_ROLLING_MIN_MPS", "0.80")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_ROLLING_MAX_MPS", "2.50")

    cfg = make_microduck_speed_residual_frontier_env_cfg()
    exposure = cfg.events["rolling_state_exposure"]

    assert exposure.params["velocity_range"] == (0.80, 2.50)
    assert exposure.params["fraction_stages"] == [{"step": 0, "fraction": 0.50}]


def test_residual_frontier_keeps_donor_actor_separate():
    assert MicroduckSpeedResidualFrontierRlCfg.actor.class_name.endswith(
        ":ResidualMLPModel"
    )
    assert MicroduckSpeedResidualFrontierRlCfg.algorithm.class_name.endswith(
        ":ResidualPPO"
    )
    assert MicroduckSpeedResidualFrontierRlCfg.actor.hidden_dims == (512, 256, 128)
    assert MicroduckSpeedResidualFrontierRlCfg.algorithm.learning_rate == 1.0e-4
    assert MicroduckSpeedResidualFrontierRlCfg.save_interval == 10


def test_sustained_stage_disables_artificial_launches(monkeypatch):
    monkeypatch.setenv("DUCKLAB_RESIDUAL_EPISODE_LENGTH_S", "30")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_ROLLING_FRACTION", "0")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LAUNCH_WEIGHT", "0")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_USABLE_SPEED_WEIGHT", "16")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_FALL_WEIGHT", "-1000")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LANE_WEIGHT", "-4")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LATERAL_WEIGHT", "-4")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_HEADING_ERROR_WEIGHT", "-3")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_HEADING_HOLD_WEIGHT", "3")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LANE_STD_M", "0.18")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_HEADING_STD_RAD", "0.12")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LATERAL_SPEED_STD_MPS", "0.15")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LINE_YAW_KP", "0.69")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LINE_LATERAL_KP", "0.21")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LINE_YAW_KD", "0.105")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LINE_MAX_WZ", "0.18")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_UPRIGHT_AT_SPEED_WEIGHT", "3.5")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_UPRIGHT_STD_RAD", "0.20")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_ANGULAR_VELOCITY_WEIGHT", "-0.03")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_LATERAL_GUARDRAIL_WEIGHT", "-0.5")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_MAX_LATERAL_SPEED_MPS", "0.05")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_TILT_GUARDRAIL_WEIGHT", "-0.75")
    monkeypatch.setenv("DUCKLAB_RESIDUAL_MAX_TILT_DEG", "16")

    cfg = make_microduck_speed_residual_frontier_env_cfg()

    assert cfg.episode_length_s == 30.0
    assert "rolling_state_exposure" not in cfg.events
    assert cfg.rewards["usable_launch"].weight == 0.0
    assert cfg.rewards["usable_speed"].weight == 16.0
    assert cfg.rewards["fall"].weight == -1000.0
    assert cfg.rewards["lane_error"].weight == -4.0
    assert cfg.rewards["world_lateral_velocity"].weight == -4.0
    assert cfg.rewards["heading_error"].weight == -3.0
    assert cfg.rewards["heading_hold"].weight == 3.0
    command = cfg.commands["twist"]
    assert command.yaw_kp == 0.69
    assert command.lateral_kp == 0.21
    assert command.yaw_kd == 0.105
    assert command.max_correction == 0.18
    assert cfg.rewards["upright_at_speed"].weight == 3.5
    assert cfg.rewards["upright_at_speed"].params["upright_std_rad"] == 0.20
    assert cfg.rewards["angular_velocity_at_speed"].weight == -0.03
    assert cfg.rewards["lateral_guardrail_at_speed"].weight == -0.5
    assert (
        cfg.rewards["lateral_guardrail_at_speed"].params[
            "max_lateral_speed_mps"
        ]
        == 0.05
    )
    assert cfg.rewards["tilt_guardrail_at_speed"].weight == -0.75
    assert cfg.rewards["tilt_guardrail_at_speed"].params["max_tilt_deg"] == 16.0
    assert cfg.rewards["usable_speed"].params["lane_std_m"] == 0.18
    assert cfg.rewards["usable_speed"].params["heading_std_rad"] == 0.12
    assert (
        cfg.rewards["usable_speed"].params["lateral_speed_std_mps"]
        == 0.15
    )
