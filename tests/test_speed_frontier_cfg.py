from mjlab_microduck.tasks.microduck_speed_frontier_env_cfg import (
    BOOTSTRAP_SPEED_RANGE_MPS,
    OFFICIAL_WHEEL_FRICTION,
    PHASE_FREQUENCY_RANGE_HZ,
    WHEEL_RADIUS_M,
    XL330_CURRENT_LIMIT_A,
    MicroduckSpeedFrontierRlCfg,
    frontier_phase_command,
    make_microduck_speed_frontier_env_cfg,
    reset_frontier_state,
)


def test_frontier_starts_at_official_drag_with_nominal_current_limited_bam():
    cfg = make_microduck_speed_frontier_env_cfg()
    friction = cfg.events["frontier_official_wheel_friction"]
    assert friction.params["ranges"] == (
        OFFICIAL_WHEEL_FRICTION,
        OFFICIAL_WHEEL_FRICTION,
    )
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    assert actuator.max_current == XL330_CURRENT_LIMIT_A == 1.75
    assert actuator.vin == 7.4
    assert actuator.vin_range is None
    assert actuator.vin_drop_gain_range is None
    assert actuator.delay_min_lag == actuator.delay_max_lag == 3


def test_frontier_uses_official_horizon_phase_conditioned_no_slip_discovery():
    cfg = make_microduck_speed_frontier_env_cfg()
    assert cfg.episode_length_s == 20.0
    assert WHEEL_RADIUS_M == 0.015
    assert PHASE_FREQUENCY_RANGE_HZ == (1.75, 5.50)
    assert BOOTSTRAP_SPEED_RANGE_MPS == (0.50, 3.00)
    reset = cfg.events["reset_frontier_state"]
    assert reset.func is reset_frontier_state
    assert "frontier_phase" not in cfg.observations["actor"].terms
    assert "frontier_phase" not in cfg.observations["critic"].terms
    assert (
        cfg.observations["actor"].terms["body_command"].func
        is frontier_phase_command
    )
    assert (
        cfg.observations["critic"].terms["body_command"].func
        is frontier_phase_command
    )


def test_frontier_reward_values_gain_and_final_world_speed_without_teacher():
    cfg = make_microduck_speed_frontier_env_cfg()
    assert cfg.rewards["world_speed"].weight == 4.0
    assert cfg.rewards["world_speed_squared"].weight == 1.0
    assert cfg.rewards["speed_gain"].weight == 6.0
    assert cfg.rewards["final_speed"].weight == 3.0
    assert cfg.rewards["lateral_speed"].weight < 0.0
    assert not cfg.curriculum
    assert MicroduckSpeedFrontierRlCfg.algorithm.class_name == "PPO"
    assert MicroduckSpeedFrontierRlCfg.algorithm.learning_rate == 1.0e-4
    assert MicroduckSpeedFrontierRlCfg.actor.distribution_cfg["init_std"] == 0.08
    assert MicroduckSpeedFrontierRlCfg.algorithm.entropy_coef == 0.0
    assert MicroduckSpeedFrontierRlCfg.max_iterations == 8_000
