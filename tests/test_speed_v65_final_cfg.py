from mjlab_microduck.tasks.microduck_speed_v65_final_env_cfg import (
    EXACT_V65_HIGH_CHECKPOINT,
    MicroduckSpeedV65FinalRlCfg,
    make_microduck_speed_v65_final_env_cfg,
)


def test_v65_final_uses_exact_official_physics_and_rest_launches():
    cfg = make_microduck_speed_v65_final_env_cfg()
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    reset = cfg.events["reset_final_speed_state"].params
    friction = cfg.events["final_official_wheel_friction"].params["ranges"]

    assert actuator.max_current == 1.75
    assert actuator.delay_min_lag == actuator.delay_max_lag == 3
    assert friction == (0.003, 0.003)
    assert reset["bootstrap_fraction_stages"] == ((0, 0.0),)
    assert reset["bootstrap_speed_range_mps"] == (0.0, 0.0)


def test_v65_final_is_a_fixed_rate_strongly_anchored_local_search():
    algorithm = MicroduckSpeedV65FinalRlCfg.algorithm

    assert algorithm.teacher_checkpoint == EXACT_V65_HIGH_CHECKPOINT
    assert algorithm.teacher_loss_coef == algorithm.teacher_loss_floor == 50.0
    assert algorithm.schedule == "fixed"
    assert algorithm.learning_rate == 2.0e-8
    assert algorithm.clip_param == 0.002
    assert algorithm.num_learning_epochs == 1
    assert MicroduckSpeedV65FinalRlCfg.save_interval == 25


def test_v65_final_fall_cost_dominates_speed_terms():
    cfg = make_microduck_speed_v65_final_env_cfg()

    assert cfg.rewards["fall"].weight == -2_000.0
    assert cfg.rewards["usable_speed"].weight == 50.0
    assert cfg.rewards["usable_launch"].weight == 25.0
    assert cfg.rewards["tilt_cost"].weight == -1.0
