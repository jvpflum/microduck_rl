import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_landing_env_cfg import (
    EPISODE_LENGTH_S,
    LANDING_STAGES,
    MicroduckRollerFrontFlipLandingRlCfg,
    make_microduck_roller_frontflip_landing_env_cfg,
)


def test_landing_school_uses_real_late_flight_resets_and_exact_limits():
    cfg = make_microduck_roller_frontflip_landing_env_cfg()
    reset = cfg.events["reset_backflip_state"]
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    assert reset.func is microduck_mdp.reset_roller_frontflip_landing_state
    assert reset.params["progress_range_deg"] == (330.0, 355.0)
    assert reset.params["offaxis_scale"] == 0.0
    assert all(
        actuator.max_current == 1.75
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )
    assert "non_skate_ground_contact" in cfg.terminations


def test_landing_school_rewards_clean_skate_touchdown_not_reset_progress():
    cfg = make_microduck_roller_frontflip_landing_env_cfg()
    rewards = cfg.rewards
    assert rewards["finish_rotation"].func is microduck_mdp.roller_backflip_rotation_progress
    assert rewards["landing_readiness"].params["minimum_rotation"] == math.radians(300.0)
    assert rewards["skate_touchdown"].func is microduck_mdp.roller_backflip_landing
    assert rewards["skate_touchdown"].params["forward_speed_tolerance"] == 1.5
    assert rewards["skate_touchdown"].weight > rewards["landing_readiness"].weight
    assert rewards["post_landing_stability"].weight > 0.0
    assert rewards["non_skate_ground_contact"].weight < 0.0


def test_landing_curriculum_moves_backward_only_on_performance():
    cfg = make_microduck_roller_frontflip_landing_env_cfg()
    curriculum = cfg.curriculum["landing_backward_progress"]
    assert curriculum.func is microduck_mdp.roller_frontflip_landing_curriculum
    assert curriculum.params["min_attempts"] == 4096
    assert curriculum.params["forward_speed_tolerance"] == 1.5
    assert LANDING_STAGES[0]["params"]["progress_range_deg"][0] == 330.0
    assert LANDING_STAGES[-1]["params"]["progress_range_deg"][0] == 240.0
    assert all(stage["required_windows"] == 2 for stage in LANDING_STAGES)


def test_landing_runner_is_exploratory_and_checkpointed():
    runner = MicroduckRollerFrontFlipLandingRlCfg
    assert runner.experiment_name == "roller_frontflip_landing"
    assert runner.max_iterations == 1_000
    assert runner.save_interval == 25
    assert runner.algorithm.learning_rate == 2.0e-4
    assert runner.algorithm.clip_param == 0.15
    assert runner.algorithm.entropy_coef == 0.005
