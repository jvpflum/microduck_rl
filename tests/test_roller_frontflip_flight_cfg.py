import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_flight_env_cfg import (
    EPISODE_LENGTH_S,
    LANDING_ROTATION,
    TARGET_ROTATION,
    MicroduckRollerFrontFlipFlightRlCfg,
    make_microduck_roller_frontflip_flight_env_cfg,
)


def test_flight_stage_preserves_unassisted_exact_launch():
    cfg = make_microduck_roller_frontflip_flight_env_cfg()
    reset = cfg.events["reset_backflip_state"].params
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    assert reset["demo_prob"] == 0.0
    assert reset["unassisted_stand_prob"] == 1.0
    assert all(
        actuator.max_current == 1.75
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )
    assert "non_skate_ground_contact" in cfg.terminations


def test_flight_stage_pushes_rotation_then_skate_landing():
    cfg = make_microduck_roller_frontflip_flight_env_cfg()
    rewards = cfg.rewards
    assert math.isclose(rewards["flight_rotation_progress"].params["target_rotation"], TARGET_ROTATION)
    assert rewards["landing"].params["landing_rotation"] == LANDING_ROTATION
    assert rewards["landing_readiness"].params["minimum_rotation"] == math.radians(150.0)
    assert rewards["sagittal_motion"].weight == -0.20
    assert rewards["non_skate_ground_contact"].weight < 0.0
    assert rewards["landing_readiness"].func is microduck_mdp.roller_backflip_landing_readiness_progress


def test_flight_runner_is_a_conservative_launch_donor_finetune():
    runner = MicroduckRollerFrontFlipFlightRlCfg
    assert runner.experiment_name == "roller_frontflip_flight"
    assert runner.max_iterations == 800
    assert runner.save_interval == 25
    assert runner.algorithm.learning_rate == 1.0e-4
    assert runner.algorithm.clip_param == 0.10
    assert runner.algorithm.entropy_coef == 0.002
