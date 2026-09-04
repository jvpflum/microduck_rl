import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_launch_env_cfg import (
    EPISODE_LENGTH_S,
    TARGET_CLEARANCE,
    TARGET_ROTATION,
    MicroduckRollerFrontFlipLaunchRlCfg,
    make_microduck_roller_frontflip_launch_env_cfg,
)


def test_launch_task_is_unassisted_forward_and_skates_only():
    cfg = make_microduck_roller_frontflip_launch_env_cfg()
    reset = cfg.events["reset_backflip_state"].params
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    assert "backflip_assistance" not in cfg.events
    assert reset["demo_prob"] == 0.0
    assert reset["assist_vz_range"] == (0.0, 0.0)
    assert reset["assist_omega_range"] == (0.0, 0.0)
    assert reset["unassisted_stand_prob"] == 1.0
    assert "non_skate_ground_contact" in cfg.terminations
    assert all(
        actuator.max_current == 1.75
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )


def test_launch_rewards_real_clearance_and_positive_pitch_momentum():
    cfg = make_microduck_roller_frontflip_launch_env_cfg()
    rewards = cfg.rewards
    assert rewards["launch_rotation_progress"].func is microduck_mdp.roller_backflip_rotation_progress
    assert math.isclose(rewards["launch_rotation_progress"].params["target_rotation"], TARGET_ROTATION)
    assert rewards["launch_clearance_progress"].params["target_clearance"] == TARGET_CLEARANCE
    assert rewards["launch_pitch_momentum"].params["target_pitch_rate"] == 30.0
    assert rewards["launch_vertical_velocity"].params["max_vz"] == 1.5
    assert rewards["non_skate_ground_contact"].weight < 0.0
    assert rewards["action_rate_l2"].weight > -1.0e-4
    assert not any("landing" in name for name in rewards)


def test_launch_runner_is_exploratory_and_checkpointed_frequently():
    runner = MicroduckRollerFrontFlipLaunchRlCfg
    assert runner.experiment_name == "roller_frontflip_launch"
    assert runner.max_iterations == 1_200
    assert runner.save_interval == 25
    assert runner.algorithm.learning_rate == 1.0e-3
    assert runner.algorithm.clip_param == 0.20
    assert runner.algorithm.entropy_coef == 0.02
