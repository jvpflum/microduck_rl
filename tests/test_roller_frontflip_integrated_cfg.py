import math

from mjlab.envs.mdp import dr

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_integrated_env_cfg import (
    EPISODE_LENGTH_S,
    INTEGRATED_STAGES,
    OFFICIAL_WHEEL_FRICTION,
    MicroduckRollerFrontFlipIntegratedRlCfg,
    make_microduck_roller_frontflip_integrated_env_cfg,
)


def test_integrated_task_stratifies_all_phases_without_assistance():
    cfg = make_microduck_roller_frontflip_integrated_env_cfg()
    reset = cfg.events["reset_backflip_state"]
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    assert reset.func is microduck_mdp.reset_roller_frontflip_integrated_state
    assert reset.params["stand_prob"] == 0.20
    assert reset.params["flight_prob"] == 0.35
    assert "backflip_assistance" not in cfg.events
    assert all(
        actuator.max_current == 1.75
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )


def test_integrated_task_enforces_official_wheel_friction():
    cfg = make_microduck_roller_frontflip_integrated_env_cfg()
    friction = cfg.events["official_wheel_friction"]
    assert friction.func is dr.dof_frictionloss
    assert friction.params["operation"] == "abs"
    assert friction.params["ranges"] == (
        OFFICIAL_WHEEL_FRICTION,
        OFFICIAL_WHEEL_FRICTION,
    )
    assert friction.params["asset_cfg"].joint_names == (r"^passive_.*wheel",)


def test_integrated_rewards_cover_launch_rotation_landing_and_durable_recovery():
    cfg = make_microduck_roller_frontflip_integrated_env_cfg()
    rewards = cfg.rewards
    assert rewards["rotation_progress"].func is microduck_mdp.roller_backflip_rotation_progress
    assert rewards["takeoff_pitch_momentum"].weight > 0.0
    assert rewards["landing_readiness"].params["minimum_rotation"] == math.radians(180.0)
    assert rewards["clean_skate_touchdown"].weight > rewards["landing_readiness"].weight
    assert rewards["rolling_recovery"].func is microduck_mdp.roller_frontflip_rolling_recovery
    assert rewards["rolling_recovery"].params["settle_seconds"] == 0.25
    assert rewards["non_skate_ground_contact"].weight < 0.0


def test_integrated_curriculum_moves_back_to_unassisted_starts_only_on_gates():
    cfg = make_microduck_roller_frontflip_integrated_env_cfg()
    curriculum = cfg.curriculum["integrated_phase_stitch"]
    assert curriculum.func is microduck_mdp.roller_frontflip_integrated_curriculum
    assert curriculum.params["min_attempts_per_kind"] == 512
    assert all(stage["required_windows"] == 2 for stage in INTEGRATED_STAGES)
    assert INTEGRATED_STAGES[0]["params"]["stand_prob"] == 0.20
    assert INTEGRATED_STAGES[-1]["params"]["stand_prob"] == 0.85
    assert INTEGRATED_STAGES[0]["params"]["flight_progress_range_deg"][0] == 220.0
    assert INTEGRATED_STAGES[-1]["params"]["flight_progress_range_deg"][0] == 30.0
    for stage in INTEGRATED_STAGES:
        params = stage["params"]
        assert params["stand_prob"] + params["flight_prob"] <= 1.0


def test_integrated_play_is_true_end_to_end_and_runner_is_conservative():
    cfg = make_microduck_roller_frontflip_integrated_env_cfg(play=True)
    reset = cfg.events["reset_backflip_state"].params
    assert reset["stand_prob"] == 1.0
    assert reset["flight_prob"] == 0.0
    assert not cfg.curriculum

    runner = MicroduckRollerFrontFlipIntegratedRlCfg
    assert runner.experiment_name == "roller_frontflip_integrated"
    assert runner.max_iterations == 2_000
    assert runner.save_interval == 25
    assert runner.algorithm.learning_rate == 1.0e-4
    assert runner.algorithm.clip_param == 0.10
    assert runner.algorithm.entropy_coef == 0.001
