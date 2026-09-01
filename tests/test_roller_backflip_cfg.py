import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_backflip_env_cfg import (
    DEMO_END_FRAME,
    DEMO_START_FRAME,
    LANDING_ROTATION,
    MicroduckRollerBackflipRlCfg,
    TARGET_CLEARANCE,
    TARGET_ROTATION,
    load_backflip_demonstration,
    make_microduck_roller_backflip_env_cfg,
)


def test_accepted_motion_drives_reverse_curriculum_without_waypoint_reward():
    demo = load_backflip_demonstration()
    assert len(demo["progress"]) == DEMO_END_FRAME - DEMO_START_FRAME
    assert demo["progress"][-1] > demo["progress"][0]
    assert demo["progress"][-1] >= LANDING_ROTATION
    assert all(row[1] == 0.0 and row[3] == 0.0 and row[5] == 0.0 for row in demo["root_qvel"])
    assert all(len(row) == 14 for row in demo["joint_pos"])

    cfg = make_microduck_roller_backflip_env_cfg()
    event = cfg.events["reset_backflip_state"]
    assert event.func is microduck_mdp.reset_roller_backflip_state
    assert event.params["demonstration"]["progress"] == demo["progress"]
    assert not any("reference" in name or "waypoint" in name for name in cfg.rewards)
    assist = cfg.events["apply_backflip_assistance"]
    assert assist.func is microduck_mdp.apply_roller_backflip_assistance
    assert assist.mode == "step"
    assert assist.params["ramp_start_clearance"] < assist.params["full_clearance"]


def test_backflip_has_hard_airborne_rotation_and_clean_landing_gates():
    cfg = make_microduck_roller_backflip_env_cfg()
    rewards = cfg.rewards
    rotation = rewards["backflip_rotation_progress"]
    clearance = rewards["backflip_clearance_progress"]
    landing = rewards["backflip_landing"]

    assert rotation.func is microduck_mdp.roller_backflip_rotation_progress
    assert math.isclose(rotation.params["target_rotation"], TARGET_ROTATION)
    assert clearance.params["target_clearance"] == TARGET_CLEARANCE
    assert landing.params["landing_rotation"] == LANDING_ROTATION
    assert rewards["backflip_takeoff_pitch"].func is microduck_mdp.roller_backflip_takeoff_pitch_progress
    assert rewards["backflip_takeoff_pitch"].weight > 0.0
    readiness = rewards["backflip_landing_readiness"]
    assert readiness.func is microduck_mdp.roller_backflip_landing_readiness_progress
    assert readiness.params["minimum_rotation"] >= math.radians(240.0)
    assert readiness.params["foot_drop_target"] >= 0.09
    assert rewards["backflip_post_landing_stability"].func is microduck_mdp.roller_backflip_post_landing_stability
    assert rewards["backflip_post_landing_stability"].weight > 0.0
    assert rewards["body_ground_contact"].weight < 0.0
    assert rewards["sagittal_motion"].weight < 0.0
    assert rewards["pitch_overspeed"].weight < 0.0
    termination = cfg.terminations["backflip_body_ground_contact"]
    assert termination.func is microduck_mdp.roller_backflip_body_ground_contact
    assert termination.params["sensor_name"] == "backflip_body_ground_contact"
    final_stage = cfg.curriculum["backflip_spawn_assistance"].params["stages"][-1]
    assert abs(final_stage["body_contact_weight"]) > (
        rewards["backflip_rotation_progress"].weight
        + rewards["backflip_clearance_progress"].weight
    )
    assert "tilt" not in rewards
    assert "hop_load_height" not in rewards
    assert any(sensor.name == "backflip_body_ground_contact" for sensor in cfg.scene.sensors)


def test_backflip_assistance_reaches_zero_and_play_is_unassisted():
    cfg = make_microduck_roller_backflip_env_cfg()
    curriculum = cfg.curriculum["backflip_spawn_assistance"]
    assert curriculum.func is microduck_mdp.backflip_performance_curriculum
    assert curriculum.params.get("min_episode_steps", 1) == 1
    stages = curriculum.params["stages"]
    assert stages[0]["params"]["demo_prob"] > stages[-1]["params"]["demo_prob"]
    assert stages[-1]["params"]["assist_vz_range"] == (0.0, 0.0)
    assert stages[-1]["params"]["assist_omega_range"] == (0.0, 0.0)
    assert stages[-1]["params"]["assist_turns_range"] == (0.0, 0.0)
    assert stages[0]["params"]["demo_frame_range"][0] > stages[-1]["params"]["demo_frame_range"][0]
    assert all(stage["required_windows"] >= 2 for stage in stages)
    assert all(stage["advance_stand_success"] > 0.0 for stage in stages)
    assert stages[-2]["advance_stand_success"] >= 0.02
    assert all(
        earlier["params"]["demo_frame_range"][0] >= later["params"]["demo_frame_range"][0]
        for earlier, later in zip(stages, stages[1:])
    )
    assert all(
        earlier["params"]["assist_vz_range"][1] >= later["params"]["assist_vz_range"][1]
        for earlier, later in zip(stages, stages[1:])
    )
    assert all(
        earlier["params"]["assist_turns_range"][1] >= later["params"]["assist_turns_range"][1]
        for earlier, later in zip(stages, stages[1:])
    )

    play = make_microduck_roller_backflip_env_cfg(play=True)
    event = play.events["reset_backflip_state"]
    assert event.params["demo_prob"] == 0.0
    assert event.params["assist_vz_range"] == (0.0, 0.0)
    assert event.params["assist_omega_range"] == (0.0, 0.0)
    assert event.params["assist_turns_range"] is None


def test_backflip_uses_action_free_two_frame_motion_prior():
    demo = load_backflip_demonstration()
    assert len(demo["style_frames"]) == len(demo["progress"])
    assert all(len(frame) == 37 for frame in demo["style_frames"])
    cfg = make_microduck_roller_backflip_env_cfg()
    assert list(cfg.observations["style"].terms) == [
        "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"
    ]
    algorithm = MicroduckRollerBackflipRlCfg.algorithm
    assert algorithm.class_name == "mjlab_microduck.wasabi.WasabiPPO"
    assert len(algorithm.expert_transitions) == len(demo["style_frames"]) - 1
    assert algorithm.discriminator_learning_rate <= 2.5e-5
    assert algorithm.discriminator_gradient_penalty >= 5.0


def test_backflip_discovery_delays_domain_randomization():
    cfg = make_microduck_roller_backflip_env_cfg()
    assert not cfg.observations["actor"].enable_corruption
    assert not {
        "randomize_wheel_friction", "randomize_com", "randomize_head_com",
        "randomize_armature", "encoder_bias",
        "base_com", "randomize_mass_inertia",
    }.intersection(cfg.events)
    assert cfg.events["randomize_joint_friction"].params["scale_range"] == (1.0, 1.0)
    assert cfg.events["expand_bam_friction_fields"].func is microduck_mdp.expand_bam_friction_fields


def test_backflip_runner_is_bounded_and_hot_swappable():
    cfg = make_microduck_roller_backflip_env_cfg()
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4
        assert cfg.observations[group].terms["body_command"].params["dim"] == 6
    assert MicroduckRollerBackflipRlCfg.experiment_name == "roller_backflip"
    assert MicroduckRollerBackflipRlCfg.max_iterations == 2_500
    assert MicroduckRollerBackflipRlCfg.save_interval == 100
