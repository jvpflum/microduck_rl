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
    assert rewards["body_ground_contact"].weight < 0.0
    assert rewards["sagittal_motion"].weight < 0.0
    assert rewards["pitch_overspeed"].weight < 0.0
    assert "tilt" not in rewards
    assert "hop_load_height" not in rewards
    assert any(sensor.name == "backflip_body_ground_contact" for sensor in cfg.scene.sensors)


def test_backflip_assistance_reaches_zero_and_play_is_unassisted():
    cfg = make_microduck_roller_backflip_env_cfg()
    stages = cfg.curriculum["backflip_spawn_assistance"].params["param_stages"]
    assert stages[0]["params"]["demo_prob"] > stages[-1]["params"]["demo_prob"]
    assert stages[-1]["params"]["assist_vz_range"] == (0.0, 0.0)
    assert stages[-1]["params"]["assist_omega_range"] == (0.0, 0.0)

    play = make_microduck_roller_backflip_env_cfg(play=True)
    event = play.events["reset_backflip_state"]
    assert event.params["demo_prob"] == 0.0
    assert event.params["assist_vz_range"] == (0.0, 0.0)
    assert event.params["assist_omega_range"] == (0.0, 0.0)


def test_backflip_runner_is_bounded_and_hot_swappable():
    cfg = make_microduck_roller_backflip_env_cfg()
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4
        assert cfg.observations[group].terms["body_command"].params["dim"] == 6
    assert MicroduckRollerBackflipRlCfg.experiment_name == "roller_backflip"
    assert MicroduckRollerBackflipRlCfg.max_iterations == 2_500
    assert MicroduckRollerBackflipRlCfg.save_interval == 100
