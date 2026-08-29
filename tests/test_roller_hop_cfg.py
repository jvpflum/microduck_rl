import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_hop_env_cfg import (
    HOP_PERIOD,
    LOAD_HEIGHT,
    STAND_HEIGHT,
    TAKEOFF_LATCH_CLEARANCE,
    TARGET_CLEARANCE,
    MicroduckRollerHopRlCfg,
    make_microduck_roller_hop_env_cfg,
)


def test_hop_is_one_stationary_phase_cycle():
    cfg = make_microduck_roller_hop_env_cfg()
    command = cfg.commands["twist"]
    assert isinstance(command, microduck_mdp.GroundPickPhaseCommandCfg)
    assert command.period == HOP_PERIOD
    assert command.randomize_phase is False
    assert cfg.episode_length_s == HOP_PERIOD

    velocity_range = cfg.events["reset_base"].params["velocity_range"]
    assert all(bounds == (0.0, 0.0) for bounds in velocity_range.values())
    assert "push_robot" not in cfg.events
    assert "fell_over" not in cfg.terminations
    assert cfg.events["reset_hop_state"].func is microduck_mdp.reset_roller_hop_state


def test_hop_requires_real_takeoff_before_landing_rewards():
    cfg = make_microduck_roller_hop_env_cfg()
    rewards = cfg.rewards
    assert rewards["hop_clearance_progress"].params == {
        "sensor_name": "feet_ground_contact",
        "stand_height": STAND_HEIGHT,
        "target_clearance": TARGET_CLEARANCE,
        "latch_clearance": TAKEOFF_LATCH_CLEARANCE,
    }
    assert rewards["hop_landing"].func is microduck_mdp.roller_hop_landing_composite
    assert rewards["hop_landing_stillness"].func is microduck_mdp.roller_hop_landing_stillness
    assert "upright" not in rewards
    assert "crouch_glide_pose" not in rewards
    assert rewards["tilt"].weight < 0.0
    assert LOAD_HEIGHT < STAND_HEIGHT < STAND_HEIGHT + TARGET_CLEARANCE


def test_hop_preserves_roller_observation_and_sim2real_stack():
    cfg = make_microduck_roller_hop_env_cfg()
    robot = cfg.scene.entities["robot"]
    assert "roller" in str(robot).lower()
    assert "wheel_vel" in cfg.observations["critic"].terms
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4
        assert cfg.observations[group].terms["body_command"].params["dim"] == 6
        selector = cfg.observations[group].terms["joint_pos"].params["asset_cfg"]
        assert selector.joint_names == (r"^(?!passive_).*",)
    for event in (
        "randomize_com",
        "randomize_head_com",
        "randomize_mass_inertia",
        "randomize_joint_friction",
        "randomize_armature",
        "randomize_wheel_friction",
        "encoder_bias",
    ):
        assert event in cfg.events
    assert "nan_state" in cfg.terminations


def test_hop_regularization_is_delayed_and_runner_is_bounded():
    cfg = make_microduck_roller_hop_env_cfg()
    action_stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    impact_stages = cfg.curriculum["landing_impact_weight"].params["weight_stages"]
    assert action_stages[0]["weight"] > action_stages[-1]["weight"]
    assert impact_stages[0]["weight"] < impact_stages[-1]["weight"]
    assert MicroduckRollerHopRlCfg.max_iterations == 1_500
    assert MicroduckRollerHopRlCfg.save_interval == 100
    assert MicroduckRollerHopRlCfg.algorithm.symmetry_cfg is not None


def test_phase_window_is_zero_outside_and_smooth_inside():
    phase = torch.tensor([0.0, 0.08, 0.12, 0.20, 0.30, 0.34, 0.50])
    gate = microduck_mdp._phase_window(phase, 0.08, 0.34, fade=0.04)
    assert torch.equal(gate[[0, 1, 5, 6]], torch.zeros(4))
    assert gate[2] > 0.99
    assert gate[3] == 1.0
    assert gate[4] > 0.99
