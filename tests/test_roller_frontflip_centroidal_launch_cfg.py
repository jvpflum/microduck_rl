import json

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_centroidal_launch_env_cfg import (
    make_microduck_roller_frontflip_centroidal_launch_env_cfg,
)


def test_centroidal_launch_is_real_start_exact_physics(monkeypatch, tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({
        "wheel_frictionloss": 0.003,
        "current_limit_a": 1.75,
        "minimum_clean_rotation_deg": 227.0,
        "body_contact_rate": 1.0,
        "knot_times_s": [0.0, 1.6],
        "full_nodes": [[0.0] * 14, [0.0] * 14],
    }))
    monkeypatch.setenv("DUCKLAB_FRONTFLIP_CENTROIDAL_PRIOR", str(prior))
    cfg = make_microduck_roller_frontflip_centroidal_launch_env_cfg()
    assert cfg.events["official_wheel_friction"].params["ranges"] == (0.003, 0.003)
    assert cfg.events["reset_base"].params["velocity_range"]["x"] == (1.20, 1.20)
    assert cfg.events["reset_backflip_state"].params["demo_prob"] == 0.0
    assert all(
        actuator.max_current == 1.75
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )
    assert (
        cfg.rewards["centroidal_pitch_momentum"].func
        is microduck_mdp.roller_frontflip_supported_pitch_angular_momentum_progress
    )
    assert cfg.rewards["launch_pitch_momentum"].weight == 0.0
    assert cfg.rewards["non_skate_ground_contact"].weight <= -100.0
