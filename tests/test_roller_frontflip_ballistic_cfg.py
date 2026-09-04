import json

import pytest

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_ballistic_env_cfg import (
    _load_ballistic_prior,
    make_microduck_roller_frontflip_ballistic_env_cfg,
)


def test_ballistic_task_keeps_official_physics_and_momentum_reward(monkeypatch):
    monkeypatch.delenv("DUCKLAB_FRONTFLIP_BALLISTIC_REFERENCE", raising=False)
    cfg = make_microduck_roller_frontflip_ballistic_env_cfg()
    friction = cfg.events["official_wheel_friction"].params
    assert friction["operation"] == "abs"
    assert friction["ranges"] == (0.003, 0.003)
    assert cfg.events["reset_base"].params["velocity_range"]["x"] == (0.8, 0.8)
    assert (
        cfg.rewards["supported_pitch_angular_momentum"].func
        is microduck_mdp.roller_frontflip_supported_pitch_angular_momentum_progress
    )


def test_ballistic_reference_gate_and_action_prior(tmp_path, monkeypatch):
    reference = tmp_path / "prior.json"
    document = {
        "wheel_frictionloss": 0.003,
        "current_limit_a": 1.75,
        "knot_times_s": [0.0, 1.0],
        "max_rotation": {"rotation_deg": 305.0},
        "max_rotation_full_nodes": [[0.0] * 14, [0.0] * 14],
    }
    reference.write_text(json.dumps(document))
    monkeypatch.setenv("DUCKLAB_FRONTFLIP_BALLISTIC_REFERENCE", str(reference))
    times, nodes = _load_ballistic_prior()
    assert times == [0.0, 1.0]
    assert len(nodes[0]) == 14
    cfg = make_microduck_roller_frontflip_ballistic_env_cfg()
    assert (
        cfg.rewards["ballistic_action_prior"].func
        is microduck_mdp.roller_frontflip_ballistic_action_prior
    )

    document["max_rotation"]["rotation_deg"] = 299.9
    reference.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="300 degrees"):
        _load_ballistic_prior()
