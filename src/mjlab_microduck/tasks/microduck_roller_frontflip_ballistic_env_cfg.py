"""Rolling front-flip stitch task using a searched ballistic launch prior."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_integrated_env_cfg import (
    LANDING_ROTATION,
    MicroduckRollerFrontFlipIntegratedRlCfg,
    make_microduck_roller_frontflip_integrated_env_cfg,
)


BALLISTIC_STAGES = [
    {
        "required_windows": 2,
        "advance": {
            "stand_rotation_rate": 0.0002,
            "flight_landing_rate": 0.08,
            "landing_stable_rate": 0.02,
            "max_invalid_rate": 0.50,
        },
        "params": {
            "stand_prob": 0.55,
            "flight_prob": 0.25,
            "flight_progress_range_deg": (120.0, 320.0),
            "landing_progress_range_deg": (285.0, 350.0),
            "landing_height_offset_range": (0.04, 0.10),
            "landing_forward_speed_range": (0.60, 1.00),
            "landing_velocity_scale_range": (0.85, 1.05),
            "landing_offaxis_scale": 0.10,
        },
    },
    {
        "required_windows": 2,
        "advance": {
            "stand_landing_rate": 0.0002,
            "stand_stable_rate": 0.0001,
            "flight_landing_rate": 0.10,
            "landing_stable_rate": 0.05,
            "max_invalid_rate": 0.55,
        },
        "params": {
            "stand_prob": 0.70,
            "flight_prob": 0.20,
            "flight_progress_range_deg": (60.0, 320.0),
            "landing_progress_range_deg": (260.0, 345.0),
            "landing_height_offset_range": (0.05, 0.12),
            "landing_forward_speed_range": (0.60, 1.00),
            "landing_velocity_scale_range": (0.90, 1.10),
            "landing_offaxis_scale": 0.15,
        },
    },
    {
        "required_windows": 2,
        "advance": {},
        "params": {
            "stand_prob": 0.85,
            "flight_prob": 0.10,
            "flight_progress_range_deg": (30.0, 320.0),
            "landing_progress_range_deg": (240.0, 345.0),
            "landing_height_offset_range": (0.05, 0.14),
            "landing_forward_speed_range": (0.60, 1.00),
            "landing_velocity_scale_range": (0.90, 1.15),
            "landing_offaxis_scale": 0.20,
        },
    },
]


def _load_ballistic_prior() -> tuple[list[float], list[list[float]]] | None:
    raw_path = os.environ.get("DUCKLAB_FRONTFLIP_BALLISTIC_REFERENCE")
    if not raw_path:
        return None
    path = Path(raw_path)
    document = json.loads(path.read_text())
    if not math.isclose(float(document["wheel_frictionloss"]), 0.003, abs_tol=1e-12):
        raise ValueError(f"ballistic prior was not searched at exact 0.003 friction: {path}")
    if not math.isclose(float(document["current_limit_a"]), 1.75, abs_tol=1e-12):
        raise ValueError(f"ballistic prior was not searched at 1.75 A: {path}")
    result = document.get("max_rotation", document.get("best", {}))
    if float(result.get("rotation_deg", 0.0)) < 300.0:
        raise ValueError(f"ballistic prior has not crossed 300 degrees: {path}")
    nodes = document.get("max_rotation_full_nodes", document.get("full_nodes"))
    if not nodes:
        raise ValueError(f"ballistic prior does not contain full action nodes: {path}")
    return list(document["knot_times_s"]), nodes


def make_microduck_roller_frontflip_ballistic_env_cfg(play: bool = False):
    cfg = make_microduck_roller_frontflip_integrated_env_cfg(play=play)
    # Discovery begins at a tightly controlled rolling entry.  A separate
    # button-triggered run-up controller can acquire this speed before the
    # final one-shot skill is invoked.
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (0.80, 0.80) if not play else (0.80, 0.80),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    cfg.rewards["rotation_progress"].weight = 60.0
    cfg.rewards["clearance_progress"].weight = 24.0
    cfg.rewards["takeoff_pitch_momentum"].weight = 8.0
    cfg.rewards["supported_pitch_angular_momentum"] = RewardTermCfg(
        func=microduck_mdp.roller_frontflip_supported_pitch_angular_momentum_progress,
        weight=28.0,
        params={
            "feet_sensor_name": "feet_ground_contact",
            "target_momentum": 0.055,
            "sensor_index": 5,
        },
    )
    prior = _load_ballistic_prior()
    if prior is not None:
        knot_times_s, full_nodes = prior
        cfg.rewards["ballistic_action_prior"] = RewardTermCfg(
            func=microduck_mdp.roller_frontflip_ballistic_action_prior,
            weight=8.0,
            params={
                "knot_times_s": knot_times_s,
                "full_nodes": full_nodes,
                "action_std": 1.0,
                "end_time_s": 0.96,
                "decay_steps": 7_200,
                "final_scale": 0.10,
            },
        )

    if not play:
        cfg.curriculum["integrated_phase_stitch"] = CurriculumTermCfg(
            func=microduck_mdp.roller_frontflip_integrated_curriculum,
            params={
                "event_name": "reset_backflip_state",
                "stages": BALLISTIC_STAGES,
                "min_attempts_per_kind": 512,
                "landing_rotation": LANDING_ROTATION,
            },
        )
        cfg.events["reset_backflip_state"].params.update(
            deepcopy(BALLISTIC_STAGES[0]["params"])
        )
    return cfg


MicroduckRollerFrontFlipBallisticRlCfg = deepcopy(
    MicroduckRollerFrontFlipIntegratedRlCfg
)
MicroduckRollerFrontFlipBallisticRlCfg.experiment_name = (
    "roller_frontflip_ballistic"
)
MicroduckRollerFrontFlipBallisticRlCfg.run_name = "roller_frontflip_ballistic_v1"
MicroduckRollerFrontFlipBallisticRlCfg.max_iterations = 1_200
MicroduckRollerFrontFlipBallisticRlCfg.save_interval = 25
MicroduckRollerFrontFlipBallisticRlCfg.algorithm.learning_rate = 8.0e-5
MicroduckRollerFrontFlipBallisticRlCfg.algorithm.clip_param = 0.08
MicroduckRollerFrontFlipBallisticRlCfg.algorithm.entropy_coef = 0.0005
