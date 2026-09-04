"""Backward curriculum for a clean skate-only front-flip landing.

The policy begins in accepted descending late-flight states and learns
untuck/alignment, tire-only touchdown, impact absorption, and quiet recovery.
Only measured landing performance widens the reset distribution backward.
"""

import math
from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_backflip_env_cfg import (
    LANDING_ROTATION,
    STAND_HEIGHT,
    TAKEOFF_CLEARANCE,
    load_backflip_demonstration,
)
from mjlab_microduck.tasks.microduck_roller_frontflip_flight_env_cfg import (
    MicroduckRollerFrontFlipFlightRlCfg,
    make_microduck_roller_frontflip_flight_env_cfg,
)


EPISODE_LENGTH_S = 1.4
TARGET_ROTATION = 2.0 * math.pi
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


LANDING_STAGES = [
    {
        "advance_landing_rate": 0.55,
        "advance_stable_rate": 0.15,
        "max_invalid_rate": 0.35,
        "required_windows": 2,
        "params": {
            "progress_range_deg": (330.0, 355.0),
            "height_offset_range": (0.03, 0.06),
            "forward_speed_range": (0.20, 0.50),
            "velocity_scale_range": (0.70, 0.90),
            "offaxis_scale": 0.0,
        },
    },
    {
        "advance_landing_rate": 0.50,
        "advance_stable_rate": 0.20,
        "max_invalid_rate": 0.40,
        "required_windows": 2,
        "params": {
            "progress_range_deg": (310.0, 355.0),
            "height_offset_range": (0.04, 0.08),
            "forward_speed_range": (0.20, 0.70),
            "velocity_scale_range": (0.80, 1.00),
            "offaxis_scale": 0.05,
        },
    },
    {
        "advance_landing_rate": 0.45,
        "advance_stable_rate": 0.25,
        "max_invalid_rate": 0.45,
        "required_windows": 2,
        "params": {
            "progress_range_deg": (285.0, 350.0),
            "height_offset_range": (0.04, 0.10),
            "forward_speed_range": (0.30, 0.90),
            "velocity_scale_range": (0.85, 1.05),
            "offaxis_scale": 0.10,
        },
    },
    {
        "advance_landing_rate": 0.40,
        "advance_stable_rate": 0.25,
        "max_invalid_rate": 0.50,
        "required_windows": 2,
        "params": {
            "progress_range_deg": (260.0, 345.0),
            "height_offset_range": (0.05, 0.12),
            "forward_speed_range": (0.40, 1.20),
            "velocity_scale_range": (0.90, 1.10),
            "offaxis_scale": 0.20,
        },
    },
    {
        "advance_landing_rate": 1.1,
        "advance_stable_rate": 1.1,
        "max_invalid_rate": 1.0,
        "required_windows": 2,
        "params": {
            "progress_range_deg": (240.0, 345.0),
            "height_offset_range": (0.05, 0.14),
            "forward_speed_range": (0.40, 1.20),
            "velocity_scale_range": (0.90, 1.15),
            "offaxis_scale": 0.30,
        },
    },
]


def make_microduck_roller_frontflip_landing_env_cfg(play: bool = False):
    cfg = make_microduck_roller_frontflip_flight_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.commands["twist"].period = EPISODE_LENGTH_S

    stage = LANDING_STAGES[-1] if play else LANDING_STAGES[0]
    cfg.events["reset_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_roller_frontflip_landing_state,
        mode="reset",
        params={
            "demonstration": load_backflip_demonstration(),
            "stand_height": STAND_HEIGHT,
            **stage["params"],
        },
    )

    state = {
        "feet_sensor_name": "feet_ground_contact",
        "body_sensor_name": "backflip_body_ground_contact",
        "stand_height": STAND_HEIGHT,
        "takeoff_clearance": TAKEOFF_CLEARANCE,
        "landing_rotation": LANDING_ROTATION,
    }
    cfg.rewards.clear()
    cfg.rewards["finish_rotation"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_rotation_progress,
        weight=15.0,
        params={**state, "target_rotation": TARGET_ROTATION},
    )
    cfg.rewards["landing_readiness"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing_readiness_progress,
        weight=90.0,
        params={**state, "minimum_rotation": math.radians(300.0), "foot_drop_target": 0.10},
    )
    cfg.rewards["skate_touchdown"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing,
        weight=220.0,
        params={
            **state,
            "joint_indices": _LEG_JOINTS,
            "forward_speed_tolerance": 1.5,
        },
    )
    cfg.rewards["post_landing_stability"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_post_landing_stability,
        weight=120.0,
        params={
            **state,
            "joint_indices": _LEG_JOINTS,
            "forward_speed_tolerance": 1.5,
        },
    )
    cfg.rewards["non_skate_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        weight=-40.0,
        params={"sensor_name": "backflip_body_ground_contact"},
    )
    cfg.rewards["sagittal_motion"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_sagittal_cost,
        weight=-0.05,
    )
    cfg.rewards["pitch_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_overspeed_cost,
        weight=-0.005,
        params={"max_pitch_rate": 22.0},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_phase_action_rate_l2,
        weight=-5.0e-5,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2,
        weight=-1.0e-5,
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": "self_collision"},
    )

    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["landing_backward_progress"] = CurriculumTermCfg(
            func=microduck_mdp.roller_frontflip_landing_curriculum,
            params={
                "event_name": "reset_backflip_state",
                "stages": LANDING_STAGES,
                "min_attempts": 4096,
                "forward_speed_tolerance": 1.5,
            },
        )
    return cfg


MicroduckRollerFrontFlipLandingRlCfg = deepcopy(MicroduckRollerFrontFlipFlightRlCfg)
MicroduckRollerFrontFlipLandingRlCfg.experiment_name = "roller_frontflip_landing"
MicroduckRollerFrontFlipLandingRlCfg.run_name = "roller_frontflip_landing_v1"
MicroduckRollerFrontFlipLandingRlCfg.max_iterations = 1_000
MicroduckRollerFrontFlipLandingRlCfg.save_interval = 25
MicroduckRollerFrontFlipLandingRlCfg.algorithm.learning_rate = 2.0e-4
MicroduckRollerFrontFlipLandingRlCfg.algorithm.clip_param = 0.15
MicroduckRollerFrontFlipLandingRlCfg.algorithm.entropy_coef = 0.005
