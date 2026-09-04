"""End-to-end skate-only front flip with performance-gated phase stitching.

One actor sees a stratified mixture of unassisted rolling launches, accepted
mid-flight states, and accepted descending landing states.  The mixture moves
backward toward real standing starts only after independent windows preserve
launch, rotation, clean touchdown, and durable rolling recovery.
"""

import math
from copy import deepcopy

from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_backflip_env_cfg import (
    LANDING_ROTATION,
    STAND_HEIGHT,
    TAKEOFF_CLEARANCE,
    load_backflip_demonstration,
)
from mjlab_microduck.tasks.microduck_roller_frontflip_landing_env_cfg import (
    MicroduckRollerFrontFlipLandingRlCfg,
    make_microduck_roller_frontflip_landing_env_cfg,
)


EPISODE_LENGTH_S = 2.5
TARGET_CLEARANCE = 0.050
TARGET_ROTATION = 2.0 * math.pi
OFFICIAL_WHEEL_FRICTION = 0.003
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


INTEGRATED_STAGES = [
    {
        "required_windows": 2,
        "advance": {
            "stand_takeoff_rate": 0.25,
            "flight_landing_rate": 0.02,
            "landing_stable_rate": 0.02,
            "max_invalid_rate": 0.25,
        },
        "params": {
            "stand_prob": 0.20,
            "flight_prob": 0.35,
            "flight_progress_range_deg": (220.0, 320.0),
            "landing_progress_range_deg": (330.0, 355.0),
            "landing_height_offset_range": (0.03, 0.06),
            "landing_forward_speed_range": (0.20, 0.50),
            "landing_velocity_scale_range": (0.70, 0.90),
            "landing_offaxis_scale": 0.0,
        },
    },
    {
        "required_windows": 2,
        "advance": {
            "stand_takeoff_rate": 0.35,
            "flight_landing_rate": 0.04,
            "landing_stable_rate": 0.05,
            "max_invalid_rate": 0.30,
        },
        "params": {
            "stand_prob": 0.30,
            "flight_prob": 0.45,
            "flight_progress_range_deg": (180.0, 320.0),
            "landing_progress_range_deg": (310.0, 355.0),
            "landing_height_offset_range": (0.04, 0.08),
            "landing_forward_speed_range": (0.20, 0.70),
            "landing_velocity_scale_range": (0.80, 1.00),
            "landing_offaxis_scale": 0.05,
        },
    },
    {
        "required_windows": 2,
        "advance": {
            "stand_takeoff_rate": 0.45,
            "stand_rotation_rate": 0.005,
            "flight_landing_rate": 0.08,
            "landing_stable_rate": 0.08,
            "max_invalid_rate": 0.35,
        },
        "params": {
            "stand_prob": 0.50,
            "flight_prob": 0.35,
            "flight_progress_range_deg": (120.0, 320.0),
            "landing_progress_range_deg": (285.0, 350.0),
            "landing_height_offset_range": (0.04, 0.10),
            "landing_forward_speed_range": (0.30, 0.90),
            "landing_velocity_scale_range": (0.85, 1.05),
            "landing_offaxis_scale": 0.10,
        },
    },
    {
        "required_windows": 2,
        "advance": {
            "stand_landing_rate": 0.002,
            "stand_stable_rate": 0.001,
            "flight_landing_rate": 0.10,
            "landing_stable_rate": 0.10,
            "max_invalid_rate": 0.40,
        },
        "params": {
            "stand_prob": 0.70,
            "flight_prob": 0.20,
            "flight_progress_range_deg": (60.0, 320.0),
            "landing_progress_range_deg": (260.0, 345.0),
            "landing_height_offset_range": (0.05, 0.12),
            "landing_forward_speed_range": (0.40, 1.20),
            "landing_velocity_scale_range": (0.90, 1.10),
            "landing_offaxis_scale": 0.20,
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
            "landing_forward_speed_range": (0.40, 1.20),
            "landing_velocity_scale_range": (0.90, 1.15),
            "landing_offaxis_scale": 0.30,
        },
    },
]


def make_microduck_roller_frontflip_integrated_env_cfg(play: bool = False):
    cfg = make_microduck_roller_frontflip_landing_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.commands["twist"].period = EPISODE_LENGTH_S
    cfg.events.pop("backflip_assistance", None)

    params = deepcopy(INTEGRATED_STAGES[0]["params"])
    if play:
        params.update({"stand_prob": 1.0, "flight_prob": 0.0})
    cfg.events["reset_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_roller_frontflip_integrated_state,
        mode="reset",
        params={
            "demonstration": load_backflip_demonstration(),
            "stand_height": STAND_HEIGHT,
            **params,
        },
    )
    # The roller MJCF uses zero passive-joint drag by default.  Make official
    # 0.003 friction explicit instead of relying on a removed DR event.
    cfg.events["official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^passive_.*wheel",)
            ),
            "operation": "abs",
            "ranges": (OFFICIAL_WHEEL_FRICTION, OFFICIAL_WHEEL_FRICTION),
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
    cfg.rewards["rotation_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_rotation_progress,
        weight=30.0,
        params={**state, "target_rotation": TARGET_ROTATION},
    )
    cfg.rewards["clearance_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_clearance_progress,
        weight=18.0,
        params={**state, "target_clearance": TARGET_CLEARANCE},
    )
    cfg.rewards["takeoff_vertical_velocity"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_takeoff_velocity,
        weight=6.0,
        params={
            "sensor_name": "feet_ground_contact",
            "max_vz": 1.5,
            "phase_start": 0.0,
            "phase_end": 0.55,
        },
    )
    cfg.rewards["takeoff_pitch_momentum"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_takeoff_pitch_progress,
        weight=12.0,
        params={"feet_sensor_name": "feet_ground_contact", "target_pitch_rate": 30.0},
    )
    cfg.rewards["landing_readiness"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing_readiness_progress,
        weight=70.0,
        params={
            **state,
            "minimum_rotation": math.radians(180.0),
            "foot_drop_target": 0.10,
        },
    )
    cfg.rewards["clean_skate_touchdown"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing,
        weight=220.0,
        params={
            **state,
            "joint_indices": _LEG_JOINTS,
            "forward_speed_tolerance": 1.5,
        },
    )
    cfg.rewards["rolling_recovery"] = RewardTermCfg(
        func=microduck_mdp.roller_frontflip_rolling_recovery,
        weight=140.0,
        params={
            **state,
            "forward_speed_tolerance": 1.5,
            "settle_seconds": 0.25,
        },
    )
    cfg.rewards["non_skate_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        weight=-50.0,
        params={"sensor_name": "backflip_body_ground_contact"},
    )
    cfg.rewards["sagittal_motion"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_sagittal_cost,
        weight=-0.08,
    )
    cfg.rewards["pitch_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_overspeed_cost,
        weight=-0.003,
        params={"max_pitch_rate": 28.0},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_phase_action_rate_l2,
        weight=-2.0e-5,
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
        cfg.curriculum["integrated_phase_stitch"] = CurriculumTermCfg(
            func=microduck_mdp.roller_frontflip_integrated_curriculum,
            params={
                "event_name": "reset_backflip_state",
                "stages": INTEGRATED_STAGES,
                "min_attempts_per_kind": 512,
                "landing_rotation": LANDING_ROTATION,
            },
        )
    return cfg


MicroduckRollerFrontFlipIntegratedRlCfg = deepcopy(
    MicroduckRollerFrontFlipLandingRlCfg
)
MicroduckRollerFrontFlipIntegratedRlCfg.experiment_name = (
    "roller_frontflip_integrated"
)
MicroduckRollerFrontFlipIntegratedRlCfg.run_name = "roller_frontflip_integrated_v1"
MicroduckRollerFrontFlipIntegratedRlCfg.max_iterations = 2_000
MicroduckRollerFrontFlipIntegratedRlCfg.save_interval = 25
MicroduckRollerFrontFlipIntegratedRlCfg.algorithm.learning_rate = 1.0e-4
MicroduckRollerFrontFlipIntegratedRlCfg.algorithm.clip_param = 0.10
MicroduckRollerFrontFlipIntegratedRlCfg.algorithm.entropy_coef = 0.001
