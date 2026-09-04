"""Dedicated V69-to-V50 closed-loop flight bridge.

This actor is never responsible for generating the launch.  It starts from a
reset-state archive containing exact native V69 flight states and accepted V50
landing states, then moves backward only after it can finish rotation and land.
That separation prevents the catastrophic launch forgetting seen in V50/V51.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path

from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_backflip_env_cfg import (
    LANDING_ROTATION,
    STAND_HEIGHT,
    TAKEOFF_CLEARANCE,
)
from mjlab_microduck.tasks.microduck_roller_frontflip_landing_env_cfg import (
    MicroduckRollerFrontFlipLandingRlCfg,
    make_microduck_roller_frontflip_landing_env_cfg,
)


EPISODE_LENGTH_S = 1.8
TARGET_ROTATION = 2.0 * math.pi
OFFICIAL_WHEEL_FRICTION = 0.003
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_DEFAULT_REFERENCE = Path(
    "/home/jarro/ducklab-private/frontflip-v21/v69-v50-bridge-reference.json"
)


BRIDGE_STAGES = [
    {
        "required_windows": 2,
        "advance_rotation_rate": 0.85,
        "advance_landing_rate": 0.55,
        # PPO rollouts include action noise: the protected V50's 43% native
        # deterministic stable rate measures about 15% in this signal.
        "advance_stable_rate": 0.12,
        "max_invalid_rate": 0.12,
        "params": {"demo_progress_range_deg": (300.0, 355.0)},
    },
    {
        "required_windows": 2,
        "advance_rotation_rate": 0.75,
        "advance_landing_rate": 0.40,
        "advance_stable_rate": 0.08,
        "max_invalid_rate": 0.16,
        "params": {"demo_progress_range_deg": (260.0, 355.0)},
    },
    {
        "required_windows": 2,
        "advance_rotation_rate": 0.60,
        "advance_landing_rate": 0.25,
        "advance_stable_rate": 0.04,
        "max_invalid_rate": 0.22,
        "params": {"demo_progress_range_deg": (220.0, 355.0)},
    },
    {
        "required_windows": 2,
        "advance_rotation_rate": 0.45,
        "advance_landing_rate": 0.15,
        "advance_stable_rate": 0.02,
        "max_invalid_rate": 0.30,
        "params": {"demo_progress_range_deg": (180.0, 355.0)},
    },
    {
        "required_windows": 2,
        "advance_rotation_rate": 0.30,
        "advance_landing_rate": 0.08,
        "advance_stable_rate": 0.01,
        "max_invalid_rate": 0.40,
        "params": {"demo_progress_range_deg": (145.0, 355.0)},
    },
    {
        "required_windows": 2,
        "advance_rotation_rate": 0.15,
        "advance_landing_rate": 0.03,
        "advance_stable_rate": 0.0,
        "max_invalid_rate": 0.55,
        "params": {"demo_progress_range_deg": (115.0, 355.0)},
    },
    {
        "required_windows": 2,
        "advance_rotation_rate": 0.05,
        "advance_landing_rate": 0.005,
        "advance_stable_rate": 0.0,
        "max_invalid_rate": 0.70,
        "params": {"demo_progress_range_deg": (95.0, 145.0)},
    },
    {
        "required_windows": 2,
        "advance_rotation_rate": 1.1,
        "max_invalid_rate": 1.0,
        "params": {"demo_progress_range_deg": (80.0, 355.0)},
    },
]


def load_bridge_reference() -> dict:
    path = Path(os.environ.get("DUCKLAB_FRONTFLIP_BRIDGE_REFERENCE", _DEFAULT_REFERENCE))
    document = json.loads(path.read_text())
    if not math.isclose(float(document["wheel_frictionloss"]), 0.003, abs_tol=1e-12):
        raise ValueError(f"bridge reference wheel friction is not 0.003: {path}")
    if not math.isclose(float(document["current_limit_a"]), 1.75, abs_tol=1e-12):
        raise ValueError(f"bridge reference current limit is not 1.75 A: {path}")
    return document["reference"]


def make_microduck_roller_frontflip_bridge_env_cfg(play: bool = False):
    cfg = make_microduck_roller_frontflip_landing_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.commands["twist"].period = 2.0
    cfg.commands["twist"].randomize_phase = False

    # Discovery and proof use the same mechanics.  No wheel-friction, mass,
    # voltage, or actuator-delay randomization is allowed in this phase.
    for name in (
        "randomize_wheel_friction",
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_mass_inertia",
        "randomize_joint_friction",
        "encoder_bias",
        "base_com",
    ):
        cfg.events.pop(name, None)
    cfg.events["official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (OFFICIAL_WHEEL_FRICTION, OFFICIAL_WHEEL_FRICTION),
        },
    )

    stage = BRIDGE_STAGES[-1] if play else BRIDGE_STAGES[0]
    cfg.events["reset_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_roller_frontflip_bridge_state,
        mode="reset",
        params={
            "demonstration": load_bridge_reference(),
            **stage["params"],
            "wheel_radius": 0.0175,
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
        weight=55.0,
        params={**state, "target_rotation": TARGET_ROTATION},
    )
    cfg.rewards["landing_readiness"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing_readiness_progress,
        weight=110.0,
        params={
            **state,
            "minimum_rotation": math.radians(120.0),
            "foot_drop_target": 0.10,
        },
    )
    cfg.rewards["skate_touchdown"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing,
        weight=260.0,
        params={**state, "joint_indices": _LEG_JOINTS, "forward_speed_tolerance": 1.5},
    )
    cfg.rewards["rolling_recovery"] = RewardTermCfg(
        func=microduck_mdp.roller_frontflip_rolling_recovery,
        weight=180.0,
        params={**state, "forward_speed_tolerance": 1.5, "settle_seconds": 0.25},
    )
    cfg.rewards["non_skate_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        weight=-160.0,
        params={"sensor_name": "backflip_body_ground_contact"},
    )
    cfg.rewards["sagittal_motion"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_sagittal_cost,
        weight=-0.12,
    )
    cfg.rewards["pitch_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_overspeed_cost,
        weight=-0.004,
        params={"max_pitch_rate": 26.0},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_phase_action_rate_l2,
        weight=-3.0e-5,
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
        cfg.curriculum["v69_to_v50_bridge"] = CurriculumTermCfg(
            func=microduck_mdp.roller_frontflip_bridge_curriculum,
            params={
                "event_name": "reset_backflip_state",
                "stages": BRIDGE_STAGES,
                # At 12,288 envs a 4,096 window advanced nearly a whole stage
                # per optimizer update.  32,768 supplies roughly ten rollout
                # iterations per independent gate before widening the problem.
                "min_attempts": 32768,
                "landing_rotation": LANDING_ROTATION,
            },
        )
    if os.environ.get("DUCKLAB_FRONTFLIP_PHYSICAL_ROLLIN") == "1":
        # Do not teleport into flight.  Every episode physically executes V69
        # through the real actuator/contact stack before the actor takes over.
        cfg.episode_length_s = 2.5
        cfg.curriculum.clear()
        cfg.events["reset_backflip_state"] = EventTermCfg(
            func=microduck_mdp.reset_roller_frontflip_rollin_state,
            mode="reset",
            params={
                "demonstration": load_bridge_reference(),
                "forward_speed_range": (
                    float(os.environ.get("DUCKLAB_FRONTFLIP_ROLLIN_SPEED_MIN", "0.60")),
                    float(os.environ.get("DUCKLAB_FRONTFLIP_ROLLIN_SPEED_MAX", "1.00")),
                ),
                "wheel_radius": 0.0175,
            },
        )
        cfg.commands["twist"].period = 2.0
        cfg.commands["twist"].randomize_phase = False
        cfg.commands["twist"].rollin_phase = True
        cfg.actions["joint_pos"].frontflip_rollin = {
            "primitive_path": str(
                Path("/home/jarro/ducklab-private/frontflip-v19/v69-best-momentum-packaged.json")
            ),
            "handoff_time_min": float(
                os.environ.get("DUCKLAB_FRONTFLIP_ROLLIN_HANDOFF_MIN", "0.70")
            ),
            "handoff_time_max": float(
                os.environ.get("DUCKLAB_FRONTFLIP_ROLLIN_HANDOFF_MAX", "0.82")
            ),
            "blend_time": float(
                os.environ.get("DUCKLAB_FRONTFLIP_ROLLIN_BLEND", "0.18")
            ),
        }
    return cfg


MicroduckRollerFrontFlipBridgeRlCfg = deepcopy(MicroduckRollerFrontFlipLandingRlCfg)
MicroduckRollerFrontFlipBridgeRlCfg.experiment_name = "roller_frontflip_bridge"
MicroduckRollerFrontFlipBridgeRlCfg.run_name = "roller_frontflip_bridge_v1"
MicroduckRollerFrontFlipBridgeRlCfg.max_iterations = 2_400
MicroduckRollerFrontFlipBridgeRlCfg.save_interval = 25
MicroduckRollerFrontFlipBridgeRlCfg.algorithm.learning_rate = 3.0e-5
MicroduckRollerFrontFlipBridgeRlCfg.algorithm.clip_param = 0.08
MicroduckRollerFrontFlipBridgeRlCfg.algorithm.entropy_coef = 0.002
