"""Rolling roller-backflip learned from an accepted arena demonstration.

The demonstration is a reverse-curriculum reset distribution, not a keyframe
reward: PPO must discover controls that satisfy state-based takeoff, airborne
rotation, clean-contact, and upright-landing gates. Initial root-velocity
assistance and mid-flip spawns are gradually reduced until standing/rolling
starts are fully unassisted.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_hop_env_cfg import (
    MicroduckRollerHopRlCfg,
    make_microduck_roller_hop_env_cfg,
)


EPISODE_LENGTH_S = 4.0
STAND_HEIGHT = 0.115
TARGET_CLEARANCE = 0.080
TAKEOFF_CLEARANCE = 0.010
TARGET_ROTATION = 2.0 * math.pi
LANDING_ROTATION = math.radians(300.0)
DEMO_START_FRAME = 132
DEMO_END_FRAME = 176
NUM_STEPS_PER_ENV = 24
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_SERVO_DOF_IDS = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
_DEMO_PATH = (
    Path(__file__).resolve().parents[3]
    / "datasets/demonstrations/backflip/rolling-backflip-v1.json"
)


def _quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    w, x, y, z = a
    W, X, Y, Z = b
    return (
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    )


def _normalized(values: list[float]) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values)


def load_backflip_demonstration(path: Path = _DEMO_PATH) -> dict:
    """Extract safe reset tensors and cumulative pitch from the curated clip."""
    document = json.loads(path.read_text())
    if not document.get("curation", {}).get("accepted"):
        raise ValueError(f"Backflip demonstration is not accepted: {path}")
    frames = document["frames"]
    quaternions: list[tuple[float, ...]] = []
    progress = [0.0]
    for frame in frames:
        quaternion = _normalized(frame["qpos"][3:7])
        if quaternions and sum(a * b for a, b in zip(quaternion, quaternions[-1])) < 0:
            quaternion = tuple(-value for value in quaternion)
        if quaternions:
            previous = quaternions[-1]
            delta = _normalized(list(_quat_mul(
                (previous[0], -previous[1], -previous[2], -previous[3]), quaternion
            )))
            if delta[0] < 0:
                delta = tuple(-value for value in delta)
            vector_length = math.sqrt(sum(value * value for value in delta[1:]))
            angle = 2.0 * math.atan2(vector_length, max(0.0, delta[0]))
            pitch_delta = 0.0 if vector_length < 1e-9 else delta[2] / vector_length * angle
            progress.append(max(progress[-1], progress[-1] + max(0.0, pitch_delta)))
        quaternions.append(quaternion)

    selected = frames[DEMO_START_FRAME:DEMO_END_FRAME]
    selected_progress = progress[DEMO_START_FRAME:DEMO_END_FRAME]
    return {
        "root_qpos": [frame["qpos"][:7] for frame in selected],
        # Preserve forward/vertical and demonstrated pitch speed, but strip
        # off-axis perturbation from the mouse-assisted recording.
        "root_qvel": [
            [frame["qvel"][0], 0.0, frame["qvel"][2], 0.0, frame["qvel"][4], 0.0]
            for frame in selected
        ],
        "joint_pos": [
            [frame["qpos"][7 + index] for index in _SERVO_DOF_IDS] for frame in selected
        ],
        "joint_vel": [
            [frame["qvel"][6 + index] for index in _SERVO_DOF_IDS] for frame in selected
        ],
        "progress": selected_progress,
    }


def make_microduck_roller_backflip_env_cfg(play: bool = False):
    cfg = make_microduck_roller_hop_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    body_ground = ContactSensorCfg(
        name="backflip_body_ground_contact",
        primary=ContactMatch(
            mode="body", pattern=r"^(trunk_base|jaw_soft)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, body_ground)

    cfg.commands["twist"].period = EPISODE_LENGTH_S
    cfg.commands["twist"].randomize_phase = False
    cfg.events.pop("reset_hop_state", None)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.112, 0.120)
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (0.12, 0.28), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events["reset_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_roller_backflip_state,
        mode="reset",
        params={
            "demonstration": load_backflip_demonstration(),
            "demo_prob": 0.0 if play else 0.65,
            "assist_vz_range": (0.0, 0.0) if play else (0.6, 0.9),
            "assist_omega_range": (0.0, 0.0) if play else (10.0, 15.0),
        },
    )

    state_params = {
        "feet_sensor_name": "feet_ground_contact",
        "body_sensor_name": "backflip_body_ground_contact",
        "stand_height": STAND_HEIGHT,
        "takeoff_clearance": TAKEOFF_CLEARANCE,
        "landing_rotation": LANDING_ROTATION,
    }
    cfg.rewards.clear()
    cfg.rewards["backflip_rotation_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_rotation_progress,
        weight=12.0,
        params={**state_params, "target_rotation": TARGET_ROTATION},
    )
    cfg.rewards["backflip_clearance_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_clearance_progress,
        weight=4.0,
        params={**state_params, "target_clearance": TARGET_CLEARANCE},
    )
    cfg.rewards["backflip_takeoff_velocity"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_takeoff_velocity,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact", "max_vz": 0.9,
            "phase_start": 0.0, "phase_end": 0.28,
        },
    )
    cfg.rewards["backflip_landing"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing,
        weight=6.0,
        params={**state_params, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["body_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        weight=-4.0,
        params={"sensor_name": "backflip_body_ground_contact"},
    )
    cfg.rewards["sagittal_motion"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_sagittal_cost,
        weight=-0.002,
    )
    cfg.rewards["pitch_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_overspeed_cost,
        weight=-0.005,
        params={"max_pitch_rate": 22.0},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-0.002,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1.0e-4,
    )
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": "self_collision"},
    )

    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["backflip_spawn_assistance"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "reset_backflip_state",
                "param_stages": [
                    {"step": 0, "params": {"demo_prob": 0.65, "assist_vz_range": (0.6, 0.9), "assist_omega_range": (10.0, 15.0)}},
                    {"step": 500 * NUM_STEPS_PER_ENV, "params": {"demo_prob": 0.50, "assist_vz_range": (0.35, 0.70), "assist_omega_range": (6.0, 12.0)}},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "params": {"demo_prob": 0.35, "assist_vz_range": (0.10, 0.40), "assist_omega_range": (2.0, 8.0)}},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"demo_prob": 0.20, "assist_vz_range": (0.0, 0.15), "assist_omega_range": (0.0, 3.0)}},
                    {"step": 2000 * NUM_STEPS_PER_ENV, "params": {"demo_prob": 0.10, "assist_vz_range": (0.0, 0.0), "assist_omega_range": (0.0, 0.0)}},
                ],
            },
        )
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.002},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.01},
                    {"step": 1800 * NUM_STEPS_PER_ENV, "weight": -0.03},
                ],
            },
        )
    return cfg


MicroduckRollerBackflipRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRollerHopRlCfg)
MicroduckRollerBackflipRlCfg.experiment_name = "roller_backflip"
MicroduckRollerBackflipRlCfg.run_name = "roller_backflip_v1"
MicroduckRollerBackflipRlCfg.max_iterations = 2_500
MicroduckRollerBackflipRlCfg.save_interval = 100
