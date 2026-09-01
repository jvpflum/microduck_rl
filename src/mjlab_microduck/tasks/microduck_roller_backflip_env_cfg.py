"""Rolling roller front flip learned from an accepted arena demonstration.

The module and task IDs retain ``backflip`` for checkpoint compatibility. In
MicroDuck's coordinate convention positive body-Y pitch is nose-down/forward,
so this environment trains a front flip.

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

from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_hop_env_cfg import (
    MicroduckRollerHopRlCfg,
    make_microduck_roller_hop_env_cfg,
)
from mjlab_microduck.wasabi import WasabiPpoAlgorithmCfg


EPISODE_LENGTH_S = 4.0
STAND_HEIGHT = 0.115
TARGET_CLEARANCE = 0.080
TAKEOFF_CLEARANCE = 0.010
TARGET_ROTATION = 2.0 * math.pi
LANDING_ROTATION = math.radians(300.0)
DEMO_START_FRAME = 132
DEMO_END_FRAME = 201
NUM_STEPS_PER_ENV = 24
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_SERVO_DOF_IDS = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
_DEFAULT_POSE = [
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
]
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


def _quat_rotate_inverse(q: tuple[float, ...], vector: list[float]) -> list[float]:
    """Rotate a world-frame vector into the quaternion's local frame."""
    pure = (0.0, vector[0], vector[1], vector[2])
    inverse = (q[0], -q[1], -q[2], -q[3])
    result = _quat_mul(_quat_mul(inverse, pure), q)
    return list(result[1:])


def load_backflip_demonstration(path: Path = _DEMO_PATH) -> dict:
    """Extract front-flip reset tensors and cumulative forward pitch."""
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
    joint_pos = [
        [frame["qpos"][7 + index] for index in _SERVO_DOF_IDS] for frame in selected
    ]
    joint_vel = [
        [frame["qvel"][6 + index] for index in _SERVO_DOF_IDS] for frame in selected
    ]
    style_frames: list[list[float]] = []
    for frame, quaternion, positions, velocities in zip(
        selected,
        quaternions[DEMO_START_FRAME:DEMO_END_FRAME],
        joint_pos,
        joint_vel,
        strict=True,
    ):
        root_velocity = frame["qvel"][:6]
        style_frames.append(
            _quat_rotate_inverse(quaternion, root_velocity[:3])
            + _quat_rotate_inverse(quaternion, root_velocity[3:6])
            + _quat_rotate_inverse(quaternion, [0.0, 0.0, -1.0])
            + [value - default for value, default in zip(positions, _DEFAULT_POSE, strict=True)]
            + velocities
        )
    return {
        "root_qpos": [frame["qpos"][:7] for frame in selected],
        # Preserve forward/vertical and demonstrated pitch speed, but strip
        # off-axis perturbation from the mouse-assisted recording.
        "root_qvel": [
            [frame["qvel"][0], 0.0, frame["qvel"][2], 0.0, frame["qvel"][4], 0.0]
            for frame in selected
        ],
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "progress": selected_progress,
        # Root-position/quaternion-free transition features used by the
        # rough-motion discriminator.  No actions are included because the
        # source clip was externally perturbed.
        "style_frames": style_frames,
    }


def make_microduck_roller_backflip_env_cfg(play: bool = False):
    cfg = make_microduck_roller_hop_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # Discover the nominal maneuver first.  Robustness randomization belongs
    # in a later fine-tune only after unassisted nominal success is reliable.
    for event_name in (
        "randomize_wheel_friction", "randomize_com", "randomize_head_com",
        "randomize_armature", "encoder_bias",
        "base_com", "randomize_mass_inertia",
    ):
        cfg.events.pop(event_name, None)
    # BAM uses this event to expand per-world friction/damping fields.  Retain
    # the infrastructure hook but make it deterministic during discovery.
    cfg.events["randomize_joint_friction"].params["scale_range"] = (1.0, 1.0)
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.observations["actor"].enable_corruption = False

    # Clean, privileged motion-prior state.  The deployed actor remains the
    # exact same 61D policy; this group exists only in training storage.
    cfg.observations["style"] = ObservationGroupCfg(
        terms={
            name: deepcopy(cfg.observations["critic"].terms[name])
            for name in ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")
        },
        enable_corruption=False,
        nan_policy="sanitize",
    )

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

    # DeepMimic-style early termination: a trunk or jaw strike makes this
    # maneuver invalid.  Previously a crashed policy could keep collecting
    # rotation and clearance progress for the rest of the episode.
    cfg.terminations["backflip_body_ground_contact"] = TerminationTermCfg(
        func=microduck_mdp.roller_backflip_body_ground_contact,
        params={"sensor_name": "backflip_body_ground_contact"},
    )

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
            "demo_frame_range": (0, DEMO_END_FRAME - DEMO_START_FRAME),
            "assist_vz_range": (0.0, 0.0) if play else (1.50, 1.65),
            "assist_omega_range": (0.0, 0.0),
            "assist_turns_range": None if play else (0.78, 0.88),
        },
    )
    cfg.events["apply_backflip_assistance"] = EventTermCfg(
        func=microduck_mdp.apply_roller_backflip_assistance,
        mode="step",
        params={
            "feet_sensor_name": "feet_ground_contact",
            "stand_height": STAND_HEIGHT,
            "ramp_start_clearance": 0.02,
            "full_clearance": 0.10,
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
    cfg.rewards["backflip_takeoff_pitch"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_takeoff_pitch_progress,
        weight=6.0,
        params={"feet_sensor_name": "feet_ground_contact", "target_pitch_rate": 18.0},
    )
    cfg.rewards["backflip_landing"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing,
        weight=6.0,
        params={**state_params, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["backflip_landing_readiness"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing_readiness_progress,
        weight=8.0,
        params={
            **state_params,
            "minimum_rotation": math.radians(240.0),
            "foot_drop_target": 0.10,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
        },
    )
    cfg.rewards["backflip_post_landing_stability"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_post_landing_stability,
        weight=4.0,
        params={**state_params, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["body_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        # The performance curriculum raises this to a strict terminal cost
        # after early-stage exploration has discovered successful landings.
        weight=-5.0,
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
            func=microduck_mdp.backflip_performance_curriculum,
            params={
                "event_name": "reset_backflip_state",
                "min_attempts": 2048,
                "stages": [
                    # Reverse curriculum: first hold the demonstrated recovery,
                    # then move the reset frontier backward through touchdown,
                    # flight, launch, and finally the unassisted rolling start.
                    # Delayed pitch assistance is combined with the compatible
                    # hop policy's learned launch impulse. A measurement probe
                    # showed that 1.75--1.90 m/s plus one injected turn produced
                    # 20 cm clearance and 423 degrees before a body strike.
                    # Stage 0 targets the demonstrated ~16 cm apex and leaves
                    # the final part of the rotation for the policy to create.
                    {"advance_success": 0.35, "advance_stand_success": 0.05, "required_windows": 2, "action_rate_weight": -0.002, "body_contact_weight": -5.0, "params": {"demo_prob": 0.65, "demo_frame_range": (56, 69), "assist_vz_range": (1.50, 1.65), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.78, 0.88)}},
                    {"advance_success": 0.30, "advance_stand_success": 0.05, "required_windows": 2, "action_rate_weight": -0.002, "body_contact_weight": -7.0, "params": {"demo_prob": 0.60, "demo_frame_range": (48, 69), "assist_vz_range": (1.25, 1.50), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.60, 0.78)}},
                    {"advance_success": 0.25, "advance_stand_success": 0.04, "required_windows": 2, "action_rate_weight": -0.003, "body_contact_weight": -9.0, "params": {"demo_prob": 0.55, "demo_frame_range": (36, 69), "assist_vz_range": (1.00, 1.30), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.45, 0.65)}},
                    {"advance_success": 0.18, "advance_stand_success": 0.03, "required_windows": 2, "action_rate_weight": -0.004, "body_contact_weight": -11.0, "params": {"demo_prob": 0.45, "demo_frame_range": (24, 69), "assist_vz_range": (0.70, 1.05), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.30, 0.50)}},
                    {"advance_success": 0.12, "advance_stand_success": 0.02, "required_windows": 2, "action_rate_weight": -0.006, "body_contact_weight": -14.0, "params": {"demo_prob": 0.35, "demo_frame_range": (12, 69), "assist_vz_range": (0.35, 0.70), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.15, 0.35)}},
                    {"advance_success": 0.08, "advance_stand_success": 0.02, "required_windows": 2, "action_rate_weight": -0.010, "body_contact_weight": -18.0, "params": {"demo_prob": 0.25, "demo_frame_range": (0, 69), "assist_vz_range": (0.10, 0.30), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.0, 0.15)}},
                    {"advance_success": 1.10, "advance_stand_success": 1.10, "required_windows": 2, "action_rate_weight": -0.020, "body_contact_weight": -18.0, "params": {"demo_prob": 0.05, "demo_frame_range": (0, 69), "assist_vz_range": (0.0, 0.0), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.0, 0.0)}},
                ],
            },
        )
    return cfg


MicroduckRollerBackflipRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRollerHopRlCfg)
MicroduckRollerBackflipRlCfg.experiment_name = "roller_backflip"
MicroduckRollerBackflipRlCfg.run_name = "roller_backflip_v1"
MicroduckRollerBackflipRlCfg.max_iterations = 2_500
MicroduckRollerBackflipRlCfg.save_interval = 100
_backflip_demo = load_backflip_demonstration()
_base_algorithm = vars(MicroduckRollerBackflipRlCfg.algorithm).copy()
_base_algorithm.update({
    "class_name": "mjlab_microduck.wasabi.WasabiPPO",
    "expert_transitions": [
        [_backflip_demo["style_frames"][index], _backflip_demo["style_frames"][index + 1]]
        for index in range(len(_backflip_demo["style_frames"]) - 1)
    ],
})
MicroduckRollerBackflipRlCfg.algorithm = WasabiPpoAlgorithmCfg(**_base_algorithm)
MicroduckRollerBackflipRlCfg.algorithm.entropy_coef = 0.005
