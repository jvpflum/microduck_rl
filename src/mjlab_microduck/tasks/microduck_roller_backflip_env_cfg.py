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
import os
from copy import deepcopy
from pathlib import Path

from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
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


EPISODE_LENGTH_S = 2.5
STAND_HEIGHT = 0.115
TARGET_CLEARANCE = 0.080
TAKEOFF_CLEARANCE = 0.010
TARGET_ROTATION = 2.0 * math.pi
LANDING_ROTATION = math.radians(300.0)
FRONT_FLIP_PITCH_SIGN = 1.0
# The prior V6 run only exposed a late 0.88 s slice of the recording.  Keep the
# full curated motion (source frame 39 through 238) available as a reverse-
# curriculum reset archive so PPO can learn the missing 90/180/270-degree
# continuations instead of repeatedly seeing only the ~50-degree launch.
DEMO_START_FRAME = 39
DEMO_END_FRAME = 239
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


def load_backflip_demonstration(path: Path | None = None) -> dict:
    """Extract front-flip reset tensors and cumulative forward pitch."""
    if path is None:
        path = Path(os.environ.get("DUCKLAB_FRONTFLIP_REFERENCE", _DEMO_PATH))
    document = json.loads(path.read_text())
    if "reference" in document:
        gate = document.get("gate", document.get("best", {}))
        if gate.get("rotation_deg", 0.0) < math.degrees(LANDING_ROTATION):
            raise ValueError(f"Front-flip reference did not pass 300-degree gate: {path}")
        if gate.get("body_contact", True):
            raise ValueError(f"Front-flip reference contains non-skate ground contact: {path}")
        if document.get("front_flip_pitch_sign") != FRONT_FLIP_PITCH_SIGN:
            raise ValueError(f"Front-flip reference uses the wrong pitch direction: {path}")
        reference = document["reference"]
        required = {
            "root_qpos", "root_qvel", "joint_pos", "joint_vel", "action", "progress"
        }
        if not required.issubset(reference):
            raise ValueError(f"Incomplete optimized front-flip reference: {path}")
        lengths = {len(reference[name]) for name in required}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 8:
            raise ValueError(f"Inconsistent optimized front-flip reference lengths: {path}")
        return reference
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
            pitch_delta = (
                0.0
                if vector_length < 1e-9
                else FRONT_FLIP_PITCH_SIGN * delta[2] / vector_length * angle
            )
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
    entry_speed = float(os.environ.get("DUCKLAB_FRONTFLIP_ENTRY_SPEED", "0.20"))

    body_ground = ContactSensorCfg(
        name="backflip_body_ground_contact",
        # A front flip is valid only when the four skate tires touch terrain.
        # Any other robot body contacting the floor ends the episode.
        primary=ContactMatch(
            mode="body", pattern=r"^(?!tire(?:_[234])?$).+$", entity="robot"
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
    # Discovery uses the exact evaluator mechanics; defer robustness randomization
    # until a clean flip is found.
    for _name in (
        "randomize_wheel_friction",
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_mass_inertia",
    ):
        cfg.events.pop(_name, None)
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup"
    )
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.138, 0.140)
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (max(0.0, entry_speed - 0.05), entry_speed + 0.05),
        "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events["reset_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_roller_backflip_state,
        mode="reset",
        params={
            "demonstration": load_backflip_demonstration(),
            "demo_prob": 0.0 if play else 0.95,
            # Start most training episodes in the demonstrated mid/late
            # rotation so PPO can learn the missing continuation and landing
            # controller before we anneal back toward standing launches.
            # The curated motion is sliced to 200 frames (source 39..238),
            # so source-frame-style values >=200 clamp to the final pose.
            # Start at selected frame 160 (source ~199) to expose an actual
            # late-flight sequence and let the target advance toward landing.
            "demo_frame_range": None,
            "assist_vz_range": (0.0, 0.0) if play else (1.6, 2.0),
            "assist_omega_range": (0.0, 0.0) if play else (8.0, 12.0),
            "assist_turns_range": None if play else (0.8, 1.1),
            "unassisted_stand_prob": 0.0,
        },
    )
    # Assistance must be applied during the rollout, not merely stored by the
    # reset event.  Run the ramp at the control cadence so the launch reaches
    # the requested angular momentum and PPO gets real full-rotation examples.
    if not play:
        cfg.events["backflip_assistance"] = EventTermCfg(
            func=microduck_mdp.apply_roller_backflip_assistance,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "feet_sensor_name": "feet_ground_contact",
                "stand_height": STAND_HEIGHT,
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
        weight=40.0,
        params={**state_params, "target_rotation": TARGET_ROTATION},
    )
    cfg.rewards["backflip_clearance_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_clearance_progress,
        weight=10.0,
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
    cfg.rewards["backflip_takeoff_pitch_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_takeoff_pitch_progress,
        weight=6.0,
        params={"feet_sensor_name": "feet_ground_contact", "target_pitch_rate": 18.0},
    )
    # Pose targets are state references only (not action labels).  Combined
    # with the phase command, this gives the landing segment a usable gradient
    # while the external launch assist is still being annealed.
    cfg.rewards["backflip_demo_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_demo_pose_tracking,
        weight=5.0,
    )
    cfg.rewards["backflip_demo_action_tracking"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_reference_action_tracking,
        weight=2.0,
    )
    cfg.rewards["backflip_landing"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing,
        weight=60.0,
        params={**state_params, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["backflip_landing_readiness"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing_readiness_progress,
        weight=70.0,
        params={**state_params},
    )
    cfg.rewards["backflip_post_landing_stability"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_post_landing_stability,
        weight=30.0,
        params={**state_params, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["body_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        weight=-10.0,
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
        func=microduck_mdp.roller_backflip_phase_action_rate_l2,
        weight=-0.00005,
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

    cfg.terminations["non_skate_ground_contact"] = TerminationTermCfg(
        func=microduck_mdp.roller_backflip_body_ground_contact,
        params={"sensor_name": "backflip_body_ground_contact"},
        time_out=False,
    )

    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["backflip_spawn_assistance"] = CurriculumTermCfg(
            func=microduck_mdp.backflip_performance_curriculum,
            params={
                "event_name": "reset_backflip_state",
                "min_attempts": 4096,
                "min_stand_attempts": 128,
                "stages": [
                    {"advance_success": 0.65, "advance_stand_success": 0.02, "required_windows": 2, "action_rate_weight": -0.00005, "body_contact_weight": -10.0, "params": {"demo_prob": 0.95, "demo_frame_range": None, "assist_vz_range": (1.6, 2.0), "assist_omega_range": (8.0, 12.0), "assist_turns_range": (0.8, 1.1), "unassisted_stand_prob": 0.0}},
                    {"advance_success": 0.60, "advance_stand_success": 0.10, "required_windows": 2, "action_rate_weight": -0.00005, "body_contact_weight": -12.0, "params": {"demo_prob": 0.80, "demo_frame_range": None, "assist_vz_range": (1.3, 1.7), "assist_omega_range": (5.0, 9.0), "assist_turns_range": (0.55, 0.85), "unassisted_stand_prob": 0.10}},
                    {"advance_success": 0.55, "advance_stand_success": 0.25, "required_windows": 2, "action_rate_weight": -0.00008, "body_contact_weight": -15.0, "params": {"demo_prob": 0.65, "demo_frame_range": None, "assist_vz_range": (0.9, 1.3), "assist_omega_range": (2.0, 5.0), "assist_turns_range": (0.25, 0.50), "unassisted_stand_prob": 0.35}},
                    {"advance_success": 0.50, "advance_stand_success": 0.50, "required_windows": 2, "action_rate_weight": -0.00010, "body_contact_weight": -18.0, "params": {"demo_prob": 0.50, "demo_frame_range": None, "assist_vz_range": (0.0, 0.0), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.0, 0.0), "unassisted_stand_prob": 0.75}},
                    {"advance_success": 1.1, "advance_stand_success": 0.80, "required_windows": 2, "action_rate_weight": -0.00015, "body_contact_weight": -20.0, "params": {"demo_prob": 0.25, "demo_frame_range": None, "assist_vz_range": (0.0, 0.0), "assist_omega_range": (0.0, 0.0), "assist_turns_range": (0.0, 0.0), "unassisted_stand_prob": 1.0}},
                ],
            },
        )
    return cfg


MicroduckRollerBackflipRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRollerHopRlCfg)
MicroduckRollerBackflipRlCfg.experiment_name = "roller_backflip"
MicroduckRollerBackflipRlCfg.run_name = "roller_backflip_v1"
MicroduckRollerBackflipRlCfg.max_iterations = 2_500
MicroduckRollerBackflipRlCfg.save_interval = 25
# This recipe learns a new phase-conditioned skill from a dynamically feasible
# reference; it is not a conservative fine-tune of the skating donor.  The old
# 1e-6/0.02 settings made the policy effectively unable to invent a launch.
MicroduckRollerBackflipRlCfg.algorithm.learning_rate = 3.0e-4
MicroduckRollerBackflipRlCfg.algorithm.clip_param = 0.20
MicroduckRollerBackflipRlCfg.algorithm.entropy_coef = 0.01
