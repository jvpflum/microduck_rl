"""GPU-first, real-start front-flip launch training around a native trajectory.

This task fixes the principal V78--V85 objective mismatch: launch authority is
measured with whole-robot centroidal angular momentum, not root pitch rate.
Every episode starts from a physically consistent rolling state and uses exact
official wheel drag and the firmware-level 1.75 A current limit.  A validated
native trajectory is only a decaying action prior; PPO must improve it through
the real actuator/contact stack.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_launch_env_cfg import (
    MicroduckRollerFrontFlipLaunchRlCfg,
    make_microduck_roller_frontflip_launch_env_cfg,
)


OFFICIAL_WHEEL_FRICTION = 0.003
CURRENT_LIMIT_A = 1.75


def _load_native_prior() -> tuple[list[float], list[list[float]]] | None:
    raw_path = os.environ.get("DUCKLAB_FRONTFLIP_CENTROIDAL_PRIOR")
    if not raw_path:
        return None
    path = Path(raw_path)
    document = json.loads(path.read_text())
    if not math.isclose(float(document["wheel_frictionloss"]), OFFICIAL_WHEEL_FRICTION, abs_tol=1e-12):
        raise ValueError(f"native prior must use frictionloss 0.003: {path}")
    if not math.isclose(float(document["current_limit_a"]), CURRENT_LIMIT_A, abs_tol=1e-12):
        raise ValueError(f"native prior must use the 1.75 A current limit: {path}")
    clean = float(document.get("minimum_clean_rotation_deg", 0.0))
    if clean < 220.0 or float(document.get("body_contact_rate", 1.0)) < 1.0:
        raise ValueError(f"native prior is not the expected >=220-degree contact frontier: {path}")
    nodes = document.get("full_nodes")
    times = document.get("knot_times_s")
    if not nodes or not times or len(nodes) != len(times) or any(len(row) != 14 for row in nodes):
        raise ValueError(f"native prior has invalid trajectory nodes: {path}")
    return list(times), nodes


def make_microduck_roller_frontflip_centroidal_launch_env_cfg(play: bool = False):
    cfg = make_microduck_roller_frontflip_launch_env_cfg(play=play)
    cfg.episode_length_s = 1.25
    cfg.commands["twist"].period = cfg.episode_length_s

    robot = cfg.scene.entities["robot"]
    robot.articulation = replace(
        robot.articulation,
        actuators=tuple(
            replace(actuator, max_current=CURRENT_LIMIT_A)
            for actuator in robot.articulation.actuators
        ),
    )
    cfg.events["official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (OFFICIAL_WHEEL_FRICTION, OFFICIAL_WHEEL_FRICTION),
        },
    )
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (1.20, 1.20),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    prior = _load_native_prior()
    cfg.rewards["launch_pitch_momentum"].weight = 0.0
    cfg.rewards["launch_rotation_progress"].weight = 18.0
    cfg.rewards["launch_clearance_progress"].weight = 24.0
    cfg.rewards["launch_vertical_velocity"].weight = 14.0
    cfg.rewards["centroidal_pitch_momentum"] = RewardTermCfg(
        func=microduck_mdp.roller_frontflip_supported_pitch_angular_momentum_progress,
        weight=48.0,
        params={
            "feet_sensor_name": "feet_ground_contact",
            "target_momentum": 0.075,
            "sensor_index": 5,
        },
    )
    if prior is not None:
        times, nodes = prior
        cfg.rewards["native_action_prior"] = RewardTermCfg(
            func=microduck_mdp.roller_frontflip_ballistic_action_prior,
            weight=16.0,
            params={
                "knot_times_s": times,
                "full_nodes": nodes,
                "action_std": 0.45,
                "end_time_s": 1.04,
                "decay_steps": 14_400,
                "final_scale": 0.20,
            },
        )
    cfg.rewards["non_skate_ground_contact"].weight = -120.0
    cfg.rewards["sagittal_motion"].weight = -0.03
    cfg.rewards["action_rate_l2"].weight = -2.0e-5
    cfg.rewards["joint_torques_l2"].weight = -2.0e-5
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.25,
        params={"sensor_name": "self_collision"},
    )
    cfg.curriculum.clear()
    return cfg


MicroduckRollerFrontFlipCentroidalLaunchRlCfg = deepcopy(
    MicroduckRollerFrontFlipLaunchRlCfg
)
MicroduckRollerFrontFlipCentroidalLaunchRlCfg.experiment_name = (
    "roller_frontflip_centroidal_launch"
)
MicroduckRollerFrontFlipCentroidalLaunchRlCfg.run_name = (
    "roller_frontflip_centroidal_launch_v1"
)
MicroduckRollerFrontFlipCentroidalLaunchRlCfg.max_iterations = 600
MicroduckRollerFrontFlipCentroidalLaunchRlCfg.save_interval = 25
MicroduckRollerFrontFlipCentroidalLaunchRlCfg.algorithm.learning_rate = 2.0e-4
MicroduckRollerFrontFlipCentroidalLaunchRlCfg.algorithm.clip_param = 0.12
MicroduckRollerFrontFlipCentroidalLaunchRlCfg.algorithm.entropy_coef = 0.006
