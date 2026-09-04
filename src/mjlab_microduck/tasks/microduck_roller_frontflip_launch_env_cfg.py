"""Dedicated discovery stage for a skates-only forward-flip launch.

This task deliberately does not ask PPO to solve flight and landing at once.
It learns the physically hard first boundary condition: leave both skates with
useful vertical velocity and positive body-Y (nose-down) angular momentum while
remaining sagittal and avoiding every non-skate terrain contact.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_backflip_env_cfg import (
    STAND_HEIGHT,
    TAKEOFF_CLEARANCE,
    MicroduckRollerBackflipRlCfg,
    make_microduck_roller_backflip_env_cfg,
)


EPISODE_LENGTH_S = 1.2
TARGET_CLEARANCE = 0.040
TARGET_ROTATION = math.radians(180.0)


def make_microduck_roller_frontflip_launch_env_cfg(play: bool = False):
    cfg = make_microduck_roller_backflip_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.commands["twist"].period = EPISODE_LENGTH_S

    robot = cfg.scene.entities["robot"]
    robot.articulation = replace(
        robot.articulation,
        actuators=tuple(
            replace(actuator, max_current=1.75)
            for actuator in robot.articulation.actuators
        ),
    )

    # This is a self-generated launch task.  Reference-state initialization and
    # external velocity assistance belong to later flight/landing training.
    cfg.events.pop("backflip_assistance", None)
    reset = cfg.events["reset_backflip_state"].params
    reset.update(
        {
            "demo_prob": 0.0,
            "demo_frame_range": None,
            "assist_vz_range": (0.0, 0.0),
            "assist_omega_range": (0.0, 0.0),
            "assist_turns_range": (0.0, 0.0),
            "unassisted_stand_prob": 1.0,
        }
    )
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (0.75, 0.85) if play else (0.40, 1.20),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    state = {
        "feet_sensor_name": "feet_ground_contact",
        "body_sensor_name": "backflip_body_ground_contact",
        "stand_height": STAND_HEIGHT,
        "takeoff_clearance": TAKEOFF_CLEARANCE,
        # A landing is outside this short task, but the common state updater
        # still requires a completion threshold.
        "landing_rotation": math.radians(300.0),
    }
    cfg.rewards.clear()
    cfg.rewards["launch_rotation_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_rotation_progress,
        weight=12.0,
        params={**state, "target_rotation": TARGET_ROTATION},
    )
    cfg.rewards["launch_clearance_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_clearance_progress,
        weight=18.0,
        params={**state, "target_clearance": TARGET_CLEARANCE},
    )
    cfg.rewards["launch_vertical_velocity"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_takeoff_velocity,
        weight=8.0,
        params={
            "sensor_name": "feet_ground_contact",
            "max_vz": 1.5,
            "phase_start": 0.0,
            "phase_end": 0.85,
        },
    )
    cfg.rewards["launch_pitch_momentum"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_takeoff_pitch_progress,
        weight=16.0,
        params={"feet_sensor_name": "feet_ground_contact", "target_pitch_rate": 30.0},
    )
    cfg.rewards["non_skate_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        weight=-15.0,
        params={"sensor_name": "backflip_body_ground_contact"},
    )
    cfg.rewards["sagittal_motion"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_sagittal_cost,
        weight=-0.01,
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_phase_action_rate_l2,
        weight=-1.0e-5,
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
    return cfg


MicroduckRollerFrontFlipLaunchRlCfg = deepcopy(MicroduckRollerBackflipRlCfg)
MicroduckRollerFrontFlipLaunchRlCfg.experiment_name = "roller_frontflip_launch"
MicroduckRollerFrontFlipLaunchRlCfg.run_name = "roller_frontflip_launch_v1"
MicroduckRollerFrontFlipLaunchRlCfg.max_iterations = 1_200
MicroduckRollerFrontFlipLaunchRlCfg.save_interval = 25
MicroduckRollerFrontFlipLaunchRlCfg.algorithm.learning_rate = 1.0e-3
MicroduckRollerFrontFlipLaunchRlCfg.algorithm.clip_param = 0.20
MicroduckRollerFrontFlipLaunchRlCfg.algorithm.entropy_coef = 0.02
