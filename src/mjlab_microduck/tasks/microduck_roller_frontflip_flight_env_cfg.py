"""Flight/line-lock stage for the roller front flip.

V47 discovers a self-generated launch.  This stage resumes that policy with
conservative PPO and teaches it to convert the launch into forward sagittal
rotation, prepare both skates below the trunk, and approach a clean landing.
"""

import math
from copy import deepcopy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_backflip_env_cfg import (
    STAND_HEIGHT,
    TAKEOFF_CLEARANCE,
)
from mjlab_microduck.tasks.microduck_roller_frontflip_launch_env_cfg import (
    MicroduckRollerFrontFlipLaunchRlCfg,
    make_microduck_roller_frontflip_launch_env_cfg,
)


EPISODE_LENGTH_S = 2.0
TARGET_CLEARANCE = 0.050
TARGET_ROTATION = 2.0 * math.pi
LANDING_ROTATION = math.radians(300.0)
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def make_microduck_roller_frontflip_flight_env_cfg(play: bool = False):
    cfg = make_microduck_roller_frontflip_launch_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.commands["twist"].period = EPISODE_LENGTH_S

    state = {
        "feet_sensor_name": "feet_ground_contact",
        "body_sensor_name": "backflip_body_ground_contact",
        "stand_height": STAND_HEIGHT,
        "takeoff_clearance": TAKEOFF_CLEARANCE,
        "landing_rotation": LANDING_ROTATION,
    }
    cfg.rewards.clear()
    cfg.rewards["flight_rotation_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_rotation_progress,
        weight=35.0,
        params={**state, "target_rotation": TARGET_ROTATION},
    )
    cfg.rewards["flight_clearance_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_clearance_progress,
        weight=8.0,
        params={**state, "target_clearance": TARGET_CLEARANCE},
    )
    cfg.rewards["launch_pitch_retention"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_takeoff_pitch_progress,
        weight=4.0,
        params={"feet_sensor_name": "feet_ground_contact", "target_pitch_rate": 30.0},
    )
    cfg.rewards["landing_readiness"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing_readiness_progress,
        weight=45.0,
        params={
            **state,
            "minimum_rotation": math.radians(150.0),
            "foot_drop_target": 0.10,
        },
    )
    cfg.rewards["landing"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_landing,
        weight=80.0,
        params={**state, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["post_landing_stability"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_post_landing_stability,
        weight=40.0,
        params={**state, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["non_skate_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_body_contact_cost,
        weight=-20.0,
        params={"sensor_name": "backflip_body_ground_contact"},
    )
    cfg.rewards["sagittal_motion"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_sagittal_cost,
        weight=-0.20,
    )
    cfg.rewards["pitch_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_overspeed_cost,
        weight=-0.002,
        params={"max_pitch_rate": 30.0},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.roller_backflip_phase_action_rate_l2,
        weight=-2.0e-5,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2,
        weight=-1.0e-5,
    )
    cfg.curriculum.clear()
    return cfg


MicroduckRollerFrontFlipFlightRlCfg = deepcopy(MicroduckRollerFrontFlipLaunchRlCfg)
MicroduckRollerFrontFlipFlightRlCfg.experiment_name = "roller_frontflip_flight"
MicroduckRollerFrontFlipFlightRlCfg.run_name = "roller_frontflip_flight_v1"
MicroduckRollerFrontFlipFlightRlCfg.max_iterations = 800
MicroduckRollerFrontFlipFlightRlCfg.save_interval = 25
MicroduckRollerFrontFlipFlightRlCfg.algorithm.learning_rate = 1.0e-4
MicroduckRollerFrontFlipFlightRlCfg.algorithm.clip_param = 0.10
MicroduckRollerFrontFlipFlightRlCfg.algorithm.entropy_coef = 0.002
