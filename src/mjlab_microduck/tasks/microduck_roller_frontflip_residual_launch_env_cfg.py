"""Residual-PPO refinement around the protected native front-flip trajectory."""

from __future__ import annotations

from copy import deepcopy
import os

from mjlab.managers import EventTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_centroidal_launch_env_cfg import (
    MicroduckRollerFrontFlipCentroidalLaunchRlCfg,
    _load_native_prior,
    make_microduck_roller_frontflip_centroidal_launch_env_cfg,
)


def make_microduck_roller_frontflip_residual_launch_env_cfg(play: bool = False):
    cfg = make_microduck_roller_frontflip_centroidal_launch_env_cfg(play=play)
    prior = _load_native_prior()
    if prior is None:
        # Keep task discovery/import safe.  The launch script always supplies
        # the validated protected primitive before constructing the real task.
        return cfg

    cfg.actions["joint_pos"].frontflip_residual_prior = {
        "primitive_path": os.environ["DUCKLAB_FRONTFLIP_CENTROIDAL_PRIOR"],
        "residual_scale": float(os.environ.get("DUCKLAB_FRONTFLIP_RESIDUAL_SCALE", "0.12")),
    }
    # Base translation without matching passive-wheel angular velocity creates
    # a large artificial skid at t=0.  The native optimizer always uses the
    # physically consistent v/r wheel state, so reproduce it exactly here.
    cfg.events["rolling_entry"] = EventTermCfg(
        func=microduck_mdp.reset_rolling_entry,
        mode="reset",
        params={"speed_range": (1.20, 1.20), "wheel_radius": 0.0175},
    )
    # The nominal motion is now part of the controller.  Penalizing deviation
    # from it again would collapse exploration toward a known crashing path.
    cfg.rewards.pop("native_action_prior", None)
    cfg.rewards["launch_rotation_progress"].weight = 30.0
    cfg.rewards["launch_clearance_progress"].weight = 28.0
    cfg.rewards["launch_vertical_velocity"].weight = 20.0
    cfg.rewards["centroidal_pitch_momentum"].weight = 64.0
    cfg.rewards["non_skate_ground_contact"].weight = -200.0
    cfg.rewards["sagittal_motion"].weight = -0.08
    cfg.rewards["residual_action_l2"] = RewardTermCfg(
        func=microduck_mdp.roller_frontflip_residual_action_l2,
        weight=-0.01,
    )
    return cfg


MicroduckRollerFrontFlipResidualLaunchRlCfg = deepcopy(
    MicroduckRollerFrontFlipCentroidalLaunchRlCfg
)
MicroduckRollerFrontFlipResidualLaunchRlCfg.experiment_name = (
    "roller_frontflip_residual_launch"
)
MicroduckRollerFrontFlipResidualLaunchRlCfg.run_name = (
    "roller_frontflip_residual_launch_v1"
)
MicroduckRollerFrontFlipResidualLaunchRlCfg.max_iterations = 300
MicroduckRollerFrontFlipResidualLaunchRlCfg.save_interval = 10
MicroduckRollerFrontFlipResidualLaunchRlCfg.algorithm.learning_rate = 5.0e-5
MicroduckRollerFrontFlipResidualLaunchRlCfg.algorithm.clip_param = 0.08
MicroduckRollerFrontFlipResidualLaunchRlCfg.algorithm.entropy_coef = 0.003
