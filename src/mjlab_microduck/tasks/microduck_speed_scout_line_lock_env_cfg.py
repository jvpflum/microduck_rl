"""Conservative zero-drag line-lock fine-tune of the 5.41 mph speed scout."""

from __future__ import annotations

import dataclasses
import os

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_race5_constrained_env_cfg import (
    race_usable_speed_squared,
)


def make_microduck_speed_scout_line_lock_env_cfg(play: bool = False):
    """Keep the donor physics fixed while optimizing heading and cross-track error."""
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.decimation = int(os.environ.get("DUCKLAB_LINE_LOCK_DECIMATION", "4"))
    ankle_action_gain = float(
        os.environ.get("DUCKLAB_LINE_LOCK_ANKLE_ACTION_GAIN", "1.0")
    )
    cfg.actions["joint_pos"].scale = {r"^(left_ankle|right_ankle)$": ankle_action_gain}
    reset_pose = cfg.events["reset_base"].params["pose_range"]
    reset_y = float(os.environ.get("DUCKLAB_LINE_LOCK_RESET_Y_M", "0.0"))
    reset_yaw = float(os.environ.get("DUCKLAB_LINE_LOCK_RESET_YAW_RAD", "0.0"))
    reset_pose["y"] = (-reset_y, reset_y)
    reset_pose["yaw"] = (-reset_yaw, reset_yaw)
    cfg.events["line_lock_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            "ranges": (
                float(os.environ.get("DUCKLAB_LINE_LOCK_FRICTION", "0.003")),
                float(os.environ.get("DUCKLAB_LINE_LOCK_FRICTION", "0.003")),
            ),
        },
    )
    cfg.curriculum.clear()
    command = cfg.commands["twist"]
    command.yaw_kp = float(os.environ.get("DUCKLAB_LINE_LOCK_YAW_KP", "0.90"))
    command.lateral_kp = float(os.environ.get("DUCKLAB_LINE_LOCK_LATERAL_KP", "0.20"))
    command.yaw_kd = float(os.environ.get("DUCKLAB_LINE_LOCK_YAW_KD", "0.12"))
    command.max_correction = float(os.environ.get("DUCKLAB_LINE_LOCK_MAX_CORRECTION", "0.30"))
    # Preserve the donor's speed objective. Straightness is the only material
    # change, and deterministic checkpoint gates decide whether an update lives.
    # Speed retention is deliberately dominant.  Friction adaptation is done in
    # short externally gated stages, so straightness cannot be bought by slowing
    # down or falling more often.
    cfg.rewards["forward_velocity_mps"].weight = 3.0
    cfg.rewards["forward_velocity_squared"].weight = 0.50
    cfg.rewards["world_forward_velocity_mps"].weight = 8.0
    cfg.rewards["world_forward_velocity_squared"].weight = 1.50
    cfg.rewards["heading_hold"].weight = float(
        os.environ.get("DUCKLAB_LINE_LOCK_HEADING_HOLD_WEIGHT", "2.5")
    )
    cfg.rewards["lane_error"].weight = float(
        os.environ.get("DUCKLAB_LINE_LOCK_LANE_WEIGHT", "-1.5")
    )
    cfg.rewards["world_lateral_velocity"].weight = float(
        os.environ.get("DUCKLAB_LINE_LOCK_LATERAL_WEIGHT", "-1.5")
    )
    cfg.rewards["heading_error"].weight = float(
        os.environ.get("DUCKLAB_LINE_LOCK_HEADING_ERROR_WEIGHT", "-1.5")
    )
    cfg.rewards["usable_speed"] = RewardTermCfg(
        func=race_usable_speed_squared,
        weight=float(os.environ.get("DUCKLAB_LINE_LOCK_USABLE_SPEED_WEIGHT", "2.0")),
        params={
            "reference_speed_mps": 0.935,
            "lane_std_m": 0.25,
            "heading_std_rad": 0.12,
            "lateral_speed_std_mps": 0.15,
        },
    )
    return cfg


MicroduckSpeedScoutLineLockRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=float(os.environ.get("DUCKLAB_LINE_LOCK_LR", "5e-7")),
        schedule="fixed",
        class_name="mjlab_microduck.algorithms.donor_anchored_ppo:DonorAnchoredPPO",
        desired_kl=float(os.environ.get("DUCKLAB_LINE_LOCK_KL", "5e-5")),
        clip_param=float(os.environ.get("DUCKLAB_LINE_LOCK_CLIP", "0.01")),
        entropy_coef=float(os.environ.get("DUCKLAB_LINE_LOCK_ENTROPY", "0.001")),
        num_learning_epochs=int(os.environ.get("DUCKLAB_LINE_LOCK_EPOCHS", "1")),
        num_mini_batches=int(os.environ.get("DUCKLAB_LINE_LOCK_MINI_BATCHES", "4")),
    ),
    experiment_name="microduck_speed_scout_line_lock",
    run_name="microduck_speed_scout_line_lock",
    save_interval=5,
)
