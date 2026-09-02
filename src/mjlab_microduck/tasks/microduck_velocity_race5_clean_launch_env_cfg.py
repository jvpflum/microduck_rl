"""V20: teach V66's speed branch to enter its stride without crab-walking."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.tasks.microduck_velocity_race5_env_cfg import (
    race_launch_heading_error_squared,
    race_launch_lateral_speed_squared,
)
from mjlab_microduck.tasks.microduck_velocity_race5_frontier_env_cfg import (
    OFFICIAL_WHEEL_FRICTIONLOSS,
)
from mjlab_microduck.tasks.microduck_velocity_race5_v59_refinement_env_cfg import (
    CONTROL_TEACHER_CHECKPOINT,
    SPEED_TEACHER_ONNX,
)
from mjlab_microduck.tasks.microduck_velocity_race5_constrained_env_cfg import (
    MicroduckRace5ConstrainedRlCfg,
    make_microduck_velocity_race5_constrained_env_cfg,
)


@dataclass
class Race5CleanLaunchPpoCfg(RslRlPpoAlgorithmCfg):
    teacher_checkpoint: str = CONTROL_TEACHER_CHECKPOINT
    speed_teacher_onnx: str = SPEED_TEACHER_ONNX
    # The V59 gait is valuable after launch, but its initial crab step is the
    # defect being removed.  Retain it softly rather than locking PPO to it.
    teacher_loss_coef: float = 0.75
    teacher_loss_decay: float = 0.99995
    teacher_loss_floor: float = 0.20
    probe_loss_share: float = 0.35
    speed_command_threshold: float = 0.45
    smooth_turn_start: float = 0.08
    smooth_turn_end: float = 0.25
    command_x_index: int = 48
    command_y_index: int = 49
    command_yaw_index: int = 50


def make_microduck_velocity_race5_clean_launch_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_race5_constrained_env_cfg(play=play)
    cfg.events["randomize_wheel_friction"].params["ranges"] = (
        OFFICIAL_WHEEL_FRICTIONLOSS,
        OFFICIAL_WHEEL_FRICTIONLOSS,
    )
    cfg.curriculum.pop("wheel_friction", None)

    # This is a straight-from-rest skill.  Steering stays owned by V66's
    # immutable control branch once this actor is composed for deployment.
    cfg.commands["twist"].ranges.lin_vel_x = (0.80, 0.80)
    cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    command = cfg.commands["twist"]
    command.yaw_kp = 0.90
    command.lateral_kp = 0.22
    command.yaw_kd = 0.10
    command.max_correction = 0.12

    cfg.rewards["race_speed_squared"].weight = 4.0
    cfg.rewards["race_forward_progress"].weight = 14.0
    cfg.rewards["race_launch_speed"].weight = 42.0
    cfg.rewards["race_usable_speed"].weight = 18.0
    cfg.rewards["race_usable_launch"].weight = 50.0
    cfg.rewards["race_finish"].weight = 900.0
    cfg.rewards["race_elapsed"].weight = -2.0
    cfg.rewards["race_launch_lateral_speed"] = RewardTermCfg(
        func=race_launch_lateral_speed_squared,
        weight=-70.0,
        params={"reference_speed": 0.18, "window_s": 1.25},
    )
    cfg.rewards["race_launch_heading"] = RewardTermCfg(
        func=race_launch_heading_error_squared,
        weight=-45.0,
        params={"reference_yaw_rad": 0.10, "window_s": 1.25},
    )
    return cfg


MicroduckRace5CleanLaunchRlCfg = dataclasses.replace(
    MicroduckRace5ConstrainedRlCfg,
    algorithm=Race5CleanLaunchPpoCfg(
        class_name="mjlab_microduck.teacher_guided_ppo.TeacherGuidedPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        entropy_coef=0.0005,
        learning_rate=1.0e-6,
        desired_kl=5.0e-5,
        clip_param=0.008,
        num_learning_epochs=1,
        num_mini_batches=4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=0.75,
    ),
    experiment_name="velocity_race5_clean_launch",
    run_name="velocity_race5_clean_launch",
    save_interval=5,
)
