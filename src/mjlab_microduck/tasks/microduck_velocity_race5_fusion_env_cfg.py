"""V18: train a replaceable speed branch inside an immutable control shell.

V17 tried to make one actor preserve every V11 skill while also discovering a
faster official-friction gait. Its early checkpoints retained control, but PPO
eventually traded away low-command behavior without producing a meaningful
speed gain. V18 removes that conflict structurally: PPO optimizes the straight,
high-effort branch, while deployment wraps each export with the already
qualified control-aware champion for idle, cruise, braking, and turns.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.tasks.microduck_velocity_race5_frontier_env_cfg import (
    OFFICIAL_WHEEL_FRICTIONLOSS,
)
from mjlab_microduck.tasks.microduck_velocity_race5_constrained_env_cfg import (
    MicroduckRace5ConstrainedRlCfg,
    make_microduck_velocity_race5_constrained_env_cfg,
)


CONTROL_TEACHER_CHECKPOINT = (
    "/home/juice/projects/microduck-lab/upstream/microduck_rl/"
    "protected_checkpoints/duckwing_v18/v17-frontier-i250.pt"
)
SPEED_TEACHER_ONNX = (
    "/home/juice/projects/microduck-lab/upstream/microduck_rl/"
    "protected_checkpoints/duckwing_v18/rtx5090-v57b-i50.onnx"
)


@dataclass
class Race5FusionTeacherPpoCfg(RslRlPpoAlgorithmCfg):
    teacher_checkpoint: str = CONTROL_TEACHER_CHECKPOINT
    speed_teacher_onnx: str = SPEED_TEACHER_ONNX
    teacher_loss_coef: float = 0.85
    teacher_loss_decay: float = 0.9998
    teacher_loss_floor: float = 0.35
    # Control is protected again by the outer deployment router, so most of
    # the gradient budget can target the V57b straight-line gait. Probes still
    # teach bounded response as line hold introduces small yaw commands.
    probe_loss_share: float = 0.20
    speed_command_threshold: float = 0.5
    smooth_turn_start: float = 0.08
    smooth_turn_end: float = 0.25
    command_x_index: int = 48
    command_y_index: int = 49
    command_yaw_index: int = 50


def make_microduck_velocity_race5_fusion_env_cfg(play: bool = False):
    """Optimize launch and speed at the exact deployment operating point."""
    cfg = make_microduck_velocity_race5_constrained_env_cfg(play=play)
    cfg.events["randomize_wheel_friction"].params["ranges"] = (
        OFFICIAL_WHEEL_FRICTIONLOSS,
        OFFICIAL_WHEEL_FRICTIONLOSS,
    )
    cfg.curriculum.pop("wheel_friction", None)

    # Match the controller used by the best qualified V57b fusion replay.
    command = cfg.commands["twist"]
    command.yaw_kp = 0.55
    command.lateral_kp = 0.10
    command.yaw_kd = 0.0422
    command.max_correction = 0.18

    # V61's only Pollen-relative loss is first-second acceleration. Keep
    # centreline-gated speed dominant, but make legal launch velocity a first-
    # class objective rather than hoping terminal time supplies its gradient.
    cfg.rewards["race_speed_squared"].weight = 4.0
    cfg.rewards["race_forward_progress"].weight = 12.0
    cfg.rewards["race_launch_speed"].weight = 12.0
    cfg.rewards["race_usable_speed"].weight = 20.0
    cfg.rewards["race_usable_launch"].weight = 20.0
    cfg.rewards["race_finish"].weight = 800.0
    cfg.rewards["race_elapsed"].weight = -1.5
    return cfg


MicroduckRace5FusionRlCfg = dataclasses.replace(
    MicroduckRace5ConstrainedRlCfg,
    algorithm=Race5FusionTeacherPpoCfg(
        class_name="mjlab_microduck.teacher_guided_ppo.TeacherGuidedPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        entropy_coef=0.001,
        learning_rate=1.0e-6,
        desired_kl=7.5e-5,
        clip_param=0.01,
        num_learning_epochs=1,
        num_mini_batches=4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
    ),
    experiment_name="velocity_race5_fusion",
    run_name="velocity_race5_fusion",
    save_interval=5,
)
