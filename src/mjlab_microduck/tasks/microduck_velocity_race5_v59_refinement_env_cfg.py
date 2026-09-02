"""V19: conservative official-friction refinement of V66's V59 speed branch.

V18 showed that a weaker V57b speed teacher can raise training reward while
giving back real race speed.  V19 instead anchors PPO to V59, the exact fast
branch inside the qualified V66 deployment router.  The final export is always
evaluated only after being wrapped by that immutable control shell.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.tasks.microduck_velocity_race5_constrained_env_cfg import (
    MicroduckRace5ConstrainedRlCfg,
    make_microduck_velocity_race5_constrained_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_race5_frontier_env_cfg import (
    OFFICIAL_WHEEL_FRICTIONLOSS,
)


CONTROL_TEACHER_CHECKPOINT = (
    "/home/juice/projects/microduck-lab/upstream/microduck_rl/"
    "protected_checkpoints/duckwing_v18/v17-frontier-i250.pt"
)
SPEED_TEACHER_ONNX = (
    "/home/juice/projects/microduck-lab/incoming/rtx5090/"
    "top5-2026-09-01/v59-i99-promoted-speed-leader/policy.onnx"
)


@dataclass
class Race5V59RefinementPpoCfg(RslRlPpoAlgorithmCfg):
    teacher_checkpoint: str = CONTROL_TEACHER_CHECKPOINT
    speed_teacher_onnx: str = SPEED_TEACHER_ONNX
    # Maintain V59 as the default high-command behavior. PPO only earns enough
    # authority to make small, validated improvements rather than gait collapse.
    teacher_loss_coef: float = 8.0
    teacher_loss_decay: float = 0.99999
    teacher_loss_floor: float = 4.0
    probe_loss_share: float = 0.35
    speed_command_threshold: float = 0.45
    smooth_turn_start: float = 0.08
    smooth_turn_end: float = 0.25
    command_x_index: int = 48
    command_y_index: int = 49
    command_yaw_index: int = 50


def make_microduck_velocity_race5_v59_refinement_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_race5_constrained_env_cfg(play=play)
    cfg.events["randomize_wheel_friction"].params["ranges"] = (
        OFFICIAL_WHEEL_FRICTIONLOSS,
        OFFICIAL_WHEEL_FRICTIONLOSS,
    )
    cfg.curriculum.pop("wheel_friction", None)

    command = cfg.commands["twist"]
    command.yaw_kp = 0.70
    command.lateral_kp = 0.14
    command.yaw_kd = 0.07
    command.max_correction = 0.15

    # Optimize V66's only material gaps—launch and usable centreline speed—
    # while keeping the output inside the teacher's neighborhood.
    cfg.rewards["race_speed_squared"].weight = 5.0
    cfg.rewards["race_forward_progress"].weight = 15.0
    cfg.rewards["race_launch_speed"].weight = 24.0
    cfg.rewards["race_usable_speed"].weight = 25.0
    cfg.rewards["race_usable_launch"].weight = 28.0
    cfg.rewards["race_finish"].weight = 1000.0
    cfg.rewards["race_elapsed"].weight = -2.0
    return cfg


MicroduckRace5V59RefinementRlCfg = dataclasses.replace(
    MicroduckRace5ConstrainedRlCfg,
    algorithm=Race5V59RefinementPpoCfg(
        class_name="mjlab_microduck.teacher_guided_ppo.TeacherGuidedPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        entropy_coef=0.0005,
        learning_rate=2.0e-7,
        desired_kl=1.0e-5,
        clip_param=0.004,
        num_learning_epochs=1,
        num_mini_batches=4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=0.75,
    ),
    experiment_name="velocity_race5_v59_refinement",
    run_name="velocity_race5_v59_refinement",
    save_interval=5,
)
