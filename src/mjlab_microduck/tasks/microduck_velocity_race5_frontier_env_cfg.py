"""V17: official-friction Race5 fine-tuning with the hybrid as teacher.

The deployed control-aware champion is an ONNX router, so it cannot be resumed
as a PPO checkpoint.  This task instead starts a normal actor from V11 and
anchors it to the exact V11/V47 routing rule while PPO optimizes the official
Race5 objective.  V47 is the transferred 5090 official-friction specialist;
synthetic command probes preserve braking, coast, turning, and recovery
behavior that a straight drag heat does not visit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.tasks.microduck_velocity_race5_constrained_env_cfg import (
    MicroduckRace5ConstrainedRlCfg,
    make_microduck_velocity_race5_constrained_env_cfg,
)


CONTROL_TEACHER_CHECKPOINT = (
    "/home/juice/projects/microduck-lab/upstream/microduck_rl/logs/rsl_rl/"
    "velocity_race5/2026-08-31_03-06-10_ducklab-race5-v11-drag-launch-i10-s42/"
    "model_10.pt"
)
SPEED_TEACHER_ONNX = (
    "/home/juice/projects/microduck-lab/upstream/microduck_rl/"
    "protected_checkpoints/microduck_5090_transfer/"
    "v47-speed-specialist/policy.onnx"
)
OFFICIAL_WHEEL_FRICTIONLOSS = 0.003


@dataclass
class Race5HybridTeacherPpoCfg(RslRlPpoAlgorithmCfg):
    teacher_checkpoint: str = CONTROL_TEACHER_CHECKPOINT
    speed_teacher_onnx: str = SPEED_TEACHER_ONNX
    teacher_loss_coef: float = 0.75
    teacher_loss_decay: float = 0.9995
    teacher_loss_floor: float = 0.15
    probe_loss_share: float = 0.45
    speed_command_threshold: float = 0.5
    smooth_turn_start: float = 0.02
    smooth_turn_end: float = 0.12
    command_x_index: int = 48
    command_y_index: int = 49
    command_yaw_index: int = 50


def make_microduck_velocity_race5_frontier_env_cfg(play: bool = False):
    """Match deployment physics from reset zero and optimize usable speed."""
    cfg = make_microduck_velocity_race5_constrained_env_cfg(play=play)

    # Earlier Race5 runs inherited the generic roller curriculum and were still
    # training at frictionloss 0.0000 after hundreds of PPO iterations.  The
    # scored robot is 0.003, so V17 removes that sim-to-sim mismatch entirely.
    cfg.events["randomize_wheel_friction"].params["ranges"] = (
        OFFICIAL_WHEEL_FRICTIONLOSS,
        OFFICIAL_WHEEL_FRICTIONLOSS,
    )
    cfg.curriculum.pop("wheel_friction", None)

    # Use the same bounded controller that produced the champion's 9/9 replay.
    command = cfg.commands["twist"]
    command.yaw_kp = 0.55
    command.lateral_kp = 0.25
    command.yaw_kd = 0.05
    command.max_correction = 0.10

    # Push the Pareto frontier only inside the usable racing envelope.  The
    # hybrid-teacher loss is the behavior constraint; these rewards provide the
    # pressure to improve beyond the teacher rather than merely clone it.
    cfg.rewards["race_speed_squared"].weight = 3.0
    cfg.rewards["race_forward_progress"].weight = 10.0
    cfg.rewards["race_launch_speed"].weight = 4.0
    cfg.rewards["race_usable_speed"].weight = 18.0
    cfg.rewards["race_usable_launch"].weight = 12.0
    cfg.rewards["race_finish"].weight = 800.0
    cfg.rewards["race_elapsed"].weight = -1.5
    return cfg


MicroduckRace5FrontierRlCfg = dataclasses.replace(
    MicroduckRace5ConstrainedRlCfg,
    algorithm=Race5HybridTeacherPpoCfg(
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
    experiment_name="velocity_race5",
    run_name="velocity_race5_frontier",
    save_interval=5,
)
