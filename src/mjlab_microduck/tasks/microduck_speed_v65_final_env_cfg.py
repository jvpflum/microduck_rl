"""V24: conservative refinement of V66's exact embedded high-speed actor.

Unlike V23, this task starts from the actual V65-high branch used by V66.  It
does not inject chassis velocity, uses a fixed optimizer rate, freezes the
deployed observation normalizer and actor body, and trains only the final
action head.  Those constraints turn this into a local policy search rather
than another attempt to rediscover the gait.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.tasks.microduck_speed_final_env_cfg import (
    MicroduckSpeedFinalRlCfg,
    make_microduck_speed_final_env_cfg,
)


EXACT_V65_HIGH_CHECKPOINT = (
    "/home/juice/projects/microduck-lab/upstream/microduck_rl/"
    "protected_checkpoints/duckwing_v24/v65-high-exact-import.pt"
)


@dataclass
class ExactV65HeadPpoCfg(RslRlPpoAlgorithmCfg):
    teacher_checkpoint: str = EXACT_V65_HIGH_CHECKPOINT
    teacher_loss_coef: float = 50.0
    teacher_loss_decay: float = 1.0
    teacher_loss_floor: float = 50.0
    probe_loss_share: float = 0.0
    speed_command_threshold: float = 0.5
    smooth_turn_start: float = 0.08
    smooth_turn_end: float = 0.25
    command_x_index: int = 48
    command_y_index: int = 49
    command_yaw_index: int = 50


def make_microduck_speed_v65_final_env_cfg(play: bool = False):
    cfg = make_microduck_speed_final_env_cfg(play=play)

    # V23's 40% moving-start population made injected kinetic energy dominate
    # the rollout statistic and policy gradient. Every V24 episode launches
    # from rest, so measured and rewarded speed must be produced by the actor.
    cfg.events["reset_final_speed_state"].params.update(
        {
            "bootstrap_speed_range_mps": (0.0, 0.0),
            "bootstrap_fraction_stages": ((0, 0.0),),
        }
    )

    # Keep the objective explicitly lexicographic: legal speed first, but a
    # fall is far more expensive than any possible 20-second speed return.
    cfg.rewards["world_speed"].weight = 2.0
    cfg.rewards["world_speed_squared"].weight = 4.0
    cfg.rewards["speed_gain"].weight = 2.0
    cfg.rewards["final_speed"].weight = 5.0
    cfg.rewards["usable_speed"].weight = 50.0
    cfg.rewards["usable_launch"].weight = 25.0
    cfg.rewards["lateral_speed"].weight = -1.0
    cfg.rewards["tilt_cost"].weight = -1.0
    cfg.rewards["alive"].weight = 0.5
    cfg.rewards["fall"].weight = -2_000.0
    cfg.rewards["self_collisions"].weight = -0.10
    cfg.rewards["action_over_limit"].weight = -0.15
    cfg.rewards["action_rate_l2"].weight = -0.05
    return cfg


_distribution = dict(MicroduckSpeedFinalRlCfg.actor.distribution_cfg)
_distribution["init_std"] = 0.015

MicroduckSpeedV65FinalRlCfg = dataclasses.replace(
    MicroduckSpeedFinalRlCfg,
    actor=dataclasses.replace(
        MicroduckSpeedFinalRlCfg.actor,
        distribution_cfg=_distribution,
    ),
    algorithm=ExactV65HeadPpoCfg(
        class_name="mjlab_microduck.teacher_guided_ppo.TeacherGuidedPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        entropy_coef=0.0,
        learning_rate=2.0e-8,
        desired_kl=1.0e-6,
        clip_param=0.002,
        num_learning_epochs=1,
        num_mini_batches=4,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        max_grad_norm=0.10,
    ),
    experiment_name="microduck_speed_v65_final",
    run_name="microduck_speed_v65_final",
    save_interval=25,
    num_steps_per_env=24,
    max_iterations=1_200,
)
