"""Deployable V70 residual search around the protected high-speed branch.

V69 showed that static action blends can buy a little speed only by giving up
heading, tilt, or contact quality.  V70 moves the search into the simulator:
it starts from the exact V65 high-command actor, keeps a frozen copy of that
actor as a control teacher, and adds a second (V47) speed teacher only for
straight high-command probes.  The extra rewards are deliberately small and
all use observations/sensors that already exist in the 61D deployment family.

The output is an ordinary 61D actor, so unlike the phase-adapter experiment it
does not require an inference-time oscillator or any new runtime inputs.  It is
still a candidate branch; promotion requires the root V67 composition and the
strict Race5/retention gates.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_v65_final_env_cfg import (
    EXACT_V65_HIGH_CHECKPOINT,
    MicroduckSpeedV65FinalRlCfg,
    make_microduck_speed_v65_final_env_cfg,
)
V47_SPEED_TEACHER_ONNX = (
    "/home/juice/projects/microduck-lab/"
    "incoming/rtx5090/v47-official-friction-speed-specialist/policy.onnx"
)


@dataclass
class V70ResidualPpoCfg(RslRlPpoAlgorithmCfg):
    """Conservative teacher routing for a full-actor local search."""

    teacher_checkpoint: str = EXACT_V65_HIGH_CHECKPOINT
    speed_teacher_onnx: str = V47_SPEED_TEACHER_ONNX
    teacher_loss_coef: float = 2.0
    teacher_loss_decay: float = 0.9997
    teacher_loss_floor: float = 0.50
    # Keep turning, braking, and low-command probes anchored to the control
    # teacher while the straight high-command samples see the speed teacher.
    probe_loss_share: float = 0.35
    speed_command_threshold: float = 0.55
    smooth_turn_start: float = 0.08
    smooth_turn_end: float = 0.25
    command_x_index: int = 48
    command_y_index: int = 49
    command_yaw_index: int = 50


def make_microduck_speed_v70_residual_env_cfg(play: bool = False):
    """Add mild contact/stride shaping without changing the 61D interface."""

    cfg = make_microduck_speed_v65_final_env_cfg(play=play)

    # These terms are intentionally an order of magnitude smaller than V65's
    # usable-speed/fall terms.  They provide gradients toward a real stroke and
    # quiet glide, but cannot make a swizzle or a stand-still look successful.
    cfg.rewards["v70_air_time"] = RewardTermCfg(
        func=microduck_mdp.skating_air_time_reward,
        weight=0.45,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.30,
        },
    )
    cfg.rewards["v70_single_support"] = RewardTermCfg(
        func=microduck_mdp.single_support_reward,
        weight=0.55,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_gate_ref": 0.30,
            "double_penalty": 0.10,
        },
    )
    cfg.rewards["v70_glide"] = RewardTermCfg(
        func=microduck_mdp.glide_reward,
        weight=0.60,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_ref": 0.30,
            "normalize_joint_count": True,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*(hip|knee|ankle).*",)
            ),
        },
    )
    cfg.rewards["v70_gait_symmetry"] = RewardTermCfg(
        func=microduck_mdp.gait_symmetry_penalty,
        weight=-0.12,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.rewards["v70_heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=0.30,
        params={"std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
    return cfg


_distribution = dict(MicroduckSpeedV65FinalRlCfg.actor.distribution_cfg)
_distribution["init_std"] = 0.012

MicroduckSpeedV70ResidualRlCfg = dataclasses.replace(
    MicroduckSpeedV65FinalRlCfg,
    actor=dataclasses.replace(
        MicroduckSpeedV65FinalRlCfg.actor,
        distribution_cfg=_distribution,
    ),
    algorithm=V70ResidualPpoCfg(
        class_name="mjlab_microduck.teacher_guided_ppo.TeacherGuidedPPO",
        teacher_checkpoint=EXACT_V65_HIGH_CHECKPOINT,
        speed_teacher_onnx=V47_SPEED_TEACHER_ONNX,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        entropy_coef=0.0,
        learning_rate=1.0e-7,
        desired_kl=2.0e-6,
        clip_param=0.003,
        num_learning_epochs=1,
        num_mini_batches=4,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        max_grad_norm=0.20,
    ),
    experiment_name="microduck_speed_v70_residual",
    run_name="microduck_speed_v70_residual",
    save_interval=25,
    num_steps_per_env=24,
    max_iterations=800,
)
