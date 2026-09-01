"""Teacher-guided official-friction transfer from the protected speed scout."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlPpoAlgorithmCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import MicroduckSpeedStraighteningRlCfg, make_microduck_speed_straightening_env_cfg

TEACHER_CHECKPOINT = "/home/juice/projects/microduck-lab/upstream/microduck_rl/logs/rsl_rl/microduck_speed_scout_transfer/2026-08-31_19-23-06_ducklab-speed-scout-transfer-e4096-i4000-s541/model_160.pt"

STAGES = (
    {"target_speed_mps": 2.25, "wheel_friction": 0.0005, "advance_mean_speed_mps": 1.25, "advance_survival_fraction": 0.97},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0015, "advance_mean_speed_mps": 1.05, "advance_survival_fraction": 0.95},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0025, "advance_mean_speed_mps": 0.90, "advance_survival_fraction": 0.93},
    {"target_speed_mps": 2.25, "wheel_friction": 0.0030, "advance_mean_speed_mps": 0.85, "advance_survival_fraction": 0.92},
)


@dataclass
class TeacherGuidedPpoCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with a frozen donor-action regularizer."""

    teacher_checkpoint: str = TEACHER_CHECKPOINT
    teacher_loss_coef: float = 0.20
    teacher_loss_decay: float = 0.999
    teacher_loss_floor: float = 0.01


def make_microduck_speed_teacher_guided_env_cfg(play: bool = False):
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    cfg.episode_length_s = 20.0
    cfg.events["teacher_guided_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)), "operation": "abs", "ranges": (0.0005, 0.0005)},
    )
    cfg.rewards["lane_error"].weight = -0.55
    cfg.rewards["world_lateral_velocity"].weight = -0.55
    cfg.rewards["heading_error"].weight = -0.55
    cfg.curriculum.clear()
    cfg.curriculum["teacher_guided_friction"] = CurriculumTermCfg(
        func=microduck_mdp.speed_discovery_performance_curriculum,
        params={"command_name": "twist", "target_reward_name": "speed_target_progress", "stages": [dict(stage) for stage in STAGES], "min_attempts": 2048, "required_windows": 2, "effort_command": 0.80, "friction_event_name": "teacher_guided_wheel_friction"},
    )
    return cfg


MicroduckSpeedTeacherGuidedRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=TeacherGuidedPpoCfg(
        class_name="mjlab_microduck.teacher_guided_ppo.TeacherGuidedPPO",
        value_loss_coef=1.0, use_clipped_value_loss=True, entropy_coef=0.001,
        learning_rate=1.0e-6, desired_kl=1.0e-4, clip_param=0.015, num_learning_epochs=2,
        num_mini_batches=4, schedule="adaptive", gamma=0.99, lam=0.95, max_grad_norm=1.0,
    ),
    experiment_name="microduck_speed_teacher_guided",
    run_name="microduck_speed_teacher_guided",
    save_interval=100,
)
