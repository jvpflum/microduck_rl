"""Race5-v14: efficient lean-and-glide development from the v11 champion."""

import dataclasses
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_race_env_cfg import (
    MicroduckRaceRlCfg,
    make_microduck_velocity_race_env_cfg,
    race_forward_progress_rate,
    race_heading_departure,
    race_heading_error_squared,
    race_invalid_heat,
    race_out_of_lane,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


MPH_TO_MPS = 0.44704
RACE5_TARGET_MPS = 5.0 * MPH_TO_MPS
RACE5_STRETCH_MPS = 10.0 * MPH_TO_MPS
RACE5_TRAIN_DISTANCE_M = 100.0 * 0.3048


def race_world_lateral_speed_squared(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize velocity across the lane in the fixed world-frame race course.

    The inherited Sprint term measures lateral velocity in the robot frame.  A
    duck that has already yawed can therefore regard movement out of the lane as
    straight ahead.  World-Y velocity is the actual cross-track error rate for
    an A-to-B heat and supplies an earlier signal than position alone.
    """
    lateral_speed = env.scene["robot"].data.root_link_lin_vel_w[:, 1]
    return torch.nan_to_num(lateral_speed.square(), nan=0.0)


def race_launch_speed_progress(
    env: "ManagerBasedRlEnv",
    reference_speed: float = 0.55,
    window_s: float = 2.0,
    safety_cap: float = RACE5_STRETCH_MPS,
) -> torch.Tensor:
    """Reward useful world-forward speed most strongly during the launch.

    Integrating this time-weighted term over the first two seconds pays a duck
    that reaches its skating gait sooner.  It avoids differentiating noisy
    contact velocities while still directly targeting launch acceleration.
    Backward motion remains negative so a rock-back cannot farm the reward.
    """
    speed = torch.clamp(
        env.scene["robot"].data.root_link_lin_vel_w[:, 0],
        min=-safety_cap,
        max=safety_cap,
    )
    age_s = env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)
    launch_weight = torch.clamp(1.0 - age_s / max(float(window_s), 0.05), min=0.0)
    return torch.nan_to_num((speed / reference_speed) * launch_weight, nan=0.0)


def race_launch_lateral_speed_squared(
    env: "ManagerBasedRlEnv",
    reference_speed: float = 0.18,
    window_s: float = 1.25,
) -> torch.Tensor:
    """Charge cross-track motion heavily during the first skating stroke.

    The ordinary race-line terms eventually correct a sideways gait, but a
    high-speed teacher can still spend its first second crab-walking before
    those costs dominate.  This is deliberately world-frame (course-relative)
    and fades after the launch window, so it does not suppress legitimate
    steering in the deployment control shell.
    """
    lateral_speed = env.scene["robot"].data.root_link_lin_vel_w[:, 1]
    age_s = env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)
    early = torch.clamp(1.0 - age_s / max(float(window_s), 0.05), min=0.0)
    cost = (lateral_speed / max(float(reference_speed), 0.01)).square() * early
    return torch.nan_to_num(cost, nan=0.0, posinf=100.0)


def race_launch_heading_error_squared(
    env: "ManagerBasedRlEnv",
    reference_yaw_rad: float = 0.10,
    window_s: float = 1.25,
) -> torch.Tensor:
    """Keep the chassis pointed down-course until the first stride is clean."""
    quat = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    age_s = env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)
    early = torch.clamp(1.0 - age_s / max(float(window_s), 0.05), min=0.0)
    cost = (yaw / max(float(reference_yaw_rad), 0.01)).square() * early
    return torch.nan_to_num(cost, nan=0.0, posinf=100.0)


def make_microduck_velocity_race5_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Use a familiar race-effort token while optimizing measured speed.

    The command is deliberately fixed at 0.80 m/s; it is an effort selector,
    not the speed target. Asking a command-conditioned actor for 2.2352 m/s was
    out of distribution and made otherwise capable skating policies crouch in
    place. The five-mph goal is enforced by measured world speed and scoring.
    """
    cfg = make_microduck_velocity_race_env_cfg(play=play)
    # V13's 100 m episodes improved lane survival but gradually selected a
    # conservative gait. Return the terminal objective to the verified 100-foot
    # race while keeping V13's larger population and rollout horizon.
    cfg.episode_length_s = 60.0
    cfg.commands["twist"].ranges.lin_vel_x = (0.80, 0.80)
    cfg.commands["twist"].ranges.ang_vel_z = (-0.30, 0.30)
    cfg.commands["twist"] = microduck_mdp.RaceLineVelocityCommandCfg(
        **vars(cfg.commands["twist"]),
        yaw_kp=0.55,
        lateral_kp=0.10,
        yaw_kd=0.08,
        max_correction=0.18,
    )
    cfg.rewards["race_finish"].params["distance_m"] = RACE5_TRAIN_DISTANCE_M
    cfg.terminations["race_finished"].params["distance_m"] = RACE5_TRAIN_DISTANCE_M

    # The inherited roller reset randomizes x and y by +/-0.5 m.  With a 0.4 m
    # legal lane that meant some heats terminated on their first step, and x
    # randomization also silently shortened/lengthened the race.  Every drag
    # heat now starts on the same measured line and centreline.
    reset_pose = cfg.events["reset_base"].params["pose_range"]
    reset_pose["x"] = (0.0, 0.0)
    reset_pose["y"] = (0.0, 0.0)
    reset_pose["yaw"] = (0.0, 0.0)

    # Normalize to the already achievable gait, so early stages retain a dense
    # learning signal. The quadratic return still increasingly favors speed.
    cfg.rewards["race_speed_squared"].weight = 4.5
    cfg.rewards["race_speed_squared"].params.update(
        reference_speed=0.55,
        safety_cap=RACE5_STRETCH_MPS,
    )
    cfg.rewards["race_forward_progress"] = RewardTermCfg(
        func=race_forward_progress_rate,
        weight=14.0,
        params={"reference_speed": 0.55, "safety_cap": RACE5_STRETCH_MPS},
    )
    # Keep one physical objective: measured world-forward speed squared, capped
    # only at the 10 mph simulator-safety ceiling. V11 improved in its first ten
    # iterations, then regressed as its speed-weight curriculum advanced. V12
    # deliberately has no moving reward weights; checkpoint selection supplies
    # the improvement pressure outside PPO. V13 applies one small, fixed speed
    # increase rather than changing the objective underneath the optimizer.
    cfg.rewards.pop("race_speed_target", None)
    cfg.curriculum.pop("race_measured_speed_target", None)
    cfg.rewards["race_launch_speed"] = RewardTermCfg(
        func=race_launch_speed_progress,
        weight=8.0,
        params={
            "reference_speed": 0.55,
            "window_s": 2.0,
            "safety_cap": RACE5_STRETCH_MPS,
        },
    )
    # At race effort, a modest 5.7-degree pitch moves the centre of mass into
    # the push without approaching the old roller policy's unstable 15-degree
    # target. This is a mechanics/traction term, not an aerodynamic claim.
    cfg.rewards["forward_lean"].weight = 0.6
    cfg.rewards["forward_lean"].params.update(
        {"target_pitch": 0.10, "std": 0.07}
    )
    # The inherited implementation sums squared velocity over ten leg joints,
    # which made the exponential underflow to a logged 0.0000 throughout V13.
    # Mean-square normalization makes a quiet single-support coast measurable;
    # keep its weight modest so speed and the evaluator still choose the gait.
    cfg.rewards["glide"].weight = 1.5
    cfg.rewards["glide"].params.update(
        {"stillness_std": 5.0, "normalize_joint_count": True}
    )
    cfg.curriculum.pop("race_speed_weight", None)
    # Terminal outcomes must be large enough to matter next to 12 seconds of
    # dense progress return. A legal finish and an invalid exit now have equal
    # and opposite one-step magnitude after dt scaling (+10 / -10).
    cfg.rewards["race_finish"].weight = 500.0
    cfg.rewards["race_elapsed"].weight = -1.0
    cfg.rewards["race_lane_error"].weight = -22.0
    cfg.rewards["race_lane_error"].params["lane_half_width_m"] = 0.35
    cfg.rewards["race_world_lateral_speed"] = RewardTermCfg(
        func=race_world_lateral_speed_squared,
        weight=-13.0,
    )
    cfg.rewards["race_heading_error"] = RewardTermCfg(
        func=race_heading_error_squared,
        weight=-22.0,
        params={"maximum_yaw_rad": 0.349066},
    )
    # Without an explicit terminal charge, quitting the lane can be cheaper
    # than enduring the remaining elapsed/heading costs. Make invalid exits an
    # unambiguously bad outcome on the exact step that ends the heat.
    cfg.rewards["race_invalid_heat"] = RewardTermCfg(
        func=race_invalid_heat,
        weight=-500.0,
        params={"lane_half_width_m": 0.75, "maximum_yaw_rad": 0.610865},
    )
    cfg.rewards["heading_hold"].weight = 16.0
    cfg.rewards["heading_hold"].params["std"] = 0.15
    cfg.rewards["lateral_speed"].weight = -5.0
    cfg.rewards["gait_symmetry"].weight = -3.0
    cfg.rewards["upright"].weight = 8.0
    # Preserve the champion's cadence regularization. The larger population,
    # not a looser action penalty, supplies V13's exploration.
    cfg.rewards["action_rate_l2"].weight = -1.2
    # The inherited roller curriculum silently changes this to -1.5 at
    # iteration 250. V10's finish rate collapsed immediately afterward. Hold
    # the champion's smoothing pressure fixed so speed is the only curriculum.
    cfg.curriculum.pop("action_rate_weight", None)
    cfg.rewards["joint_torques_l2"].weight = -8.0e-4
    # Permit a small recovery window beyond the dense 20-degree shaping range.
    # v7 terminated at 20 degrees so often that it learned mostly from aborted
    # heats instead of from successful centreline finishes.
    cfg.terminations["race_out_of_lane"] = TerminationTermCfg(
        func=race_out_of_lane,
        params={"lane_half_width_m": 0.75},
        time_out=False,
    )
    cfg.terminations["race_heading_departure"] = TerminationTermCfg(
        func=race_heading_departure,
        params={"maximum_yaw_rad": 0.610865},
        time_out=False,
    )
    return cfg


MicroduckRace5RlCfg = dataclasses.replace(
    MicroduckRaceRlCfg,
    # Twice the rollout horizon gives each PPO update more temporal context;
    # 2,048 environments in the launch script produce 98,304 transitions per
    # iteration while retaining the exact deployable actor architecture.
    num_steps_per_env=48,
    algorithm=dataclasses.replace(
        MicroduckRaceRlCfg.algorithm,
        # Keep PPO inside a small trust region around the warm-start policy.
        # The race evaluator is the second line of defense: it will reject a
        # faster checkpoint if cruise, braking, steering, or stability regresses.
        learning_rate=5.0e-6,
        desired_kl=2.5e-4,
        clip_param=0.02,
        num_learning_epochs=1,
        symmetry_cfg={
            **SYMMETRY_CFG,
            "use_data_augmentation": True,
            "mirror_loss_coeff": 2.0,
        },
    ),
    experiment_name="velocity_race5",
    run_name="velocity_race5",
    save_interval=5,
)
