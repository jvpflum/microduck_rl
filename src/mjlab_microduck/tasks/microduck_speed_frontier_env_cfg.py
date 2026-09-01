"""Direct official-friction, phase-conditioned MicroDuck speed discovery.

This task searches for a periodic residual around the protected official
friction champion at the evaluator's bearing drag from the first transition.
The custom runner freezes the champion actor and trains only phase-input
weights. A randomized oscillator makes cadence changes easy to represent,
while physically consistent moving resets expose the adapter to cruise states.

The oscillator reuses three of the six normally zero-padded body-command
channels.  That preserves the deployment family's 61D actor and 78D critic
interfaces, and lets a proven skating checkpoint start with exactly its old
behavior because those input channels were zero throughout its training.
"""

from __future__ import annotations

import dataclasses
import math
from copy import deepcopy

import torch

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, MetricsTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.actuator import FrictionDRBamActuatorCfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_discovery_env_cfg import (
    MicroduckSpeedDiscoveryRlCfg,
    SPEED_DISCOVERY_CAP_MPS,
    make_microduck_speed_discovery_env_cfg,
)


OFFICIAL_WHEEL_FRICTION = 0.003
WHEEL_RADIUS_M = 0.015
XL330_CURRENT_LIMIT_A = 1.75
PHASE_FREQUENCY_RANGE_HZ = (1.75, 5.50)
BOOTSTRAP_SPEED_RANGE_MPS = (0.50, 3.00)


def reset_frontier_state(
    env,
    env_ids: torch.Tensor | slice | None,
    frequency_range_hz: tuple[float, float] = PHASE_FREQUENCY_RANGE_HZ,
    bootstrap_speed_range_mps: tuple[float, float] = BOOTSTRAP_SPEED_RANGE_MPS,
    bootstrap_fraction_stages: tuple[tuple[int, float], ...] = (
        (0, 0.20),
        (2_500 * 24, 0.12),
        (5_000 * 24, 0.08),
    ),
    wheel_radius_m: float = WHEEL_RADIUS_M,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset oscillator state and optional no-slip forward launch.

    Moving resets set both chassis velocity and wheel angular velocity.  Eighty
    percent of the early population starts from rest, so acceleration cannot
    be solved by relying on injected kinetic energy.  The moving fraction falls
    later while retaining a small high-speed cohort to train the cruise regime.
    """
    if env_ids is None or isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(ids) == 0:
        return

    if not hasattr(env, "_frontier_phase_frequency_hz"):
        env._frontier_phase_frequency_hz = torch.zeros(env.num_envs, device=env.device)
        env._frontier_phase_offset = torch.zeros(env.num_envs, device=env.device)
        env._frontier_initial_speed_mps = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_speed_discovery_peak"):
        env._speed_discovery_peak = torch.zeros(env.num_envs, device=env.device)

    f_lo, f_hi = frequency_range_hz
    env._frontier_phase_frequency_hz[ids] = torch.empty(
        len(ids), device=env.device
    ).uniform_(f_lo, f_hi)
    env._frontier_phase_offset[ids] = torch.empty(
        len(ids), device=env.device
    ).uniform_(0.0, 2.0 * math.pi)

    fraction = float(bootstrap_fraction_stages[0][1])
    step = int(env.common_step_counter)
    for threshold, candidate in bootstrap_fraction_stages:
        if step >= threshold:
            fraction = float(candidate)

    moving = torch.rand(len(ids), device=env.device) < fraction
    lo, hi = bootstrap_speed_range_mps
    speed = torch.zeros(len(ids), device=env.device)
    speed[moving] = torch.empty(int(moving.sum()), device=env.device).uniform_(lo, hi)
    env._frontier_initial_speed_mps[ids] = speed
    env._speed_discovery_peak[ids] = speed

    asset = env.scene[asset_cfg.name]
    root_velocity = torch.zeros(len(ids), 6, device=env.device)
    root_velocity[:, 0] = speed
    asset.write_root_link_velocity_to_sim(root_velocity, env_ids=ids)

    wheel_ids, _ = asset.find_joints(r"^passive_.*wheel$")
    if wheel_ids:
        wheel_id_tensor = torch.tensor(wheel_ids, dtype=torch.long, device=env.device)
        wheel_velocity = (speed / wheel_radius_m).unsqueeze(-1).expand(-1, len(wheel_ids))
        asset.write_joint_velocity_to_sim(
            wheel_velocity, joint_ids=wheel_id_tensor, env_ids=ids
        )


def frontier_phase_command(env) -> torch.Tensor:
    """Six-channel command slot carrying phase and cadence in its first 3D."""
    if not hasattr(env, "_frontier_phase_frequency_hz"):
        # ObservationManager can query shapes before the first reset callback.
        env._frontier_phase_frequency_hz = torch.full(
            (env.num_envs,), sum(PHASE_FREQUENCY_RANGE_HZ) / 2.0, device=env.device
        )
        env._frontier_phase_offset = torch.zeros(env.num_envs, device=env.device)
        env._frontier_initial_speed_mps = torch.zeros(env.num_envs, device=env.device)
    time_s = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
    phase = env._frontier_phase_offset + 2.0 * math.pi * env._frontier_phase_frequency_hz * time_s
    f_lo, f_hi = PHASE_FREQUENCY_RANGE_HZ
    normalized_frequency = 2.0 * (env._frontier_phase_frequency_hz - f_lo) / (f_hi - f_lo) - 1.0
    phase_features = torch.stack(
        (torch.sin(phase), torch.cos(phase), normalized_frequency), dim=-1
    )
    return torch.nn.functional.pad(phase_features, (0, 3))


def _world_speed(env) -> torch.Tensor:
    speed = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
    speed = torch.nan_to_num(
        speed,
        nan=0.0,
        posinf=SPEED_DISCOVERY_CAP_MPS,
        neginf=-SPEED_DISCOVERY_CAP_MPS,
    ).clamp(-SPEED_DISCOVERY_CAP_MPS, SPEED_DISCOVERY_CAP_MPS)
    if hasattr(env, "_speed_discovery_peak"):
        env._speed_discovery_peak = torch.maximum(env._speed_discovery_peak, speed)
    return speed


def frontier_world_speed(env) -> torch.Tensor:
    return _world_speed(env)


def frontier_world_speed_squared(env) -> torch.Tensor:
    speed = _world_speed(env)
    return speed * torch.abs(speed)


def frontier_speed_gain(env) -> torch.Tensor:
    initial = getattr(env, "_frontier_initial_speed_mps", None)
    if initial is None:
        initial = torch.zeros(env.num_envs, device=env.device)
    return (_world_speed(env) - initial).clamp(
        -SPEED_DISCOVERY_CAP_MPS, SPEED_DISCOVERY_CAP_MPS
    )


def frontier_final_speed(env, final_fraction: float = 0.40) -> torch.Tensor:
    first_final_step = int((1.0 - final_fraction) * env.max_episode_length)
    active = (env.episode_length_buf >= first_final_step).to(torch.float32)
    return active * _world_speed(env)


def frontier_lateral_speed_squared(env) -> torch.Tensor:
    lateral = env.scene["robot"].data.root_link_lin_vel_w[:, 1]
    return torch.nan_to_num(lateral, nan=0.0).square()


def frontier_initial_speed_metric(env) -> torch.Tensor:
    return getattr(
        env,
        "_frontier_initial_speed_mps",
        torch.zeros(env.num_envs, device=env.device),
    )


def frontier_frequency_metric(env) -> torch.Tensor:
    return getattr(
        env,
        "_frontier_phase_frequency_hz",
        torch.zeros(env.num_envs, device=env.device),
    )


def make_microduck_speed_frontier_env_cfg(play: bool = False):
    cfg = make_microduck_speed_discovery_env_cfg(play=play)
    # Optimize the same sustained horizon used by the official evaluator.
    # Short launch episodes hid slow heading drift that erased world-X speed.
    cfg.episode_length_s = 20.0

    # Nominal BAM physics with the deployable firmware current ceiling.  This
    # avoids spending the clean-discovery budget on voltage/mass randomization
    # and prevents a high-current simulation exploit from becoming champion.
    frontier_robot = deepcopy(MICRODUCK_WALK_ROLLERS_ROBOT_CFG)
    frontier_robot.articulation.actuators = (
        FrictionDRBamActuatorCfg(
            motor_name="xl330",
            model="m6",
            target_names_expr=(r"^(?!passive_).*",),
            kp_fw=200.0,
            vin=7.4,
            vin_range=None,
            vin_drop_gain_range=None,
            vin_min=None,
            max_current=XL330_CURRENT_LIMIT_A,
            delay_min_lag=3,
            delay_max_lag=3,
        ),
    )
    cfg.scene.entities["robot"] = frontier_robot

    cfg.events["frontier_official_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^passive_.*wheel$",)
            ),
            "operation": "abs",
            "ranges": (OFFICIAL_WHEEL_FRICTION, OFFICIAL_WHEEL_FRICTION),
        },
    )
    cfg.events["reset_frontier_state"] = EventTermCfg(
        func=reset_frontier_state,
        mode="reset",
    )

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.80, 0.80)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)

    # Preserve checkpoint and firmware observation sizes by replacing the
    # existing zero-padded 6D body-command slot in place.  Assignment to an
    # existing dict key keeps the original observation ordering.
    phase_term = ObservationTermCfg(func=frontier_phase_command)
    cfg.observations["actor"].terms["body_command"] = phase_term
    cfg.observations["critic"].terms["body_command"] = deepcopy(phase_term)

    safety_terms = {
        name: deepcopy(cfg.rewards[name])
        for name in ("self_collisions", "action_over_limit", "action_rate_l2")
    }
    cfg.rewards.clear()
    cfg.rewards["world_speed"] = RewardTermCfg(func=frontier_world_speed, weight=4.0)
    cfg.rewards["world_speed_squared"] = RewardTermCfg(
        func=frontier_world_speed_squared, weight=1.0
    )
    cfg.rewards["speed_gain"] = RewardTermCfg(func=frontier_speed_gain, weight=6.0)
    cfg.rewards["final_speed"] = RewardTermCfg(func=frontier_final_speed, weight=3.0)
    cfg.rewards["alive"] = RewardTermCfg(
        func=microduck_mdp.speed_discovery_alive, weight=0.25
    )
    cfg.rewards["fall"] = RewardTermCfg(
        func=microduck_mdp.speed_discovery_fall, weight=-500.0
    )
    cfg.rewards["lateral_speed"] = RewardTermCfg(
        func=frontier_lateral_speed_squared, weight=-0.35
    )
    # Preserve policy-side safety terms with their exact action/entity params.
    for name, weight in (
        ("self_collisions", -0.05),
        ("action_over_limit", -0.10),
        ("action_rate_l2", -0.02),
    ):
        term = safety_terms[name]
        term.weight = weight
        cfg.rewards[name] = term

    cfg.curriculum.clear()
    cfg.metrics["frontier_initial_speed_mps"] = MetricsTermCfg(
        func=frontier_initial_speed_metric
    )
    cfg.metrics["frontier_speed_gain_mps"] = MetricsTermCfg(func=frontier_speed_gain)
    cfg.metrics["frontier_phase_frequency_hz"] = MetricsTermCfg(
        func=frontier_frequency_metric
    )
    return cfg


_actor_distribution = dict(MicroduckSpeedDiscoveryRlCfg.actor.distribution_cfg)
_actor_distribution["init_std"] = 0.08

MicroduckSpeedFrontierRlCfg = dataclasses.replace(
    MicroduckSpeedDiscoveryRlCfg,
    actor=dataclasses.replace(
        MicroduckSpeedDiscoveryRlCfg.actor,
        distribution_cfg=_actor_distribution,
    ),
    algorithm=dataclasses.replace(
        MicroduckSpeedDiscoveryRlCfg.algorithm,
        learning_rate=1.0e-4,
        desired_kl=0.01,
        clip_param=0.15,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
    ),
    experiment_name="microduck_speed_frontier",
    run_name="microduck_speed_frontier",
    save_interval=50,
    max_iterations=8_000,
)
