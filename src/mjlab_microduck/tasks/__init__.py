from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


class MicroduckFrontierAdapterRunner(MicroduckOnPolicyRunner):
    """Train a phase adapter while keeping the proven skate actor immutable."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        actor = self.alg.actor
        for parameter in actor.parameters():
            parameter.requires_grad_(False)

        phase_input_weights = actor.mlp[0].weight
        if phase_input_weights.shape[1] != 61:
            raise ValueError(
                "Frontier adapter requires the unified 61D actor observation"
            )
        phase_input_weights.requires_grad_(True)
        gradient_mask = phase_input_weights.new_zeros(phase_input_weights.shape)
        gradient_mask[:, -6:] = 1.0
        phase_input_weights.register_hook(lambda gradient: gradient * gradient_mask)
        # RSL-RL updates normalizer buffers separately from optimizer
        # parameters. Keep the champion's original observation calibration
        # immutable; phase inputs use the unit moments installed by the
        # frontier warm-start script.
        actor.obs_normalizer.eval()
        actor.update_normalization = lambda observations: None

        self.frontier_trainable_actor_parameters = int(gradient_mask.sum().item())
        print(
            "[FrontierAdapter] froze champion actor; training "
            f"{self.frontier_trainable_actor_parameters} phase-input weights"
        )


class MicroduckConservativeHeadRunner(MicroduckOnPolicyRunner):
    """Adapt only the action head while preserving a deployed speed gait."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        actor = self.alg.actor
        for parameter in actor.parameters():
            parameter.requires_grad_(False)

        action_head = actor.mlp[-1]
        action_head.weight.requires_grad_(True)
        action_head.bias.requires_grad_(True)
        # The exact deployment normalizer is part of the policy. Updating it
        # during a short refinement changes every hidden activation at once.
        actor.obs_normalizer.eval()
        actor.update_normalization = lambda observations: None

        trainable = sum(
            parameter.numel() for parameter in actor.parameters()
            if parameter.requires_grad
        )
        print(
            "[ConservativeHead] froze actor normalizer/body; training "
            f"{trainable} action-head parameters"
        )


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_sprint_env_cfg import (
    make_microduck_velocity_sprint_env_cfg,
    MicroduckSprintRlCfg,
)
from .microduck_velocity_race_env_cfg import (
    make_microduck_velocity_race_env_cfg,
    MicroduckRaceRlCfg,
)
from .microduck_velocity_race5_env_cfg import (
    make_microduck_velocity_race5_env_cfg,
    MicroduckRace5RlCfg,
)
from .microduck_velocity_race5_constrained_env_cfg import (
    make_microduck_velocity_race5_constrained_env_cfg,
    MicroduckRace5ConstrainedRlCfg,
)
from .microduck_velocity_race5_frontier_env_cfg import (
    make_microduck_velocity_race5_frontier_env_cfg,
    MicroduckRace5FrontierRlCfg,
)
from .microduck_velocity_race5_fusion_env_cfg import (
    MicroduckRace5FusionRlCfg,
    make_microduck_velocity_race5_fusion_env_cfg,
)
from .microduck_velocity_race5_v59_refinement_env_cfg import (
    MicroduckRace5V59RefinementRlCfg,
    make_microduck_velocity_race5_v59_refinement_env_cfg,
)
from .microduck_velocity_race5_clean_launch_env_cfg import (
    MicroduckRace5CleanLaunchRlCfg,
    make_microduck_velocity_race5_clean_launch_env_cfg,
)
from .microduck_speed_discovery_env_cfg import (
    make_microduck_speed_discovery_env_cfg,
    MicroduckSpeedDiscoveryRlCfg,
)
from .microduck_speed_straightening_env_cfg import (
    make_microduck_speed_straightening_env_cfg,
    MicroduckSpeedStraighteningRlCfg,
)
from .microduck_speed_retention_env_cfg import (
    make_microduck_speed_retention_env_cfg,
    MicroduckSpeedRetentionRlCfg,
)
from .microduck_speed_retention_boost_env_cfg import (
    make_microduck_speed_retention_boost_env_cfg,
    MicroduckSpeedRetentionBoostRlCfg,
)
from .microduck_speed_friction_transfer_env_cfg import (
    make_microduck_speed_friction_transfer_env_cfg,
    MicroduckSpeedFrictionTransferRlCfg,
)
from .microduck_speed_scout_transfer_env_cfg import (
    make_microduck_speed_scout_transfer_env_cfg,
    MicroduckSpeedScoutTransferRlCfg,
)
from .microduck_speed_friction_tether_env_cfg import (
    make_microduck_speed_friction_tether_env_cfg,
    MicroduckSpeedFrictionTetherRlCfg,
)
from .microduck_speed_official_adaptation_env_cfg import (
    make_microduck_speed_official_adaptation_env_cfg,
    MicroduckSpeedOfficialAdaptationRlCfg,
)
from .microduck_speed_command_breakthrough_env_cfg import (
    make_microduck_speed_command_breakthrough_env_cfg,
    MicroduckSpeedCommandBreakthroughRlCfg,
)
from .microduck_speed_teacher_guided_env_cfg import (
    make_microduck_speed_teacher_guided_env_cfg,
    MicroduckSpeedTeacherGuidedRlCfg,
)
from .microduck_speed_frontier_env_cfg import (
    make_microduck_speed_frontier_env_cfg,
    MicroduckSpeedFrontierRlCfg,
)
from .microduck_speed_final_env_cfg import (
    make_microduck_speed_final_env_cfg,
    MicroduckSpeedFinalRlCfg,
)
from .microduck_speed_v65_final_env_cfg import (
    make_microduck_speed_v65_final_env_cfg,
    MicroduckSpeedV65FinalRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_hop_env_cfg import (
    make_microduck_roller_hop_env_cfg,
    MicroduckRollerHopRlCfg,
)
from .microduck_roller_backflip_env_cfg import (
    make_microduck_roller_backflip_env_cfg,
    MicroduckRollerBackflipRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SPRINT — straight-line target-speed specialist.
register_mjlab_task(
    task_id="Mjlab-Velocity-Sprint-MicroDuck",
    env_cfg=make_microduck_velocity_sprint_env_cfg(),
    play_env_cfg=make_microduck_velocity_sprint_env_cfg(play=True),
    rl_cfg=MicroduckSprintRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller RACE — fixed-distance, uncapped forward-speed specialist.
register_mjlab_task(
    task_id="Mjlab-Velocity-Race-MicroDuck",
    env_cfg=make_microduck_velocity_race_env_cfg(),
    play_env_cfg=make_microduck_velocity_race_env_cfg(play=True),
    rl_cfg=MicroduckRaceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller RACE5 — 5 mph target with a 10 mph simulation stretch cap.
register_mjlab_task(
    task_id="Mjlab-Velocity-Race5-MicroDuck",
    env_cfg=make_microduck_velocity_race5_env_cfg(),
    play_env_cfg=make_microduck_velocity_race5_env_cfg(play=True),
    rl_cfg=MicroduckRace5RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller RACE5 CONSTRAINED — V11 fine-tuning for speed that remains on line.
register_mjlab_task(
    task_id="Mjlab-Velocity-Race5Constrained-MicroDuck",
    env_cfg=make_microduck_velocity_race5_constrained_env_cfg(),
    play_env_cfg=make_microduck_velocity_race5_constrained_env_cfg(play=True),
    rl_cfg=MicroduckRace5ConstrainedRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller RACE5 FRONTIER — distill the control-aware hybrid, then exceed it at
# the exact official-friction operating point without forgetting other commands.
register_mjlab_task(
    task_id="Mjlab-Velocity-Race5Frontier-MicroDuck",
    env_cfg=make_microduck_velocity_race5_frontier_env_cfg(),
    play_env_cfg=make_microduck_velocity_race5_frontier_env_cfg(play=True),
    rl_cfg=MicroduckRace5FrontierRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller RACE5 FUSION — optimize a V57b-guided speed branch, then deploy it
# inside the immutable qualified control-aware shell.
register_mjlab_task(
    task_id="Mjlab-Velocity-Race5Fusion-MicroDuck",
    env_cfg=make_microduck_velocity_race5_fusion_env_cfg(),
    play_env_cfg=make_microduck_velocity_race5_fusion_env_cfg(play=True),
    rl_cfg=MicroduckRace5FusionRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller RACE5 V59 REFINEMENT — make conservative improvements to the exact
# V59 high-speed branch used by V66; deployment retains V66's control router.
register_mjlab_task(
    task_id="Mjlab-Velocity-Race5V59Refinement-MicroDuck",
    env_cfg=make_microduck_velocity_race5_v59_refinement_env_cfg(),
    play_env_cfg=make_microduck_velocity_race5_v59_refinement_env_cfg(play=True),
    rl_cfg=MicroduckRace5V59RefinementRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller RACE5 CLEAN LAUNCH — remove the V59 speed branch's initial crab gait;
# V66 remains responsible for all non-straight deployment commands.
register_mjlab_task(
    task_id="Mjlab-Velocity-Race5CleanLaunch-MicroDuck",
    env_cfg=make_microduck_velocity_race5_clean_launch_env_cfg(),
    play_env_cfg=make_microduck_velocity_race5_clean_launch_env_cfg(play=True),
    rl_cfg=MicroduckRace5CleanLaunchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SPEED DISCOVERY — nominal physics and unconstrained chassis velocity.
register_mjlab_task(
    task_id="Mjlab-SpeedDiscovery-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_discovery_env_cfg(),
    play_env_cfg=make_microduck_speed_discovery_env_cfg(play=True),
    rl_cfg=MicroduckSpeedDiscoveryRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SPEED STRAIGHTENING — retain the fast gait while restoring race line.
register_mjlab_task(
    task_id="Mjlab-SpeedStraightening-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_straightening_env_cfg(),
    play_env_cfg=make_microduck_speed_straightening_env_cfg(play=True),
    rl_cfg=MicroduckSpeedStraighteningRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedRetention-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_retention_env_cfg(),
    play_env_cfg=make_microduck_speed_retention_env_cfg(play=True),
    rl_cfg=MicroduckSpeedRetentionRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedRetentionBoost-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_retention_boost_env_cfg(),
    play_env_cfg=make_microduck_speed_retention_boost_env_cfg(play=True),
    rl_cfg=MicroduckSpeedRetentionBoostRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedFrictionTransfer-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_friction_transfer_env_cfg(),
    play_env_cfg=make_microduck_speed_friction_transfer_env_cfg(play=True),
    rl_cfg=MicroduckSpeedFrictionTransferRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedScoutTransfer-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_scout_transfer_env_cfg(),
    play_env_cfg=make_microduck_speed_scout_transfer_env_cfg(play=True),
    rl_cfg=MicroduckSpeedScoutTransferRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedFrictionTether-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_friction_tether_env_cfg(),
    play_env_cfg=make_microduck_speed_friction_tether_env_cfg(play=True),
    rl_cfg=MicroduckSpeedFrictionTetherRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedOfficialAdaptation-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_official_adaptation_env_cfg(),
    play_env_cfg=make_microduck_speed_official_adaptation_env_cfg(play=True),
    rl_cfg=MicroduckSpeedOfficialAdaptationRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedCommandBreakthrough-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_command_breakthrough_env_cfg(),
    play_env_cfg=make_microduck_speed_command_breakthrough_env_cfg(play=True),
    rl_cfg=MicroduckSpeedCommandBreakthroughRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedTeacherGuided-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_teacher_guided_env_cfg(),
    play_env_cfg=make_microduck_speed_teacher_guided_env_cfg(play=True),
    rl_cfg=MicroduckSpeedTeacherGuidedRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SpeedFrontier-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_frontier_env_cfg(),
    play_env_cfg=make_microduck_speed_frontier_env_cfg(play=True),
    rl_cfg=MicroduckSpeedFrontierRlCfg,
    runner_cls=MicroduckFrontierAdapterRunner,
)

# Roller SPEED FINAL — exact V59 initialization plus a bounded official-friction
# search for the replaceable V66 high-command branch.
register_mjlab_task(
    task_id="Mjlab-SpeedFinal-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_final_env_cfg(),
    play_env_cfg=make_microduck_speed_final_env_cfg(play=True),
    rl_cfg=MicroduckSpeedFinalRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller V65 FINAL — exact V66 high branch, fixed-rate action-head-only search.
register_mjlab_task(
    task_id="Mjlab-SpeedV65Final-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_speed_v65_final_env_cfg(),
    play_env_cfg=make_microduck_speed_v65_final_env_cfg(play=True),
    rl_cfg=MicroduckSpeedV65FinalRlCfg,
    runner_cls=MicroduckConservativeHeadRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller HOP — stationary two-skate takeoff and controlled two-skate landing.
register_mjlab_task(
    task_id="Mjlab-RollerHop-Flat-MicroDuck",
    env_cfg=make_microduck_roller_hop_env_cfg(),
    play_env_cfg=make_microduck_roller_hop_env_cfg(play=True),
    rl_cfg=MicroduckRollerHopRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller BACKFLIP — rolling takeoff, airborne backward rotation, skate landing.
register_mjlab_task(
    task_id="Mjlab-RollerBackflip-Flat-MicroDuck",
    env_cfg=make_microduck_roller_backflip_env_cfg(),
    play_env_cfg=make_microduck_roller_backflip_env_cfg(play=True),
    rl_cfg=MicroduckRollerBackflipRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_allcollisions_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# allcollisions backlash robot — same model as their base task in each case.
_BL_ALLCOL = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_ALLCOL),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Sprint-Backlash-MicroDuck", make_microduck_velocity_sprint_env_cfg, {}, MicroduckSprintRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Race-Backlash-MicroDuck", make_microduck_velocity_race_env_cfg, {}, MicroduckRaceRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Race5-Backlash-MicroDuck", make_microduck_velocity_race5_env_cfg, {}, MicroduckRace5RlCfg, _BL_ROLLERS),
    ("Mjlab-SpeedDiscovery-Flat-Backlash-MicroDuck-Rollers", make_microduck_speed_discovery_env_cfg, {}, MicroduckSpeedDiscoveryRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerHop-Flat-Backlash-MicroDuck", make_microduck_roller_hop_env_cfg, {}, MicroduckRollerHopRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerBackflip-Flat-Backlash-MicroDuck", make_microduck_roller_backflip_env_cfg, {}, MicroduckRollerBackflipRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
