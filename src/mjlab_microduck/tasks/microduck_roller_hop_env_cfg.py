"""One-shot roller hop built on Pollen's proven roller-crouch stack.

The policy starts upright and stationary, preloads, leaves the ground with both
skates, then lands upright on both skates.  The phase command is only a trigger
and timing cue.  Completion and landing rewards are state-gated: standing still
at reset cannot earn the hop or landing rewards.

V1 deliberately targets a repeatable 20 mm clearance.  Higher jumps and moving
entries belong in later curricula after this stationary landing is reliable.
"""

from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


HOP_PERIOD = 3.0
STAND_HEIGHT = 0.115
LOAD_HEIGHT = 0.085
TARGET_CLEARANCE = 0.020
TAKEOFF_LATCH_CLEARANCE = 0.004

LOAD_START = 0.08
LOAD_END = 0.34
LAUNCH_START = 0.25
LAUNCH_END = 0.52

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def make_microduck_roller_hop_env_cfg(play: bool = False):
    """Create the stationary roller-hop environment."""
    cfg = make_microduck_roller_crouch_env_cfg(play=play)
    cfg.episode_length_s = HOP_PERIOD

    # One stationary attempt per episode.  Keep the roller model, BAM actuator,
    # 61D observation layout, sensor setup, NaN guards, and sim2real DR inherited
    # from roller-crouch, but remove mid-hop pushes during skill discovery.
    cfg.events.pop("push_robot", None)
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events["reset_hop_state"] = EventTermCfg(
        func=microduck_mdp.reset_roller_hop_state,
        mode="reset",
    )
    # Random exploration commonly tilts before the launch window. Terminating
    # there prevents PPO from ever observing takeoff/landing outcomes, so v1
    # runs the full one-shot episode and penalizes tilt continuously instead.
    cfg.terminations.pop("fell_over", None)

    command = cfg.commands["twist"]
    command.period = HOP_PERIOD
    command.randomize_phase = False

    # Replace the crouch-glide objective.  The dense load/launch hints make the
    # maneuver discoverable, while takeoff progress and all post-flight rewards
    # remain locked behind real two-skate flight.
    body_ang_vel = cfg.rewards["body_ang_vel"]
    body_ang_vel.params["asset_cfg"].body_names = ("trunk_base",)
    body_ang_vel.weight = -0.01
    cfg.rewards.clear()
    cfg.rewards["hop_load_height"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_load_height,
        weight=1.0,
        params={
            "target_height": LOAD_HEIGHT,
            "std": 0.018,
            "phase_start": LOAD_START,
            "phase_end": LOAD_END,
        },
    )
    cfg.rewards["hop_takeoff_velocity"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_takeoff_velocity,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "max_vz": 0.45,
            "phase_start": LAUNCH_START,
            "phase_end": LAUNCH_END,
        },
    )
    cfg.rewards["hop_clearance_progress"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_clearance_progress,
        weight=8.0,
        params={
            "sensor_name": "feet_ground_contact",
            "stand_height": STAND_HEIGHT,
            "target_clearance": TARGET_CLEARANCE,
            "latch_clearance": TAKEOFF_LATCH_CLEARANCE,
        },
    )
    cfg.rewards["hop_flight_upright"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_flight_upright,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "stand_height": STAND_HEIGHT,
            "latch_clearance": TAKEOFF_LATCH_CLEARANCE,
            "upright_std": 0.25,
        },
    )
    cfg.rewards["hop_landing"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_landing_composite,
        weight=5.0,
        params={
            "sensor_name": "feet_ground_contact",
            "stand_height": STAND_HEIGHT,
            "latch_clearance": TAKEOFF_LATCH_CLEARANCE,
            "height_std": 0.025,
            "upright_std": 0.35,
            "pose_std": 0.45,
            "joint_indices": _LEG_JOINTS,
        },
    )
    cfg.rewards["hop_landing_stillness"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_landing_stillness,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "stand_height": STAND_HEIGHT,
            "latch_clearance": TAKEOFF_LATCH_CLEARANCE,
            "linear_std": 0.08,
            "angular_std": 1.0,
        },
    )

    # Low discovery-time regularization: enough to discourage sideways launch,
    # wild commands, shell strikes, and destructive impacts without suppressing
    # the high-force extension the hop physically requires.
    cfg.rewards["horizontal_drift"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_horizontal_drift_penalty,
        weight=-0.5,
    )
    cfg.rewards["tilt"] = RewardTermCfg(
        func=microduck_mdp.roller_hop_tilt_penalty,
        weight=-0.5,
    )
    cfg.rewards["body_ang_vel"] = body_ang_vel
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2,
        weight=-1.0e-4,
    )
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": "self_collision"},
    )

    # Polish only after discovery.  Step counts are PPO iterations × 24 rollout
    # steps, matching the rest of this repository's curriculum convention.
    cfg.curriculum.clear()
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.01},
                {"step": 400 * 24, "weight": -0.03},
                {"step": 900 * 24, "weight": -0.08},
            ],
        },
    )
    cfg.curriculum["landing_impact_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "gentle_landing",
            "weight_stages": [
                {"step": 0, "weight": 2.0e-4},
                {"step": 500 * 24, "weight": 5.0e-4},
                {"step": 1_000 * 24, "weight": 1.0e-3},
            ],
        },
    )

    return cfg


MicroduckRollerHopRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_hop",
    run_name="roller_hop_v1",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1_500,
)
