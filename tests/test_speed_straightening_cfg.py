from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
    world_forward_velocity,
    world_forward_velocity_squared,
)


def test_straightening_keeps_speed_discovery_small_reward_surface():
    cfg = make_microduck_speed_straightening_env_cfg()

    assert cfg.rewards["forward_velocity_mps"].weight == 2.0
    assert cfg.rewards["forward_velocity_squared"].weight == 0.25
    assert cfg.rewards["world_forward_velocity_mps"].func is world_forward_velocity
    assert cfg.rewards["world_forward_velocity_mps"].weight == 5.0
    assert (
        cfg.rewards["world_forward_velocity_squared"].func
        is world_forward_velocity_squared
    )
    assert cfg.rewards["heading_hold"].func is microduck_mdp.heading_hold_reward
    assert cfg.rewards["lane_error"].weight == -1.0
    assert cfg.rewards["world_lateral_velocity"].weight == -1.0
    assert cfg.rewards["heading_error"].weight == -1.0
    assert "pose" not in cfg.rewards
    assert "joint_torques_l2" not in cfg.rewards


def test_straightening_exposes_closed_loop_correction_without_actor_shape_change():
    cfg = make_microduck_speed_straightening_env_cfg()
    command = cfg.commands["twist"]

    assert isinstance(command, microduck_mdp.RaceLineVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (0.8, 0.8)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (-0.18, 0.18)
    assert command.max_correction == 0.18
    # No observation term is added or removed; the inherited contract resolves
    # to 61D when the environment binds its 14 actuated joints.
    assert tuple(cfg.observations["actor"].terms) == (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    )
    assert MicroduckSpeedStraighteningRlCfg.experiment_name == "microduck_speed_straightening"
