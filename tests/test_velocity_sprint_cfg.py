from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks.microduck_velocity_sprint_env_cfg import (
    MicroduckSprintRlCfg,
    make_microduck_velocity_sprint_env_cfg,
)


def test_sprint_uses_measured_speed_as_primary_objective():
    cfg = make_microduck_velocity_sprint_env_cfg()
    tracking = cfg.rewards["track_linear_velocity"]

    assert tracking.func is mdp.track_linear_velocity
    assert tracking.weight == 8.0
    assert tracking.params == {"command_name": "twist", "std": 0.35}
    assert cfg.rewards["forward_speed_progress"].weight == 6.0
    assert tracking.weight > cfg.rewards["wheel_speed"].weight
    assert "braking" not in cfg.rewards


def test_sprint_command_is_forward_target_speed_with_stop_samples():
    command = make_microduck_velocity_sprint_env_cfg().commands["twist"]

    assert command.ranges.lin_vel_x == (0.30, 0.60)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.rel_standing_envs == 0.0


def test_sprint_keeps_deployable_runner_and_stability_pressure():
    cfg = make_microduck_velocity_sprint_env_cfg()

    assert MicroduckSprintRlCfg.experiment_name == "velocity_sprint"
    assert MicroduckSprintRlCfg.actor.obs_normalization is True
    assert MicroduckSprintRlCfg.critic.obs_normalization is True
    assert MicroduckSprintRlCfg.algorithm.learning_rate == 2.0e-4
    assert MicroduckSprintRlCfg.algorithm.desired_kl == 0.003
    assert MicroduckSprintRlCfg.algorithm.clip_param == 0.10
    assert cfg.rewards["upright"].weight > 0.0
    assert cfg.rewards["body_ang_vel"].weight < 0.0
    assert cfg.rewards["heading_hold"].weight > 0.0
