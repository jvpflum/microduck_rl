from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_speed_discovery_env_cfg import (
    MicroduckSpeedDiscoveryRlCfg,
    SPEED_DISCOVERY_CAP_MPS,
    SPEED_DISCOVERY_STAGES,
    make_microduck_speed_discovery_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)


def test_speed_discovery_is_separate_from_normal_roller_recipe():
    normal = make_microduck_velocity_rollers_env_cfg()
    discovery = make_microduck_speed_discovery_env_cfg()

    assert "pose" in normal.rewards
    assert "wheel_speed" in normal.rewards
    assert "push_robot" in normal.events
    assert "pose" not in discovery.rewards
    assert "wheel_speed" not in discovery.rewards
    assert "push_robot" not in discovery.events


def test_speed_discovery_rewards_actual_chassis_speed_beyond_command():
    cfg = make_microduck_speed_discovery_env_cfg()

    assert cfg.rewards["forward_velocity_mps"].func is microduck_mdp.speed_discovery_forward_velocity
    assert cfg.rewards["forward_velocity_mps"].weight == 5.0
    assert cfg.rewards["forward_velocity_mps"].params["safety_cap_mps"] == 7.5
    assert cfg.rewards["forward_velocity_squared"].weight == 0.75
    assert cfg.rewards["speed_target_progress"].weight == 1.0
    assert cfg.rewards["speed_target_progress"].params["target_speed_mps"] == 2.5
    assert cfg.rewards["fall"].weight == -500.0
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    assert cfg.rewards["action_over_limit"].weight == -0.05
    assert SPEED_DISCOVERY_CAP_MPS > 6.7


def test_speed_discovery_uses_nominal_flat_physics_and_clean_launch():
    cfg = make_microduck_speed_discovery_env_cfg()
    reset = cfg.events["reset_base"].params

    assert cfg.scene.terrain.terrain_type == "plane"
    assert set(cfg.events) == {
        "reset_base",
        "reset_robot_joints",
        "expand_bam_friction_fields",
        "reset_action_history",
    }
    assert reset["pose_range"]["x"] == (0.0, 0.0)
    assert reset["pose_range"]["y"] == (0.0, 0.0)
    assert reset["pose_range"]["roll"] == (0.0, 0.0)
    assert reset["pose_range"]["pitch"] == (0.0, 0.0)
    assert reset["pose_range"]["yaw"] == (0.0, 0.0)
    assert set(cfg.terminations) == {"time_out", "fell_over", "nan_state"}
    for term in cfg.observations["actor"].terms.values():
        assert term.noise is None
        assert term.delay_min_lag == 0
        assert term.delay_max_lag == 0


def test_speed_curriculum_is_performance_gated_through_15_mph():
    cfg = make_microduck_speed_discovery_env_cfg()
    term = cfg.curriculum["speed_stage"]
    targets = [stage["target_speed_mps"] for stage in SPEED_DISCOVERY_STAGES]

    assert targets == [2.5, 3.5, 4.5, 5.5, 6.7]
    assert term.func is microduck_mdp.speed_discovery_performance_curriculum
    assert term.params["min_attempts"] == 4096
    assert term.params["required_windows"] == 2
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.8, 0.8)
    assert cfg.commands["twist"].ranges.lin_vel_y == (-0.02, 0.02)
    assert cfg.commands["twist"].ranges.ang_vel_z == (-0.05, 0.05)


def test_speed_discovery_ppo_batch_profiles_remain_sensible():
    algo = MicroduckSpeedDiscoveryRlCfg.algorithm

    assert MicroduckSpeedDiscoveryRlCfg.num_steps_per_env == 24
    assert MicroduckSpeedDiscoveryRlCfg.save_interval == 25
    assert MicroduckSpeedDiscoveryRlCfg.max_iterations == 6000
    assert algo.learning_rate == 3.0e-5
    assert algo.clip_param == 0.10
    assert algo.entropy_coef == 0.001
    assert algo.num_learning_epochs == 5
    assert algo.num_mini_batches == 4
    assert algo.symmetry_cfg is None
    assert 4096 * 24 == 98_304
    assert 8192 * 24 == 196_608
    assert 4096 * 24 // 4 == 8192 * 24 // 8 == 24_576


def test_speed_curriculum_promotes_from_speed_and_survival_not_iteration():
    command = SimpleNamespace(cfg=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=None)))
    target_reward = SimpleNamespace(params={"target_speed_mps": 2.5})
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        max_episode_length=100,
        episode_length_buf=torch.tensor([100, 90]),
        _speed_discovery_sum=torch.tensor([220.0, 205.0]),
        _speed_discovery_count=torch.tensor([100, 100]),
        _speed_discovery_peak=torch.tensor([2.4, 2.3]),
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        reward_manager=SimpleNamespace(get_term_cfg=lambda _name: target_reward),
    )
    result = microduck_mdp.speed_discovery_performance_curriculum(
        env,
        torch.tensor([0, 1]),
        command_name="twist",
        target_reward_name="speed_target_progress",
        stages=[dict(stage) for stage in SPEED_DISCOVERY_STAGES],
        min_attempts=2,
        required_windows=1,
    )

    assert result["stage"].item() == 1.0
    assert result["target_speed_mps"].item() == 3.5
    assert command.cfg.ranges.lin_vel_x == (0.8, 0.8)
    assert target_reward.params["target_speed_mps"] == 3.5
    assert result["window_mean_speed_mps"].item() > 2.0
    assert result["window_survival_fraction"].item() >= 0.9
from types import SimpleNamespace

import torch
