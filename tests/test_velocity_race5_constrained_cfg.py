from mjlab_microduck.tasks.microduck_velocity_race5_constrained_env_cfg import (
    MicroduckRace5ConstrainedRlCfg,
    make_microduck_velocity_race5_constrained_env_cfg,
    race_usable_launch_speed,
    race_usable_speed_squared,
)


def test_constrained_race5_makes_speed_pay_only_when_it_is_usable():
    cfg = make_microduck_velocity_race5_constrained_env_cfg()

    assert cfg.rewards["race_usable_speed"].func is race_usable_speed_squared
    assert cfg.rewards["race_usable_launch"].func is race_usable_launch_speed
    assert cfg.rewards["race_usable_speed"].weight > cfg.rewards["race_speed_squared"].weight
    assert cfg.rewards["race_usable_launch"].weight > cfg.rewards["race_launch_speed"].weight
    assert cfg.rewards["race_finish"].weight == 650.0
    assert cfg.rewards["race_elapsed"].weight == -1.2


def test_constrained_race5_uses_a_tighter_v11_finetune_trust_region():
    algorithm = MicroduckRace5ConstrainedRlCfg.algorithm

    assert algorithm.learning_rate == 2.0e-6
    assert algorithm.desired_kl == 1.0e-4
    assert algorithm.clip_param == 0.01
    assert algorithm.num_learning_epochs == 1
