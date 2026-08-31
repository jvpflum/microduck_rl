"""Official-friction speed adaptation for the preserved 5.41 mph skate gait.

This task intentionally does *not* ramp from zero friction.  Every rollout is
near the Race5 bearing-drag profile, so PPO cannot keep a frictionless-only
solution and call it progress.  World-X speed stays the primary objective;
the small line costs merely reject the obvious steer-by-drifting exploit.
"""

from __future__ import annotations

import dataclasses
import os

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.microduck_speed_straightening_env_cfg import (
    MicroduckSpeedStraighteningRlCfg,
    make_microduck_speed_straightening_env_cfg,
)


# Small, explicit recipe set used by DuckLab's sequential sweep.  This is not
# a blind PPO search: each recipe changes one tradeoff from the same 5.41 mph
# donor and all candidates are certified with the same Race5 evaluator.
OFFICIAL_RECIPES = {
    "balanced": {
        "forward": 3.0, "forward_sq": 0.75, "target": 0.75,
        "world": 7.0, "world_sq": 1.50, "heading_hold": 1.5,
        "lane": -0.60, "lateral": -0.60, "heading": -0.60,
    },
    "speed_retention": {
        "forward": 3.3, "forward_sq": 0.90, "target": 0.80,
        "world": 7.7, "world_sq": 1.80, "heading_hold": 1.20,
        "lane": -0.45, "lateral": -0.45, "heading": -0.45,
    },
    "line_hold": {
        "forward": 2.7, "forward_sq": 0.60, "target": 0.65,
        "world": 6.1, "world_sq": 1.20, "heading_hold": 2.2,
        "lane": -0.90, "lateral": -0.90, "heading": -0.90,
    },
}


def _recipe() -> tuple[str, dict[str, float]]:
    name = os.environ.get("DUCKLAB_OFFICIAL_RECIPE", "balanced")
    if name not in OFFICIAL_RECIPES:
        raise ValueError(
            f"Unknown DUCKLAB_OFFICIAL_RECIPE={name!r}; "
            f"choose one of {sorted(OFFICIAL_RECIPES)}"
        )
    return name, OFFICIAL_RECIPES[name]


def make_microduck_speed_official_adaptation_env_cfg(play: bool = False):
    """Robust speed training around official Race5 roller friction."""
    cfg = make_microduck_speed_straightening_env_cfg(play=play)
    _, recipe = _recipe()
    cfg.episode_length_s = 20.0
    cfg.events["official_band_wheel_friction"] = EventTermCfg(
        func=dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
            "operation": "abs",
            # Realistic variation around the benchmark value; 0.003 remains
            # the deterministic evaluation point.
            "ranges": (0.0025, 0.0035),
        },
    )
    command = cfg.commands["twist"]
    command.yaw_kp = 0.80
    command.lateral_kp = 0.16
    command.yaw_kd = 0.10
    command.max_correction = 0.22

    # Do not let a stable slow cruise dominate.  The original speed-discovery
    # terms remain continuous (not command-capped); world progress carries the
    # largest weight.  Direction costs are deliberately modest.
    cfg.rewards["forward_velocity_mps"].weight = recipe["forward"]
    cfg.rewards["forward_velocity_squared"].weight = recipe["forward_sq"]
    cfg.rewards["speed_target_progress"].weight = recipe["target"]
    cfg.rewards["world_forward_velocity_mps"].weight = recipe["world"]
    cfg.rewards["world_forward_velocity_squared"].weight = recipe["world_sq"]
    cfg.rewards["heading_hold"].weight = recipe["heading_hold"]
    cfg.rewards["lane_error"].weight = recipe["lane"]
    cfg.rewards["world_lateral_velocity"].weight = recipe["lateral"]
    cfg.rewards["heading_error"].weight = recipe["heading"]
    cfg.curriculum.clear()
    return cfg


MicroduckSpeedOfficialAdaptationRlCfg = dataclasses.replace(
    MicroduckSpeedStraighteningRlCfg,
    algorithm=dataclasses.replace(
        MicroduckSpeedStraighteningRlCfg.algorithm,
        learning_rate=float(os.environ.get("DUCKLAB_OFFICIAL_LR", "2e-6")),
        desired_kl=1.0e-4,
        clip_param=0.015,
        num_learning_epochs=2,
    ),
    experiment_name="microduck_speed_official_adaptation",
    run_name="microduck_speed_official_adaptation",
    save_interval=10,
)
