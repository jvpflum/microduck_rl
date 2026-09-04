"""Conservative PPO variant that keeps the actor close to its loaded donor."""

from __future__ import annotations

import copy
import os

import torch

from rsl_rl.algorithms import PPO


class DonorAnchoredPPO(PPO):
    """Apply cumulative damped PPO updates inside a donor-centered trust region.

    ``DUCKLAB_DONOR_UPDATE_RETENTION`` controls how much of each incremental PPO
    update is retained. ``DUCKLAB_DONOR_MAX_RELATIVE_DRIFT`` caps the total MLP
    parameter displacement relative to the actor loaded at startup.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.donor_update_retention = float(
            os.environ.get("DUCKLAB_DONOR_UPDATE_RETENTION", "0.05")
        )
        self.donor_max_relative_drift = float(
            os.environ.get("DUCKLAB_DONOR_MAX_RELATIVE_DRIFT", "0.0")
        )
        self.freeze_obs_normalizer = os.environ.get(
            "DUCKLAB_DONOR_FREEZE_OBS_NORMALIZER", "1"
        ).lower() not in {"0", "false", "no"}
        self.donor_max_action_rms = float(
            os.environ.get("DUCKLAB_DONOR_MAX_ACTION_RMS", "0.0")
        )
        self.donor_action_guard_samples = int(
            os.environ.get("DUCKLAB_DONOR_ACTION_GUARD_SAMPLES", "4096")
        )
        self.donor_fixed_action_guard = os.environ.get(
            "DUCKLAB_DONOR_FIXED_ACTION_GUARD", "0"
        ).lower() not in {"0", "false", "no"}
        self.donor_guard_high_speed_fraction = float(
            os.environ.get("DUCKLAB_DONOR_GUARD_HIGH_SPEED_FRACTION", "0.0")
        )
        self.donor_guard_high_speed_min = float(
            os.environ.get("DUCKLAB_DONOR_GUARD_HIGH_SPEED_MIN", "0.70")
        )
        self.donor_guard_command_index = int(
            os.environ.get("DUCKLAB_DONOR_GUARD_COMMAND_INDEX", "48")
        )
        if not 0.0 <= self.donor_update_retention <= 1.0:
            raise ValueError("DUCKLAB_DONOR_UPDATE_RETENTION must be in [0, 1]")
        if self.donor_max_relative_drift < 0.0:
            raise ValueError("DUCKLAB_DONOR_MAX_RELATIVE_DRIFT must be non-negative")
        if self.donor_max_action_rms < 0.0:
            raise ValueError("DUCKLAB_DONOR_MAX_ACTION_RMS must be non-negative")
        if self.donor_action_guard_samples < 1:
            raise ValueError("DUCKLAB_DONOR_ACTION_GUARD_SAMPLES must be positive")
        if not 0.0 <= self.donor_guard_high_speed_fraction <= 1.0:
            raise ValueError(
                "DUCKLAB_DONOR_GUARD_HIGH_SPEED_FRACTION must be in [0, 1]"
            )
        if self.donor_guard_command_index < 0:
            raise ValueError("DUCKLAB_DONOR_GUARD_COMMAND_INDEX must be non-negative")
        self._donor_actor: dict[str, torch.Tensor] | None = None
        self._donor_actor_model = None
        self._fixed_guard_obs = None
        self._fixed_guard_actions = None

    def load(self, loaded_dict, load_cfg, strict):
        result = super().load(loaded_dict, load_cfg, strict)
        self._donor_actor = {
            name: value.detach().clone()
            for name, value in self.actor.state_dict().items()
        }
        self._donor_actor_model = copy.deepcopy(self.actor).to(self.device).eval()
        for parameter in self._donor_actor_model.parameters():
            parameter.requires_grad_(False)
        print(
            "[DonorAnchoredPPO] actor anchor captured; "
            f"retaining {self.donor_update_retention:.3f} of each PPO displacement"
            f"; max relative drift {self.donor_max_relative_drift:.3e}"
            f"; max action RMS {self.donor_max_action_rms:.3e}"
            f"; fixed action guard {self.donor_fixed_action_guard}"
            f"; high-speed guard fraction {self.donor_guard_high_speed_fraction:.2f}"
            f"; freeze obs normalizer {self.freeze_obs_normalizer}"
        )
        return result

    def _sample_action_guard_observations(self):
        flat_obs = self.storage.observations.flatten(0, 1)
        population = flat_obs.batch_size[0]
        sample_count = min(self.donor_action_guard_samples, population)
        general_count = sample_count
        high_speed_indices = None
        selected_high_count = 0

        if self.donor_guard_high_speed_fraction > 0.0:
            try:
                actor_obs = flat_obs["actor"]
                if actor_obs.shape[-1] <= self.donor_guard_command_index:
                    raise ValueError(
                        "actor observation is too short for "
                        f"DUCKLAB_DONOR_GUARD_COMMAND_INDEX={self.donor_guard_command_index}"
                    )
                eligible = torch.nonzero(
                    actor_obs[:, self.donor_guard_command_index]
                    >= self.donor_guard_high_speed_min,
                    as_tuple=False,
                ).flatten()
                requested = int(round(sample_count * self.donor_guard_high_speed_fraction))
                high_count = min(requested, eligible.numel())
                if high_count > 0:
                    high_positions = torch.linspace(
                        0, eligible.numel() - 1, high_count, device=self.device
                    ).long()
                    high_speed_indices = eligible[high_positions]
                    selected_high_count = high_count
                    general_count = sample_count - high_count
            except KeyError:
                # Older storage layouts may not name the policy observation
                # group. The general reservoir remains a safe fallback.
                high_speed_indices = None

        parts = []
        if general_count > 0:
            parts.append(
                torch.linspace(
                    0, population - 1, general_count, device=self.device
                ).long()
            )
        if high_speed_indices is not None:
            parts.append(high_speed_indices)
        indices = torch.cat(parts) if len(parts) > 1 else parts[0]
        if self.donor_fixed_action_guard and self._fixed_guard_obs is None:
            print(
                "[DonorAnchoredPPO] fixed action reservoir captured; "
                f"samples {indices.numel()}; high-speed samples {selected_high_count}; "
                f"high-speed threshold {self.donor_guard_high_speed_min:.3f}"
            )
        return flat_obs[indices].detach().clone()

    def update(self):
        reference_obs = None
        donor_actions = None
        if self.donor_max_action_rms > 0.0 and self._donor_actor_model is not None:
            if self.donor_fixed_action_guard and self._fixed_guard_obs is not None:
                reference_obs = self._fixed_guard_obs
                donor_actions = self._fixed_guard_actions
            else:
                reference_obs = self._sample_action_guard_observations()
                with torch.inference_mode():
                    donor_actions = self._donor_actor_model(reference_obs).detach().clone()
                if self.donor_fixed_action_guard:
                    self._fixed_guard_obs = reference_obs
                    self._fixed_guard_actions = donor_actions

        previous_actor = {
            name: value.detach().clone()
            for name, value in self.actor.state_dict().items()
        }
        losses = super().update()
        if self._donor_actor is None:
            self._donor_actor = {
                name: value.detach().clone()
                for name, value in self.actor.state_dict().items()
            }
            return losses

        retention = self.donor_update_retention
        with torch.no_grad():
            # Dampen this update relative to the immediately preceding actor.
            # Unlike fixed-anchor interpolation, changes accumulate over time.
            for name, value in self.actor.state_dict().items():
                if torch.is_floating_point(value):
                    if self.freeze_obs_normalizer and name.startswith("obs_normalizer."):
                        value.copy_(self._donor_actor[name])
                    else:
                        previous = previous_actor[name]
                        value.copy_(previous + retention * (value - previous))

            relative_drift = 0.0
            if self.donor_max_relative_drift > 0.0:
                donor_norm_sq = torch.zeros((), device=self.device)
                delta_norm_sq = torch.zeros((), device=self.device)
                for name, value in self.actor.state_dict().items():
                    if torch.is_floating_point(value) and name.startswith("mlp."):
                        anchor = self._donor_actor[name]
                        donor_norm_sq.add_(torch.sum(anchor * anchor))
                        delta = value - anchor
                        delta_norm_sq.add_(torch.sum(delta * delta))
                donor_norm = torch.sqrt(donor_norm_sq)
                delta_norm = torch.sqrt(delta_norm_sq)
                relative_drift = (delta_norm / donor_norm.clamp_min(1e-12)).item()
                if relative_drift > self.donor_max_relative_drift:
                    scale = self.donor_max_relative_drift / relative_drift
                    for name, value in self.actor.state_dict().items():
                        if torch.is_floating_point(value) and name.startswith("mlp."):
                            anchor = self._donor_actor[name]
                            value.copy_(anchor + scale * (value - anchor))
                    relative_drift = self.donor_max_relative_drift

            action_rms = 0.0
            action_scale = 1.0
            if reference_obs is not None and donor_actions is not None:
                candidate = {
                    name: value.detach().clone()
                    for name, value in self.actor.state_dict().items()
                    if torch.is_floating_point(value) and name.startswith("mlp.")
                }

                def set_actor_scale(scale: float) -> None:
                    for name, value in self.actor.state_dict().items():
                        if name in candidate:
                            anchor = self._donor_actor[name]
                            value.copy_(anchor + scale * (candidate[name] - anchor))

                def measure_action_rms() -> float:
                    actions = self.actor(reference_obs)
                    return torch.sqrt(torch.mean((actions - donor_actions).square())).item()

                action_rms = measure_action_rms()
                if action_rms > self.donor_max_action_rms:
                    low, high = 0.0, 1.0
                    for _ in range(12):
                        middle = 0.5 * (low + high)
                        set_actor_scale(middle)
                        if measure_action_rms() <= self.donor_max_action_rms:
                            low = middle
                        else:
                            high = middle
                    action_scale = low
                    set_actor_scale(action_scale)
                    action_rms = measure_action_rms()
        losses["donor_retention"] = retention
        losses["donor_relative_drift"] = relative_drift
        if reference_obs is not None:
            losses["donor_action_rms"] = action_rms
            losses["donor_action_scale"] = action_scale
        return losses
