"""Frozen-donor residual actor and PPO loader for speed-frontier training.

The donor network remains the deployable fallback throughout training.  A
zero-initialized, bounded residual network learns only the state-dependent
correction needed for official wheel drag, heading recovery, and higher-speed
balance.  Unlike weight interpolation, this architecture cannot silently erase
the donor gait during the first PPO updates.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules import HiddenState


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


def _remap_residual_as_frozen_base(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Move a legacy residual branch into a stacked actor's frozen branch."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key == "_residual_max_action":
            remapped["_frozen_residual_max_action"] = value
        elif key.startswith("residual_mlp."):
            remapped[f"frozen_residual_mlp.{key.removeprefix('residual_mlp.')}"] = value
        else:
            remapped[key] = value
    return remapped


def _parse_trainable_action_rows(
    value: str | None, output_dim: int
) -> tuple[int, ...] | None:
    """Parse an optional comma-separated residual output-row allowlist."""
    if value is None or not value.strip():
        return None
    try:
        rows = tuple(sorted({int(part.strip()) for part in value.split(",")}))
    except ValueError as exc:
        raise ValueError(
            "DUCKLAB_RESIDUAL_TRAINABLE_ACTION_ROWS must contain integers"
        ) from exc
    if not rows or rows[0] < 0 or rows[-1] >= output_dim:
        raise ValueError(
            "DUCKLAB_RESIDUAL_TRAINABLE_ACTION_ROWS must select rows in "
            f"[0, {output_dim - 1}]"
        )
    return rows


def _configure_surgical_output_training(
    residual_mlp: nn.Module,
    output_dim: int,
    rows: tuple[int, ...] | None,
) -> None:
    """Freeze residual features and mask the final layer to selected actions."""
    if rows is None:
        return
    for parameter in residual_mlp.parameters():
        parameter.requires_grad_(False)
    final_linear = next(
        module
        for module in reversed(list(residual_mlp.modules()))
        if isinstance(module, nn.Linear)
    )
    final_linear.weight.requires_grad_(True)
    final_linear.bias.requires_grad_(True)
    row_mask = torch.zeros(output_dim, dtype=final_linear.weight.dtype)
    row_mask[list(rows)] = 1.0
    final_linear.register_buffer(
        "_ducklab_trainable_row_mask", row_mask, persistent=False
    )
    final_linear.weight.register_hook(
        lambda grad: grad * final_linear._ducklab_trainable_row_mask[:, None]
    )
    final_linear.bias.register_hook(
        lambda grad: grad * final_linear._ducklab_trainable_row_mask
    )


class ResidualMLPModel(MLPModel):
    """An immutable donor MLP plus a bounded, trainable neural residual.

    With ``DUCKLAB_RESIDUAL_STACK_ON_RESUME=1``, a legacy residual checkpoint
    is loaded as an immutable donor+residual base and ``residual_mlp`` becomes a
    second, zero-output adapter.  This makes iteration zero exactly reproduce
    the proven policy while giving PPO a deliberately small correction channel.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        residual_dims = tuple(
            int(value)
            for value in os.environ.get(
                "DUCKLAB_RESIDUAL_HIDDEN_DIMS", "256,128"
            ).split(",")
            if value.strip()
        )
        if not residual_dims:
            raise ValueError("DUCKLAB_RESIDUAL_HIDDEN_DIMS must not be empty")
        residual_max_action = float(
            os.environ.get("DUCKLAB_RESIDUAL_MAX_ACTION", "0.18")
        )
        if not 0.0 < residual_max_action <= 1.0:
            raise ValueError("DUCKLAB_RESIDUAL_MAX_ACTION must be in (0, 1]")
        # Keep the residual authority in the checkpoint so deterministic
        # evaluation and export cannot silently use a different scale.
        self.register_buffer(
            "_residual_max_action",
            torch.tensor(residual_max_action, dtype=torch.float32),
        )

        self.stacked_adapter = _env_flag(
            "DUCKLAB_RESIDUAL_STACK_ON_RESUME", default=False
        )

        self.residual_mlp = MLP(
            self.obs_dim,
            output_dim,
            residual_dims,
            activation,
        )
        # Only the output layer is zeroed.  This produces exactly the donor at
        # initialization while allowing useful gradients into the output layer
        # on the first PPO update and into the features thereafter.
        final_linear = next(
            module
            for module in reversed(list(self.residual_mlp.modules()))
            if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        if self.stacked_adapter:
            self.frozen_residual_mlp = MLP(
                self.obs_dim,
                output_dim,
                residual_dims,
                activation,
            )
            self.register_buffer(
                "_frozen_residual_max_action",
                torch.tensor(0.0, dtype=torch.float32),
            )
        self.trainable_action_rows = _parse_trainable_action_rows(
            os.environ.get("DUCKLAB_RESIDUAL_TRAINABLE_ACTION_ROWS"), output_dim
        )
        _configure_surgical_output_training(
            self.residual_mlp, output_dim, self.trainable_action_rows
        )
        self.freeze_donor()

    def freeze_donor(self) -> None:
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)
        for parameter in self.obs_normalizer.parameters():
            parameter.requires_grad_(False)
        if self.stacked_adapter:
            for parameter in self.frozen_residual_mlp.parameters():
                parameter.requires_grad_(False)

    def load_donor_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        result = super().load_state_dict(state_dict, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [
            key
            for key in result.missing_keys
            if not key.startswith("residual_mlp.")
            and key != "_residual_max_action"
        ]
        if unexpected or missing:
            raise RuntimeError(
                "Donor checkpoint is incompatible with residual actor: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.freeze_donor()

    def load_residual_as_frozen_base(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        """Freeze a legacy residual checkpoint and leave the new adapter zero."""
        if not self.stacked_adapter:
            raise RuntimeError(
                "DUCKLAB_RESIDUAL_STACK_ON_RESUME is required for stacked loading"
            )
        if any(key.startswith("frozen_residual_mlp.") for key in state_dict):
            raise RuntimeError("Checkpoint is already a stacked residual actor")
        remapped = _remap_residual_as_frozen_base(state_dict)
        result = self.load_state_dict(remapped, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [
            key
            for key in result.missing_keys
            if not key.startswith("residual_mlp.")
            and key != "_residual_max_action"
        ]
        if unexpected or missing:
            raise RuntimeError(
                "Residual checkpoint is incompatible with stacked actor: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.freeze_donor()

    def _combined_output(self, latent: torch.Tensor) -> torch.Tensor:
        donor = self.mlp(latent)
        if self.stacked_adapter:
            donor = donor + self._frozen_residual_max_action * torch.tanh(
                self.frozen_residual_mlp(latent)
            )
        residual = self._residual_max_action * torch.tanh(
            self.residual_mlp(latent)
        )
        return donor + residual

    @property
    def residual_max_action(self) -> float:
        return float(self._residual_max_action.item())

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self._combined_output(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def update_normalization(self, obs: TensorDict) -> None:
        # The donor's normalization is part of the behavior being preserved.
        return None

    def as_jit(self) -> nn.Module:
        return _TorchResidualModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxResidualModel(self, verbose)


class _TorchResidualModel(nn.Module):
    def __init__(self, model: ResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.residual_mlp = copy.deepcopy(model.residual_mlp)
        # Keep a shape-compatible zero branch for legacy actors so the exported
        # module has one stable forward path in both modes.
        self.frozen_residual_mlp = copy.deepcopy(
            model.frozen_residual_mlp
            if model.stacked_adapter
            else model.residual_mlp
        )
        self.register_buffer(
            "residual_max_action",
            model._residual_max_action.detach().clone(),
        )
        self.register_buffer(
            "frozen_residual_max_action",
            (
                model._frozen_residual_max_action.detach().clone()
                if model.stacked_adapter
                else torch.tensor(0.0, dtype=torch.float32)
            ),
        )
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = (
            self.mlp(x)
            + self.frozen_residual_max_action
            * torch.tanh(self.frozen_residual_mlp(x))
            + self.residual_max_action
            * torch.tanh(self.residual_mlp(x))
        )
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxResidualModel(_TorchResidualModel):
    is_recurrent = False

    def __init__(self, model: ResidualMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class ResidualPPO(PPO):
    """PPO that can warm-start a residual actor from a conventional checkpoint."""

    def load(self, loaded_dict, load_cfg, strict):
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
        actor_state = loaded_dict["actor_state_dict"]
        residual_checkpoint = any(
            key.startswith("residual_mlp.") for key in actor_state
        )
        stacked_checkpoint = any(
            key.startswith("frozen_residual_mlp.") for key in actor_state
        )
        stacked_warmstart = bool(
            residual_checkpoint
            and not stacked_checkpoint
            and isinstance(self.actor, ResidualMLPModel)
            and self.actor.stacked_adapter
        )

        if load_cfg.get("actor"):
            if not isinstance(self.actor, ResidualMLPModel):
                raise TypeError("ResidualPPO requires ResidualMLPModel as actor")
            if stacked_checkpoint:
                self.actor.load_state_dict(actor_state, strict=strict)
                self.actor.freeze_donor()
            elif stacked_warmstart:
                self.actor.load_residual_as_frozen_base(actor_state)
            elif residual_checkpoint:
                self.actor.load_state_dict(actor_state, strict=strict)
                self.actor.freeze_donor()
            else:
                self.actor.load_donor_state_dict(actor_state)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(
                loaded_dict["critic_state_dict"], strict=strict
            )
        surgical_resume = bool(
            (stacked_checkpoint or (residual_checkpoint and not stacked_warmstart))
            and self.actor.trainable_action_rows is not None
        )
        compatible_resume = bool(
            stacked_checkpoint or (residual_checkpoint and not stacked_warmstart)
        )
        if load_cfg.get("optimizer") and compatible_resume and not surgical_resume:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd and compatible_resume:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(
                loaded_dict["rnd_optimizer_state_dict"]
            )

        if stacked_checkpoint:
            kind = "stacked residual resume"
        elif stacked_warmstart:
            kind = "frozen residual-base warm-start"
        elif residual_checkpoint:
            kind = "residual resume"
        else:
            kind = "frozen donor warm-start"
        print(
            f"[ResidualPPO] {kind}; max residual action "
            f"{self.actor.residual_max_action:.3f}"
        )
        if surgical_resume:
            print(
                "[ResidualPPO] surgical output-row training with fresh optimizer; "
                f"rows={self.actor.trainable_action_rows}"
            )
        # A conventional donor never carries compatible residual optimizer or
        # iteration state, so its frontier run intentionally begins at zero.
        return bool(load_cfg.get("iteration", False) and compatible_resume)

    def update(self):
        losses = super().update()
        if isinstance(self.actor, ResidualMLPModel):
            with torch.no_grad():
                distribution = self.actor.distribution
                min_std = float(os.environ.get("DUCKLAB_RESIDUAL_MIN_STD", "0.015"))
                max_std = float(os.environ.get("DUCKLAB_RESIDUAL_MAX_STD", "0.20"))
                if hasattr(distribution, "std_param"):
                    distribution.std_param.clamp_(min_std, max_std)
                output_layer = next(
                    module
                    for module in reversed(
                        list(self.actor.residual_mlp.modules())
                    )
                    if isinstance(module, nn.Linear)
                )
                output_sq = (
                    output_layer.weight.square().sum()
                    + output_layer.bias.square().sum()
                )
                losses["residual_output_layer_norm"] = torch.sqrt(
                    output_sq
                ).item()
        return losses
