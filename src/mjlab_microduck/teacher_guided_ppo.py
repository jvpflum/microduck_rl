"""PPO with a frozen policy-action anchor for conservative sim transfer."""

from __future__ import annotations

import copy
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import load as load_onnx
from onnx import numpy_helper
from rsl_rl.algorithms import PPO


class DuckLabOnnxSpeedTeacher(nn.Module):
    """Torch replay of DuckLab's distilled donor-residual ONNX policy.

    The 5090 review bundle intentionally contains no optimizer checkpoint.  Its
    V47 policy is nevertheless a small, deterministic MLP graph, so loading the
    graph weights into frozen Torch buffers lets thousands of GPU environments
    use it as a teacher without a CPU ONNX Runtime round trip.
    """

    def __init__(self, policy_path: str) -> None:
        super().__init__()
        model = load_onnx(Path(policy_path))
        weights = {
            item.name: torch.from_numpy(numpy_helper.to_array(item).copy())
            for item in model.graph.initializer
        }
        required = {
            "observation_mean",
            "observation_std",
            "actor.base_actor.donor.0.weight",
            "actor.base_actor.residual.0.weight",
            "actor.dagger_residual.0.weight",
            "ducklab_action_multiplier",
            "ducklab_action_offset",
        }
        missing = sorted(required - weights.keys())
        if missing:
            raise ValueError(
                f"Unsupported DuckLab speed-teacher graph; missing {missing}"
            )
        for name, value in weights.items():
            self.register_buffer(name.replace(".", "__"), value)

    def _weight(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "__"))

    def _mlp(self, observations: torch.Tensor, prefix: str, layers: tuple[int, ...]) -> torch.Tensor:
        output = observations
        for index, layer in enumerate(layers):
            output = F.linear(
                output,
                self._weight(f"{prefix}.{layer}.weight"),
                self._weight(f"{prefix}.{layer}.bias"),
            )
            if index + 1 < len(layers):
                output = F.elu(output)
        return output

    def forward(self, observations) -> torch.Tensor:
        actor_observations = (
            observations if isinstance(observations, torch.Tensor) else observations["actor"]
        )
        normalized = (
            actor_observations - self._weight("observation_mean")
        ) / self._weight("observation_std")
        donor = self._mlp(
            normalized, "actor.base_actor.donor", (0, 2, 4, 6)
        )
        residual = 0.08 * torch.tanh(
            self._mlp(normalized, "actor.base_actor.residual", (0, 2, 4))
        )
        dagger = 0.20 * torch.tanh(
            self._mlp(normalized, "actor.dagger_residual", (0, 2, 4))
        )
        # The exported graph reconstructs raw command_x from its normalized
        # input, then applies a smooth gate over 0.70--0.75 m/s.
        gate = torch.clamp(
            (actor_observations[:, 48:49] - 0.70) / 0.05,
            min=0.0,
            max=1.0,
        )
        gate = gate.square() * (3.0 - 2.0 * gate)
        actions = donor + residual + gate * dagger
        return (
            actions * self._weight("ducklab_action_multiplier")
            + self._weight("ducklab_action_offset")
        )


class TeacherGuidedPPO(PPO):
    """Keep a PPO actor near a frozen donor while it adapts to changed physics.

    The teacher is never used by the acting policy or exporter.  It only adds a
    decaying MSE term during updates, so the saved/exported actor is a normal
    single-policy MicroDuck ONNX.
    """

    def __init__(
        self,
        *args,
        teacher_checkpoint: str,
        speed_teacher_checkpoint: str | None = None,
        speed_teacher_onnx: str | None = None,
        teacher_loss_coef: float = 0.20,
        teacher_loss_decay: float = 0.999,
        teacher_loss_floor: float = 0.01,
        probe_loss_share: float = 0.0,
        speed_command_threshold: float = 0.5,
        smooth_turn_start: float = 0.02,
        smooth_turn_end: float = 0.12,
        command_x_index: int = 48,
        command_y_index: int = 49,
        command_yaw_index: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("TeacherGuidedPPO requires RND and symmetry to be disabled.")
        checkpoint = torch.load(teacher_checkpoint, weights_only=False, map_location=self.device)
        self.teacher = copy.deepcopy(self.actor).eval()
        self.teacher.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        if speed_teacher_checkpoint is not None and speed_teacher_onnx is not None:
            raise ValueError("Configure only one speed teacher source.")
        self.speed_teacher = None
        if speed_teacher_checkpoint is not None:
            speed_checkpoint = torch.load(
                speed_teacher_checkpoint,
                weights_only=False,
                map_location=self.device,
            )
            self.speed_teacher = copy.deepcopy(self.actor).eval()
            self.speed_teacher.load_state_dict(
                speed_checkpoint["actor_state_dict"], strict=True
            )
            for parameter in self.speed_teacher.parameters():
                parameter.requires_grad_(False)
        elif speed_teacher_onnx is not None:
            self.speed_teacher = DuckLabOnnxSpeedTeacher(speed_teacher_onnx).to(
                self.device
            )
            self.speed_teacher.eval()
            for parameter in self.speed_teacher.parameters():
                parameter.requires_grad_(False)
        if not 0.0 <= probe_loss_share <= 1.0:
            raise ValueError("probe_loss_share must be between zero and one.")
        if not 0.0 <= smooth_turn_start < smooth_turn_end:
            raise ValueError("smooth turn routing requires 0 <= start < end.")
        self.teacher_loss_coef = teacher_loss_coef
        self.teacher_loss_decay = teacher_loss_decay
        self.teacher_loss_floor = teacher_loss_floor
        self.probe_loss_share = probe_loss_share
        self.speed_command_threshold = speed_command_threshold
        self.smooth_turn_start = smooth_turn_start
        self.smooth_turn_end = smooth_turn_end
        self.command_x_index = command_x_index
        self.command_y_index = command_y_index
        self.command_yaw_index = command_yaw_index

    def _teacher_actions(self, observations: torch.Tensor) -> torch.Tensor:
        control_actions = self.teacher(observations)
        if self.speed_teacher is None:
            return control_actions
        speed_actions = self.speed_teacher(observations)
        actor_observations = observations["actor"]
        command_x = actor_observations[
            :, self.command_x_index : self.command_x_index + 1
        ]
        command_yaw = actor_observations[
            :, self.command_yaw_index : self.command_yaw_index + 1
        ].abs()
        speed_command = (command_x > self.speed_command_threshold).to(
            dtype=actor_observations.dtype
        )
        turn_gate = torch.clamp(
            (self.smooth_turn_end - command_yaw)
            / (self.smooth_turn_end - self.smooth_turn_start),
            min=0.0,
            max=1.0,
        )
        speed_weight = speed_command * turn_gate
        return control_actions + speed_weight * (speed_actions - control_actions)

    def _probe_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Cover commands absent from a straight race so they cannot regress."""
        probes = observations.clone()
        actor_probes = probes["actor"]
        commands = actor_probes.new_tensor(
            (
                (-0.40, 0.00, 0.00),
                (0.00, 0.00, 0.00),
                (0.30, 0.00, 0.25),
                (0.30, 0.00, -0.25),
                (0.80, 0.00, 0.30),
                (0.80, 0.00, -0.30),
                (0.80, 0.00, 0.00),
            )
        )
        indices = torch.arange(
            actor_probes.shape[0], device=actor_probes.device
        ) % len(commands)
        selected = commands[indices]
        actor_probes[:, self.command_x_index] = selected[:, 0]
        actor_probes[:, self.command_y_index] = selected[:, 1]
        actor_probes[:, self.command_yaw_index] = selected[:, 2]
        probes["actor"] = actor_probes
        return probes

    def update(self) -> dict[str, float]:
        mean_value_loss = mean_surrogate_loss = mean_entropy = mean_teacher_loss = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, self.actor.output_distribution_params)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-7, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate_loss = torch.max(
                -torch.squeeze(batch.advantages) * ratio,
                -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
            ).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            student_actions = self.actor(batch.observations)
            with torch.no_grad():
                teacher_actions = self._teacher_actions(batch.observations)
            teacher_loss = nn.functional.mse_loss(student_actions, teacher_actions)
            if self.speed_teacher is not None and self.probe_loss_share > 0.0:
                probe_observations = self._probe_observations(batch.observations)
                probe_student_actions = self.actor(probe_observations)
                with torch.no_grad():
                    probe_teacher_actions = self._teacher_actions(probe_observations)
                probe_loss = nn.functional.mse_loss(
                    probe_student_actions, probe_teacher_actions
                )
                teacher_loss = (
                    (1.0 - self.probe_loss_share) * teacher_loss
                    + self.probe_loss_share * probe_loss
                )
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            loss = loss + self.teacher_loss_coef * teacher_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_teacher_loss += teacher_loss.item()

        updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.teacher_loss_coef = max(self.teacher_loss_floor, self.teacher_loss_coef * self.teacher_loss_decay)
        return {
            "value": mean_value_loss / updates,
            "surrogate": mean_surrogate_loss / updates,
            "entropy": mean_entropy / updates,
            "teacher": mean_teacher_loss / updates,
        }
