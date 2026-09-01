"""PPO with a frozen policy-action anchor for conservative sim transfer."""

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


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
        teacher_loss_coef: float = 0.20,
        teacher_loss_decay: float = 0.999,
        teacher_loss_floor: float = 0.01,
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
        self.teacher_loss_coef = teacher_loss_coef
        self.teacher_loss_decay = teacher_loss_decay
        self.teacher_loss_floor = teacher_loss_floor

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
                teacher_actions = self.teacher(batch.observations)
            teacher_loss = nn.functional.mse_loss(student_actions, teacher_actions)
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
