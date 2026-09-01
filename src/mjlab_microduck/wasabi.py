"""Small WASABI-style adversarial motion prior for RSL-RL PPO.

The discriminator learns from two-frame, root-position-free state transitions.
It is intentionally independent of demonstration actions: the arena clip used
an external mouse force, so those actions are not valid behavior-cloning labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from mjlab.rl import RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO


@dataclass
class WasabiPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict | None = None
    expert_transitions: list[list[list[float]]] = field(default_factory=list)
    style_reward_coef: float = 0.05
    task_reward_lerp: float = 0.8
    discriminator_hidden_dims: tuple[int, ...] = (256, 128)
    discriminator_learning_rate: float = 2.5e-5
    discriminator_updates: int = 4
    discriminator_batch_size: int = 2048
    discriminator_gradient_penalty: float = 5.0
    discriminator_warmup_updates: int = 5
    expert_corruption: float = 0.03


class WasabiPPO(PPO):
    """PPO with a WASABI rough-demonstration transition reward."""

    def __init__(
        self,
        *args,
        expert_transitions: list[list[list[float]]],
        style_reward_coef: float = 0.05,
        task_reward_lerp: float = 0.8,
        discriminator_hidden_dims: tuple[int, ...] = (256, 128),
        discriminator_learning_rate: float = 2.5e-5,
        discriminator_updates: int = 4,
        discriminator_batch_size: int = 2048,
        discriminator_gradient_penalty: float = 5.0,
        discriminator_warmup_updates: int = 5,
        expert_corruption: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        expert = torch.as_tensor(expert_transitions, dtype=torch.float32, device=self.device)
        if expert.ndim != 3 or expert.shape[1] != 2:
            raise ValueError(f"expert_transitions must have shape [N, 2, D], got {tuple(expert.shape)}")
        self.style_dim = expert.shape[-1]
        self.style_mean = torch.zeros(self.style_dim, device=self.device)
        self.style_var = torch.ones(self.style_dim, device=self.device)
        self.style_count = 1.0e-2
        self.expert_transitions = expert.flatten(1)
        input_dim = self.expert_transitions.shape[-1]
        layers: list[nn.Module] = []
        previous = input_dim
        for width in discriminator_hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        # WASABI's least-squares discriminator uses an unconstrained output.
        # A tanh here saturated on the first failed probe and made every poor
        # policy transition receive exactly the same zero style reward.
        layers.append(nn.Linear(previous, 1))
        self.discriminator = nn.Sequential(*layers).to(self.device)
        self.discriminator_optimizer = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=discriminator_learning_rate,
            momentum=0.9,
            weight_decay=1.0e-3,
        )
        self.style_reward_coef = style_reward_coef
        self.task_reward_lerp = task_reward_lerp
        self.discriminator_updates = discriminator_updates
        self.discriminator_batch_size = discriminator_batch_size
        self.discriminator_gradient_penalty = discriminator_gradient_penalty
        self.discriminator_warmup_updates = discriminator_warmup_updates
        self.expert_corruption = expert_corruption
        self._update_count = 0
        self._previous_style: torch.Tensor | None = None
        self._policy_transitions: list[torch.Tensor] = []

    def _normalize(self, transitions: torch.Tensor) -> torch.Tensor:
        frames = transitions.reshape(-1, 2, self.style_dim)
        std = torch.sqrt(self.style_var + 1.0e-2)
        return torch.clamp((frames - self.style_mean) / std, -10.0, 10.0).flatten(1)

    @torch.no_grad()
    def _update_style_statistics(self, policy: torch.Tensor, expert: torch.Tensor) -> None:
        """Update one shared normalizer from equal policy/expert batches.

        Expert-only variance made near-constant demonstration coordinates have
        millimetric scales, allowing the discriminator to separate resets from
        the clip immediately.  WASABI instead updates a shared online state
        normalizer with both distributions.
        """
        values = torch.cat((policy, expert), dim=0).reshape(-1, self.style_dim)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = float(values.shape[0])
        delta = batch_mean - self.style_mean
        total = self.style_count + batch_count
        new_mean = self.style_mean + delta * (batch_count / total)
        m_a = self.style_var * self.style_count
        m_b = batch_var * batch_count
        correction = delta.pow(2) * (self.style_count * batch_count / total)
        self.style_mean = new_mean
        self.style_var = (m_a + m_b + correction) / total
        self.style_count = total

    def act(self, obs):
        self._previous_style = obs["style"].detach()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        if self._previous_style is None:
            raise RuntimeError("WasabiPPO.process_env_step called before act")
        current_style = obs["style"].detach()
        transition = torch.stack((self._previous_style, current_style), dim=1).flatten(1)
        valid = ~dones.bool()
        if valid.any():
            self._policy_transitions.append(transition[valid].detach())
        if self._update_count >= self.discriminator_warmup_updates:
            with torch.no_grad():
                policy_score = self.discriminator(self._normalize(transition)).squeeze(-1)
                # Least-squares mapping supported by WASABI.  Bounded scores
                # avoid the runaway Wasserstein gap exposed by probe seed 11.
                style_reward = self.style_reward_coef * torch.clamp(
                    1.0 - 0.25 * (policy_score - 1.0).pow(2), min=0.0
                )
                rewards = self.task_reward_lerp * rewards + (1.0 - self.task_reward_lerp) * style_reward
        super().process_env_step(obs, rewards, dones, extras)

    def _update_discriminator(self) -> dict[str, float]:
        if not self._policy_transitions:
            return {"wasabi": 0.0, "wasabi_gp": 0.0, "wasabi_expert": 0.0, "wasabi_policy": 0.0}
        policy = torch.cat(self._policy_transitions, dim=0)
        self._policy_transitions.clear()
        totals = torch.zeros(4, device=self.device)
        batch_size = min(self.discriminator_batch_size, policy.shape[0])
        for _ in range(self.discriminator_updates):
            policy_batch = policy[torch.randint(policy.shape[0], (batch_size,), device=self.device)]
            expert_batch = self.expert_transitions[
                torch.randint(self.expert_transitions.shape[0], (batch_size,), device=self.device)
            ].clone()
            if self.expert_corruption > 0.0:
                expert_batch += torch.randn_like(expert_batch) * self.expert_corruption
            self._update_style_statistics(policy_batch, expert_batch)
            policy_score = self.discriminator(self._normalize(policy_batch))
            expert_batch.requires_grad_(True)
            normalized_expert = self._normalize(expert_batch)
            expert_score = self.discriminator(normalized_expert)
            gradient = torch.autograd.grad(
                expert_score.sum(), normalized_expert, create_graph=True, retain_graph=True
            )[0]
            gradient_penalty = self.discriminator_gradient_penalty * gradient.pow(2).sum(dim=1).mean()
            adversarial = 0.5 * (
                (expert_score - 1.0).pow(2).mean()
                + (policy_score + 1.0).pow(2).mean()
            )
            loss = adversarial + gradient_penalty
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()
            totals += torch.stack((loss.detach(), gradient_penalty.detach(), expert_score.mean().detach(), policy_score.mean().detach()))
        totals /= max(1, self.discriminator_updates)
        return {
            "wasabi": float(totals[0]),
            "wasabi_gp": float(totals[1]),
            "wasabi_expert": float(totals[2]),
            "wasabi_policy": float(totals[3]),
        }

    def update(self) -> dict[str, float]:
        losses = super().update()
        losses.update(self._update_discriminator())
        self._update_count += 1
        return losses

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        state = super().save()
        state.update({
            "discriminator_state_dict": self.discriminator.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "wasabi_update_count": self._update_count,
            "wasabi_style_mean": self.style_mean,
            "wasabi_style_var": self.style_var,
            "wasabi_style_count": self.style_count,
        })
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
            self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
            self._update_count = int(loaded_dict.get("wasabi_update_count", 0))
            self.style_mean = loaded_dict.get("wasabi_style_mean", self.style_mean).to(self.device)
            self.style_var = loaded_dict.get("wasabi_style_var", self.style_var).to(self.device)
            self.style_count = float(loaded_dict.get("wasabi_style_count", self.style_count))
        return load_iteration
