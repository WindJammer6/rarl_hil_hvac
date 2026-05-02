import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

class ReplayMemory:
    """Fixed-size circular replay buffer."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden[0]),
        nn.ReLU(),
        nn.Linear(hidden[0], hidden[1]),
        nn.ReLU(),
        nn.Linear(hidden[1], out_dim),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAC  (Soft Actor-Critic – discrete version)
# ─────────────────────────────────────────────────────────────────────────────

class _SoftQNetwork(nn.Module):
    """Q(s, ·) → (batch, n_actions)."""
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = _mlp(n_obs, [256, 256], n_actions)

    def forward(self, x):
        return self.net(x)


class _SACActor(nn.Module):
    """π(·|s) logits over discrete actions."""
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = _mlp(n_obs, [256, 128], n_actions)

    def forward(self, x):
        return self.net(x)

    def get_probs_and_log_probs(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class SACAgent:
    """
    Soft Actor-Critic for discrete action spaces.

    Fixes relative to the original version:
    - safe handling when a whole batch is terminal
    - cleaner target-value computation without fake forward passes
    - corrected alpha-loss sign for entropy tuning
    """

    def __init__(
        self,
        n_observations, n_actions, device,
        lr=3e-4, gamma=0.99, tau=0.005,
        alpha=0.2, auto_alpha=True, target_entropy=None,
        memory_size=10_000, batch_size=64,
    ):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha

        self.actor = _SACActor(n_observations, n_actions).to(device)
        self.q1 = _SoftQNetwork(n_observations, n_actions).to(device)
        self.q2 = _SoftQNetwork(n_observations, n_actions).to(device)
        self.q1_tgt = _SoftQNetwork(n_observations, n_actions).to(device)
        self.q2_tgt = _SoftQNetwork(n_observations, n_actions).to(device)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())
        self.q1_tgt.eval()
        self.q2_tgt.eval()

        self.policy_net = self.actor

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        if target_entropy is None:
            target_entropy = 0.5 * math.log(float(n_actions))
        self.target_entropy = float(target_entropy)

        self.log_alpha = torch.tensor(
            math.log(alpha), dtype=torch.float32,
            requires_grad=auto_alpha, device=device,
        )
        if auto_alpha:
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = float(alpha)

        self.memory = ReplayMemory(memory_size)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs, _ = self.actor.get_probs_and_log_probs(state)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        return action.view(1, 1)

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device, dtype=torch.bool,
        )
        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device).float()
        else:
            non_final_next_states = torch.empty((0, self.n_observations), device=self.device, dtype=torch.float32)

        states = torch.cat(batch.state).to(self.device).float()
        actions = torch.cat(batch.action).view(-1, 1).to(self.device)
        rewards = torch.cat(batch.reward).view(-1).to(self.device).float()

        alpha = self.log_alpha.exp().detach()

        # ---- Critic targets ----
        with torch.no_grad():
            v_next = torch.zeros(self.batch_size, device=self.device)
            if non_final_mask.any():
                next_probs, next_log_probs = self.actor.get_probs_and_log_probs(non_final_next_states)
                q1_next = self.q1_tgt(non_final_next_states)
                q2_next = self.q2_tgt(non_final_next_states)
                min_q_next = torch.min(q1_next, q2_next)
                v_next_non_final = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=1)
                v_next[non_final_mask] = v_next_non_final

            target_q = rewards + self.gamma * v_next

        q1_pred = self.q1(states).gather(1, actions).squeeze(1)
        q2_pred = self.q2(states).gather(1, actions).squeeze(1)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optim.step()

        # ---- Actor update ----
        probs, log_probs = self.actor.get_probs_and_log_probs(states)
        with torch.no_grad():
            min_q = torch.min(self.q1(states), self.q2(states))

        actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ---- Temperature update ----
        if self.auto_alpha:
            with torch.no_grad():
                probs_detached, log_probs_detached = self.actor.get_probs_and_log_probs(states)
                entropy = -(probs_detached * log_probs_detached).sum(dim=1).mean()

            alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach())
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha.item()

        self._soft_update(self.q1, self.q1_tgt)
        self._soft_update(self.q2, self.q2_tgt)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        if "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
            if "q1" in checkpoint:
                self.q1.load_state_dict(checkpoint["q1"])
                self.q2.load_state_dict(checkpoint["q2"])
                self.q1_tgt.load_state_dict(checkpoint["q1"])
                self.q2_tgt.load_state_dict(checkpoint["q2"])
        else:
            self.actor.load_state_dict(checkpoint)
        print(f"SACAgent: model loaded from {model_path}")

    def save_model(self, path: str):
        torch.save(
            {"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict()},
            path,
        )

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
