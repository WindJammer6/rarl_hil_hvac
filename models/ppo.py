import math
import random
from collections import deque, namedtuple

import numpy as np
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
    """Build a 3-layer MLP matching the DQN hidden-layer sizes."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden[0]),
        nn.ReLU(),
        nn.Linear(hidden[0], hidden[1]),
        nn.ReLU(),
        nn.Linear(hidden[1], out_dim),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. PPO  (Proximal Policy Optimisation – discrete, actor-critic, GAE)
# ─────────────────────────────────────────────────────────────────────────────

class _PPOActor(nn.Module):
    """
    Returns logits over discrete actions.
    Compatible with .max(1).indices in the existing test harness.
    """
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = _mlp(n_obs, [256, 128], n_actions)

    def forward(self, x):
        return self.net(x)

    def get_action_and_log_prob(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class _PPOCritic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.net = _mlp(n_obs, [256, 128], 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOAgent:
    """
    Proximal Policy Optimisation with Generalised Advantage Estimation (GAE).

    Notes
    -----
    - On-policy rollout buffer.
    - Uses clipped surrogate objective.
    - Uses proper one-step bootstrap from the final next_state.
    - Updates actor and critic separately (no double-backward bug).
    """

    def __init__(
        self,
        n_observations, n_actions, device,
        lr=3e-4, gamma=0.99, lam=0.95,
        clip_eps=0.2, n_epochs=4, batch_size=64,
        ent_coef=0.01, vf_coef=0.5,
        update_every=48,
    ):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_every = update_every

        self.actor = _PPOActor(n_observations, n_actions).to(device)
        self.critic = _PPOCritic(n_observations).to(device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.policy_net = self.actor

        # Rollout buffer
        self._buf_states = []
        self._buf_actions = []
        self._buf_rewards = []
        self._buf_dones = []
        self._buf_log_probs = []
        self._buf_values = []
        self._last_next_state = None

        self._step_counter = 0
        self._pending_log_prob = None
        self._pending_value = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        state : (1, n_obs) float tensor
        returns: (1, 1) long tensor
        """
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action_and_log_prob(state)
            value = self.critic(state)

        self._pending_log_prob = log_prob.detach()
        self._pending_value = value.detach()
        return action.view(1, 1)

    def memorize(self, state, action, next_state, reward):
        """Store one transition. next_state may be None (terminal)."""
        self._buf_states.append(state)
        self._buf_actions.append(action)
        self._buf_rewards.append(reward)
        self._buf_dones.append(next_state is None)
        self._buf_log_probs.append(self._pending_log_prob)
        self._buf_values.append(self._pending_value)
        self._last_next_state = next_state
        self._step_counter += 1

    def optimize_model(self):
        """Run a PPO update every `update_every` steps."""
        if self._step_counter < self.update_every:
            return
        if len(self._buf_states) < 2:
            return

        self._ppo_update()
        self._clear_buffer()

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        if "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
            if "critic" in checkpoint:
                self.critic.load_state_dict(checkpoint["critic"])
        else:
            self.actor.load_state_dict(checkpoint)
        print(f"PPOAgent: model loaded from {model_path}")

    def save_model(self, path: str):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_last_value(self) -> torch.Tensor:
        """Bootstrap value V(s_{T+1}) from the final next_state if non-terminal."""
        with torch.no_grad():
            if self._last_next_state is None:
                return torch.tensor(0.0, device=self.device)
            return self.critic(self._last_next_state).view(-1)[0]

    def _ppo_update(self):
        T = len(self._buf_states)

        states = torch.cat(self._buf_states).to(self.device)
        actions = torch.cat(self._buf_actions).view(-1).to(self.device)
        rewards = torch.cat([r.view(1) for r in self._buf_rewards]).to(self.device)
        dones = torch.tensor(self._buf_dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.stack(self._buf_log_probs).view(-1).to(self.device)
        values = torch.stack(self._buf_values).view(-1).to(self.device)

        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        last_value = self._compute_last_value()

        with torch.no_grad():
            for t in reversed(range(T)):
                next_value = last_value if t == T - 1 else values[t + 1]
                not_done = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_value * not_done - values[t]
                gae = delta + self.gamma * self.lam * not_done * gae
                advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        indices = np.arange(T)
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                idx = indices[start:start + self.batch_size]
                idx_t = torch.as_tensor(idx, device=self.device, dtype=torch.long)

                s_b = states[idx_t]
                a_b = actions[idx_t]
                adv_b = advantages[idx_t].detach()
                ret_b = returns[idx_t].detach()
                old_lp_b = old_log_probs[idx_t].detach()

                logits = self.actor(s_b)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(a_b)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                value_pred = self.critic(s_b)
                critic_loss = self.vf_coef * F.mse_loss(value_pred, ret_b)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

    def _clear_buffer(self):
        self._buf_states.clear()
        self._buf_actions.clear()
        self._buf_rewards.clear()
        self._buf_dones.clear()
        self._buf_log_probs.clear()
        self._buf_values.clear()
        self._last_next_state = None
        self._step_counter = 0
        self._pending_log_prob = None
        self._pending_value = None
