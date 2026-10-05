"""
PPO Agent — Proximal Policy Optimisation for Dynamic Task Sequencing.

Architecture
────────────
  Observation → [Shared MLP encoder]
                        │
               ┌────────┴────────┐
           [Actor head]    [Critic head]
               │
         Action logits (num_workers * num_tasks + 1)
               │
         Masked softmax  ← legality supplied by the environment
               │
           Sampled action

Key design choices
──────────────────
  1. Action masking.  The mask is produced by the ENVIRONMENT
     (`env.legal_action_mask()`, surfaced as `info["action_mask"]`) and simply
     read here.  Policy and environment therefore cannot disagree about what is
     legal — a class of bug that is easy to introduce when the mask is
     re-derived inside the agent.
  2. A WAIT action is always legal, so the agent is never forced into an
     illegal move when the team is busy or blocked.
  3. Separate actor / critic heads over a shared trunk (standard PPO).
  4. Entropy bonus for exploration; optional linear learning-rate anneal.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict


# ── Network ───────────────────────────────────────────────────────────── #

class TaskSequencingNetwork(nn.Module):
    """Shared encoder + actor + critic heads."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[Categorical, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.actor(h)
        if mask is not None:
            # Guard against an all-False row producing NaNs after softmax.
            safe = mask.clone()
            empty = ~safe.any(dim=-1)
            if empty.any():
                safe[empty] = True
            logits = logits.masked_fill(~safe, -1e9)
        dist  = Categorical(logits=logits)
        value = self.critic(h).squeeze(-1)
        return dist, value


# ── Rollout buffer ────────────────────────────────────────────────────── #

class RolloutBuffer:
    """Stores a single fixed-length PPO rollout."""

    def __init__(self, size: int, obs_dim: int, action_dim: int,
                 device: torch.device):
        self.size   = size
        self.device = device
        self.obs    = torch.zeros(size, obs_dim)
        self.acts   = torch.zeros(size, dtype=torch.long)
        self.rews   = torch.zeros(size)
        self.vals   = torch.zeros(size)
        self.logps  = torch.zeros(size)
        self.dones  = torch.zeros(size)
        self.masks  = torch.zeros(size, action_dim, dtype=torch.bool)
        self.ptr    = 0

    def reset(self):
        self.ptr = 0

    @property
    def is_full(self) -> bool:
        return self.ptr >= self.size

    def store(self, obs, act, rew, val, logp, done, mask):
        if self.is_full:
            raise RuntimeError("rollout buffer overflow — call update() first")
        i = self.ptr
        self.obs[i]   = torch.as_tensor(obs, dtype=torch.float32)
        self.acts[i]  = int(act)
        self.rews[i]  = float(rew)
        self.vals[i]  = float(val)
        self.logps[i] = float(logp)
        self.dones[i] = float(done)
        self.masks[i] = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        self.ptr += 1

    def compute_returns(self, last_value: float, gamma: float, lam: float
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation over the filled portion."""
        n = self.ptr
        advantages = torch.zeros(n)
        gae = 0.0
        next_val = last_value
        for t in reversed(range(n)):
            non_terminal = 1.0 - self.dones[t].item()
            td = self.rews[t].item() + gamma * next_val * non_terminal - self.vals[t].item()
            gae = td + gamma * lam * non_terminal * gae
            advantages[t] = gae
            next_val = self.vals[t].item()
        returns = advantages + self.vals[:n]
        return advantages.to(self.device), returns.to(self.device)

    def to(self, device: torch.device):
        for name in ("obs", "acts", "rews", "vals", "logps", "dones", "masks"):
            setattr(self, name, getattr(self, name).to(device))
        self.device = device
        return self


# ── PPO Agent ─────────────────────────────────────────────────────────── #

class PPOAgent:
    """
    Proximal Policy Optimisation agent for dynamic task sequencing.

    Parameters
    ----------
    obs_dim       dimension of the (observer-enhanced) observation
    action_dim    size of the environment's discrete action space
    hidden        hidden layer width
    lr            learning rate
    gamma         discount factor
    lam           GAE lambda
    clip_eps      PPO clip ratio
    ent_coef      entropy coefficient
    vf_coef       value-loss coefficient
    update_epochs PPO epochs per rollout
    minibatch     minibatch size
    rollout_len   transitions per rollout
    """

    def __init__(
        self,
        obs_dim:       int,
        action_dim:    int,
        hidden:        int   = 256,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        lam:           float = 0.95,
        clip_eps:      float = 0.2,
        ent_coef:      float = 0.01,
        vf_coef:       float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int   = 4,
        minibatch:     int   = 64,
        rollout_len:   int   = 2048,
        device:        str   = "cpu",
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.ent_coef   = ent_coef
        self.vf_coef    = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch     = minibatch
        self.rollout_len   = rollout_len
        self.device        = torch.device(device)
        self.initial_lr    = lr

        self.net = TaskSequencingNetwork(
            obs_dim=obs_dim, action_dim=action_dim, hidden=hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(rollout_len, obs_dim, action_dim, self.device)
        self.buffer.to(self.device)

        self._total_updates = 0
        self.config: Dict[str, object] = {}

    # ── mask ─────────────────────────────────────────────────────────── #
    def build_mask(self, info: dict) -> np.ndarray:
        """
        Read the environment's canonical legality mask.

        Falls back to 'everything legal' only if an info dict without a mask is
        supplied, which should not happen with CollaborativeTaskEnv.
        """
        mask = info.get("action_mask")
        if mask is None:
            return np.ones(self.action_dim, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            mask = mask.copy()
            mask[-1] = True                 # WAIT is the last action
        return mask

    # ── action selection ─────────────────────────────────────────────── #
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, info: dict,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        obs_t  = torch.as_tensor(obs, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(self.build_mask(info),
                                 device=self.device).unsqueeze(0)
        dist, value = self.net(obs_t, mask_t)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return (int(action.item()),
                float(dist.log_prob(action).item()),
                float(value.item()))

    @torch.no_grad()
    def value_of(self, obs: np.ndarray, info: dict) -> float:
        obs_t  = torch.as_tensor(obs, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(self.build_mask(info),
                                 device=self.device).unsqueeze(0)
        _, value = self.net(obs_t, mask_t)
        return float(value.item())

    # ── store transition ─────────────────────────────────────────────── #
    def store(self, obs, action, reward, value, logp, done, info):
        self.buffer.store(obs, action, reward, value, logp,
                          float(done), self.build_mask(info))

    # ── learning-rate anneal ─────────────────────────────────────────── #
    def set_lr_fraction(self, frac: float):
        """frac = 1.0 at the start of training, 0.0 at the end."""
        lr = max(self.initial_lr * frac, 1e-6)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ── PPO update ───────────────────────────────────────────────────── #
    def update(self, last_obs: np.ndarray, last_info: dict,
               last_done: bool = False) -> Dict[str, float]:
        """Run a PPO update over the collected rollout. Returns loss metrics."""
        n = self.buffer.ptr
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}

        last_val = 0.0 if last_done else self.value_of(last_obs, last_info)
        advantages, returns = self.buffer.compute_returns(last_val, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_all   = self.buffer.obs[:n]
        acts_all  = self.buffer.acts[:n]
        logp_all  = self.buffer.logps[:n]
        masks_all = self.buffer.masks[:n]

        total_pg = total_vf = total_ent = total_kl = total_clip = 0.0
        batches = 0

        for _ in range(self.update_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch):
                mb = perm[start:start + self.minibatch]
                if mb.numel() < 2:
                    continue

                dist, val = self.net(obs_all[mb], masks_all[mb])
                new_logp = dist.log_prob(acts_all[mb])
                entropy  = dist.entropy().mean()

                log_ratio = new_logp - logp_all[mb]
                ratio     = log_ratio.exp()
                adv_mb    = advantages[mb]

                pg_loss = -torch.min(
                    ratio * adv_mb,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb,
                ).mean()
                vf_loss = F.mse_loss(val, returns[mb])
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    total_kl   += float(((ratio - 1) - log_ratio).mean())
                    total_clip += float(((ratio - 1).abs() > self.clip_eps).float().mean())
                total_pg  += pg_loss.detach().item()
                total_vf  += vf_loss.detach().item()
                total_ent += entropy.detach().item()
                batches   += 1

        self.buffer.reset()
        self._total_updates += 1
        b = max(batches, 1)
        return {
            "policy_loss": total_pg / b,
            "value_loss":  total_vf / b,
            "entropy":     total_ent / b,
            "approx_kl":   total_kl / b,
            "clip_frac":   total_clip / b,
        }

    # ── save / load ──────────────────────────────────────────────────── #
    def save(self, path: str, config: Optional[dict] = None):
        """`config` records the ablation flags and partner pool this agent was
        trained with, so evaluation can reproduce the same setup."""
        torch.save({
            "net":        self.net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "updates":    self._total_updates,
            "obs_dim":    self.obs_dim,
            "action_dim": self.action_dim,
            "config":     dict(config or {}),
        }, path)
        print(f"[PPOAgent] Saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_obs, saved_act = ckpt.get("obs_dim"), ckpt.get("action_dim")
        if saved_obs is not None and saved_obs != self.obs_dim:
            raise ValueError(
                f"checkpoint obs_dim={saved_obs} but agent expects {self.obs_dim}. "
                "The environment configuration must match the one used for training.")
        if saved_act is not None and saved_act != self.action_dim:
            raise ValueError(
                f"checkpoint action_dim={saved_act} but agent expects {self.action_dim}.")
        self.net.load_state_dict(ckpt["net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._total_updates = ckpt.get("updates", 0)
        self.config = dict(ckpt.get("config", {}))
        note = ""
        if self.config:
            flags = [k for k in ("no_observer", "no_reassignment", "curriculum")
                     if self.config.get(k)]
            if flags:
                note = "  [" + ", ".join(flags) + "]"
        print(f"[PPOAgent] Loaded checkpoint <- {path} "
              f"(updates={self._total_updates}){note}")
