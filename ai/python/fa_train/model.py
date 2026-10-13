"""Actor and critic for FactoredEnv (docs/training-design.md, section 7).

Actor: a unit-set encoder (per-unit MLP over dynamic features, static blueprint features and a learned
blueprint embedding; masked mean and max pooling per group: own / seen / previously seen / radar) plus
a scalar MLP, a two-layer torso and one masked categorical head per action head. Small on purpose: it is
the part that must later run in FAF's Lua.

Critic: the same encoders plus the privileged vector (true totals of both armies), as a separate tower.
It is only used in training.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fa_env.factored import HEAD_NAMES, HEADS, observation_sizes

VOCAB = 1024      # blueprint indices are clamped into this (FAF has ~750 unit blueprints)
N_GROUPS = 4


class UnitEncoder(nn.Module):
    def __init__(self, n_features: int, embed: int = 16, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, embed)
        self.mlp = nn.Sequential(nn.Linear(n_features + embed, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out_size = 2 * N_GROUPS * hidden

    def forward(self, units, bp, group, mask):
        h = self.mlp(torch.cat([units, self.embed(bp.clamp(0, VOCAB - 1))], -1))      # [B, N, H], >= 0
        pooled = []
        for g in range(N_GROUPS):
            m = (mask & (group == g)).unsqueeze(-1).to(h.dtype)                     # [B, N, 1]
            count = m.sum(1).clamp(min=1.0)
            pooled.append((h * m).sum(1) / count)
            pooled.append((h * m).max(1).values)                                     # 0 when the group is empty
        return torch.cat(pooled, -1)


class Actor(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        sizes = observation_sizes()
        self.units = UnitEncoder(sizes["units"])
        self.scalars = nn.Sequential(nn.Linear(sizes["scalars"], 128), nn.ReLU())
        self.torso = nn.Sequential(nn.Linear(self.units.out_size + 128, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleDict({h: nn.Linear(hidden, len(HEADS[h])) for h in HEAD_NAMES})
        for head in self.heads.values():          # start near uniform over the allowed options
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, obs: dict) -> dict[str, torch.Tensor]:
        u = self.units(obs["units"], obs["unit_bp"], obs["unit_group"], obs["unit_mask"])
        z = self.torso(torch.cat([u, self.scalars(obs["scalars"])], -1))
        return {h: self.heads[h](z) for h in HEAD_NAMES}


class Critic(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        sizes = observation_sizes()
        self.units = UnitEncoder(sizes["units"])
        self.scalars = nn.Sequential(nn.Linear(sizes["scalars"] + sizes["privileged"], 128), nn.ReLU())
        self.torso = nn.Sequential(nn.Linear(self.units.out_size + 128, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs: dict) -> torch.Tensor:
        u = self.units(obs["units"], obs["unit_bp"], obs["unit_group"], obs["unit_mask"])
        s = self.scalars(torch.cat([obs["scalars"], obs["privileged"]], -1))
        return self.torso(torch.cat([u, s], -1)).squeeze(-1)


def masked_logits(logits: dict, obs: dict) -> dict:
    return {h: logits[h].masked_fill(~obs[f"mask_{h}"], -1e9) for h in HEAD_NAMES}


def distributions(logits: dict, obs: dict) -> dict:
    return {h: torch.distributions.Categorical(logits=l) for h, l in masked_logits(logits, obs).items()}


def to_torch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        t = torch.as_tensor(v, device=device)
        if k.startswith("mask_") or k == "unit_mask":
            t = t.bool()
        elif t.dtype == torch.float64:
            t = t.float()
        out[k] = t
    return out


def stack(observations: list[dict]) -> dict:
    import numpy as np
    return {k: np.stack([o[k] for o in observations]) for k in observations[0]}


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
