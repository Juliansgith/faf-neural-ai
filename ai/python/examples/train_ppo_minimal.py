"""Minimal PPO over MacroEnv (CleanRL-style, one file): proves the loop end to end -- parallel games, a
policy on the GPU, masked actions, advantage estimation, updates, checkpoints. It is a starting point,
not a tuned trainer; expect to replace the network, the reward and the macro set.

    python train_ppo_minimal.py --envs 4 --steps 64 --updates 10

Needs torch (see ai/README.md).
"""
import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import GameConfig  # noqa: E402
from fa_env.featurize import SPATIAL_CHANNELS, UNIT_FEATURES  # noqa: E402
from fa_env.gym_env import MACROS, MacroEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=64, help="decisions per env per update")
parser.add_argument("--updates", type=int, default=10)
parser.add_argument("--map", default="SCMP_016")
parser.add_argument("--decision-ticks", type=int, default=20)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--clip", type=float, default=0.2)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--save", default="ppo_macro.pt")
args = parser.parse_args()

MAP_SIZE, MAX_UNITS, VOCAB = 64, 256, 1024


class Policy(nn.Module):
    """Map CNN + unit set encoder (blueprint embedding, masked mean) + scalars -> policy and value."""

    def __init__(self, n_actions):
        super().__init__()
        c = len(SPATIAL_CHANNELS)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 16, 256), nn.ReLU(),
        )
        self.bp_embed = nn.Embedding(VOCAB, 16)
        self.unit_mlp = nn.Sequential(nn.Linear(len(UNIT_FEATURES) + 16, 128), nn.ReLU(), nn.Linear(128, 128))
        self.scalar_mlp = nn.Sequential(nn.Linear(11, 64), nn.ReLU())
        self.torso = nn.Sequential(nn.Linear(256 + 128 + 64, 256), nn.ReLU())
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def forward(self, obs):
        m = self.cnn(obs["map"])
        u = torch.cat([obs["units"], self.bp_embed(obs["unit_bp"].clamp(0, VOCAB - 1))], -1)
        u = self.unit_mlp(u)
        mask = obs["unit_mask"].unsqueeze(-1).float()
        u = (u * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        h = self.torso(torch.cat([m, u, self.scalar_mlp(obs["scalars"])], -1))
        return self.pi(h), self.v(h).squeeze(-1)


def masked_dist(logits, action_mask):
    logits = logits.masked_fill(action_mask == 0, -1e8)
    return torch.distributions.Categorical(logits=logits)


def to_torch(obs, device):
    return {k: torch.as_tensor(np.asarray(v), device=device) for k, v in obs.items()}


def make(i):
    def thunk():
        return MacroEnv(GameConfig(map=args.map, ai_faction=1 + i % 4, seed=1000 + i, window=(640, 480)),
                        decision_ticks=args.decision_ticks, max_minutes=15, map_size=MAP_SIZE, max_units=MAX_UNITS)
    return thunk


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.AsyncVectorEnv([make(i) for i in range(args.envs)], shared_memory=False)
    policy = Policy(len(MACROS)).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    print(f"device {device}; {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M parameters")

    obs, _ = envs.reset(seed=list(range(args.envs)))
    obs_t = to_torch(obs, device)
    done = torch.zeros(args.envs, device=device)
    T, N = args.steps, args.envs
    for update in range(1, args.updates + 1):
        t0 = time.monotonic()
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []
        ep_returns = []
        for _ in range(T):
            with torch.no_grad():
                logits, value = policy(obs_t)
                dist = masked_dist(logits, obs_t["action_mask"])
                action = dist.sample()
            buf_obs.append(obs_t)
            buf_act.append(action)
            buf_logp.append(dist.log_prob(action))
            buf_val.append(value)
            buf_done.append(done)
            obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            obs_t = to_torch(obs, device)
            buf_rew.append(torch.as_tensor(reward, device=device, dtype=torch.float32))
            done = torch.as_tensor(terminated | truncated, device=device, dtype=torch.float32)
        with torch.no_grad():
            _, next_value = policy(obs_t)
        # GAE
        adv = torch.zeros(T, N, device=device)
        last = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - (done if t == T - 1 else buf_done[t + 1])
            nv = next_value if t == T - 1 else buf_val[t + 1]
            delta = buf_rew[t] + args.gamma * nv * next_nonterminal - buf_val[t]
            last = delta + args.gamma * args.gae_lambda * next_nonterminal * last
            adv[t] = last
        returns = adv + torch.stack(buf_val)

        flat = {k: torch.cat([o[k] for o in buf_obs]) for k in buf_obs[0]}
        b_act, b_logp = torch.cat(buf_act), torch.cat(buf_logp)
        b_adv, b_ret = adv.reshape(-1), returns.reshape(-1)
        idx = np.arange(T * N)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for mb in np.array_split(idx, args.minibatches):
                mb_obs = {k: v[mb] for k, v in flat.items()}
                logits, value = policy(mb_obs)
                dist = masked_dist(logits, mb_obs["action_mask"])
                ratio = (dist.log_prob(b_act[mb]) - b_logp[mb]).exp()
                a = b_adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * F.mse_loss(value, b_ret[mb])
                loss = pg + 0.5 * v_loss - args.ent_coef * dist.entropy().mean()
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()
        rew = torch.stack(buf_rew)
        counts = np.bincount(b_act.cpu().numpy(), minlength=len(MACROS))
        print(f"update {update}: {T * N / (time.monotonic() - t0):.1f} decisions/s  reward/step {rew.mean():+.4f}  "
              f"loss {loss.item():.3f}  macros {dict(zip(MACROS, counts.tolist()))}")
        torch.save(policy.state_dict(), args.save)
    envs.close()
