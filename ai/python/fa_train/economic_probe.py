"""Does the network's memory understand where the economy is heading? A frozen-feature probe (nothing is trained in the
network; 2026-11-07 review and plateau report).

    python -m fa_train.economic_probe --checkpoint rl_258.pt --cache CACHE --games 240 --processes 8 --out probe.json

The memory z of every decision is computed once. Small heads then predict own mass income, army mass and mass
production capacity 30, 120 and 300 seconds ahead, and whether the game is won, on whole held-out games. Each is compared
with the simple baseline of carrying today's value forward, by game phase and faction, and the capacity head is also
scored on the change: can the memory tell which positions are about to gain capacity? Games that end before a horizon
use their final value only if they lasted at least half of that horizon; other rows are left out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from fa_env.tools_obs import PRIV_SCALE, PRIVILEGED

from .learn_parallel import first_pass
from .learn_rl import chunk_batch, load_net, micro_view
from .tools_bc import Games

SECONDS_PER_DECISION = 2
HORIZONS = (30, 120, 300)


def targets(g: dict, n: int) -> dict:
    priv = np.asarray(g["privileged"][:n], np.float32)
    out = {"income": priv[:, PRIVILEGED.index("massIncome")] * PRIV_SCALE[PRIVILEGED.index("massIncome")],
           "army": priv[:, PRIVILEGED.index("armyMass")] * PRIV_SCALE[PRIVILEGED.index("armyMass")]}
    if "mass_capacity" in g:
        out["capacity"] = np.asarray(g["mass_capacity"][:n], np.float32)
    return out


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def fit(x: np.ndarray, y: np.ndarray, classify: bool = False, epochs: int = 12, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    X, Y = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(order), 1024):
            b = order[i:i + 1024]
            p = net(X[b]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(p, Y[b]) if classify else ((p - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--games", type=int, default=240)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    net, _ = load_net(a.checkpoint, device)
    net.eval()
    games = Games([Path(a.cache)], newest=a.games)
    games.units = True
    games.teacher = {}
    usable = [i for i, g in enumerate(games.games) if "pol_has" in g][: a.games]
    lengths = {i: int(games.lengths[i]) for i in usable}
    tokens_of = {i: int(games.games[i]["offsets_t"][lengths[i]]) for i in usable}
    ns = SimpleNamespace(length=32, temperature=.5, vtrace=False, stop_file=None)
    _, _, _, _, _, features, _, _ = first_pass(net, net, games, usable, lengths, tokens_of, ns, a.processes, False, True,
                                               chunk_batch, micro_view, 1)
    rng = np.random.default_rng(0)
    shuffled = [int(i) for i in rng.permutation(usable)]
    cut = int(len(shuffled) * .8)
    train_games, test_games = shuffled[:cut], shuffled[cut:]

    def rows(game_ids, horizon, name):
        xs, ys, now, minute = [], [], [], []
        h = horizon // SECONDS_PER_DECISION
        for gi in game_ids:
            n = lengths[gi]
            t = targets(games.games[gi], n)
            if name not in t:
                continue
            keep = np.arange(n)
            keep = keep[keep + h // 2 < n]
            future = np.minimum(keep + h, n - 1)
            xs.append(features[gi][0][keep])
            ys.append(t[name][future])
            now.append(t[name][keep])
            minute.append(keep * SECONDS_PER_DECISION / 60)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(now), np.concatenate(minute)

    report = {"checkpoint": str(a.checkpoint), "train_games": len(train_games), "test_games": len(test_games), "results": {}}
    for name in ("income", "army", "capacity"):
        for horizon in HORIZONS:
            xa, ya, na, _ = rows(train_games, horizon, name)
            xb, yb, nb, mb = rows(test_games, horizon, name)
            if not len(xa):
                continue
            sub = rng.choice(len(xa), min(len(xa), 300_000), replace=False)
            scale = max(float(ya.std()), 1e-6)
            head = fit(xa[sub], (ya[sub] - na[sub]) / scale)      # the change from today's value
            with torch.no_grad():
                d = head(torch.as_tensor(xb)).squeeze(-1).numpy() * scale
            phases = {"0-10": mb < 10, "10-20": (mb >= 10) & (mb < 20), "20+": mb >= 20}
            report["results"][f"{name}+{horizon}s"] = {
                "r2_head": round(r2(nb + d, yb), 4), "r2_carry_forward": round(r2(nb, yb), 4),
                "r2_of_the_change": round(r2(d, yb - nb), 4),
                "by_phase": {k: {"n": int(v.sum()), "r2_head": round(r2(nb[v] + d[v], yb[v]), 4),
                                 "r2_carry": round(r2(nb[v], yb[v]), 4), "r2_change": round(r2(d[v], yb[v] - nb[v]), 4)}
                             for k, v in phases.items() if v.sum() > 100}}
            print(name, horizon, json.dumps(report["results"][f"{name}+{horizon}s"]), flush=True)

    def wins(game_ids):
        xs, ys, minute = [], [], []
        for gi in game_ids:
            n = lengths[gi]
            won = float(np.asarray(games.games[gi]["rew_terminal"][:n]).sum() > 0)
            xs.append(features[gi][0][:n])
            ys.append(np.full(n, won, np.float32))
            minute.append(np.arange(n) * SECONDS_PER_DECISION / 60)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(minute)

    xa, ya, _ = wins(train_games)
    xb, yb, mb = wins(test_games)
    sub = rng.choice(len(xa), min(len(xa), 300_000), replace=False)
    head = fit(xa[sub], ya[sub], classify=True)
    with torch.no_grad():
        s = head(torch.as_tensor(xb)).squeeze(-1).numpy()
    report["results"]["win"] = {"auc": round(auc(s, yb), 4), "by_minute": {
        f"{lo}-{hi}": round(auc(s[(mb >= lo) & (mb < hi)], yb[(mb >= lo) & (mb < hi)]), 4)
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30)) if ((mb >= lo) & (mb < hi)).sum() > 100}}
    print("win", json.dumps(report["results"]["win"]), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
