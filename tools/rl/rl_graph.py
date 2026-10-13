"""Graph of every RL network's results on the server:  python rl_graph.py [run] [out.png]"""
import os
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAF = Path(os.environ.get("FAF_ROOT", r"C:\faf"))   # the workspace (README: Setup)
sys.path.insert(0, str(FAF / "faf-neural-ai" / "ai" / "python"))
from fa_train.rl_loop import _names, matched, win_rates_by_ai  # noqa: E402

run = Path(sys.argv[1]) if len(sys.argv) > 1 else FAF / "_runs" / "server" / "rl2"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else FAF / "_runs" / "server" / "rl2_progress.png"
start = FAF / "_runs" / "server" / "vaa_M" / "checkpoints" / "step_01.pt"
names = _names(start, run / "checkpoints")
by_ai = win_rates_by_ai(run / "rounds", names)
defeats = defaultdict(lambda: [0, 0])
for r in (run / "rounds").glob("lane*_*"):
    if not (r / "DONE").exists():
        continue
    net = names.get(json.loads((r / "round.json").read_text()).get("checkpoint_sha256"))
    if net is None:
        continue
    for line in open(r / "eval" / "episodes.jsonl"):
        e = json.loads(line)
        if e["ai"] in ("adaptive", "rush", "tech"):
            defeats[net][0] += e["outcome"] == "defeat"
            defeats[net][1] += 1

nets = sorted(n for n in by_ai if n != "start")
idx, rate, err, base, dfx, dfy = [], [], [], [], [], []
m28 = defaultdict(lambda: ([], [], []))
for n in nets:
    i = int(n.split("_")[1])
    m = matched(by_ai, n)
    if m and m[1] >= 30:
        w, g, b = m
        p = w / g
        idx.append(i); rate.append(100 * p); err.append(196 * math.sqrt(p * (1 - p) / g)); base.append(100 * b)
    d = defeats.get(n)
    if d and d[1] >= 30:
        dfx.append(i); dfy.append(100 * d[0] / d[1])
    for ai, (w, g) in by_ai[n].items():
        if ai.startswith("m28") and g >= 12:
            opp, _, km = ai.partition("#")
            key = ("M28 full" if opp == "m28ai" else f"M28 x{opp.split('@')[1]}") + (f" ({km})" if km else "")
            m28[key][0].append(i); m28[key][1].append(100 * w / g); m28[key][2].append(g)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
ax1.errorbar(idx, rate, yerr=err, fmt="o-", color="#2b6cb0", ecolor="#90cdf4", capsize=2, lw=1.5, ms=4,
             label="RL network: wins vs adaptive/rush/tech (95 % range)")
ax1.plot(idx, base, "--", color="#718096", lw=1.2, label="starting network on the same opponents")
ax1.plot(dfx, dfy, "s:", color="#c53030", ms=3, lw=1, label="defeat rate (same games)")
for x, text in ((12, "spatial credit"), (17.5, "M28 in the mix"), (24.5, "M28 ladder"),
                (38, "PPO fix, lr up"), (43, "reward v2 +\ndanger maps"), (45, "lr 2e-5"), (51, "4 passes")):
    ax1.axvline(x, color="#a0aec0", lw=0.8, alpha=0.7)
    ax1.text(x + 0.2, 101, text, fontsize=7.5, color="#4a5568", va="top", rotation=90)
ax1.set_ylim(0, 102)
ax1.set_ylabel("% of games")
ax1.set_title("RL progress on the server (rl2), every network with 30+ comparable games")
ax1.legend(loc="lower left", fontsize=8.5)
ax1.grid(alpha=0.3)
colors = {"M28 x0.5": "#68d391", "M28 x0.6": "#38a169", "M28 x0.65": "#2f855a", "M28 x0.7": "#d69e2e",
          "M28 full": "#c53030"}
for key in sorted(m28, key=lambda k: (k == "M28 full", k)):
    xs, ys, gs = m28[key]
    ax2.plot(xs, ys, "o-", color=colors.get(key, "#4a5568"), ms=[2 + g / 12 for g in gs][0], lw=1.2, label=key)
ax2.set_ylim(-3, 103)
ax2.set_ylabel("% won against M28")
ax2.set_xlabel("RL iteration (network rl_NNN)")
ax2.legend(loc="upper left", fontsize=8.5, ncol=5)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=130)
print(out)
