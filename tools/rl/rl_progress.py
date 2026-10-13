"""Matched win rates (adaptive/rush/tech, re-weighted start), defeat rates and M28 results of the newest RL
networks of a run:  python rl_progress.py [run] [--since rl_036] [--wait-for rl_043 --games 100]

With --wait-for: polls every 10 minutes until that network has --games matched games, then prints the table."""
import os
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

FAF = Path(os.environ.get("FAF_ROOT", r"C:\faf"))   # the workspace (README: Setup)
sys.path.insert(0, str(FAF / "faf-neural-ai" / "ai" / "python"))
from fa_train.rl_loop import _names, matched, win_rates_by_ai  # noqa: E402

STANDARD = ("adaptive", "rush", "tech")


def table(run: Path, since: str) -> tuple[str, dict]:
    start = FAF / "_runs" / "server" / "vaa_M" / "checkpoints" / "step_01.pt"
    names = _names(start, run / "checkpoints")
    by_ai = win_rates_by_ai(run / "rounds", names)
    outcomes = defaultdict(lambda: defaultdict(int))
    for r in (run / "rounds").glob("lane*_*"):
        if not (r / "DONE").exists():
            continue
        net = names.get(json.loads((r / "round.json").read_text()).get("checkpoint_sha256"))
        if net is None:
            continue
        for line in open(r / "eval" / "episodes.jsonl"):
            e = json.loads(line)
            if e["ai"] in STANDARD:
                outcomes[net][e["outcome"]] += 1
    lines, games = [], {}
    for net in sorted(k for k in by_ai if k != "start" and k >= since):
        m = matched(by_ai, net)
        if not m:
            continue
        w, g, base = m
        games[net] = g
        se = math.sqrt(max(1e-9, (w / g) * (1 - w / g) / g))
        o = outcomes[net]
        n = max(1, sum(o.values()))
        m28 = " ".join(f"{ai.replace('m28aicheat@', 'm28x')}:{a}/{b}" for ai, (a, b) in sorted(by_ai[net].items())
                       if ai.startswith("m28"))
        lines.append(f"{net}: {w}/{g} = {w / g:.0%} (+-{1.96 * se:.0%}) vs start {base:.0%} ({100 * (w / g - base):+.0f}); "
                     f"defeat {o['defeat'] / n:.0%} timeout {o['timeout'] / n:.0%} | {m28}")
    return "\n".join(lines), games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=str(FAF / "_runs" / "server" / "rl2"))
    ap.add_argument("--since", default="rl_030")
    ap.add_argument("--wait-for")
    ap.add_argument("--games", type=int, default=100)
    a = ap.parse_args()
    while True:
        text, games = table(Path(a.run), a.since)
        if not a.wait_for or games.get(a.wait_for, 0) >= a.games:
            print(text, flush=True)
            return
        time.sleep(600)


if __name__ == "__main__":
    main()
