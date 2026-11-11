"""Paired-condition comparison of a player's lanes: each network or lane setting against a control lane, on the same maps
and factions (a difference is taken within map + own faction + opponent faction, weighted by the arm's own games).

    python -m fa_train.arena_report --rounds ~/fa/player/rounds [--control 0] [--since STAMP]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics


def load(rounds: str, since: str) -> dict:
    arms = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(rounds, "lane*_*", ""))):
        name = os.path.basename(d.rstrip("/"))
        if name.split("_", 1)[1] < since:
            continue
        lane = int(name[4:].split("_", 1)[0])
        try:
            with open(d + "eval/episodes.jsonl", encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    arms[lane].append({k: e.get(k) for k in ("ai", "map", "faction", "ai_faction", "outcome", "minutes", "mass_income")})
        except OSError:
            continue
    return arms


def standard(control: list, arm: list, ai: str):
    key = lambda r: (r["map"], r["faction"], r["ai_faction"])
    gc, ga = collections.defaultdict(list), collections.defaultdict(list)
    for r in control:
        if r["ai"] == ai:
            gc[key(r)].append(r["outcome"] == "victory")
    for r in arm:
        if r["ai"] == ai:
            ga[key(r)].append(r["outcome"] == "victory")
    common = [k for k in ga if k in gc]
    w = sum(len(ga[k]) for k in common)
    if not w:
        return None
    d = v = 0.0
    for k in common:
        share = len(ga[k]) / w
        pc, pa = sum(gc[k]) / len(gc[k]), sum(ga[k]) / len(ga[k])
        d += share * (pa - pc)
        v += share * share * (max(pc * (1 - pc), .03) / len(gc[k]) + max(pa * (1 - pa), .03) / len(ga[k]))
    return 100 * d, 196 * math.sqrt(v), w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--control", type=int, default=0)
    ap.add_argument("--since", default="")
    a = ap.parse_args()
    arms = load(os.path.expanduser(a.rounds), a.since)
    ais = sorted({r["ai"] for rows in arms.values() for r in rows})
    for lane in sorted(arms):
        rows = arms[lane]
        parts = []
        for ai in ais:
            sub = [r for r in rows if r["ai"] == ai]
            if not sub:
                continue
            win = sum(r["outcome"] == "victory" for r in sub) / len(sub)
            parts.append(f"{ai}: {100 * win:.1f} % n={len(sub)} min {statistics.mean(r['minutes'] or 0 for r in sub):.1f}")
            if lane != a.control and arms.get(a.control):
                s = standard(arms[a.control], rows, ai)
                if s:
                    parts[-1] += f" diff {s[0]:+.1f} +-{s[1]:.1f}"
        print(f"lane {lane}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
