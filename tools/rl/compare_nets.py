"""Matched comparison of RL networks on the games they played (rl2 rounds): each game's conditions (opponent, map,
start slots, both factions) are fixed per schedule row, so two networks can be compared condition by condition --
map and faction luck is not counted as skill. Stratified (Mantel-Haenszel weights) difference of win rates, with a
95 % range, per opponent group.

    python compare_nets.py rl_089 rl_106 rl_107 ...      # each against the first
    python compare_nets.py rl_089 --newest 6              # the newest 6 networks against rl_089
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

FAF = Path(os.environ.get("FAF_ROOT", r"C:\faf"))   # the workspace (README: Setup)
sys.path.insert(0, str(FAF / "faf-neural-ai" / "ai" / "python"))
from fa_train.rl_loop import _map_km, _names  # noqa: E402

SCRIPTED = ("adaptive", "rush", "tech")


def group_of(ai: str, km: int) -> str:
    size = "large" if km > 10 else "small"
    if ai in SCRIPTED:
        return f"scripted/{size}"
    if ai.startswith("self@"):
        return f"self-play/{size}"
    return f"{ai.replace('m28aicheat@', 'M28 x').replace('m28ai', 'M28 full')}/{size}"


def load(run: Path) -> dict:
    """network -> group -> condition -> [wins, games]"""
    names = _names(FAF / "_runs" / "server" / "vaa_M" / "checkpoints" / "step_01.pt", run / "checkpoints")
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in (run / "rounds").glob("lane*_*"):
        ep = r / "eval" / "episodes.jsonl"
        if not (r / "DONE").exists() or not ep.exists():
            continue
        net = names.get(json.loads((r / "round.json").read_text()).get("checkpoint_sha256"))
        if net is None:
            continue
        for line in open(ep):
            e = json.loads(line) if line.strip() else None
            if not e:
                continue
            ai = str(e.get("ai"))
            cond = (ai, e.get("map"), tuple(e.get("slots") or ()), e.get("faction"), e.get("ai_faction"))
            cell = out[net][group_of(ai, _map_km(e.get("map", "")))][cond]
            cell[0] += int(e.get("outcome") == "victory")
            cell[1] += 1
    return out


MATCH = {"full": lambda c: c,                      # opponent, map, start slots, both factions
         "map": lambda c: (c[0], c[1], c[3]),      # opponent, map, our faction
         "faction": lambda c: (c[0], c[3], c[4])}  # opponent, both factions


def coarsen(cells: dict, match: str) -> dict:
    """Since 2026-10-14 the conditions are drawn per round (vary_conditions): exact rows rarely repeat across
    networks, so match on fewer of them (still not counting faction or map luck as skill, only fewer pairs)."""
    out = defaultdict(lambda: [0, 0])
    for cond, (w, n) in cells.items():
        k = MATCH[match](cond)
        out[k][0] += w
        out[k][1] += n
    return out


def compare(ref: dict, cand: dict) -> tuple[float, float, int, int] | None:
    """Stratified difference (cand - ref) in win rate, its standard error, and the games compared (each side)."""
    num = den = var = 0.0
    n_ref = n_cand = 0
    for cond, (wc, nc) in cand.items():
        if cond not in ref:
            continue
        wr, nr = ref[cond]
        if nc == 0 or nr == 0:
            continue
        w = nc * nr / (nc + nr)
        pc, pr = wc / nc, wr / nr
        num += w * (pc - pr)
        den += w
        p = (wc + wr) / (nc + nr)
        var += w * w * p * (1 - p) * (1 / nc + 1 / nr)
        n_ref += nr
        n_cand += nc
    if den == 0:
        return None
    return num / den, math.sqrt(var) / den, n_ref, n_cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cands", nargs="*")
    ap.add_argument("--newest", type=int, default=0)
    ap.add_argument("--match", choices=sorted(MATCH), default="full",
                    help="which conditions must agree: full row, opponent+map+our faction, or opponent+factions")
    ap.add_argument("--run", type=Path, default=FAF / "_runs" / "server" / "rl2")
    a = ap.parse_args()
    data = load(a.run)
    cands = list(a.cands)
    if a.newest:
        cands += sorted(n for n in data if n.startswith("rl_") and n > a.ref)[-a.newest:]
    for cand in cands:
        print(f"{cand} vs {a.ref} (match {a.match}):")
        for group in sorted(set(data[a.ref]) | set(data[cand])):
            res = compare(coarsen(data[a.ref].get(group, {}), a.match),
                          coarsen(data[cand].get(group, {}), a.match))
            if res is None:
                continue
            d, se, nr, nc = res
            flag = "  better" if d - 1.96 * se > 0 else "  worse" if d + 1.96 * se < 0 else ""
            print(f"  {group:22s} {100 * d:+6.1f} points (95 % range {100 * (d - 1.96 * se):+.1f} .. "
                  f"{100 * (d + 1.96 * se):+.1f}); {nc} games vs {nr}{flag}")


if __name__ == "__main__":
    main()
