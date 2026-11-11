"""Per-lane game throughput from the player's round logs (eval.log), for A/B tests of player settings.

  python -m fa_train.lane_speed --since 20261110T0230Z [--home ~/fa/player] [--run ~/fa/rl3]

Each round's eval.log prints every minute "[hh:mm:ss] D decisions/s, G game-s/s ..." and "per decision: game step
S ms, waited for orders W ms; batches of B; decider: ... network N ... ms per decision; idle I %". This averages them
per lane over the rounds started at or after --since, with each round's worker count and inference threads."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RATE = re.compile(r"\] ([\d.]+) decisions/s, ([\d.]+) game-s/s")
STEP = re.compile(r"per decision: game step (\d+) ms, waited for orders (\d+) ms; batches of ([\d.]+); decider: "
                  r"prepare ([\d.]+), network ([\d.]+), orders ([\d.]+) ms per decision; idle (\d+) %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="UTC stamp as in round folder names, e.g. 20261110T0230Z")
    ap.add_argument("--home", type=Path, default=Path.home() / "fa" / "player")
    ap.add_argument("--run", type=Path, default=Path.home() / "fa" / "rl3")
    ap.add_argument("--skip-minutes", type=int, default=5, help="ignore each round's first minutes (loading)")
    a = ap.parse_args()
    lanes: dict[str, dict] = {}
    for root in (a.home / "rounds", a.run / "rounds"):
        for folder in sorted(root.glob("lane*_*")):
            lane, stamp = folder.name[4:].split("_", 1)
            if stamp < a.since or not (folder / "eval.log").exists():
                continue
            try:
                cfg = json.loads((folder / "round.json").read_text()).get("config", {})
            except (OSError, ValueError):
                cfg = {}
            extra = (cfg.get("lane_env") or {}).get(lane) or {}
            threads = extra.get("OMP_NUM_THREADS", cfg.get("torch_threads"))
            key = (f"lane {lane} (workers {cfg.get('workers')}, threads {threads}"
                   + (", bf16" if extra.get("FA_ENV_ACT_BF16") == "1" else "") + ")")
            s = lanes.setdefault(key, {"rounds": set(), "rate": [], "step": []})
            s["rounds"].add(folder.name)
            minutes = 0
            for line in (folder / "eval.log").read_text(errors="replace").splitlines():
                m = RATE.search(line)
                if m:
                    minutes += 1
                    if minutes > a.skip_minutes:
                        s["rate"].append((float(m.group(1)), float(m.group(2))))
                    continue
                m = STEP.search(line)
                if m and minutes > a.skip_minutes:
                    s["step"].append(tuple(float(v) for v in m.groups()))
    print(f"rounds started since {a.since}; minute averages after each round's first {a.skip_minutes} minutes\n")
    print("| lane | rounds | minutes | decisions/s | game-s/s | game step ms | wait for orders ms | batch | network ms | decider idle % |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, s in sorted(lanes.items()):
        if not s["rate"]:
            continue
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        st = list(zip(*s["step"])) if s["step"] else [[]] * 7
        print(f"| {key} | {len(s['rounds'])} | {len(s['rate'])} | {mean([r[0] for r in s['rate']]):.1f} | "
              f"{mean([r[1] for r in s['rate']]):.1f} | {mean(st[0]):.0f} | {mean(st[1]):.0f} | {mean(st[2]):.2f} | "
              f"{mean(st[4]):.1f} | {mean(st[6]):.0f} |")


if __name__ == "__main__":
    main()
