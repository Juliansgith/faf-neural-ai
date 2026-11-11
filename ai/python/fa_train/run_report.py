"""A plain status report of a server-side RL run: strength by network, throughput, learner health, disk. Standard
library only, read-only apart from the report it writes.

    python -m fa_train.run_report --run ~/fa/rl3 --home ~/fa/player [--days 7] [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def journal(run: Path) -> list:
    rows = []
    try:
        with open(run / "rl.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def rounds(*folders) -> list:
    """(stamp, meta) of every finished round in these folders."""
    out = []
    for folder in folders:
        for d in glob.glob(str(Path(folder) / "lane*_*")):
            try:
                if not os.path.exists(os.path.join(d, "DONE")):
                    continue
                with open(os.path.join(d, "round.json"), encoding="utf-8") as f:
                    meta = json.load(f)
                out.append((os.path.basename(d).split("_", 1)[1], meta))
            except (OSError, ValueError):
                pass
    return sorted(out, key=lambda x: x[0])


def sha_names(run: Path) -> dict:
    names = {}
    cache = run / "report_shas.json"
    try:
        names = json.loads(cache.read_text())
    except (OSError, ValueError):
        pass
    changed = False
    known = set(names.values())
    for p in sorted((run / "checkpoints").glob("*.pt")):
        if p.stem in known:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        names[h.hexdigest()] = p.stem
        changed = True
    if changed:
        try:
            cache.write_text(json.dumps(names))
        except OSError:
            pass
    return names


def pct(win: int, n: int) -> str:
    if not n:
        return "-"
    p = win / n
    return f"{100 * p:.1f} % +-{196 * math.sqrt(p * (1 - p) / n):.1f} ({n})"


def build(run: Path, home: Path, days: float) -> str:
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).strftime("%Y%m%dT%H%M%SZ")
    rows = journal(run)
    all_rounds = rounds(run / "rounds", home / "rounds")
    names = sha_names(run)
    lines = [f"# RL run report, {now:%Y-%m-%d %H:%M} UTC, last {days:g} days", ""]
    try:
        champ = json.loads((run / "champion.json").read_text())
        lines += [f"Champion: **{champ.get('name')}** since {champ.get('since')}. {champ.get('why', '')[:160]}", ""]
    except (OSError, ValueError):
        lines += ["Champion: none recorded", ""]

    recent = [(s, m) for s, m in all_rounds if s >= since]
    games = sum(int(m.get("games") or 0) for _, m in recent)
    lines += [f"## Throughput", "", f"* Games finished: {games} ({games / max(days, 1e-6):.0f} a day)"]
    done = [r for r in rows if r.get("checkpoint")]
    recent_learn = [r for r in done if str(r.get("time", "")) >= (now - timedelta(days=days)).isoformat()[:19]]
    lines.append(f"* Networks trained: {len(recent_learn)}")
    if recent_learn:
        minutes = sorted(float(r.get("minutes") or 0) for r in recent_learn)
        lines.append(f"* Minutes per training iteration: median {minutes[len(minutes) // 2]:.0f}, longest {minutes[-1]:.0f}")
    failed = [r for r in rows if r.get("exit") not in (None, 0) and str(r.get("time", "")) >= (now - timedelta(days=days)).isoformat()[:19]]
    lines.append(f"* Failed iterations: {len(failed)}")

    per_net = collections.OrderedDict()
    for _, meta in all_rounds:
        name = names.get(meta.get("checkpoint_sha256") or "", (meta.get("checkpoint_sha256") or "?")[:10])
        table = per_net.setdefault(name, collections.defaultdict(lambda: [0, 0]))
        for ai, o in (meta.get("outcomes") or {}).items():
            table[ai][0] += int((o.get("outcomes") or {}).get("victory") or 0)
            table[ai][1] += int(o.get("games") or 0)
    opponents = sorted({ai for t in per_net.values() for ai in t})
    lines += ["", "## Win rate by network (newest 25, every game kept in round records)", ""]
    lines.append("| network | " + " | ".join(opponents) + " |")
    lines.append("|---|" + "---|" * len(opponents))
    for name, t in list(per_net.items())[-25:]:
        lines.append(f"| {name} | " + " | ".join(pct(*t[ai]) if ai in t else "-" for ai in opponents) + " |")

    tiers = collections.OrderedDict()
    spread = collections.OrderedDict()                 # network -> {"5 km" | "10 km": [(extractors, income)]} at minute 10
    groups = collections.OrderedDict()                 # network -> {column: [wins, games]} (2026-11-11)
    try:
        from fa_env.maps import MAP_POOL
    except Exception:                                   # noqa: BLE001
        MAP_POOL = {}
    try:
        equiv = json.loads((run / "acting_equivalence.json").read_text())
        new_maps = set(equiv["new_maps"] if "new_maps" in equiv else equiv.get("gate_exclude_maps") or ())
    except (OSError, ValueError):
        new_maps = set()
    for stamp, meta in all_rounds[-150:]:
        name = names.get(meta.get("checkpoint_sha256") or "", (meta.get("checkpoint_sha256") or "?")[:10])
        folder = None
        for root in (run / "rounds", home / "rounds"):
            hit = root / f"lane{meta.get('lane')}_{stamp}"
            if hit.exists():
                folder = hit
        if folder is None:
            continue
        try:
            with open(folder / "eval" / "episodes.jsonl", encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    ai, map_id = str(e.get("ai", "")), e.get("map")
                    kind = ("M28 " + ai.partition("@")[2] if ai.startswith("m28") else
                            "self-play" if ai.startswith("self@") else "easy AIs")
                    where = ("new maps" if map_id in new_maps else
                             f"{MAP_POOL.get(map_id, {}).get('km', '?')} km")
                    cell = groups.setdefault(name, {}).setdefault(f"{kind}, {where}", [0, 0])
                    cell[0] += e.get("outcome") == "victory"
                    cell[1] += 1
                    if not ai.startswith("self@"):
                        for snap in e.get("economy_snapshots") or ():
                            if snap.get("minute") == 10:
                                km = MAP_POOL.get(map_id, {}).get("km", "?")
                                spread.setdefault(name, {}).setdefault(f"{km} km", []).append(
                                    ((snap.get("completed_roles") or {}).get("mex", 0), snap.get("mass_income") or 0))
                    if not ai.startswith("m28") or (e.get("minutes") or 0) < 15:
                        continue
                    last_state = (e.get("recent_state_snapshots") or [{}])[-1]
                    mex = ((last_state.get("completed_role_activity") or {}).get("mex") or {}).get("units")
                    income = last_state.get("mass_income")
                    if mex and income is not None:
                        tiers.setdefault(name, []).append(income / mex)
        except (OSError, ValueError):
            continue
    if groups:
        cols = sorted({c for t in groups.values() for c in t if not c.startswith("self-play")})
        lines += ["", "## Win rate by map group (the newest 150 rounds)", "",
                  "The network is weakest on 10 km maps, and weaker still on the four maps added on 2026-11-10 (\"new maps\", "
                  "acting_equivalence.json). M28 plays the 10 km maps at a level of its own since 2026-11-11.", "",
                  "| network | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name, t in list(groups.items())[-10:]:
            lines.append(f"| {name} | " + " | ".join(pct(*t[c]) if c in t else "-" for c in cols) + " |")
    if spread:
        sizes = sorted({c for t in spread.values() for c in t}, key=lambda c: (len(c), c))
        lines += ["", "## Expansion: extractors and mass income at minute 10 (games that lasted that long, no self-play)", "",
                  "A 10 km map has 40-60 mass deposits. On 2026-11-11 the network held about 14 extractors there at minute 10 "
                  "and still 14 at minute 15, where M28 x0.75 had 29: it does not spread beyond its own side.", "",
                  "| network | " + " | ".join(f"{c}: extractors | {c}: income | {c}: games" for c in sizes) + " |",
                  "|---|" + "---|---|---|" * len(sizes)]
        for name, t in list(spread.items())[-10:]:
            cells = []
            for c in sizes:
                v = t.get(c) or []
                mex, income = sorted(x[0] for x in v), sorted(x[1] for x in v)
                cells.append(f"{mex[len(v) // 2]} | {income[len(v) // 2]:.0f} | {len(v)}" if v else "- | - | 0")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if tiers:
        lines += ["", "## Economy tier: mass income per extractor at the end of games of 15+ minutes against M28", "",
                  "T1 alone gives 2.0. M28 reaches 5-7 in long games. This is the number the plateau report says is stuck.", "",
                  "| network | median | games |", "|---|---|---|"]
        for name, v in list(tiers.items())[-12:]:
            v = sorted(v)
            lines.append(f"| {name} | {v[len(v) // 2]:.2f} | {len(v)} |")

    if done:
        last = done[-1]
        history = last.get("learn_rl") or [{}]
        first = history[0] if history else {}
        lines += ["", "## Learner, last iteration", "",
                  f"Iteration {last.get('iteration')}, from {Path(str(last.get('from', '?'))).stem}, "
                  f"{last.get('new_games')} new games, {last.get('minutes')} minutes.", "",
                  "`" + json.dumps({k: first.get(k) for k in ("pg", "v", "ent", "kl", "ref_kl", "distill", "gnorm", "chunks")}) + "`"]
    try:
        gate = json.loads((run / "gate_state.json").read_text()).get("verdicts", {})
        lines += ["", "## Gate, newest verdicts", ""]
        for key in list(gate)[-8:]:
            v = gate[key]
            lines.append(f"* {key}: back-check {(v.get('rollback') or {}).get('verdict')} "
                         f"{(v.get('rollback') or {}).get('diff')} (n {(v.get('rollback') or {}).get('n')}), "
                         f"promotion {(v.get('promotion') or {}).get('verdict')} {(v.get('promotion') or {}).get('diff')} "
                         f"(n {(v.get('promotion') or {}).get('n')})")
    except (OSError, ValueError):
        pass

    usage = shutil.disk_usage(run)
    lines += ["", "## Machine", "", f"* Disk: {usage.free / 2 ** 30:.0f} GB free of {usage.total / 2 ** 30:.0f} GB"]
    try:
        lines.append(f"* Load: {os.getloadavg()[0]:.0f} (1 min), uptime {float(open('/proc/uptime').read().split()[0]) / 86400:.1f} days")
    except (OSError, AttributeError):
        pass
    try:
        lines.append(f"* Newest journal row age: {(time.time() - (run / 'rl.jsonl').stat().st_mtime) / 3600:.1f} hours")
    except OSError:
        pass
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    text = build(a.run, a.home, a.days)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
