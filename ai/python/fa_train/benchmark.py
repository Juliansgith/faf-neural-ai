"""Fixed ordinary-start suites and strict completion accounting (CPU/stdlib only).

Generate a development suite with --repeats 1, or a 640-game confirmation with
--repeats 2. Use different seed bases for development and confirmation. Reusing a
suite across candidates is useful for development, not an independent holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from fa_env.maps import MAP_POOL

WEIGHTED_MAPS = ("SCMP_012", "SCMP_013", "SCMP_016", "X1MP_001", "X1MP_002",
                 "SCMP_019", "SCMP_023", "SCMP_012", "SCMP_013", "SCMP_016")
OUTCOMES = {"victory", "defeat", "timeout", "draw"}


def game_key(row: dict) -> tuple:
    return (row["ai"], row["map"], tuple(row["slots"]), int(row["faction"]),
            int(row["ai_faction"]), int(row["seed"]))


def validate_rows(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("schedule must contain a nonempty list of rows")
    seen = set()
    clean = []
    fields = {"ai", "map", "slots", "faction", "ai_faction", "seed", "minutes"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"schedule rows must have exactly {sorted(fields)}")
        if row["map"] not in MAP_POOL or tuple(row["slots"]) not in MAP_POOL[row["map"]]["slots"]:
            raise ValueError("unknown map or unsupported ordinary start slots")
        if any(type(row[k]) is not int or row[k] not in range(1, 5) for k in ("faction", "ai_faction")):
            raise ValueError("factions must be integers 1..4")
        if type(row["seed"]) is not int or not 0 <= row["seed"] < 2**31:
            raise ValueError("seed must be a nonnegative signed 32-bit integer")
        if not isinstance(row["ai"], str) or not row["ai"]:
            raise ValueError("opponent must be named")
        if not isinstance(row["minutes"], (float, int)) or not math.isfinite(row["minutes"]) or row["minutes"] <= 0:
            raise ValueError("minutes must be finite and positive")
        key = game_key(row)
        if key in seen:
            raise ValueError("duplicate scheduled game")
        seen.add(key)
        clean.append(dict(row, slots=list(row["slots"])))
    return clean


def suite(seed_base: int, repeats: int = 2, ai: str = "m28aicheat@0.9", minutes: float = 40.0) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    # Interleave maps/sides/factions to make interrupted runs visibly broad, but
    # only a completely finished schedule is an accepted benchmark.
    for repeat in range(repeats):
        for own in range(1, 5):
            for enemy in range(1, 5):
                for side in range(2):
                    for map_id in WEIGHTED_MAPS:
                        rows.append(dict(ai=ai, map=map_id, slots=MAP_POOL[map_id]["slots"][side],
                                         faction=own, ai_faction=enemy, seed=seed_base + len(rows), minutes=minutes))
    return validate_rows(rows)


def schedule_hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(validate_rows(rows), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lua_tree_hash(root: Path) -> str:
    """Fingerprint loaded Lua sources, independent of machine-specific root paths."""
    digest = hashlib.sha256()
    # WindowsPath sorts case-insensitively while PosixPath does not. Sort the
    # portable relative strings so identical Windows/Wine opponents match.
    paths = sorted(root.rglob('*.lua'), key=lambda p: p.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f'No Lua sources in {root}')
    for path in paths:
        name, data = path.relative_to(root).as_posix().encode(), path.read_bytes()
        digest.update(len(name).to_bytes(8, 'big') + name + len(data).to_bytes(8, 'big') + data)
    return digest.hexdigest()


def wilson(wins: int, games: int) -> list[float] | None:
    if not games:
        return None
    z = 1.959963984540054
    p, z2 = wins / games, z * z
    centre = (p + z2 / (2 * games)) / (1 + z2 / games)
    half = z * math.sqrt(p * (1 - p) / games + z2 / (4 * games**2)) / (1 + z2 / games)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def counts(games: list[dict]) -> dict:
    outcomes = Counter(g.get("outcome", "missing") for g in games)
    n, wins = len(games), outcomes["victory"]
    return dict(games=n, wins=wins, rate=wins / n if n else None, wilson95=wilson(wins, n), outcomes=dict(outcomes))


def summarize(rows: list[dict], results: list[dict]) -> dict:
    """Errors, duplicates, extras and missing rows invalidate completion. Never
    silently drop failed games and inflate the denominator's apparent quality.
    Wilson is a descriptive binomial interval, not a multiple-testing correction.
    """
    rows = validate_rows(rows)
    expected = {game_key(r) for r in rows}
    found = defaultdict(list)
    malformed = 0
    for result in results:
        try:
            found[game_key(result)].append(result)
        except (KeyError, TypeError, ValueError):
            malformed += 1
    duplicates = sum(max(0, len(v) - 1) for v in found.values())
    extras = len(set(found) - expected)
    missing = len(expected - set(found))
    matched = [r for k in expected for r in found.get(k, [])]
    invalid = sum(r.get("outcome") not in OUTCOMES for r in matched)
    complete = not (malformed or duplicates or extras or missing or invalid)
    groups = {}
    for key in ("ai", "map", "faction", "ai_faction", "slots"):
        bins = defaultdict(list)
        for result in matched:
            bins[str(result[key])].append(result)
        groups[key] = {label: counts(games) for label, games in sorted(bins.items())}
    total = counts(matched)
    lower = total["wilson95"][0] if total["wilson95"] else 0.0
    return dict(schedule_sha256=schedule_hash(rows), expected=len(rows), complete=complete,
                missing=missing, duplicate_results=duplicates, extra_games=extras, malformed=malformed,
                invalid_outcomes=invalid, total=total, strata=groups,
                statistical_threshold_met=bool(complete and len(rows) >= 640 and lower > 0.60),
                note="Threshold alone does not certify holdout independence, opponent identity, or provenance.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    rows = suite(a.seed_base, a.repeats)
    # Refuse to overwrite a frozen suite accidentally.
    with a.out.open("x", encoding="utf-8") as f:
        json.dump({"schema": 1, "rows": rows, "schedule_sha256": schedule_hash(rows)}, f, indent=2)
    print(f"{len(rows)} ordinary-start games; schedule {schedule_hash(rows)}")


if __name__ == "__main__":
    main()
