"""RL from a player machine's games: pull the newest rounds, one PPO iteration on them (fa_train.learn_rl), push the
result, again. The player records each decision's own choice; this side learns from them.

    python -m fa_train.rl_loop --run RUN --init CKPT [--reference REF] [--iterations 0] [--games-per-iteration 120]

RUN/rl.jsonl: one line per iteration (rounds, games, learn_rl's statistics, the checkpoint, the time, the experiment
--tag), per fallback (a "fallback" row: the lineage is cut back to its checkpoint) and per regression warning.
RUN/STOP ends the loop after the current step (and stops a running learn_rl without saving: the GPU watchdog's
brake). RUN/YIELD ends it at the next safe point, never inside a learn_rl (the autopilot's switch). Each iteration:
  1. pull finished rounds (fa_train.server_sync) into RUN/rounds and encode the new ones into RUN/cache;
  2. learn_rl over the newest --games-per-iteration games played by the last --recent-networks networks (V-trace
     corrects for the older ones; games of networks before them are left out);
  3. push the new network (the lanes play it from their next round) and wait until enough new games exist.
--m28-curriculum: one opponent slot of the player is M28 at a resource/build multiplier that follows the newest
networks' results against it (up a step above --m28-band's top, down a step below its bottom): the network always
has an M28 it beats about half the time (RUN/curriculum.json keeps the level).
The reference (KL anchor) stays the starting network unless --reference, --reference-lag or --anchor-champion say
otherwise.

The champion gate (--gate-games): every lineage network after the champion (the best network so far) is judged ONCE
against it, on its first --gate-games matched games: back to the champion when the 95 % upper bound of the
difference is below zero. A network that passed is judged once more, on its first --gate-promote-games, for
promotion (z >= --gate-promote-z and >= --gate-promote-points). Verdicts are kept in RUN/gate_state.json, so a
restart never judges a network again. After --max-rejections fallbacks from the same champion with the same
configuration the loop writes RUN/GATE_EXHAUSTED.json and exits (code 6): the autopilot switches experiments.

Exit codes (also written to RUN/rl_loop_exit.json with the reason, for the supervisor):
  0  STOP, or --iterations reached                  4  YIELD: a safe-point exit asked for by the autopilot
  1  an unexpected error (a traceback)              5  another rl_loop holds RUN/rl_loop.lock
  3  learn_rl failed twice in a row (one retry)     6  the champion gate's rejection cap (RUN/GATE_EXHAUSTED.json)
  7  a STOP file was there at the start (not ours: the GPU watchdog's or the user's; --clear-stop to remove it)
(2 was the regression guard's stop; the guard only warns now.)
Recoverable trouble never ends the loop: ssh/scp failures are retried with backoff and then left pending (pushes and
player configs are retried every pass and on every start), an incomplete round is set aside, a full disk pauses the
intake of new games (--min-free-gb) instead of crashing.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXIT_DONE, EXIT_ERROR, EXIT_LEARN_FAILED, EXIT_YIELD, EXIT_LOCKED, EXIT_GATE_EXHAUSTED, EXIT_STOP_PRESENT = \
    0, 1, 3, 4, 5, 6, 7
EXIT_FILE = "rl_loop_exit.json"
OWN_STOP = "rl_loop"          # a STOP (or local-lane PAUSE) whose content starts with this is the loop's own


def say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M}] {msg}", flush=True)


def atomic_write_json(path: Path, obj) -> None:
    """Written beside, then renamed: a crash or a blue screen never leaves half a state file (a torn champion.json or
    curriculum.json made every later start crash, 2026-10-16 review F17)."""
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_journal(path: Path) -> list:
    """The journal's rows; a line that does not parse (a torn last line after a power loss) is skipped with a
    message instead of crashing every later start."""
    rows = []
    if not path.exists():
        return rows
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            say(f"journal line {i + 1} unreadable (a torn write?): skipped")
    return rows


def append_row(journal: Path, row: dict, tag: str = "") -> None:
    """A journal row with its wall-clock time and the experiment's name (the autopilot's hour-based rules)."""
    row.setdefault("time", datetime.now().isoformat(timespec="seconds"))
    if tag:
        row.setdefault("experiment", tag)
    with journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# sha256 of a checkpoint by (path, size, mtime): every pass hashed every checkpoint several times (12.4 s per
# _names() call with 111 checkpoints, 2-4 min of CPU a pass, growing all run; 2026-10-16 review F13)
_SHA: dict = {}
_SHA_DIRTY = [False]


def sha256_cached(path: Path) -> str:
    st = Path(path).stat()
    key = f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}"
    h = _SHA.get(key)
    if h is None:
        h = _SHA[key] = sha256(path)
        _SHA_DIRTY[0] = True
    return h


def load_sha_cache(path: Path) -> None:
    _SHA.update(read_json(path, {}) or {})


def save_sha_cache(path: Path) -> None:
    if _SHA_DIRTY[0]:
        try:
            atomic_write_json(path, _SHA)
            _SHA_DIRTY[0] = False
        except OSError:
            pass


# a finished round never changes: its round.json and episodes are read once (by file time), not every pass
_META: dict = {}
_EPISODES: dict = {}
_ENVIRONMENTS: dict = {}
GATE_SCHEMA = "engine-runtime-v5"
EPISODE_KEYS = ("ai", "faction", "ai_faction", "map", "slots", "outcome", "seed")


def round_meta(r: Path) -> dict:
    p = r / "round.json"
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        return {}
    hit = _META.get(str(p))
    if hit and hit[0] == stamp:
        return hit[1]
    meta = read_json(p, None)
    if not isinstance(meta, dict):
        return {}
    _META[str(p)] = (stamp, meta)
    return meta


def episodes(r: Path) -> list | None:
    """The round's games from eval/episodes.jsonl (only the fields the loop reads); None without the file."""
    p = r / "eval" / "episodes.jsonl"
    try:
        st = p.stat()
    except OSError:
        return None
    key = (st.st_size, st.st_mtime_ns)
    hit = _EPISODES.get(str(p))
    if hit and hit[0] == key:
        return hit[1]
    out = []
    try:
        lines = p.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        out.append({k: e.get(k) for k in EPISODE_KEYS})
    _EPISODES[str(p)] = (key, out)
    return out


def round_stamp(r: Path) -> str:
    return r.name.split("_", 1)[-1]


def done_rounds(rounds: Path) -> list:
    """The finished rounds (DONE), oldest first by their start time (by name, lane10_ sorted between lane1_ and
    lane2_)."""
    return sorted((r for r in rounds.glob("lane*_*") if (r / "DONE").exists()), key=round_stamp)


def round_raw_dir(r: Path) -> Path | None:
    """A round's recordings: rounds/<r>/raw, or where archive_old moved them (RAW_ARCHIVED). None when gone."""
    if (r / "raw").is_dir():
        return r / "raw"
    note = r / "RAW_ARCHIVED"
    if note.exists():
        try:
            p = Path(note.read_text().strip())
        except OSError:
            return None
        return p if p.is_dir() else None
    return None


def round_games(r: Path) -> int:
    meta = round_meta(r)
    if isinstance(meta.get("games"), int):
        return meta["games"]
    raw = round_raw_dir(r)
    return len(list(raw.glob("*.jsonl.gz"))) if raw else 0


def arrived_at(r: Path) -> float:
    """When a round reached this machine: its ARRIVED stamp (server_sync.pull, local_collect), else the folder's
    creation time (Windows: extraction and a same-disk move keep the player's modification time, not the creation
    time), else its modification time."""
    try:
        if (r / "ARRIVED").exists():
            return (r / "ARRIVED").stat().st_mtime
        st = r.stat()
        return getattr(st, "st_birthtime", None) or (st.st_ctime if sys.platform == "win32" else st.st_mtime)
    except OSError:
        return time.time()


LARGE_KM = 10        # maps above this size count under their own keys ("rush#20km")


@functools.lru_cache(maxsize=None)
def _map_km(map_id: str) -> int:
    try:
        from fa_env.maps import MAP_POOL
        return int(MAP_POOL.get(map_id, {}).get("km", 5))
    except Exception:                                   # noqa: BLE001
        return 5


def win_rates_by_ai(rounds: Path, names: dict, by_faction: bool = False) -> dict:
    """network name -> {opponent: [wins, games]} over the finished rounds it played. Games on maps larger than
    LARGE_KM count under "<opponent>#<km>km": the starting network played small maps only, and a large map's
    results compared with its small-map rate would read as a regression (2026-10-11, large maps join)."""
    out = {}
    for r in done_rounds(rounds):
        meta = round_meta(r)
        net = names.get(meta.get("checkpoint_sha256"))
        if net is None:
            continue
        eps = episodes(r)
        if eps is not None:
            for e in eps:
                km = _map_km(e.get("map") or "")
                key = e["ai"] if km <= LARGE_KM else f"{e['ai']}#{km}km"
                if e.get("map") in NEW_MAPS:               # maps being learnt (2026-11-11: 25 % wins against the
                    key = f"{e['ai']}#new"                  # easy AIs there is no collapse and no regression)
                if by_faction:                  # the regression guard: our faction apart (a mix change is no regression)
                    key += f"|f{e.get('faction')}"
                w = out.setdefault(net, {}).setdefault(key, [0, 0])
                w[0] += int(e.get("outcome") == "victory")
                w[1] += 1
            continue
        for ai, v in (meta.get("outcomes") or {}).items():
            w = out.setdefault(net, {}).setdefault(ai, [0, 0])
            w[0] += int((v.get("outcomes") or {}).get("victory", 0))
            w[1] += int(v.get("games", 0))
    return out


def win_rates(rounds: Path, names: dict) -> dict:
    """network name -> [wins, games] over the finished rounds it played (all opponents but M28 pooled)."""
    out = {}
    for net, by_ai in win_rates_by_ai(rounds, names).items():
        for ai, (w, g) in by_ai.items():
            if ai.startswith("m28"):                  # (all losses for now: they say nothing about a change)
                continue
            t = out.setdefault(net, [0, 0])
            t[0] += w
            t[1] += g
    return out


def matched(by_ai: dict, net: str, min_base_games: int = 20):
    """(net's wins, games, the starting network's rate on the same opponent mix) over the opponents both played
    (M28 aside): the player's mix of opponents changes, and a harder mix is no regression. None without games."""
    w = g = 0
    base = 0.0
    for ai, (wi, gi) in by_ai.get(net, {}).items():
        sw, sg = by_ai.get("start", {}).get(ai, (0, 0))
        if ai.startswith("m28") or sg < min_base_games or not gi:
            continue
        w, g = w + wi, g + gi
        base += gi * sw / sg
    return (w, g, base / g) if g else None


def _names(start: Path, ckpts: Path) -> dict:
    names = {sha256_cached(start): "start"}
    for p in sorted(ckpts.glob("rl_*.pt")):
        names[sha256_cached(p)] = p.stem
    return names


def newest_rate(rounds: Path, start: Path, ckpts: Path, min_games: int = 100, order: list | None = None):
    """(name, win rate, games, the starting network's rate on the same opponents) of the newest RL network with at
    least `min_games` games on the player; None when there is none yet. `order`: the lineage's network names,
    oldest first (by name, an abandoned rl_097 set the imitation weight after the fallback to rl_089)."""
    by_ai = win_rates_by_ai(rounds, _names(start, ckpts))
    ready = []
    for net in by_ai:
        m = matched(by_ai, net) if net != "start" else None
        if m and m[1] >= min_games:
            ready.append((net, m))
    if not ready:
        return None
    if order is not None:
        rank = {n: i for i, n in enumerate(order)}
        ready = [r for r in ready if r[0] in rank]
        if not ready:
            return None
        net, (w, g, base) = max(ready, key=lambda r: rank[r[0]])
    else:
        net, (w, g, base) = max(ready)
    return net, w / g, g, base


def m28_rate(rounds: Path, start: Path, ckpts: Path) -> str:
    """Each network's results against M28, per M28 variant (its cheat multiplier), for the log."""
    by_ai = win_rates_by_ai(rounds, _names(start, ckpts))
    parts = []
    for ai in sorted({ai for v in by_ai.values() for ai in v if ai.startswith("m28")}):
        nets = [f"{net} {by_ai[net][ai][0]}/{by_ai[net][ai][1]}" for net in sorted(by_ai) if ai in by_ai[net]]
        parts.append(f"{ai}: " + ", ".join(nets[-4:]))     # the newest few networks (and the start)
    return " | ".join(parts)


def regression(rounds: Path, start: Path, ckpts: Path, points: float, min_games: int, order: list | None = None):
    """The newest networks, pooled back until `min_games` games, playing clearly worse than the starting one on
    the player (the rl1 lesson: an entropy bonus ~100x too strong took the win rate from 80 % to 64 % in two
    iterations, seen only by hand). Compared on the same opponents: the starting network's rate is re-weighted to
    the networks' own mix. Returns None, or (what happened, the network to go back to: the newest one measured at
    least as good as the starting network over 100+ games, else None for the starting network).
    Pooled newest, not each network alone: one network's noisy 144 games (rl_038: 68 % +-8 against 76 %, the next
    one 78 %) are not a trend, and going back to the starting network threw away every iteration since.
    `order`: the lineage's network names, oldest first (abandoned branches are not the newest networks: by name,
    the restart after a fallback to rl_089 pooled rl_091+092+096 and fell back again, 2026-10-13).
    Since 2026-10-17 the loop only WARNS on it (journal + log): the champion gate decides, and the guard's stop
    ended the run with nothing to restart it (2026-10-16 review F3)."""
    # compared per opponent and our faction: since 2026-10-14 the factions vary (they were 67 % UEF, the start's
    # games too), and Cybran or Seraphim games where the start played none would read as a regression
    by_ai = win_rates_by_ai(rounds, _names(start, ckpts), by_faction=True)
    if sum(g for ai, (w, g) in by_ai.get("start", {}).items() if not ai.startswith("m28")) < min_games:
        return None
    nets = [n for n in order if n in by_ai] if order is not None else sorted(n for n in by_ai if n != "start")
    w = g = 0
    base_sum = 0.0
    used = []
    for net in reversed(nets):
        m = matched(by_ai, net)
        if not m:
            continue
        w, g, base_sum = w + m[0], g + m[1], base_sum + m[2] * m[1]
        used.append(net)
        if g >= min_games:
            break
    # (not g: --guard-games 0 with no matched network divided by zero)
    if not g or g < min_games or w / g >= base_sum / g - points / 100:
        return None
    fallback = None
    for net in reversed(nets):
        m = matched(by_ai, net)
        if m and m[1] >= 100 and m[0] / m[1] >= m[2]:
            fallback = ckpts / f"{net}.pt"
            break
    return (f"{'+'.join(reversed(used))} win {w}/{g} ({w / g:.0%}) against the starting network's "
            f"{base_sum / g:.0%} on the same opponents"), fallback


def collapsed(rounds: Path, start: Path, ckpts: Path, order: list, floor: float, min_games: int,
              with_net: bool = False):
    """The lineage's newest network with `min_games` games against the standard AIs on small maps (no M28, no
    self-play, maps up to LARGE_KM) winning under `floor`: broken, not noise (a healthy network wins 75-90 % there;
    under 40 % of 20 games is ~1e-5 for it). Returns None, or (what happened, the newest earlier lineage network
    measured at or above the floor, or None when there is none) -- with `with_net` also the broken network's name
    first. The regression guard pools networks and takes ~140 games; 2026-10-14 rl_115 all but stopped building
    economy (mass income 18 against 40-60, 66 units against ~185) and won 9 of 54 such games, pooled with its
    healthy neighbours it was not a trend."""
    by_ai = win_rates_by_ai(rounds, _names(start, ckpts))
    measured = []
    for net in reversed(order):
        w = g = 0
        for ai, (wi, gi) in by_ai.get(net, {}).items():
            if ai.startswith(("m28", "self@")) or "#" in ai:
                continue
            w, g = w + wi, g + gi
        if g >= max(1, min_games):
            measured.append((net, w, g))
    if not measured or measured[0][1] / measured[0][2] >= floor:
        return None
    net, w, g = measured[0]
    back = next((n for n, wi, gi in measured[1:] if wi / gi >= floor), None)
    what = f"{net} won {w}/{g} ({w / g:.0%}) against the standard AIs on small maps (floor {floor:.0%})"
    return (net, what, back) if with_net else (what, back)


def _stratified(ref: dict, cand: dict) -> tuple[float, float, int] | None:
    """Mantel-Haenszel difference (cand - ref) of win rates over the conditions both played, its standard error,
    and the candidate's games counted. ref/cand: condition -> [wins, games]."""
    num = den = var = 0.0
    n_cand = 0
    for cond, (wc, nc) in cand.items():
        wr, nr = ref.get(cond, (0, 0))
        if not nc or not nr:
            continue
        w = nc * nr / (nc + nr)
        num += w * (wc / nc - wr / nr)
        den += w
        p = (wc + wr) / (nc + nr)
        # Map/start/engine matching creates small cells. The pooled Bernoulli
        # variance is biased low by (N-1)/N; correct it before summing cells.
        # With one game on each side this matters by a factor of two.
        var += w * w * p * (1 - p) * (1 / nc + 1 / nr) * (nc + nr) / (nc + nr - 1)
        n_cand += nc
    if den == 0:
        return None
    return num / den, math.sqrt(var) / den, n_cand


_ACTING_TRAIN_SOURCES = frozenset({
    "fa_train/tools_model.py", "fa_train/tools_trainer.py", "fa_train/trainer.py",
    "fa_train/worker.py", "fa_train/tools_cadence.py", "fa_train/checkpoint_contract.py",
    "fa_train/tools_distillation.py",
})


# Run-level declarations, read from <run>/acting_equivalence.json when the loop starts (2026-11-10): source revisions
# a person checked to leave play unchanged (speed, logging, a head acting never reads), so the champion's games
# still match a candidate's after such a deploy. Without them every deploy of an acting file started an environment
# the champion had never played in, and no gate look could happen (rl_004: 0 matched games in 3 h).
# {"sources": {"fa_env/bridge.py": {"<recorded digest>": "<digest it counts as>", ...}, ...},
#  "gate_exclude_maps": ["SCMP_017", ...],   # maps added after the champion played: kept out of the gate
#  "notes": [...]}
ACTING_ALIASES: dict = {}
GATE_EXCLUDE_MAPS: frozenset = frozenset()
NEW_MAPS: frozenset = frozenset()     # "new_maps" (absent: the gate's excluded maps): the guards count them apart


def load_acting_equivalence(path: Path) -> str | None:
    """Set ACTING_ALIASES and GATE_EXCLUDE_MAPS from `path` (absent: none). Returns a summary, or None."""
    global ACTING_ALIASES, GATE_EXCLUDE_MAPS, NEW_MAPS
    data = read_json(path, None)
    if not isinstance(data, dict):
        return None
    sources = data.get("sources") or {}
    ACTING_ALIASES = {str(k): {str(d): str(c) for d, c in (v or {}).items()} for k, v in sources.items()}
    GATE_EXCLUDE_MAPS = frozenset(str(m) for m in data.get("gate_exclude_maps") or ())
    # 2026-11-11: with a control lane the champion plays the new maps too and the gate may count them, while
    # the collapse and regression guards still read them apart ("<ai>#new")
    NEW_MAPS = frozenset(str(m) for m in (data["new_maps"] if "new_maps" in data else GATE_EXCLUDE_MAPS) or ())
    _ENVIRONMENTS.clear()
    return (f"{sum(len(v) for v in ACTING_ALIASES.values())} source revisions declared equal to earlier ones, "
            f"{len(GATE_EXCLUDE_MAPS)} maps kept out of the gate")


_EQUIVALENCE_STAMP: list = [None]


def refresh_acting_equivalence(path: Path) -> str | None:
    """load_acting_equivalence again when the file changed (a deploy adds its revisions without a loop restart).
    Returns the summary when it (re)loaded, else None."""
    try:
        st = path.stat()
        stamp = (st.st_size, st.st_mtime_ns)
    except OSError:
        stamp = None
    if stamp == _EQUIVALENCE_STAMP[0]:
        return None
    _EQUIVALENCE_STAMP[0] = stamp
    if stamp is None:
        global ACTING_ALIASES, GATE_EXCLUDE_MAPS, NEW_MAPS
        ACTING_ALIASES, GATE_EXCLUDE_MAPS, NEW_MAPS = {}, frozenset(), frozenset()
        _ENVIRONMENTS.clear()
        return "none (file absent)"
    return load_acting_equivalence(path)


@functools.lru_cache(maxsize=None)
def _source_newline_aliases(path: Path) -> dict:
    """Recognize only byte-verified newline variants of an available source.

    A recorded digest is never replaced just because its old path now contains
    another version. It must hash exactly to these known bytes, their LF form,
    or their CRLF form. Unknown historical revisions remain distinct. The
    cached evidence is fixed for this gate process's lifetime.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    canonical = hashlib.sha256(lf).hexdigest()
    return {hashlib.sha256(data).hexdigest(): canonical
            for data in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def acting_source_key(manifest: dict) -> tuple:
    """Actor/environment sources; learner and reporting edits do not change this key.

    Exact source matching is deliberately conservative: even a comment-only
    revision may need new controls. No behavior-equivalence inference is made.
    """
    sources = manifest.get("policy_source_sha256") or {}
    selected = []
    for name, digest in sources.items():
        if not isinstance(name, str) or ".." in name.split("/"):
            continue
        if not (name.startswith("fa_env/") or name in _ACTING_TRAIN_SOURCES):
            continue
        digest = ACTING_ALIASES.get(name, {}).get(digest, digest)
        aliases = _source_newline_aliases(ROOT / name)
        selected.append((name, aliases.get(digest, digest)))
    return tuple(sorted(selected))


def round_environment(r: Path) -> tuple:
    """Immutable evaluator provenance, shared by the gate and experiment judge.

    Unknown historical provenance matches only other unknown provenance. Never
    infer a historical engine hash from the current contents of its old path.
    """
    p = r / "eval" / "eval_manifest.json"
    try:
        stat = p.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (None, (), None, None, None, None, None, ())
    hit = _ENVIRONMENTS.get(str(p))
    if hit and hit[0] == stamp:
        return hit[1]
    manifest = read_json(p, {}) or {}
    game_sources = manifest.get("game_sources") or {}
    mods = game_sources.get("mods") or {}
    key = (manifest.get("engine_sha256"),
           tuple(sorted((name, value.get("lua_sha256")) for name, value in mods.items())),
           manifest.get("policy"), manifest.get("build_budget"), manifest.get("commander_greedy"),
           game_sources.get("bridge_sha256"), game_sources.get("hooks_sha256"), acting_source_key(manifest))
    _ENVIRONMENTS[str(p)] = (stamp, key)
    return key


def gate_condition(e: dict, environment: tuple) -> tuple:
    """Opponent, factions and the engine/opponent/acting environment. Map and starting slots pool (v5, 2026-10-26):
    every network plays the same seeded schedule, so they balance in expectation, and as cells (v4) they left half
    of a candidate's games without a champion game in the same cell (rl_233: 352 of 728 matched, a network took
    ~3.5 h to reach its 400-game look instead of ~75 min)."""
    return (str(e.get("ai") or ""), e.get("faction"), e.get("ai_faction"), environment)


def ordered_gate_episodes(eps: list) -> list | None:
    """Fixed pseudo-random order within a completed round, independent of finish time.

    Episodes are logged as games finish. Taking that prefix can manufacture a
    gain when wins finish sooner, even with identical full-round outcomes. Use
    only scheduled identity fields, not outcomes, duration, or file position.
    Hash ordering also avoids the schedule's opponent-grouped prefix. An absent
    or duplicate identity cannot supply a reproducible fixed look: omit the round.
    This does not certify independence of whole-round arrival or repeated models;
    final performance still needs a separately reserved complete evaluation.
    """
    ordered, seen = [], set()
    fields = ("ai", "faction", "ai_faction", "map", "slots", "seed")
    for e in eps:
        if not isinstance(e.get("seed"), int) or isinstance(e["seed"], bool):
            return None
        identity = json.dumps({k: e.get(k) for k in fields}, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            return None
        seen.add(identity)
        rank = hashlib.sha256(("faf-gate-order-v1:" + identity).encode("utf-8")).digest()
        ordered.append((rank, identity, e))
    return [e for _, _, e in sorted(ordered, key=lambda row: (row[0], row[1]))]


def gate_sequences(rounds: Path, names: dict, since: str) -> dict:
    """network -> [(condition, won)] of its gate-usable games (rounds by start time,
    games by fixed identity hash): small maps, the standard AIs and M28 (self-play aside: its opponents change), condition =
    gate_condition (opponent, factions, engine/opponent/acting settings).
    Rounds from `since` on; source changes cannot masquerade as policy gains."""
    out: dict = {}
    for r in done_rounds(rounds):
        if round_stamp(r) < since:
            continue
        net = names.get(round_meta(r).get("checkpoint_sha256"))
        eps = episodes(r)
        if net is None or eps is None:
            continue
        eps = ordered_gate_episodes(eps)
        if eps is None:
            continue
        environment = round_environment(r)
        seq = out.setdefault(net, [])
        for e in eps:
            ai = str(e.get("ai") or "")
            if ai.startswith("self@") or _map_km(e.get("map") or "") > LARGE_KM or e.get("map") in GATE_EXCLUDE_MAPS:
                continue
            seq.append((gate_condition(e, environment), int(e.get("outcome") == "victory")))
    return out


def cells_of(seq: list, matched_to: dict | None = None, limit: int | None = None) -> tuple[dict, int]:
    """condition -> [wins, games] of a game sequence: with `matched_to`, only conditions it has games in; with
    `limit`, only the first `limit` (matched) games -- the fixed sample a network is judged on."""
    cells: dict = {}
    n = 0
    for cond, won in seq:
        if matched_to is not None and not matched_to.get(cond, (0, 0))[1]:
            continue
        if limit is not None and n >= limit:
            break
        c = cells.setdefault(cond, [0, 0])
        c[0] += won
        c[1] += 1
        n += 1
    return cells, n


def candidate_games_pending(seqs: dict, champion: str, candidate: str, minimum: int) -> int:
    """Keep a candidate on the players until it can receive its fixed gate look.
    Self-play/large maps are already excluded by gate_sequences; unmatched games
    cannot satisfy the threshold. The champion itself needs no candidate test.
    """
    if minimum <= 0 or candidate == champion:
        return 0
    reference, _ = cells_of(seqs.get(champion, []))
    _, games = cells_of(seqs.get(candidate, []), matched_to=reference)
    return max(0, minimum - games)


def champion_gate(rounds: Path, start: Path, ckpts: Path, champion: str, order: list, min_games: int,
                  since: str, z: float = 1.96):
    """The first gate (2026-10-14), kept for comparisons and its tests; the loop uses judge_gate since 2026-10-17.
    It re-tested every lineage network after the champion on every pass (every 5 minutes while waiting): with
    repeated looks ~9-11 % of truly equal networks were judged "worse" and ~8-10 % "better" instead of 2.5 % each,
    and its standard error was 10-20 % under the bootstrap's (2026-10-16 review: bugs-global L1, data-analysis 7,
    bugs-loop F12). Returns None, ("worse", what, name to go on from) when the newest measured network is clearly
    worse, or ("better", what, name) for the newest clearly better one."""
    seqs = gate_sequences(rounds, _names(start, ckpts), since)
    cells = {net: cells_of(seq)[0] for net, seq in seqs.items()}
    if champion not in order or champion not in cells:
        return None
    measured = []
    for net in order[order.index(champion) + 1:]:
        res = _stratified(cells[champion], cells.get(net, {}))
        if res and res[2] >= min_games:
            measured.append((net, *res))
    if not measured:
        return None
    say_ = lambda m: (f"{m[0]} {100 * m[1]:+.1f} points vs {champion} (95 % {100 * (m[1] - z * m[2]):+.1f} .. "  # noqa: E731
                      f"{100 * (m[1] + z * m[2]):+.1f}; {m[3]} matched games)")
    better = [m for m in measured if m[1] - z * m[2] > 0]
    newest = measured[-1]
    if newest[1] + z * newest[2] < 0:
        return "worse", say_(newest), better[-1][0] if better else champion
    if better:
        return "better", say_(better[-1]), better[-1][0]
    return None


def gate_verdict_key(net: str, champion: str, schema: str | None = None) -> str:
    """gate_state.json's key of a network's verdicts against a champion (under a comparison schema)."""
    return f"{net}@{champion}" + (f"#{schema}" if schema else "")


GATE_DEFAULTS = {"judge_games": 150, "promote_games": 400, "z_back": 1.96, "z_promote": 2.5,
                 "promote_points": 3.0, "se_inflation": 1.2}


def judge_gate(seqs: dict, champion: str, order: list, state: dict, shas: dict | None = None,
               notes: list | None = None, comparison_schema: str | None = None, **params):
    """The champion gate, judged once per network (2026-10-17). `seqs`: gate_sequences(); `order`: the lineage's
    network names, oldest first (the champion among them); `state`: gate_state.json's dict, where each verdict is
    kept ("verdicts": "<net>@<champion>" -> {"rollback": ..., "promotion": ...}) so that no network is judged twice,
    across restarts too; `shas`: name -> sha256 (a network file replaced under the same name is judged anew).
    Each network after the champion is judged on exactly its first `judge_games` matched games (conditions the
    champion played; the champion's own games all count): back to the champion when the 95 % upper bound of the
    Mantel-Haenszel difference is below zero (z_back 1.96). A network that passed gets one more look, on its first
    `promote_games`: promoted when z >= z_promote (2.5) and the difference >= promote_points (+3). The standard
    error is multiplied by `se_inflation` (1.2: the formula's SE was 10-20 % under the bootstrap SE, rl_124 0.041 vs
    0.049, rl_118 0.031 vs 0.037; bugs-global L1). Data-analysis section 7 (Monte Carlo of this estimator on
    rl_089's real rates): one look at 150 games catches a 10-point drop 85 % of the time with ~3.5 % false
    alarms; z 2.5 at 400+ games and +3 points promotes ~2-3 % of truly equal networks (z 1.96 with repeated looks
    promoted ~9 %).
    Returns None, ("better", what, network) for the newest newly promoted network, or ("worse", what, champion)
    when a lineage network was judged worse and no network after it passed. `notes` gets one line per new
    verdict (for the log)."""
    p = {**GATE_DEFAULTS, **params}
    notes = notes if notes is not None else []
    verdicts = state.setdefault("verdicts", {})
    if champion not in order:
        return None
    champ_cells, _ = cells_of(seqs.get(champion, []))
    if not champ_cells:
        return None
    after = order[order.index(champion) + 1:]

    def verdict_key(net):
        # Preserve historical verdicts without reusing their different matching
        # method. Direct callers with legacy sequences keep the old namespace.
        return gate_verdict_key(net, champion, comparison_schema)

    def measure(seq, limit):
        cells, n = cells_of(seq, champ_cells, limit)
        if n < limit:
            return None
        res = _stratified(champ_cells, cells)
        if res is None:
            return None
        diff, se, n = res
        return diff, se * p["se_inflation"], n

    for net in after:
        key = verdict_key(net)
        v = verdicts.get(key)
        if v is not None and shas and v.get("sha") and shas.get(net) and v["sha"] != shas[net]:
            v = None                                    # another file under the same name
        if v is None:
            v = verdicts[key] = {"net": net, "champion": champion, "sha": (shas or {}).get(net)}
        seq = seqs.get(net, [])
        if "rollback" not in v:
            m = measure(seq, p["judge_games"])
            if m:
                diff, se, n = m
                upper = diff + p["z_back"] * se
                v["rollback"] = {"diff": round(diff, 4), "se": round(se, 4), "n": n, "upper": round(upper, 4),
                                 "verdict": "worse" if upper < 0 else "ok",
                                 "time": datetime.now().isoformat(timespec="seconds")}
                notes.append(f"gate: {net} {100 * diff:+.1f} points vs {champion} (95 % "
                             f"{100 * (diff - p['z_back'] * se):+.1f} .. {100 * upper:+.1f}; first {n} matched "
                             f"games): {'WORSE' if upper < 0 else 'kept'}")
        if (v.get("rollback") or {}).get("verdict") == "ok" and "promotion" not in v:
            m = measure(seq, p["promote_games"])
            if m:
                diff, se, n = m
                z = diff / se if se > 0 else 0.0
                better = z >= p["z_promote"] and 100 * diff >= p["promote_points"]
                # clearly worse on the larger sample: a fallback too (2026-10-26 data review B1: rl_202 z -4.1 and
                # rl_203 z -5.1 were only "not promoted", and the line drifted 2-3 more networks down)
                worse = diff + p["z_back"] * se < 0 and 100 * diff <= -p["promote_points"]
                v["promotion"] = {"diff": round(diff, 4), "se": round(se, 4), "n": n, "z": round(z, 2),
                                  "verdict": "better" if better else "worse" if worse else "not better",
                                  "time": datetime.now().isoformat(timespec="seconds"),
                                  "acted": False}
                notes.append(f"gate: {net} {100 * diff:+.1f} points vs {champion} (z {z:.2f}, first {n} matched "
                             f"games): {'PROMOTED' if better else 'WORSE' if worse else 'not promoted'}")
    def promotion(n):
        return (verdicts.get(verdict_key(n)) or {}).get("promotion") or {}

    promoted = [n for n in after if promotion(n).get("verdict") == "better" and not promotion(n).get("acted")]
    if promoted:
        net = promoted[-1]
        pr = verdicts[verdict_key(net)]["promotion"]
        pr["acted"] = True                              # (a champion set back by hand is not re-promoted from it)
        return "better", (f"{net} {100 * pr['diff']:+.1f} points vs {champion} (z {pr['z']:.2f} >= "
                          f"{p['z_promote']:g}; first {pr['n']} matched games)"), net
    worse_at = None
    for i, net in enumerate(after):
        v = verdicts.get(verdict_key(net)) or {}
        rb = v.get("rollback") or {}
        if rb.get("verdict") == "worse" or (v.get("promotion") or {}).get("verdict") == "worse":
            worse_at = i
        elif rb.get("verdict") == "ok" and worse_at is not None:
            worse_at = None                             # a later network passed: the line recovered
    if worse_at is not None:
        net = after[worse_at]
        v = verdicts[verdict_key(net)]
        if (v.get("rollback") or {}).get("verdict") == "worse":
            rb = v["rollback"]
            return "worse", (f"{net} {100 * rb['diff']:+.1f} points vs {champion} (95 % upper "
                             f"{100 * rb['upper']:+.1f}; first {rb['n']} matched games)"), champion
        pr = v["promotion"]
        return "worse", (f"{net} {100 * pr['diff']:+.1f} points vs {champion} (z {pr['z']:.2f}; first {pr['n']} "
                         f"matched games)"), champion
    return None


def m28_level_rate(rounds: Path, allowed: set, level: float, maps=None) -> tuple[int, int]:
    """(wins, games) against M28 at `level` by the networks in `allowed`, on maps up to LARGE_KM: large maps are
    learned later and their losses would move the level for the wrong reason (2026-10-12: x0.65 -> x0.6 on 40 %
    of small-map games plus 0 of 5 on 20 km maps). `maps`: only games on these (a level with its own maps)."""
    key = f"m28aicheat@{level:g}"
    w = g = 0
    for r in done_rounds(rounds):
        meta = round_meta(r)
        if meta.get("checkpoint_sha256") not in allowed:
            continue
        eps = episodes(r)
        if eps is not None:
            for e in eps:
                if e.get("ai") == key and _map_km(e.get("map") or "") <= LARGE_KM \
                        and (maps is None or e.get("map") in maps):
                    w += int(e.get("outcome") == "victory")
                    g += 1
            continue
        v = (meta.get("outcomes") or {}).get(key)
        if v and maps is None:
            w += int((v.get("outcomes") or {}).get("victory", 0))
            g += int(v.get("games", 0))
    return w, g


STRETCH = 0.15       # the stretch slot: M28 this much above the curriculum level


def opponent_mix(level: float, standard: str, weight: float, league: str = "", league_weight: float = 1.0,
                 full_weight: float = 1.0, stretch_weight: float = 1.0,
                 target_m28: float | None = None, target_weight: float = 0.0, stretch_offset: float = STRETCH,
                 wide_level: float | None = None, wide_weight: float = 0.0) -> str:
    """The player's opponents: the standard AIs, M28 at the curriculum level (weighted), a stretch M28 above it
    and full M28 (to measure). `full_weight` 0 drops full M28: it wins ~98 % of games and tells the gate almost
    nothing (data-analysis 7; research E2: --full-m28-weight 0 --stretch-weight 2 --m28-weight 3). The stretch slot
    (2026-10-12): full M28 rushed our commander in 62 % of games and won 99 % of those, at the curriculum's 0.65
    M28 rushed in 21 %; at about 0.8 it rushes often and can still be beaten, where defending such a rush can be
    learned."""
    stretch = round(min(1.0, level + stretch_offset), 3)
    mix = [*[s for s in standard.split(",") if s], f"m28aicheat@{level:g}:{weight:g}"]
    if stretch < 1.0:
        mix.append(f"m28aicheat@{stretch:g}" + (f":{stretch_weight:g}" if stretch_weight != 1.0 else ""))
    if full_weight > 0:
        mix.append("m28ai" + (f":{full_weight:g}" if full_weight != 1.0 else ""))
    # self-play (2026-10-13): past networks of ours (<player home>/league/<name>.pt) as opponents -- the only ones
    # that play like us and do not stand still while we learn
    mix += [f"self@{n}" + (f":{league_weight:g}" if league_weight != 1.0 else "") for n in league.split(",") if n]
    extra = []
    if wide_level is not None and wide_weight > 0:
        # the wide maps' own level (2026-11-11; m28_ai_maps says which maps): one more slot
        extra.append((f'm28aicheat@{wide_level:g}', wide_weight))
    if target_weight:
        if not target_weight > 0 or target_weight == float('inf') or target_m28 is None or not 0 < target_m28 <= 1:
            raise ValueError('Target M28 needs multiplier in (0,1] and a finite positive weight')
        extra.append((f'm28aicheat@{target_m28:g}', target_weight))
    if extra:
        # A fixed objective remains present when curriculum and stretch move.
        # Merge overlap: duplicate names otherwise give ambiguous round counts.
        weights = {}
        for item in mix:
            name, colon, amount = item.rpartition(':')
            name, amount = (name, float(amount)) if colon else (item, 1.0)
            weights[name] = weights.get(name, 0.0) + amount
        for name, amount in extra:
            weights[name] = weights.get(name, 0.0) + amount
        mix = [name + (f':{amount:g}' if amount != 1 else '') for name, amount in weights.items()]
    return ",".join(mix)


def m28_ai_maps(existing: dict | None, base_mix: str, wide_level: float, small_maps: str, wide_maps: str) -> dict:
    """The player config's "ai_maps" with a wide-map level (2026-11-11). `base_mix`: the opponents without the wide
    slot; its M28 opponents play `small_maps`, M28 at the wide level plays `wide_maps` (both lists when the two share
    a name), and the other opponents' entries of `existing` stay. Every M28 name is written out: tools_eval takes an
    exact name before a prefix, and "m28aicheat@0.5" is a prefix of "m28aicheat@0.55"."""
    out = {str(k): v for k, v in (existing or {}).items() if not str(k).startswith("m28ai")}
    for item in base_mix.split(","):
        name = item.rpartition(":")[0] if ":" in item else item
        if name.startswith("m28ai"):
            out[name] = small_maps
    wide = f"m28aicheat@{wide_level:g}"
    out[wide] = f"{small_maps},{wide_maps}" if wide in out else wide_maps
    return out


def league_push(host: str, home: str, local_home: Path | None, ckpts: Path, names: list) -> None:
    """The self-play opponents' networks into the players' league folders (<home>/league/<name>.pt)."""
    from fa_train.server_sync import SSH_OPTS, ssh
    if not names:
        return
    if host:
        ssh(host, f"mkdir -p {home}/league", timeout=300)
    for name in names:
        src = ckpts / f"{name}.pt"
        if host:
            subprocess.run(["scp", "-q", *SSH_OPTS, str(src), f"{host}:{home}/league/{name}.pt"], check=True,
                           timeout=1800)
        if local_home:
            (local_home / "league").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, local_home / "league" / f"{name}.pt")
    say(f"league: {', '.join(names)}")


RETRYABLE = (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError)


def retry(fn, *args, what: str = "", waits=(30, 120), **kw):
    """(True, result) or (False, None): ssh/scp to the player. A VM reboot or a minute of network loss ended the
    run (push/set_config/league_push used check=True; 2026-10-16 review F2): tried len(waits) + 1 times with
    those pauses (seconds), then the caller leaves it pending and tries again on the next pass."""
    for i in range(len(waits) + 1):
        try:
            return True, fn(*args, **kw)
        except RETRYABLE as exc:
            msg = f"{what or fn.__name__} failed ({type(exc).__name__}: {str(exc)[:200]})"
            if i < len(waits):
                say(f"{msg}; again in {waits[i]} s")
                time.sleep(waits[i])
            else:
                say(f"{msg}; left pending, tried again next pass")
    return False, None


def archive_old(rounds: Path, cache: Path, archive: Path, raw_hours: float, cache_hours: float,
                hold=None, hold_hours: float = 72.0, keep_cache=None) -> tuple[int, int]:
    """Move finished rounds' raw recordings older than `raw_hours` and cache files older than `cache_hours` to
    `archive` (a round keeps its round.json and eval/, and a RAW_ARCHIVED note with the new place). 2026-10-14:
    recordings grow ~7 MB a game, 50-70 GB a day; the PC's system disk filled up and a round transfer failed.
    A round's age counts from its arrival here (ARRIVED, arrived_at): tar keeps the server's folder time, and a round
    pulled more than a day late was archived before it was encoded -- the encoder then found no recordings and
    reported success (2026-10-16 review F6). `hold(round) -> True` keeps a round's recordings here (a round of a
    lineage network not encoded yet), but never longer than `hold_hours` after its arrival. `keep_cache(round name)
    -> True` keeps a cache file here whatever its age: the games training uses (the current networks', the
    champion's and its ancestors', the self-imitation archive's). By age alone, a restart after a stop of more than
    `cache_hours` archived the whole training window -- 2026-10-20 it took 980 freshly re-encoded rl_089 games with
    the stale ones, and the storage budget then deleted them."""
    now = time.time()
    raws = files = 0
    for r in rounds.glob("lane*_*"):
        if not ((r / "raw").is_dir() and (r / "DONE").exists()):
            continue
        age = now - arrived_at(r)
        if age <= raw_hours * 3600 or (hold is not None and hold(r) and age <= hold_hours * 3600):
            continue
        (archive / "rounds_raw" / r.name).mkdir(parents=True, exist_ok=True)
        shutil.move(str(r / "raw"), str(archive / "rounds_raw" / r.name / "raw"))
        (r / "RAW_ARCHIVED").write_text(str(archive / "rounds_raw" / r.name / "raw"))
        raws += 1
    (archive / "cache").mkdir(parents=True, exist_ok=True)
    rounds_of = {}
    if keep_cache is not None:
        from fa_train.tools_bc import cache_index
        rounds_of = {name: Path(str((e.get("manifest") or {}).get("source_root") or "")).parent.name
                     for name, e in cache_index(cache).items()}
    for f in cache.glob("*.npz"):
        if now - f.stat().st_mtime <= cache_hours * 3600:
            continue
        if keep_cache is not None and keep_cache(rounds_of.get(f.name, "")):
            continue
        shutil.move(str(f), str(archive / "cache" / f.name))
        files += 1
    return raws, files


def lineage(init: Path, done: list) -> list:
    """The networks the current one descends from, oldest first, from the journal rows: a fallback to a network
    already in the chain cuts the branch after it (abandoned networks are no KL anchor, no source of the games
    used, no self-imitation archive). 2026-10-13: counted plainly, rl_096 -- trained from the fallback rl_089 --
    took rl_083 as its anchor (rejected, 77 % vs scripted) and lost M28 x0.55 63 % to rl_089's 18 %."""
    chain = [init]
    for row in done:
        if not row.get("checkpoint"):
            continue
        p = Path(row["checkpoint"])
        if row.get("fallback") and p in chain:
            chain = chain[: chain.index(p) + 1]
        else:
            chain.append(p)
    return chain


def success_archive(rounds: Path, nets: set, level: float, limit: int) -> list:
    """[source_root, source_name] of the recordings of won games against the strongest opponents -- M28 at the
    curriculum's level or above, full M28 -- played by the networks in `nets` (sha256), strongest and newest
    first, at most `limit` (for learn_rl --sil). A recording's name is <uid>_<opponent>_<seed>.jsonl.gz.
    The recordings are found where archive_old moved them too (RAW_ARCHIVED): looking only in rounds/<r>/raw, the
    archive held only the last 24 h of wins (2026-10-16 review, bugs-data F6)."""
    picks = []
    for r in done_rounds(rounds):
        if round_meta(r).get("checkpoint_sha256") not in nets:
            continue
        eps = episodes(r)
        raw = round_raw_dir(r)
        if not eps or raw is None:
            continue
        for e in eps:
            ai = str(e.get("ai") or "")
            if e.get("outcome") != "victory" or not ai.startswith("m28"):
                continue
            strength = 1.0 if ai == "m28ai" else float(ai.split("@")[1]) if "@" in ai else 0.0
            if strength < level - 1e-6:
                continue
            for f in raw.glob(f"*_{ai}_{e.get('seed')}.jsonl.gz"):
                picks.append((strength, f.stat().st_mtime, str(f.parent.resolve()), f.name))
    picks.sort(key=lambda p: (-p[0], -p[1]))
    return [[root, name] for _, _, root, name in picks[:limit]]


def ensure_encoded(archive: list, cache: Path, should_stop=None) -> int:
    """Encode the archive's recordings that the cache lacks in its current format (rounds of networks too old
    for the PPO games, or encoded before a format change). Returns how many were encoded."""
    from collections import defaultdict
    from fa_train.tools_bc import _cache_is_current, _encoder_semantics_hash, build_cache, cache_index
    if should_stop is not None and should_stop():
        return 0
    sem = _encoder_semantics_hash()
    have = {((e.get("manifest") or {}).get("source_root"), (e.get("manifest") or {}).get("source_name"))
            for e in cache_index(cache).values() if _cache_is_current(e.get("manifest"), sem) and e.get("bc")}
    missing = defaultdict(list)
    for root, name in archive:
        if (root, name) not in have and (Path(root) / name).exists():
            missing[root].append(Path(root) / name)
    encoded = 0
    for root, files in missing.items():
        if should_stop is not None and should_stop():
            break
        result = build_cache(Path(root), cache, log=lambda *_: None, files=files, should_stop=should_stop)
        encoded += result["games"]
        if result.get("cancelled"):
            break
    return encoded


def stale_training_rounds(rounds: Path, cache: Path, names: set, consumed: set, limit: int) -> list[Path]:
    """Previously consumed rounds needing re-encoding in the newest training window.

    A journal entry proves consumption, not that a current-format cache still
    exists. Count distinct current sources: one SIL game must not certify an
    entire round. Re-encoding does not make these games fresh again.
    """
    from collections import defaultdict
    from fa_train.tools_bc import _cache_is_current, _encoder_semantics_hash, cache_index
    sem, present = _encoder_semantics_hash(), defaultdict(set)
    for entry in cache_index(cache).values():
        manifest = entry.get('manifest') or {}
        if entry.get('bc') and _cache_is_current(manifest, sem):
            name = Path(str(manifest.get('source_root') or '')).parent.name
            present[name].add(manifest.get('source_name'))
    missing, total = [], 0
    for name in sorted(names, key=lambda n: n.split('_', 1)[-1], reverse=True):
        if total >= limit:
            break
        p = rounds / name
        count = round_games(p)
        if count <= 0:
            continue
        if len(present[name]) < count and name in consumed:
            if round_raw_dir(p) is None:
                # An expired recording cannot be reconstructed. Older retained
                # rounds can still fill the window; Games excludes stale caches.
                total += len(present[name])
                continue
            missing.append(p)
        total += count
    return missing


def local_push(home: Path, checkpoint: Path, name: str = "current.pt") -> None:
    """The local player's network (written beside, then renamed: a lane never reads half a file)."""
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / f"incoming_{name}"
    shutil.copyfile(checkpoint, tmp)
    tmp.replace(home / name)


def local_config(home: Path, updates: dict) -> None:
    path = home / "config.json"
    cfg = read_json(path, {}) or {}
    cfg.update(updates)
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / "config.json.new"
    tmp.write_text(json.dumps(cfg, indent=1))
    tmp.replace(path)


def local_collect(home: Path, rounds: Path) -> int:
    """The local player's finished rounds into the rounds folder (moved: a round is collected once), each with
    its ARRIVED stamp."""
    from fa_train.server_sync import mark_arrived
    n = 0
    for r in sorted((home / "rounds").glob("lane*_*")):
        if (r / "DONE").exists() and not (rounds / r.name).exists():
            shutil.move(str(r), str(rounds / r.name))
            mark_arrived(rounds / r.name)
            n += 1
    if n:
        print(f"{n} new local rounds", flush=True)
    return n


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return float("inf")


def disks_low(paths: list, floor_gb: float) -> list:
    """'C:\\ 21 GB' for each disk (of `paths`) under `floor_gb` free."""
    out, seen = [], set()
    for p in paths:
        if p is None:
            continue
        anchor = Path(p).anchor or str(p)
        if anchor in seen:
            continue
        seen.add(anchor)
        free = free_gb(Path(p) if Path(p).exists() else Path(anchor))
        if free < floor_gb:
            out.append(f"{anchor} {free:.0f} GB")
    return out


def run(args: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"\n$ {' '.join(args)}\n")
        out.flush()
        return subprocess.call([sys.executable, "-u", *args], cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)


def learn_history(out: Path):
    """learn_rl's per-epoch statistics from the checkpoint it wrote."""
    import torch
    return torch.load(out, map_location="cpu", weights_only=False).get("learn_rl", {}).get("history")


def unstable_update(history, max_gnorm: float, max_clip: float) -> str | None:
    """Why a learn_rl update looks unstable (its last PPO epoch: mean gradient norm above `max_gnorm` or clip
    fraction above `max_clip`; 0 turns a check off), else None. 2026-10-26 data review T1: of 66 judged networks the
    8 with gnorm > 0.9 all ended >= 5 points under the champion (mean -11), the 58 others +1 on average; both
    lineages that collapsed after rl_188 showed it before a single game."""
    epochs = [h for h in (history or []) if isinstance(h, dict) and "gnorm" in h]
    if not epochs:
        return None
    last = epochs[-1]
    if max_gnorm and float(last.get("gnorm") or 0.0) > max_gnorm:
        return f"gradient norm {float(last['gnorm']):.2f} > {max_gnorm:g}"
    if max_clip and float(last.get("clip") or 0.0) > max_clip:
        return f"clip fraction {float(last['clip']):.3f} > {max_clip:g}"
    return None


class Finish(SystemExit):
    """A deliberate end of the loop with one of the documented exit codes."""


def _lock(run_dir: Path):
    """One loop per run, an OS lock its death releases (a supervisor and a hand restart racing a live loop meant
    two trainings on one GPU, 2026-10-16 review F16). Returns the open lock file, or None when another loop holds
    it."""
    f = open(run_dir / "rl_loop.lock", "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _unlock(f) -> None:
    if f is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    f.close()


def config_key(a) -> tuple[str, dict]:
    """A short hash of what makes one configuration (the gate's rejection cap counts per champion and this)."""
    desc = {"tag": a.tag, "learn": a.learn_args, "games": a.games_per_iteration, "recent": a.recent_networks,
            "reference": str(a.reference or ""), "lag": a.reference_lag, "anchor_champion": a.anchor_champion,
            "sil": a.sil_archive, "league": a.league, "distill": [a.distill, a.adaptive_distill],
            "standard_ai": a.standard_ai, "m28_weights": [a.m28_weight, a.stretch_weight, a.full_m28_weight],
            "min_new": a.min_new_games, "gate_schema": GATE_SCHEMA}
    if getattr(a, 'target_weight', 0):
        desc['target_m28'] = [a.target_m28, a.target_weight]
    if getattr(a, 'stretch_offset', STRETCH) != STRETCH:
        desc['stretch_offset'] = a.stretch_offset
    if getattr(a, 'm28_wide_start', None) is not None:
        desc['m28_wide'] = [a.m28_wide_weight, a.m28_wide_maps, a.m28_maps]
    return hashlib.sha1(json.dumps(desc, sort_keys=True).encode()).hexdigest()[:12], desc


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--sil-archive", type=int, default=0,
                    help="self-imitation: up to this many won games against M28 at the curriculum level or above "
                         "(the last --sil-networks networks' games) go to learn_rl --sil every iteration")
    ap.add_argument("--sil-networks", type=int, default=30)
    ap.add_argument("--learn-alone", action="store_true",
                    help="with --local-home: pause the games while learn_rl trains (see pause_for_learning)")
    ap.add_argument("--drain-minutes", type=float, default=45.0,
                    help="with --learn-alone: how long to wait for running rounds to finish")
    ap.add_argument("--learn-processes-alone", type=int, default=22,
                    help="with --learn-alone: learn_rl --processes when no game is running")
    ap.add_argument("--no-remote", action="store_true",
                    help="learner and player on one machine: only --local-home, nothing over ssh (2026-11-07)")
    ap.add_argument("--local-home", type=Path,
                    help="a player on this machine as well (fa_train.player --home): it gets every new network and "
                         "the M28 curriculum's opponents, and its finished rounds are collected with the server's")
    ap.add_argument("--control-lane", action="store_true",
                    help="the local player plays the champion (<local-home>/champion.pt, player config "
                         "network=champion.pt) instead of the newest network: every candidate is compared with "
                         "champion games played at the same time (data-analysis 7: the champion's own games put a "
                         "floor under the gate's error)")
    ap.add_argument("--reference-lag", type=int, default=0,
                    help="the KL anchor follows the networks: the one this many iterations back (0: --reference, "
                         "fixed). Fixed at the starting network -- M28's imitation -- it pulled every iteration back "
                         "towards M28 harder the further the policy moved (ref_kl 0.006 -> 0.035), while beating M28 "
                         "needs a policy unlike it")
    ap.add_argument("--anchor-champion", action="store_true",
                    help="the KL anchor is the champion (over --reference/--reference-lag): with the lag counted along "
                         "the lineage, the anchor after a fallback to rl_089 was rl_070, older than the champion and "
                         "its fine head (2026-10-16 review F15, research E3)")
    ap.add_argument("--host", default="my-server")
    ap.add_argument("--iterations", type=int, default=0, help="0: until RUN/STOP")
    ap.add_argument("--games-per-iteration", type=int, default=120)
    ap.add_argument("--recent-networks", type=int, default=2,
                    help="learn from games of this many newest networks (2: the current one and the one before)")
    ap.add_argument("--m28-curriculum", action="store_true", help="adapt the player's M28 opponent (see above)")
    ap.add_argument("--m28-start", type=float, default=0.6, help="the curriculum's first M28 multiplier")
    ap.add_argument("--m28-band", default="0.4,0.6", help="win rates below/above which the level moves")
    ap.add_argument("--m28-step", type=float, default=0.05)
    ap.add_argument("--m28-min-games", type=int, default=36, help="games at a level before it may move")
    ap.add_argument("--m28-weight", type=float, default=2.0, help="the curriculum slot's games per standard AI's")
    ap.add_argument("--stretch-weight", type=float, default=1.0, help="the stretch M28 slot's weight")
    ap.add_argument("--m28-wide-start", type=float,
                    help="a second curriculum level, for --m28-wide-maps (2026-11-11: 10 km maps were 1-15 %% wins "
                         "against M28 x0.75 where the 5 km maps were 67 %%: the economy stays at tech 1 and the "
                         "commander dies 200-300 from home; games that are almost never won teach little). The "
                         "other M28 slots then play --m28-maps only, and the loop writes the player's ai_maps")
    ap.add_argument("--m28-wide-maps", default="", help="comma list (a map named twice is played twice as often)")
    ap.add_argument("--m28-maps", default="", help="with --m28-wide-start: the maps of the other M28 slots")
    ap.add_argument("--m28-wide-weight", type=float, default=4.0)
    ap.add_argument("--m28-wide-step", type=float, default=0.05)
    ap.add_argument("--m28-wide-min-games", type=int, default=400)
    ap.add_argument("--m28-wide-band", default="0.4,0.6")
    ap.add_argument("--target-m28", type=float, help="fixed M28 multiplier kept alongside the moving curriculum")
    ap.add_argument("--target-weight", type=float, default=0.0, help="additional fixed-target weight (0: disabled)")
    ap.add_argument("--full-m28-weight", type=float, default=1.0, help="full M28's weight in the mix (0: none)")
    ap.add_argument("--standard-ai", default="adaptive,rush,tech")
    ap.add_argument("--archive-dir", type=Path, help="move old raw recordings and cache files here each iteration")
    ap.add_argument("--archive-raw-hours", type=float, default=24.0)
    ap.add_argument("--archive-cache-hours", type=float, default=72.0)
    ap.add_argument("--keep-cache-rounds", type=int, default=24,
                    help="the newest this many rounds of the training window keep their cache whatever its age")
    ap.add_argument("--archive-hold-hours", type=float, default=72.0,
                    help="a lineage round not encoded yet keeps its recordings here up to this long after arrival")
    ap.add_argument("--storage-budget-gb", type=float, default=0.0,
                    help="keep --archive-dir under this many GB (fa_train.storage's policy; 0: off): archived cache, "
                         "abandoned networks' raw games, then the lineage's oldest archived raw games")
    ap.add_argument("--storage-keep-networks", type=int, default=8,
                    help="the newest lineage networks whose raw games go last when over the budget")
    ap.add_argument("--storage-min-age-hours", type=float, default=24.0)
    ap.add_argument("--storage-every-minutes", type=float, default=30.0)
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="under this much free on the run's, the archive's or the local player's disk: no new games "
                         "are taken in (no pull, the local lane paused, no encoding) until there is again")
    ap.add_argument("--encode-backlog-games", type=int, default=0,
                    help="at most this many games of not yet encoded rounds are encoded in one pass, newest first "
                         "(0: twice --games-per-iteration); older ones are passed over (a restart after two days "
                         "found ~10,000 games waiting, ~90 GB of cache)")
    ap.add_argument("--league", default="", help="self-play opponents: past networks' names (rl_NNN), copied to the "
                    "players' league folders at start")
    ap.add_argument("--league-weight", type=float, default=1.0)
    ap.add_argument("--min-new-games", type=int, default=60, help="wait for this many new games between iterations")
    ap.add_argument("--min-candidate-games", type=int, default=0,
                    help="before training the next network, keep the current candidate playing until this many "
                         "gate-matched games exist (0: legacy behavior); early rejection still takes precedence")
    ap.add_argument("--max-candidate-wait-hours", type=float, default=3.0,
                    help="a candidate waits at most this long for its --min-candidate-games (0: no limit)")
    ap.add_argument("--hold-min-points", type=float,
                    help="with --min-candidate-games: hold only a candidate whose 150-game look passed at this many "
                         "points or more (unset: hold every candidate)")
    ap.add_argument("--unstable-gnorm", type=float, default=0.0,
                    help="a new network whose last learn_rl epoch had a mean gradient norm above this is not played: "
                         "the current one plays on and the next iteration trains from it on new games (0: off; "
                         "2026-10-26 data review T1: all 8 judged networks above 0.9 ended >= 5 points worse)")
    ap.add_argument("--unstable-clip", type=float, default=0.0,
                    help="as --unstable-gnorm, for the last epoch's PPO clip fraction (0: off; review: above 0.03, 9 "
                         "of 11 ended >= 5 points worse)")
    ap.add_argument("--unstable-max", type=int, default=2,
                    help="this many unstable networks in a row from the same parent: back to the champion")
    ap.add_argument("--control-lanes", default="",
                    help="lanes of the local player that always play the champion (comma separated lane numbers; "
                         "<local-home>/champion.pt, player config lane_networks). The champion then has games in "
                         "every environment, schedule and M28 level the candidates play in (2026-11-11: a deploy of "
                         "an acting file, new maps or a curriculum step each left the gate without champion games). "
                         "Their rounds are not trained on unless the champion is one of the recent networks")
    ap.add_argument("--control-games", type=int, default=0,
                    help="with --control-lane: the local lane plays the champion only while the champion has fewer "
                         "than this many gate games in the local lane's environment, then the current network "
                         "(0: always the champion; review I1: the champion took 43-60 %% of all games)")
    ap.add_argument("--stretch-offset", type=float, default=0.15,
                    help="the stretch M28 slot this much above the curriculum level")
    ap.add_argument("--learn-args", default="", help="more learn_rl options, e.g. '--epochs 2 --distill 0.3'")
    ap.add_argument("--learn-retry-wait", type=float, default=300.0,
                    help="seconds before the one retry of a failed learn_rl (CUDA/host memory)")
    ap.add_argument("--push", action="store_true", help="push each new network to the player")
    ap.add_argument("--tag", default="", help="the experiment's name, written into every journal row (autopilot)")
    ap.add_argument("--start-from", default="",
                    help="'champion' or a network name: unless the lineage already ends there, go back to it first "
                         "(a fallback row) -- the autopilot starts each experiment from the champion")
    ap.add_argument("--clear-stop", action="store_true",
                    help="a STOP file at the start is renamed aside (STOP.<time>) and the loop starts; without it "
                         "the loop does not start (exit 7): a STOP is the GPU watchdog's or the user's")
    ap.add_argument("--guard-points", type=float, default=8.0,
                    help="WARN (journal + log) when the newest networks' win rate on the player falls this many "
                         "points below the starting network's (over --guard-games games); it no longer stops")
    ap.add_argument("--guard-games", type=int, default=140)
    ap.add_argument("--collapse-floor", type=float, default=0.4,
                    help="a network winning under this against the standard AIs on small maps (over "
                         "--collapse-games) is broken: back to the champion (or, with the gate off or the champion "
                         "the broken one, the newest earlier network above the floor), without stopping")
    ap.add_argument("--collapse-games", type=int, default=20, help="0: off")
    ap.add_argument("--champion", help="the INITIAL champion, used only while <run>/champion.json does not exist "
                                       "(default: the network the loop starts from); after that the file is the "
                                       "loop's (a restart must not forget a promotion)")
    ap.add_argument("--set-champion", help="replace the champion in champion.json with this network (on purpose)")
    ap.add_argument("--gate-games", type=int, default=GATE_DEFAULTS["judge_games"],
                    help="each lineage network after the champion is judged once, on its first this many matched "
                         "games: back to the champion when the 95 %% upper bound is below zero (0: gate off)")
    ap.add_argument("--gate-promote-games", type=int, default=GATE_DEFAULTS["promote_games"])
    ap.add_argument("--gate-z", type=float, default=GATE_DEFAULTS["z_back"])
    ap.add_argument("--gate-promote-z", type=float, default=GATE_DEFAULTS["z_promote"])
    ap.add_argument("--gate-promote-points", type=float, default=GATE_DEFAULTS["promote_points"])
    ap.add_argument("--gate-se-inflation", type=float, default=GATE_DEFAULTS["se_inflation"])
    ap.add_argument("--gate-since", default="20261013T2310Z", help="rounds from this stamp on (varied conditions)")
    ap.add_argument("--train-since", default="",
                    help="train only on rounds from this stamp on (<lane>_<UTC start>): after a deploy that changes "
                         "the game or the observations, older games were played with other code -- 2026-10-20: the "
                         "first X1 iteration took 46 pre-deploy games (some by rl_086/087) and its KL to the data read "
                         "17,000 against ~0.02")
    ap.add_argument("--max-rejections", type=int, default=3,
                    help="gate/collapse fallbacks from the same champion with the same configuration before the loop "
                         "exits with code 6 (0: never)")
    ap.add_argument("--reset-rejections", action="store_true",
                    help="forget the rejections counted for the current champion and configuration")
    ap.add_argument("--distill", type=float, default=0.3, help="starting weight of M28's labels in learn_rl")
    ap.add_argument("--distill-min", type=float, default=0.05)
    ap.add_argument("--distill-max", type=float, default=0.6)
    ap.add_argument("--adaptive-distill", type=int, default=1,
                    help="1: the labels' weight follows the newest network's results on the player: x0.8 while it "
                         "wins at least as often as the starting network, x1.5 when it is 3+ points below (Zhao et "
                         "al. 2022, adaptive behaviour-cloning regularisation); 0: fixed --distill")
    return ap.parse_args(argv)


def main(argv=None):
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # finally blocks run (the learning pause is lifted)
    a = parse_args(argv)
    run_dir = a.run.resolve()
    for d in (run_dir / "rounds", run_dir / "cache", run_dir / "checkpoints"):
        d.mkdir(parents=True, exist_ok=True)
    lock = _lock(run_dir)
    exit_file = run_dir / EXIT_FILE
    declared = refresh_acting_equivalence(run_dir / "acting_equivalence.json")
    if declared:
        say(f"acting_equivalence.json: {declared}")
    try:
        if lock is None:
            say(f"another rl_loop runs on {run_dir}")
            raise Finish(EXIT_LOCKED)                 # (its exit file is the other loop's: not written)
        try:
            atomic_write_json(exit_file, {"code": None, "running": True, "pid": os.getpid(),
                                          "started": datetime.now().isoformat(timespec="seconds")})
            _main(a, run_dir)
        except Finish as f:
            _write_exit(exit_file, int(f.code or 0), str(getattr(f, "why", "")))
            raise
        except KeyboardInterrupt:
            _write_exit(exit_file, EXIT_ERROR, "interrupted")
            raise
        except BaseException as exc:                  # noqa: BLE001 -- recorded for the supervisor, then re-raised
            if isinstance(exc, SystemExit):
                _write_exit(exit_file, exc.code if isinstance(exc.code, int) else EXIT_ERROR, str(exc.code))
            else:
                _write_exit(exit_file, EXIT_ERROR, f"{type(exc).__name__}: {exc}",
                            traceback.format_exc()[-2000:])
            raise
    finally:
        _unlock(lock)


def _write_exit(path: Path, code: int, why: str, trace: str = "") -> None:
    try:
        atomic_write_json(path, {"code": code, "why": why, "trace": trace, "pid": os.getpid(),
                                 "time": datetime.now().isoformat(timespec="seconds")})
    except OSError:
        pass


def finish(code: int, why: str):
    say(f"exit {code}: {why}")
    f = Finish(code)
    f.why = why
    raise f


def _main(a, run_dir: Path):
    rounds, cache, ckpts = run_dir / "rounds", run_dir / "cache", run_dir / "checkpoints"
    init = a.init.resolve()
    stop_file, yield_file = run_dir / "STOP", run_dir / "YIELD"
    # a STOP at the start is an instruction, not a leftover (the GPU watchdog's emergency brake, or the user's): it
    # was deleted here, and a supervisor restarting the loop then resumed training on a GPU logging driver errors
    # (2026-10-16 review F16). Only the loop's own STOP goes; --clear-stop (a deliberate start) renames it aside.
    if stop_file.exists():
        content = stop_file.read_text(errors="replace").strip()
        if content.startswith(OWN_STOP):
            stop_file.unlink(missing_ok=True)
        elif a.clear_stop:
            aside = run_dir / f"STOP.{datetime.now():%Y%m%d-%H%M%S}"
            stop_file.rename(aside)
            say(f"STOP found ({content[:80] or 'empty'}): moved to {aside.name} (--clear-stop)")
        else:
            finish(EXIT_STOP_PRESENT, f"STOP present ({content[:80] or 'empty: the GPU watchdog or the user'}); "
                                      f"not starting (remove it, or --clear-stop)")
    journal = run_dir / "rl.jsonl"
    done = read_journal(journal)
    sha_file = run_dir / "sha_cache.json"
    load_sha_cache(sha_file)
    # the newest network an iteration produced (a failed or stopped iteration leaves a row without one)
    last = next((row for row in reversed(done) if row.get("checkpoint")), None)
    current = Path(last["checkpoint"]) if last else init
    reference = (a.reference or a.init).resolve()
    sys.path.insert(0, str(ROOT))
    from fa_train import server_sync
    home = "~/fa/player"
    chain = lineage(init, done)
    shas = {}

    def sha_of(p: Path) -> str:
        if p not in shas:
            shas[p] = sha256_cached(p)
        return shas[p]

    # the champion: champion.json is the loop's once it exists. 2026-10-16 review (B8/F7): rl2.cmd's
    # `--champion rl_089` overrode the file on every start, so any restart forgot a promoted champion
    champ_file = run_dir / "champion.json"
    kept = read_json(champ_file, None)
    if isinstance(kept, dict) and kept.get("name"):
        champion = kept["name"]
        if a.champion and a.champion != champion:
            say(f"champion.json says {champion}: --champion {a.champion} ignored (it seeds a missing file only; "
                f"--set-champion replaces it)")
    else:
        champion = a.champion or current.stem
        say(f"no champion.json: champion {champion} (from {'--champion' if a.champion else 'the current network'})")
        atomic_write_json(champ_file, {"name": champion, "since": datetime.now().isoformat(timespec="seconds"),
                                       "why": "initial"})
    if a.set_champion and a.set_champion != champion:
        say(f"champion {champion} replaced by {a.set_champion} (--set-champion)")
        champion = a.set_champion
        atomic_write_json(champ_file, {"name": champion, "since": datetime.now().isoformat(timespec="seconds"),
                                       "why": "--set-champion"})
    say(f"champion: {champion}")
    if a.gate_games:
        say(f"gate comparison schema: {GATE_SCHEMA}")

    def net_path(name: str) -> Path:
        p = ckpts / f"{name}.pt"
        if p.exists():
            return p
        for q in reversed(chain):
            if q.stem == name:
                return q
        return init if init.stem == name else p

    # the gate's verdicts and rejection counts (gate_state.json): a restart never judges a network again
    gate_file = run_dir / "gate_state.json"
    gate_state = read_json(gate_file, {}) or {}
    gate_state.setdefault("verdicts", {})
    gate_state.setdefault("rejections", {})
    ckey, cdesc = config_key(a)

    def rejection_key() -> str:
        return f"{champion}|{ckey}"

    if a.reset_rejections:
        gate_state["rejections"].pop(rejection_key(), None)
        atomic_write_json(gate_file, gate_state)

    def check_exhausted() -> None:
        n = gate_state["rejections"].get(rejection_key(), 0)
        if a.max_rejections and n >= a.max_rejections:
            atomic_write_json(run_dir / "GATE_EXHAUSTED.json", {
                "champion": champion, "config": ckey, "desc": cdesc, "rejections": n,
                "time": datetime.now().isoformat(timespec="seconds")})
            finish(EXIT_GATE_EXHAUSTED, f"{n} fallbacks from {champion} with this configuration ({ckey}): "
                                        f"exhausted (GATE_EXHAUSTED.json; another configuration, a new champion "
                                        f"or --reset-rejections goes on)")

    check_exhausted()

    # pushes and player configs that failed are retried every pass (and a push of the current network on every
    # start): the players must play what the journal says
    pending = {"push": None, "config": {}, "local_config": {}, "league": False}

    def push_all(ckpt: Path) -> bool:
        ok = True
        if a.push:
            ok, _ = retry(server_sync.push, a.host, home, ckpt, what=f"push of {ckpt.name}")
        if a.local_home:
            try:
                local_push(a.local_home, ckpt)
            except OSError as exc:
                say(f"local push of {ckpt.name} failed ({exc}); left pending")
                ok = False
        pending["push"] = None if ok else ckpt
        return ok

    control_now = [None]               # the local lane's network file while --control-games decides it
    hold_since: dict = {}              # candidate -> when its --min-candidate-games wait began
    hold_expired: set = set()
    env_drift_said = [None]

    def push_champion_local() -> None:
        """--control-lane: the local lane plays the champion."""
        if a.local_home and a.control_lane:
            try:
                local_push(a.local_home, net_path(champion), "champion.pt")
                local_config(a.local_home, {"network": "champion.pt"})
                control_now[0] = "champion.pt"
            except OSError as exc:
                say(f"control lane: champion not copied ({exc})")
        if a.local_home and a.control_lanes:
            lanes = [x.strip() for x in a.control_lanes.split(",") if x.strip()]
            try:
                local_push(a.local_home, net_path(champion), "champion.pt")
                local_config(a.local_home, {"lane_networks": {lane: "champion.pt" for lane in lanes}})
                say(f"control lanes {', '.join(lanes)}: play the champion {champion}")
            except OSError as exc:
                say(f"control lanes: champion not copied ({exc})")

    def configure(updates: dict) -> None:
        pending["config"].update(updates)
        ok = a.no_remote
        if not a.no_remote:
            ok, _ = retry(server_sync.set_config, a.host, home, dict(pending["config"]), what="player config")
        if ok:
            pending["config"] = {}
        if a.local_home:
            try:
                local_config(a.local_home, updates)
            except OSError as exc:
                say(f"local player config not written ({exc})")

    def save_gate() -> None:
        try:
            atomic_write_json(gate_file, gate_state)
        except OSError as exc:
            say(f"gate_state.json not written ({exc})")

    def candidate_hold(seqs: dict) -> int:
        """Games the current network still needs before it is replaced (--min-candidate-games); with
        --hold-min-points only a promising one waits: its 150-game look passed at that many points or more and its
        400-game look is still to come (2026-10-26 data review B2: 13 of 59 networks never got a promotion look,
        rl_187 +9.3 over 312 games among them; holding every network idled the GPU ~3.5 h per network)."""
        if not (a.gate_games and a.min_candidate_games):
            return 0
        if a.hold_min_points is not None:
            v = gate_state.get("verdicts", {}).get(gate_verdict_key(current.stem, champion, GATE_SCHEMA)) or {}
            rb = v.get("rollback") or {}
            if rb.get("verdict") != "ok" or 100 * rb.get("diff", 0) < a.hold_min_points or "promotion" in v:
                return 0
        pending = candidate_games_pending(seqs, champion, current.stem, a.min_candidate_games)
        # a deadline: matched games need champion games in the same environment, and after a deploy on one machine
        # a candidate can get none at all -- the loop then waited forever (code review 2)
        if pending and a.max_candidate_wait_hours:
            since = hold_since.setdefault(current.stem, time.time())
            if time.time() - since > a.max_candidate_wait_hours * 3600:
                if current.stem not in hold_expired:
                    say(f"{current.stem}: still {pending} gate games short after {a.max_candidate_wait_hours:g} h; "
                        f"training goes on")
                    hold_expired.add(current.stem)
                return 0
        return pending

    def recent() -> set:
        return {sha_of(p) for p in chain[-a.recent_networks:] if p.exists()}

    allowed = recent()

    def anchor() -> Path:
        """The KL reference of the next iteration."""
        if a.anchor_champion and net_path(champion).exists():
            return net_path(champion)
        if a.reference_lag > 0 and len(chain) > a.reference_lag:
            return chain[-1 - a.reference_lag]
        return reference

    k = len(done)
    new_pending: list = []                             # encoded rounds not yet counted by an iteration

    def back_to(name: str, why: str, key: str) -> None:
        """In place: the lineage cut back to `name` (a journal fallback row, as lineage() reads it), the players
        given it; the loop goes on (the row counts as an iteration: k = len(done) on a restart). The push comes
        first and the row records whether it went through: a row written before a push that then failed left the
        server on a rejected network, whose games were all thrown away (2026-10-16 review F8)."""
        nonlocal chain, current, allowed, new_pending, k
        back = net_path(name)
        pushed = push_all(back)
        append_row(journal, {"iteration": k + 1, key: why, "checkpoint": str(back), "fallback": True,
                             "pushed": pushed}, a.tag)
        stems = [p.stem for p in chain]
        chain = chain[: stems.index(name) + 1] if name in stems else chain + [back]
        current = back
        allowed = recent()
        new_pending = []
        k += 1

    def reject(name: str, why: str, key: str) -> None:
        """A gate or collapse fallback to the champion: counted against the champion and this configuration."""
        back_to(name, why, key)
        rk = rejection_key()
        gate_state["rejections"][rk] = gate_state["rejections"].get(rk, 0) + 1
        save_gate()
        say(f"{gate_state['rejections'][rk]} fallback(s) from {champion} with this configuration"
            + (f" (cap {a.max_rejections})" if a.max_rejections else ""))
        check_exhausted()

    # every start: the players play the loop's network (it pushed only on an empty journal, and a push that failed
    # half-way left the server on a rejected network after a restart, F8); --start-from goes back first
    started_from = False
    if a.start_from:
        name = champion if a.start_from == "champion" else a.start_from
        if current.stem != name:
            say(f"experiment {a.tag or '?'}: from {name} (was {current.stem})")
            back_to(name, f"start of experiment {a.tag or '?'}", "start")
            started_from = True
    if not started_from:
        push_all(current)
    if a.local_home:
        try:
            local_config(a.local_home, {"network": "champion.pt" if a.control_lane else "current.pt"})
        except OSError as exc:
            say(f"local player config not written ({exc})")
    push_champion_local()
    league = [n for n in a.league.split(",") if n]
    if league:
        ok, _ = retry(league_push, None if a.no_remote else a.host, home, a.local_home, ckpts, league, what="league push")
        pending["league"] = not ok

    state_file = run_dir / "curriculum.json"
    level = (read_json(state_file, {}) or {}).get("m28_level", a.m28_start)
    low, high = (float(x) for x in a.m28_band.split(","))
    wide_on = bool(a.m28_curriculum and a.m28_wide_start is not None)
    if wide_on and not (a.m28_wide_maps and a.m28_maps and a.local_home):
        raise SystemExit("--m28-wide-start needs --m28-wide-maps, --m28-maps and --local-home")
    wide_level = (read_json(state_file, {}) or {}).get("m28_wide_level", a.m28_wide_start) if wide_on else None
    wide_low, wide_high = (float(x) for x in a.m28_wide_band.split(","))
    small_set = frozenset(m for m in a.m28_maps.split(",") if m) if wide_on else None
    wide_set = frozenset(m for m in a.m28_wide_maps.split(",") if m)

    def mix(with_wide: bool = True) -> str:
        return opponent_mix(level, a.standard_ai, a.m28_weight, a.league, a.league_weight, a.full_m28_weight,
                            a.stretch_weight, a.target_m28, a.target_weight, a.stretch_offset,
                            wide_level if with_wide else None, a.m28_wide_weight)

    def set_opponents() -> None:
        updates = {"ai": mix()}
        if wide_on:
            existing = (read_json(a.local_home / "config.json", {}) or {}).get("ai_maps")
            updates["ai_maps"] = m28_ai_maps(existing, mix(False), wide_level, a.m28_maps, a.m28_wide_maps)
        configure(updates)
        atomic_write_json(state_file, {"m28_level": level, **({"m28_wide_level": wide_level} if wide_on else {})})

    if a.m28_curriculum:
        set_opponents()
        say(f"M28 curriculum at x{level:g}" + (f", on the wide maps at x{wide_level:g}" if wide_on else ""))
    distill = next((row["distill"] for row in reversed(done) if "distill" in row), a.distill)
    distill_basis = None
    # the network the imitation weight was last adjusted for, kept across restarts: forgotten, every restart
    # adjusted again for the same network (2026-10-11: 0.112 -> 0.169 -> 0.253, both times for rl_040)
    distill_file = run_dir / "adaptive_distill.json"
    kept = read_json(distill_file, None)
    if isinstance(kept, dict) and "distill" in kept:
        distill, distill_basis = float(kept["distill"]), kept.get("basis")

    # A stopped/failed learner did not save an update. Its intake is still fresh
    # after restart; merely recording the attempt must not consume the games.
    consumed = {n for row in done if row.get("checkpoint") or row.get("unstable") for n in row.get("rounds", [])}
    unstable_run = [0]                 # unstable updates in a row from the current network (--unstable-gnorm/-clip)
    encoded = set(consumed)
    passed_over: set = set()           # rounds not encoded on purpose (the backlog cap, failures): may be archived
    encode_failures: dict = {}
    warned = {row.get("what") for row in done if row.get("warning")}
    gate_off_said = [None]
    intake_paused = [False]
    storage_t = [0.0]
    backlog = a.encode_backlog_games or 2 * a.games_per_iteration
    pause_file = a.local_home / "PAUSE" if a.local_home else None

    LEARNING = f"{OWN_STOP}: learning"

    def running_games() -> int:
        try:
            return int(subprocess.run(["pgrep", "-c", "-f", r"main\.exe"], capture_output=True, text=True).stdout.strip() or 0)
        except (OSError, ValueError):
            return 0

    def pause_for_learning(args: list) -> bool:
        """--learn-alone (2026-11-08): the games wait while the network trains, so the learner has every CPU and the
        games then play only the newest network. PAUSE stops new rounds; the running ones are waited for (their games
        are kept), up to --drain-minutes. The learner then gets --learn-processes-alone workers, or the ones in its
        arguments while games still run (their memory leaves room for fewer)."""
        if not (a.learn_alone and pause_file is not None):
            return False
        try:
            if pause_file.exists() and not pause_file.read_text().startswith(LEARNING):
                say("PAUSE present (not the loop's learning pause): the games are not waited for")
                return False
            pause_file.write_text(f"{LEARNING} since {datetime.now().isoformat(timespec='seconds')} (pid {os.getpid()})")
        except OSError as exc:
            say(f"could not pause the games for learning ({exc})")
            return False
        t0 = time.monotonic()
        while running_games() and time.monotonic() - t0 < a.drain_minutes * 60:
            if stop_file.exists():
                break
            time.sleep(30)
        left = running_games()
        if left == 0 and "--processes" in args:
            i = args.index("--processes")
            args[i + 1] = str(a.learn_processes_alone)
        say(f"games paused for learning: {left} still running after {(time.monotonic() - t0) / 60:.0f} min; "
            f"learner workers {args[args.index('--processes') + 1] if '--processes' in args else '-'}")
        return True

    def resume_after_learning() -> None:
        try:
            if pause_file is not None and pause_file.exists() and pause_file.read_text().startswith(LEARNING):
                pause_file.unlink()
                say("games resume")
        except OSError as exc:
            say(f"could not remove the learning pause ({exc}): remove {pause_file} by hand")

    # a learning pause left by a loop that died (a kill, a reboot): the games must not wait for it forever
    resume_after_learning()

    def set_intake(low_disks: list) -> bool:
        """Under --min-free-gb: no pull, no local games, no encoding; announced once, lifted when space is back."""
        if low_disks and not intake_paused[0]:
            say(f"DISK LOW ({', '.join(low_disks)} free, floor {a.min_free_gb:g} GB): new games paused (no pull, "
                f"local lane paused, no encoding) until there is space again")
            if pause_file is not None:
                try:
                    pause_file.write_text(f"{OWN_STOP}: disk low {', '.join(low_disks)}")
                except OSError:
                    pass
        elif not low_disks and intake_paused[0]:
            say("disk space back above the floor: new games resume")
            if pause_file is not None and pause_file.exists():
                try:
                    if pause_file.read_text().startswith(OWN_STOP):
                        pause_file.unlink()
                except OSError:
                    pass
        intake_paused[0] = bool(low_disks)
        return intake_paused[0]

    while not stop_file.exists() and (not a.iterations or k < a.iterations):
        # the autopilot's safe point: never inside a learn_rl (STOP would throw its work away)
        if yield_file.exists():
            finish(EXIT_YIELD, f"YIELD ({yield_file.read_text(errors='replace').strip()[:80]})")
        log = run_dir / f"rl_{k + 1:03d}.log"
        if pending["push"] is not None:
            push_all(pending["push"])
        if pending["config"]:
            configure({})
        if pending["league"]:
            ok, _ = retry(league_push, None if a.no_remote else a.host, home, a.local_home, ckpts, league, what="league push")
            pending["league"] = not ok
        paused = set_intake(disks_low([run_dir, a.archive_dir, a.local_home], a.min_free_gb))
        if not paused:
            if not a.no_remote:
                try:
                    server_sync.pull(a.host, home, rounds, keep=False, min_free_gb=a.min_free_gb)
                except Exception as exc:                # noqa: BLE001 -- the next pass tries again
                    say(f"pull failed ({type(exc).__name__}: {exc}); next pass")
            if a.local_home:
                try:
                    local_collect(a.local_home, rounds)
                except OSError as exc:
                    say(f"collecting the local rounds failed ({exc}); next pass")
        # Gate eligibility needs completed game metadata, not encoded replay.
        # Keep intake untouched while a candidate still needs its fixed look:
        # a cache-format change otherwise re-encodes hours of archived games
        # before reaching the very gate wait that prevents another update.
        seqs = {}
        if a.gate_games:
            names = _names(init, ckpts)
            if champion not in names.values() and init.stem == champion:
                names[sha256_cached(init)] = champion
            declared = refresh_acting_equivalence(run_dir / "acting_equivalence.json")
            if declared:
                say(f"acting_equivalence.json: {declared}")
            seqs = gate_sequences(rounds, names, a.gate_since)
        server_env = None
        if a.gate_games and seqs:
            # the lanes must play the same code for the champion's local games to match the candidates' server
            # games: since 2026-10-25 21:31 a comment-only difference in the server's tools_model.py split them
            # (code review 1), and the gate lost the control lane's 1,100 champion games without a word
            all_rounds = done_rounds(rounds)
            locals_ = [r for r in all_rounds if r.name.startswith("lane10_")]
            servers = [r for r in all_rounds if not r.name.startswith("lane10_")]
            local_env = round_environment(max(locals_, key=round_stamp)) if locals_ else None
            server_env = round_environment(max(servers, key=round_stamp)) if servers else None
            drift = bool(local_env and server_env and local_env != server_env)
            if drift != bool(env_drift_said[0]):
                say("WARNING: the local lane and the server play different code or engines (newest rounds' "
                    "provenance differs): their games are not compared with each other" if drift else
                    "local lane and server play the same code again")
                env_drift_said[0] = drift
        if a.local_home and a.control_lane and a.control_games and a.gate_games:
            # the local lane plays the champion only until it has --control-games gate games in the environment the
            # candidates are judged in, the server's (review I1: past ~2,000 they add almost nothing, and they took
            # 43-60 % of all games)
            if server_env is not None:
                env = server_env
                have = sum(1 for cond, _ in seqs.get(champion, []) if cond[-1] == env)
                want = "champion.pt" if have < a.control_games else "current.pt"
                if want != control_now[0]:
                    try:
                        local_config(a.local_home, {"network": want})
                        control_now[0] = want
                        say(f"control lane: plays {'the champion' if want == 'champion.pt' else 'the current network'}"
                            f" ({have} champion games in its environment, target {a.control_games})")
                    except OSError as exc:
                        say(f"control lane: config not written ({exc})")
        candidate_waiting = bool(candidate_hold(seqs))
        # encode the allowed networks' new rounds, newest first, at most `backlog` games a pass. Rounds of other
        # networks are NOT marked done: after a fallback in this very pass the champion's rounds become allowed
        # (they were marked "encoded" for good, and ~2 days of rl_089 games would have been lost; F5)
        if not paused and not candidate_waiting:
            todo = [p for p in done_rounds(rounds) if p.name not in encoded and p.name not in passed_over
                    and round_meta(p).get("checkpoint_sha256") in allowed]
            budget = backlog
            gone, older = [], []
            for p in reversed(todo):
                if stop_file.exists() or yield_file.exists():
                    break
                raw = round_raw_dir(p)
                if budget <= 0 or raw is None:
                    passed_over.add(p.name)
                    (gone if raw is None else older).append(p.name)
                    continue
                if run(["-m", "fa_train.encode_parallel", "--raw", str(raw), "--cache", str(cache),
                        "--workers", "12"], log) == 0:
                    new_pending.append(p.name)
                    encoded.add(p.name)
                    budget -= round_games(p)
                else:
                    encode_failures[p.name] = encode_failures.get(p.name, 0) + 1
                    if encode_failures[p.name] >= 3:
                        passed_over.add(p.name)
                        say(f"{p.name}: encoding failed 3 times; passed over (see {log.name})")
            if stop_file.exists() or yield_file.exists():
                # Finish the active CPU encoding, then honor the safe-point
                # request before housekeeping or starting another GPU learner.
                continue
            if older:
                say(f"{len(older)} older rounds of the allowed networks not encoded: more than the newest {backlog} "
                    f"games waiting ({older[-1]} .. {older[0]})")
            if gone:
                say(f"{len(gone)} rounds without their recordings, not encoded: {', '.join(gone[:5])}")
        new = list(new_pending)
        if a.archive_dir and not disks_low([a.archive_dir], a.min_free_gb):
            # after the encoding (it ran before it: a late round's recordings left before they were encoded, F6);
            # a lineage round not encoded yet stays (the champion's and its recent ancestors' after a fallback)
            stems = [p.stem for p in chain]
            back_chain = chain[: stems.index(champion) + 1] if champion in stems else chain
            usable = allowed | {sha_of(p) for p in back_chain[-a.recent_networks:] if p.exists()}

            def hold(r: Path) -> bool:
                return (r.name not in encoded and r.name not in passed_over
                        and round_meta(r).get("checkpoint_sha256") in usable)

            sil_names = set()
            if (run_dir / "sil.json").exists():
                try:
                    sil_names = {Path(root).parent.name for root, _ in json.loads((run_dir / "sil.json").read_text())}
                except (OSError, ValueError, TypeError):
                    pass
            lineage_keep = usable       # the training window: recent networks, the champion and its ancestors before it
            # only the NEWEST of those rounds: the control lane plays the champion around the clock, and keeping all
            # of its rounds grew the cache to 87 GB in a day (2026-10-22) while training reads the newest ~540 games
            window = sorted((p.name for p in done_rounds(rounds)
                             if round_meta(p).get("checkpoint_sha256") in lineage_keep),
                            key=lambda n: n.split("_", 1)[-1])[-a.keep_cache_rounds:]
            window = set(window)

            def keep_cache(name: str) -> bool:
                return bool(name) and (name in sil_names or name in window)
            try:
                moved = archive_old(rounds, cache, a.archive_dir, a.archive_raw_hours, a.archive_cache_hours,
                                    hold=hold, hold_hours=a.archive_hold_hours, keep_cache=keep_cache)
                if any(moved):
                    say(f"archived: {moved[0]} rounds' recordings, {moved[1]} cache files -> {a.archive_dir}")
            except (OSError, shutil.Error) as exc:     # a full disk: logged, never the end of the run (F1)
                say(f"archiving failed ({type(exc).__name__}: {str(exc)[:200]}); next pass")
        if a.storage_budget_gb and (time.time() - storage_t[0] > a.storage_every_minutes * 60 or paused):
            storage_t[0] = time.time()
            try:
                from fa_train import storage
                lineage_shas = {sha_of(p) for p in chain if p.exists()}
                keep = {sha_of(p) for p in chain[-a.storage_keep_networks:] if p.exists()}
                known = set(_names(init, ckpts))
                champ = net_path(champion)
                plan = storage.plan(rounds, a.archive_dir, a.storage_budget_gb, lineage_shas, keep,
                                    sha_of(champ) if champ.exists() else None, known,
                                    storage.sil_rounds(run_dir / "sil.json"), a.storage_min_age_hours)
                if plan:
                    freed = storage.apply(plan, rounds)
                    say(f"storage: {freed / 2 ** 30:.1f} GB freed ({storage.summary(plan)})")
            except Exception as exc:                    # noqa: BLE001 -- housekeeping never ends the run
                say(f"storage cleanup failed ({type(exc).__name__}: {exc})")
        if a.m28_curriculum:
            w, g = m28_level_rate(rounds, allowed, level, small_set)
            if g >= a.m28_min_games and not (low <= w / g <= high):
                old = level
                level = round(min(1.0, max(0.3, level + (a.m28_step if w / g > high else -a.m28_step))), 3)
                if level != old:
                    set_opponents()
                    say(f"M28 curriculum x{old:g} -> x{level:g} (won {w}/{g})")
            if wide_on:
                w, g = m28_level_rate(rounds, allowed, wide_level, wide_set)
                if g >= a.m28_wide_min_games and not (wide_low <= w / g <= wide_high):
                    old = wide_level
                    step = a.m28_wide_step if w / g > wide_high else -a.m28_wide_step
                    wide_level = round(min(1.0, max(0.3, wide_level + step)), 3)
                    if wide_level != old:
                        set_opponents()
                        say(f"M28 curriculum on the wide maps x{old:g} -> x{wide_level:g} (won {w}/{g})")
        order = [p.stem for p in chain[1:]]
        if chain[0].stem == champion:                  # a run started from the champion itself (B4: the gate was off)
            order = [champion] + order
        broke = collapsed(rounds, init, ckpts, [p.stem for p in chain[1:]], a.collapse_floor,
                          a.collapse_games, with_net=True) if a.collapse_games else None
        if broke:
            net, what, older = broke
            # back to the champion, not merely the newest network above the floor: replay B (2026-10-16 review
            # F11) went from the collapsed rl_115 to rl_114, 14 points under the champion
            target = champion if a.gate_games and net != champion and champion in order[:order.index(net)] \
                else older
            if target:
                say(f"COLLAPSE: {what}; going on from {target}")
                reject(target, what, "collapse") if target == champion else back_to(target, what, "collapse")
                continue
            say(f"COLLAPSE: {what}; no earlier network to go back to, going on")
        if a.gate_games:
            off = (f"the champion {champion} is not in the lineage" if champion not in order else
                   f"the champion {champion} has no games since {a.gate_since}" if not seqs.get(champion) else None)
            if off != gate_off_said[0]:
                say(f"GATE OFF: {off}" if off else "gate on")
                gate_off_said[0] = off
            notes: list = []
            gate = judge_gate(seqs, champion, order, gate_state, {p.stem: sha_of(p) for p in chain if p.exists()},
                              notes, judge_games=a.gate_games, promote_games=a.gate_promote_games, z_back=a.gate_z,
                              z_promote=a.gate_promote_z, promote_points=a.gate_promote_points,
                              se_inflation=a.gate_se_inflation, comparison_schema=GATE_SCHEMA)
            for line in notes:
                say(line)
            if notes or gate:
                save_gate()
            if gate:
                kind, what, name = gate
                if kind == "better":
                    say(f"CHAMPION: {name} (was {champion}): {what}")
                    champion = name
                    atomic_write_json(champ_file, {"name": champion,
                                                   "since": datetime.now().isoformat(timespec="seconds"),
                                                   "why": what})
                    push_champion_local()
                    check_exhausted()                  # (a new champion: a fresh count)
                else:
                    say(f"GATE: {what}; going on from {name}")
                    reject(name, what, "gate")
                    continue
        bad = regression(rounds, init, ckpts, a.guard_points, a.guard_games, order=[p.stem for p in chain[1:]])
        if bad and bad[0] not in warned:
            # a warning, not a stop: its stop ended the run with nothing to restart it, and it measured against the
            # imitation start, 10 points under the champion (2026-10-16 review F3)
            say(f"REGRESSION WARNING: {bad[0]} (not stopping: the champion gate decides)")
            append_row(journal, {"iteration": k + 1, "warning": "regression", "what": bad[0]}, a.tag)
            warned.add(bad[0])
            k += 1
        # only rounds training may use count (--train-since): 2026-10-20 397 "new" games were all from before the
        # cutoff, learn_rl got 0 games, failed twice and the loop exited
        n_new = sum(round_games(rounds / n) for n in new if n.split("_", 1)[-1] >= a.train_since)
        if a.gate_games and a.min_candidate_games:
            pending_games = candidate_hold(seqs)
            if pending_games:
                say(f"{current.stem}: waiting for {pending_games} more gate-matched games before replacing it "
                    f"(target {a.min_candidate_games})")
                save_sha_cache(sha_file)
                time.sleep(300)
                continue
        if n_new < a.min_new_games and k > 0:
            say(f"{n_new} new games; waiting for {a.min_new_games}")
            save_sha_cache(sha_file)
            time.sleep(300)
            continue
        if a.adaptive_distill:
            seen = newest_rate(rounds, init, ckpts, order=[p.stem for p in chain[1:]])
            if seen is not None and seen[0] != distill_basis:
                name, rate, games, base = seen
                old = distill
                if rate >= base:
                    distill = max(a.distill_min, distill * 0.8)
                elif rate < base - 0.03:
                    distill = min(a.distill_max, distill * 1.5)
                distill_basis = name
                atomic_write_json(distill_file, {"distill": distill, "basis": name})
                say(f"imitation weight {old:.3f} -> {distill:.3f} ({name} wins {rate:.0%} of {games} games, the "
                    f"starting network {base:.0%} on the same opponents)")
                vs_m28 = m28_rate(rounds, init, ckpts)
                if vs_m28:
                    say(f"against M28: {vs_m28}")
        # a new name: a crash between a push and its journal row left rl_{k+1}.pt played on the server, and the next
        # start wrote over it -- its games then belonged to no network (F8)
        out = ckpts / f"rl_{k + 1:03d}.pt"
        suffix = "b"
        while out.exists():
            out = ckpts / f"rl_{k + 1:03d}{suffix}.pt"
            suffix = chr(ord(suffix) + 1)
        lineage_rounds = sorted(p.name for p in done_rounds(rounds)
                                if round_meta(p).get("checkpoint_sha256") in allowed
                                and p.name.split("_", 1)[-1] >= a.train_since)
        if not lineage_rounds:              # nothing training may use yet: wait, never a learn_rl without games
            say(f"no rounds for training since {a.train_since or 'the start'}; waiting")
            save_sha_cache(sha_file)
            time.sleep(300)
            continue
        stale = stale_training_rounds(rounds, cache, set(lineage_rounds), consumed, a.games_per_iteration + 60)
        if stale:
            say(f"re-encoding {len(stale)} consumed rounds for the current cache format; these are not new games")
        repaired = True
        for old in stale:
            if run(["-m", "fa_train.encode_parallel", "--raw", str(round_raw_dir(old)), "--cache", str(cache),
                    "--workers", "12"], log) != 0:
                repaired = False
                break
            if yield_file.exists() or stop_file.exists():
                repaired = False
                break
        if not repaired:
            # The top of the loop handles STOP/YIELD; a failed encoder retries
            # without launching training on a silently reduced data window.
            if not (yield_file.exists() or stop_file.exists()):
                time.sleep(60)
            continue
        atomic_write_json(run_dir / "train_rounds.json", lineage_rounds)
        sil_args = []
        if a.sil_archive:
            archive = success_archive(rounds, {sha_of(p) for p in chain[-a.sil_networks:] if p.exists()}, level,
                                      a.sil_archive)
            fresh = ensure_encoded(archive, cache,
                                   should_stop=lambda: stop_file.exists() or yield_file.exists())
            if stop_file.exists() or yield_file.exists():
                continue
            atomic_write_json(run_dir / "sil.json", archive)
            sil_args = ["--sil", str(run_dir / "sil.json")]
            say(f"success archive: {len(archive)} won games against M28 x{level:g}+ ({fresh} encoded now)")
        args = ["-m", "fa_train.learn_rl", "--init", str(current), "--out", str(out), "--cache", str(cache),
                "--reference", str(anchor()), "--max-games", str(a.games_per_iteration),
                "--stop-file", str(stop_file), "--rounds-file", str(run_dir / "train_rounds.json")] \
            + a.learn_args.split() + ["--distill", f"{distill:.4f}"] \
            + sil_args
        t0 = time.monotonic()
        paused_here = pause_for_learning(args)
        try:
            code = run(args, log)
        finally:
            if paused_here:
                resume_after_learning()
        retried = False
        if code != 0 and not stop_file.exists():
            # one retry after a pause: CUDA out of memory (rl_088) and host-memory deaths (rl_037, rl_127) were
            # transient; the loop ended on the first one (F2)
            say(f"learn_rl failed (exit {code}, see {log.name}); once more in {a.learn_retry_wait:.0f} s")
            time.sleep(a.learn_retry_wait)
            if not stop_file.exists():
                retried = True
                code = run(args, log)
        row = {"iteration": k + 1, "from": str(current), "rounds": new, "new_games": n_new, "exit": code,
               "distill": round(distill, 4), "config": ckey,
               "minutes": round((time.monotonic() - t0) / 60, 1)}
        if retried:
            row["retried"] = True
        fall_back = None
        if code == 0 and out.exists():
            row["learn_rl"] = learn_history(out)
            shaky = unstable_update(row["learn_rl"], a.unstable_gnorm, a.unstable_clip)
            if shaky:
                # not played: the current network plays on, and the next iteration trains from it on new games
                unstable_run[0] += 1
                row["unstable"] = str(out)
                row["why"] = shaky
                say(f"UNSTABLE: {out.name} {shaky}; not played, {current.stem} plays on "
                    f"({unstable_run[0]} in a row, at {a.unstable_max} back to {champion})")
                if unstable_run[0] >= a.unstable_max and current.stem != champion:
                    fall_back = f"{unstable_run[0]} unstable updates in a row from {current.stem} ({shaky})"
                    unstable_run[0] = 0
            else:
                unstable_run[0] = 0
                row["checkpoint"] = str(out)
                current = out
                chain.append(out)
                allowed = recent()                      # the new network and the ones the lanes still play
                if not yield_file.exists():             # (YIELD: the next experiment pushes its own start)
                    row["pushed"] = push_all(out)
        if a.m28_curriculum:
            row["m28_level"] = level
            if wide_on:
                row["m28_wide_level"] = wide_level
        new_pending = []
        if row.get("checkpoint") or row.get("unstable"):
            consumed.update(new)
        append_row(journal, row, a.tag)
        save_sha_cache(sha_file)
        say(f"iteration {k + 1}: {n_new} new games -> "
            f"{out.name if code == 0 else 'FAILED (see ' + log.name + ')'} in {row['minutes']} min")
        if code != 0:
            if stop_file.exists():
                break
            finish(EXIT_LEARN_FAILED, f"learn_rl failed twice (exit {code}; see {log.name})")
        k += 1
        if fall_back:
            say(f"GATE: {fall_back}; going on from {champion}")
            reject(champion, fall_back, "unstable")
    save_sha_cache(sha_file)
    finish(EXIT_DONE, "STOP" if stop_file.exists() else f"{a.iterations} iterations")


if __name__ == "__main__":
    main()
