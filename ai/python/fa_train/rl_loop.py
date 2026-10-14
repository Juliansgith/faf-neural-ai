"""RL from a player machine's games: pull the newest rounds, one PPO iteration on them (fa_train.learn_rl), push the
result, again. The player records each decision's own choice; this side learns from them.

    python -m fa_train.rl_loop --run RUN --init CKPT [--reference REF] [--iterations 0] [--games-per-iteration 120]

RUN/rl.jsonl: one line per iteration (rounds, games, learn_rl's statistics, the checkpoint). RUN/STOP ends the loop
after the current step. Each iteration:
  1. pull finished rounds (fa_train.server_sync) into RUN/rounds and encode the new ones into RUN/cache;
  2. learn_rl over the newest --games-per-iteration games played by the last --recent-networks networks (V-trace
     corrects for the older ones; games of networks before them are left out);
  3. push the new network (the lanes play it from their next round) and wait until enough new games exist.
--m28-curriculum: one opponent slot of the player is M28 at a resource/build multiplier that follows the newest
networks' results against it (up a step above --m28-band's top, down a step below its bottom): the network always
has an M28 it beats about half the time (RUN/curriculum.json keeps the level).
The reference (KL anchor) stays the starting network unless --reference says otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


LARGE_KM = 10        # maps above this size count under their own keys ("rush#20km")


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
    for r in sorted(rounds.glob("lane*_*")):
        if not (r / "DONE").exists():
            continue
        meta = json.loads((r / "round.json").read_text())
        net = names.get(meta.get("checkpoint_sha256"))
        if net is None:
            continue
        episodes = r / "eval" / "episodes.jsonl"
        if episodes.exists():
            for line in episodes.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                km = _map_km(e.get("map", ""))
                key = e["ai"] if km <= LARGE_KM else f"{e['ai']}#{km}km"
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
    names = {sha256(start): "start"}
    for p in sorted(ckpts.glob("rl_*.pt")):
        names[sha256(p)] = p.stem
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
    the restart after a fallback to rl_089 pooled rl_091+092+096 and fell back again, 2026-10-13)."""
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
    if g < min_games or w / g >= base_sum / g - points / 100:
        return None
    fallback = None
    for net in reversed(nets):
        m = matched(by_ai, net)
        if m and m[1] >= 100 and m[0] / m[1] >= m[2]:
            fallback = ckpts / f"{net}.pt"
            break
    return (f"{'+'.join(reversed(used))} win {w}/{g} ({w / g:.0%}) against the starting network's "
            f"{base_sum / g:.0%} on the same opponents"), fallback


def m28_level_rate(rounds: Path, allowed: set, level: float) -> tuple[int, int]:
    """(wins, games) against M28 at `level` by the networks in `allowed`, on maps up to LARGE_KM: large maps are
    learned later and their losses would move the level for the wrong reason (2026-10-12: x0.65 -> x0.6 on 40 %
    of small-map games plus 0 of 5 on 20 km maps)."""
    key = f"m28aicheat@{level:g}"
    w = g = 0
    for r in rounds.glob("lane*_*"):
        if not (r / "DONE").exists():
            continue
        meta = json.loads((r / "round.json").read_text())
        if meta.get("checkpoint_sha256") not in allowed:
            continue
        episodes = r / "eval" / "episodes.jsonl"
        if episodes.exists():
            for line in episodes.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("ai") == key and _map_km(e.get("map", "")) <= LARGE_KM:
                    w += int(e.get("outcome") == "victory")
                    g += 1
            continue
        v = (meta.get("outcomes") or {}).get(key)
        if v:
            w += int((v.get("outcomes") or {}).get("victory", 0))
            g += int(v.get("games", 0))
    return w, g


STRETCH = 0.15       # the stretch slot: M28 this much above the curriculum level


def opponent_mix(level: float, standard: str, weight: float, league: str = "", league_weight: float = 1.0) -> str:
    """The player's opponents: the standard AIs, M28 at the curriculum level (weighted), a stretch M28 above it
    and full M28 (to measure). The stretch slot (2026-10-12): full M28 rushed our commander in 62 % of games and
    won 99 % of those, at the curriculum's 0.65 M28 rushed in 21 %; at about 0.8 it rushes often and can still be
    beaten, where defending such a rush can be learned."""
    stretch = round(min(1.0, level + STRETCH), 3)
    mix = [*[s for s in standard.split(",") if s], f"m28aicheat@{level:g}:{weight:g}"]
    if stretch < 1.0:
        mix.append(f"m28aicheat@{stretch:g}")
    mix.append("m28ai")
    # self-play (2026-10-13): past networks of ours (<player home>/league/<name>.pt) as opponents -- the only ones
    # that play like us and do not stand still while we learn
    mix += [f"self@{n}" + (f":{league_weight:g}" if league_weight != 1.0 else "") for n in league.split(",") if n]
    return ",".join(mix)


def league_push(host: str, home: str, local_home: Path | None, ckpts: Path, names: list) -> None:
    """The self-play opponents' networks into the players' league folders (<home>/league/<name>.pt)."""
    import shutil
    import subprocess
    from fa_train.server_sync import SSH_OPTS, ssh
    if not names:
        return
    ssh(host, f"mkdir -p {home}/league")
    for name in names:
        src = ckpts / f"{name}.pt"
        subprocess.run(["scp", "-q", *SSH_OPTS, str(src), f"{host}:{home}/league/{name}.pt"], check=True)
        if local_home:
            (local_home / "league").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, local_home / "league" / f"{name}.pt")
    print(f"[{datetime.now():%H:%M}] league: {', '.join(names)}", flush=True)


def archive_old(rounds: Path, cache: Path, archive: Path, raw_hours: float, cache_hours: float) -> tuple[int, int]:
    """Move finished rounds' raw recordings older than `raw_hours` and cache files older than `cache_hours` to
    `archive` (a round keeps its round.json and eval/, and a RAW_ARCHIVED note with the new place). 2026-10-14:
    recordings grow ~7 MB a game, 50-70 GB a day; the PC's system disk filled up and a round transfer failed."""
    import shutil
    import time as _time
    now = _time.time()
    raws = files = 0
    for r in rounds.glob("lane*_*"):
        if (r / "raw").is_dir() and (r / "DONE").exists() and now - r.stat().st_mtime > raw_hours * 3600:
            (archive / "rounds_raw" / r.name).mkdir(parents=True, exist_ok=True)
            shutil.move(str(r / "raw"), str(archive / "rounds_raw" / r.name / "raw"))
            (r / "RAW_ARCHIVED").write_text(str(archive / "rounds_raw" / r.name / "raw"))
            raws += 1
    (archive / "cache").mkdir(parents=True, exist_ok=True)
    for f in cache.glob("*.npz"):
        if now - f.stat().st_mtime > cache_hours * 3600:
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
    first, at most `limit` (for learn_rl --sil). A recording's name is <uid>_<opponent>_<seed>.jsonl.gz."""
    picks = []
    for r in rounds.glob("lane*_*"):
        if not (r / "DONE").exists() or not (r / "eval" / "episodes.jsonl").exists():
            continue
        if json.loads((r / "round.json").read_text()).get("checkpoint_sha256") not in nets:
            continue
        for line in (r / "eval" / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            ai = str(e.get("ai", ""))
            if e.get("outcome") != "victory" or not ai.startswith("m28"):
                continue
            strength = 1.0 if ai == "m28ai" else float(ai.split("@")[1]) if "@" in ai else 0.0
            if strength < level - 1e-6:
                continue
            for f in (r / "raw").glob(f"*_{ai}_{e.get('seed')}.jsonl.gz"):
                picks.append((strength, f.stat().st_mtime, str(f.parent.resolve()), f.name))
    picks.sort(key=lambda p: (-p[0], -p[1]))
    return [[root, name] for _, _, root, name in picks[:limit]]


def ensure_encoded(archive: list, cache: Path) -> int:
    """Encode the archive's recordings that the cache lacks in its current format (rounds of networks too old
    for the PPO games, or encoded before a format change). Returns how many were encoded."""
    from collections import defaultdict
    from fa_train.tools_bc import _cache_is_current, _encoder_semantics_hash, build_cache, cache_index
    sem = _encoder_semantics_hash()
    have = {((e.get("manifest") or {}).get("source_root"), (e.get("manifest") or {}).get("source_name"))
            for e in cache_index(cache).values() if _cache_is_current(e.get("manifest"), sem) and e.get("bc")}
    missing = defaultdict(list)
    for root, name in archive:
        if (root, name) not in have and (Path(root) / name).exists():
            missing[root].append(Path(root) / name)
    for root, files in missing.items():
        build_cache(Path(root), cache, log=lambda *_: None, files=files)
    return sum(len(f) for f in missing.values())


def local_push(home: Path, checkpoint: Path) -> None:
    """The local player's network (written beside, then renamed: a lane never reads half a file)."""
    import shutil
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / "incoming.pt"
    shutil.copyfile(checkpoint, tmp)
    tmp.replace(home / "current.pt")


def local_config(home: Path, updates: dict) -> None:
    path = home / "config.json"
    cfg = json.loads(path.read_text()) if path.exists() else {}
    cfg.update(updates)
    tmp = home / "config.json.new"
    tmp.write_text(json.dumps(cfg, indent=1))
    tmp.replace(path)


def local_collect(home: Path, rounds: Path) -> int:
    """The local player's finished rounds into the rounds folder (moved: a round is collected once)."""
    import shutil
    n = 0
    for r in sorted((home / "rounds").glob("lane*_*")):
        if (r / "DONE").exists() and not (rounds / r.name).exists():
            shutil.move(str(r), str(rounds / r.name))
            n += 1
    if n:
        print(f"{n} new local rounds", flush=True)
    return n


def run(args: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"\n$ {' '.join(args)}\n")
        out.flush()
        return subprocess.call([sys.executable, "-u", *args], cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--sil-archive", type=int, default=0,
                    help="self-imitation: up to this many won games against M28 at the curriculum level or above "
                         "(the last --sil-networks networks' games) go to learn_rl --sil every iteration")
    ap.add_argument("--sil-networks", type=int, default=30)
    ap.add_argument("--local-home", type=Path,
                    help="a player on this machine as well (fa_train.player --home): it gets every new network and "
                         "the M28 curriculum's opponents, and its finished rounds are collected with the server's")
    ap.add_argument("--reference-lag", type=int, default=0,
                    help="the KL anchor follows the networks: the one this many iterations back (0: --reference, "
                         "fixed). Fixed at the starting network -- M28's imitation -- it pulled every iteration back "
                         "towards M28 harder the further the policy moved (ref_kl 0.006 -> 0.035), while beating M28 "
                         "needs a policy unlike it")
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
    ap.add_argument("--standard-ai", default="adaptive,rush,tech")
    ap.add_argument("--archive-dir", type=Path, help="move old raw recordings and cache files here each iteration")
    ap.add_argument("--archive-raw-hours", type=float, default=24.0)
    ap.add_argument("--archive-cache-hours", type=float, default=72.0)
    ap.add_argument("--league", default="", help="self-play opponents: past networks' names (rl_NNN), copied to the "
                    "players' league folders at start")
    ap.add_argument("--league-weight", type=float, default=1.0)
    ap.add_argument("--min-new-games", type=int, default=60, help="wait for this many new games between iterations")
    ap.add_argument("--learn-args", default="", help="more learn_rl options, e.g. '--epochs 2 --distill 0.3'")
    ap.add_argument("--push", action="store_true", help="push each new network to the player")
    ap.add_argument("--guard-points", type=float, default=8.0,
                    help="stop and put the starting network back when a network's win rate on the player falls this "
                         "many points below the starting network's (over --guard-games games)")
    ap.add_argument("--guard-games", type=int, default=140)
    ap.add_argument("--distill", type=float, default=0.3, help="starting weight of M28's labels in learn_rl")
    ap.add_argument("--distill-min", type=float, default=0.05)
    ap.add_argument("--distill-max", type=float, default=0.6)
    ap.add_argument("--adaptive-distill", type=int, default=1,
                    help="1: the labels' weight follows the newest network's results on the player: x0.8 while it "
                         "wins at least as often as the starting network, x1.5 when it is 3+ points below (Zhao et "
                         "al. 2022, adaptive behaviour-cloning regularisation); 0: fixed --distill")
    a = ap.parse_args()
    run_dir = a.run.resolve()
    rounds, cache, ckpts = run_dir / "rounds", run_dir / "cache", run_dir / "checkpoints"
    for d in (rounds, cache, ckpts):
        d.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)
    journal = run_dir / "rl.jsonl"
    done = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    # the newest network an iteration produced (a failed or stopped iteration leaves a row without one)
    last = next((row for row in reversed(done) if row.get("checkpoint")), None)
    current = Path(last["checkpoint"]) if last else a.init.resolve()
    reference = (a.reference or a.init).resolve()
    sys.path.insert(0, str(ROOT))
    from fa_train.server_sync import pull, push, set_config
    if a.push and not done:
        push(a.host, "~/fa/player", current)
    if a.local_home:
        local_push(a.local_home, current)
    league_push(a.host, "~/fa/player", a.local_home, ckpts, [n for n in a.league.split(",") if n])
    # games played by the newest --recent-networks networks are used
    chain = lineage(a.init.resolve(), done)
    shas = {}

    def recent() -> set:
        for p in chain[-a.recent_networks:]:
            if p not in shas:
                shas[p] = sha256(p)
        return {shas[p] for p in chain[-a.recent_networks:]}

    allowed = recent()

    def anchor() -> Path:
        """The KL reference of the next iteration."""
        if a.reference_lag > 0 and len(chain) > a.reference_lag:
            return chain[-1 - a.reference_lag]
        return reference
    encoded = {n for row in done for n in row.get("rounds", [])}
    pending = []                                       # encoded rounds not yet counted by an iteration
    state_file = run_dir / "curriculum.json"
    level = json.loads(state_file.read_text())["m28_level"] if state_file.exists() else a.m28_start
    low, high = (float(x) for x in a.m28_band.split(","))
    if a.m28_curriculum:
        set_config(a.host, "~/fa/player", {"ai": opponent_mix(level, a.standard_ai, a.m28_weight, a.league, a.league_weight)})
        if a.local_home:
            local_config(a.local_home, {"ai": opponent_mix(level, a.standard_ai, a.m28_weight, a.league, a.league_weight)})
        state_file.write_text(json.dumps({"m28_level": level}))
        print(f"[{datetime.now():%H:%M}] M28 curriculum at x{level:g}", flush=True)
    k = len(done)
    distill = next((row["distill"] for row in reversed(done) if "distill" in row), a.distill)
    distill_basis = None
    # the network the imitation weight was last adjusted for, kept across restarts: forgotten, every restart
    # adjusted again for the same network (2026-10-11: 0.112 -> 0.169 -> 0.253, both times for rl_040)
    distill_file = run_dir / "adaptive_distill.json"
    if distill_file.exists():
        kept = json.loads(distill_file.read_text())
        distill, distill_basis = float(kept["distill"]), kept.get("basis")
    while not (run_dir / "STOP").exists() and (not a.iterations or k < a.iterations):
        log = run_dir / f"rl_{k + 1:03d}.log"
        pull(a.host, "~/fa/player", rounds, keep=False)
        if a.local_home:
            local_collect(a.local_home, rounds)
        if a.archive_dir:
            moved = archive_old(rounds, cache, a.archive_dir, a.archive_raw_hours, a.archive_cache_hours)
            if any(moved):
                print(f"[{datetime.now():%H:%M}] archived: {moved[0]} rounds' recordings, {moved[1]} cache files "
                      f"-> {a.archive_dir}", flush=True)
        new = []
        for p in sorted(rounds.glob("lane*_*")):
            if not (p / "DONE").exists() or p.name in encoded:
                continue
            meta = json.loads((p / "round.json").read_text())
            if meta.get("checkpoint_sha256") not in allowed:
                encoded.add(p.name)                     # an older network's games: not for this policy
                continue
            if run(["-m", "fa_train.encode_parallel", "--raw", str(p / "raw"), "--cache", str(cache),
                    "--workers", "12"], log) == 0:
                pending.append(p.name)
                encoded.add(p.name)
        new = list(pending)
        if a.m28_curriculum:
            w, g = m28_level_rate(rounds, allowed, level)
            if g >= a.m28_min_games and not (low <= w / g <= high):
                old = level
                level = round(min(1.0, max(0.3, level + (a.m28_step if w / g > high else -a.m28_step))), 3)
                if level != old:
                    set_config(a.host, "~/fa/player", {"ai": opponent_mix(level, a.standard_ai, a.m28_weight, a.league, a.league_weight)})
                    if a.local_home:
                        local_config(a.local_home, {"ai": opponent_mix(level, a.standard_ai, a.m28_weight, a.league, a.league_weight)})
                    state_file.write_text(json.dumps({"m28_level": level}))
                    print(f"[{datetime.now():%H:%M}] M28 curriculum x{old:g} -> x{level:g} (won {w}/{g})", flush=True)
        bad = regression(rounds, a.init.resolve(), ckpts, a.guard_points, a.guard_games,
                         order=[p.stem for p in chain[1:]])
        if bad:
            what, fallback = bad
            back = fallback or a.init.resolve()
            print(f"[{datetime.now():%H:%M}] REGRESSION: {what}; stopping and putting {back.name} back", flush=True)
            with journal.open("a", encoding="utf-8") as f:   # a restart goes on from `back`
                f.write(json.dumps({"iteration": k + 1, "regression": what, "checkpoint": str(back),
                                    "fallback": True}) + "\n")
            if a.push:
                push(a.host, "~/fa/player", back)
            break
        n_new = sum(len(list((rounds / n / "raw").glob("*.jsonl.gz"))) for n in new)
        if n_new < a.min_new_games and k > 0:
            print(f"[{datetime.now():%H:%M}] {n_new} new games; waiting for {a.min_new_games}", flush=True)
            time.sleep(300)
            continue
        if a.adaptive_distill:
            seen = newest_rate(rounds, a.init.resolve(), ckpts, order=[p.stem for p in chain[1:]])
            if seen is not None and seen[0] != distill_basis:
                name, rate, games, base = seen
                old = distill
                if rate >= base:
                    distill = max(a.distill_min, distill * 0.8)
                elif rate < base - 0.03:
                    distill = min(a.distill_max, distill * 1.5)
                distill_basis = name
                distill_file.write_text(json.dumps({"distill": distill, "basis": name}))
                print(f"[{datetime.now():%H:%M}] imitation weight {old:.3f} -> {distill:.3f} ({name} wins {rate:.0%} "
                      f"of {games} games, the starting network {base:.0%} on the same opponents)", flush=True)
                vs_m28 = m28_rate(rounds, a.init.resolve(), ckpts)
                if vs_m28:
                    print(f"[{datetime.now():%H:%M}] against M28: {vs_m28}", flush=True)
        out = ckpts / f"rl_{k + 1:03d}.pt"
        sil_args = []
        if a.sil_archive:
            for p in chain[-a.sil_networks:]:
                if p not in shas:
                    shas[p] = sha256(p)
            archive = success_archive(rounds, {shas[p] for p in chain[-a.sil_networks:]}, level, a.sil_archive)
            fresh = ensure_encoded(archive, cache)
            (run_dir / "sil.json").write_text(json.dumps(archive))
            sil_args = ["--sil", str(run_dir / "sil.json")]
            print(f"[{datetime.now():%H:%M}] success archive: {len(archive)} won games against M28 x{level:g}+ "
                  f"({fresh} encoded now)", flush=True)
        args = ["-m", "fa_train.learn_rl", "--init", str(current), "--out", str(out), "--cache", str(cache),
                "--reference", str(anchor()), "--max-games", str(a.games_per_iteration),
                "--stop-file", str(run_dir / "STOP")] + a.learn_args.split() + ["--distill", f"{distill:.4f}"] \
            + sil_args
        t0 = time.monotonic()
        code = run(args, log)
        row = {"iteration": k + 1, "from": str(current), "rounds": new, "new_games": n_new, "exit": code,
               "distill": round(distill, 4),
               "minutes": round((time.monotonic() - t0) / 60, 1)}
        current_before = current
        if code == 0 and out.exists():
            import torch
            row["learn_rl"] = torch.load(out, map_location="cpu", weights_only=False).get("learn_rl", {}).get("history")
            row["checkpoint"] = str(out)
            current = out
            chain.append(out)
            allowed = recent()                          # the new network and the ones the lanes still play
            if a.push:
                push(a.host, "~/fa/player", out)
                if a.local_home:
                    local_push(a.local_home, out)
        if a.m28_curriculum:
            row["m28_level"] = level
        pending = []
        with journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{datetime.now():%H:%M}] iteration {k + 1}: {n_new} new games -> "
              f"{out.name if code == 0 else 'FAILED (see ' + log.name + ')'} in {row['minutes']} min", flush=True)
        if code != 0:
            break
        k += 1


if __name__ == "__main__":
    main()
