"""A player machine (the server, 2026-10-09): plays recorded, teacher-labelled games around the clock with the
newest network, for a learner elsewhere that pulls them, trains, and pushes the next network.

    python -m fa_train.player --home ~/fa/player

HOME/config.json (re-read before every round, so it can be changed while the service runs):
    {"lanes": 3, "workers": 14, "torch_threads": 4, "games_per_ai": 2, "ai": "medium,adaptive,m28ai",
     "labels": "m28", "minutes": 40, "maps": "small", "temperature": 0.5, "bin_dir": "~/fa/bin"}

- HOME/current.pt is the network to play. The learner uploads a new one as HOME/incoming.pt and renames it to
  current.pt (atomic); each lane picks it up at its next round.
- Each lane loops on its own (one deciding process and `workers` games): a slow game in one lane never leaves
  another lane's CPUs idle. A round is `games_per_ai` games per opponent per worker. A lane starts its next round
  once no more than `tail` games of the current one are still playing (default workers // 3): the last long games
  of a round (up to `minutes`) no longer leave the lane's other workers idle.
- A finished round is HOME/rounds/<lane>_<UTC time>/ with raw/ (recordings, the encoder's input), eval/ (the
  evaluator's results) and round.json (network sha256, config, outcomes), and a DONE file last. The learner pulls
  DONE rounds and deletes them here.
- HOME/STOP: every lane finishes its round and stops. HOME/PAUSE: lanes wait before their next round.
- A round that did not finish (a crash, a reboot) is moved to HOME/rounds/failed/ at the next start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE_OFFSET = 0
DEFAULTS = {"lanes": 3, "workers": 14, "torch_threads": 4, "games_per_ai": 2, "ai": "medium,adaptive,m28ai",
            "labels": "m28", "minutes": 40, "maps": "small", "temperature": 0.5, "bin_dir": "~/fa/bin"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(home: Path) -> dict:
    cfg = dict(DEFAULTS)
    path = home / "config.json"
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            log(f"config.json unreadable ({exc}); using the last good values")
    return cfg


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def outcomes(results: Path) -> dict | None:
    try:
        return json.loads(results.read_text()).get("summary")
    except (OSError, ValueError):
        return None


def play_round(home: Path, lane: int, cfg: dict, seed_base: int, state: dict | None = None) -> Path | None:
    """One round of one lane; returns its folder when it finished. `state` gets the round's folder and its number
    of games as soon as it starts (for the lane's overlap of rounds)."""
    current = home / "current.pt"
    if not current.exists():
        log(f"lane {lane}: no {current} yet; waiting")
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = home / "rounds" / f"lane{lane + LANE_OFFSET}_{stamp}"
    (folder / "raw").mkdir(parents=True, exist_ok=True)
    # the round's own copy: a new current.pt never changes a running round (two rounds of a lane may overlap)
    snapshot = folder / "network.pt"
    shutil.copyfile(current, snapshot)
    # "name" or "name:weight" (weight x games_per_ai x workers games against it)
    per = int(cfg["games_per_ai"]) * int(cfg["workers"])
    ais = []
    for spec in (s for s in str(cfg["ai"]).split(",") if s):
        name, _, weight = spec.partition(":")
        ais.append(f"{name}:{max(1, round(per * float(weight)))}" if weight else f"{name}:{per}")
    if state is not None:
        state.update(folder=folder, games=sum(int(s.rpartition(":")[2]) for s in ais))
    meta = {"lane": lane, "started": stamp, "checkpoint_sha256": sha256(snapshot), "config": cfg,
            "seed_base": seed_base, "host": os.uname().nodename if hasattr(os, "uname") else ""}
    (folder / "round.json").write_text(json.dumps(meta, indent=1))
    env = dict(os.environ, OMP_NUM_THREADS=str(cfg["torch_threads"]), MKL_NUM_THREADS=str(cfg["torch_threads"]),
               FA_ENV_BIN_DIR=str(Path(os.path.expanduser(cfg["bin_dir"]))))
    cmd = [sys.executable, "-u", "examples/tools_eval.py", "--checkpoint", str(snapshot), "--run", str(folder / "eval"),
           "--ai", ",".join(ais), "--games", str(int(cfg["games_per_ai"]) * int(cfg["workers"])),
           "--maps", str(cfg["maps"]), "--factions", "1,2,3,4", "--ai-factions", "1,2,3,4",
           "--minutes", str(cfg["minutes"]), "--workers", str(cfg["workers"]), "--seed-base", str(seed_base),
           "--mode", "sample", "--temperature", str(cfg["temperature"]), "--record", str(folder / "raw"),
           "--bin-dir", env["FA_ENV_BIN_DIR"]]
    if cfg.get("labels"):
        cmd += ["--labels", str(cfg["labels"])]
    if cfg.get("device"):
        cmd += ["--device", str(cfg["device"])]
    if cfg.get("vary_conditions"):
        cmd += ["--vary-conditions"]
    if cfg.get("act_gather_ms"):
        cmd += ["--act-gather-ms", str(cfg["act_gather_ms"]), "--act-min-batch", str(cfg.get("act_min_batch", 1))]
    t0 = time.monotonic()
    with open(folder / "eval.log", "w") as out:
        code = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=subprocess.STDOUT)
    meta.update(exit=code, minutes=round((time.monotonic() - t0) / 60, 1),
                games=len(list((folder / "raw").glob("*.jsonl.gz"))),
                outcomes=outcomes(folder / "eval" / "eval_results.json"))
    (folder / "round.json").write_text(json.dumps(meta, indent=1))
    if code != 0 or not meta["games"]:
        snapshot.unlink(missing_ok=True)
        log(f"lane {lane}: round {folder.name} failed (exit {code}, {meta['games']} games); see its eval.log")
        return None
    snapshot.unlink(missing_ok=True)
    (folder / "DONE").write_text(stamp)
    log(f"lane {lane}: round {folder.name}: {meta['games']} games in {meta['minutes']} min: "
        f"{json.dumps({k: v.get('win_rate') for k, v in (meta['outcomes'] or {}).items()})}")
    return folder


def finished_games(folder: Path) -> int:
    try:
        with open(folder / "eval" / "episodes.jsonl", "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def lane_loop(home: Path, lane: int) -> None:
    rounds = []                                       # (thread, state) of this lane's running rounds (at most 2)

    def run(cfg, seed_base, state):
        try:
            state["done"] = play_round(home, lane, cfg, seed_base, state)
        except Exception as exc:                      # noqa: BLE001 -- a lane must survive one bad round
            log(f"lane {lane}: round error: {type(exc).__name__}: {exc}")
            state["done"] = None

    while not (home / "STOP").exists():
        ended = [st for t, st in rounds if not t.is_alive()]
        rounds = [(t, st) for t, st in rounds if t.is_alive()]
        if any(st.get("done") is None for st in ended):
            time.sleep(60)                            # a failed round: not straight into the next one
        if (home / "PAUSE").exists():
            time.sleep(30)
            continue
        cfg = load_config(home)
        if lane >= int(cfg["lanes"]) or len(rounds) >= 2:   # fewer lanes configured now: this one rests
            time.sleep(30)
            continue
        if rounds:
            st = rounds[-1][1]
            tail = int(cfg.get("tail", int(cfg["workers"]) // 3))
            if "folder" not in st or st["games"] - finished_games(st["folder"]) > tail:
                time.sleep(20)
                continue
        # distinct per lane and round, and a valid game seed: minute of the epoch (mod 10^5) x 10^4 + lane x 10^3;
        # a round uses seed_base + game index (< 10^3 per opponent)
        seed_base = (int(time.time()) // 60 % 100_000) * 10_000 + lane * 1_000
        state = {}
        t = threading.Thread(target=run, args=(cfg, seed_base, state), daemon=True)
        t.start()
        rounds.append((t, state))
        time.sleep(61)                                # the next round's seed_base is another minute's
    for t, _ in rounds:
        t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--max-lanes", type=int, default=8, help="lane threads started (config.json says how many play)")
    ap.add_argument("--lane-offset", type=int, default=0,
                    help="added to the lanes' numbers in round names: a second player machine's rounds (lane10_...) "
                         "never collide with the first's in the learner's rounds folder")
    a = ap.parse_args()
    global LANE_OFFSET
    LANE_OFFSET = a.lane_offset
    home = a.home.expanduser().resolve()
    (home / "rounds" / "failed").mkdir(parents=True, exist_ok=True)
    for old in (home / "rounds").glob("lane*_*"):     # unfinished rounds of an earlier run
        if not (old / "DONE").exists():
            shutil.move(str(old), str(home / "rounds" / "failed" / old.name))
            log(f"moved unfinished round {old.name} to rounds/failed")
    (home / "STOP").unlink(missing_ok=True)
    log(f"player started in {home}; lanes per config.json (up to {a.max_lanes})")
    threads = [threading.Thread(target=lane_loop, args=(home, i), daemon=True) for i in range(a.max_lanes)]
    for i, t in enumerate(threads):
        t.start()
        time.sleep(20)                                # stagger the lanes' game loading
    for t in threads:
        t.join()
    log("player stopped (STOP)")


if __name__ == "__main__":
    main()
