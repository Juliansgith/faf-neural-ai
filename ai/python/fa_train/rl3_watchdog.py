"""Unattended upkeep of the server run (fa-watchdog.timer, every 15 minutes). Standard library only.

1. A learning pause (~/fa/player/PAUSE written by rl_loop --learn-alone) whose loop process is gone and with no learner
   running for 30 minutes is stale (the loop died before it lifted it): it is removed, so the games never wait for nothing.
2. The forced-upgrade test (player config lane_networks/lane_env for lanes 2 and 3): once both lanes have
   --test-games games against M28 since the test began, its result is written to ~/fa/rl3/reports/ and the two lanes go
   back to training games (lane_networks and lane_env removed).
3. A line per run in ~/fa/rl3/watchdog.log.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def learner_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "fa_train.learn_rl"], capture_output=True, text=True).stdout
    return bool(out.strip())


def owner_alive(text: str) -> bool:
    """The pause reads "rl_loop: learning since <time> (pid N)". The loop drains the games for up to 45 minutes before
    its learner starts, so "no learner" alone is not stale (2026-11-09 20:17 the watchdog lifted a draining pause)."""
    try:
        pid = int(text.rsplit("(pid ", 1)[1].split(")", 1)[0])
    except (IndexError, ValueError):
        return False
    return os.path.exists(f"/proc/{pid}")


def test_games(run: Path, home: Path, since: str, lanes=("2", "3")) -> dict:
    counts = {lane: 0 for lane in lanes}
    for folder in (run / "rounds", home / "rounds"):
        for d in glob.glob(str(folder / "lane*_*")):
            name = os.path.basename(d)
            lane, stamp = name[4:].split("_", 1)
            if lane not in counts or stamp < since or not os.path.exists(os.path.join(d, "DONE")):
                continue
            try:
                with open(os.path.join(d, "eval", "episodes.jsonl"), encoding="utf-8") as f:
                    counts[lane] += sum(1 for line in f if '"m28aicheat@' in line)
            except OSError:
                pass
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path.home() / "fa" / "rl3")
    ap.add_argument("--home", type=Path, default=Path.home() / "fa" / "player")
    ap.add_argument("--test-since", default="20261108T0600Z")
    ap.add_argument("--test-games", type=int, default=1200)
    a = ap.parse_args()
    notes = []

    pause = a.home / "PAUSE"
    marker = a.run / "watchdog_pause_seen"
    text = pause.read_text(errors="replace") if pause.exists() else ""
    if text.startswith("rl_loop: learning") and not learner_running() and not owner_alive(text):
        if not marker.exists():
            marker.write_text(str(time.time()))
        elif time.time() - float(marker.read_text() or 0) > 1800:
            pause.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            notes.append("stale learning pause removed (no learner for 30 minutes)")
    else:
        marker.unlink(missing_ok=True)

    cfg_path = a.home / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        cfg = {}
    # the test ended 2026-11-09 17:46; lane_env is used for other A/B tests since (2026-11-10: inference threads)
    if cfg.get("lane_env") and not (a.run / "reports" / "forced_upgrade_test.md").exists():
        counts = test_games(a.run, a.home, a.test_since)
        notes.append(f"forced-upgrade test games: {counts}")
        if all(v >= a.test_games for v in counts.values()):
            report = subprocess.run(
                [str(Path.home() / "venv" / "bin" / "python"), "-m", "fa_train.arena_report", "--rounds", str(a.run / "rounds"),
                 "--control", "3", "--since", a.test_since], capture_output=True, text=True,
                cwd=str(Path.home() / "faf-neural-ai" / "ai" / "python")).stdout
            (a.run / "reports").mkdir(parents=True, exist_ok=True)
            (a.run / "reports" / "forced_upgrade_test.md").write_text(
                "# Forced-upgrade test (lane 2: rl_000 plus upgrade orders at minutes 7, 10, 13, 16, 19; lane 3: rl_000)\n\n"
                f"Finished {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Lanes 0 and 1 are training lanes; read lane 2 "
                "against lane 3 (the control).\n\n```\n" + report + "```\n")
            cfg.pop("lane_env", None)
            cfg.pop("lane_networks", None)
            tmp = cfg_path.with_name("config.json.watchdog")
            tmp.write_text(json.dumps(cfg, indent=1))
            tmp.replace(cfg_path)
            notes.append("forced-upgrade test finished: report written, lanes 2 and 3 back to training")

    with open(a.run / "watchdog.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} {'; '.join(notes) or 'ok'}\n")


if __name__ == "__main__":
    main()
