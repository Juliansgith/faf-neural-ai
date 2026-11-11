"""The autopilot's own watchdog. Task Scheduler runs it every 10 minutes with pythonw.exe (no window; registered by
tools/autopilot/install_autopilot_task.ps1, which the user runs once). When the autopilot's heartbeat is older than
STALE_MIN minutes it relaunches the autopilot hidden and detached; a hung autopilot (alive, no heartbeat) is ended
first -- it is plain Python and never uses CUDA (rl_loop and learn_rl are separate processes and are never
touched here: the relaunched autopilot adopts the running loop). More than MAX_RESTARTS relaunches in 6 hours: it
alerts and pauses relaunching for an hour (a crash loop is a bug to look at). <state dir>/WATCHDOG_OFF switches
it off without removing the task.

    pythonw -m fa_train.autopilot_watchdog --config C:\\faf\\_runs\\server\\rl2\\autopilot\\autopilot.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CREATE_NO_WINDOW, DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP = 0x08000000, 0x00000008, 0x00000200
STALE_MIN, MAX_RESTARTS = 10, 5
ROOT = Path(__file__).resolve().parents[1]          # ai/python
WORKSPACE = Path(__file__).resolve().parents[4]


def autopilot_pids() -> list[int]:
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','pythonw.exe' -and "
          "$_.CommandLine -like '*fa_train.autopilot *' -and $_.CommandLine -notlike '*autopilot_watchdog*' } | "
          "Select-Object -ExpandProperty ProcessId")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                             text=True, timeout=60, creationflags=CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def decide(state_dir: Path, now: float, w: dict) -> tuple[bool, float]:
    """(relaunch?, heartbeat age in minutes) -- the pure part, for the tests."""
    if (state_dir / "WATCHDOG_OFF").exists():
        return False, 0.0
    try:
        hb = json.loads((state_dir / "heartbeat.json").read_text())
        age = (now - float(hb["time"])) / 60
    except (OSError, ValueError, KeyError, TypeError):
        age = 1e9
    if age < STALE_MIN:
        return False, age
    w["restarts"] = [t for t in w.get("restarts", []) if now - t < 6 * 3600]
    if len(w["restarts"]) >= MAX_RESTARTS and now - w["restarts"][-1] < 3600:
        return False, age                               # alerted when the budget ran out; wait
    return True, age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args()
    try:
        cfg = json.loads(a.config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    run = Path(cfg.get("run") or WORKSPACE / "_runs" / "server" / "rl2")
    state_dir = Path(cfg.get("state_dir") or run / "autopilot")
    state_dir.mkdir(parents=True, exist_ok=True)
    wd = state_dir / "watchdog.json"
    try:
        w = json.loads(wd.read_text())
    except (OSError, ValueError):
        w = {"restarts": []}
    now = time.time()
    go, age = decide(state_dir, now, w)
    if not go:
        return 0
    for pid in autopilot_pids():                       # alive but silent: hung (plain Python, never CUDA)
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)
    python = Path(cfg.get("python") or WORKSPACE / "ai-venv" / "Scripts" / "python.exe")
    pythonw = python.with_name("pythonw.exe")
    subprocess.Popen([str(pythonw if pythonw.exists() else python), "-m", "fa_train.autopilot", "--config",
                      str(a.config)], cwd=str(ROOT),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, close_fds=True)
    w["restarts"].append(now)
    tmp = wd.with_name(wd.name + ".new")
    tmp.write_text(json.dumps(w))
    tmp.replace(wd)
    with open(state_dir / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "watchdog_relaunch",
                            "heartbeat_age_min": round(age, 1)}) + "\n")
    topic = (cfg.get("alerts") or {}).get("ntfy_topic")
    if topic and len(w["restarts"]) >= MAX_RESTARTS:
        try:
            server = (cfg.get("alerts") or {}).get("ntfy_server", "https://ntfy.sh")
            urllib.request.urlopen(urllib.request.Request(
                f"{server}/{topic}", data=f"autopilot relaunched {len(w['restarts'])}x in 6 h; pausing relaunches "
                                          f"for 1 h".encode(), headers={"Title": "FAF RL: watchdog",
                                                                         "Priority": "high"}), timeout=20)
        except Exception:                               # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
