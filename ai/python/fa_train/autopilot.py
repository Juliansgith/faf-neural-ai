"""Autopilot: an experiment manager and supervisor for an unattended RL run (2026-10-17; designed in the
2026-10-16 review, autopilot.md, and adapted to rl_loop's exit-code contract).

It supervises fa_train.rl_loop (one training process, the only CUDA user), runs a queue of EXPERIMENTS -- a name,
rl_loop/learn_rl flag overrides, player-config overrides and a start network (normally the champion) -- judges each
by its networks' matched results against the champion (the gate's statistic), and switches to the next experiment
when one has FAILED, STALLED or used its time. It restarts the loop and the local game lane after crashes (with
backoff and a crash budget), notices wedges and stalls, writes a report (markdown, HTML with a PNG chart; phone
readable) and sends alerts (ntfy, opt-in). Everything it starts is hidden (CREATE_NO_WINDOW, itself under pythonw):
no window ever opens or takes the focus.

What it never does: kill a CUDA process (learn_rl), delete a STOP it did not write (the GPU watchdog's or the
user's; with gpu.auto_resume, which is off by default, an empty one after a quiet period), or switch an experiment
inside a learn_rl. It switches at rl_loop's safe point, the YIELD file (the loop exits 4 at its next loop top).
Its own STOP (content "autopilot: ...") is written only for a silent learn_rl or a wedged loop and removed by it
before the next launch.

    pythonw -m fa_train.autopilot --config C:\\faf\\_runs\\server\\rl2\\autopilot\\autopilot.json [--once] [--dry-run]

rl_loop's exit codes (rl_loop_exit.json): 0 STOP/iterations, 1 traceback, 3 learn_rl failed twice, 4 YIELD,
5 locked (another loop), 6 the gate's rejection cap (the experiment FAILs), 7 a STOP was there at its start.

Files (all under <run>/autopilot/): autopilot.json (settings), queue.json (experiments; edited by hand at any time),
state.json (the autopilot's own state), events.jsonl (every decision), heartbeat.json (every tick, for the
watchdog: fa_train.autopilot_watchdog), override.json (user commands), report/ (report.md, report.html,
progress.png), champions/ (a copy of every champion), WATCHDOG_OFF (the user's switch for the watchdog).

USER COMMANDS (override.json, or a message on the ntfy command topic): {"cmd": "skip"} ends the current experiment
at the next safe point; "hold" stops training at the next safe point and waits; "resume"; {"cmd": "run",
"experiment": NAME} runs that queue entry next; {"cmd": "champion", "name": "rl_NNN"} makes that network the
champion (rl_loop --set-champion at the next launch; the current experiment restarts from it); {"cmd": "extend",
"hours": H} gives the current experiment H more hours before STALL/MAX; "report" writes the report now. A
whitelist: no command can pass flags, paths or shell text.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # ai/python
WORKSPACE = Path(__file__).resolve().parents[4]     # the folder holding the repo (C:\faf in the docs)
CREATE_NO_WINDOW = 0x08000000                       # no console window: no focus change, nothing to close
CREATE_NEW_PROCESS_GROUP = 0x00000200
OWN_STOP = "autopilot"                              # the content prefix of a STOP this autopilot wrote

# rl_loop's exit codes (fa_train.rl_loop: EXIT_*)
EXIT_DONE, EXIT_ERROR, EXIT_LEARN_FAILED, EXIT_YIELD, EXIT_LOCKED, EXIT_GATE_EXHAUSTED, EXIT_STOP_PRESENT = \
    0, 1, 3, 4, 5, 6, 7

# ---------------------------------------------------------------------------------------------------- settings

_RUN = WORKSPACE / "_runs" / "server" / "rl2"
_INIT = WORKSPACE / "_runs" / "server" / "vaa_M" / "checkpoints" / "step_01.pt"
_LOCAL = WORKSPACE / "_runs" / "local_player"
SMALL_MAPS = "SCMP_012,SCMP_013,SCMP_016,X1MP_001,X1MP_002,SCMP_019,SCMP_023,SCMP_012,SCMP_013,SCMP_016"

DEFAULTS = {
    "run": str(_RUN),
    "python": str(WORKSPACE / "ai-venv" / "Scripts" / "python.exe"),
    "init": str(_INIT),
    "log": str(WORKSPACE / "_runs" / "server" / "rl2.log"),
    "host": "my-server",
    "server_home": "~/fa/player",
    "local_home": str(_LOCAL),
    "local_player_cmd": None,       # None: python -u -m fa_train.player --home <local_home> --max-lanes 1 --lane-offset 10
    "control_lane": True,           # the local lane plays the champion (rl_loop --control-lane)
    "gpu_watchdog_ps1": str(WORKSPACE / "_runs" / "tools" / "gpu_watchdog.ps1"),
    # rl2.cmd's command line as of 2026-10-14 19:30 minus --run/--init/--champion, plus the 2026-10-17 loop flags:
    # the judge-once gate at 150 games, the 350 GB storage budget (REPORT decision 1), the disk floor
    "base_loop": {"--reference": str(_INIT), "--push": None, "--games-per-iteration": "480",
                  "--min-new-games": "240", "--recent-networks": "4", "--m28-curriculum": None, "--m28-start": "0.6",
                  "--reference-lag": "10", "--local-home": str(_LOCAL), "--sil-archive": "60",
                  "--archive-dir": r"D:\faf_archive\rl2", "--gate-games": "150", "--max-rejections": "3",
                  "--storage-budget-gb": "350", "--min-free-gb": "30"},
    "base_learn": {"--opponent-critic": None, "--timeout-scale": "10", "--timeout-cost": "2",
                   "--mapped": str(_RUN / "mapped"), "--distill-large": "4", "--epochs": "4", "--lr": "2e-5",
                   "--distill": "0.3", "--ref-coef": "0.2", "--streams": "16", "--micro": "4",
                   "--grad-checkpoint": None, "--local-credit": "0.5", "--econ-coef": "1.0",
                   "--econ-source": "capacity", "--reclaim-coef": "0.5", "--intel-coef": "1.0", "--map-coef": "2.0",
                   "--waste-coef": "1.0", "--stall-coef": "1.0"},
    # player-config keys an experiment may override, with their normal values (restored when it ends)
    "base_player": {"maps": "SCMP_012,SCMP_013,SCMP_003,SCMP_016,X1MP_001,SCMP_004,X1MP_002,SCMP_019,SCMP_006,"
                            "SCMP_023,SCMP_012,SCMP_013"},
    "gate_since": "20261013T2310Z",
    # judge(): FAIL/STALL/MAX (autopilot.md 3.4, calibrated on rl2's history in 2.2-2.4); se_inflation as rl_loop's
    # gate (the formula SE was 10-20 % under the bootstrap's)
    "rules": {"z": 1.96, "se_inflation": 1.2, "k_fallbacks": 2, "fail_upper": -3.0, "fail_min_games": 300,
              "stall_hours": 36, "stall_upper": 2.0, "window_games": 700, "max_hours": 60,
              "max_quiet_hours": 24, "entropy_alarm": 0.15},
    "restart": {"backoff_minutes": [1, 5, 15, 30, 60], "exp_crashes": 6, "exp_window_hours": 12,
                "global_crashes": 15, "global_window_hours": 24, "learn_failures_in_a_row": 3},
    "stall": {"wedge_minutes": 120, "wedge_kill_minutes": 30, "learn_log_hours": 4, "rounds_alert_hours": 4,
              "local_restart_hours": 6},
    "disk": {"warn_gb": 60, "check_minutes": 30, "archive": r"D:\faf_archive\rl2"},
    "memory": {"max_commit_gb": 110},
    # gpu.auto_resume removes an EMPTY STOP (the GPU watchdog's -- or a user's) after quiet_hours without driver
    # errors, at most max_resumes_per_day: the user's decision (REPORT decision 4), off by default
    "gpu": {"auto_resume": False, "quiet_hours": 2, "max_resumes_per_day": 1},
    "report": {"mirror_dir": None, "daily_hour": 8, "every_hours": 6},
    "alerts": {"ntfy_topic": None, "ntfy_command_topic": None, "ntfy_server": "https://ntfy.sh",
               "min_minutes_between_same": 60, "min_minutes_between_high": 30},
    "end_time": None,               # ISO time the user is back: finalize (champion on all lanes) some hours before
    "finalize_hours_before": 6,
    "on_exhausted": "best_stalled",  # or "hold"
    "tick_seconds": 60,
    "state_dir": None,             # default <run>/autopilot
}


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) and \
            not k.startswith("base_") else v
    return out


def load_settings(path: Path) -> dict:
    s = deep_merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {})
    s["dir"] = s.get("state_dir") or str(Path(s["run"]) / "autopilot")
    return s


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------- experiments

def merged_args(base: dict, over: dict) -> list[str]:
    """Flags as a command line: base overridden by `over` (value None: a bare flag; False: removed)."""
    d = dict(base)
    for k, v in (over or {}).items():
        if v is False:
            d.pop(k, None)
        else:
            d[k] = v
    out = []
    for k, v in d.items():
        out += [k] if v is None else [k, str(v)]
    return out


def resolve(name_or_path: str, run: Path, champion: str) -> str:
    """'champion' -> the champion's checkpoint; 'rl_NNN' -> its checkpoint; a path stays a path."""
    if name_or_path == "champion":
        name_or_path = champion
    if name_or_path.startswith("rl_") and "\\" not in name_or_path and "/" not in name_or_path:
        return str(run / "checkpoints" / f"{name_or_path}.pt")
    return name_or_path


def expand(exp: dict, queue: list) -> dict:
    """An entry with "inherit": NAME takes that entry's loop/learn/player overrides under its own ("X1 + E5")."""
    base_name = exp.get("inherit")
    if not base_name:
        return exp
    base = next((e for e in queue if e.get("name") == base_name and e is not exp), None)
    if base is None:
        return exp
    base = expand(base, [e for e in queue if e is not exp])
    out = dict(exp)
    for key in ("loop", "learn", "player"):
        out[key] = {**(base.get(key) or {}), **(exp.get(key) or {})}
    return out


def loop_command(s: dict, exp: dict, champion: str, first: bool = True, set_champion: str | None = None) -> list:
    """The rl_loop command line of an experiment. `first`: the experiment's first launch goes back to its start
    network (--start-from); a restart after a crash resumes where it was (with --start-from on every launch, a crash
    after a promotion inside the experiment threw away the new champion's descendants). --champion only seeds a
    missing champion.json (rl_loop); `set_champion` (the user's champion command) replaces it."""
    run = Path(s["run"])
    loop = {**s["base_loop"], **(exp.get("loop") or {})}
    if s.get("control_lane") and s.get("local_home"):
        loop.setdefault("--control-lane", None)
    loop = {k: (resolve(v, run, champion) if k == "--reference" and isinstance(v, str)
                else v.replace("CHAMPION", champion) if isinstance(v, str) else v)
            for k, v in loop.items() if v is not False}
    learn = merged_args(s["base_learn"], exp.get("learn") or {})
    start = exp.get("start", "champion")
    return [s["python"], "-u", "-m", "fa_train.rl_loop", "--run", str(run), "--init", s["init"],
            "--champion", champion, "--tag", exp["name"],
            *(["--set-champion", set_champion] if set_champion else []),
            *(["--start-from", "champion" if start == "champion" else start] if first else []),
            *merged_args(loop, {}), "--learn-args", " ".join(learn)]


# The REPORT's section 4C queue (research.md section 3, "Named experiments"), each from the champion. "inherit":
# X1's changes under the entry's own. Flags marked PLACEHOLDER do not exist yet: such an entry is skipped (INVALID,
# alerted) until someone fills it in queue.json. The research agent's day-0 items (E0 two half-data trainings,
# E4 the weight-averaged soups) and the final weight-averaged network are hand steps, not queue entries.
X1_PLAYER = {"maps": SMALL_MAPS, "minutes": 40}
DEFAULT_QUEUE = [
    {"name": "X1-clean", "why": "all the review's A-fixes (reward denoise: capacity clip 1, stall 0.25; a "
     "Monte-Carlo critic warm-up of 3 epochs at its own learning rate, no value clip; no 'undo' imitation labels), "
     "small maps only and no full-M28 slot, the KL anchor on the champion (research E1+E2+E3, REPORT 4A)",
     "start": "champion",
     "loop": {"--anchor-champion": None, "--m28-weight": "3", "--stretch-weight": "2", "--full-m28-weight": "0"},
     "learn": {"--capacity-clip": "1", "--stall-coef": "0.25", "--mc-warmup": None, "--value-warmup": "3",
               "--value-lr": "1e-4", "--value-clip": "0", "--ref-coef": "0.3"},
     "player": X1_PLAYER},
    {"name": "X2-terminal", "inherit": "X1-clean", "why": "win/loss dominates the return: less shaping, higher "
     "lambda and gamma (research E5)", "start": "champion",
     "learn": {"--shaping": "0.4", "--lam": "0.98", "--gamma": "0.999"}},
    {"name": "X3-slow", "inherit": "X1-clean", "why": "each update was ~80 % noise: smaller steps on more data "
     "(research E6)", "start": "champion",
     "loop": {"--games-per-iteration": "720"}, "learn": {"--lr": "1e-5", "--epochs": "2", "--target-kl": "0.01"}},
    {"name": "X4-less-imitation", "inherit": "X1-clean", "why": "M28's labels pull ~6x the RL term: imitation "
     "0.02 fixed, large maps x1, gentler self-imitation (research E7; its --distill-temperature/--distill-smoothing "
     "do not exist in learn_rl and are left out)", "start": "champion",
     "loop": {"--adaptive-distill": "0", "--distill": "0.02"},
     "learn": {"--distill-large": "1", "--sil-coef": "0.01"}},
    {"name": "X5-critic", "inherit": "X1-clean", "why": "a separate (wider) critic (research E8/E9)",
     "start": "champion", "learn": {"--value-trunk-grad": "PLACEHOLDER"}},
    {"name": "X6-self-play", "inherit": "X3-slow", "why": "X3's small updates plus one self-play slot, the champion, as a measured "
     "experiment (research E10)", "start": "champion",
     "loop": {"--league": "CHAMPION", "--league-weight": "0.5"}},
    {"name": "X7-offline-awr", "why": "offline advantage-weighted training on the stored games (research E11): not "
     "an rl_loop run; needs its own tool", "start": "champion", "loop": {"--offline": "PLACEHOLDER"}},
]


# ---------------------------------------------------------------------------------------------------- data

def names_of(init: Path, ckpts: Path, sha_file: Path) -> dict:
    """sha256 -> network name (rl_loop's hash cache, by size and time: hashing 130+ checkpoints costs ~10 s)."""
    from fa_train import rl_loop
    rl_loop.load_sha_cache(sha_file)
    out = {}
    if init.exists():
        out[rl_loop.sha256_cached(init)] = "start"
    for p in sorted(ckpts.glob("rl_*.pt")):
        out[rl_loop.sha256_cached(p)] = p.stem
    rl_loop.save_sha_cache(sha_file)
    return out


def scan_rounds(rounds: Path, names: dict, since: str) -> dict:
    """network -> {"cells": {gate_condition: [w, n]} (same provenance and conditions as the champion gate),
    "large": [w, n], "self": [w, n], "games": n, "first": stamp, "last": stamp} (rl_loop's per-round caches)."""
    from fa_train.rl_loop import (LARGE_KM, _map_km, done_rounds, episodes, round_meta, round_stamp,
                                 round_environment, gate_condition)
    out: dict = {}
    for r in done_rounds(rounds):
        stamp = round_stamp(r)
        if stamp < since:
            continue
        net = names.get(round_meta(r).get("checkpoint_sha256"))
        eps = episodes(r)
        if net is None or eps is None:
            continue
        environment = round_environment(r)
        d = out.setdefault(net, {"cells": {}, "large": [0, 0], "self": [0, 0], "games": 0, "first": stamp,
                                 "last": stamp})
        d["last"] = max(d["last"], stamp)
        for e in eps:
            won = int(e.get("outcome") == "victory")
            d["games"] += 1
            ai = str(e.get("ai") or "")
            if ai.startswith("self@"):
                d["self"][0] += won
                d["self"][1] += 1
            elif _map_km(e.get("map") or "") > LARGE_KM:
                d["large"][0] += won
                d["large"][1] += 1
            else:
                c = d["cells"].setdefault(gate_condition(e, environment), [0, 0])
                c[0] += won
                c[1] += 1
    return out


def pool(scan: dict, nets: list) -> dict:
    out: dict = {}
    for n in nets:
        for k, (w, g) in scan.get(n, {}).get("cells", {}).items():
            c = out.setdefault(k, [0, 0])
            c[0] += w
            c[1] += g
    return out


def versus(scan: dict, champion: str, nets: list, inflation: float = 1.0):
    """(diff, se, games) of the pooled networks vs the champion (SE x `inflation`), or None."""
    from fa_train.rl_loop import _stratified
    res = _stratified(scan.get(champion, {}).get("cells", {}), pool(scan, nets))
    return (res[0], res[1] * inflation, res[2]) if res else None


def read_journal(run: Path) -> list:
    from fa_train.rl_loop import read_journal as rj
    return rj(run / "rl.jsonl")


def experiment_networks(rows: list, after_row: int = 0) -> list:
    """The experiment's own networks (trained rows, not fallbacks), oldest first, from row `after_row` on."""
    return [Path(r["checkpoint"]).stem for r in rows[after_row:] if r.get("checkpoint") and not r.get("fallback")]


def fallbacks(rows: list, after_row: int = 0) -> int:
    return sum(1 for r in rows[after_row:] if r.get("fallback") and any(k in r for k in ("gate", "collapse")))


# ---------------------------------------------------------------------------------------------------- rules

@dataclass
class Verdict:
    kind: str                     # "continue", "FAIL", "STALL", "MAX", "ERROR", "SKIP"
    why: str
    numbers: dict = field(default_factory=dict)


def judge(rules: dict, scan: dict, champion: str, nets: list, n_fallbacks: int, hours: float,
          hours_since_promo: float, extra_hours: float = 0.0) -> Verdict:
    """The experiment's verdict (autopilot.md 3.4). `nets`: its networks since the champion became champion (or
    since it started), oldest first; `n_fallbacks`: gate/collapse fallbacks since then; `hours`: since it started;
    `hours_since_promo`: since its last promotion (= hours without one).
    FAIL: k_fallbacks (2) fallbacks without a promotion (K=2 ends a harmful configuration, mu -2 points an
    iteration, by 24 h with p 0.70 and keeps a useful one, mu +1, past 24 h with p 0.94), or >= 300 pooled games
    with the 95 % upper bound under -3 points (caught every post-rl_089 restart within 1-3 h, never fired on the
    rl_044-062 climb or the rl_070-089 plateau). STALL: 36 h without a promotion and the newest ~700 games' upper
    bound under +2. MAX: 60 h and no promotion in the last 24 h."""
    z, infl = rules["z"], rules.get("se_inflation", 1.0)
    pooled = versus(scan, champion, nets, infl) if nets else None
    window, n = [], 0
    for m in reversed(nets):
        window.append(m)
        n += sum(g for _, g in scan.get(m, {}).get("cells", {}).values())
        if n >= rules["window_games"]:
            break
    win = versus(scan, champion, window, infl) if window else None
    num = {"pooled": pooled, "window": win, "fallbacks": n_fallbacks, "hours": round(hours, 1),
           "hours_since_promo": round(hours_since_promo, 1), "networks": len(nets),
           "max_hours": rules["max_hours"], "hard_max_hours": rules.get("hard_max_hours", False)}
    if n_fallbacks >= rules["k_fallbacks"]:
        return Verdict("FAIL", f"{n_fallbacks} gate fallbacks to the champion without a promotion", num)
    if pooled and pooled[2] >= rules["fail_min_games"] and 100 * (pooled[0] + z * pooled[1]) < rules["fail_upper"]:
        return Verdict("FAIL", f"pooled {100 * pooled[0]:+.1f} points vs {champion} (95 % upper "
                               f"{100 * (pooled[0] + z * pooled[1]):+.1f}) over {pooled[2]} games", num)
    if hours_since_promo >= rules["stall_hours"] + extra_hours and (
            win is None or 100 * (win[0] + z * win[1]) < rules["stall_upper"]):
        return Verdict("STALL", f"{hours_since_promo:.0f} h without a promotion and the newest games cannot be "
                                f"{rules['stall_upper']:g}+ points better"
                       + (f" ({100 * win[0]:+.1f}, upper {100 * (win[0] + z * win[1]):+.1f})" if win else ""), num)
    if hours >= rules["max_hours"] + extra_hours and (rules.get("hard_max_hours") or
                                                    hours_since_promo >= rules["max_quiet_hours"]):
        return Verdict("MAX", f"{hours:.0f} h used, none of the last {hours_since_promo:.0f} h promoted", num)
    return Verdict("continue", "", num)


def experiment_rules(rules: dict, spec: dict) -> dict:
    """Explicit per-run budgets, including a productive retry's shorter budget, are hard caps.

    The default policy keeps its promotion grace. A queue entry's max_hours promises an actual bounded
    experiment and must not silently inherit that grace or the global 60-hour value.
    """
    out = dict(rules)
    if spec.get("max_hours") is not None:
        hours = float(spec["max_hours"])
        if not 0 < hours <= 24 * 365:
            raise ValueError("experiment max_hours must be positive and at most one year")
        out.update(max_hours=hours, hard_max_hours=True)
    return out


def next_experiment(queue: list, results: dict, on_exhausted: str) -> dict | None:
    """A productive experiment first (it produced a champion and has not had its turn since: from the new champion,
    shorter); then the first pending queue entry; then (best_stalled) the STALLed experiment with the best window
    score once more; None: hold. Productive first since 2026-10-22: X3 made the first champion past the plateau
    (rl_147, +10 points) and would otherwise have waited behind every untried entry."""
    for e in queue:
        r = results.get(e["name"]) or {}
        if r.get("requeue") and not r.get("requeued_done"):
            return {**expand(e, queue), "max_hours": max(12, (e.get("max_hours") or 60) / 2), "_requeue": True}
    for e in queue:
        if e["name"] not in results:
            return expand(e, queue)
    if on_exhausted == "best_stalled":
        stalled = [(results[e["name"]].get("score") or -99, e) for e in queue
                   if results[e["name"]].get("verdict") == "STALL" and not results[e["name"]].get("rerun")
                   and not results[e["name"]].get("requeued_done")]      # one second turn per experiment
        if stalled:
            return {**expand(max(stalled, key=lambda x: x[0])[1], queue), "_rerun": True}
    return None


def entropy_alarm(rows: list, champion_entropy: float | None, limit: float) -> str | None:
    """The newest trained row's final-epoch entropy far from the champion's (autopilot.md 2.3: 3 of 4 such networks
    were bad; an alert, never a decision)."""
    if champion_entropy is None:
        return None
    for r in reversed(rows):
        h = [e for e in (r.get("learn_rl") or []) if isinstance(e, dict) and "epoch" in e]
        if r.get("checkpoint") and h:
            ent = h[-1].get("ent")
            if ent is not None and abs(ent - champion_entropy) > limit:
                return f"{Path(r['checkpoint']).stem}: entropy {ent:.3f} vs the champion's {champion_entropy:.3f}"
            return None
    return None


# ---------------------------------------------------------------------------------------------------- processes

def processes() -> list[dict] | None:
    """[{pid, ppid, name, cmd}] of python/pythonw/powershell processes (CIM; psutil is not in the venv). None when
    the listing failed: the caller must then do nothing (an empty list would read as "rl_loop is gone" and start a
    second one -- two CUDA trainings; rl_loop's own lock refuses that too, exit 5)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','pythonw.exe','powershell.exe' } "
          "| Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                             text=True, timeout=60, creationflags=CREATE_NO_WINDOW).stdout
        rows = json.loads(out) if out.strip() else []
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    rows = rows if isinstance(rows, list) else [rows]
    return [{"pid": r["ProcessId"], "ppid": r["ParentProcessId"], "name": r["Name"], "cmd": r.get("CommandLine") or ""}
            for r in rows if r.get("ProcessId") != os.getpid()]


def find(procs: list, *needles: str) -> list[dict]:
    return [p for p in procs if all(n.lower() in p["cmd"].lower() for n in needles)]


def _command_args(command: str) -> list[str] | None:
    """Read top-level arguments; flags inside a quoted Python -c program are not options."""
    try:
        if sys.platform != "win32":
            import shlex
            return shlex.split(command)
        import ctypes
        split = ctypes.WinDLL("shell32", use_last_error=True).CommandLineToArgvW
        split.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        split.restype = ctypes.POINTER(ctypes.c_wchar_p)
        free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        free.argtypes, free.restype = [ctypes.c_void_p], ctypes.c_void_p
        count = ctypes.c_int()
        argv = split(command, ctypes.byref(count))
        if not argv:
            return None
        try:
            return [argv[i] for i in range(count.value)]
        finally:
            free(ctypes.cast(argv, ctypes.c_void_p))
    except (OSError, ValueError):
        return None


def learner_processes(procs: list, run: Path) -> list[dict]:
    """GPU/unknown learners and this run's learners block lifecycle transitions.

    An explicitly CPU-only job with independent output, mapped data and STOP paths
    does not own this run or the GPU. Keep ambiguous arguments and managed-loop
    descendants blocking, including CPU children that the loop must finish first.
    """
    root = run.resolve()
    owned = {p["pid"] for p in find(procs, "fa_train.rl_loop", str(run)) if "pid" in p}
    parents = {p["pid"]: p.get("ppid") for p in procs if "pid" in p}
    blockers = []
    for process in find(procs, "fa_train.learn_rl"):
        args = _command_args(process["cmd"])
        def values(flag):
            return [args[i + 1] if i + 1 < len(args) else "" for i, a in enumerate(args) if a == flag] + \
                   [a.split("=", 1)[1] for a in args if a.startswith(flag + "=")]
        independent = False
        if args and values("--device") == ["cpu"] and len(values("--out")) == 1:
            writes = values("--out") + values("--mapped") + values("--stop-file")
            try:
                independent = all(Path(p).is_absolute() and not Path(p).resolve().is_relative_to(root)
                                  for p in writes)
            except OSError:
                pass
            pid, seen = process.get("pid"), set()
            while pid is not None and pid not in seen:
                if pid in owned:
                    independent = False
                    break
                seen.add(pid)
                pid = parents.get(pid)
        if not independent:
            blockers.append(process)
    return blockers


def launch(cmd: list[str], cwd: Path, log: Path | None) -> int:
    """A hidden child (no window, no focus; it outlives the autopilot)."""
    out = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
    if log:
        out.write(f"\n[{datetime.now():%H:%M}] autopilot: $ {' '.join(cmd)}\n")
        out.flush()
    try:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, close_fds=True)
    finally:
        if log:
            out.close()                                 # the child has its own handle
    return p.pid


def kill_tree(pid: int) -> None:
    """Only for CPU loop trees; callers exclude GPU/unknown and managed learners."""
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)


def free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return float("nan")


def commit_gb() -> float:
    ps = "$o = Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1MB"
    try:
        return float(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                    timeout=60, creationflags=CREATE_NO_WINDOW).stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def gpu_driver_errors(hours: float) -> int:
    ps = (f"(Get-WinEvent -FilterHashtable @{{LogName='System'; ProviderName='nvlddmkm'; "
          f"StartTime=(Get-Date).AddHours(-{hours})}} -ErrorAction SilentlyContinue | Measure-Object).Count")
    try:
        return int(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                  timeout=60, creationflags=CREATE_NO_WINDOW).stdout.strip() or 0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


# ---------------------------------------------------------------------------------------------------- alerts

def log_event(s: dict, kind: str, **kw) -> None:
    d = Path(s["dir"])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "event": kind, **kw},
                           default=str) + "\n")


class Alerts:
    """Every alert goes to events.jsonl; with alerts.ntfy_topic (opt-in, off by default) also to the phone. ntfy.sh
    needs no account: the topic name is the secret (long and random); messages carry run statistics only.
    Rate-limited per kind (min_minutes_between_same), "high" ones always."""

    def __init__(self, s: dict, state: dict):
        self.s, self.state = s, state

    def send(self, key: str, text: str, priority: str = "default") -> None:
        last = self.state.setdefault("alerts_sent", {}).get(key, 0)
        gap = self.s["alerts"].get("min_minutes_between_high", 30) if priority == "high" else self.s["alerts"]["min_minutes_between_same"]
        if time.time() - last < gap * 60:          # high ones too: a stalled-rounds alert repeated every minute (2026-10-20)
            return
        self.state["alerts_sent"][key] = time.time()
        log_event(self.s, "alert", key=key, text=text)
        topic = self.s["alerts"]["ntfy_topic"]
        if not topic:
            return
        try:
            req = urllib.request.Request(f"{self.s['alerts']['ntfy_server']}/{topic}", data=text.encode()[:3800],
                                         headers={"Title": f"FAF RL: {key}", "Priority": priority})
            urllib.request.urlopen(req, timeout=20).read()
        except Exception as exc:                        # noqa: BLE001 -- an alert must never stop the autopilot
            log_event(self.s, "alert_failed", error=str(exc))


def ntfy_commands(s: dict, state: dict) -> list[dict]:
    """User commands from the ntfy command topic (opt-in; whitelisted in apply_command)."""
    topic = s["alerts"]["ntfy_command_topic"]
    if not topic:
        return []
    since = state.get("ntfy_since", "all")
    try:
        with urllib.request.urlopen(f"{s['alerts']['ntfy_server']}/{topic}/json?poll=1&since={since}",
                                    timeout=20) as r:
            msgs = [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]
    except Exception:                                   # noqa: BLE001
        return []
    out = []
    for m in msgs:
        if m.get("event") != "message":
            continue
        state["ntfy_since"] = m["id"]
        words = str(m.get("message", "")).split()
        if words:
            cmd = {"cmd": words[0].lower()}
            if len(words) > 1:
                cmd["experiment" if cmd["cmd"] == "run" else "name" if cmd["cmd"] == "champion" else "hours"] = words[1]
            out.append(cmd)
    return out


# ---------------------------------------------------------------------------------------------------- report

def write_report(s: dict, state: dict, scan: dict, journal: list, champion: str, verdict: Verdict | None) -> Path:
    """report.md + progress.png + report.html (the PNG inlined: one file to open on a phone), mirrored."""
    out = Path(s["dir"]) / "report"
    out.mkdir(parents=True, exist_ok=True)
    z, infl = s["rules"]["z"], s["rules"].get("se_inflation", 1.0)
    exp = state.get("experiment") or {}
    lines = [f"# RL autopilot report {datetime.now():%Y-%m-%d %H:%M}", "",
             f"**Mode** {state.get('mode')}  **Champion** {champion}  **Experiment** {exp.get('name', '-')} "
             f"(since {exp.get('started', '-')})", ""]
    if verdict:
        lines += [f"Current verdict: {verdict.kind} {verdict.why}",
                  f"Numbers: `{json.dumps(verdict.numbers, default=str)}`", ""]
    lines += ["## Experiments", "", "Lifetime trained counts and promotions; verdict/score and segment counts describe the latest judged segment.",
              "", "| experiment | first started | latest ended | lifetime trained | segment networks | lifetime promotions | segment fallbacks | verdict | score |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, r in state.get("results", {}).items():
        trained = sum(1 for row in journal if row.get("experiment") == name and row.get("checkpoint")
                      and not row.get("fallback") and row.get("learn_rl"))
        lines.append(f"| {name} | {r.get('started', '')} | {r.get('ended', '')} | {trained} | {r.get('networks', '')} | "
                     f"{r.get('promotions', 0)} | {r.get('fallbacks', '')} | {r.get('verdict', '')} {r.get('why', '')} | "
                     f"{r.get('score', '')} |")
    if state.get("runs"):
        lines += ["", "## Completed experiment runs", "", "| experiment | started | ended | trained | promotions | fallbacks | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in state["runs"][-30:]:
            lines.append(f"| {r['name']} | {r['started']} | {r['ended']} | {r['trained']} | {r['promotions']} | "
                         f"{r['fallbacks']} | {r['verdict']} {r['why']} |")
    lines += ["", "## Champions", ""] + [f"* {c['time']} {c['name']} ({c.get('why', '')})"
                                         for c in state.get("champions", [])]
    gate = {}
    try:
        gate = json.loads((Path(s["run"]) / "gate_state.json").read_text()).get("verdicts", {})
    except (OSError, ValueError):
        pass
    if gate:
        lines += ["", "## Frozen gate verdicts (newest 20; each network judged once)", "",
                  "| network | vs | rollback (diff, upper, games) | promotion (diff, z, games) |", "|---|---|---|---|"]
        for v in list(gate.values())[-20:]:
            rb, pr = v.get("rollback") or {}, v.get("promotion") or {}
            lines.append(f"| {v.get('net')} | {v.get('champion')} | "
                         + (f"{rb.get('verdict')} {100 * rb['diff']:+.1f} / {100 * rb['upper']:+.1f} / {rb['n']}"
                            if rb else "-") + " | "
                         + (f"{pr.get('verdict')} {100 * pr['diff']:+.1f} / {pr['z']} / {pr['n']}" if pr else "-")
                         + " |")
    lines += ["", "## Retrospective comparisons against the current champion", "",
              "Win-rate differences against common opponents, not head-to-head games. Unlike frozen gate verdicts, these update as more games arrive.",
              "The legacy comparison matches opponent and factions, not map/start/seed. Large-map 0/0 means untested.", "",
              "| network | experiment | diff | 95 % | games | large maps won | entropy | ref_kl |",
              "|---|---|---|---|---|---|---|---|"]
    from fa_train.rl_loop import _stratified
    pts = []
    tags = {}
    for row in journal:
        if row.get("checkpoint") and not row.get("fallback"):
            tags[Path(row["checkpoint"]).stem] = row
    for net, d in sorted(scan.items(), key=lambda kv: kv[1]["first"]):
        if net in ("start", champion):
            continue
        res = _stratified(scan.get(champion, {}).get("cells", {}), d["cells"])
        if not res or res[2] < 30:
            continue
        res = (res[0], res[1] * infl, res[2])
        row = tags.get(net, {})
        h = [e for e in (row.get("learn_rl") or []) if isinstance(e, dict) and "epoch" in e]
        pts.append((d["first"], net, res, row.get("experiment", "?")))
        lines.append(f"| {net} | {row.get('experiment', '?')} | {100 * res[0]:+.1f} | "
                     f"{100 * (res[0] - z * res[1]):+.1f}..{100 * (res[0] + z * res[1]):+.1f} | {res[2]} | "
                     f"{d['large'][0]}/{d['large'][1]} | {h[-1].get('ent', '') if h else ''} | "
                     f"{h[-1].get('ref_kl', '') if h else ''} |")
    lines += ["", "## Health", "", f"* disks: C: {free_gb(s['run']):.0f} GB free, D: "
              f"{free_gb(s['disk']['archive']):.0f} GB free", f"* loop exits (24 h): "
              f"{sum(1 for c in state.get('crashes', []) if time.time() - c['t'] < 86400)}",
              f"* newest round {state.get('newest_round', '?')}", "", "## Events (newest 15)", ""]
    ev = Path(s["dir"]) / "events.jsonl"
    if ev.exists():
        lines += [f"* {x}" for x in ev.read_text(encoding="utf-8").splitlines()[-15:]]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    png = out / "progress.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = {}
        for stamp, net, res, tag in pts[-80:]:
            c = colors.setdefault(tag, f"C{len(colors) % 10}")
            x = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
            ax.errorbar([x], [100 * res[0]], yerr=[100 * z * res[1]], fmt="o", color=c, ms=4, capsize=2,
                        label=tag if tag not in ax.get_legend_handles_labels()[1] else None)
        for c in state.get("champions", []):
            ax.axvline(datetime.fromisoformat(c["time"]), color="#2f855a", lw=0.8, ls="--")
        ax.axhline(0, color="#4a5568", lw=1)
        ax.set_ylabel(f"points vs {champion} (95 %)")
        ax.set_title("Each network vs the current champion; green lines: promotions")
        if pts:
            ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(png, dpi=110)
        plt.close(fig)
    except Exception as exc:                            # noqa: BLE001 -- no chart is no reason to stop
        log_event(s, "chart_failed", error=str(exc))
    img = base64.b64encode(png.read_bytes()).decode() if png.exists() else ""
    import html as _html
    body = _html.escape("\n".join(lines))
    (out / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>RL autopilot</title><style>body{font:14px system-ui;margin:12px;background:#fff;color:#111}"
        "pre{white-space:pre-wrap}img{max-width:100%}@media(prefers-color-scheme:dark){body{background:#111;"
        "color:#eee}}</style>" + (f"<img src='data:image/png;base64,{img}'>" if img else "") + f"<pre>{body}</pre>",
        encoding="utf-8")
    mirror = s["report"]["mirror_dir"]
    if mirror:
        try:
            Path(mirror).mkdir(parents=True, exist_ok=True)
            for f in ("report.html", "report.md", "progress.png"):
                if (out / f).exists():
                    shutil.copyfile(out / f, Path(mirror) / f)
        except OSError as exc:
            log_event(s, "mirror_failed", error=str(exc))
    return out / "report.html"


# ---------------------------------------------------------------------------------------------------- the autopilot

class Autopilot:
    def __init__(self, config: Path, dry_run: bool = False):
        self.config = config
        self.s = load_settings(config)
        self.dry = dry_run
        self.dir = Path(self.s["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = Path(self.s["run"])
        self.state_file = self.dir / "state.json"
        try:
            self.state = json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            self.state = {"mode": "start", "results": {}, "champions": [], "crashes": []}
        self.state.setdefault("crashes", [])
        self.alerts = Alerts(self.s, self.state)
        self.last_scan = 0.0
        self.scan: dict = {}

    # -- state ------------------------------------------------------------------------------------------------
    def save(self) -> None:
        write_json(self.state_file, self.state)
        write_json(self.dir / "heartbeat.json", {"pid": os.getpid(), "time": time.time(),
                                                 "mode": self.state.get("mode")})

    def names(self) -> dict:
        return names_of(Path(self.s["init"]), self.run / "checkpoints", self.dir / "sha_cache.json")

    def queue(self) -> list:
        q = self.dir / "queue.json"
        try:
            items = json.loads(q.read_text(encoding="utf-8-sig"))
            assert isinstance(items, list) and all(isinstance(e, dict) and "name" in e for e in items)
            self.state["queue_ok"] = items
            return items
        except (OSError, ValueError, AssertionError) as exc:
            if q.exists():
                self.alerts.send("queue", f"queue.json unreadable ({exc}); using the last good one")
            return self.state.get("queue_ok") or DEFAULT_QUEUE

    def champion(self) -> str:
        try:
            return json.loads((self.run / "champion.json").read_text())["name"]
        except (OSError, ValueError, KeyError):
            return self.state.get("champion") or "rl_089"

    def loop_exit(self) -> dict:
        """rl_loop_exit.json when it was written after the loop's last launch, else {}."""
        try:
            e = json.loads((self.run / "rl_loop_exit.json").read_text())
            t = datetime.fromisoformat(e.get("time") or e.get("started")).timestamp()
        except (OSError, ValueError, TypeError):
            return {}
        return e if t >= self.state.get("loop_started", 0) - 5 and e.get("code") is not None else {}

    def stop_is_ours(self) -> bool:
        try:
            return (self.run / "STOP").read_text(errors="replace").startswith(OWN_STOP)
        except OSError:
            return False

    def write_stop(self, why: str) -> None:
        """Our own STOP (a silent learn_rl, a wedged loop): rl_loop and learn_rl stop at their next check."""
        if not self.dry and not (self.run / "STOP").exists():
            (self.run / "STOP").write_text(f"{OWN_STOP}: {why}")
        self.state["own_stop_t"] = time.time()
        log_event(self.s, "own_stop", why=why)

    # -- the rl_loop child ------------------------------------------------------------------------------------
    def loop_procs(self, procs: list) -> list:
        return [p for p in find(procs, "fa_train.rl_loop") if str(self.run).lower() in p["cmd"].lower()]

    def start_experiment(self, exp: dict, champion: str, why: str) -> None:
        """Prepare the run's state for `exp` (curriculum level, imitation weight, player config) and launch."""
        st = self.state
        if "PLACEHOLDER" in json.dumps(exp):
            st.setdefault("results", {})[exp["name"]] = {"verdict": "INVALID", "why": "unfilled placeholder"}
            self.alerts.send("queue", f"{exp['name']} has an unfilled placeholder: skipped", "high")
            return
        level = st.get("champion_level", 0.6)
        # the imitation weight the experiment starts at: its own (rl_loop --distill), else the champion's
        loop_distill = (exp.get("loop") or {}).get("--distill")
        distill = float(loop_distill) if loop_distill else float(st.get("champion_distill") or 0.3)
        if not self.dry:
            write_json(self.run / "curriculum.json", {"m28_level": level})
            write_json(self.run / "adaptive_distill.json", {"distill": distill, "basis": champion})
            self.player_config(exp)
            (self.run / "YIELD").unlink(missing_ok=True)
        st["experiment"] = {"name": exp["name"], "started": datetime.now().isoformat(timespec="seconds"),
                            "t0": time.time(), "start_row": len(read_journal(self.run)), "extra_hours": 0.0,
                            "spec": exp}
        st["mode"] = "running"
        st.setdefault("results", {}).setdefault(exp["name"], {"started": st["experiment"]["started"]})
        log_event(self.s, "experiment_start", name=exp["name"], champion=champion, why=why)
        self.alerts.send("experiment", f"starting {exp['name']} from {champion}: {exp.get('why', '')} ({why})")
        self.launch_loop(champion)

    def launch_loop(self, champion: str) -> None:
        st = self.state
        e = st["experiment"]
        exp = e["spec"]
        # --start-from until the journal shows a row of this experiment (its start row or its first iteration)
        first = not any(r.get("experiment") == exp["name"] for r in read_journal(self.run)[e["start_row"]:])
        set_champ = st.pop("set_champion", None)
        cmd = loop_command(self.s, exp, champion, first, set_champ)
        st["loop_started"] = time.time()
        st.pop("expected_exit", None)                   # (it meant an exit before this launch)
        if self.dry:
            log_event(self.s, "dry_launch", cmd=cmd)
            return
        (self.run / "YIELD").unlink(missing_ok=True)   # only ever ours: it would end the loop at once
        if self.stop_is_ours():
            (self.run / "STOP").unlink(missing_ok=True)
        st["mode"] = "running"
        pid = launch(cmd, ROOT, Path(self.s["log"]))
        st["loop_pid"] = pid
        log_event(self.s, "loop_launch", pid=pid, first=first)

    def player_config(self, exp: dict) -> None:
        """The experiment's player-config overrides on both players; keys it does not set go back to base."""
        updates = {**self.s["base_player"], **(exp.get("player") or {})}
        if not updates:
            return
        from fa_train.rl_loop import local_config
        from fa_train.server_sync import set_config
        try:
            set_config(self.s["host"], self.s["server_home"], updates)
        except Exception as exc:                        # noqa: BLE001
            self.alerts.send("server", f"player config not set on the server: {exc}", "high")
        if self.s["local_home"]:
            try:
                local_config(Path(self.s["local_home"]), updates)
            except OSError as exc:
                self.alerts.send("local", f"local player config not written: {exc}")

    # -- one tick -----------------------------------------------------------------------------------------------
    def tick(self) -> None:
        st, s = self.state, self.s
        now = time.time()
        procs = processes()
        if procs is None:                               # no process list: decide nothing this tick
            self.save()
            return
        loops = self.loop_procs(procs)
        learns = learner_processes(procs, self.run)
        champion = self.champion()
        journal = read_journal(self.run)
        self.commands(champion)

        # champion changes (the gate's promotions, or a user command)
        if champion != st.get("champion"):
            if st.get("champion"):
                exp = st.get("experiment") or {}
                st.setdefault("results", {}).setdefault(exp.get("name", "?"), {}).setdefault("promotions", 0)
                st["results"][exp.get("name", "?")]["promotions"] += 1
                exp["run_promotions"] = exp.get("run_promotions", 0) + 1     # this turn's (requeue again)
                if exp:
                    exp["promo_t"] = now
                self.alerts.send("champion", f"new champion {champion} (was {st['champion']})", "high")
            self.keep_champion(champion, journal)
            st["champion"] = champion

        # a STOP: ours (a stall) is removed before the next launch; anyone else's holds training
        stop = self.run / "STOP"
        if stop.exists() and not self.stop_is_ours():
            if st.get("mode") != "gpu_hold":
                # the mode before the STOP comes back when it goes: a hold or finalize was lost (code review 5)
                st["mode_before_gpu_hold"] = st.get("mode")
                st["mode"], st["gpu_hold_since"] = "gpu_hold", now
                self.alerts.send("gpu", "STOP present (GPU watchdog or the user): training held", "high")
            g = s["gpu"]
            resumes = [t for t in st.get("gpu_resumes", []) if now - t < 86400]
            empty = not stop.read_text(errors="replace").strip()
            if (g["auto_resume"] and empty and not loops and not learns
                    and now - st["gpu_hold_since"] > g["quiet_hours"] * 3600
                    and len(resumes) < g["max_resumes_per_day"] and gpu_driver_errors(g["quiet_hours"]) == 0):
                if not self.dry:
                    stop.unlink(missing_ok=True)
                st["gpu_resumes"] = resumes + [now]
                before = st.pop("mode_before_gpu_hold", None)
                st["mode"] = before if before in ("hold", "finalized", "exhausted") else "running"
                st["expected_exit"] = True                 # the loop's exit on STOP is no crash
                self.alerts.send("gpu", f"no driver errors for {g['quiet_hours']} h: STOP removed, resuming", "high")
            self.housekeeping(procs, journal, champion)
            self.save()
            return
        if st.get("mode") == "gpu_hold":                # the STOP is gone (removed by the user): carry on
            before = st.pop("mode_before_gpu_hold", None)
            st["mode"] = before if before in ("hold", "finalized", "exhausted") else "running"
            st["expected_exit"] = True
        else:
            st.pop("mode_before_gpu_hold", None)
        if st.get("mode") != "hold":
            st.pop("shutdown_requested", None)          # a shutdown belongs to a hold

        # user hold / finalize / exhausted
        if st.get("mode") in ("hold", "finalized", "exhausted"):
            if loops and not (self.run / "YIELD").exists() and not self.dry:
                (self.run / "YIELD").write_text(st["mode"])
            self.housekeeping(procs, journal, champion)
            self.save()
            return

        end = s.get("end_time")
        if end and datetime.now() >= datetime.fromisoformat(end) - timedelta(hours=s["finalize_hours_before"]):
            self.finalize(loops, learns, champion)
            self.save()
            return

        exp = st.get("experiment")
        if not exp:
            nxt = next_experiment(self.queue(), st.get("results", {}), s["on_exhausted"])
            wanted = st.get("run_next")          # a "run" command: that entry next, even the one that just ended
            if wanted:
                entry = next((e for e in self.queue() if e["name"] == wanted), None)
                if entry is not None:
                    nxt = expand(entry, self.queue())
                else:                                    # the entry left the queue: forget the request
                    st.pop("run_next", None)
                    wanted = None
            if nxt is None:
                st["mode"] = "exhausted"
                self.alerts.send("queue", "queue exhausted: holding (the champion keeps playing)", "high")
            elif not learns and not loops:
                if wanted:
                    st.pop("run_next", None)
                    st.setdefault("results", {}).pop(wanted, None)
                self.start_experiment(nxt, champion, "command" if wanted else "queue")
            self.save()
            return

        # scan the games (every 10 minutes) and judge the experiment
        if now - self.last_scan > 600:
            self.scan = scan_rounds(self.run / "rounds", self.names(), s["gate_since"])
            self.last_scan = now
        rows = journal[exp["start_row"]:]
        # after a promotion inside the experiment, judge only the new champion's descendants
        after = next((i + 1 for i in range(len(rows) - 1, -1, -1) if rows[i].get("checkpoint") and not
                      rows[i].get("fallback") and Path(rows[i]["checkpoint"]).stem == champion), 0)
        nets = experiment_networks(rows, after)
        v = judge(experiment_rules(s["rules"], exp["spec"]), self.scan, champion, nets, fallbacks(rows, after), (now - exp["t0"]) / 3600,
                  (now - (exp.get("promo_t") or exp["t0"])) / 3600, exp.get("extra_hours", 0.0))
        st["verdict"] = {"kind": v.kind, "why": v.why, "numbers": v.numbers}
        alarm = entropy_alarm(rows, st.get("champion_entropy"), s["rules"]["entropy_alarm"])
        if alarm:
            self.alerts.send("entropy", alarm)

        if v.kind != "continue" or exp.get("skip"):
            self.end_experiment(v if v.kind != "continue" else Verdict("SKIP", "user skip", v.numbers), loops)
        elif not loops:
            self.restart(journal, learns)
        else:
            self.wedge(loops, learns)
        self.housekeeping(procs, journal, champion)
        self.save()

    def end_experiment(self, v: Verdict, loops: list) -> None:
        """Safe-point switch: YIELD, wait for rl_loop to exit (it finishes a running learn_rl first), then record."""
        st = self.state
        exp = st["experiment"]
        if loops:
            if not (self.run / "YIELD").exists() and not self.dry:
                (self.run / "YIELD").write_text(v.kind)
                log_event(self.s, "yield", experiment=exp["name"], verdict=v.kind, why=v.why)
            return                                      # next tick: still running, or gone
        if not self.dry:
            (self.run / "YIELD").unlink(missing_ok=True)
        pooled = v.numbers.get("pooled")
        r = st["results"].setdefault(exp["name"], {})
        r.update(ended=datetime.now().isoformat(timespec="seconds"), verdict=v.kind, why=v.why,
                 networks=v.numbers.get("networks"), fallbacks=v.numbers.get("fallbacks"),
                 score=round(100 * pooled[0], 1) if pooled else None)
        run_rows = read_journal(self.run)[exp["start_row"]:]
        st.setdefault("runs", []).append({"name": exp["name"], "started": exp["started"], "ended": r["ended"],
                                          "trained": len(experiment_networks(run_rows)),
                                          "promotions": exp.get("run_promotions", 0), "fallbacks": fallbacks(run_rows),
                                          "verdict": v.kind, "why": v.why, "numbers": v.numbers,
                                          "spec": exp["spec"]})
        if exp["spec"].get("_requeue"):
            r["requeued_done"] = True
        if exp["spec"].get("_rerun"):
            r["rerun"] = True
        if r.get("promotions") and v.kind in ("FAIL", "STALL", "MAX") and not exp["spec"].get("_requeue"):
            r["requeue"] = True                         # it produced a champion: worth another, shorter turn
        if exp.get("run_promotions") and v.kind in ("FAIL", "STALL", "MAX"):
            r["requeue"], r["requeued_done"] = True, False      # a champion again this turn: another turn after it
        log_event(self.s, "experiment_end", name=exp["name"], verdict=v.kind, why=v.why, numbers=v.numbers)
        self.alerts.send("experiment", f"{exp['name']} ended: {v.kind} {v.why}", "high")
        st["experiment"] = None

    def restart(self, journal: list, learns: list) -> None:
        """rl_loop is gone without a switch of ours: read why (rl_loop_exit.json), apply backoff and the crash
        budgets, relaunch."""
        st, r = self.state, self.s["restart"]
        if learns:                                      # an orphaned learn_rl still owns the GPU: wait for it
            st["mode"] = "waiting_orphan_learn_rl"
            return
        now = time.time()
        if now - st.get("loop_started", 0) < 180:      # just launched: the process list may lag
            return
        ex = self.loop_exit()
        code = ex.get("code")
        if st.get("last_exit_seen") != st.get("loop_started"):
            st["last_exit_seen"] = st.get("loop_started")
            if code == EXIT_GATE_EXHAUSTED:             # the loop's own rejection cap: this configuration is done
                self.end_experiment(Verdict("FAIL", f"gate exhausted: {ex.get('why', '')}"), [])
                return
            if code == EXIT_LOCKED:                     # another loop holds the run: it shows in the next list
                log_event(self.s, "loop_exit", code=code, why=ex.get("why"))
                return
            if st.pop("expected_exit", False) or code == EXIT_YIELD:
                pass
            else:
                reason = f"exit {code}: {ex.get('why', '')}" if ex else "no exit record (killed or crashed hard)"
                st["crashes"].append({"t": now, "exp": st["experiment"]["name"], "why": reason, "code": code})
                log_event(self.s, "loop_exit", why=reason)
                self.alerts.send("loop", f"rl_loop exited: {reason}")
        crashes = st["crashes"]
        exp_n = sum(1 for c in crashes if c["exp"] == st["experiment"]["name"]
                    and now - c["t"] < r["exp_window_hours"] * 3600)
        all_n = sum(1 for c in crashes if now - c["t"] < r["global_window_hours"] * 3600)
        tail = [row for row in journal[st["experiment"]["start_row"]:] if "exit" in row][-r["learn_failures_in_a_row"]:]
        if len(tail) == r["learn_failures_in_a_row"] and all(row["exit"] != 0 for row in tail):
            self.end_experiment(Verdict("ERROR", f"learn_rl failed {len(tail)} times in a row"), [])
            return
        if all_n >= r["global_crashes"]:
            st["mode"] = "hold"
            self.alerts.send("crashes", f"{all_n} loop exits in {r['global_window_hours']} h: holding", "high")
            return
        if exp_n >= r["exp_crashes"]:
            self.end_experiment(Verdict("ERROR", f"{exp_n} loop exits in {r['exp_window_hours']} h"), [])
            return
        wait = r["backoff_minutes"][min(exp_n, len(r["backoff_minutes"])) - 1] * 60 if exp_n else 0
        if crashes and now - crashes[-1]["t"] < wait:
            return
        if commit_gb() > self.s["memory"]["max_commit_gb"]:
            self.alerts.send("memory", "committed memory too high: relaunch postponed")
            return
        self.launch_loop(st["champion"])

    def wedge(self, loops: list, learns: list) -> None:
        """rl_loop alive but silent. A silent learn_rl gets our STOP (it exits at its next check; never killed: a
        possible CUDA process). With no GPU/unknown or managed learners, a silent loop gets our STOP first;
        still silent wedge_kill_minutes later (a hung ssh/tar never looks at STOP), its CPU tree is ended."""
        w = self.s["stall"]
        logs = [Path(self.s["log"])] + sorted(self.run.glob("rl_*.log"))[-1:]
        newest = max((p.stat().st_mtime for p in logs if p.exists()), default=time.time())
        quiet = (time.time() - newest) / 60
        st = self.state
        if quiet < w["wedge_minutes"] and not learns:
            st.pop("wedge_stop_t", None)
        if learns:
            if quiet > w["learn_log_hours"] * 60 and not (self.run / "STOP").exists():
                self.write_stop(f"learn_rl silent for {quiet / 60:.1f} h")
                self.alerts.send("learn_rl", f"learn_rl silent for {quiet / 60:.1f} h: STOP written (never killed)",
                                 "high")
            return
        if quiet > w["wedge_minutes"]:
            if "wedge_stop_t" not in st:
                st["wedge_stop_t"] = time.time()
                self.write_stop(f"rl_loop silent for {quiet:.0f} min")
                self.alerts.send("wedge", f"rl_loop silent for {quiet:.0f} min with no learn_rl: STOP written",
                                 "high")
            elif time.time() - st["wedge_stop_t"] > w["wedge_kill_minutes"] * 60:
                self.alerts.send("wedge", f"rl_loop still silent {w['wedge_kill_minutes']} min after STOP and no "
                                          f"learn_rl runs: ending its process tree (CPU only)", "high")
                if not self.dry:
                    for p in loops:
                        kill_tree(p["pid"])
                st.pop("wedge_stop_t", None)

    def finalize(self, loops: list, learns: list, champion: str) -> None:
        st = self.state
        if loops or learns:
            if loops and not (self.run / "YIELD").exists() and not self.dry:
                (self.run / "YIELD").write_text("finalize")
            return
        if st.get("mode") != "finalized":
            from fa_train.rl_loop import local_push
            from fa_train.server_sync import push
            ck = self.run / "checkpoints" / f"{champion}.pt"
            if not self.dry:
                try:
                    push(self.s["host"], self.s["server_home"], ck)
                except Exception as exc:                # noqa: BLE001
                    self.alerts.send("server", f"final push failed: {exc}", "high")
                if self.s["local_home"]:
                    try:
                        local_push(Path(self.s["local_home"]), ck)
                    except OSError as exc:
                        self.alerts.send("local", f"final local push failed: {exc}", "high")
            st["mode"] = "finalized"
            self.alerts.send("final", f"finalized: {champion} on all lanes; report written", "high")
            self.write(champion)

    def keep_champion(self, champion: str, journal: list) -> None:
        """A copy of every champion outside the checkpoints folder, its curriculum level, imitation weight and
        training entropy (an experiment starts from those)."""
        src = self.run / "checkpoints" / f"{champion}.pt"
        dst = self.dir / "champions" / f"{len(self.state.get('champions', [])):03d}_{champion}.pt"
        if src.exists() and not self.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        row = next((r for r in reversed(journal) if r.get("checkpoint") and Path(r["checkpoint"]).stem == champion
                    and not r.get("fallback")), {})
        h = [e for e in (row.get("learn_rl") or []) if isinstance(e, dict) and "epoch" in e]
        self.state["champion_entropy"] = h[-1].get("ent") if h else self.state.get("champion_entropy")
        if row.get("m28_level"):
            self.state["champion_level"] = row["m28_level"]
        if row.get("distill"):
            self.state["champion_distill"] = row["distill"]
        self.state.setdefault("champions", []).append({"time": datetime.now().isoformat(timespec="seconds"),
                                                       "name": champion, "copy": str(dst)})

    def housekeeping(self, procs: list, journal: list, champion: str) -> None:
        st, s, now = self.state, self.s, time.time()
        # the GPU watchdog runs beside training, hidden
        if s["gpu_watchdog_ps1"] and Path(s["gpu_watchdog_ps1"]).exists() and not find(procs, "gpu_watchdog.ps1") \
                and not self.dry:
            launch(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File",
                    s["gpu_watchdog_ps1"], "-StopFile", str(self.run / "STOP")], ROOT, None)
            log_event(s, "gpu_watchdog_launch")
        # the local lane (CPU games): relaunched when gone or silent, never while its own STOP is there
        home = Path(s["local_home"]) if s["local_home"] else None
        if home and not (home / "STOP").exists() and st.get("mode") not in ("finalized",):
            players = [p for p in find(procs, "fa_train.player") if str(home).lower() in p["cmd"].lower()]
            plog = home / "player.log"
            age_h = (now - plog.stat().st_mtime) / 3600 if plog.exists() else 0
            paused = (home / "PAUSE").exists()          # (rl_loop pauses it when a disk is low: no log then)
            if not players or (age_h > s["stall"]["local_restart_hours"] and not paused):
                if players:
                    self.alerts.send("local", f"local lane silent {age_h:.1f} h: restarting it")
                    if not self.dry:
                        for p in players:
                            kill_tree(p["pid"])     # CPU games only (device cpu); never a CUDA process
                if not self.dry:
                    cmd = s["local_player_cmd"] or [s["python"], "-u", "-m", "fa_train.player", "--home", str(home),
                                                    "--max-lanes", "1", "--lane-offset", "10"]
                    launch(cmd, ROOT, home / "player.log")
                log_event(s, "local_lane_launch")
        # rounds arriving (their ARRIVED stamps)
        rounds = sorted((self.run / "rounds").glob("lane*_*"), key=lambda p: p.name.split("_", 1)[-1])[-10:]
        if rounds:
            from fa_train.rl_loop import arrived_at
            newest = max(arrived_at(r) for r in rounds)
            st["newest_round"] = datetime.fromtimestamp(newest).isoformat(timespec="minutes")
            if (now - newest) / 3600 > s["stall"]["rounds_alert_hours"] and st.get("mode") == "running":
                self.alerts.send("rounds", f"no new round for {(now - newest) / 3600:.1f} h (server down?)", "high")
        # disks: rl_loop pauses its intake under its floor and applies the storage budget; here only the alert
        if now - st.get("disk_t", 0) > s["disk"]["check_minutes"] * 60:
            st["disk_t"] = now
            c, d = free_gb(s["run"]), free_gb(s["disk"]["archive"])
            if min(c, d) < s["disk"]["warn_gb"]:
                self.alerts.send("disk", f"disk low: C: {c:.0f} GB, D: {d:.0f} GB free (rl_loop pauses new games "
                                         f"under its --min-free-gb)", "high")
        # the report
        rep = s["report"]
        due_daily = datetime.now().hour == rep["daily_hour"] and st.get("daily_done") != datetime.now().date().isoformat()
        if due_daily or now - st.get("report_t", 0) > rep["every_hours"] * 3600 or len(journal) != st.get("report_rows"):
            self.write(champion)
            st["report_t"], st["report_rows"] = now, len(journal)
            if due_daily:
                st["daily_done"] = datetime.now().date().isoformat()
                v = st.get("verdict") or {}
                self.alerts.send("daily", f"champion {champion}; experiment {(st.get('experiment') or {}).get('name')}"
                                          f"; {v.get('kind', '')} {v.get('why', '')}")

    def write(self, champion: str) -> None:
        if not self.scan:
            self.scan = scan_rounds(self.run / "rounds", self.names(), self.s["gate_since"])
        v = self.state.get("verdict")
        write_report(self.s, self.state, self.scan, read_journal(self.run), champion,
                     Verdict(v["kind"], v["why"], v.get("numbers", {})) if v else None)

    # -- user commands ------------------------------------------------------------------------------------------
    def commands(self, champion: str) -> None:
        cmds = ntfy_commands(self.s, self.state)
        f = self.dir / "override.json"
        if f.exists():
            try:
                # utf-8-sig: PowerShell 5.1 writes a BOM, and the command was dropped silently (code review 6)
                c = json.loads(f.read_text(encoding="utf-8-sig"))
                cmds += c if isinstance(c, list) else [c]
                f.replace(self.dir / f"override.{int(time.time())}.done.json")
            except ValueError as exc:
                self.alerts.send("override", f"override.json unreadable: {exc}", "high")
                log_event(self.s, "override_unreadable", error=str(exc)[:200])
                f.replace(self.dir / f"override.{int(time.time())}.bad.json")
        for c in cmds:
            if isinstance(c, dict):
                self.apply_command(c, champion)

    def apply_command(self, c: dict, champion: str) -> None:
        st = self.state
        cmd = str(c.get("cmd", "")).lower()
        exp = st.get("experiment")
        if cmd == "skip" and exp:
            exp["skip"] = True
        elif cmd in ("hold", "shutdown"):
            st["mode"] = "hold"
            if cmd == "shutdown":
                st["shutdown_requested"] = True
            else:
                st.pop("shutdown_requested", None)
        elif cmd == "resume" and st.get("mode") in ("hold", "exhausted"):
            st.pop("shutdown_requested", None)
            st["mode"] = "running"
            st["expected_exit"] = True
        elif cmd == "extend" and exp:
            try:
                exp["extra_hours"] = exp.get("extra_hours", 0.0) + float(c.get("hours", 24))
            except (TypeError, ValueError):
                self.alerts.send("override", f"ignored command {c}")
                return
        elif cmd == "run" and any(e["name"] == c.get("experiment") for e in self.queue()):
            # run_next: the named entry starts next even when it is the one running now (its end records SKIP,
            # and the queue order alone then started the entry after it: 2026-10-26 X3d -> X10)
            st["run_next"] = c["experiment"]
            st.setdefault("results", {}).pop(c["experiment"], None)
            q = self.queue()
            q.sort(key=lambda e: e["name"] != c["experiment"])
            if not self.dry:
                write_json(self.dir / "queue.json", q)
            if exp:
                exp["skip"] = True
        elif cmd == "champion" and isinstance(c.get("name"), str) and c["name"].startswith("rl_") \
                and (self.run / "checkpoints" / f"{Path(c['name']).name}.pt").exists():
            st["set_champion"] = Path(c["name"]).name   # rl_loop --set-champion at the next launch
            if exp:
                exp["skip"] = True                      # the experiment ends and runs again from the new one
                st["results"].pop(exp["name"], None)
                st["run_next"] = exp["name"]            # (its end records SKIP: without this the queue moved on)
        elif cmd == "report":
            self.write(champion)
        else:
            self.alerts.send("override", f"ignored command {c}")
            return
        log_event(self.s, "command", **c)

    def shutdown_ready(self, procs: list | None) -> bool:
        """An operator shutdown drains through YIELD and never stops a learner.

        The saved hold survives the supervisor's exit (and a watchdog restart).
        Unknown process state cannot establish that a shutdown is safe.
        """
        return bool(not self.dry and self.state.get("shutdown_requested")
                    and self.state.get("mode") == "hold" and procs is not None
                    and not self.loop_procs(procs) and not learner_processes(procs, self.run))


def _lock(path: Path):
    f = open(path, "a+")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--once", action="store_true", help="one tick, then exit (tests, a manual look)")
    ap.add_argument("--dry-run", action="store_true", help="decide and report, launch/kill/delete nothing")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    pilot = Autopilot(a.config, a.dry_run)
    # one autopilot per run (the watchdog may race a slow start): an OS lock, released when the process dies
    lock = _lock(pilot.dir / "autopilot.lock")
    if lock is None:
        sys.exit("another autopilot holds the lock")
    queue = pilot.dir / "queue.json"
    if not queue.exists():
        write_json(queue, DEFAULT_QUEUE)
    log_event(pilot.s, "autopilot_start", pid=os.getpid(), dry_run=a.dry_run)
    while True:
        try:
            pilot.tick()
        except Exception as exc:                        # noqa: BLE001 -- the supervisor must outlive its bugs
            import traceback
            log_event(pilot.s, "tick_error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
            pilot.alerts.send("autopilot", f"tick error: {type(exc).__name__}: {exc}")
            try:
                pilot.save()
            except OSError:
                pass
        if a.once:
            break
        if pilot.state.get("shutdown_requested") and pilot.shutdown_ready(processes()):
            pilot.state.pop("shutdown_requested", None)
            pilot.save()
            log_event(pilot.s, "autopilot_shutdown", pid=os.getpid(), reason="operator; drained and held")
            break
        time.sleep(pilot.s["tick_seconds"])


if __name__ == "__main__":
    main()
