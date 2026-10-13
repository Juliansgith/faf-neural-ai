"""Play a replay under beat_probe with FAF's ForgedAlliance.exe or our main.exe.

    probe_run.py official|ours [--beats N] [--fast] [--watch A:B] [--bp ADDR|SYMBOL --bp-window A:B]
                               [--mount DIR] [--tag name] [--replay FILE]

Writes <work>/probe/<which><tag>.jsonl (one RNG fingerprint per beat, plus draw sites / breakpoint hits)
and the game log next to it. <work> is ../_fidelity next to this repository, or FA_FIDELITY_DIR.
--mount puts a folder first on the Lua search path (e.g. the tracer from make_lua_trace.py); --fast keeps
tapping numpad + so the replay runs at maximum speed.
"""
import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("FA_FIDELITY_DIR", REPO.parent / "_fidelity"))
sys.path.insert(0, str(REPO / "ai" / "python"))
sys.path.insert(0, str(REPO / "tools"))
from fa_env.bridge import DEFAULT_BIN_DIR, make_private_prefs, write_init_file  # noqa: E402
import beat_probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["official", "ours"])
    ap.add_argument("--beats", type=int, default=80)
    ap.add_argument("--watch", help="record every twister draw site during beats A:B")
    ap.add_argument("--bp", action="append", help="extra breakpoint (hex address for official, PDB symbol for ours)")
    ap.add_argument("--bp-window", help="record --bp hits only during beats A:B")
    ap.add_argument("--tag", default="")
    ap.add_argument("--replay", default=str(WORK / "run1" / "recorded.SCFAReplay"))
    ap.add_argument("--extra", nargs="*", default=[], help="extra game switches")
    ap.add_argument("--fast", action="store_true", help="tap numpad + to raise the replay speed")
    ap.add_argument("--mount", help="folder mounted first on the Lua search path")
    a = ap.parse_args()

    out = WORK / "probe"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.which}{a.tag}"
    exe = DEFAULT_BIN_DIR / ("ForgedAlliance.exe" if a.which == "official" else "main.exe")
    prefs_name, prefs_path = make_private_prefs(f"{name}_{uuid.uuid4().hex[:6]}", prefix="probe")
    init_path = write_init_file(out / f"init_{name}.lua")
    if a.mount:
        with open(init_path, "a", encoding="utf-8") as f:
            f.write(f"table.insert(path, 1, {{ dir = [[{Path(a.mount).as_posix()}]], mountpoint = '/' }})\n")
    game_args = ["/init", str(init_path), "/nobugreport", "/nosound", "/nomovie", "/log", str(out / f"{name}.log"),
                 "/windowed", "800", "600", "/prefs", prefs_name, "/replay", a.replay, "/exitongameover", *a.extra]
    argv = ["--exe", str(exe), "--out", str(out / f"{name}.jsonl"), "--beats", str(a.beats)]
    if a.watch:
        argv += ["--watch", a.watch]
    for b in a.bp or []:
        argv += ["--bp", b]
    if a.bp_window:
        argv += ["--bp-window", a.bp_window]
    sys.argv = ["beat_probe.py", *argv, "--", *game_args]

    if a.fast:
        from replay_fidelity import speed_up

        def keep_fast(pid):
            def loop():
                time.sleep(12)
                while True:
                    speed_up(pid)
                    time.sleep(10)
            threading.Thread(target=loop, daemon=True).start()

        beat_probe.ON_STARTED = keep_fast
    try:
        beat_probe.main()
    finally:
        prefs_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
