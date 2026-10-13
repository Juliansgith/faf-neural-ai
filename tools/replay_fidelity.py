"""Measure how faithfully our engine reproduces FAF's official one, by replay.

    replay_fidelity.py run [--map SCMP_016] [--seed 7] [--minutes 12] [--out DIR]

1. record:   FAF's ForgedAlliance.exe plays an all-AI skirmish with /synclog (every beat's checksum
             inputs traced to a file) and writes a replay.
2. playback: our main.exe plays that replay with /synclog. The replay carries the official run's
             per-beat checksums, so the engine itself reports the first beat it computes differently.
3. compare:  the two runs' per-beat trace files are compared: first beat whose final checksum
             differs, then the first differing line inside it (entity id, health, position bits ...).

Sub-commands `record`, `playback` and `compare` run the steps separately.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai" / "python"))
from fa_env.bridge import DEFAULT_BIN_DIR, make_private_prefs, write_init_file  # noqa: E402

OFFICIAL_EXE = DEFAULT_BIN_DIR / "ForgedAlliance.exe"
OUR_EXE = DEFAULT_BIN_DIR / "main.exe"
REPLAYS_DIR = Path(os.environ["USERPROFILE"]) / "Documents" / "My Games" / "Gas Powered Games" / \
    "Supreme Commander Forged Alliance" / "replays"

# ------------------------------------------------------------------------------------------------ helpers
u32 = ctypes.WinDLL("user32", use_last_error=True)
EnumProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
WM_KEYDOWN, WM_KEYUP, VK_ADD = 0x0100, 0x0101, 0x6B


def speed_up(pid: int, presses: int = 12) -> None:
    """Tap numpad '+' in the game's windows: raises the game speed to its maximum."""
    hwnds = []

    def child(h, _):
        hwnds.append(h)
        return True

    def top(h, _):
        p = wt.DWORD()
        u32.GetWindowThreadProcessId(h, ctypes.byref(p))
        if p.value == pid:
            hwnds.append(h)
            u32.EnumChildWindows(h, EnumProc(child), 0)
        return True

    u32.EnumWindows(EnumProc(top), 0)
    scan = u32.MapVirtualKeyW(VK_ADD, 0)
    for _ in range(presses):
        for h in hwnds:
            if u32.IsWindowVisible(h):
                u32.PostMessageW(h, WM_KEYDOWN, VK_ADD, 1 | (scan << 16))
                u32.PostMessageW(h, WM_KEYUP, VK_ADD, 1 | (scan << 16) | (1 << 30) | (1 << 31))
        time.sleep(0.05)


def launch(exe: Path, out: Path, name: str, session_args: list[str]) -> tuple[subprocess.Popen, Path, list[Path]]:
    run_id = f"{name}_{uuid.uuid4().hex[:6]}"
    log_path = out / f"{name}.log"
    prefs_name, prefs_path = make_private_prefs(run_id, prefix="fidelity")
    init_path = write_init_file(out / f"init_{name}.lua")
    args = [str(exe), "/init", str(init_path), "/nobugreport", "/nosound", "/nomovie", "/log", str(log_path),
            "/windowed", "800", "600", "/prefs", prefs_name, *session_args]
    (out / f"{name}.args.json").write_text(json.dumps(args, indent=1))
    return subprocess.Popen(args, cwd=str(DEFAULT_BIN_DIR)), log_path, [prefs_path]


def run_with_speedup(proc: subprocess.Popen, timeout: float, stop_when=None) -> bool:
    """Wait for the process, keeping the game speed up; True if it exited on its own."""
    t0 = time.time()
    next_tap = t0 + 20
    while proc.poll() is None and time.time() - t0 < timeout:
        time.sleep(1)
        if time.time() >= next_tap:
            speed_up(proc.pid)
            next_tap = time.time() + 15
        if stop_when and stop_when():
            break
    exited = proc.poll() is not None
    if not exited:
        proc.kill()
        proc.wait()
    return exited


def newest_replay(since: float) -> Path | None:
    found = [p for p in REPLAYS_DIR.rglob("*.SCFAReplay") if p.stat().st_mtime >= since]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def synclog_dir(root: Path) -> Path | None:
    """The per-run folder /synclog creates under its argument."""
    dirs = [d for d in root.iterdir() if d.is_dir()] if root.exists() else []
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


# ------------------------------------------------------------------------------------------------ steps
def record(args) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    session = ["/map", args.map, "/seed", str(args.seed), "/faction", str(args.faction),
               "/agentaifaction", str(args.ai_faction), "/agentai", args.ai, "/agentallai",
               "/synclog", str(out / "synclog_official"), "/exitongameover"]
    proc, log_path, leftovers = launch(OFFICIAL_EXE, out, "official", session)
    exited = run_with_speedup(proc, args.minutes * 60)
    for p in leftovers:
        p.unlink(missing_ok=True)
    replay = newest_replay(started)
    if replay is None:
        raise SystemExit(f"no replay written (log: {log_path})")
    target = out / "recorded.SCFAReplay"
    shutil.copyfile(replay, target)
    game_time = last_game_time(log_path)
    print(f"recorded {target} ({target.stat().st_size // 1024} KB), official run "
          f"{'ended on its own' if exited else 'stopped at the time limit'}, last game time {game_time}")
    return target


def playback(args) -> Path:
    out = Path(args.out)
    replay = Path(args.replay) if getattr(args, "replay", None) else out / "recorded.SCFAReplay"
    session = ["/replay", str(replay), "/synclog", str(out / "synclog_ours"), "/exitongameover", *args.extra]
    proc, log_path, leftovers = launch(OUR_EXE, out, "ours", session)
    mismatch_seen = [None]

    def first_mismatch_long_ago():
        if not log_path.exists():
            return False
        text = log_path.read_text(encoding="latin-1", errors="replace")
        if "mismatched" in text and mismatch_seen[0] is None:
            mismatch_seen[0] = time.time()
        return mismatch_seen[0] is not None and time.time() - mismatch_seen[0] > 20

    run_with_speedup(proc, args.minutes * 60, stop_when=first_mismatch_long_ago)
    for p in leftovers:
        p.unlink(missing_ok=True)
    print(f"playback done, game time {last_game_time(log_path)}; log {log_path}")
    return log_path


def last_game_time(log_path: Path) -> str:
    if not log_path.exists():
        return "?"
    times = re.findall(r"Game time: (\d+:\d+:\d+)", log_path.read_text(encoding="latin-1", errors="replace"))
    return times[-1] if times else "?"


BEAT_RE = re.compile(r"beat(\d{5})\.log$")
FINAL_RE = re.compile(r"beat (\d+) final checksum: ([0-9a-fA-F]+)")


def beat_checksums(folder: Path) -> dict[int, str]:
    sums = {}
    for f in folder.glob("beat*.log"):
        for m in FINAL_RE.finditer(f.read_text(encoding="latin-1", errors="replace")):
            sums[int(m.group(1))] = m.group(2).lower()
    return sums


def compare(args) -> dict:
    out = Path(args.out)
    official, ours = synclog_dir(out / "synclog_official"), synclog_dir(out / "synclog_ours")
    report = {"engine_reported_mismatches": []}
    ours_log = out / "ours.log"
    if ours_log.exists():
        report["engine_reported_mismatches"] = re.findall(
            r"Checksum for beat (\d+) mismatched", ours_log.read_text(encoding="latin-1", errors="replace"))[:5]
    if official is None or ours is None:
        print("missing synclog folders", official, ours)
        return report
    a, b = beat_checksums(official), beat_checksums(ours)
    common = sorted(set(a) & set(b))
    report.update(official_beats=len(a), our_beats=len(b), common_beats=len(common))
    first = next((beat for beat in common if a[beat] != b[beat]), None)
    report["first_differing_beat"] = first
    report["matching_beats_before_it"] = sum(1 for beat in common if first is None or beat < first)
    print(json.dumps(report, indent=1))
    if first is None:
        return report

    fa, fb = official / f"beat{first:05d}.log", ours / f"beat{first:05d}.log"
    la = fa.read_text(encoding="latin-1", errors="replace").splitlines()
    lb = fb.read_text(encoding="latin-1", errors="replace").splitlines()
    idx = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
    print(f"\nfirst differing line in beat {first} (line {idx + 1}):")
    for label, lines in (("official", la), ("ours", lb)):
        print(f"--- {label}")
        for i in range(max(0, idx - 12), min(len(lines), idx + 6)):
            print(f"{'>>' if i == idx else '  '} {lines[i]}")
    report["first_differing_line"] = {"official": la[idx] if idx < len(la) else None,
                                      "ours": lb[idx] if idx < len(lb) else None}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "record", "playback", "compare"])
    ap.add_argument("--out", default=str(Path(os.environ.get("TEMP", ".")) / "fa_fidelity"))
    ap.add_argument("--map", default="SCMP_016")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--faction", type=int, default=1)
    ap.add_argument("--ai-faction", type=int, default=2)
    ap.add_argument("--ai", default="rush")
    ap.add_argument("--minutes", type=float, default=6.0, help="wall-clock limit per run")
    ap.add_argument("--replay")
    ap.add_argument("--extra", nargs="*", default=[], help="extra engine switches for the playback run")
    args = ap.parse_args()
    if args.step in ("run", "record"):
        record(args)
    if args.step in ("run", "playback"):
        playback(args)
    if args.step in ("run", "compare"):
        compare(args)


if __name__ == "__main__":
    main()
