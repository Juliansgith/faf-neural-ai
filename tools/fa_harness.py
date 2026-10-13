"""Run one Forged Alliance executable (original or rebuilt) on a scripted session and summarise it.

usage: fa_harness.py <label> <exe name in FAF bin> <outdir> <timeout seconds> -- <session args...>

Launches windowed with no sound/movies and a private prefs file, taps numpad '+' every 20 s so the
sim runs at maximum speed, and stops the process at the timeout. Writes <outdir>/summary.json with
the exit status, run time, Lua errors, and game-flow lines (defeats, game over) found in the log.
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import shutil
import subprocess
import sys
import time

BIN = r"C:\ProgramData\FAForever\bin"
PREFS_DIR = os.path.join(os.environ["LOCALAPPDATA"], "Gas Powered Games", "Supreme Commander Forged Alliance")

label, exe_name, outdir, timeout = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3]), int(sys.argv[4])
session_args = sys.argv[sys.argv.index("--") + 1:]
os.makedirs(outdir, exist_ok=True)
log_path = os.path.join(outdir, "game.log")

# Private prefs per run so neither the user's Game.prefs nor a parallel run is touched.
prefs_name = f"claudetest_{label}.prefs"
src_prefs = os.path.join(PREFS_DIR, "Game.prefs")
if os.path.exists(src_prefs):
    shutil.copyfile(src_prefs, os.path.join(PREFS_DIR, prefs_name))

u32 = ctypes.WinDLL("user32", use_last_error=True)
EnumProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
WM_KEYDOWN, WM_KEYUP, VK_ADD = 0x0100, 0x0101, 0x6B


def speed_up(pid, presses=12):
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


args = [os.path.join(BIN, exe_name), "/init", os.path.join(BIN, "init.lua"), "/nobugreport",
        "/log", log_path, "/windowed", "1024", "640", "/nosound", "/nomovie", "/prefs", prefs_name,
        *session_args]
t0 = time.time()
proc = subprocess.Popen(args, cwd=BIN)
next_tap = t0 + 30
while proc.poll() is None and time.time() - t0 < timeout:
    time.sleep(2)
    if time.time() >= next_tap:
        speed_up(proc.pid)
        next_tap = time.time() + 20
timed_out = proc.poll() is None
if timed_out:
    proc.kill()
    proc.wait()
elapsed = time.time() - t0

text = open(log_path, encoding="latin-1", errors="replace").read() if os.path.exists(log_path) else ""
lines = text.splitlines()
err_re = re.compile(r"(?i)(error running|attempt to (call|index|perform|compare|concatenate)|stack traceback|"
                    r"script error|lua error|\berror:|exception|crash|assert)")
flow_re = re.compile(r"(?i)(defeat|victory|game ?over|has been (defeated|abandoned)|GRN:|draw|armies? .*(won|lost))")
errors = [l for l in lines if err_re.search(l)]
unique_errors = []
for l in errors:
    key = re.sub(r"0x[0-9A-Fa-f]+|\d+", "#", l.strip())[:200]
    if key not in unique_errors:
        unique_errors.append(key)
summary = {
    "label": label, "exe": exe_name, "args": session_args,
    "exit_code": None if timed_out else proc.returncode,
    "timed_out": timed_out, "wall_seconds": round(elapsed),
    "log_lines": len(lines),
    "run_time_line": next((l for l in reversed(lines) if "Run time" in l), None),
    "error_line_count": len(errors), "unique_errors": unique_errors[:80],
    "flow_lines": [l.strip()[:200] for l in lines if flow_re.search(l)][:80],
    "last_lines": [l[:200] for l in lines[-15:]],
}
with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: summary[k] for k in ("label", "exit_code", "timed_out", "wall_seconds", "log_lines",
                                            "error_line_count", "run_time_line")}))
