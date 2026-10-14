"""Checks that a training game never takes the focus or shows its window on screen, from launch until it is
minimized (the no-focus-steal rule; engine /agentbackground). Launches one minimized game through FAEnv and
samples GetForegroundWindow and the game's visible windows every 10 ms.

    python focus_watch.py [--exe PATH] [--steps 20]

Prints how often the game owned the foreground window and how often one of its windows intersected a monitor.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai" / "python"))
from fa_env import FAEnv, GameConfig  # noqa: E402
from fa_env import bridge as bridge_mod  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[2]   # the folder holding this repo

user32 = ctypes.windll.user32


def game_windows(pid):
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def visit(hwnd, _):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


def on_screen(hwnd):
    if user32.IsIconic(hwnd):
        return False
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)     # virtual screen
    vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    return r.right > vx and r.left < vx + vw and r.bottom > vy and r.top < vy + vh and r.right > r.left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(WORKSPACE / "_enginetest" / "bin_e" / "main.exe"))
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    exe = Path(a.exe)
    stats = {"samples": 0, "foreground": 0, "on_screen": 0}
    holder = {}
    stop = threading.Event()
    orig = bridge_mod.GameProcess.__init__

    def init(self, cfg, port):
        orig(self, cfg, port)
        holder["pid"] = self.process.pid

    bridge_mod.GameProcess.__init__ = init

    def watch():
        while not stop.is_set():
            pid = holder.get("pid")
            if pid:
                stats["samples"] += 1
                fg = user32.GetForegroundWindow()
                owner = wt.DWORD()
                user32.GetWindowThreadProcessId(fg, ctypes.byref(owner))
                stats["foreground"] += owner.value == pid
                stats["on_screen"] += any(on_screen(h) for h in game_windows(pid))
            time.sleep(0.01)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    env = FAEnv(GameConfig(map="SCMP_016", ai="easy", seed=3, minimized=True, exe=exe, bin_dir=exe.parent,
                           script=HERE.parent / "ai" / "agentbridge.lua"), step_ticks=10)
    try:
        env.reset()
        for _ in range(a.steps):
            env.step()
    finally:
        stop.set()
        env.close()
    print(f"samples {stats['samples']}: game in the foreground {stats['foreground']}, "
          f"a game window on screen {stats['on_screen']}")
    sys.exit(1 if stats["foreground"] or stats["on_screen"] else 0)


if __name__ == "__main__":
    main()
