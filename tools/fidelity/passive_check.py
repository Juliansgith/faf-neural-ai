"""Behavioural check: does an FAF AI kill a passive player, and when -- in FAF's exe and in ours?

    passive_check.py --exe official|ours --ai medium --seed 1 --minutes 20

Launches a skirmish (SCMP_016, player UEF in army 1 doing nothing, AI of the given personality), keeps the
game speed up, and reports whether the player was defeated (the game ends with /exitongameover) or
survived to the time limit, with the last game time seen in the log. The same test through the agent
bridge is `evaluate.py --agent noop`; this one needs no bridge, so it runs on the official exe too.
"""
import argparse
import re
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ai" / "python"))
sys.path.insert(0, str(REPO / "tools"))
from fa_env.bridge import DEFAULT_BIN_DIR, make_private_prefs, write_init_file  # noqa: E402
from replay_fidelity import speed_up  # noqa: E402
import subprocess  # noqa: E402

GAME_TIME = re.compile(r"Game time: (\d+):(\d+):(\d+)")


def last_game_seconds(log: Path) -> int:
    if not log.exists():
        return 0
    times = GAME_TIME.findall(log.read_text(encoding="latin-1", errors="replace"))
    if not times:
        return 0
    h, m, s = map(int, times[-1])
    return h * 3600 + m * 60 + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", choices=["official", "ours"], default="official")
    ap.add_argument("--ai", default="medium")
    ap.add_argument("--ai-faction", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--wall-limit", type=float, default=900, help="seconds")
    ap.add_argument("--out", default=str(REPO.parent / "_fidelity" / "passive"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.exe}_{a.ai}_f{a.ai_faction}_s{a.seed}"
    exe = DEFAULT_BIN_DIR / ("ForgedAlliance.exe" if a.exe == "official" else "main.exe")
    prefs_name, prefs_path = make_private_prefs(f"{name}_{uuid.uuid4().hex[:6]}", prefix="passive")
    init = write_init_file(out / f"init_{name}.lua")
    log = out / f"{name}.log"
    log.unlink(missing_ok=True)
    args = [str(exe), "/init", str(init), "/nobugreport", "/nosound", "/nomovie", "/log", str(log),
            "/windowed", "640", "480", "/prefs", prefs_name, "/map", "SCMP_016", "/seed", str(a.seed),
            "/faction", "1", "/agentaifaction", str(a.ai_faction), "/agentai", a.ai, "/exitongameover",
            "/victory", "demoralization"]
    proc = subprocess.Popen(args, cwd=str(DEFAULT_BIN_DIR))
    t0 = time.time()
    next_tap = t0 + 20
    try:
        while proc.poll() is None and time.time() - t0 < a.wall_limit:
            time.sleep(2)
            if time.time() >= next_tap:
                speed_up(proc.pid)
                next_tap = time.time() + 10
            if last_game_seconds(log) >= a.minutes * 60:
                break
    finally:
        ended = proc.poll() is not None
        if not ended:
            proc.kill()
            proc.wait()
        prefs_path.unlink(missing_ok=True)
    seconds = last_game_seconds(log)
    verdict = "ended (player defeated or game over)" if ended else "survived to the limit"
    print(f"{name}: {verdict} at game time {seconds // 60}:{seconds % 60:02d}, wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
