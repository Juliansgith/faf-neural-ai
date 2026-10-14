"""Does the sim replay identically? Plays the same game twice with an agent that never acts (only the AI opponent
plays) and compares a fingerprint of every observation.

    python determinism_probe.py [--exe PATH] [--minutes 8] [--ai adaptive] [--seed 32]

Prints the first tick at which the two runs differ (or that they never do).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai" / "python"))
from fa_env import FAEnv, GameConfig  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[2]   # the folder holding this repo


def run(exe: Path, minutes: float, ai: str, seed: int, omniscient: bool) -> list:
    env = FAEnv(GameConfig(map="SCMP_016", ai=ai, ai_faction=2, seed=seed, minimized=True, exe=exe, bin_dir=exe.parent,
                           script=HERE.parent / "ai" / "agentbridge.lua"), step_ticks=20, omniscient=omniscient)
    out = []
    try:
        obs, _ = env.reset()
        while obs.get("tick", 0) < minutes * 600:
            obs, _, done, trunc, _ = env.step()
            out.append((obs.get("tick"), hashlib.blake2b(json.dumps(obs, sort_keys=True).encode(),
                                                          digest_size=8).hexdigest(), obs))
            if done or trunc:
                break
    finally:
        env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(WORKSPACE / "_enginetest" / "bin_e" / "main.exe"))
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--ai", default="adaptive")
    ap.add_argument("--seed", type=int, default=32)
    a = ap.parse_args()
    exe = Path(a.exe)
    r1 = run(exe, a.minutes, a.ai, a.seed, True)
    r2 = run(exe, a.minutes, a.ai, a.seed, True)
    for (t1, d1, o1), (t2, d2, o2) in zip(r1, r2):
        if t1 != t2 or d1 != d2:
            u1 = {u["id"]: u for u in o1.get("units", [])}
            u2 = {u["id"]: u for u in o2.get("units", [])}
            diff = [i for i in u1 if i in u2 and u1[i] != u2[i]]
            print(f"runs differ from tick {t1}/{t2}: {len(diff)} units differ, only in run 1 {len(set(u1) - set(u2))}, "
                  f"only in run 2 {len(set(u2) - set(u1))}; e.g. {[(i, u1[i], u2[i]) for i in diff[:2]]}")
            return 1
    print(f"identical over {min(len(r1), len(r2))} observations (to tick {r1[-1][0] if r1 else 0})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
