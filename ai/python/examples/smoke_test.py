"""Bridge smoke test: launch a game, handshake, step N times without acting, report what came back.

    python smoke_test.py [--map SCMP_016] [--steps 60] [--ticks 10] [--speed 50]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import FAEnv, GameConfig  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="SCMP_016")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--ticks", type=int, default=10)
parser.add_argument("--speed", type=int, default=50)
parser.add_argument("--exe", default=None)
args = parser.parse_args()

config = GameConfig(map=args.map, speed=args.speed, seed=1234)
if args.exe:
    config.exe = Path(args.exe)

env = FAEnv(config, step_ticks=args.ticks)
t0 = time.monotonic()
try:
    obs, info = env.reset()
    print(f"connected after {info['load_seconds']:.1f} s; log {info['log_path']}")
    print(f"map {info['map']['name']!r} {info['map']['width']}x{info['map']['height']}, "
          f"{len(info['map']['deposits'])} deposits, {len(info['blueprints'])} unit blueprints")
    for a in info["armies"]:
        print(f"  army {a['index']} {a['name']:<18} {a['nickname']!s:<22} faction {a['faction']} "
              f"{a['brainType']:<6} {a['relation']:<8} start {a['start']}")
    if info.get("errors"):
        print("hello errors:", info["errors"])

    for step in range(args.steps):
        obs, reward, terminated, truncated, step_info = env.step()
        if step_info["errors"]:
            print(f"step {step} errors: {step_info['errors']}")
        if step % 10 == 0 or terminated or truncated:
            me = next(a for a in obs["armies"] if a["index"] == obs["me"])
            mass, energy = me["eco"]["mass"], me["eco"]["energy"]
            mine = sum(1 for u in obs["units"] if u["army"] == obs["me"])
            print(f"tick {obs['tick']:>5} units(all/mine) {len(obs['units']):>4}/{mine:<3} "
                  f"mass {mass['stored']:.0f}/{mass['max'] or 0:.0f} +{mass['income']:.1f}/s  "
                  f"energy {energy['stored']:.0f} +{energy['income']:.1f}/s  timing {step_info['timing']}")
        if terminated or truncated:
            print("episode over:", step_info["outcome"])
            break

    s = env.stats
    ticks = obs["tick"]
    wall = time.monotonic() - s["t0"]
    print(json.dumps({
        "steps": s["steps"], "game_ticks": ticks, "wall_seconds": round(wall, 2),
        "ticks_per_second": round(ticks / wall, 1), "game_speedup": round(ticks / 10 / wall, 1),
        "bytes_received": env.server.bytes_received,
        "last_obs_bytes": len(json.dumps(obs)),
        "sample_unit": next((u for u in obs["units"] if u["army"] == obs["me"]), None),
    }, indent=1))
finally:
    env.close()
    print(f"total {time.monotonic() - t0:.1f} s")
