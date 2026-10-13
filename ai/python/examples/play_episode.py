"""Play one episode with a reference agent against FAF's rush AI and report progress and throughput.

    python play_episode.py [--agent scripted|random] [--map SCMP_016] [--minutes 20] [--ticks 10]
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import FAEnv, GameConfig  # noqa: E402
from fa_env.agents import RandomAgent, ScriptedAgent  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--agent", choices=["scripted", "random"], default="scripted")
parser.add_argument("--map", default="SCMP_016")
parser.add_argument("--minutes", type=float, default=20.0)
parser.add_argument("--ticks", type=int, default=10, help="game ticks per agent step (10 ticks = 1 s)")
parser.add_argument("--speed", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--faction", type=int, default=1)
parser.add_argument("--omniscient", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

env = FAEnv(GameConfig(map=args.map, speed=args.speed, seed=args.seed, faction=args.faction),
            step_ticks=args.ticks, max_ticks=int(args.minutes * 600), omniscient=args.omniscient)
agent = ScriptedAgent() if args.agent == "scripted" else RandomAgent(seed=args.seed)

t0 = time.monotonic()
obs, info = env.reset()
agent.reset(obs, info)
print(f"loaded in {info['load_seconds']:.1f} s; log {info['log_path']}")
action_counts, action_errors = collections.Counter(), collections.Counter()
total_reward, step_info, done = 0.0, None, False
last_report = -1
while not done:
    actions, queries = agent.act(obs, step_info)
    for a in actions:
        action_counts[a["type"]] += 1
    obs, reward, terminated, truncated, step_info = env.step(actions, queries)
    total_reward += reward
    for r in step_info["results"]:
        if not r.get("ok"):
            action_errors[r.get("error", "?")[:90]] += 1
    for e in step_info["errors"]:
        print("bridge error:", e[:300])
    minute = obs["tick"] // 600
    if minute != last_report:
        last_report = minute
        me = next(a for a in obs["armies"] if a["index"] == obs["me"])
        mine = [u for u in obs["units"] if u["army"] == obs["me"]]
        enemy = [u for u in obs["units"] if u["army"] != obs["me"]]
        kinds = collections.Counter(u.get("bp") for u in mine)
        wall = time.monotonic() - env.stats["t0"]
        print(f"[{obs['tick'] // 600:>2}:{(obs['tick'] // 10) % 60:02d}] units {len(mine):>3} (visible enemies {len(enemy):>3}) "
              f"mass +{me['eco']['mass']['income']:.1f}/s energy +{me['eco']['energy']['income']:.0f}/s "
              f"killed {me['stats'].get('Enemies_Killed', 0):.0f} lost {me['stats'].get('Units_Killed', 0):.0f} "
              f"reward {total_reward:+.2f} | {obs['tick'] / 10 / max(wall, 1e-6):.0f}x real time | "
              f"{dict(kinds.most_common(5))}")
        if args.verbose:
            print(f"        timing {step_info['timing']}  action errors so far {dict(action_errors.most_common(3))}")
    done = terminated or truncated

wall = time.monotonic() - env.stats["t0"]
print(json.dumps({
    "outcome": step_info["outcome"] or ("time limit" if truncated else None),
    "game_minutes": round(obs["tick"] / 600, 1),
    "wall_seconds": round(wall, 1),
    "speedup": round(obs["tick"] / 10 / wall, 1),
    "steps": env.stats["steps"],
    "agent_seconds": round(env.stats["agent_wall"], 1),
    "reward": round(total_reward, 2),
    "actions": dict(action_counts),
    "action_errors": dict(action_errors.most_common(10)),
}, indent=1))
env.close()
print(f"total {time.monotonic() - t0:.1f} s")
