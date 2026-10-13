"""Play one game with RulePolicy in FactoredEnv and print progress each game minute.

    python rule_game.py --ai easy --seed 1 --minutes 30
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import GameConfig  # noqa: E402
from fa_env.factored import HEADS, FactoredEnv  # noqa: E402
from fa_env.rule_policy import RulePolicy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ai", default="easy")
ap.add_argument("--ai-faction", type=int, default=2)
ap.add_argument("--faction", type=int, default=1)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--minutes", type=float, default=30)
ap.add_argument("--map", default="SCMP_016")
ap.add_argument("--minimized", action="store_true", help="minimize the game window (it then skips drawing)")
ap.add_argument("--probe-minute", type=int, help="at this game minute, sample the game's per-thread CPU")
ap.add_argument("--profile-minute", type=int, help="at this game minute, sample-profile the busiest thread")
ap.add_argument("--profile-rank", type=int, default=1, help="profile the rank-th busiest thread instead")
ap.add_argument("--record", help="save a raw recording of the game to this folder (fa_env/record.py)")
a = ap.parse_args()
TOOLS = Path(__file__).resolve().parents[3] / "tools"

env = FactoredEnv(GameConfig(map=a.map, faction=a.faction, ai=a.ai, ai_faction=a.ai_faction, seed=a.seed,
                             window=(640, 480), minimized=a.minimized), time_limit_minutes=a.minutes)
policy = RulePolicy()
if a.record:
    from fa_env.record import Recorder
    env.recorder = Recorder(Path(a.record), f"rule_{a.ai}_s{a.seed}")
t0 = time.monotonic()
obs, info = env.reset()
policy.reset(env)
print(f"loaded in {info['load_seconds']:.0f} s; choices: {env.chosen}")
counts = {h: collections.Counter() for h in HEADS}
totals = collections.Counter()
last_minute, done = -1, False
while not done:
    choice = policy.act(env)
    for h, i in choice.items():
        counts[h][HEADS[h][i]] += 1
    obs, rewards, done, step = env.step(dict(choice))
    for k, v in rewards.items():
        totals[k] += v
    for e in step["errors"]:
        print("script error:", e[:200])
    minute = step["tick"] // 600
    if minute != last_minute:
        last_minute = minute
        if minute in (a.probe_minute, a.profile_minute):
            import subprocess
            tool = "thread_cpu.py" if minute == a.probe_minute else "thread_profile.py"
            extra = [] if minute == a.probe_minute else ["--rank", str(a.profile_rank)]
            subprocess.Popen([sys.executable, str(TOOLS / tool), "--pid", str(env.fa.game.process.pid),
                              "--seconds", "5"] + extra)
        s = env.state
        roles = {k: len(v) for k, v in s.own.items() if v}
        m = s.eco.get("mass", {})
        print(f"[{minute:2d} min] mass +{m.get('income', 0):.1f} stored {m.get('stored', 0):.0f} | army {len(s.army)} "
              f"order {env.army_order} | seen {len(s.enemies_seen)} ghosts {len(s.ghosts)} | {roles} | "
              f"errors {step['action_errors']} | {time.monotonic() - t0:.0f} s")
print(json.dumps(step.get("episode"), indent=1))
print("reward totals", {k: round(v, 2) for k, v in totals.items()})
for h in HEADS:
    print(h, dict(counts[h].most_common()))
env.close()
