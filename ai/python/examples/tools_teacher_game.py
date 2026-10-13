"""Gate T1 (docs/tools-design.md): the rule teacher plays through the tools executor.

Every decision the teacher's orders are turned into commands (verb, subject, blueprint, count, cell, target) and
executed by ToolKit -- exactly what a network's commands go through. Builds away from deposits get the cell
of the spot the teacher aimed at; the executor finds a legal site in or next to that cell. If the teacher
still wins, the vocabulary and the executor cover what it needs.

    python tools_teacher_game.py --ai medium --seed 3 --minutes 30
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import GameConfig  # noqa: E402
from fa_env.rule_policy import RulePolicy  # noqa: E402
from fa_env.tools_env import ToolsEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ai", default="easy")
ap.add_argument("--ai-faction", type=int, default=2)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--minutes", type=float, default=30)
ap.add_argument("--record", help="save a raw recording to this folder")
a = ap.parse_args()

env = ToolsEnv(GameConfig(map="SCMP_016", ai=a.ai, ai_faction=a.ai_faction, seed=a.seed, window=(640, 480),
                          minimized=True), time_limit_minutes=a.minutes)
if a.record:
    from fa_env.record import Recorder
    env.recorder = Recorder(Path(a.record), f"tools_teacher_{a.ai}_s{a.seed}")
policy = RulePolicy()
t0 = time.monotonic()
obs, info = env.reset()
policy.reset(env)
verbs, failures = collections.Counter(), 0
done, last_minute = False, -1
while not done:
    cmds = env.teacher_commands(policy, dry=False)
    for c in cmds:
        if c["verb"] == "build" and c.get("place") == "cell" and c.get("cell") is None:   # an open cell near the aim
            c["cell"] = env.free_cell_near(*c.get("aim", env.start), c["bp"])
        verbs[c["verb"]] += 1
    obs, rewards, done, step = env.step_tools(cmds)
    failures += step["action_errors"]
    minute = step["tick"] // 600
    if minute != last_minute:
        last_minute = minute
        s = env.state
        roles = {k: len(v) for k, v in s.own.items() if v}
        print(f"[{minute:2d} min] mass +{s.eco.get('mass', {}).get('income', 0):.1f} | army {len(s.army)} | "
              f"{roles} | failed orders so far {failures} | {time.monotonic() - t0:.0f} s")
print(json.dumps(step.get("episode"), indent=1))
print("commands", dict(verbs.most_common()), "failed orders", failures)
env.close()
