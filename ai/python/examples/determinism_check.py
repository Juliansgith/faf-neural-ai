"""Run the same episode twice (same seed, same scripted agent) and compare observations step by step.

    python determinism_check.py [--steps 300] [--map SCMP_016] [--seed 5]

The sim is deterministic by design (lock-step multiplayer depends on it), so two runs must match
exactly -- as long as the opponents are the same. FAF's command-line launcher picks the AI's faction
with UI-side randomness, so a mismatch that coincides with different factions says nothing about the
sim; the script reports the factions of both runs for that reason.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import FAEnv, GameConfig  # noqa: E402
from fa_env.agents import ScriptedAgent  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--map", default="SCMP_016")
parser.add_argument("--seed", type=int, default=5)
parser.add_argument("--ai-faction", type=int, default=2, help="pin the opponents' faction (0: FAF's random pick)")
parser.add_argument("--ai", default="rush")
args = parser.parse_args()
if args.ai_faction == 0:
    args.ai_faction = None


def run():
    config = GameConfig(map=args.map, seed=args.seed, ai_faction=args.ai_faction, ai=args.ai)
    env = FAEnv(config, step_ticks=10, omniscient=True)
    obs, info = env.reset()
    factions = [(a["index"], a["faction"]) for a in info["armies"]]
    agent = ScriptedAgent()
    agent.reset(obs, info)
    hashes, step_info = [], None
    for _ in range(args.steps):
        actions, queries = agent.act(obs, step_info)
        obs, _, terminated, truncated, step_info = env.step(actions, queries)
        hashes.append(hashlib.sha1(json.dumps(obs, sort_keys=True).encode()).hexdigest())
        if terminated or truncated:
            break
    last = obs
    env.close()
    return factions, hashes, last


factions_a, hashes_a, last_a = run()
factions_b, hashes_b, last_b = run()
print("factions A:", factions_a)
print("factions B:", factions_b)
mismatch = next((i for i, (a, b) in enumerate(zip(hashes_a, hashes_b)) if a != b), None)
if mismatch is None and len(hashes_a) == len(hashes_b):
    print(f"IDENTICAL: {len(hashes_a)} steps ({len(hashes_a) * 10} ticks) matched exactly")
else:
    print(f"DIVERGED at step {mismatch} (tick {(mismatch or 0) * 10 + 12})")
    if factions_a != factions_b:
        print("  (the opponents' factions differ between the runs, so this is expected)")
    print("  units A/B:", len(last_a.get("units", [])), len(last_b.get("units", [])))
