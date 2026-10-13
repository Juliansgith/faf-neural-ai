"""Aggregate throughput of N game instances stepped in parallel (gymnasium AsyncVectorEnv, one process each).

    python vec_throughput.py --envs 4 --seconds 120

Random valid macro actions; prints decisions/s and game-seconds/s summed over all instances.
"""
import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import GameConfig  # noqa: E402
from fa_env.gym_env import MacroEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=120)
parser.add_argument("--map", default="SCMP_016")
parser.add_argument("--decision-ticks", type=int, default=20)
args = parser.parse_args()


def make(i):
    def thunk():
        return MacroEnv(GameConfig(map=args.map, ai_faction=1 + i % 4, window=(640, 480)),
                        decision_ticks=args.decision_ticks, max_minutes=15)
    return thunk


if __name__ == "__main__":
    envs = gym.vector.AsyncVectorEnv([make(i) for i in range(args.envs)], shared_memory=False)
    t0 = time.monotonic()
    obs, _ = envs.reset(seed=list(range(args.envs)))
    print(f"{args.envs} games loaded in {time.monotonic() - t0:.1f} s")
    rng = np.random.default_rng(0)
    decisions, t_start = 0, time.monotonic()
    episodes = 0
    while time.monotonic() - t_start < args.seconds:
        masks = obs["action_mask"]
        actions = [rng.choice(np.flatnonzero(m)) for m in masks]
        obs, rewards, terminated, truncated, infos = envs.step(np.array(actions))
        decisions += args.envs
        episodes += int(np.sum(terminated | truncated))
    wall = time.monotonic() - t_start
    game_seconds = decisions * args.decision_ticks / 10
    print(f"{decisions} decisions in {wall:.1f} s: {decisions / wall:.1f} decisions/s, "
          f"{game_seconds / wall:.0f} game-seconds/s across {args.envs} games "
          f"({game_seconds / wall / args.envs:.0f}x real time each), {episodes} episodes finished")
    envs.close()
