"""Exact offline returns when the recording contains live reward components."""
import math
import numpy as np


def recorded_returns(steps, end, gamma, shaping=1.0):
    records = [row["step"].get("transition") for row in steps]
    if all(t is None for t in records):
        return None  # Legacy recordings need explicitly identified approximate targets.
    if any(t is None for t in records):
        raise ValueError("Partially recorded reward transitions")
    if not (math.isfinite(gamma) and 0 <= gamma <= 1 and math.isfinite(shaping)):
        raise ValueError("Invalid discount or shaping weight")
    rewards = []
    for i, (row, transition) in enumerate(zip(steps, records)):
        t = transition
        if t.get("schema") != "fa_tools_transition_v1":
            raise ValueError("Unknown reward transition schema")
        if (t["start_tick"] != row["step"]["tick"] or t["start_tick"] != row["msg"]["obs"]["tick"] or
                t["end_tick"] <= t["start_tick"] or
                (i and records[i-1]["end_tick"] != t["start_tick"])):
            raise ValueError("Misaligned reward transition")
        if bool(t["done"]) != (i == len(records) - 1):
            raise ValueError("Reward recording must contain one final terminal transition")
        c = t["reward_components"]
        if set(c) != {"terminal", "trade", "economy", "commander"} or not all(math.isfinite(v) for v in c.values()):
            raise ValueError("Invalid recorded reward components")
        rewards.append(c["terminal"] + shaping * (c["trade"] + c["economy"] + c["commander"]))
    if records[-1]["outcome"] != end["outcome"]:
        raise ValueError("Recorded outcome disagrees with episode summary")
    returns, acc = np.empty(len(rewards), np.float32), 0.0
    for i in reversed(range(len(rewards))):
        acc = rewards[i] + gamma * acc
        returns[i] = acc
    return returns
