"""Small, bounded diagnostics without changing commands or reward values."""
from __future__ import annotations


def action_failures(sent, results, limit=16):
    failures = []
    count = 0
    for index, result in enumerate(results):
        if result.get("ok"):
            continue
        count += 1
        if len(failures) >= limit:
            continue
        action = sent[index] if index < len(sent) else {}
        reason = result.get("error") or result.get("message") or "no error description supplied"
        failures.append({"index": index, "type": action.get("type", "unknown"),
                         "units": list(action.get("units") or [])[:16],
                         "bp": action.get("bp"), "error": str(reason)[:500]})
    return count, failures


def transition_record(start_tick, end_tick, rewards, done, outcome, time_limit_ticks):
    # Components are actual environment outputs. Learner shaping weights are
    # deliberately not guessed: PPO can change those weights during a run.
    return {"schema": "fa_tools_transition_v1", "start_tick": int(start_tick),
            "end_tick": int(end_tick), "reward_components": dict(rewards),
            "done": bool(done), "outcome": outcome, "time_limit_ticks": int(time_limit_ticks),
            "reward_weighting": "raw_components; learner shaping weights not applied"}
