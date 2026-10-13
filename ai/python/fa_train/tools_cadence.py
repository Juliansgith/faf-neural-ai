"""Explicit decision cadence for tools training, evaluation and worker resets."""
from collections.abc import Mapping
from numbers import Integral

LEGACY_DECISION_TICKS = 20


def decision_ticks(value):
    # Placement/reclaim queries can require one tick before the final action tick.
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError("decision_ticks must be an integer of at least 2 simulation ticks")
    return int(value)


def spec_decision_ticks(spec):
    if 'decision_ticks' not in spec:
        raise ValueError('Bounded experiment spec must explicitly declare decision_ticks')
    return decision_ticks(spec['decision_ticks'])


def checkpoint_cadence(saved_config, active_ticks):
    active = decision_ticks(active_ticks)
    saved = saved_config.get("decision_ticks") if isinstance(saved_config, Mapping) else None
    if saved is not None:
        saved = decision_ticks(saved)
    return {"saved_decision_ticks": saved, "active_decision_ticks": active,
            "saved_provenance": "explicit" if saved is not None else "unavailable",
            "mismatch": saved is not None and saved != active}
