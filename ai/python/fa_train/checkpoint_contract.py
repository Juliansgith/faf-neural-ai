"""Reject silent changes to the SET policy when loading shape-compatible weights.

Legacy independent-Bernoulli models/checkpoints have no set_distribution metadata.
Alternative implementations must explicitly declare the exact saved distribution
as a class-level set_distribution_contract. A matching state_dict is insufficient.
"""
from copy import deepcopy
from collections.abc import Mapping


def _contract(value, origin):
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("acting"), str) or not value["acting"]:
        raise ValueError(f"Invalid SET distribution contract in {origin}: expected a mapping with an acting identifier")
    return deepcopy(dict(value))


def runtime_set_distribution(learner, actor):
    # Read the class declaration explicitly. An arbitrary instance attribute or
    # dynamically generated mock attribute must not claim decoder compatibility.
    left = _contract(getattr(type(learner), "set_distribution_contract", None), "learner")
    right = _contract(getattr(type(actor), "set_distribution_contract", None), "actor")
    if left != right:
        raise ValueError("Actor and learner SET distribution contracts differ; acting and scoring must agree")
    return left


def validate_checkpoint_set_distribution(state, learner, actor):
    active = runtime_set_distribution(learner, actor)
    saved = _contract(state.get("set_distribution"), "checkpoint")
    if saved != active:
        saved_name = saved["acting"] if saved else "legacy independent Bernoulli"
        active_name = active["acting"] if active else "legacy independent Bernoulli"
        raise ValueError(
            f"Checkpoint SET distribution mismatch: saved={saved_name}, runtime={active_name}. "
            "Use the matching decoder and explicitly declare its complete set_distribution_contract; "
            "do not remove checkpoint metadata or silently reinterpret its weights."
        )
    return active
