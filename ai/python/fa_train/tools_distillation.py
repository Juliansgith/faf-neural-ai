"""Small, allocation-free validation and provenance helpers for online tools teachers."""
from __future__ import annotations

import math


def validate_source(source: str) -> str:
    # m28: M28AI sits in our slot in shadow mode (fa_env.env.FAEnv(shadow=True)) -- its orders for the state the
    # policy has brought the game to are recorded, not given, and become the labels of the decision before
    if source not in ("v3", "m28", "none"):
        raise ValueError(f"unsupported distill_teacher {source!r}; expected 'v3', 'm28' or 'none'")
    return source


def validate_distillation(config) -> None:
    validate_source(config.distill_teacher)
    for name in ("distill_start", "distill_end", "distill_fraction"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")


def distillation_metadata(config, beta: float | None = None) -> dict:
    validate_distillation(config)
    enabled = config.distill_teacher != "none"
    result = {"distill_teacher": config.distill_teacher,
              "distill_start": config.distill_start, "distill_end": config.distill_end,
              "distill_fraction": config.distill_fraction,
              "effective_distill_start": config.distill_start if enabled else 0.0,
              "effective_distill_end": config.distill_end if enabled else 0.0}
    if beta is not None:
        result["effective_beta"] = beta if enabled else 0.0
    return result


def episode_teacher(episode: dict) -> str:
    """Legacy label requests mean v3; explicit none always suppresses stale requests."""
    source = validate_source(episode.get("distill_teacher", "v3"))
    return source if episode.get("teacher_labels") else "none"


def checkpoint_distillation(saved_config: dict | None, active_config, beta: float) -> dict:
    active = distillation_metadata(active_config, beta)
    if saved_config is None:
        return {"saved": None, "active": active, "saved_provenance": "unavailable",
                "mismatches": [], "diagnostics": ["checkpoint has no config; PPO teacher provenance unavailable"]}
    saved = {key: saved_config.get(key) for key in ("distill_start", "distill_end", "distill_fraction")}
    saved["distill_teacher"] = saved_config.get("distill_teacher", "v3")
    legacy = "distill_teacher" not in saved_config
    diagnostics = ["missing saved selector interpreted as legacy v3"] if legacy else []
    try:
        validate_source(saved["distill_teacher"])
    except ValueError as exc:
        diagnostics.append(str(exc) + "; active runtime config remains authoritative")
    mismatches = [key for key, value in saved.items()
                  if (key == "distill_teacher" or key in saved_config) and value != active[key]]
    return {"saved": saved, "active": active, "saved_provenance": "legacy_v3" if legacy else "explicit",
            "mismatches": mismatches, "diagnostics": diagnostics}
