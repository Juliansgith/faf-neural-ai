"""Shared numeric precision for live tools observations and offline caches.

Only the large token/map arrays are compressed on the worker pipe. Candidate
and enhancement features stay float32; caching them at lower precision creates
different actor inputs even when observations and command bookkeeping agree.
"""
import numpy as np

PRECISION_CONTRACT = "tools_f16_tokens_map_last_f32_candidates_v2"
TRANSPORT_HALF_KEYS = ("tokens", "map", "token_last_feat")
CACHE_FLOAT_DTYPES = {"tokens": np.float16, "map": np.float16, "token_last_feat": np.float16,
                      "cand_feat": np.float32, "enh_feat": np.float32}


def compact_observation(obs: dict) -> dict:
    out = dict(obs)
    for key in TRANSPORT_HALF_KEYS:
        if key in obs:
            out[key] = obs[key].astype(np.float16)
    return out


def widen_observation(obs: dict) -> dict:
    out = dict(obs)
    for key in TRANSPORT_HALF_KEYS:
        if key in obs:
            out[key] = obs[key].astype(np.float32)
    return out
