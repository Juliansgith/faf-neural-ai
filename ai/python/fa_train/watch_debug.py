"""The watch debugger: what each decision of a watched game (tools_eval --watch) was made of.

The trainer's acting network keeps its internals for watched games (ToolsNet.debug -> last_debug):
watch_payload() turns one game's share into plain data -- the value estimate, each unit's three likeliest verbs
(with 'end': nothing), the acting units' likeliest cells, targets and blueprints, the spatial critic's map and
the encoder's attention. The game's worker adds what it knows (write_decision): token positions and names, the
map layers the network saw (danger maps included), the economy, and the true state (obs.privileged: what we know
of the enemy, map control). Files in the run's watch folder, read by tools/watch_viewer.py:
  latest.json            the newest decision
  timeline.jsonl         one row per decision (value, intel, map control, economy)
  decisions/<tick>.json  every decision, for scrubbing back
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fa_env.tools import VERBS

LAYERS = ("height", "mass_spots", "last_looked", "enemy_strength", "enemy_ground_fire", "enemy_air_fire",
          "own_ground_fire", "enemy_shield", "own_shield")


def _np(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def watch_payload(dbg: dict, i: int, obs: dict) -> dict:
    """Game `i` of an acting batch's last_debug, as plain data (token indices refer to `obs`)."""
    n = len(obs["ids"])
    out = {"value": float(_np(dbg["value"])[i])}
    actor = _np(dbg["actor"])[i][:n] > 0
    vp, vi = _np(dbg["vtop_p"])[i][:n], _np(dbg["vtop_i"])[i][:n].astype(int)
    out["verbs"] = {int(t): [[VERBS[vi[t, k]], round(float(vp[t, k]), 3)] for k in range(vi.shape[1])]
                    for t in np.nonzero(actor)[0]}
    idx, on, verb = _np(dbg["idx"])[i].astype(int), _np(dbg["on"])[i] > 0, _np(dbg["verb"])[i].astype(int)
    parts = {k: (_np(dbg[f"{k}_p"])[i], _np(dbg[f"{k}_i"])[i].astype(int), _np(dbg[f"{k}_on"])[i] > 0)
             for k in ("cell", "tgt", "bp")}
    acting = {}
    for u in np.nonzero(on)[0]:
        t = int(idx[u])
        if t >= n:
            continue
        e = {"verb": VERBS[verb[u]]}
        for k, (p, ix, o) in parts.items():
            if o[u]:
                e[k] = [[int(ix[u, j]), round(float(p[u, j]), 3)] for j in range(ix.shape[1])]
        acting[t] = e
    out["acting"] = acting
    out["local_value"] = np.round(_np(dbg["local_value"])[i], 3).tolist()
    if "attn" in dbg:
        a = _np(dbg["attn"])[i][:n, :n]
        types = np.asarray(obs["token_type"])[:n]
        own = types == 0
        weight = a[own].mean(0) if own.any() else a.mean(0)    # what our units look at, on average
        out["attention"] = [[int(t), round(float(weight[t]), 4)] for t in np.argsort(-weight)[:15]]
    return out


def _plain(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_decision(folder: Path, env, obs: dict, commands: list, payload: dict) -> None:
    """One decision of a watched game into `folder` (see the module docstring)."""
    from fa_env.tools_obs import MAP_CHANNELS
    from fa_train.tools_bc import shaping_arrays
    folder.mkdir(parents=True, exist_ok=True)
    raw = env.obs
    tick = int(raw.get("tick", 0))
    n = len(obs["ids"])
    tok = np.asarray(obs["tokens"])[:n]
    types = np.asarray(obs["token_type"])[:n]
    units = {str(u["id"]): u for u in raw.get("units") or []}
    tokens = []
    for t in range(n):
        uid = obs["ids"][t]
        u = units.get(uid) if uid else None
        bp = (env.bps.get(u.get("bp")) or {}) if u else {}
        name = bp.get("name") or (u or {}).get("bp") or ""
        tokens.append([int(types[t]), round(float(tok[t, 7]), 4), round(float(tok[t, 8]), 4), name,
                       round(float(tok[t, 13]), 2)])
    maps = np.asarray(obs["map"])
    layers = {k: np.round(maps[MAP_CHANNELS.index(k)], 2).tolist() for k in LAYERS if k in MAP_CHANNELS}
    me = env.me
    armies = raw.get("armies") or []
    army = next((a for a in armies if a.get("index") == me), {})
    enemies = [a["index"] for a in armies if a.get("relation") == "enemy" and not a.get("civilian")]
    priv = raw.get("privileged") or {}
    keys = ("mexes", "armyMass", "massIncome", "engineers", "factories", "engineerMass", "structureMass")
    mine = {k: float((priv.get(str(me)) or {}).get(k) or 0.0) for k in keys}
    theirs = {k: sum(float((priv.get(str(e)) or {}).get(k) or 0.0) for e in enemies) for k in keys}
    shaped = shaping_arrays([{"msg": {"obs": raw}}], env.kit, env.bps, me)
    eco = army.get("eco") or {}
    rec = {"tick": tick, "minute": round(tick / 600, 2), "map_size": [env.observer.width, env.observer.height],
           "tokens": tokens, "layers": layers, "commands": commands,
           "candidates": [(env.bps.get(c) or {}).get("name") or c for c in obs.get("cand_ids") or []],
           "eco": {r: {k: round(float((eco.get(r) or {}).get(k) or 0.0), 1) for k in ("income", "requested", "stored", "max")}
                   for r in ("mass", "energy")},
           "mine": mine, "theirs": theirs, "intel": round(float(shaped["phi_intel"][0]), 3),
           "map_control": round(float(shaped["phi_map"][0]), 3), **payload}
    text = json.dumps(rec, default=_plain)
    (folder / "decisions").mkdir(exist_ok=True)
    (folder / "decisions" / f"{tick:06d}.json").write_text(text)
    tmp = folder / "latest.json.tmp"
    tmp.write_text(text)
    tmp.replace(folder / "latest.json")
    row = {"tick": tick, "minute": rec["minute"], "value": round(payload["value"], 4), "intel": rec["intel"],
           "map_control": rec["map_control"], "mass_income": rec["eco"]["mass"]["income"],
           "own_mexes": mine["mexes"], "enemy_mexes": theirs["mexes"], "own_army": round(mine["armyMass"]),
           "enemy_army": round(theirs["armyMass"]), "orders": len(commands)}
    with open(folder / "timeline.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
