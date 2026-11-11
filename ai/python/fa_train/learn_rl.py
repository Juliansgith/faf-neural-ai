"""PPO on recorded games: the RL step for games played elsewhere (fa_train.player records each decision's own
choice: log-probabilities, per-unit log-probabilities, value, actions; tools_bc encodes them as pol_* arrays).

    python -m fa_train.learn_rl --init CKPT --out OUT.pt --cache RUN/cache [--cache ...] --reference REF.pt \
        [--epochs 2] [--ref-coef 0.2] [--distill 0.5] [--lr 3e-5]

One call is one PPO iteration over the games that carry recorded choices:
  1. the games are cut into consecutive chunks of `--length` decisions and run as parallel streams (a game per
     stream, the next game when one ends), the network's memory carried from chunk to chunk exactly as it was live
     (_runs/tools/policy_replay_check.py: replayed log-probabilities equal the recorded ones);
  2. a first pass (no gradients) gives every decision's value; advantages are GAE over the recorded reward
     components (shaping weight --shaping, as the cloned value head was trained on);
  3. --epochs passes of factored PPO (each unit's ratio against its recorded log-probability, the decision's
     advantage; tools_trainer.ToolsConfig.unit_ratio), a value loss, a small entropy bonus, a KL anchor to the frozen
     --reference network (k3 estimator per unit) and M28's labels (the recording's teacher) as a cloning term of
     weight --distill. An epoch stops early when the per-unit KL to the recorded policy passes 1.5 x --target-kl.
The data may be a little stale (the player picks up a new network between rounds): the ratios are against the
choices as recorded, so PPO's clipping covers the lag; games played by much older networks should be dropped
(--max-games keeps the newest).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import time

T0 = time.monotonic()


def at() -> str:
    """Minutes since the learner started, for the phase lines (2026-11-10: where an iteration's hours go)."""
    return f" [{(time.monotonic() - T0) / 60:.1f} min]"
from pathlib import Path

import numpy as np
import torch
from fa_env.tools import VERBS

from .cpu_runtime import add_cpu_arguments, configure_cpu, configure_model
from .tools_bc import REWARD_COMPONENTS, Games, recency_key
from fa_env.tools_obs import CAND_KEYS, ENH_KEYS, TOKEN_KEYS, TOKEN_T_KEYS
from .tools_model import (AUX_ECON_HORIZONS, N_AUX_ECON, N_OPPONENT, UNIT_KEYS, ToolsNet, apply_model_options, collate, model_options,
                          opponent_features, opponent_of_recording, parameter_count, stack_actions)


def load_net(path: Path, device) -> tuple[ToolsNet, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("decoder", "slots") != "units":
        raise ValueError("learn_rl needs a units-decoder network")
    net = ToolsNet(state.get("size", "M"), decoder="units").to(device)
    net.load_state_dict(state["model"])
    apply_model_options(net, state)
    return net, state


SHAPING_ARRAYS = ("phi_intel", "phi_map", "waste", "mass_capacity", "reclaimed_mass", "econ_trade")


def prepare_aux_econ(games, usable: list, lengths: dict) -> None:
    """--aux-econ: for every decision, the change of log income, log army mass and log capacity AUX_ECON_HORIZONS
    decisions ahead, and which of those exist. A game that ends before a horizon counts if it lasted at least half of
    it (its final value). The inputs are the recorded own totals; `games.aux_econ` holds the arrays per game."""
    from fa_env.tools_obs import PRIV_SCALE, PRIVILEGED
    i_inc, i_army = PRIVILEGED.index("massIncome"), PRIVILEGED.index("armyMass")
    out = {}
    for gi in usable:
        g, n = games.games[gi], lengths[gi]
        priv = np.asarray(g["privileged"][:n], np.float32)
        series = [np.log1p(np.maximum(priv[:, i_inc] * PRIV_SCALE[i_inc], 0.0)),
                  np.log1p(np.maximum(priv[:, i_army] * PRIV_SCALE[i_army], 0.0)),
                  np.log1p(np.maximum(np.asarray(g["mass_capacity"][:n], np.float32), 0.0))
                  if "mass_capacity" in g else None]
        y, m = np.zeros((n, N_AUX_ECON), np.float32), np.zeros((n, N_AUX_ECON), np.float32)
        for k, values in enumerate(series):
            if values is None:
                continue
            for j, h in enumerate(AUX_ECON_HORIZONS):
                c = k * len(AUX_ECON_HORIZONS) + j
                keep = np.arange(n)
                keep = keep[keep + h // 2 < n]
                y[keep, c] = np.clip(values[np.minimum(keep + h, n - 1)] - values[keep], -4.0, 4.0)
                m[keep, c] = 1.0
        out[gi] = (y, m)
    games.aux_econ = out


def potential(phi: np.ndarray, gamma: float) -> np.ndarray:
    """gamma * phi(next) - phi(now) per decision; the game's end is absorbing with phi = 0 (Ng et al. 1999). Keeping
    the last value ("its next state is its own") paid every game gamma^m * phi_end: the final economy, intel and map
    control as a terminal bonus the size of a win -- a timeout while ahead +15.7 against a victory's +19, a long 20 km
    defeat +3.6 (2026-10-17 review)."""
    return gamma * np.append(phi[1:], 0.0) - phi


def clip_steps(phi: np.ndarray, limit: float | None) -> np.ndarray:
    """A potential whose step-to-step changes are clipped to +-limit, then summed again: one bad reading (an
    upgrading extractor's production) moves it by at most `limit`, and the telescoping (what is gained is given
    back at the end) still holds."""
    if not limit or len(phi) < 2:
        return phi
    return np.concatenate([phi[:1], phi[0] + np.cumsum(np.clip(np.diff(phi), -limit, limit))]).astype(np.float32)


LARGE_KM = 10


def map_km(g: dict) -> int:
    """The size of a game's map in km (5 when unknown)."""
    from fa_env.maps import MAP_POOL
    try:
        return int(MAP_POOL.get(json.loads(str(g["meta"].item())).get("map"), {}).get("km", 5))
    except Exception:                                   # noqa: BLE001
        return 5


ACU_SPEED = 1.7          # game units per second: every faction's commander


def phi_commander(g: dict, n: int, grace: float, span: float) -> np.ndarray | None:
    """How far out our commander is, in [0, 1]: its walk home in minutes (a straight line at ACU_SPEED from where
    it started) past `grace` minutes, over `span` minutes; where it is not seen (dead) it stays as it was last.
    From the close-ups' focus points (cache format 12). None: a game without them."""
    if "focus" not in g or "fine" not in g:
        return None
    foc = np.asarray(g["focus"], np.float32)[:n, 0]                  # [n, 3]: x / width, z / height, present
    size = np.asarray(g["fine"], np.float32)[:n, 2:4]
    on = foc[:, 2] > 0.5
    if not on.any():
        return None
    pos = foc[:, :2] * size
    home = pos[int(np.argmax(on))]
    walk = np.hypot(pos[:, 0] - home[0], pos[:, 1] - home[1]) / ACU_SPEED / 60.0
    out = np.clip((walk - grace) / span, 0.0, 1.0).astype(np.float32)
    last = 0.0
    for t in range(n):
        if on[t]:
            last = out[t]
        else:
            out[t] = last
    return out


def discounted(r: np.ndarray, gamma: float) -> np.ndarray:
    out, acc = np.zeros_like(r), 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        out[t] = acc
    return out


def _meta(g: dict) -> dict:
    try:
        return json.loads(str(g["meta"].item()))
    except (KeyError, ValueError, AttributeError):
        return {}


def _outcome(g: dict) -> str | None:
    return _meta(g).get("outcome")


def reward_spec(a) -> str:
    """What the reward is made of (stored in each checkpoint: a change is noticed by the next iteration)."""
    return (f"econ={a.econ_source};kills={a.econ_coef:g};reclaim={a.reclaim_coef:g};intel={a.intel_coef:g};"
            f"map={a.map_coef:g};waste={a.waste_coef:g};shaping={a.shaping:g}"
            + (f";stall={a.stall_coef:g};capacity=actual" if a.stall_coef else "")
            + (f";acu={a.acu_coef:g}/{a.acu_grace:g}/{a.acu_span:g}{'' if a.acu_all_maps else '/large'}"
               if a.acu_coef else "")
            + (f";timeout={a.timeout_scale:g}/{a.timeout_cost:g}" if getattr(a, "timeout_scale", None) is not None
               else "")
            + (f";capclip={a.capacity_clip:g}" if getattr(a, "capacity_clip", None) else "")
            + (f";capscale={a.capacity_scale:g}" if getattr(a, "capacity_scale", .05) != .05 else "")
            + ";phi_end=0"
            # Discount changes both the critic's return and potential shaping.
            # Legacy checkpoints used .997; keep their signature stable.
            + (f";gamma={a.gamma:g}" if a.gamma != .997 else ""))


LEGACY_SPEC = "econ=recorded;kills=0;reclaim=0.5;intel=0;map=0;waste=0;shaping=1"


_TOKEN_FIRST = set(TOKEN_KEYS) | {"token_mask"}
_TOKEN_SECOND = set(TOKEN_T_KEYS)
_CAND_FIRST = set(CAND_KEYS) | {"cand_mask"}
_ENH_FIRST = set(ENH_KEYS) | {"enh_mask"}
_UNIT_ROWS = {"old_ulp", "prox_ulp", "ref_ulp"}


def needs_shaping_arrays(a) -> bool:
    return a.econ_source == "capacity" or any((a.econ_coef, a.reclaim_coef, a.intel_coef, a.map_coef, a.waste_coef))


def rewards_of(g: dict, a) -> np.ndarray:
    """Per decision: the recorded terminal, trade and commander components, and the economy as chosen.
    econ_source 'recorded': the recorded economy component (change of mass income / 20). Measured on 144 games
    (2026-10-11), it was 99 % of the shaping's variance: reclaim makes the income jump ~17/s between decisions, so
    the fight signal (trade) was 2 % of it. 'capacity': the change of the mass production capacity / 20 (extractors,
    fabricators, commander), plus reclaim as mass gained (--reclaim-coef x mass / 1000: at half the trade's rate,
    so that a mex, credited once, is not out-valued by the finite wrecks of the map).
    The rest (tools_bc.shaping_arrays, local_trade_arrays): kills/losses of economy units beyond their cost
    (--econ-coef), the change of how much of the enemy we know (--intel-coef) and of map control (--map-coef)
    as potentials (gain rewarded, loss penalised, nothing for holding: they cannot be farmed or reward long
    games), and resources wasted to full storage (--waste-coef, a real cost)."""
    c = {k: np.asarray(g[f"rew_{k}"], np.float32) for k in REWARD_COMPONENTS}
    n = len(c["terminal"])
    if getattr(a, "timeout_scale", None) is not None and _outcome(g) == "timeout":
        # recorded as 5 x the value advantage (-1..1): at worst -5 against a defeat's -10, so in a game it is
        # losing the network gained ~14 a game by stalling to the clock (4646 games, 2026-10-14: timeout -4.5,
        # defeat -18.6 with shaping). Scored by who is ahead, a small cost for not finishing
        va = float(c["terminal"][n - 1]) / 5.0
        c["terminal"] = c["terminal"].copy()
        # never below a defeat's -10 (clip(10 va) - 2 reached -12: dying before the clock paid), never above a win
        c["terminal"][n - 1] = float(np.clip(a.timeout_scale * va - a.timeout_cost, -9.0, 10.0))
    shaped = c["trade"] + c["commander"]
    if a.econ_source == "capacity":
        shaped = shaped + potential(clip_steps(np.asarray(g["mass_capacity"], np.float32)[:n] * a.capacity_scale,
                                               getattr(a, "capacity_clip", None)), a.gamma)
        shaped = shaped + a.reclaim_coef * np.append(np.diff(np.asarray(g["reclaimed_mass"], np.float32)[:n]), 0.0) / 1000.0
    else:
        shaped = shaped + c["economy"]
    if a.econ_coef and "econ_trade" in g:
        shaped = shaped + a.econ_coef * np.asarray(g["econ_trade"], np.float32)[:n]
    if a.intel_coef:
        shaped = shaped + a.intel_coef * potential(np.asarray(g["phi_intel"], np.float32)[:n], a.gamma)
    if a.map_coef:
        shaped = shaped + a.map_coef * potential(np.asarray(g["phi_map"], np.float32)[:n], a.gamma)
    if a.waste_coef:
        shaped = shaped - a.waste_coef * np.asarray(g["waste"], np.float32)[:n] / 1000.0
    if a.stall_coef and "energy_stall" in g:          # games encoded since 2026-10-12
        shaped = shaped - a.stall_coef * np.asarray(g["energy_stall"], np.float32)[:n] / 1000.0
    if a.acu_coef and (a.acu_all_maps or map_km(g) > LARGE_KM):
        # the commander's exposure as a potential: walking out costs, walking back gives it back, dying out there
        # keeps the cost (the last decision is its own next state), staying out bleeds (1 - gamma) a decision
        phi = phi_commander(g, n, a.acu_grace, a.acu_span)
        if phi is not None:
            # a won game gives the exposure back at its end (the 20 km wins so far ended with the commander at the
            # enemy, 2026-10-13); a lost one keeps it: dying out there costs more than dying at home
            won = float(c["terminal"][n - 1]) > 0
            shaped = shaped - a.acu_coef * (a.gamma * np.append(phi[1:], 0.0 if won else phi[-1]) - phi)
    return c["terminal"] + a.shaping * shaped


def local_returns(g: dict, n: int, gamma: float, scale: float) -> np.ndarray | None:
    """[n, GRID*GRID] discounted local trade per cell (tools_bc.local_trade_arrays), spread over each cell's 3 x 3
    neighbourhood (a unit two cells from a fight still took part), divided by `scale`. None: the game has none."""
    if "loc_cells" not in g:
        return None
    from fa_env.tools import GRID
    off, cells, vals = np.asarray(g["offsets_l"]), np.asarray(g["loc_cells"]), np.asarray(g["loc_values"])
    r = np.zeros((n, GRID, GRID), np.float32)
    for t in range(min(n, len(off) - 1)):
        a, b = int(off[t]), int(off[t + 1])
        if b > a:
            np.add.at(r[t], (cells[a:b] // GRID, cells[a:b] % GRID), vals[a:b])
    pad = np.pad(r, ((0, 0), (1, 1), (1, 1)))
    spread = sum(pad[:, 1 + dz:1 + dz + GRID, 1 + dx:1 + dx + GRID] for dz in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    out = np.zeros_like(spread)
    acc = np.zeros((GRID, GRID), np.float32)
    for t in reversed(range(n)):
        acc = spread[t] + gamma * acc
        out[t] = acc
    return (out / scale).reshape(n, GRID * GRID)


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns of one whole game (the last decision is terminal)."""
    n = len(rewards)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def vtrace(rewards: np.ndarray, values: np.ndarray, log_rho: np.ndarray, gamma: float, lam: float,
           rho_bar: float = 1.0, c_bar: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """V-trace (IMPALA, Espeholt et al. 2018) over one whole game: value targets and policy advantages corrected for
    the games having been played by an older network. `log_rho` is each decision's log importance ratio; here the
    mean of its units' log ratios (the geometric mean of the per-unit ratios): a decision's joint ratio is a product
    over ~100 units and swings far from 1 even when every unit's policy barely moved, so textbook V-trace would
    truncate nearly every trace. Returns (advantages, value targets)."""
    n = len(rewards)
    ratio = np.exp(np.clip(log_rho, -20, 20))
    rho, c = np.minimum(rho_bar, ratio), lam * np.minimum(c_bar, ratio)
    next_v = np.append(values[1:], 0.0)
    delta = rho * (rewards + gamma * next_v - values)
    vs_minus_v = np.zeros(n, np.float32)
    acc = 0.0
    for t in reversed(range(n)):
        acc = delta[t] + gamma * c[t] * acc
        vs_minus_v[t] = acc
    vs = values + vs_minus_v
    next_vs = np.append(vs[1:], 0.0)
    return (rho * (rewards + gamma * next_vs - values)).astype(np.float32), vs.astype(np.float32)


def upgo(rewards: np.ndarray, values: np.ndarray, gamma: float) -> np.ndarray:
    """UPGO (AlphaStar, Vinyals et al. 2019): the return follows the game only while it goes better than expected
    (the next action's one-step value at least the state's value), and bootstraps from the value otherwise: a
    self-imitation of the game's better-than-expected stretches. Returns its advantages (G - V)."""
    n = len(rewards)
    v_next = np.append(values[1:], 0.0)
    q_next = np.append(rewards[1:] + gamma * np.append(values[2:], 0.0), 0.0)   # Q(s_{t+1}, a_{t+1}) estimate
    g = np.zeros(n, np.float32)
    for t in reversed(range(n)):
        if t == n - 1:
            g[t] = rewards[t]
        else:
            follow = q_next[t] >= v_next[t]
            g[t] = rewards[t] + gamma * (g[t + 1] if follow else v_next[t])
    return (g - values).astype(np.float32)


class Streams:
    """Games laid out as `n` parallel streams of consecutive chunks of `length` decisions: rounds() yields, per
    round, each stream's (game, start) or None (the stream has run out), and whether the chunk starts its game."""

    def __init__(self, lengths: dict[int, int], n: int, length: int, order: list[int], queues=None):
        self.length = length
        self.queues = queues if queues is not None else [order[i::n] for i in range(n)]
        self.lengths = lengths

    def rounds(self):
        pos = [[0, 0] for _ in self.queues]           # [index in queue, start in game]
        while True:
            row, alive = [], False
            for s, q in enumerate(self.queues):
                qi, start = pos[s]
                if qi >= len(q):
                    row.append(None)
                    continue
                alive = True
                g = q[qi]
                row.append((g, start))
                start += self.length
                if start >= self.lengths[g]:
                    qi, start = qi + 1, 0
                pos[s] = [qi, start]
            if not alive:
                return
            yield row


def balanced_queues(order: list[int], lengths: dict[int, int], n: int, rng) -> list[list[int]]:
    """Games dealt to `n` streams so that every stream holds about as many decisions (longest game first to the
    stream with the fewest; `order` breaks ties), each stream's queue then shuffled: dealt round-robin, the streams
    ran out at different times and 18-28 % of an epoch's chunk rounds were padding (2026-10-11 review)."""
    queues, load = [[] for _ in range(n)], [0] * n
    for g in sorted(order, key=lambda i: -lengths[i]):          # sorted() is stable: ties keep `order`'s shuffle
        k = min(range(n), key=lambda j: load[j])
        queues[k].append(g)
        load[k] += lengths[g]
    for q in queues:
        rng.shuffle(q)
    return queues


def chunk_batch(games: Games, row: list, length: int, device, values=None, adv=None, ret=None, adv_up=None,
                loc=None, kl0=None, prox=None, refp=None):
    """One round of streams as a sequence-major batch: observations, recorded actions and log-probabilities,
    teacher labels, masks. Finished streams are padded (invalid)."""
    obs, acts, labels, valid, starts, old_unit_lp, old_lp, sup, v_old, a_, r_ = [], [], [], [], [], [], [], [], [], [], []
    u_ = []
    k_ = []
    supw = []
    prox_rows, ref_rows = [], []
    from fa_env.tools import GRID
    loc_rows = []
    opp_rows = []
    opponents = getattr(games, "opponent", None)
    pad_obs = None
    from fa_env.factory_upgrades import label_prefix as capability_label_prefix
    label_prefix = capability_label_prefix(games)
    for item in row:
        if item is None:
            continue
        pad_obs = pad_obs or games.step_obs(games.games[item[0]], 0)
    for item in row:
        for t in range(length):
            if item is None or item[1] + t >= games.lengths[item[0]]:
                g, step = (item[0], games.lengths[item[0]] - 1) if item is not None else (None, 0)
                obs.append(games.step_obs(games.games[g], step) if g is not None else pad_obs)
                opp_rows.append(opponents[g] if opponents is not None and g is not None else np.zeros(N_OPPONENT, np.float32))
                n = len(obs[-1]["tokens"])
                acts.append({k: np.full(n, 0 if k == "uverb" else -1, np.int64) for k in UNIT_KEYS})
                labels.append({k: np.full(n, -1, np.int64) for k in UNIT_KEYS})
                valid.append(False)
                starts.append(False)
                old_unit_lp.append(np.zeros(n, np.float32))
                prox_rows.append(np.zeros(n, np.float32))
                ref_rows.append(np.zeros(n, np.float32))
                old_lp.append(0.0)
                sup.append(False)
                supw.append(0.0)
                v_old.append(0.0), a_.append(0.0), r_.append(0.0), u_.append(0.0), k_.append(0.0)
                loc_rows.append(None)
                continue
            gi, step = item[0], item[1] + t
            g = games.games[gi]
            o, e = int(g["offsets_t"][step]), int(g["offsets_t"][step + 1])
            obs.append(games.step_obs(g, step))
            opp_rows.append(opponents[gi] if opponents is not None else np.zeros(N_OPPONENT, np.float32))
            acts.append({k: np.asarray(g[f"pol_{k}"][o:e], np.int64) for k in UNIT_KEYS})
            if label_prefix and label_prefix + 'uverb' not in g:
                raise ValueError('Air-refueling labels require current cache format')
            labels.append({k: np.asarray(g[label_prefix + k][o:e], np.int64) for k in UNIT_KEYS})
            valid.append(bool(g["pol_has"][step]))
            starts.append(step == 0)
            old_unit_lp.append(np.asarray(g["pol_unit_logp"][o:e], np.float32))
            prox_rows.append(prox[gi][o:e] if prox is not None else old_unit_lp[-1])
            ref_rows.append(refp[gi][o:e] if refp is not None else np.zeros(e - o, np.float32))
            old_lp.append(float(g["pol_logp"][step]))
            # imitation only where a teacher labelled the game: without one, encoding takes the network's own
            # commands as the labels (tools_eval --labels-share leaves some games unlabelled)
            sup.append(bool(g[label_prefix + "unit_supervise"][step]) and getattr(games, "teacher", {}).get(gi, True))
            supw.append(games.distill_weight.get(gi, 1.0) if hasattr(games, "distill_weight") else 1.0)
            if values is not None:
                v_old.append(float(values[gi][step]))
                a_.append(float(adv[gi][step]))
                r_.append(float(ret[gi][step]))
                u_.append(float(adv_up[gi][step]) if adv_up is not None else 0.0)
                loc_rows.append(loc[gi][step] if loc is not None and loc.get(gi) is not None else None)
                k_.append(float(kl0[gi][step]) if kl0 is not None else 0.0)
            else:
                v_old.append(0.0), a_.append(0.0), r_.append(0.0), u_.append(0.0), k_.append(0.0)
                loc_rows.append(None)
    b = collate(obs, device)
    if opponents is not None:
        b["opponent"] = torch.as_tensor(np.stack(opp_rows), dtype=torch.float32, device=device)
    T = b["tokens"].shape[1]
    act = {k: torch.as_tensor(v, device=device) for k, v in stack_actions(acts, T, "units").items()}
    lab = {k: torch.as_tensor(v, device=device) for k, v in stack_actions(labels, T, "units").items()}
    def rows(xs):
        out = np.zeros((len(obs), T), np.float32)
        for i, x in enumerate(xs):
            out[i, : len(x)] = x
        return out

    ulp = rows(old_unit_lp)
    f = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
    extra = {}
    aux = getattr(games, "aux_econ", None)
    if aux is not None:
        y, m = np.zeros((len(obs), N_AUX_ECON), np.float32), np.zeros((len(obs), N_AUX_ECON), np.float32)
        for s, item in enumerate(row):
            if item is not None and item[0] in aux:
                gi, start = item
                n = min(length, games.lengths[gi] - start)
                y[s * length:s * length + n] = aux[gi][0][start:start + n]
                m[s * length:s * length + n] = aux[gi][1][start:start + n]
        extra = dict(aux_y=f(y), aux_m=f(m))
    return dict(b=b, act=act, lab=lab, valid=f(valid), starts=f(starts, torch.bool), old_ulp=f(ulp), **extra,
                old_lp=f(old_lp), sup=f(sup), supw=f(supw), v_old=f(v_old), adv=f(a_), ret=f(r_), adv_up=f(u_),
                loc=f(np.stack([x if x is not None else np.zeros(GRID * GRID, np.float32) for x in loc_rows])),
                loc_has=f([x is not None for x in loc_rows]), kl0=f(k_), prox_ulp=f(rows(prox_rows)),
                ref_ulp=f(rows(ref_rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, type=Path, help="the network to improve (the one that played)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", action="append", required=True, help="encoded recorded games (with pol_* arrays)")
    ap.add_argument("--reference", type=Path, help="frozen KL anchor (default: --init)")
    ap.add_argument("--max-games", type=int, default=0, help="only the newest this many games (0: all)")
    ap.add_argument("--mapped", type=Path,
                    help="read the games memory-mapped from this folder (unpacked there once; folders of games no "
                         "longer chosen are removed) instead of loaded into RAM")
    ap.add_argument("--capacity-clip", type=float, default=None,
                    help="the economy capacity potential's change per decision clipped to +-this (capacity/20 units)")
    ap.add_argument("--mc-warmup", action="store_true",
                    help="the value warm-up fits Monte-Carlo returns (the games' own discounted rewards), not V-trace "
                         "targets built from the old critic, which a reward change barely reaches")
    ap.add_argument("--value-lr", type=float, default=None, help="the critic's own learning rate (default: --lr)")
    ap.add_argument("--value-clip", type=float, default=None,
                    help="clip of the value update around the iteration's start values (default: --clip; 0: off)")
    ap.add_argument("--local-clip", type=float, default=3.0,
                    help="local-credit advantages clipped to +-this many standard deviations (0: off)")
    ap.add_argument("--no-undo-labels", action="store_true",
                    help="keep M28's 'nothing' labels on units the network ordered in the same decision")
    ap.add_argument("--opponent-critic", action="store_true",
                    help="the critic is told the game's opponent (from the recording's name; never when playing)")
    ap.add_argument("--timeout-scale", type=float, default=None,
                    help="a timeout's terminal: this x the value advantage (-1..1), clipped to +-10 (default: as recorded, 5x)")
    ap.add_argument("--timeout-cost", type=float, default=0.0, help="taken off a timeout's terminal (with --timeout-scale)")
    ap.add_argument("--rounds-file", type=Path,
                    help="JSON list of round folder names: only games recorded in them (rl_loop: the lineage's)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--streams", type=int, default=16, help="games side by side")
    ap.add_argument("--length", type=int, default=32, help="decisions per chunk")
    ap.add_argument("--micro", type=int, default=4, help="streams per forward pass (gradients accumulated)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=30,
                    help="optimizer steps over which the learning rate rises to --lr: the policy acts at temperature "
                         "0.5 and is sharp, and a fresh Adam moves every weight by ~lr at once (6 steps at 3e-5: "
                         "KL 0.001 -> 0.19)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--econ-source", choices=("recorded", "capacity"), default="recorded",
                    help="the economy reward: the recorded change of mass income, or of mass production capacity "
                         "plus reclaim (see rewards_of)")
    ap.add_argument("--reclaim-coef", type=float, default=0.5, help="with --econ-source capacity: mass reclaimed")
    ap.add_argument("--intel-coef", type=float, default=0.0, help="potential: how much of the enemy we know")
    ap.add_argument("--map-coef", type=float, default=0.0, help="potential: (our mexes - enemy mexes) / deposits")
    ap.add_argument("--waste-coef", type=float, default=0.0, help="resources wasted to full storage (mass)")
    ap.add_argument("--stall-coef", type=float, default=0.0,
                    help="mass spending lost to energy stalls (the game slows all building to what it can power)")
    ap.add_argument("--value-warmup", type=int, default=1,
                    help="when the reward differs from the one the network's critic learned: fit the value head "
                         "alone for this many epochs first (the policy unchanged), then compute the advantages")
    ap.add_argument("--value-warmup-every", type=int, default=0,
                    help="fit the value head alone for this many epochs EVERY iteration, before the advantages (with "
                         "--mc-warmup to the games' own returns): from rl_167 the critic was +0.2..+0.6 too optimistic "
                         "at the first decision, most against M28 (2026-10-26 data review T2) -- after centring, an "
                         "opponent-dependent bias still tilts the advantages (0: only after a reward change)")
    ap.add_argument("--acu-coef", type=float, default=0.0,
                    help="the commander's exposure (its walk home, phi_commander) as a potential, this much at full "
                         "exposure; large maps only unless --acu-all-maps. 2026-10-13: on 20 km maps our commander "
                         "pushed to the enemy after minute 20 (74 %% of its orders past halfway, M28 37 %%) and died "
                         "there 25-35 minutes in, too late for the -10 to reach the walk out")
    ap.add_argument("--acu-grace", type=float, default=1.0, help="minutes of walk home that cost nothing (its base)")
    ap.add_argument("--acu-span", type=float, default=4.0, help="minutes beyond the grace to full exposure")
    ap.add_argument("--acu-all-maps", action="store_true")
    ap.add_argument("--distill-large-min", type=float, default=0.0,
                    help="the large maps' imitation weight is at least this. Off: at 0.4 (rl_090-092, 2026-10-13) the "
                         "pull toward M28 on the large maps took the small maps with it -- M28 x0.6 won 61 %% (rl_089), "
                         "47 %%, 20 %%; the network is shared, so is the imitation")
    ap.add_argument("--sil", type=Path,
                    help="self-imitation (Oh et al. 2018): a JSON list of [source_root, source_name] recordings -- "
                         "the success archive of fa_train.rl_loop: wins against the strongest opponents, kept across "
                         "iterations. After the PPO epochs, the policy imitates their decisions where the game went "
                         "better than the critic expected ((R - V)+), and the critic learns those returns")
    # 2026-10-12: at 0.1, a cap of 2 and the critic pulled to the archive's returns, one self-imitation pass after
    # rl_080 (archive of wins against M28 x0.65+, 89 % of its decisions better than expected) moved the policy by
    # ref KL +0.05 and left the critic optimistic: explained variance 0.79 -> 0.16, rl_081 stopped in epoch 1
    ap.add_argument("--sil-coef", type=float, default=0.03)
    ap.add_argument("--sil-value-coef", type=float, default=0.0,
                    help="the critic pulled up to the archive's returns (Oh et al.): off -- wins picked from many games "
                         "teach it that every state is as good as the best outcome")
    ap.add_argument("--sil-epochs", type=int, default=1)
    ap.add_argument("--sil-max-adv", type=float, default=1.0, help="self-imitation advantages capped at this many "
                    "standard deviations of the PPO advantages")
    ap.add_argument("--probe-games", type=int, default=12,
                    help="complete recorded games replayed before/after SIL to measure final policy drift (0: off)")
    ap.add_argument("--econ-coef", type=float, default=0.0,
                    help="weight of the economy part of the trade (mexes and engineers destroyed or lost, valued "
                         "beyond their cost: tools_bc.economy_bonus), on top of the recorded trade reward")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="entropy bonus per unit (none by default: sampling at temperature 0.5 explores already)")
    ap.add_argument("--ref-coef", type=float, default=0.2)
    ap.add_argument("--distill", type=float, default=0.5, help="weight of M28's labels (the recording's teacher)")
    ap.add_argument("--distill-builder-weight", type=float, default=1.0,
                    help="relative weight of all builder teacher labels, including END (1: legacy RL objective)")
    ap.add_argument("--distill-idle-factory-end-weight", type=float, default=1.0,
                    help="teacher END weight for idle empty-queue factories with >=25%% mass/energy storage "
                         "and unit-cap room (0..1; default unchanged; an action-rate ablation)")
    ap.add_argument("--distill-large", type=float, default=1.0,
                    help="M28's labels count this many times more in games on maps above 10 km. 2026-10-12: on "
                         "20 km maps the network lost 99 %% of games even to the scripted AIs, its commander pushing "
                         "to the enemy (median 72 %% of the way after minute 20) where M28 pulls it home (19 %%): the "
                         "small maps' commander push, generalised. M28 plays big maps better than we do"),
    ap.add_argument("--shaping", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.5, help="the temperature the games were played at")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--processes", type=int, default=0,
                    help="CPU: this many worker processes for the first pass and the epochs (fa_train.learn_parallel), "
                         "each with --cpu-threads threads; this process then uses one thread until they are done")
    ap.add_argument("--capacity-scale", type=float, default=0.05,
                    help="with --econ-source capacity: reward per mass/s of production capacity gained (a potential: "
                         "credit now for income later). 0.05 is what it always was; at 0.05 a T1 to T2 extractor upgrade "
                         "earns 0.2 for 900 mass while the same mass of army can trade for about 0.9")
    ap.add_argument("--aux-econ", type=float, default=0.0,
                    help="weight of the auxiliary loss on the memory: income, army and capacity 1, 5 and 20 minutes ahead")
    ap.add_argument("--chunks-per-step", type=int, default=0,
                    help="with --processes: finished chunks per optimizer step (0: --streams); --streams can then be "
                         "more than this, so that every worker has a stream to work on")
    ap.add_argument("--parallel-groups", type=int, default=1,
                    help="micro-batches run side by side on this many threads, each with --cpu-threads threads")
    ap.add_argument("--no-critic-features", dest="critic_features", action="store_false",
                    help="run the trunk again for the critic's warm-up and values instead of keeping its first pass")
    ap.add_argument("--installed-enhancements", action="store_true",
                    help="enable the zero-initialized input for installed own-upgrade identities")
    ap.add_argument("--air-refueling", action="store_true", help="own fuel and typed staging actions")
    ap.add_argument("--queued-factory-upgrades", action="store_true", help="allow upgrades behind one factory production order")
    ap.add_argument("--smooth-verb-floor", action="store_true", help="opt-in exact continuous verb distribution; requires matching fresh behavior records")
    ap.add_argument("--device")
    add_cpu_arguments(ap)
    ap.add_argument("--step-lstm", action="store_true", help="use timestep LSTM calls for comparison")
    ap.add_argument("--separate-decoder", action="store_true", help="score policy and teacher separately for comparison")
    ap.add_argument("--no-trim-batches", action="store_true", help="retain padded microbatches for comparison")
    ap.add_argument("--stop-file", type=Path, help="stop at once, saving nothing, when this exists (the GPU watchdog)")
    ap.add_argument("--vtrace", type=int, default=1, help="1: V-trace targets and advantages (per-unit geometric-mean "
                    "ratios); 0: plain GAE")
    ap.add_argument("--upgo", type=float, default=0.5, help="weight of the UPGO policy term (0: off)")
    ap.add_argument("--local-credit", type=float, default=0.0,
                    help="spatial credit: weight of each unit's local advantage (its cell's coming local trade "
                         "against the local critic) added to the decision's (0: off)")
    ap.add_argument("--local-gamma", type=float, default=0.9, help="discount of the local trade (~10 decisions)")
    ap.add_argument("--local-scale", type=float, default=500.0, help="mass per unit of local return")
    ap.add_argument("--local-coef", type=float, default=0.5, help="weight of the local critic's loss")
    a = ap.parse_args()
    if not math.isfinite(a.distill_builder_weight) or a.distill_builder_weight <= 0:
        ap.error("--distill-builder-weight must be finite and positive")
    if not math.isfinite(a.distill_idle_factory_end_weight) or not 0 <= a.distill_idle_factory_end_weight <= 1:
        ap.error("--distill-idle-factory-end-weight must be finite and in [0, 1]")
    check_stop(a)                            # an intake-time stop must not start a new GPU learner
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.backends.cuda.matmul.allow_tf32 = False           # exact ratios
    for name in ("streams", "length", "micro"):
        number = getattr(a, name)
        if number is not None and number < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    try:
        cpu_runtime = configure_cpu(a, device)
        if int(a.processes or 0) > 1:
            torch.set_num_threads(1)
    except (ValueError, OSError, RuntimeError) as error:
        ap.error(str(error))
    if device.type == "cpu":
        print(f"CPU execution {json.dumps(cpu_runtime)}, checkpointing {a.grad_checkpoint}", flush=True)
    t0 = time.monotonic()
    net, state = load_net(a.init, device)
    if a.smooth_verb_floor:
        net.smooth_verb_floor = True
    if a.installed_enhancements:
        net.installed_enhancements = True
    if a.air_refueling:
        net.air_refueling = True
    if a.queued_factory_upgrades:
        net.queued_factory_upgrades = True
    net.grad_checkpoint = bool(a.grad_checkpoint)
    net.sequence_lstm = not a.step_lstm
    configure_model(net, a)
    ref, ref_state = load_net(a.reference or a.init, device)
    ref.sequence_lstm = net.sequence_lstm
    configure_model(ref, a)
    # An anchor from before the fine head and the close-ups has none: it takes the network's own (as they were
    # trained when they joined), else its KL would price the fine pixels against nothing (a pull to certainty)
    joined = {k: v for k, v in net.state_dict().items()
              if k.startswith(("fine_", "focus_")) and k not in ref_state["model"]}
    if joined:
        ref.load_state_dict({**ref.state_dict(), **joined})
        ref.fine_positions = net.fine_positions
        print(f"reference: the network's fine head and close-ups ({len(joined)} tensors), it had none", flush=True)
    if net.air_refueling:
        # The same legal action support is needed by the anchor; an old masked
        # action must not impose an infinite KL penalty on a newly legal action.
        if 'refuel_weight' not in ref_state['model']:
            ref.refuel_weight.data.copy_(net.refuel_weight.data)
    # both ways: a reference saved with the flag kept its wider masks for a network without it
    ref.air_refueling = net.air_refueling
    ref.queued_factory_upgrades = net.queued_factory_upgrades
    from .probability_accounting import align_reference_distribution
    align_reference_distribution(net, ref)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    # only the newest games are loaded (with a margin for games without recorded choices): the cache keeps them all
    # only the lineage's rounds (rl_loop): the cache keeps every game encoded, and after a fallback the newest
    # were the abandoned branch's -- every restart from rl_089 trained first on them (rl_102 on rl_096-098's
    # games, rl_113 on the commander-reward networks', 2026-10-14) and went downhill
    only = set(json.loads(a.rounds_file.read_text())) if a.rounds_file else None
    # in RAM a game is ~99 MB since the fine positions and close-ups (2026-10-14: 300 games 30 GB, 540 games
    # 53 GB; training died next to a 42 GB program): --mapped reads them from disk as they are needed
    games = Games([Path(c) for c in a.cache], newest=a.max_games + 60 if a.max_games else None, rounds=only,
                  mapped=a.mapped)
    if a.mapped:
        import shutil
        keep = {f.name for f in games.folders}
        for old in a.mapped.iterdir():
            if old.is_dir() and old.name not in keep:
                shutil.rmtree(old, ignore_errors=True)
    if only is not None:
        print(f"games: {len(games.games)} from the lineage's {len(only)} rounds", flush=True)
    games.teacher = {i: bool(_meta(g).get("label_source")) for i, g in enumerate(games.games)}
    print(f"imitation: {sum(games.teacher.values())} of {len(games.games)} games carry a teacher's labels", flush=True)
    if a.opponent_critic:
        games.opponent = {i: opponent_features(opponent_of_recording(g["_cache_manifest"].get("source_name", "")))
                          for i, g in enumerate(games.games)}
        kinds = {}
        for i, g in enumerate(games.games):
            k = opponent_of_recording(g["_cache_manifest"].get("source_name", "")).split("@")[0] or "?"
            kinds[k] = kinds.get(k, 0) + 1
        print(f"critic: told the opponent ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})", flush=True)
    games.units = True
    games.air_refueling = net.air_refueling
    games.queued_factory_upgrades = net.queued_factory_upgrades
    usable = [i for i, g in enumerate(games.games)
              if "pol_has" in g and bool(np.asarray(g["pol_has"]).any()) and all(f"rew_{c}" in g for c in REWARD_COMPONENTS)]
    from .probability_accounting import compatible_ppo_games, semantics
    usable, incompatible = compatible_ppo_games(net, games, usable, a.temperature)
    print(f"PPO probability semantics: {semantics(net)}; {len(usable)} compatible, {len(incompatible)} incompatible games excluded", flush=True)
    # the large maps' weight, at least --distill-large-min whatever the adaptive weight is: it falls as the small
    # maps are won (0.05 on 2026-10-13, so x4 was 0.2), while the large maps are still lost and M28 plays them better
    large = max(a.distill_large, a.distill_large_min / max(a.distill, 1e-6)) if a.distill > 0 else a.distill_large
    games.distill_weight = {i: large for i in usable if map_km(games.games[i]) > LARGE_KM}
    if large != 1.0:
        print(f"imitation: {len(games.distill_weight)} games on maps above {LARGE_KM} km, M28's labels x{large:g} "
              f"({a.distill * large:.3f})", flush=True)
    if a.max_games:                                    # the newest games (by when they were encoded)
        usable = sorted(usable, key=lambda i: recency_key(games.games[i]["_cache_manifest"]))[-a.max_games:]
    if not usable:
        raise SystemExit("no compatible games with recorded choices and rewards; collect matching fresh behavior data")
    if needs_shaping_arrays(a):
        carrying = [i for i in usable if all(k in games.games[i] for k in SHAPING_ARRAYS)]
        if len(carrying) >= max(60, len(usable) // 2):
            usable = carrying
        else:                                   # games encoded before the arrays existed: this iteration as before
            print(f"reward: only {len(carrying)} of {len(usable)} games carry the shaping arrays; the recorded "
                  f"reward this iteration", flush=True)
            a.econ_source, a.econ_coef, a.intel_coef, a.map_coef, a.waste_coef = "recorded", 0.0, 0.0, 0.0, 0.0
    spec = reward_spec(a)
    learned = (state.get("learn_rl") or {}).get("reward_spec")
    warm = a.value_warmup > 0 and learned != spec and learned is not None or (learned is None and spec != LEGACY_SPEC
                                                                             and a.value_warmup > 0)
    print(f"reward: {spec}" + (f" (the critic learned: {learned or 'the recorded reward'}; value warm-up first)"
                               if warm else ""), flush=True)
    ret_std = float(state.get("ret_std") or 1.0)
    lengths = {i: int(games.lengths[i]) for i in usable}
    print(f"{len(usable)} games with recorded choices ({sum(lengths.values())} decisions); network "
          f"{parameter_count(net) / 1e6:.2f} M, options {model_options(net)}; returns scaled by {ret_std:.2f}",
          flush=True)
    if a.aux_econ > 0:
        prepare_aux_econ(games, usable, lengths)
    order = sorted(usable, key=lambda i: -lengths[i])          # long games first: streams end together
    streams = Streams(lengths, a.streams, a.length, order)

    # 1. values of every decision with the network as it is (no gradients), memory carried chunk to chunk
    net.eval()
    values = {i: np.zeros(lengths[i], np.float32) for i in usable}
    log_rho = {i: np.zeros(lengths[i], np.float32) for i in usable}
    kl_sum, kl_n = 0.0, 0.0
    kl0 = {i: np.zeros(lengths[i], np.float32) for i in usable}
    # decoupled PPO (Hilton et al. 2021): the clip and the KL stop are taken against the network as it is now (the
    # 'proximal' policy), the games' own policy enters only through V-trace's truncated ratio in the advantages.
    # Before (to 2026-10-11) the ratio was taken against the recorded policy: with games of older networks that
    # corrected the lag twice, and the KL stop measured the data's lag, not the update (it never fired).
    tokens_of = {i: int(games.games[i]["offsets_t"][lengths[i]]) for i in usable}
    prox = {i: np.zeros(tokens_of[i], np.float32) for i in usable}
    refp = {i: np.zeros(tokens_of[i], np.float32) for i in usable} if a.ref_coef > 0 else None
    state_h = net.initial_state(a.streams, device)
    ref_h1 = ref.initial_state(a.streams, device)
    features = {} if a.critic_features else None     # the trunk's output per decision: the critic's fit needs no other pass
    processes = int(a.processes or 0) if device.type == "cpu" else 0
    if processes > 1:
        from .learn_parallel import first_pass
        values, log_rho, kl0, prox, refp, features, kl_sum, kl_n = first_pass(
            net, ref, games, usable, lengths, tokens_of, a, processes, a.ref_coef > 0, features is not None,
            chunk_batch, micro_view, CHUNK_THREADS(a))
    with torch.no_grad():
        for row in (streams.rounds() if processes <= 1 else ()):
            c = chunk_batch(games, row, a.length, device)
            N, T_all = a.streams * a.length, c["b"]["tokens"].shape[1]
            v_all = torch.zeros(N)
            z_all = torch.zeros(N, net.core_dim)
            ulp_all = torch.zeros(N, T_all)
            known_all = torch.zeros(N, T_all)
            ref_all = torch.zeros(N, T_all) if refp is not None else None
            def first_pass_group(group):
                sub, b_g, act_g, _ = micro_view(c, group, a.streams, a.length, not getattr(a, "no_trim_batches", False))
                rows_g = (group.cpu().unsqueeze(1) * a.length + torch.arange(a.length)).view(-1)
                st = (state_h[0][:, group].contiguous(), state_h[1][:, group].contiguous())
                h, m, z, (hs, cs) = net.core(b_g, st, a.length, sub["starts"], return_state=True)
                state_h[0][:, group], state_h[1][:, group] = hs, cs
                v_all[rows_g] = net.value_of(z, b_g).float().cpu()
                z_all[rows_g] = z.float().cpu()
                _, ent, _ = net.score(h, m, z, b_g, act_g, temperature=a.temperature)
                T_g = ent["unit_logp"].shape[1]
                ulp_all[rows_g, :T_g] = ent["unit_logp"].float().cpu()
                known_all[rows_g, :T_g] = ent["unit_known"].float().cpu()
                if refp is not None:
                    # the frozen reference's log-probabilities, once here instead of a forward pass every PPO step
                    st = (ref_h1[0][:, group].contiguous(), ref_h1[1][:, group].contiguous())
                    hr, mr, zr, (hs, cs) = ref.core(b_g, st, a.length, sub["starts"], return_state=True)
                    ref_h1[0][:, group], ref_h1[1][:, group] = hs, cs
                    _, ent_r, _ = ref.score(hr, mr, zr, b_g, act_g, temperature=a.temperature)
                    ref_all[rows_g, :T_g] = ent_r["unit_logp"].float().cpu()
            map_groups(first_pass_group, micro_groups(c, a.streams, a.length, a.micro,
                                                      not getattr(a, "no_trim_batches", False)), a)
            v = v_all.view(a.streams, a.length).numpy()
            if features is not None:
                keep_features(features, row, z_all, c["b"]["privileged"], lengths, a)
            lr_ = np.zeros((a.streams, a.length), np.float32)
            ent = {"unit_logp": ulp_all.to(device), "unit_known": known_all.to(device)}
            ulp_now = ulp_all.numpy()
            if refp is not None:
                ulp_ref = ref_all.numpy()
            for s_, item in enumerate(row):
                if item is None:
                    continue
                gi, start = item
                off = games.games[gi]["offsets_t"]
                for t in range(min(a.length, lengths[gi] - start)):
                    o, e = int(off[start + t]), int(off[start + t + 1])
                    prox[gi][o:e] = ulp_now[s_ * a.length + t, : e - o]
                    if refp is not None:
                        refp[gi][o:e] = ulp_ref[s_ * a.length + t, : e - o]
            if a.vtrace:
                known = ent["unit_known"]
                d_unit = (ent["unit_logp"] - c["old_ulp"]) * known
                per_decision = d_unit.sum(1) / known.sum(1).clamp(min=1.0)
                ok = c["valid"] > 0
                kl_unit = (((d_unit.exp() - 1) - d_unit) * known).sum(1) / known.sum(1).clamp(min=1.0)
                kl_sum += float((kl_unit * ok).sum())
                kl_n += float(ok.sum())
                kl_rows = kl_unit.view(a.streams, a.length).cpu().numpy()
                lr_ = per_decision.view(a.streams, a.length).cpu().numpy()
            for s, item in enumerate(row):
                if item is None:
                    continue
                gi, start = item
                n = min(a.length, lengths[gi] - start)
                values[gi][start:start + n] = v[s, :n]
                # a decision without a usable recorded choice was scored as "every unit chose nothing" (verb 0,
                # log-prob 0): its ratio is meaningless (0.36 measured), so treat it as on-policy (rho = c = 1)
                log_rho[gi][start:start + n] = lr_[s, :n] * np.asarray(games.games[gi]["pol_has"][start:start + n],
                                                                    np.float32)
                if a.vtrace:
                    kl0[gi][start:start + n] = kl_rows[s, :n]
    rewards = {gi: rewards_of(games.games[gi], a)[: lengths[gi]] / ret_std for gi in usable}

    def advantages():
        adv, ret, adv_up = {}, {}, {}
        for gi in usable:
            r = rewards[gi]
            if a.vtrace:
                adv[gi], ret[gi] = vtrace(r, values[gi], log_rho[gi], a.gamma, a.lam)
            else:
                adv[gi], ret[gi] = gae(r, values[gi], a.gamma, a.lam)
            adv_up[gi] = upgo(r, values[gi], a.gamma) if a.upgo > 0 else np.zeros(lengths[gi], np.float32)
            if a.vtrace and a.upgo > 0:
                adv_up[gi] = adv_up[gi] * np.minimum(1.0, np.exp(np.clip(log_rho[gi], -20, 20)))
        return adv, ret, adv_up

    adv, ret, adv_up = advantages()
    mc_ret = {gi: discounted(rewards[gi], a.gamma).astype(np.float32) for gi in usable}
    critic_bias(values, mc_ret, usable, games, "before")
    changed = warm or (a.mc_warmup and a.value_warmup > 0 and learned != spec)
    if changed or a.value_warmup_every > 0:
        before = 1 - float(np.var(np.concatenate([ret[gi] - values[gi] for gi in usable]))
                           / max(np.var(np.concatenate([ret[gi] for gi in usable])), 1e-8))
        # Monte-Carlo targets (the data is near on-policy: KL ~0.02): V-trace targets are built from the stale
        # critic, so a reward change reached 0.6 % of the true value change 100 decisions before the end (review B4)
        if features is not None:
            value_warmup_on_features(net, games, usable, features, mc_ret if a.mc_warmup else ret, a, device,
                                     epochs=a.value_warmup if changed else a.value_warmup_every)
            values = values_from_features(net, games, usable, features, device)
        else:
            value_warmup(net, games, usable, lengths, mc_ret if a.mc_warmup else ret, a, device,
                         epochs=a.value_warmup if changed else a.value_warmup_every)
            values = value_pass(net, games, usable, lengths, a, device)
        adv, ret, adv_up = advantages()
        after = 1 - float(np.var(np.concatenate([ret[gi] - values[gi] for gi in usable]))
                          / max(np.var(np.concatenate([ret[gi] for gi in usable])), 1e-8))
        print(f"value warm-up: explained variance {before:+.2f} -> {after:+.2f}", flush=True)
        critic_bias(values, mc_ret, usable, games, "after the warm-up")
    all_adv = np.concatenate([adv[gi] for gi in usable])
    mean, std = float(all_adv.mean()), float(all_adv.std() + 1e-8)
    for gi in usable:
        adv[gi] = (adv[gi] - mean) / std
        adv_up[gi] = adv_up[gi] / std                       # the same scale as the main advantages
    loc = None
    if a.local_credit > 0:
        loc = {gi: local_returns(games.games[gi], lengths[gi], a.local_gamma, a.local_scale) for gi in usable}
        with_loc = [gi for gi in usable if loc[gi] is not None]
        if with_loc:
            active = np.concatenate([loc[gi][np.abs(loc[gi]) > 1e-4] for gi in with_loc] or [np.zeros(1)])
            loc_std = float(active.std() + 1e-8) if active.size > 1 else 1.0
            for gi in with_loc:
                loc[gi] = loc[gi] / loc_std                     # the scale of the decision advantages
                if a.local_clip > 0:
                    # tails to 132 sd: 0.74 % of the values carried 18.7 % of the total (review L2)
                    loc[gi] = np.clip(loc[gi], -a.local_clip, a.local_clip)
            print(f"local credit: {len(with_loc)} of {len(usable)} games carry local trade; {active.size} active "
                  f"cell-decisions, std {loc_std:.4f} (normalised to 1)", flush=True)
        else:
            loc = None
            print("local credit: no game carries local trade (encoded before it): off for this iteration", flush=True)
    kl_start = kl_sum / max(1.0, kl_n)
    print(f"KL of the data to the network now (the lag of the games): {kl_start:.4f}{at()}", flush=True)
    if a.vtrace:
        rho = np.exp(np.concatenate([log_rho[gi] for gi in usable]))
        print(f"v-trace: per-unit geometric-mean ratio median {np.median(rho):.4f}, "
              f"above 1 (truncated) {float((rho > 1).mean()):.1%}", flush=True)
    returns = np.concatenate([ret[gi] for gi in usable])
    vals = np.concatenate([values[gi] for gi in usable])
    explained = 1 - float(np.var(returns - vals) / max(np.var(returns), 1e-8))
    # against the games' own discounted returns: the V-trace targets above are built from the values themselves,
    # which makes that figure look better than the critic is (2026-10-11 review)
    mc = np.concatenate([mc_ret[gi] for gi in usable])
    explained_mc = 1 - float(np.var(mc - vals) / max(np.var(mc), 1e-8))
    print(f"values: explained variance {explained:+.2f} (of the games' returns {explained_mc:+.2f}); "
          f"advantages normalised (mean {mean:+.3f}, std {std:.3f})", flush=True)
    if a.upgo > 0:
        up_all = np.concatenate([adv_up[gi] for gi in usable])
        print(f"upgo: mean advantage {up_all.mean():+.3f} (scale of the normalised advantages), "
              f"positive {float((up_all > 0).mean()):.0%}", flush=True)

    # 2. PPO epochs
    net.train()
    value_params = {id(p) for mod in (net.value_head, net.opp_value, net.aux_econ) for p in mod.parameters()}
    groups = [{"params": [p for p in net.parameters() if id(p) not in value_params], "base_lr": a.lr}]
    groups.append({"params": [p for p in net.parameters() if id(p) in value_params], "base_lr": a.value_lr or a.lr})
    opt = torch.optim.Adam([dict(g, lr=g["base_lr"]) for g in groups], eps=1e-5)
    history = []
    a.step = 0
    epoch_pool = None
    for epoch in range(a.epochs):
        rng = np.random.default_rng(epoch)
        order = [int(i) for i in rng.permutation(usable)]
        streams = Streams(lengths, a.streams, a.length, order, balanced_queues(order, lengths, a.streams, rng))
        state_h = net.initial_state(a.streams, device)
        ref_h = ref.initial_state(a.streams, device)
        sums = {k: 0.0 for k in ("pg", "v", "ent", "kl", "clip", "ref_kl", "distill", "local", "kl_data", "aux", "gnorm",
                                 "n")}
        stopped = False
        if processes > 1:
            from .learn_parallel import EpochPool
            if epoch_pool is None:
                epoch_pool = EpochPool(net, games, usable, lengths, a, processes, CHUNK_THREADS(a), chunk_batch, ppo_group,
                                       dict(values=values, adv=adv, ret=ret, adv_up=adv_up, loc=loc,
                                            kl0=kl0 if a.vtrace else None, prox=prox, refp=refp))
            sums, stopped = epoch_pool.run_epoch(streams.queues, opt)
            # fresh workers every epoch: a worker's heap only grows (2026-11-08: 26 workers at 5 GB each after 45
            # minutes, and the kernel killed the learner)
            epoch_pool.close()
            epoch_pool = None
        for row, c in (() if processes > 1 else prefetched(streams.rounds(), lambda r: chunk_batch(
                games, r, a.length, device, values, adv, ret, adv_up, loc, kl0 if a.vtrace else None, prox, refp))):
            if a.stop_file and a.stop_file.exists():
                raise SystemExit(f"{a.stop_file} found: stopping without saving")
            stats, state_h, ref_h = ppo_step(net, ref, opt, c, state_h, ref_h, a)
            for k, v in stats.items():
                sums[k] += v
            # this iteration's own movement: KL to the network at the iteration's start (the proximal policy);
            # judged after a few chunks, a single chunk is too noisy
            if sums["n"] >= 8 and sums["kl"] / sums["n"] > 1.5 * a.target_kl:
                stopped = True
                break
        n = max(1.0, sums["n"])
        rec = {k: round(v / n, 5) for k, v in sums.items() if k != "n"}
        rec.update(epoch=epoch + 1, chunks=int(sums["n"]), stopped_early=stopped, kl_start=round(kl_start, 5))
        history.append(rec)
        print(f"epoch {epoch + 1}: {json.dumps(rec)}{at()}", flush=True)
        if stopped:
            break
    if epoch_pool is not None:
        epoch_pool.close()
    probe_ids = select_probe_games(games, usable, a.probe_games)
    diagnostics = []
    before_sil = None
    if probe_ids:
        diag, before_sil = policy_probe(net, games, probe_ids, lengths, a, device, prox, refp)
        diag["phase"] = "after_ppo"
        diagnostics.append(diag)
        print(f"policy probe: {json.dumps(diag)}", flush=True)
    if a.sil and a.sil.exists() and not stopped:
        sil_rec = self_imitation(net, opt, a, device, ret_std, std)
        if sil_rec:
            history.append(sil_rec)
            print(f"self-imitation: {json.dumps(sil_rec)}{at()}", flush=True)
            if probe_ids:
                diag, _ = policy_probe(net, games, probe_ids, lengths, a, device, prox, refp, before_sil)
                diag["phase"] = "after_sil"
                diagnostics.append(diag)
                print(f"policy probe: {json.dumps(diag)}", flush=True)
    check_stop(a)
    out = {"model": net.state_dict(), "size": state.get("size", "M"), "decoder": "units",
           "model_options": model_options(net), "ret_std": ret_std, "kind": "tools",
           "learn_rl": {"init": str(a.init), "reference": str(a.reference or a.init), "games": len(usable),
                        "reward_spec": spec, "policy_diagnostics": diagnostics,
                        "execution": cpu_runtime if device.type == "cpu" else {"device": str(device)},
                        "probability_semantics": semantics(net), "incompatible_games_excluded": len(incompatible),
                        "history": history, "explained_variance": explained, "explained_variance_mc": explained_mc, "args": {k: str(v) for k, v in vars(a).items()},
                        "minutes": round((time.monotonic() - t0) / 60, 1)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(a.out)
    print(f"saved {a.out} after {out['learn_rl']['minutes']} min", flush=True)


def sil_group(net: ToolsNet, c: dict, group, state, new_h, new_c, n_streams: int, a, params: list, total_valid: float,
              total_sup: float):
    """One micro-batch of the self-imitation pass (the loss of self_imitation), for the worker processes."""
    L = a.length
    sub, b, act, _ = micro_view(c, group, n_streams, L, not getattr(a, "no_trim_batches", False))
    st = (state[0][:, group].contiguous(), state[1][:, group].contiguous())
    h, m, z, (hs, cs) = net.core(b, st, L, sub["starts"], return_state=True)
    new_h[:, group], new_c[:, group] = hs.detach(), cs.detach()
    _, ent, _ = net.score(h, m, z, b, act, temperature=a.temperature)
    value = net.value_of(z, b)
    w = sub["valid"] / total_valid
    known = ent["unit_known"]
    nll = (-ent["unit_logp"]).clamp(max=10.0) * known
    pg = ((nll.sum(1) / known.sum(1).clamp(min=1.0)) * sub["adv"] * w).sum()
    gap = (sub["ret"] - value).clamp(min=0.0)
    vl = 0.5 * ((gap ** 2) * w).sum()
    loss = a.sil_coef * pg + a.sil_value_coef * a.vf_coef * vl
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return grads, [("sil_pg", pg.detach()), ("sil_v", vl.detach())]


def self_imitation(net: ToolsNet, opt, a, device, ret_std: float, adv_std: float) -> dict | None:
    """The success archive's games (--sil): their decisions where the discounted return beat the value now,
    imitated in proportion to by how much (the main advantages' scale), and the critic pulled up to those returns.
    Positive advantage selects successful examples; it does not guarantee improvement on unselected games.
    The final policy probe measures movement on the same complete games before and after this pass."""
    check_stop(a)
    # Adam still moves parameters from its PPO momentum with zero new gradients.
    # A zero-coefficient ablation must skip the pass, not take zero-loss steps.
    if a.sil_coef == 0 and a.sil_value_coef == 0:
        return None
    sources = {tuple(x) for x in json.loads(a.sil.read_text())}
    if not sources:
        return None
    games = Games([Path(c) for c in a.cache], sources=sources)
    games.units = True
    games.air_refueling = net.air_refueling
    games.queued_factory_upgrades = net.queued_factory_upgrades
    if a.opponent_critic:          # the same critic as the main pass (it was the base head alone here)
        games.opponent = {i: opponent_features(opponent_of_recording(g["_cache_manifest"].get("source_name", "")))
                          for i, g in enumerate(games.games)}
    usable = [i for i, g in enumerate(games.games)
              if "pol_has" in g and bool(np.asarray(g["pol_has"]).any()) and all(f"rew_{c}" in g for c in REWARD_COMPONENTS)
              and (not needs_shaping_arrays(a) or all(k in g for k in SHAPING_ARRAYS))]
    if not usable:
        return None
    lengths = {i: int(games.lengths[i]) for i in usable}
    processes = int(a.processes or 0) if device.type == "cpu" else 0
    if processes > 1:
        from .learn_parallel import first_pass
        tokens_of = {i: int(games.games[i]["offsets_t"][lengths[i]]) for i in usable}
        values = first_pass(net, net, games, usable, lengths, tokens_of, a, processes, False, False, chunk_batch,
                            micro_view, CHUNK_THREADS(a))[0]
        values = {gi: np.asarray(v) for gi, v in values.items()}
    else:
        values = value_pass(net, games, usable, lengths, a, device)
    ret, adv = {}, {}
    for gi in usable:
        r = rewards_of(games.games[gi], a)[: lengths[gi]] / ret_std
        ret[gi] = discounted(r, a.gamma).astype(np.float32)
        # capped: one decision of an old game may not outweigh the PPO advantages (smoke test: mean 3 std units)
        adv[gi] = np.clip((ret[gi] - values[gi]) / max(adv_std, 1e-6), 0.0, a.sil_max_adv).astype(np.float32)
    pos = np.concatenate([adv[gi] for gi in usable])
    stats = {"sil_games": len(usable), "sil_positive": round(float((pos > 0).mean()), 3),
             "sil_adv_mean": round(float(pos.mean()), 4), "sil_pg": 0.0, "sil_v": 0.0, "sil_chunks": 0}
    net.train()
    zero = {gi: np.zeros(lengths[gi], np.float32) for gi in usable}
    if processes > 1:
        from .learn_parallel import EpochPool
        for epoch in range(a.sil_epochs):
            rng = np.random.default_rng(4000 + epoch)
            order = [int(i) for i in rng.permutation(usable)]
            streams = Streams(lengths, a.streams, a.length, order, balanced_queues(order, lengths, a.streams, rng))
            pool = EpochPool(net, games, usable, lengths, a, processes, CHUNK_THREADS(a), chunk_batch, sil_group,
                             dict(values=zero, adv=adv, ret=ret, adv_up=None, loc=None, kl0=None, prox=None, refp=None),
                             kl_stop=False)
            try:
                sums, _ = pool.run_epoch(streams.queues, opt)
            finally:
                pool.close()
            stats["sil_pg"] += sums.get("sil_pg", 0.0)
            stats["sil_v"] += sums.get("sil_v", 0.0)
            stats["sil_chunks"] += int(sums.get("n", 0))
        n = max(1, stats["sil_chunks"])
        stats["sil_pg"], stats["sil_v"] = round(stats["sil_pg"] / n, 5), round(stats["sil_v"] / n, 5)
        return stats
    for epoch in range(a.sil_epochs):
        rng = np.random.default_rng(4000 + epoch)
        order = [int(i) for i in rng.permutation(usable)]
        streams = Streams(lengths, a.streams, a.length, order, balanced_queues(order, lengths, a.streams, rng))
        state_h = net.initial_state(a.streams, device)
        for row in streams.rounds():
            check_stop(a)
            c = chunk_batch(games, row, a.length, device, values=zero, adv=adv, ret=ret, adv_up=None)
            n_streams, L = state_h[0].shape[1], a.length
            new_h, new_c = state_h[0].clone(), state_h[1].clone()
            total_valid = float(c["valid"].sum()) or 1.0
            opt.zero_grad(set_to_none=True)
            for s0 in range(0, n_streams, a.micro):
                s1 = min(n_streams, s0 + a.micro)
                idx = slice(s0 * L, s1 * L)
                sub = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == n_streams * L else v) for k, v in c.items()
                       if k not in ("b", "act", "lab")}
                b = {k: v[idx] for k, v in c["b"].items()}
                act = {k: v[idx] for k, v in c["act"].items()}
                st = (state_h[0][:, s0:s1].contiguous(), state_h[1][:, s0:s1].contiguous())
                h, m, z, (hs, cs) = net.core(b, st, L, sub["starts"], return_state=True)
                new_h[:, s0:s1], new_c[:, s0:s1] = hs.detach(), cs.detach()
                _, ent, _ = net.score(h, m, z, b, act, temperature=a.temperature)
                value = net.value_of(z, b)
                w = sub["valid"] / total_valid
                known = ent["unit_known"]
                nll = (-ent["unit_logp"]).clamp(max=10.0) * known                 # a very unlikely unit: capped
                pg = ((nll.sum(1) / known.sum(1).clamp(min=1.0)) * sub["adv"] * w).sum()
                gap = (sub["ret"] - value).clamp(min=0.0)
                vl = 0.5 * ((gap ** 2) * w).sum()
                loss = a.sil_coef * pg + a.sil_value_coef * a.vf_coef * vl
                loss.backward()
                stats["sil_pg"] += float(pg.detach())
                stats["sil_v"] += float(vl.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad)
            opt.step()
            opt.zero_grad(set_to_none=True)
            state_h = (new_h, new_c)
            stats["sil_chunks"] += 1
    n = max(1, stats["sil_chunks"])
    stats["sil_pg"], stats["sil_v"] = round(stats["sil_pg"] / n, 5), round(stats["sil_v"] / n, 5)
    return stats


def check_stop(a) -> None:
    path = getattr(a, "stop_file", None)
    if path and path.exists():
        raise SystemExit(f"{path} found: stopping without saving")


def builder_distillation_labels(labels: dict, batch: dict, weight: float = 1.0) -> dict:
    """Weight a builder's entire teacher lesson, including END, without changing actions.

    Match the existing BC builder definition (own units with a build menu).
    Weighting only positive orders would artificially increase their learned rate.
    The default returns the original labels and preserves the legacy objective.
    """
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("distill builder weight must be finite and positive")
    if weight == 1.0:
        return labels
    builder = batch["tokens"][..., 67] > 0
    weights = 1.0 + (weight - 1.0) * builder.float()
    if "uweight" in labels:
        weights = weights * labels["uweight"]
    return dict(labels, uweight=weights)


def idle_factory_distillation_labels(labels: dict, batch: dict, weight: float = 1.0) -> dict:
    """Opt-in action-rate ablation: weaken idle-factory END advice, not execution.

    The shadow teacher often also waits in these states. Downweighting its END
    changes the learned rate deliberately; it is not a faithful imitation fix.
    Only existing player-visible observations are used. Scarcity, queued work,
    unit-cap pressure, positive orders and unknown labels retain their weights.
    """
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("idle factory END weight must be finite and in [0, 1]")
    if weight == 1.0:
        return labels
    tok, scalars = batch['tokens'], batch['scalars']
    # ToolsObserver: mass stored/max = scalars[1:3], energy = [6:8], cap use = [13].
    resources = ((scalars[:, 2] > 0) & (scalars[:, 7] > 0)
                 & (scalars[:, 1] >= .25 * scalars[:, 2])
                 & (scalars[:, 6] >= .25 * scalars[:, 7]) & (scalars[:, 13] < .95))
    # Produce eligibility includes the legal mobile build menu. Own token idle
    # and queue features are columns 15 and 17; enemy and padding never qualify.
    selected = (batch['mask_subject'][:, VERBS.index('produce')] & batch['token_mask']
                & (batch['token_type'] == 0) & (tok[..., 15] > .5) & (tok[..., 17] == 0)
                & resources[:, None] & (labels['uverb'] == 0))
    weights = torch.where(selected, weight, 1.0)
    if 'uweight' in labels:
        weights = weights * labels['uweight']
    return dict(labels, uweight=weights)


def select_probe_games(games, usable: list[int], limit: int) -> list[int]:
    """Round-robin across opponent/faction/map strata, newest first within a stratum.
    These are diagnostic training games, never an independent performance test.
    """
    if limit <= 0:
        return []
    bins = {}
    for gi in sorted(usable, key=lambda i: recency_key(games.games[i]["_cache_manifest"]), reverse=True):
        g = games.games[gi]
        meta = _meta(g)
        key = (opponent_of_recording(g["_cache_manifest"].get("source_name", "")),
               str(meta.get("faction", "?")), str(meta.get("map", "?")))
        bins.setdefault(key, []).append(gi)
    picked = []
    while bins and len(picked) < limit:
        for key in list(bins):
            picked.append(bins[key].pop(0))
            if not bins[key]:
                del bins[key]
            if len(picked) >= limit:
                break
    return picked


def policy_probe(net, games, ids, lengths, a, device, prox, refp=None, before=None):
    """Replay complete games from initial recurrent state. Sample-action k3
    statistics are diagnostic surrogates, not exact KL or a win-rate prediction.
    Return token scores/modes to isolate the movement caused by SIL on the same data.
    """
    streams_n = min(a.micro, len(ids))
    streams = Streams(lengths, streams_n, a.length, ids)
    state = net.initial_state(streams_n, device)
    payload = {i: {"logp": np.zeros(int(games.games[i]["offsets_t"][lengths[i]]), np.float32),
                   "mode": np.zeros(int(games.games[i]["offsets_t"][lengths[i]]), np.int16)} for i in ids}
    sums = {k: 0.0 for k in ("sample_k3_to_init", "sample_k3_to_ref", "sample_k3_from_sil",
                              "verb_mode_changed_by_sil", "verb_removed_mass", "verb_below_floor")}
    commander_sums = dict.fromkeys(sums, 0.0)
    commander_units = 0.0
    decisions, units, commander_decisions = 0, 0.0, 0.0
    commander_probability = np.zeros(len(VERBS), np.float64)
    commander_removed = np.zeros(len(VERBS), np.float64)
    training, audit = net.training, net.audit_probabilities
    net.eval()
    net.audit_probabilities = True
    try:
        with torch.no_grad():
            for row in streams.rounds():
                check_stop(a)
                c = chunk_batch(games, row, a.length, device, prox=prox, refp=refp)
                h, m, z, state = net.core(c["b"], state, a.length, c["starts"], return_state=True)
                _, ent, _ = net.score(h, m, z, c["b"], c["act"], temperature=a.temperature)
                weight = ent["unit_known"] * c["valid"].unsqueeze(1)
                # The army-wide average can hide a large change in the one
                # unit whose death loses the game. Use the same valid/scored
                # action mask; padding and unknown commander actions do not
                # contribute to this diagnostic's denominator.
                commander_weight = weight * c["b"]["mask_subject"][:, VERBS.index("commander_move")]
                commander_units += float(commander_weight.sum())
                lp = ent["unit_logp"]
                units += float(weight.sum())
                decisions += int(c["valid"].sum())
                commander_decisions += float((ent['commander_count'] * c['valid']).sum())
                commander_probability += (ent['commander_verb_probability'] * c['valid'].unsqueeze(1)).sum(0).cpu().numpy()
                commander_removed += (ent['commander_verb_removed_probability'] * c['valid'].unsqueeze(1)).sum(0).cpu().numpy()
                for key, other in (("sample_k3_to_init", c["prox_ulp"]), ("sample_k3_to_ref", c["ref_ulp"])):
                    if key.endswith("ref") and refp is None:
                        continue
                    delta = lp - other
                    sums[key] += float(((torch.expm1(delta) - delta) * weight).sum())
                    commander_sums[key] += float(((torch.expm1(delta) - delta) * commander_weight).sum())
                for key in ("verb_removed_mass", "verb_below_floor"):
                    sums[key] += float((ent[key] * weight).sum())
                    commander_sums[key] += float((ent[key] * commander_weight).sum())
                arr, modes, weights = lp.cpu().numpy(), ent["verb_mode"].cpu().numpy(), weight.cpu().numpy()
                commander_weights = commander_weight.cpu().numpy()
                for s, item in enumerate(row):
                    if item is None:
                        continue
                    gi, start = item
                    off = games.games[gi]["offsets_t"]
                    for t in range(min(a.length, lengths[gi] - start)):
                        o, e = int(off[start + t]), int(off[start + t + 1])
                        j, n = s * a.length + t, e - o
                        payload[gi]["logp"][o:e] = arr[j, :n]
                        payload[gi]["mode"][o:e] = modes[j, :n]
                        if before is not None:
                            d = arr[j, :n] - before[gi]["logp"][o:e]
                            sums["sample_k3_from_sil"] += float(((np.expm1(d) - d) * weights[j, :n]).sum())
                            sums["verb_mode_changed_by_sil"] += float(((modes[j, :n] != before[gi]["mode"][o:e])
                                                                       * weights[j, :n]).sum())
                            commander_sums["sample_k3_from_sil"] += float(
                                ((np.expm1(d) - d) * commander_weights[j, :n]).sum())
                            commander_sums["verb_mode_changed_by_sil"] += float(
                                ((modes[j, :n] != before[gi]["mode"][o:e]) * commander_weights[j, :n]).sum())
    finally:
        net.audit_probabilities = audit
        net.train(training)
    stats = {key: round(value / max(units, 1.0), 7) for key, value in sums.items()
             if (before is not None or key not in ("sample_k3_from_sil", "verb_mode_changed_by_sil"))
             and (refp is not None or key != "sample_k3_to_ref")}
    if commander_units:
        stats.update({"commander_" + key: round(commander_sums[key] / commander_units, 7)
                      for key in stats})
    stats['commander_scored_units'] = int(commander_units)
    stats.update(games=len(ids), decisions=decisions, units=int(units),
                 sources=[games.games[i]["_cache_manifest"].get("source_name") for i in ids])
    stats['commander_decisions'] = int(commander_decisions)
    stats['commander_verbs'] = {verb: dict(raw_probability=round(float(p / max(1, commander_decisions)), 7),
                                          removed_probability=round(float(r / max(1, commander_decisions)), 7))
                                for verb, p, r in zip(VERBS, commander_probability, commander_removed) if p or r}
    return stats, payload


def keep_features(features: dict, row: list, z, privileged, lengths: dict, a) -> None:
    """The memory and privileged inputs of a round of chunks, by game: a game's memory is reset at its start and
    carried chunk to chunk, so these are what any later pass with the same trunk would compute."""
    z = z.view(a.streams, a.length, -1).float().cpu().numpy()
    privileged = privileged.view(a.streams, a.length, -1).float().cpu().numpy()
    for s, item in enumerate(row):
        if item is None:
            continue
        gi, start = item
        if gi not in features:
            features[gi] = (np.zeros((lengths[gi], z.shape[-1]), np.float32),
                            np.zeros((lengths[gi], privileged.shape[-1]), np.float32))
        n = min(a.length, lengths[gi] - start)
        features[gi][0][start:start + n] = z[s, :n]
        features[gi][1][start:start + n] = privileged[s, :n]


def _critic_batch(games: Games, gi: int, zs, ps, device) -> dict:
    b = {"privileged": torch.as_tensor(ps, device=device)}
    opponents = getattr(games, "opponent", None)
    if opponents is not None:
        b["opponent"] = torch.as_tensor(np.broadcast_to(opponents[gi], (len(zs), len(opponents[gi]))).copy(),
                                        dtype=torch.float32, device=device)
    return b


def values_from_features(net: ToolsNet, games: Games, usable: list, features: dict, device, batch: int = 8192) -> dict:
    """Every decision's value with the critic as it is now and the trunk's kept output."""
    net.eval()
    values = {}
    with torch.no_grad():
        for gi in usable:
            zs, ps = features[gi]
            out = []
            for i in range(0, len(zs), batch):
                z = torch.as_tensor(zs[i:i + batch], device=device)
                out.append(net.value_of(z, _critic_batch(games, gi, zs[i:i + batch], ps[i:i + batch], device)).cpu().numpy())
            values[gi] = np.concatenate(out).astype(np.float32)
    return values


def value_warmup_on_features(net: ToolsNet, games: Games, usable: list, features: dict, ret: dict, a, device,
                             epochs: int) -> None:
    """value_warmup without running the trunk again: the value head on the kept memory, in shuffled batches the size
    of a round of chunks, as many steps per epoch as a round-based epoch has (2026-11-07: a trunk pass is expensive on
    the CPU, and this fit needs none)."""
    head = list(net.value_head.parameters()) + list(net.opp_value.parameters())
    lr = a.value_lr if a.value_lr is not None else a.lr
    opt = torch.optim.Adam(head, lr=lr)
    print(f"value warm-up (kept features) learning rate {lr:g}", flush=True)
    rows = []
    for gi in usable:
        has = np.asarray(games.games[gi]["pol_has"][:len(features[gi][0])]).astype(bool)
        rows.append(np.stack([np.full(int(has.sum()), gi), np.flatnonzero(has)], 1))
    rows = np.concatenate(rows)
    if not len(rows):
        raise ValueError("Critic fitting requires recorded decisions")
    size = int(getattr(a, "chunks_per_step", 0) or a.streams) * a.length
    for epoch in range(epochs):
        rng = np.random.default_rng(1000 + epoch)
        order = rng.permutation(len(rows))
        total, n = 0.0, 0
        for i in range(0, len(order), size):
            check_stop(a)
            part = rows[order[i:i + size]]
            by_game: dict = {}
            for gi, t in part:
                by_game.setdefault(int(gi), []).append(int(t))
            feature_rows, input_rows, targets = [], [], []
            for gi, ts in by_game.items():
                zs, ps = features[gi][0][ts], features[gi][1][ts]
                feature_rows.append(torch.as_tensor(zs, device=device))
                input_rows.append(_critic_batch(games, gi, zs, ps, device))
                targets.append(torch.as_tensor(ret[gi][ts], dtype=torch.float32, device=device))
            batch = {key: torch.cat([row[key] for row in input_rows]) for key in input_rows[0]}
            value = net.value_of(torch.cat(feature_rows), batch)
            loss = 0.5 * ((value - torch.cat(targets)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + float(loss.detach()), n + 1
        print(f"value warm-up epoch {epoch + 1}: loss {total / max(1, n):.5f} over {n} batches{at()}", flush=True)


def value_pass(net: ToolsNet, games: Games, usable: list, lengths: dict, a, device) -> dict:
    """Every decision's value with the network as it is (no gradients), memory carried chunk to chunk."""
    net.eval()
    values = {i: np.zeros(lengths[i], np.float32) for i in usable}
    streams = Streams(lengths, a.streams, a.length, sorted(usable, key=lambda i: -lengths[i]))
    state_h = net.initial_state(a.streams, device)
    with torch.no_grad():
        for row in streams.rounds():
            check_stop(a)
            c = chunk_batch(games, row, a.length, device)
            v = torch.zeros(a.streams * a.length)
            for group in micro_groups(c, a.streams, a.length, a.micro, not getattr(a, "no_trim_batches", False)):
                sub, b_g, _, _ = micro_view(c, group, a.streams, a.length, not getattr(a, "no_trim_batches", False))
                rows_g = (group.cpu().unsqueeze(1) * a.length + torch.arange(a.length)).view(-1)
                st = (state_h[0][:, group].contiguous(), state_h[1][:, group].contiguous())
                _, _, z, (hs, cs) = net.core(b_g, st, a.length, sub["starts"], return_state=True)
                state_h[0][:, group], state_h[1][:, group] = hs, cs
                v[rows_g] = net.value_of(z, b_g).float().cpu()
            v = v.view(a.streams, a.length).numpy()
            for s_, item in enumerate(row):
                if item is not None:
                    gi, start = item
                    n = min(a.length, lengths[gi] - start)
                    values[gi][start:start + n] = v[s_, :n]
    return values


def critic_bias(values: dict, mc: dict, usable: list, games, when: str) -> None:
    """mean(V - Monte-Carlo return) overall, at the first decision, and per opponent including multiplier. Explained variance
    ignores a constant bias and is measured against targets built from the values themselves: rl_089's critic was
    +1.54 critic units (~13.6 reward units) too optimistic and no logged number showed it (2026-10-17 review)."""
    d = np.concatenate([values[gi] - mc[gi] for gi in usable])
    first = {}
    for gi in usable:
        kind = opponent_of_recording(games.games[gi]["_cache_manifest"].get("source_name", "")) or "?"
        first.setdefault(kind, []).append(float(values[gi][0] - mc[gi][0]))
    print(f"critic bias {when}: mean(V - MC) {float(d.mean()):+.3f}, at the first decision "
          + ", ".join(f"{k} {np.mean(v):+.2f}" for k, v in sorted(first.items())), flush=True)


def value_warmup(net: ToolsNet, games: Games, usable: list, lengths: dict, ret: dict, a, device,
                 epochs: int | None = None) -> None:
    """The value head alone fitted to the new returns for --value-warmup epochs (or `epochs`): the memory and the
    policy stay as they are (the trunk runs without gradients), so the first iteration after a reward change does
    not compute its advantages with a critic of the old reward."""
    head = list(net.value_head.parameters()) + list(net.opp_value.parameters())
    # Honor the same critic rate used during PPO; a hidden 3e-4 warm-up
    # previously overrode experiments explicitly requesting smaller updates.
    lr = a.value_lr if a.value_lr is not None else a.lr
    opt = torch.optim.Adam(head, lr=lr)
    print(f"value warm-up learning rate {lr:g}", flush=True)
    for epoch in range(a.value_warmup if epochs is None else epochs):
        rng = np.random.default_rng(1000 + epoch)
        order = [int(i) for i in rng.permutation(usable)]
        streams = Streams(lengths, a.streams, a.length, order, balanced_queues(order, lengths, a.streams, rng))
        state_h = net.initial_state(a.streams, device)
        total, n = 0.0, 0
        for row in streams.rounds():
            check_stop(a)
            c = chunk_batch(games, row, a.length, device, ret=ret, values={i: ret[i] * 0 for i in usable},
                            adv={i: ret[i] * 0 for i in usable}, adv_up=None)
            with torch.no_grad():
                _, _, z, state_h = net.core(c["b"], state_h, a.length, c["starts"], return_state=True)
            value = net.value_of(z, c["b"])
            w = c["valid"] / float(c["valid"].sum() or 1.0)
            loss = 0.5 * (((value - c["ret"]) ** 2) * w).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + float(loss.detach()), n + 1
        print(f"value warm-up epoch {epoch + 1}: loss {total / max(1, n):.5f} over {n} chunks{at()}", flush=True)


def CHUNK_THREADS(a) -> int:
    return int(getattr(a, "cpu_threads", None) or 1)


_GROUP_POOL: dict = {}


def map_groups(fn, groups: list, a) -> list:
    """fn(group) for the groups of a round, side by side on --parallel-groups threads. One thread team of 28 was no
    faster than 14 on the server (2026-11-07): most tensors are too small to split well, and whole micro-batches are
    a far better unit of work. Each worker has its own OpenMP team of --cpu-threads threads."""
    workers = int(getattr(a, "parallel_groups", 1) or 1)
    if workers <= 1 or len(groups) <= 1:
        return [fn(g) for g in groups]
    pool = _GROUP_POOL.get(workers)
    if pool is None:
        threads = torch.get_num_threads()
        pool = _GROUP_POOL[workers] = concurrent.futures.ThreadPoolExecutor(
            workers, initializer=torch.set_num_threads, initargs=(threads,))
    grad = torch.is_grad_enabled()      # thread-local: a worker does not inherit the caller's no_grad

    def call(group):
        with torch.set_grad_enabled(grad):
            return fn(group)
    return list(pool.map(call, groups))


def prefetched(rows, fn, ahead: int = 2):
    """fn(row) for each row, computed ahead on one background thread (assembling a chunk is mostly numpy)."""
    pending: collections.deque = collections.deque()
    rows = iter(rows)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        for row in rows:
            pending.append((row, pool.submit(fn, row)))
            if len(pending) > ahead:
                done_row, future = pending.popleft()
                yield done_row, future.result()
        while pending:
            done_row, future = pending.popleft()
            yield done_row, future.result()


def micro_groups(c: dict, n_streams: int, length: int, micro: int, trim: bool = True) -> list:
    """The streams of a round in groups of `micro`, similar sizes together: a group's tensors are cut to its own
    longest lists (micro_view), and a round's padding cost about 1.9x on linear work and 2.7x on attention
    (2026-10-26 audit of real caches; a padded token costs the same as a real one)."""
    if not trim:
        return list(torch.arange(n_streams, device=c["b"]["tokens"].device).split(micro))
    tokens = c["b"]["token_mask"].sum(1).view(n_streams, length).max(1).values
    order = torch.argsort(tokens, descending=True, stable=True)
    return [order[i:i + micro].sort().values for i in range(0, n_streams, micro)]


def micro_view(c: dict, streams, n_streams: int, length: int, trim: bool = True) -> tuple[dict, dict, dict, dict]:
    """(sub, b, act, lab) of these streams, every tensor cut to the group's longest token, candidate and enhancement
    lists. Cutting removes only padding, so what is computed for the real rows does not change."""
    rows = (streams.unsqueeze(1) * length + torch.arange(length, device=streams.device)).view(-1)
    b_all = c["b"]
    def extent(mask):
        if not trim:
            return mask.shape[1]
        # Last present row also supports sparse masks, not just packed prefixes.
        positions = torch.arange(1, mask.shape[1] + 1, device=mask.device)
        return max(1, int((mask[rows] * positions).max()))
    T, C, E = (extent(b_all[key]) for key in ("token_mask", "cand_mask", "enh_mask"))
    b = {}
    for k, v in b_all.items():
        v = v[rows]
        if k in _TOKEN_FIRST:
            v = v[:, :T]
        elif k in _TOKEN_SECOND:
            v = v[:, :, :T]
        elif k in _CAND_FIRST:
            v = v[:, :C]
        elif k in _ENH_FIRST:
            v = v[:, :E]
        elif k == "can_build":
            v = v[:, :T, :C]
        elif k == "can_enh":
            v = v[:, :T, :E]
        b[k] = v
    act = {k: v[rows][:, :T] for k, v in c["act"].items()}
    lab = {k: v[rows][:, :T] for k, v in c["lab"].items()}
    N = n_streams * length
    sub = {}
    for k, v in c.items():
        if k in ("b", "act", "lab"):
            continue
        if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == N:
            v = v[rows]
            if v.dim() == 2 and k in _UNIT_ROWS:
                v = v[:, :T]
        sub[k] = v
    return sub, b, act, lab


def ppo_group(net: ToolsNet, c: dict, group, state, new_h, new_c, n_streams: int, a, params: list, total_valid: float,
              total_sup: float):
    """One micro-batch of a round: its gradients (not accumulated: `autograd.grad`) and statistics. `total_valid` and
    `total_sup` are the whole round's, so the sum over a round's groups is the round's gradient however they ran.
    The memory after the chunk goes to new_h, new_c."""
    L = a.length
    sub, b, act, lab = micro_view(c, group, n_streams, L, not getattr(a, "no_trim_batches", False))
    st = (state[0][:, group].contiguous(), state[1][:, group].contiguous())
    h, m, z, (hs, cs) = net.core(b, st, L, sub["starts"], return_state=True)
    new_h[:, group], new_c[:, group] = hs.detach(), cs.detach()
    if a.distill > 0:
        if not a.no_undo_labels:
            lab = dict(lab, uverb=torch.where((act["uverb"] > 0) & (lab["uverb"] == 0),
                                              torch.full_like(lab["uverb"], -1), lab["uverb"]))
        lab = builder_distillation_labels(lab, b, getattr(a, "distill_builder_weight", 1.0))
        lab = idle_factory_distillation_labels(lab, b, getattr(a, "distill_idle_factory_end_weight", 1.0))
        if getattr(a, "separate_decoder", False):
            lp, ent, _ = net.score(h, m, z, b, act, temperature=a.temperature)
            _, _, bc = net.score(h, m, z, b, lab, smoothing=0.05)
        else:
            lp, ent, bc = net.score_pair(h, m, z, b, act, lab, temperature=a.temperature)
    else:
        lp, ent, _ = net.score(h, m, z, b, act, temperature=a.temperature)
    value = net.value_of(z, b)
    w = sub["valid"] / total_valid
    ulp, known = ent["unit_logp"], ent["unit_known"]
    n_units = known.sum(1).clamp(min=1.0)
    per_unit = lambda x: ((x * known).sum(1) / n_units * w).sum()  # noqa: E731
    u_log_ratio = (ulp - sub["prox_ulp"]) * known           # against the network at the iteration's start
    u_ratio = u_log_ratio.exp()
    adv = sub["adv"].unsqueeze(1)
    local_loss = torch.zeros((), device=u_ratio.device)
    if a.local_credit > 0 and "loc" in sub and float(sub["loc_has"].sum()) > 0:
        # spatial credit: each unit's advantage gets the local one of its cell -- the coming local trade there
        # against what the local critic expected -- on top of the decision's
        from fa_env.tools import GRID
        lv = net.local_value(m).flatten(1)                                          # [n, G*G]
        has = sub["loc_has"].unsqueeze(1)
        local_loss = (((lv - sub["loc"]) ** 2).mean(1) * sub["loc_has"] * w).sum()
        tok = b["tokens"]
        cell = ((tok[..., 8] * GRID).long().clamp(0, GRID - 1) * GRID
                + (tok[..., 7] * GRID).long().clamp(0, GRID - 1))                  # [n, T]
        local_adv = (sub["loc"] - lv.detach()).gather(1, cell) * has                  # [n, T]
        adv = adv + a.local_credit * local_adv
        loss_extra = a.local_coef * local_loss
    else:
        loss_extra = 0.0
    pg = per_unit(torch.max(-adv * u_ratio, -adv * u_ratio.clamp(1 - a.clip, 1 + a.clip)))
    if a.upgo > 0:
        up = sub["adv_up"].unsqueeze(1)
        pg = pg + a.upgo * per_unit(torch.max(-up * u_ratio, -up * u_ratio.clamp(1 - a.clip, 1 + a.clip)))
    # this iteration's movement, KL(start || now) estimated on the games' own actions: each unit weighted by
    # min(1, start / games' policy). Unweighted, an action the start network all but ruled out (picked by an
    # older network) gave a ratio of e^15 and one chunk an average KL of 1.1 (iteration 40, kl_data 0.03)
    with torch.no_grad():
        iw = ((sub["prox_ulp"] - sub["old_ulp"]) * known).clamp(max=0.0).exp()
    kl = per_unit(iw * ((u_ratio - 1) - u_log_ratio))
    with torch.no_grad():
        d_data = (ulp - sub["old_ulp"]) * known
        kl_data = per_unit((d_data.exp() - 1) - d_data)                  # to the games' own policy (lag + movement)
    clipfrac = per_unit(((u_ratio - 1).abs() > a.clip).float())
    vc = a.clip if a.value_clip is None else a.value_clip
    if vc > 0:
        v_clip = sub["v_old"] + (value - sub["v_old"]).clamp(-vc, vc)
        v_loss = 0.5 * (torch.max((value - sub["ret"]) ** 2, (v_clip - sub["ret"]) ** 2) * w).sum()
    else:                   # no value clip (Andrychowicz et al. 2021: it does not help and can hurt)
        v_loss = 0.5 * (((value - sub["ret"]) ** 2) * w).sum()
    # per unit, like the policy term: ent[k] is a decision's sum over its units. Summed, the bonus was ~100x the
    # policy term and flattened the policy (rl1, 2026-10-10: entropy per decision 43 -> 116 in five iterations,
    # win rate 80 % -> 64 %).
    entropy = sum(ent[k] for k in ("verb", "blueprint", "cell", "target")) / n_units * w
    loss = pg + a.vf_coef * v_loss - a.ent_coef * entropy.sum() + loss_extra
    ref_kl = torch.zeros((), device=pg.device)
    if a.ref_coef > 0:
        # k3 estimator of KL(policy || reference), capped: one unit the policy has all but ruled out would
        # otherwise dominate (exp(d) - d - 1 is ~e^10 at d = 10). The reference's log-probabilities come from
        # the first pass (it is frozen)
        d = ((sub["ref_ulp"] - ulp) * known).clamp(max=5.0)
        ref_kl = per_unit(d.exp() - d - 1)
        loss = loss + a.ref_coef * ref_kl
    distill = torch.zeros((), device=pg.device)
    if a.distill > 0:
        sup = sub["sup"] * (sub["valid"] > 0).float()
        # per unit, like the policy term: bc is a decision's sum over its units (~100 units: ~100x the scale)
        distill = (bc / n_units * sup * sub["supw"]).sum() / total_sup
        loss = loss + a.distill * distill
    aux_loss = torch.zeros((), device=pg.device)
    if getattr(a, "aux_econ", 0.0) > 0 and "aux_y" in sub:
        err = ((net.aux_econ(z) - sub["aux_y"]) ** 2) * sub["aux_m"]
        aux_loss = (err.sum(1) / sub["aux_m"].sum(1).clamp(min=1.0) * w).sum()
        loss = loss + a.aux_econ * aux_loss
    # autograd.grad, not backward: backward from several threads races on the shared .grad
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return grads, [(key, val.detach()) for key, val in (
        ("pg", pg), ("v", v_loss), ("ent", entropy.sum()), ("kl", kl), ("clip", clipfrac),
        ("ref_kl", ref_kl), ("distill", distill), ("local", local_loss), ("kl_data", kl_data), ("aux", aux_loss))]


def ppo_step(net: ToolsNet, ref: ToolsNet, opt, c: dict, state, ref_state, a):
    """One optimizer step on a round of chunks, in micro-batches of streams (gradients accumulated). Returns the
    statistics and both networks' memories after the chunks (detached, to carry into the next round)."""
    new_h, new_c = state[0].clone(), state[1].clone()
    ref_h, ref_c = ref_state[0].clone(), ref_state[1].clone()
    n_streams = state[0].shape[1]
    L = a.length
    total_valid = float(c["valid"].sum()) or 1.0
    opt.zero_grad(set_to_none=True)
    sums = {k: 0.0 for k in ("pg", "v", "ent", "kl", "clip", "ref_kl", "distill", "local", "kl_data", "aux")}
    params = [q for q in net.parameters() if q.requires_grad]

    def one_group(group):
        return ppo_group(net, c, group, state, new_h, new_c, n_streams, a, params, total_valid,
                         max(1.0, float(c["sup"].sum())))

    groups = micro_groups(c, n_streams, L, a.micro, not getattr(a, "no_trim_batches", False))
    for grads, stats in map_groups(one_group, groups, a):
        for q, g in zip(params, grads):
            if g is not None:
                q.grad = g if q.grad is None else q.grad.add_(g)
        for key, val in stats:
            sums[key] += float(val)
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad)
    a.step = getattr(a, "step", 0) + 1
    for group in opt.param_groups:
        group["lr"] = group.get("base_lr", a.lr) * min(1.0, a.step / max(1, a.warmup))
    if torch.isfinite(norm):
        opt.step()
    opt.zero_grad(set_to_none=True)
    sums["gnorm"] = float(norm) if torch.isfinite(norm) else 0.0
    sums["n"] = 1.0
    return sums, (new_h, new_c), (ref_h, ref_c)


if __name__ == "__main__":
    main()
