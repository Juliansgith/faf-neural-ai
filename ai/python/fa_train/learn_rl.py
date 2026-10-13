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
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .tools_bc import REWARD_COMPONENTS, Games
from .tools_model import (UNIT_KEYS, ToolsNet, apply_model_options, collate, model_options, parameter_count,
                          stack_actions)


def load_net(path: Path, device) -> tuple[ToolsNet, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("decoder", "slots") != "units":
        raise ValueError("learn_rl needs a units-decoder network")
    net = ToolsNet(state.get("size", "M"), decoder="units").to(device)
    net.load_state_dict(state["model"])
    apply_model_options(net, state)
    return net, state


SHAPING_ARRAYS = ("phi_intel", "phi_map", "waste", "mass_capacity", "reclaimed_mass", "econ_trade")


def potential(phi: np.ndarray, gamma: float) -> np.ndarray:
    """gamma * phi(next) - phi(now) per decision (the last decision's next state is its own: nothing after it)."""
    return gamma * np.append(phi[1:], phi[-1]) - phi


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


def reward_spec(a) -> str:
    """What the reward is made of (stored in each checkpoint: a change is noticed by the next iteration)."""
    return (f"econ={a.econ_source};kills={a.econ_coef:g};reclaim={a.reclaim_coef:g};intel={a.intel_coef:g};"
            f"map={a.map_coef:g};waste={a.waste_coef:g};shaping={a.shaping:g}"
            + (f";stall={a.stall_coef:g};capacity=actual" if a.stall_coef else "")
            + (f";acu={a.acu_coef:g}/{a.acu_grace:g}/{a.acu_span:g}{'' if a.acu_all_maps else '/large'}"
               if a.acu_coef else ""))


LEGACY_SPEC = "econ=recorded;kills=0;reclaim=0.5;intel=0;map=0;waste=0;shaping=1"


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
    shaped = c["trade"] + c["commander"]
    if a.econ_source == "capacity":
        shaped = shaped + potential(np.asarray(g["mass_capacity"], np.float32)[:n] / 20.0, a.gamma)
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
    pad_obs = None
    for item in row:
        if item is None:
            continue
        pad_obs = pad_obs or games.step_obs(games.games[item[0]], 0)
    for item in row:
        for t in range(length):
            if item is None or item[1] + t >= games.lengths[item[0]]:
                g, step = (item[0], games.lengths[item[0]] - 1) if item is not None else (None, 0)
                obs.append(games.step_obs(games.games[g], step) if g is not None else pad_obs)
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
            acts.append({k: np.asarray(g[f"pol_{k}"][o:e], np.int64) for k in UNIT_KEYS})
            labels.append({k: np.asarray(g[k][o:e], np.int64) for k in UNIT_KEYS})
            valid.append(bool(g["pol_has"][step]))
            starts.append(step == 0)
            old_unit_lp.append(np.asarray(g["pol_unit_logp"][o:e], np.float32))
            prox_rows.append(prox[gi][o:e] if prox is not None else old_unit_lp[-1])
            ref_rows.append(refp[gi][o:e] if refp is not None else np.zeros(e - o, np.float32))
            old_lp.append(float(g["pol_logp"][step]))
            sup.append(bool(g["unit_supervise"][step]))
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
    return dict(b=b, act=act, lab=lab, valid=f(valid), starts=f(starts, torch.bool), old_ulp=f(ulp),
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
    ap.add_argument("--econ-coef", type=float, default=0.0,
                    help="weight of the economy part of the trade (mexes and engineers destroyed or lost, valued "
                         "beyond their cost: tools_bc.economy_bonus), on top of the recorded trade reward")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="entropy bonus per unit (none by default: sampling at temperature 0.5 explores already)")
    ap.add_argument("--ref-coef", type=float, default=0.2)
    ap.add_argument("--distill", type=float, default=0.5, help="weight of M28's labels (the recording's teacher)")
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
    ap.add_argument("--device")
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
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.backends.cuda.matmul.allow_tf32 = False           # exact ratios
    t0 = time.monotonic()
    net, state = load_net(a.init, device)
    net.grad_checkpoint = bool(a.grad_checkpoint)
    ref, ref_state = load_net(a.reference or a.init, device)
    # An anchor from before the fine head and the close-ups has none: it takes the network's own (as they were
    # trained when they joined), else its KL would price the fine pixels against nothing (a pull to certainty)
    joined = {k: v for k, v in net.state_dict().items()
              if k.startswith(("fine_", "focus_")) and k not in ref_state["model"]}
    if joined:
        ref.load_state_dict({**ref.state_dict(), **joined})
        ref.fine_positions = net.fine_positions
        print(f"reference: the network's fine head and close-ups ({len(joined)} tensors), it had none", flush=True)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    # only the newest games are loaded (with a margin for games without recorded choices): the cache keeps them all
    games = Games([Path(c) for c in a.cache], newest=a.max_games + 60 if a.max_games else None)
    games.units = True
    usable = [i for i, g in enumerate(games.games)
              if "pol_has" in g and bool(np.asarray(g["pol_has"]).any()) and all(f"rew_{c}" in g for c in REWARD_COMPONENTS)]
    # the large maps' weight, at least --distill-large-min whatever the adaptive weight is: it falls as the small
    # maps are won (0.05 on 2026-10-13, so x4 was 0.2), while the large maps are still lost and M28 plays them better
    large = max(a.distill_large, a.distill_large_min / max(a.distill, 1e-6)) if a.distill > 0 else a.distill_large
    games.distill_weight = {i: large for i in usable if map_km(games.games[i]) > LARGE_KM}
    if large != 1.0:
        print(f"imitation: {len(games.distill_weight)} games on maps above {LARGE_KM} km, M28's labels x{large:g} "
              f"({a.distill * large:.3f})", flush=True)
    if a.max_games:                                    # the newest games (by when they were encoded)
        usable = sorted(usable, key=lambda i: int(games.games[i]["_cache_manifest"].get("created_ns", 0)))[-a.max_games:]
    if not usable:
        raise SystemExit("no games with recorded choices and rewards")
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
    with torch.no_grad():
        for row in streams.rounds():
            c = chunk_batch(games, row, a.length, device)
            h, m, z, state_h = net.core(c["b"], state_h, a.length, c["starts"], return_state=True)
            v, _ = net.heads(z, c["b"])
            v = v.view(a.streams, a.length).cpu().numpy()
            lr_ = np.zeros((a.streams, a.length), np.float32)
            _, ent, _ = net.score(h, m, z, c["b"], c["act"], temperature=a.temperature)
            ulp_now = ent["unit_logp"].float().cpu().numpy()
            if refp is not None:
                # the frozen reference's log-probabilities, once here instead of a forward pass every PPO step
                hr, mr, zr, ref_h1 = ref.core(c["b"], ref_h1, a.length, c["starts"], return_state=True)
                _, ent_r, _ = ref.score(hr, mr, zr, c["b"], c["act"], temperature=a.temperature)
                ulp_ref = ent_r["unit_logp"].float().cpu().numpy()
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
                log_rho[gi][start:start + n] = lr_[s, :n]
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
    if warm:
        before = 1 - float(np.var(np.concatenate([ret[gi] - values[gi] for gi in usable]))
                           / max(np.var(np.concatenate([ret[gi] for gi in usable])), 1e-8))
        value_warmup(net, games, usable, lengths, ret, a, device)
        values = value_pass(net, games, usable, lengths, a, device)
        adv, ret, adv_up = advantages()
        after = 1 - float(np.var(np.concatenate([ret[gi] - values[gi] for gi in usable]))
                          / max(np.var(np.concatenate([ret[gi] for gi in usable])), 1e-8))
        print(f"value warm-up: explained variance {before:+.2f} -> {after:+.2f}", flush=True)
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
            print(f"local credit: {len(with_loc)} of {len(usable)} games carry local trade; {active.size} active "
                  f"cell-decisions, std {loc_std:.4f} (normalised to 1)", flush=True)
        else:
            loc = None
            print("local credit: no game carries local trade (encoded before it): off for this iteration", flush=True)
    kl_start = kl_sum / max(1.0, kl_n)
    print(f"KL of the data to the network now (the lag of the games): {kl_start:.4f}", flush=True)
    if a.vtrace:
        rho = np.exp(np.concatenate([log_rho[gi] for gi in usable]))
        print(f"v-trace: per-unit geometric-mean ratio median {np.median(rho):.4f}, "
              f"above 1 (truncated) {float((rho > 1).mean()):.1%}", flush=True)
    returns = np.concatenate([ret[gi] for gi in usable])
    vals = np.concatenate([values[gi] for gi in usable])
    explained = 1 - float(np.var(returns - vals) / max(np.var(returns), 1e-8))
    # against the games' own discounted returns: the V-trace targets above are built from the values themselves,
    # which makes that figure look better than the critic is (2026-10-11 review)
    def discounted(r):
        out, acc = np.zeros_like(r), 0.0
        for t in range(len(r) - 1, -1, -1):
            acc = r[t] + a.gamma * acc
            out[t] = acc
        return out
    mc = np.concatenate([discounted(rewards[gi]) for gi in usable])
    explained_mc = 1 - float(np.var(mc - vals) / max(np.var(mc), 1e-8))
    print(f"values: explained variance {explained:+.2f} (of the games' returns {explained_mc:+.2f}); "
          f"advantages normalised (mean {mean:+.3f}, std {std:.3f})", flush=True)
    if a.upgo > 0:
        up_all = np.concatenate([adv_up[gi] for gi in usable])
        print(f"upgo: mean advantage {up_all.mean():+.3f} (scale of the normalised advantages), "
              f"positive {float((up_all > 0).mean()):.0%}", flush=True)

    # 2. PPO epochs
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    history = []
    a.step = 0
    for epoch in range(a.epochs):
        rng = np.random.default_rng(epoch)
        order = [int(i) for i in rng.permutation(usable)]
        streams = Streams(lengths, a.streams, a.length, order, balanced_queues(order, lengths, a.streams, rng))
        state_h = net.initial_state(a.streams, device)
        ref_h = ref.initial_state(a.streams, device)
        sums = {k: 0.0 for k in ("pg", "v", "ent", "kl", "clip", "ref_kl", "distill", "local", "kl_data", "gnorm",
                                 "n")}
        stopped = False
        for row in streams.rounds():
            if a.stop_file and a.stop_file.exists():
                raise SystemExit(f"{a.stop_file} found: stopping without saving")
            c = chunk_batch(games, row, a.length, device, values, adv, ret, adv_up, loc, kl0 if a.vtrace else None,
                            prox, refp)
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
        print(f"epoch {epoch + 1}: {json.dumps(rec)}", flush=True)
        if stopped:
            break
    if a.sil and a.sil.exists() and not stopped:
        sil_rec = self_imitation(net, opt, a, device, ret_std, std)
        if sil_rec:
            history.append(sil_rec)
            print(f"self-imitation: {json.dumps(sil_rec)}", flush=True)
    out = {"model": net.state_dict(), "size": state.get("size", "M"), "decoder": "units",
           "model_options": model_options(net), "ret_std": ret_std, "kind": "tools",
           "learn_rl": {"init": str(a.init), "reference": str(a.reference or a.init), "games": len(usable),
                        "reward_spec": spec,
                        "history": history, "explained_variance": explained, "explained_variance_mc": explained_mc, "args": {k: str(v) for k, v in vars(a).items()},
                        "minutes": round((time.monotonic() - t0) / 60, 1)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(a.out)
    print(f"saved {a.out} after {out['learn_rl']['minutes']} min", flush=True)


def self_imitation(net: ToolsNet, opt, a, device, ret_std: float, adv_std: float) -> dict | None:
    """The success archive's games (--sil): their decisions where the discounted return beat the value now,
    imitated in proportion to by how much (the main advantages' scale), and the critic pulled up to those returns.
    Only better-than-expected decisions of won games count, so it can only reinforce what worked; the archive's
    games are older networks' choices, which self-imitation needs no importance correction for."""
    sources = {tuple(x) for x in json.loads(a.sil.read_text())}
    if not sources:
        return None
    games = Games([Path(c) for c in a.cache], sources=sources)
    games.units = True
    usable = [i for i, g in enumerate(games.games)
              if "pol_has" in g and bool(np.asarray(g["pol_has"]).any()) and all(f"rew_{c}" in g for c in REWARD_COMPONENTS)
              and (not needs_shaping_arrays(a) or all(k in g for k in SHAPING_ARRAYS))]
    if not usable:
        return None
    lengths = {i: int(games.lengths[i]) for i in usable}
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
    for epoch in range(a.sil_epochs):
        rng = np.random.default_rng(4000 + epoch)
        order = [int(i) for i in rng.permutation(usable)]
        streams = Streams(lengths, a.streams, a.length, order, balanced_queues(order, lengths, a.streams, rng))
        state_h = net.initial_state(a.streams, device)
        for row in streams.rounds():
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
                value, _ = net.heads(z, b)
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


def value_pass(net: ToolsNet, games: Games, usable: list, lengths: dict, a, device) -> dict:
    """Every decision's value with the network as it is (no gradients), memory carried chunk to chunk."""
    net.eval()
    values = {i: np.zeros(lengths[i], np.float32) for i in usable}
    streams = Streams(lengths, a.streams, a.length, sorted(usable, key=lambda i: -lengths[i]))
    state_h = net.initial_state(a.streams, device)
    with torch.no_grad():
        for row in streams.rounds():
            c = chunk_batch(games, row, a.length, device)
            _, _, z, state_h = net.core(c["b"], state_h, a.length, c["starts"], return_state=True)
            v, _ = net.heads(z, c["b"])
            v = v.view(a.streams, a.length).cpu().numpy()
            for s_, item in enumerate(row):
                if item is not None:
                    gi, start = item
                    n = min(a.length, lengths[gi] - start)
                    values[gi][start:start + n] = v[s_, :n]
    return values


def value_warmup(net: ToolsNet, games: Games, usable: list, lengths: dict, ret: dict, a, device) -> None:
    """The value head alone fitted to the new returns for --value-warmup epochs: the memory and the policy stay
    as they are (the trunk runs without gradients), so the first iteration after a reward change does not
    compute its advantages with a critic of the old reward."""
    head = list(net.value_head.parameters())
    opt = torch.optim.Adam(head, lr=3e-4)
    for epoch in range(a.value_warmup):
        rng = np.random.default_rng(1000 + epoch)
        order = [int(i) for i in rng.permutation(usable)]
        streams = Streams(lengths, a.streams, a.length, order, balanced_queues(order, lengths, a.streams, rng))
        state_h = net.initial_state(a.streams, device)
        total, n = 0.0, 0
        for row in streams.rounds():
            c = chunk_batch(games, row, a.length, device, ret=ret, values={i: ret[i] * 0 for i in usable},
                            adv={i: ret[i] * 0 for i in usable}, adv_up=None)
            with torch.no_grad():
                _, _, z, state_h = net.core(c["b"], state_h, a.length, c["starts"], return_state=True)
            value = net.value_head(torch.cat([z, c["b"]["privileged"]], -1)).squeeze(-1)
            w = c["valid"] / float(c["valid"].sum() or 1.0)
            loss = 0.5 * (((value - c["ret"]) ** 2) * w).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + float(loss.detach()), n + 1
        print(f"value warm-up epoch {epoch + 1}: loss {total / max(1, n):.5f} over {n} chunks", flush=True)


def ppo_step(net: ToolsNet, ref: ToolsNet, opt, c: dict, state, ref_state, a):
    """One optimizer step on a round of chunks, in micro-batches of streams (gradients accumulated). Returns the
    statistics and both networks' memories after the chunks (detached, to carry into the next round)."""
    new_h, new_c = state[0].clone(), state[1].clone()
    ref_h, ref_c = ref_state[0].clone(), ref_state[1].clone()
    n_streams = state[0].shape[1]
    L = a.length
    total_valid = float(c["valid"].sum()) or 1.0
    opt.zero_grad(set_to_none=True)
    sums = {k: 0.0 for k in ("pg", "v", "ent", "kl", "clip", "ref_kl", "distill", "local", "kl_data")}
    for s0 in range(0, n_streams, a.micro):
        s1 = min(n_streams, s0 + a.micro)
        idx = slice(s0 * L, s1 * L)
        sub = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == n_streams * L else v) for k, v in c.items()
               if k not in ("b", "act", "lab")}
        b = {k: v[idx] for k, v in c["b"].items()}
        act = {k: v[idx] for k, v in c["act"].items()}
        lab = {k: v[idx] for k, v in c["lab"].items()}
        st = (state[0][:, s0:s1].contiguous(), state[1][:, s0:s1].contiguous())
        h, m, z, (hs, cs) = net.core(b, st, L, sub["starts"], return_state=True)
        new_h[:, s0:s1], new_c[:, s0:s1] = hs.detach(), cs.detach()
        lp, ent, _ = net.score(h, m, z, b, act, temperature=a.temperature)
        value, _ = net.heads(z, b)
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
        v_clip = sub["v_old"] + (value - sub["v_old"]).clamp(-a.clip, a.clip)
        v_loss = 0.5 * (torch.max((value - sub["ret"]) ** 2, (v_clip - sub["ret"]) ** 2) * w).sum()
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
            _, _, bc = net.score(h, m, z, b, lab, smoothing=0.05, temperature=1.0)
            sup = sub["sup"] * (sub["valid"] > 0).float()
            # per unit, like the policy term: bc is a decision's sum over its units (~100 units: ~100x the scale)
            distill = (bc / n_units * sup * sub["supw"]).sum() / max(1.0, float(c["sup"].sum()))
            loss = loss + a.distill * distill
        loss.backward()
        for key, val in (("pg", pg), ("v", v_loss), ("ent", entropy.sum()), ("kl", kl), ("clip", clipfrac),
                         ("ref_kl", ref_kl), ("distill", distill), ("local", local_loss), ("kl_data", kl_data)):
            sums[key] += float(val.detach())
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad)
    a.step = getattr(a, "step", 0) + 1
    for group in opt.param_groups:
        group["lr"] = a.lr * min(1.0, a.step / max(1, a.warmup))
    if torch.isfinite(norm):
        opt.step()
    opt.zero_grad(set_to_none=True)
    sums["gnorm"] = float(norm) if torch.isfinite(norm) else 0.0
    sums["n"] = 1.0
    return sums, (new_h, new_c), (ref_h, ref_c)


if __name__ == "__main__":
    main()
