"""Cloning the teacher into the tools network, from raw recordings (docs/tools-design.md, sections 5-6).

    python -m fa_train.tools_bc --raw <run>/raw --cache <run>/tools_cache --out <run>/checkpoints/tools_bc.pt
    python -m fa_train.tools_bc --raw ... --cache ... --out ... --sizes S,M,L --steps 4000     # the size study

1. Every recorded game is replayed through ToolsObserver; the teacher's recorded orders become command
   labels (orders the masks forbid -- the teacher's own mistakes -- are dropped and counted). Each game is
   cached as one compressed .npz (ragged tokens), so re-training never re-encodes.
2. Training samples chunks of consecutive decisions (the LSTM memory warms up over the first `burn` of
   them, without loss). Loss: smoothed cross-entropy over every argument + value (returns / std) + belief.
3. Held-out: whole games. Metrics are per argument type, teacher-forced: verb, subject, per-unit selection,
   blueprint, enhancement, option, count, cell (exact / within one cell), target. The best evaluation is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.record import read_blueprints, read_game  # noqa: E402
from fa_env.observation_precision import CACHE_FLOAT_DTYPES, PRECISION_CONTRACT
from fa_env.reclaim_memory import after_step as reclaim_after_step
from fa_train.recorded_rewards import recorded_returns
from fa_env.tools import (CELL_VERBS, COUNTS, FINE, FIRE_STATES, GRID, MAX_COMMANDS, N_TOGGLES, OPTION_VERBS,  # noqa: E402
                          SET_SUBJECT, SILO_KINDS, SINGLE_SUBJECT, SPOTS, TARGET_ROWS, TARGET_VERBS, ToolKit)
from fa_env.tools_obs import (CAND_KEYS, CAND_MOBILE, ENH_KEYS, PAIR_KEYS, STEP_KEYS, TOKEN_KEYS,  # noqa: E402
                              TOKEN_T_KEYS, V_INDEX, ToolsObserver)
from fa_train.tools_model import (ACTION_KEYS, METRIC_KEYS, UNIT_KEYS, UNIT_METRIC_KEYS, V_END, ToolsNet,  # noqa: E402
                                  apply_model_options, collate, commands_to_indices, empty_unit_labels,
                                  model_options, parameter_count, upgrade_options)

K = MAX_COMMANDS
CARRY_VERBS = SET_SUBJECT | {"commander_move", "commander_attack_move"}
GAMMA = 0.997
CACHE_SUFFIX = ".v3.npz"                     # observation encoding stays v3; unverified older artifacts are ignored
CACHE_FORMAT = 12                             # fine pixels: usub, the map's fine split (11: danger maps)
# 10:                              # tokens carry the last order given to the unit (9: move-then-build)
UNIT_LABEL_CAP = 128                          # units given a verb per decision in the labels; more: unknown
LABELS = tuple(k for k in ACTION_KEYS if k != "set")           # one index per command slot
# how the ragged arrays are stored: concatenated along their first axis, with per-step offsets
DTYPES = {**CACHE_FLOAT_DTYPES, "token_bp": np.int16, "token_order": np.int8, "token_type": np.int8,
          "token_last": np.int8,
          "cand_bp": np.int16, "cand_kind": np.int8, "enh_idx": np.int16,
          "can_build": np.int16, "can_enh": np.int16, "uverb": np.int8, **{k: np.int16 for k in UNIT_KEYS[1:]}}
GROUPS = {"t": TOKEN_KEYS + tuple(k + "_t" for k in TOKEN_T_KEYS) + ("set_t",) + UNIT_KEYS, "c": CAND_KEYS,
          "e": ENH_KEYS, "pb": ("can_build",), "pe": ("can_enh",)}
TOKEN_LABEL_KEYS = ("set_t",) + UNIT_KEYS     # labels stored along the tokens (not part of an observation)


# ---------------------------------------------------------------------- recordings -> cached arrays
def lookups(x: dict) -> dict:
    """Index tables of an observation for `command_allowed` (built once per observation)."""
    return {"index": {u: i for i, u in enumerate(x["ids"]) if u is not None},
            "cand": {b: j for j, b in enumerate(x.get("cand_ids") or [])},
            "can_build": {(int(i), int(j)) for i, j in x["can_build"]},
            "enh": {(int(i), x["enh_names"][j]) for i, j in x["can_enh"]},
            "upgrades": upgrade_options(x)}


def _pair(value) -> tuple[int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    if any(not isinstance(n, (int, np.integer)) for n in value):
        return None
    return int(value[0]), int(value[1])


def command_rejection_reason(c: dict, x: dict, look: dict, used: set | None = None,
                             *, shared_only: bool = False) -> str | None:
    """Return the first deterministic mask/schema reason a recorded command is not labelable."""
    if not isinstance(c, dict):
        return "malformed_command"
    verb = c.get("verb")
    if not isinstance(verb, str) or verb not in V_INDEX:
        return "unknown_verb"
    v = V_INDEX[verb]
    index = look["index"]
    if not x["mask_verb"][v]:
        return "verb_mask"
    raw_subject = c.get("subject")
    if not isinstance(raw_subject, (list, tuple)) or not raw_subject:
        return "subject_missing"
    subj = [index.get(u) for u in raw_subject]
    if not shared_only and any(i is None for i in subj):
        return "subject_missing"
    if not shared_only and any(i >= x["mask_subject"].shape[1] or not x["mask_subject"][v, i] for i in subj):
        return "subject_mask"
    if verb in SINGLE_SUBJECT and len(subj) != 1:
        return "subject_cardinality"
    if verb in SINGLE_SUBJECT and used is not None and (v, subj[0]) in used:
        # Conversion preserves the truthful queue, but the decoder reserves each verb/unit pair.
        return ("unrepresentable_queue_duplicate" if verb in {"build", "produce", "upgrade", "silo_build"}
                else "duplicate_single_subject")

    kind = None
    if verb in ("build", "produce"):
        j = look["cand"].get(c.get("bp"))
        if j is None:
            return "blueprint_missing"
        if (subj[0], j) not in look["can_build"]:
            return "blueprint_not_buildable"
        kind = int(x["cand_kind"][j])
        if (kind == CAND_MOBILE) != (verb == "produce"):
            return "blueprint_kind_mismatch"

    if verb == "build":
        place = c.get("place")
        if place not in (None, "cell", "next"):
            return "place_invalid"
        if place == "next":
            if kind >= 2:
                return "place_kind_mismatch"
            i = index.get(c.get("anchor"))
            if i is None:
                return "anchor_missing"
            if i >= len(x["mask_anchor"]) or not x["mask_anchor"][i]:
                return "anchor_mask"
        else:
            cell = c.get("cell")
            if cell is not None:
                pair = _pair(cell)
                if pair is None:
                    return "build_cell_invalid"
                gx, gz = pair
                mask = x["mask_build_cell"]
                if not (0 <= gx < mask.shape[2] and 0 <= gz < mask.shape[1] and
                        0 <= kind < mask.shape[0]):
                    return "build_cell_invalid"
                if not mask[kind, gz, gx]:
                    return "build_cell_mask"
            elif kind >= 2:                 # extractors and hydro plants need a deposit cell
                return "build_cell_missing"
            spot = c.get("spot")
            if spot is not None:
                pair = _pair(spot)
                if pair is None or any(not 0 <= n < SPOTS for n in pair):
                    return "spot_invalid"
    elif verb in CELL_VERBS and c.get("cell") is not None:
        pair = _pair(c["cell"])
        if pair is None:
            return "move_cell_invalid"
        gx, gz = pair
        mask = x["mask_move_cell"]
        if not (0 <= gx < mask.shape[1] and 0 <= gz < mask.shape[0]):
            return "move_cell_invalid"
        if not mask[gz, gx]:
            return "move_cell_mask"

    if verb == "upgrade" and subj[0] in look["upgrades"]:  # a factory: which upgrade is part of the label
        j = look["cand"].get(c.get("bp"))
        if j is None or j not in look["upgrades"][subj[0]]:
            return "upgrade_missing"
    if verb == "enhance" and (subj[0], c.get("enhancement")) not in look["enh"]:
        return "enhancement_missing"

    if verb in ("fire_state", "silo_build", "toggle"):
        try:
            option = int(c.get("option") or 0)
        except (TypeError, ValueError, OverflowError):
            return "option_invalid"
        if verb == "fire_state" and not 0 <= option < len(FIRE_STATES):
            return "option_invalid"
        if verb == "silo_build":
            if not 0 <= option < len(SILO_KINDS):
                return "option_invalid"
            if not x["mask_silo"][subj[0], option]:
                return "option_mask"
    if verb == "toggle":
        try:
            option = int(c.get("option") or 0)
        except (TypeError, ValueError, OverflowError):
            return "option_invalid"
        if not 0 <= option < N_TOGGLES:
            return "option_invalid"
        if not x["mask_toggle"][subj[0], option]:
            return "option_mask"
    if verb == "produce":
        if c.get("count") not in COUNTS:
            return "count_invalid"

    if verb in TARGET_VERBS:
        i = index.get(c.get("target"))
        if i is None:
            return "target_missing"
        row = TARGET_ROWS.index(TARGET_VERBS[verb])
        if i >= x["mask_target"].shape[1] or not x["mask_target"][row, i]:
            return "target_mask"
    return None


def partition_subjects(c: dict, x: dict, look: dict) -> tuple[dict | None, list, str | None]:
    """Validate command-wide arguments before splitting a set; known masked IDs never get retried."""
    if not isinstance(c, dict) or c.get("verb") not in SET_SUBJECT:
        return c, [], None
    reason = command_rejection_reason(c, x, look, shared_only=True)
    if reason is not None:
        return None, [], reason
    v, index = V_INDEX[c["verb"]], look["index"]
    present, missing = [], []
    for u in c["subject"]:
        i = index.get(u)
        if i is None:
            missing.append(u)
        elif i < x["mask_subject"].shape[1] and x["mask_subject"][v, i]:
            if u not in present:
                present.append(u)
    reason = None if present else ("subject_missing" if missing else "subject_mask")
    return dict(c, subject=present) if present else None, missing, reason


def carry_commands(commands: list[dict], x: dict, look: dict) -> list[dict]:
    """One retry for absent IDs only, with shared validation and original provenance."""
    carry = []
    for c in commands:
        if c["verb"] not in CARRY_VERBS or c.get("carried"):
            continue
        if c["verb"] in SET_SUBJECT:
            _, missing, reason = partition_subjects(c, x, look)
            if reason not in (None, "subject_missing", "subject_mask"):
                continue
        else:
            missing = [u for u in c.get("subject") or [] if u not in look["index"]]
            # Commander navigation shares the same cell validation as set navigation.
            reason = command_rejection_reason(c, x, look, shared_only=True)
            if reason not in (None, "subject_missing", "subject_mask"):
                continue
        if missing:
            carry.append(dict(c, subject=missing, carried=True))
    return carry


def _reserve_single_subject(c: dict, x: dict, look: dict, used: set) -> None:
    verb = c["verb"]
    if verb in SINGLE_SUBJECT:
        used.add((V_INDEX[verb], look["index"][c["subject"][0]]))


def command_allowed(c: dict, x: dict, look: dict, used: set) -> bool:
    """Validate a decoder-representable label; reserve single subjects after valid arguments."""
    if command_rejection_reason(c, x, look, used) is not None:
        return False
    _reserve_single_subject(c, x, look, used)
    return True


def _new_command_diagnostics(source: str) -> dict:
    return {"partial_subjects": {}, "source": source, "before_filter": 0, "after_filter": 0, "overflow_past_k": 0,
            "kept_for_labels": 0, "nonempty_unrepresentable_turns": 0, "clear_only_turns": 0,
            "rejections": {}, "per_verb": {}}


def _bump(counter: dict, key: str, amount: int = 1) -> None:
    counter[key] = int(counter.get(key, 0)) + amount


def _merge_command_diagnostics(total: dict, part: dict) -> dict:
    for key in ("before_filter", "after_filter", "overflow_past_k", "kept_for_labels",
                "nonempty_unrepresentable_turns", "clear_only_turns"):
        total[key] = int(total.get(key, 0)) + int(part.get(key, 0))
    for key, count in part.get("partial_subjects", {}).items():
        _bump(total.setdefault("partial_subjects", {}), key, int(count))
    for reason, count in part.get("rejections", {}).items():
        _bump(total["rejections"], reason, int(count))
    for verb, v in part.get("per_verb", {}).items():
        dest = total["per_verb"].setdefault(verb, {"before_filter": 0, "after_filter": 0,
                                                   "overflow_past_k": 0, "rejections": {}})
        for key in ("before_filter", "after_filter", "overflow_past_k"):
            dest[key] += int(v.get(key, 0))
        for reason, count in v.get("rejections", {}).items():
            _bump(dest["rejections"], reason, int(count))
    return total


def filter_commands(commands: list[dict], x: dict, look: dict | None = None,
                    source: str = "unspecified") -> tuple[list[dict], dict]:
    """Filter labels with stable per-reason counts; returned commands are the first K valid commands."""
    look = look or lookups(x)
    diagnostics = _new_command_diagnostics(source)
    diagnostics["before_filter"] = len(commands)
    used: set = set()
    accepted = []
    for c in commands:
        verb = c.get("verb") if isinstance(c, dict) and isinstance(c.get("verb"), str) else "unknown"
        per_verb = diagnostics["per_verb"].setdefault(
            verb, {"before_filter": 0, "after_filter": 0, "overflow_past_k": 0, "rejections": {}})
        per_verb["before_filter"] += 1
        original = c
        c, missing, reason = partition_subjects(c, x, look)
        if (verb in SET_SUBJECT and isinstance(original.get("subject"), (list, tuple)) and
                original["subject"] and reason in (None, "subject_missing", "subject_mask")):
            counters = diagnostics["partial_subjects"]
            retained = len(c["subject"]) if c else 0
            masked = len(original["subject"]) - retained - len(missing)
            _bump(counters, "retained_subjects", retained)
            _bump(counters, "missing_subjects", len(missing))
            _bump(counters, "masked_subjects", masked)
            if retained and (missing or masked):
                _bump(counters, "partially_retained_groups")
            if missing and not retained and not masked:
                _bump(counters, "all_missing_groups")
        if reason is None:
            reason = command_rejection_reason(c, x, look, used)
        if reason is not None:
            _bump(diagnostics["rejections"], reason)
            _bump(per_verb["rejections"], reason)
            continue
        _reserve_single_subject(c, x, look, used)
        accepted.append(c)
        per_verb["after_filter"] += 1
    from fa_env.build_cancellation import atomic_cancellation_groups
    groups, broken = atomic_cancellation_groups(accepted)
    for c in broken:
        diagnostics['per_verb'][c['verb']]['after_filter'] -= 1
        _bump(diagnostics['rejections'], 'paired_build_cancellation_rejected')
        _bump(diagnostics['per_verb'][c['verb']]['rejections'], 'paired_build_cancellation_rejected')
    accepted = [c for group in groups for c in group]
    kept = []
    for group in groups:
        if len(kept) + len(group) > K:
            break
        kept.extend(group)
    diagnostics["after_filter"] = len(accepted)
    diagnostics["overflow_past_k"] = len(accepted) - len(kept)
    diagnostics["kept_for_labels"] = len(kept)
    for c in accepted[len(kept):]:
        verb = c.get("verb", "unknown")
        diagnostics["per_verb"][verb]["overflow_past_k"] += 1
    return kept, diagnostics


def command_arg_indices(c: dict, x: dict, look: dict) -> dict | None:
    """One command's verb and argument indices, as `commands_to_indices` gives them for a slot, from the
    observation's lookups (that function rebuilds every table per call: 4x the encoding time of a long M28
    game when called per command). None: the observation cannot express the command."""
    verb, index = c["verb"], look["index"]
    nf = x.get("fine")
    fine = (int(round(float(nf[0]))), int(round(float(nf[1])))) if nf is not None and max(nf[0], nf[1]) > 1.5         else None                           # the map's fine pixels per cell side, when it has them
    subj = [index[u] for u in c.get("subject") or [] if u in index]
    if not subj:
        return None
    one = {"verb": V_INDEX[verb]}
    if verb in TARGET_VERBS:
        if c.get("target") not in index:
            return None
        one["target"] = index[c["target"]]
    if verb in ("build", "produce") or (verb == "upgrade" and subj[0] in look["upgrades"]):
        j = look["cand"].get(c.get("bp"))
        if j is None or (verb == "upgrade" and j not in look["upgrades"][subj[0]]):
            return None
        one["bp"] = j
    if verb == "enhance":
        j = next((int(j) for i, j in x["can_enh"] if int(i) == subj[0] and x["enh_names"][j] == c.get("enhancement")),
                 None)
        if j is None:
            return None
        one["enh"] = j
    if verb in OPTION_VERBS:
        one["option"] = int(c.get("option") or 0)
    nxt = False
    if verb == "build" and int(x["cand_kind"][one["bp"]]) < 2:           # placed, not on a deposit
        nxt = c.get("place") == "next"
        one["place"] = 1 if nxt else 0
        if nxt and c.get("anchor") in index:
            one["anchor"] = index[c["anchor"]]
        # on a map with fine pixels a spot is one of its pixel's: without the pixel (a recording from before
        # them) it was the cell's, a different place -- no lesson
        if not nxt and c.get("spot") is not None and (not fine or c.get("sub") is not None):
            one["spot"] = c["spot"][1] * SPOTS + c["spot"][0]
    if verb == "produce":
        one["count"] = COUNTS.index(c["count"]) if c.get("count") in COUNTS else 0
    if verb in CELL_VERBS and c.get("cell") is not None and not nxt:
        gx, gz = c["cell"]
        one["cell"] = gz * GRID + gx
        sub = _pair(c.get("sub")) if fine else None
        if sub is not None and 0 <= sub[0] < fine[0] and 0 <= sub[1] < fine[1]:
            one["sub"] = sub[1] * FINE + sub[0]
    return one


WAY_VERBS = frozenset({V_INDEX["move"], V_INDEX["commander_move"]})


def unit_labels(commands: list[dict], x: dict, look: dict) -> dict:
    """One decision's commands as per-unit labels (the 'units' decoder): every commanded unit gets its first
    command the observation can express; a unit whose command it cannot express is unknown (-1: no lesson),
    never 'nothing'. No slot limit: only UNIT_LABEL_CAP units per decision.

    A move ahead of another order is only the way there: M28 gives 43 % of its engineers' builds as "move, then
    build", and labelled 'move' they taught its clones to walk and never build (one extractor after 5 minutes)."""
    index = look["index"]
    lab = empty_unit_labels(len(x["ids"]))
    used: set = set()

    def unknown(units):
        for u in units:
            i = index.get(u)
            if i is not None and lab["uverb"][i] == 0:
                lab["uverb"][i] = -1

    for c in commands:
        subject = list(c.get("subject") or []) if isinstance(c, dict) else []
        part, _, reason = partition_subjects(c, x, look)
        if reason is None:
            reason = command_rejection_reason(part, x, look, used)
        one = command_arg_indices(part, x, look) if reason is None else None
        if one is None:
            unknown(subject)
            continue
        _reserve_single_subject(part, x, look, used)
        unknown(u for u in subject if u not in part["subject"])          # members the verb's mask leaves out
        for u in part["subject"]:
            i = index[u]
            if lab["uverb"][i] > 0 and not (int(lab["uverb"][i]) in WAY_VERBS and one["verb"] not in WAY_VERBS):
                continue                                                 # its first command stands
            lab["uverb"][i] = one["verb"]
            for key in UNIT_KEYS[1:]:
                lab[key][i] = one.get(key[1:], -1)
    for i in np.flatnonzero(lab["uverb"] > 0)[UNIT_LABEL_CAP:]:
        lab["uverb"][i] = -1
        for key in UNIT_KEYS[1:]:
            lab[key][i] = -1
    return lab


def should_supervise_commands(commands: list[dict], accepted: list[dict]) -> bool:
    """Empty teacher turns mean END; invalid-only nonempty turns are unknown and must not teach END."""
    return not commands or bool(accepted)


def supervise_recorded_step(step: dict, commands: list, retained: list) -> bool:
    """External interventions occur after the input observation: never label that transition."""
    return (step.get("supervised") is not False and
            not bool(step.get("teacher_interventions")) and
            should_supervise_commands(commands, retained) and
            (not step.get("sent") or bool(retained)))


def encode_game(path: Path, bps: dict) -> tuple[dict | None, dict]:
    hello, steps, end = read_game(path)
    diag = _new_command_diagnostics("recording_bc")
    stats = {"steps": len(steps), "commands": 0, "dropped": 0, "kept": 0,
             "command_diagnostics": diag}
    if end is None or not steps:
        return None, stats
    info, meta = hello["hello"], hello["meta"]
    policy_labels = meta.get("policy_command_labels") is True
    # a game the policy played while a teacher said what it would do (DAgger): the steps' `labels` are the
    # lesson, the played commands only keep the observer's bookkeeping as it was live
    teacher_labels = bool(meta.get("label_source"))
    me = meta["me"]
    kit = ToolKit(info, bps)
    ob = ToolsObserver(info, bps, me, kit)
    rows, trade, income, bc_supervise, unit_supervise = [], [], [], [], []
    carry: list[dict] = []          # orders to units not observed yet, tried again at the next decision
    for st in steps:
        s, obs = st["step"], st["msg"]["obs"]
        x = ob.observe(obs)
        units = {str(u["id"]): u for u in obs.get("units", []) if u.get("army") == me}
        sent = s.get("sent") or []
        if teacher_labels:
            # the shadow AI's raw orders (recorded since 2026-10-12) keep exact positions: its commands again from
            # them, with fine pixels on a map that has them; else the recorded labels (cells only)
            cmds = kit.commands_from_actions(s["shadow"], units) if kit.has_fine and s.get("shadow") else \
                s.get("labels") or []
        elif policy_labels:
            if s.get("policy_command_labels") is not True or not isinstance(s.get("commands"), list):
                raise ValueError("policy-command recording requires explicit per-step commands and flag")
            cmds = s["commands"]
        else:
            if s.get("policy_command_labels") is True:
                raise ValueError("policy-command step requires matching recording metadata")
            cmds = kit.commands_from_actions(sent, units, prior_commands=carry)
        fresh = [c for c in cmds if not c.get("carried")]
        look = lookups(x)
        keep, step_diag = filter_commands(cmds, x, look, source="recording_bc")
        _merge_command_diagnostics(diag, step_diag)
        # END is evidence of an empty raw turn, not nonempty intent lost in conversion/filtering.
        # This also covers a current clear that cancels all prior carried commands.
        labelled = not teacher_labels or s.get("labels") is not None
        bc_supervise.append(labelled and (should_supervise_commands(cmds, keep) if teacher_labels
                                          else supervise_recorded_step(s, cmds, keep)))
        if sent and not keep and not teacher_labels:
            diag["nonempty_unrepresentable_turns"] += 1
            if all(a.get("type") == "clear" for a in sent):
                diag["clear_only_turns"] += 1
        for c in fresh:
            stats.setdefault("verbs", {})[c["verb"]] = stats.get("verbs", {}).get(c["verb"], 0) + 1
        # An AI orders a unit the moment it leaves the factory, after the decision's observation was taken
        # (M28: a fifth of its commands). The order still stands at the next decision, when the unit is there.
        # The same for units still finishing at the observation (the order comes when they complete).
        carry = [] if policy_labels or teacher_labels else carry_commands(cmds, x, look)
        lab = commands_to_indices(keep, x)
        lab.update(unit_labels(cmds, x, look))
        unit_supervise.append(labelled and s.get("supervised") is not False
                              and not bool(s.get("teacher_interventions")))
        rows.append((x, lab, int(obs.get("tick", 0))))
        army = next((a for a in obs.get("armies", []) if a.get("index") == me), {})
        st_ = army.get("stats") or {}
        trade.append(float(st_.get("Enemies_MassValue_Destroyed", 0)) - float(st_.get("Units_MassValue_Lost", 0)))
        income.append(float(((army.get("eco") or {}).get("mass") or {}).get("income", 0.0)))
        # what the acting side does after the decision (as ToolsEnv will do live)
        ob.set_wrecks(reclaim_after_step(ob.wrecks, s.get("queries") or [], s.get("answers") or [],
                                        s.get("sent") or [], s.get("results") or []))
        for a in s.get("sent") or []:
            if a.get("type") == "build" and kit.role.get(a.get("bp")) in ("mex", "hydro"):
                ob.claim(a["pos"][0], a["pos"][1], s["tick"])
        if policy_labels or teacher_labels:
            # ToolsEnv notes every attempted policy command, including masked prefix
            # actions and intent suppressed by navigation deduplication/site reservation.
            ob.note_commands(s.get("commands") or [] if teacher_labels else cmds, s["tick"], s.get("results"))
        else:
            ob.note_commands(keep, s["tick"])
    # Older recordings lack commander damage and the exact terminal transition.
    # Preserve their approximate targets with explicit provenance; new recordings
    # use the actual live components at the same decision chronology.
    n = len(rows)
    rewards = np.zeros(n, np.float32)
    for t in range(n - 1):
        rewards[t] = (trade[t + 1] - trade[t]) / 1000.0 + (income[t + 1] - income[t]) / 20.0
    outcome = end.get("outcome")
    rewards[-1] += 10.0 if outcome == "victory" else -10.0 if outcome == "defeat" else 5.0 * float(end.get("value_advantage") or 0)
    returns = np.zeros(n, np.float32)
    acc = 0.0
    for t in reversed(range(n)):
        acc = rewards[t] + GAMMA * acc
        returns[t] = acc
    exact = recorded_returns(steps, end, GAMMA)
    reward_source = "recorded_components_shaping_1" if exact is not None else "legacy_approximation"
    if exact is not None:
        returns = exact
    out = pack([x for x, _, _ in rows], [lab for _, lab, _ in rows])
    out.update(policy_arrays(steps, [x for x, _, _ in rows], fine=kit.has_fine))
    out.update(local_trade_arrays(steps, kit, bps, me))
    out.update(shaping_arrays(steps, kit, bps, me))
    out["returns"] = returns
    out["reward_source"] = np.array(reward_source)
    out["tick"] = np.array([t for _, _, t in rows], np.int64)
    out["bc_supervise"] = np.asarray(bc_supervise, dtype=bool)
    out["unit_supervise"] = np.asarray(unit_supervise, dtype=bool)   # per-unit labels have no lost turns
    out["command_diagnostics"] = np.array(json.dumps(diag, sort_keys=True))
    out["meta"] = np.array(json.dumps({**meta, "outcome": outcome, "minutes": end.get("minutes"),
                                      "reward_source": reward_source}))
    stats["reward_source"] = reward_source
    stats["commands"] = diag["before_filter"]
    stats["dropped"] = sum(diag["rejections"].values()) + diag["overflow_past_k"]
    stats["kept"] = diag["kept_for_labels"]
    return out, stats


REWARD_COMPONENTS = ("terminal", "trade", "economy", "commander")
# An economy unit is worth more than its mass cost: a mex lost is the mass it would still have produced, an engineer
# lost is the building it would have done (the 2026-10-11 loss analysis: against M28 our mexes and engineers were
# raided away in minutes 5-10 while the trade reward counted a T1 mex as 36 mass)
ECONOMY_HORIZON = 120.0        # seconds of a unit's mass production its loss costs beyond its build cost
ENGINEER_BONUS = 1.0           # an engineer (not the commander) is worth its cost again


def economy_bonus(bp: dict) -> float:
    """What losing the unit costs beyond its build cost, in mass."""
    cats_ = set(bp.get("categories") or [])
    bonus = float(bp.get("massProduction") or 0.0) * ECONOMY_HORIZON
    if "ENGINEER" in cats_ and "COMMAND" not in cats_:
        bonus += ENGINEER_BONUS * float(bp.get("mass") or 0.0)
    return bonus


OWN_REPLACED_FRACTION = 0.98    # an own unit that vanishes at least this healthy was upgraded or loaded, not killed
LOCAL_KILL_FRACTION = 0.3        # an enemy that vanishes from view counts as destroyed only below this much health


def local_trade_arrays(steps: list[dict], kit, bps: dict, me: int) -> dict:
    """Where damage happened between each decision and the next, valued in mass (the spatial credit of
    fa_train.learn_rl, 2026-10-10): own units' lost health negative (exact: own units are always seen), visible
    enemy units' lost health positive, and an enemy that vanished from view counted as destroyed only if it was
    already below LOCAL_KILL_FRACTION of its health (otherwise it probably walked out of sight). Values are
    health fraction x the unit's value (its mass cost plus economy_bonus), summed per map cell (cell = cz * GRID +
    cx, the network's indexing). Arrays: offsets_l [steps + 1], loc_cells (int16), loc_values (float32), and
    econ_trade [steps]: the economy bonus part of each decision's trade (what follows it until the next decision),
    in the trade reward's unit (mass / 1000), for learn_rl --econ-coef."""
    from fa_env.tools import GRID
    prev, cells, values, offsets = None, [], [], [0]
    econ = np.zeros(len(steps), np.float32)
    for i, st in enumerate(steps):
        units = {}
        for u in (st["msg"]["obs"].get("units") or []):
            maxhp = float(u.get("maxhp") or 0.0)
            if maxhp <= 0:
                continue
            bp = bps.get(u.get("bp")) or {}
            cost = float(bp.get("mass") or 0.0)
            bonus = economy_bonus(bp)
            cx, cz = kit.cell(float(u["x"]), float(u["z"]))
            units[u["id"]] = (float(u.get("hp") or 0.0) / maxhp, cost + bonus, u.get("army") == me, cz * GRID + cx,
                              bonus)
        if prev is not None:                                 # events between decision i - 1 and decision i
            acc = {}
            for uid, (frac, cost, own, cell, bonus) in prev.items():
                now = units.get(uid)
                if now is not None:
                    lost = max(0.0, frac - now[0])
                elif (own and frac < OWN_REPLACED_FRACTION) or (not own and frac < LOCAL_KILL_FRACTION):
                    lost = frac
                else:
                    # an own unit gone at full health was replaced or put away, not killed: an upgrade (a mex or
                    # factory becomes a new unit), a unit loaded into a transport. Counted as lost (to 2026-10-11),
                    # it gave every mex upgrade a negative local advantage
                    continue
                if lost <= 0 or cost <= 0:
                    continue
                acc[cell] = acc.get(cell, 0.0) + (-1.0 if own else 1.0) * lost * cost
                econ[i - 1] += (-1.0 if own else 1.0) * lost * bonus / 1000.0
            for cell, v in acc.items():
                cells.append(cell)
                values.append(v)
            offsets.append(len(cells))
        prev = units
    offsets.append(len(cells))                               # the last decision: nothing after it
    return {"offsets_l": np.asarray(offsets, np.int64), "loc_cells": np.asarray(cells, np.int16),
            "loc_values": np.asarray(values, np.float32), "econ_trade": econ}


FACTORY_ROLES = ("land_factory", "air_factory", "naval_factory")
ENERGY_IN_MASS = 0.02          # energy wasted, in mass: a T1 power generator (75 mass) makes ~2400 energy in 2 minutes


def shaping_arrays(steps: list[dict], kit, bps: dict, me: int) -> dict:
    """Per decision, measured against the true state (obs.privileged: every army's real totals, which only rewards
    and critics may use; the network never sees them), for learn_rl's shaping (2026-10-11):
      phi_intel [steps]  how much of the enemy we know, in [0, 1]: known enemy mexes / their real mexes (0.35),
                         known factories / real (0.15), enemy army mass seen or on radar / real (0.35), their
                         commander seen (0.15). A part with nothing to know counts as known. Measured against the
                         truth on purpose: by what we see alone, not scouting would look like an empty map.
      phi_map [steps]    map control: (our mexes - enemy mexes) / the map's mass deposits.
      waste [steps]      mass wasted to full storage until the next decision, plus energy x ENERGY_IN_MASS.
      mass_capacity [steps]   mass production per second of our extractors, fabricators and commander.
      reclaimed_mass [steps]  mass reclaimed so far (army stats).
      energy_stall [steps]    mass spending lost to an energy stall until the next decision.
    The potentials are used as differences (gamma * phi' - phi): a gain is rewarded when it happens, a loss
    penalised, and holding nothing earns nothing over time, so they cannot be farmed or bias towards long games."""
    n = len(steps)
    intel, mapc, waste = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    capacity, reclaimed = np.zeros(n, np.float32), np.zeros(n, np.float32)
    stall = np.zeros(n, np.float32)
    n_spots = max(1, sum(1 for d in kit.deposits if d[2] == "mass"))
    for i, st in enumerate(steps):
        obs = st["msg"]["obs"]
        priv = obs.get("privileged") or {}
        armies = obs.get("armies") or []
        enemies = [a["index"] for a in armies if a.get("relation") == "enemy" and not a.get("civilian")]
        real = {k: 0.0 for k in ("mexes", "factories", "armyMass", "combat")}
        for e in enemies:
            pe = priv.get(str(e)) or {}
            for k in real:
                real[k] += float(pe.get(k) or 0.0)
        mine = priv.get(str(me)) or {}
        known_mex = known_fac = 0
        seen_army = 0.0
        blips = 0
        commander_seen = False
        for u in obs.get("units") or []:
            if u.get("army") == me or u.get("army") not in enemies:
                continue
            if u.get("radar"):
                blips += 1
                continue
            bp = bps.get(u.get("bp")) or {}
            role = kit.role.get(u.get("bp"))
            if role == "mex":
                known_mex += 1
            elif role in FACTORY_ROLES:
                known_fac += 1
            c = set(bp.get("categories") or [])
            if "COMMAND" in c and not u.get("ghost"):
                commander_seen = True
            elif not u.get("ghost") and "MOBILE" in c and "ENGINEER" not in c and bp.get("weapons"):
                seen_army += float(bp.get("mass") or 0.0)
        ratio = lambda known, total: 1.0 if total <= 0 else min(1.0, known / total)  # noqa: E731
        mean_unit = real["armyMass"] / max(1.0, real["combat"])
        army_known = ratio(seen_army + 0.5 * blips * mean_unit, real["armyMass"]) if real["armyMass"] > 50 else 1.0
        intel[i] = (0.35 * ratio(known_mex, real["mexes"]) + 0.15 * ratio(known_fac, real["factories"])
                    + 0.35 * army_known + 0.15 * float(commander_seen))
        mapc[i] = (float(mine.get("mexes") or 0.0) - real["mexes"]) / n_spots
        # mass production capacity (extractors, fabricators, the commander), per second: the economy without the
        # reclaim bursts that make the recorded income jump by ~17/s from one decision to the next
        # the unit's actual output when the observation has it (pm: adjacency bonuses included, so a mass
        # storage beside a mex counts), else its blueprint's
        capacity[i] = sum(float(u["pm"]) if u.get("pm") is not None else
                          float((bps.get(u.get("bp")) or {}).get("massProduction") or 0.0) * float(u.get("frac", 1.0))
                          for u in obs.get("units") or [] if u.get("army") == me)
        own_army = next((a for a in armies if a.get("index") == me), {})
        reclaimed[i] = float((own_army.get("stats") or {}).get("Economy_Reclaimed_Mass") or 0.0)
        # waste: storage full and more coming in than asked for (rates per second; the decision's span in ticks)
        army = next((a for a in armies if a.get("index") == me), {})
        eco = army.get("eco") or {}
        nxt = steps[i + 1]["msg"]["obs"].get("tick", 0) if i + 1 < n else obs.get("tick", 0) + 20
        dt = max(0.0, (float(nxt) - float(obs.get("tick", 0))) / 10.0)
        for res, scale in (("mass", 1.0), ("energy", ENERGY_IN_MASS)):
            e = eco.get(res) or {}
            full = float(e.get("max") or 0.0) > 0 and float(e.get("stored") or 0.0) >= 0.98 * float(e.get("max") or 0.0)
            if full:
                waste[i] += scale * max(0.0, float(e.get("income") or 0.0) - float(e.get("requested") or 0.0)) * dt
        # energy stall: storage empty and more asked for than comes in: the game slows all building to the share
        # it can power, so that share of the mass spending is lost time
        en, ms = eco.get("energy") or {}, eco.get("mass") or {}
        e_req, e_inc = float(en.get("requested") or 0.0), float(en.get("income") or 0.0)
        if e_req > e_inc > 0 and float(en.get("stored") or 0.0) < 0.02 * max(1.0, float(en.get("max") or 0.0)):
            stall[i] = (1.0 - e_inc / e_req) * float(ms.get("requested") or 0.0) * dt
    return {"phi_intel": intel, "phi_map": mapc, "waste": waste, "mass_capacity": capacity,
            "reclaimed_mass": reclaimed, "energy_stall": stall}


def policy_arrays(steps: list[dict], xs: list[dict], fine: bool = False) -> dict:
    """What the network itself chose at each decision, when the recording has it (step["policy"], since
    2026-10-10), for learning from recorded games with PPO, and each decision's reward components:
      pol_has [steps]            the step's choice is usable (recorded, and its units match the observation)
      pol_logp, pol_value        the decision's log-probability and value estimate when it was played
      pol_<UNIT_KEYS> [tokens]   the chosen per-unit actions (rows as the labels', along offsets_t)
      pol_unit_logp [tokens]     each unit's own log-probability (factored PPO)
      rew_<component> [steps]    the live reward components of the decision (recorded transitions)
    Absent arrays: an older recording (no choices) or a slots-decoder game. Empty dict when nothing applies.
    `fine`: the map has fine pixels; a decision whose cell orders came without one (a network before the fine
    head) placed its spots in the cell, not the pixel -- unusable (pol_has False)."""
    from fa_train.tools_model import UNIT_KEYS
    out = {}
    transitions = [st["step"].get("transition") for st in steps]
    if transitions and all(t and t.get("reward_components") for t in transitions):
        for c in REWARD_COMPONENTS:
            out[f"rew_{c}"] = np.array([float(t["reward_components"].get(c, 0.0)) for t in transitions], np.float32)
    policies = [st["step"].get("policy") for st in steps]
    if not any(p and isinstance(p.get("action"), dict) and "units" in p["action"] for p in policies):
        return out
    n_tok = [len(x["ids"]) for x in xs]
    has = np.zeros(len(xs), bool)
    logp = np.zeros(len(xs), np.float32)
    value = np.zeros(len(xs), np.float32)
    rows = {k: [] for k in UNIT_KEYS}
    unit_lp = []
    for i, (p, n) in enumerate(zip(policies, n_tok)):
        row = {k: np.full(n, 0 if k == "uverb" else -1, np.int64) for k in UNIT_KEYS}
        ulp = np.zeros(n, np.float32)
        ok = bool(p) and isinstance(p.get("action"), dict) and len(p.get("unit_logp") or []) == n
        if ok:
            try:
                for entry in p["action"].get("units") or []:
                    for k, v in zip(UNIT_KEYS, entry[1:]):
                        row[k][entry[0]] = v
                if fine and np.any((row["ucell"] >= 0) & (row["usub"] < 0)):
                    ok = False
                ulp[:] = np.asarray(p["unit_logp"], np.float32)
            except (IndexError, TypeError, ValueError):
                ok = False
        if ok:
            has[i], logp[i], value[i] = True, float(p["logp"]), float(p.get("value") or 0.0)
        else:
            row = {k: np.full(n, 0 if k == "uverb" else -1, np.int64) for k in UNIT_KEYS}
            ulp[:] = 0.0
        for k in UNIT_KEYS:
            rows[k].append(row[k])
        unit_lp.append(ulp)
    out["pol_has"], out["pol_logp"], out["pol_value"] = has, logp, value
    for k in UNIT_KEYS:
        out[f"pol_{k}"] = np.concatenate(rows[k]).astype(DTYPES.get(k, np.int16))
    out["pol_unit_logp"] = np.concatenate(unit_lp).astype(np.float32)
    return out


def pack(xs: list[dict], labels: list[dict]) -> dict:
    """A game's observations and labels as flat arrays: ragged ones (per token, candidate, enhancement, pair)
    concatenated with per-step offsets, the rest stacked."""
    ragged = []
    for x, lab in zip(xs, labels):
        r = {k: x[k] for k in TOKEN_KEYS + CAND_KEYS + ENH_KEYS + PAIR_KEYS}
        r.update({k + "_t": x[k].T for k in TOKEN_T_KEYS})
        r["set_t"] = lab["set"].T
        units = lab if "uverb" in lab else empty_unit_labels(len(x["tokens"]))    # slot labels only: nothing known
        r.update({k: units[k] for k in UNIT_KEYS})
        ragged.append(r)
    out = {}
    for g, keys in GROUPS.items():
        sizes = np.array([len(r[keys[0]]) for r in ragged], np.int64)
        out["offsets_" + g] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        for k in keys:
            out[k] = np.concatenate([r[k] for r in ragged]).astype(DTYPES.get(k, ragged[0][k].dtype))
    for k in STEP_KEYS:
        out[k] = np.stack([x[k] for x in xs]).astype(DTYPES.get(k, xs[0][k].dtype))
    for k in LABELS:
        out[k] = np.stack([lab[k] for lab in labels])
    return out


def _encoder_semantics_hash() -> str:
    """The encoder's version as caches record it. Bump CACHE_FORMAT whenever observations, labels or their
    interpretation change. (Hashing the source files instead, as on 2026-10-04, threw every cached game away at
    any edit of these files, comments included.)"""
    return f"fa-tools-cache-format-{CACHE_FORMAT}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cache_manifest(raw: Path, blueprints: Path, semantics: str) -> dict:
    raw_hash, blueprint_hash = _sha256_file(raw), _sha256_file(blueprints)
    key_hash = hashlib.sha256(b"fa-tools-cache-key-v1\0")
    source_root = str(raw.parent.resolve())
    source_name = raw.name
    for value in (raw_hash, blueprint_hash, semantics, source_root, source_name):
        key_hash.update(value.encode("utf-8") + b"\0")
    return {"format": CACHE_FORMAT, "observation_precision": PRECISION_CONTRACT,
            "raw_sha256": raw_hash, "blueprints_sha256": blueprint_hash,
            "encoder_sha256": semantics, "key": key_hash.hexdigest(),
            "source_name": source_name, "source_root": source_root, "created_ns": time.time_ns()}


def _cache_target(cache: Path, raw: Path, key: str) -> Path:
    stem = raw.name[:-len(".jsonl.gz")] if raw.name.endswith(".jsonl.gz") else raw.stem
    return cache / f"{stem}.{key[:20]}{CACHE_SUFFIX}"


def _read_manifest(z) -> dict | None:
    if "cache_manifest" not in z.files:
        return None
    try:
        value = z["cache_manifest"].item()
        return json.loads(str(value))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _cache_is_current(manifest: dict | None, semantics: str, key: str | None = None) -> bool:
    return bool(manifest and manifest.get("format") == CACHE_FORMAT and
                manifest.get("observation_precision") == PRECISION_CONTRACT and
                manifest.get("encoder_sha256") == semantics and manifest.get("key") and
                (key is None or manifest.get("key") == key))


CACHE_INDEX = "index.json"


def cache_index(cache: Path) -> dict[str, dict]:
    """name -> {"bytes", "mtime_ns", "manifest", "bc"} of every encoded game in `cache`. Opening every .npz for its
    manifest -- twice per training, 5,700 games -- took most of a run's start (6-40 min, 2026-10-10); a file's entry
    is kept in cache/index.json while its size and time are unchanged, and only new or changed files are opened."""
    index_path = cache / CACHE_INDEX
    try:
        old = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    out, changed = {}, False
    for path in sorted(cache.glob("*" + CACHE_SUFFIX)):
        try:
            st = path.stat()
        except OSError:
            continue
        entry = old.get(path.name)
        if entry and entry.get("bytes") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns:
            out[path.name] = entry
            continue
        changed = True
        try:
            with np.load(path, allow_pickle=False) as z:
                out[path.name] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "manifest": _read_manifest(z),
                                  "bc": "bc_supervise" in z.files}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            continue
    if changed or set(out) != set(old):
        try:
            tmp = index_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(out), encoding="utf-8")
            os.replace(tmp, index_path)
        except OSError:
            pass                                       # a read-only cache still works, only slower
    return out


def current_cache_paths(cache: Path) -> list[Path]:
    """List cache artifacts safe to reuse with the current encoder; leave legacy/stale files untouched."""
    semantics = _encoder_semantics_hash()
    return [cache / name for name, e in cache_index(cache).items()
            if _cache_is_current(e.get("manifest"), semantics) and e.get("bc")]


def _add_game_stats(totals: dict, stats: dict) -> None:
    for key in ("steps", "commands", "dropped", "kept"):
        totals[key] += int(stats.get(key, 0))
    for verb, count in stats.get("verbs", {}).items():
        _bump(totals.setdefault("verbs", {}), verb, int(count))
    diagnostics = stats.get("command_diagnostics")
    if diagnostics:
        _merge_command_diagnostics(totals["command_diagnostics"], diagnostics)


def _write_cache_atomic(target: Path, arrays: dict) -> None:
    """Readers only see complete ZIP archives, including during parallel encoding."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_cache(raw: Path, cache: Path, log=print, files=None) -> dict:
    """Encode finished recordings in `raw`, or only the explicit paths in `files`.

    Blueprints are always read from `raw`; when `files` is passed, every recording must use that folder's
    blueprint table. Existing two-argument callers keep the full-folder behavior.
    """
    cache.mkdir(parents=True, exist_ok=True)
    bps = {b["id"]: b for b in read_blueprints(raw)}
    blueprint_path = raw / "blueprints.json.gz"
    semantics = _encoder_semantics_hash()
    totals = {"games": 0, "skipped": 0, "steps": 0, "commands": 0, "dropped": 0, "kept": 0,
              "command_diagnostics": _new_command_diagnostics("recording_bc")}
    cut_off = []
    paths = sorted(Path(p) for p in files) if files is not None else sorted(raw.glob("*.jsonl.gz"))
    for path in paths:
        manifest = _cache_manifest(path, blueprint_path, semantics)
        target = _cache_target(cache, path, manifest["key"])
        if target.exists():
            try:
                with np.load(target, allow_pickle=False) as z:
                    cached_manifest = _read_manifest(z)
                    if _cache_is_current(cached_manifest, semantics, manifest["key"]):
                        diag = json.loads(str(z["command_diagnostics"].item()))
                        stats = {"steps": len(z["returns"]), "commands": diag.get("before_filter", 0),
                                 "dropped": sum(diag.get("rejections", {}).values()) +
                                            diag.get("overflow_past_k", 0),
                                 "kept": diag.get("kept_for_labels", 0), "command_diagnostics": diag}
                        _add_game_stats(totals, stats)
                        totals["games"] += 1
                        continue
            except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError, zipfile.BadZipFile):
                pass
        try:
            arrays, stats = encode_game(path, bps)
        except EOFError:                              # a game still running when its collection stopped
            cut_off.append(path.name)
            arrays, stats = None, {}
        except Exception as exc:                      # an odd recording must not stop the rest
            log(f"skipping {path.name}: {type(exc).__name__}: {exc}")
            arrays, stats = None, {}
        if arrays is None:
            totals["skipped"] += 1
            continue
        arrays["cache_manifest"] = np.array(json.dumps(manifest, sort_keys=True))
        _write_cache_atomic(target, arrays)
        totals["games"] += 1
        _add_game_stats(totals, stats)
    if cut_off:
        log(f"skipped {len(cut_off)} unfinished recordings (games cut off when a collection stopped; normal)")
    log(f"cache: {totals}")
    return totals


# ---------------------------------------------------------------------- batches
MAPPED_DISK_RESERVE = 15 << 30         # bytes a memory-mapped corpus never takes from the disk


# The memory-mapped copy of a game is ~14x its compressed .npz (16 MB against 1.1 MB for a 230-decision game,
# 2026-10-09). Layout 2 stores it compactly, read back as the identical arrays (training batches compared bit for
# bit on real games, _runs/tools/check_layout.py):
# - the map channels that do not change during the game (height, water, slope, deposits, start positions,
#   buildable: 8 of 16) once instead of at every decision, the others as distinct planes plus an index (structures,
#   remembered, radar repeat for long stretches);
# - rows that repeat stored once plus an index: only 2 % of the candidate rows and 44 % of the unit rows are
#   distinct (the same units and buildable blueprints, decision after decision);
# - the yes/no arrays of two or more dimensions (masks, set labels) eight to a byte.
# Opening a game costs ~1 ms per file (11,000 files for 150 games: 9.7 s), so an array is split into distinct rows
# and an index only when that saves at least MIN_SPLIT_SAVING bytes: layout 2 split a dozen per-slot label arrays
# to save a few KB each, 78 files per game (layout 3: ~64).
MAPPED_LAYOUT = 3
MIN_SPLIT_SAVING = 64 << 10


class SplitMap:
    """A [steps, C, G, G] map array stored as its constant channels (once) and its changing ones (per step);
    indexing by a step or a slice of steps returns exactly the original rows."""

    def __init__(self, static: np.ndarray, dynamic: np.ndarray, static_idx, dynamic_idx, steps: int):
        self.static, self.dynamic = static, dynamic
        self.static_idx, self.dynamic_idx = np.asarray(static_idx, np.int64), np.asarray(dynamic_idx, np.int64)
        self.dtype = static.dtype                          # static and dynamic come from the same array
        channels = len(self.static_idx) + len(self.dynamic_idx)
        self.shape = (steps, channels) + tuple((static if static.ndim == 3 else dynamic).shape[-2:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        rows = np.arange(self.shape[0])[key]
        out = np.empty(rows.shape + self.shape[1:], self.dtype)
        if len(self.static_idx):
            out[..., self.static_idx, :, :] = self.static
        if len(self.dynamic_idx):
            out[..., self.dynamic_idx, :, :] = self.dynamic[key]
        return out

    def __array__(self, dtype=None, copy=None):
        a = self[:]
        return a.astype(dtype) if dtype is not None else a


class IndexedRows:
    """An array stored as its distinct rows (along the first axis) and, per row, which one it is; indexing along
    the first axis returns exactly the original rows."""

    def __init__(self, rows: np.ndarray, index: np.ndarray):
        self.rows, self.index = rows, index
        self.dtype = rows.dtype
        self.shape = tuple(index.shape) + tuple(rows.shape[1:])
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        return self.rows[np.asarray(self.index[key])]

    def __array__(self, dtype=None, copy=None):
        a = self[:]
        return a.astype(dtype) if dtype is not None else a


def _distinct_rows(a: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(distinct rows, index) of `a` along its first axis when that is clearly smaller than `a`, else None."""
    if a.ndim < 2 or len(a) < 64 or a.dtype.kind not in "fiub":
        return None
    n = len(a)
    flat = np.ascontiguousarray(a).reshape(n, -1)
    width = flat.dtype.itemsize * flat.shape[1]
    if width <= 8:                                     # an index of 4 bytes would not pay for itself
        return None
    _, first, inverse = np.unique(flat.view(np.dtype((np.void, width))).ravel(), return_index=True,
                                  return_inverse=True)
    stored = len(first) * width + 4 * n
    if stored > 0.8 * n * width or n * width - stored < MIN_SPLIT_SAVING:
        return None
    return np.ascontiguousarray(a[first]), inverse.reshape(-1).astype(np.int32)


class PackedBool:
    """A bool array of two or more dimensions stored bit-packed along its last axis; indexing along the first
    axis returns exactly the original bools."""

    def __init__(self, bits: np.ndarray, width: int):
        self.bits, self.width = bits, int(width)
        self.dtype = np.dtype(bool)
        self.shape = tuple(bits.shape[:-1]) + (self.width,)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        return np.unpackbits(self.bits[key], axis=-1, count=self.width).view(bool)

    def __array__(self, dtype=None, copy=None):
        a = self[:]
        return a.astype(dtype) if dtype is not None else a


def _compact(name: str, a: np.ndarray) -> dict:
    """Files (name -> array) and a layout entry for one cached array: layout 2 (see MAPPED_LAYOUT)."""
    if name == "map" and a.ndim == 4 and len(a):
        constant = (a == a[:1]).reshape(a.shape[0], a.shape[1], -1).all(axis=(0, 2))
        static_idx, dynamic_idx = np.flatnonzero(constant), np.flatnonzero(~constant)
        dynamic = np.ascontiguousarray(a[:, dynamic_idx])
        layout = {"kind": "split", "static": static_idx.tolist(), "dynamic": dynamic_idx.tolist(),
                  "steps": int(a.shape[0]), "planes": False}
        files = {"map.static": np.ascontiguousarray(a[0, static_idx])}
        planes = _distinct_rows(dynamic.reshape((-1,) + dynamic.shape[2:])) if dynamic.size else None
        if planes is not None:                       # distinct [G, G] planes, and per (step, channel) which one
            files["map.planes"], files["map.plane_index"] = planes[0], planes[1].reshape(dynamic.shape[:2])
            layout["planes"] = True
        else:
            files["map.dynamic"] = dynamic
        return {"files": files, "layout": layout}
    if a.dtype == np.bool_ and a.ndim >= 2:
        return {"files": {f"{name}.bits": np.packbits(a, axis=-1)},
                "layout": {"kind": "bits", "width": int(a.shape[-1])}}
    rows = _distinct_rows(a) if a.dtype != np.bool_ else None
    if rows is not None:
        return {"files": {f"{name}.rows": rows[0], f"{name}.index": rows[1]}, "layout": {"kind": "rows"}}
    return {"files": {name: a}, "layout": None}


def open_mapped(folder: Path) -> dict:
    """A game's memory-mapped arrays (either layout), compact ones behind SplitMap / PackedBool / IndexedRows."""
    files = {p.name[:-4]: p for p in Path(folder).glob("*.npy")}
    layout_path = Path(folder) / "layout.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else {}
    load = lambda stem: np.load(files[stem], mmap_mode="r", allow_pickle=False)  # noqa: E731
    out = {}
    for name, entry in layout.items():
        if entry["kind"] == "split":
            dynamic = IndexedRows(load(f"{name}.planes"), load(f"{name}.plane_index")) if entry.get("planes")                 else load(f"{name}.dynamic")
            out[name] = SplitMap(load(f"{name}.static"), dynamic, entry["static"], entry["dynamic"], entry["steps"])
        elif entry["kind"] == "bits":
            out[name] = PackedBool(load(f"{name}.bits"), entry["width"])
        elif entry["kind"] == "rows":
            out[name] = IndexedRows(load(f"{name}.rows"), load(f"{name}.index"))
    for stem in files:
        if "." not in stem and stem not in out:
            out[stem] = load(stem)
    return out


class LazyMapped:
    """A memory-mapped game whose arrays are opened on first use (open_mapped's arrays, one at a time). A training
    run and each of its loader processes used to open all ~65 files of every game up front (370,000 files for the
    corpus) though a run samples only part of it."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._arrays: dict = {}
        self._files: dict | None = None
        self._layout: dict | None = None

    def _scan(self) -> None:
        if self._files is None:
            self._files = {p.name[:-4]: p for p in self.folder.glob("*.npy")}
            layout_path = self.folder / "layout.json"
            self._layout = json.loads(layout_path.read_text()) if layout_path.exists() else {}

    def _names(self) -> set:
        self._scan()
        return set(self._layout) | {s for s in self._files if "." not in s} | set(self._arrays)

    def _open(self, name: str):
        self._scan()
        load = lambda stem: np.load(self._files[stem], mmap_mode="r", allow_pickle=False)  # noqa: E731
        entry = self._layout.get(name)
        if entry is None:
            return load(name)
        if entry["kind"] == "split":
            dynamic = IndexedRows(load(f"{name}.planes"), load(f"{name}.plane_index")) if entry.get("planes")                 else load(f"{name}.dynamic")
            return SplitMap(load(f"{name}.static"), dynamic, entry["static"], entry["dynamic"], entry["steps"])
        if entry["kind"] == "bits":
            return PackedBool(load(f"{name}.bits"), entry["width"])
        return IndexedRows(load(f"{name}.rows"), load(f"{name}.index"))

    def __getitem__(self, name: str):
        if name not in self._arrays:
            if name not in self._names():
                raise KeyError(name)
            self._arrays[name] = self._open(name)
        return self._arrays[name]

    def __setitem__(self, name: str, value) -> None:
        self._arrays[name] = value

    def __contains__(self, name) -> bool:
        return name in self._names()

    def get(self, name, default=None):
        return self[name] if name in self else default

    def keys(self):
        return sorted(self._names())

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def pop(self, name, *default):
        value = self.get(name, *default) if default else self[name]
        self._arrays.pop(name, None)
        if self._files is not None:
            self._files.pop(name, None)
        if self._layout is not None:
            self._layout.pop(name, None)
        return value


def _mapped_arrays(path: Path, folder: Path, lazy: bool = False):
    """A cached game's arrays memory-mapped from `folder`, unpacked from its .npz there once: a full corpus is
    ~113 KB per decision (89 GB for 1,500 M28 games) -- more than the RAM next to everything else. Unpacked in the
    compact layout (MAPPED_LAYOUT); a copy in an older layout is rewritten."""
    stat = path.stat()
    marker = folder / "complete.json"
    identity = {"source": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "layout": MAPPED_LAYOUT}
    if not marker.exists() or json.loads(marker.read_text()) != identity:
        folder.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)                   # an interrupted rewrite is never taken for a finished one
        with zipfile.ZipFile(path) as z:
            needed = sum(item.file_size for item in z.infolist())
            if shutil.disk_usage(folder).free < needed + MAPPED_DISK_RESERVE:
                raise RuntimeError(f"not enough disk space to map {path.name}: keeping "
                                   f"{MAPPED_DISK_RESERVE >> 30} GB free for the rest of the computer")
            written, layout = set(), {}
            for item in z.infolist():
                if Path(item.filename).name != item.filename or not item.filename.endswith(".npy"):
                    raise ValueError(f"{path.name}: unexpected member {item.filename}")
                with z.open(item) as src:
                    a = np.lib.format.read_array(src, allow_pickle=False)
                part = _compact(item.filename[:-4], a)
                if part["layout"] is not None:
                    layout[item.filename[:-4]] = part["layout"]
                for stem, arr in part["files"].items():
                    tmp = folder / f"{stem}.npy.tmp"
                    with open(tmp, "wb") as dst:
                        np.lib.format.write_array(dst, np.asarray(arr), allow_pickle=False)
                    os.replace(tmp, folder / f"{stem}.npy")
                    written.add(f"{stem}.npy")
        (folder / "layout.json").write_text(json.dumps(layout))
        for old in folder.glob("*.npy"):                 # the previous layout's files
            if old.name not in written:
                try:
                    old.unlink()
                except OSError:                          # still mapped by another process: harmless, unread
                    pass
        marker.write_text(json.dumps(identity))
    return LazyMapped(folder) if lazy else open_mapped(folder)


class Games:
    def __init__(self, cache: Path | list, max_games: int | None = None, mapped: Path | None = None,
                 newest: int | None = None, sources: set | None = None):
        """All current cached games in `cache` (a folder or several); with `mapped`, read from memory-mapped
        copies in that folder instead of loaded into RAM. `newest`: only the newest this many (by encoding time,
        chosen from the cache index before anything is loaded)."""
        self.games = []
        semantics = _encoder_semantics_hash()
        current = []
        for c in (cache if isinstance(cache, (list, tuple)) else [cache]):
            for name, entry in cache_index(Path(c)).items():
                manifest = entry.get("manifest")
                if _cache_is_current(manifest, semantics) and entry.get("bc"):
                    current.append((Path(c) / name, manifest))   # never train on unverified legacy artifacts
        # A raw recording may be replaced in place. Keep the newest encoding for that source identity, while
        # retaining distinct recordings from different folders even when their basenames match.
        latest = {}
        for path, manifest in current:
            source = (manifest.get("source_root"), manifest.get("source_name"))
            previous = latest.get(source)
            if previous is None or int(manifest.get("created_ns", 0)) > int(previous[1].get("created_ns", 0)):
                latest[source] = (path, manifest)
        if sources is not None:             # only these recordings: {(source_root, source_name)}
            latest = {k: v for k, v in latest.items() if k in sources}
        if newest is not None:
            # an RL cache keeps every game ever encoded (3655 on 2026-10-11, ~30 MB each once loaded): loading all
            # of them to train on the newest few hundred ran the PC out of memory
            latest = dict(sorted(latest.items(), key=lambda kv: int(kv[1][1].get("created_ns", 0)))[-newest:])
        selected = sorted((item[0] for item in latest.values()))
        manifests = {item[0]: item[1] for item in latest.values()}
        if max_games is not None:
            selected = selected[:max_games]
        folders = [Path(c).resolve() for c in (cache if isinstance(cache, (list, tuple)) else [cache])]
        # which of the cache folders each game is from (0: the first)
        self.origin = np.array([folders.index(q) if (q := p.parent.resolve()) in folders else 0 for p in selected],
                               np.int64)
        self.folders: list[Path] = []       # with `mapped`: each game's folder of arrays, for loader processes
        for path in selected:
            if mapped is not None:
                folder = Path(mapped) / path.name.removesuffix(".npz")
                arrays = _mapped_arrays(path, folder, lazy=True)
                arrays.pop("cache_manifest", None)
                arrays["_cache_manifest"] = manifests[path]          # from the cache index: nothing opened
                self.games.append(arrays)
                self.folders.append(folder)
                continue
            with np.load(path, allow_pickle=False) as z:
                self.games.append({k: z[k] for k in z.files if k != "cache_manifest"})
                self.games[-1]["_cache_manifest"] = json.loads(str(z["cache_manifest"].item()))
        self.lengths = np.array([len(g["returns"]) for g in self.games])
        self.units = False              # which decoder's turns count as supervised (set by train)

    @classmethod
    def reopen(cls, folders: list[Path], units: bool) -> "Games":
        """The same memory-mapped games in another process (a batch loader): no cache scan, no checks."""
        self = object.__new__(cls)
        self.games = []
        for f in folders:                       # opened on first use: a loader samples only part of the corpus
            g = LazyMapped(Path(f))
            g.pop("cache_manifest", None)
            self.games.append(g)
        self.lengths = np.array([len(g["returns"]) for g in self.games])
        self.units, self.folders = units, list(folders)
        return self

    WIDE = {"tokens": np.float32, "token_bp": np.int64, "token_order": np.int64, "token_type": np.int64,
            "token_last": np.int64, "token_last_feat": np.float32,
            "cand_feat": np.float32, "cand_bp": np.int64, "cand_kind": np.int64, "enh_idx": np.int64,
            "enh_feat": np.float32, "can_build": np.int64, "can_enh": np.int64, "map": np.float32}

    def step_obs(self, g: dict, t: int) -> dict:
        out = {}
        for grp, keys in GROUPS.items():
            a, b = int(g["offsets_" + grp][t]), int(g["offsets_" + grp][t + 1])
            for k in keys:
                if k in TOKEN_LABEL_KEYS:
                    continue
                v = g[k][a:b]
                v = v.astype(self.WIDE[k]) if k in self.WIDE else v
                out[k[:-2] if k.endswith("_t") else k] = v.T if k.endswith("_t") else v
        for k in STEP_KEYS:
            out[k] = g[k][t].astype(self.WIDE[k]) if k in self.WIDE else g[k][t]
        return out

    def batch(self, picks: list[tuple[int, int]], length: int, device):
        """Chunks (game, start) of `length` decisions, sequence-major, as model tensors and labels."""
        obs, labels, starts, returns, supervised = [], [], [], [], []
        for gi, a in picks:
            g = self.games[gi]
            for t in range(a, a + length):
                obs.append(self.step_obs(g, t))
                o, b = int(g["offsets_t"][t]), int(g["offsets_t"][t + 1])
                labels.append({**{k: g[k][t] for k in LABELS},
                               "set": g["set_t"][o:b].T, **{k: g[k][o:b] for k in UNIT_KEYS}})
                starts.append(t == 0)
                returns.append(g["returns"][t])
                supervised.append(g["unit_supervise" if self.units else "bc_supervise"][t])
        b = collate(obs, device)
        T = b["tokens"].shape[1]
        lab = {}
        for key in ACTION_KEYS:
            if key == "set":
                arr = np.zeros((len(labels), K, T), bool)
                for i, l in enumerate(labels):
                    arr[i, :, : l["set"].shape[1]] = l["set"]
            else:
                arr = np.stack([l[key] for l in labels])
            lab[key] = torch.as_tensor(arr, device=device)
        for key in UNIT_KEYS:
            arr = np.full((len(labels), T), 0 if key == "uverb" else -1, np.int64)
            for i, l in enumerate(labels):
                arr[i, : len(l[key])] = l[key]
            lab[key] = torch.as_tensor(arr, device=device)
        return b, lab, torch.as_tensor(np.array(starts), device=device), \
            torch.as_tensor(np.array(returns, np.float32), device=device), \
            torch.as_tensor(np.array(supervised, np.float32), device=device)


def chunks(games: Games, ids: list[int], length: int, rng: np.random.Generator, n: int,
           burn: int = 8, weight: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Random chunks (game, start). A chunk that starts the game needs no memory warm-up, so every position is
    trained; starts are drawn from before the game too and clamped to it, which gives the opening as many chunks
    as any other stretch (uniform starts left the first decisions of every game untrained: night 1). `weight`
    (per game) makes a game's decisions count that many times."""
    ok = [i for i in ids if games.lengths[i] >= length]
    w = games.lengths[ok].astype(np.float64)
    if weight is not None:
        w = w * weight[ok]
    picks = rng.choice(ok, size=n, p=w / w.sum())
    return [(int(g), max(0, int(rng.integers(-(length - burn), games.lengths[g] - length + 1)))) for g in picks]


def keep_mask(starts: torch.Tensor, n_chunks: int, length: int, burn: int) -> torch.Tensor:
    """[n_chunks * length] weights: a chunk's first `burn` decisions only warm the memory up, unless the chunk
    starts the game (the memory is then exactly what it is live)."""
    keep = torch.ones(n_chunks, length, device=starts.device)
    at_start = starts.view(n_chunks, length)[:, 0]
    keep[~at_start, :burn] = 0
    return keep.flatten()


def all_chunks(games: Games, ids: list[int], length: int) -> list[tuple[int, int]]:
    return [(i, a) for i in ids for a in range(0, int(games.lengths[i]) - length + 1, length)]


def balanced_held_out_chunks(games: Games, ids: list[int], length: int) -> tuple[list[tuple[int, int]], dict]:
    """Select opening/middle/late windows for every held game; `length` is shortened for brief games.

    The former global first-N cap could evaluate only the first few held games and mostly their openings. The
    returned plan gives each game up to three windows at equal phase weights. `max_chunks` is deliberately not
    used as a hard limit: full coverage is more important, and at most three chunks per held game is cheap.
    """
    ids = list(ids)
    if not ids:
        return [], {"games_requested": 0, "games_covered": 0, "chunks": 0, "chunk_length": 0,
                    "phases": {}}
    chunk_length = min(length, min(int(games.lengths[i]) for i in ids))
    if chunk_length <= 0:
        return [], {"games_requested": len(ids), "games_covered": 0, "chunks": 0, "chunk_length": 0,
                    "phases": {}}
    picks = []
    by_game = []
    phases = ("opening", "middle", "late")
    phase_counts = {p: 0 for p in phases}
    for gi in ids:
        n = int(games.lengths[gi])
        last = max(0, n - chunk_length)
        starts = (0, last // 2, last)
        selected = []
        phase_map = {}
        for phase, start in zip(phases, starts):
            if start not in selected:
                selected.append(start)
                phase_map[phase] = start
                phase_counts[phase] += 1
        for start in selected:
            picks.append((gi, start))
        manifest = games.games[gi].get("_cache_manifest", {})
        try:
            meta = json.loads(str(games.games[gi]["meta"].item()))
        except (KeyError, ValueError, TypeError, AttributeError):
            meta = {}
        by_game.append({"game": int(gi), "source": manifest.get("source_name"), "decisions": n,
                        "map": meta.get("map"), "faction": meta.get("faction"), "ai": meta.get("ai"),
                        "ai_faction": meta.get("ai_faction"), "phase_starts": phase_map})
    coverage = {"games_requested": len(ids), "games_covered": len(by_game), "chunks": len(picks),
                "chunk_length": chunk_length, "phases": phase_counts, "by_game": by_game}
    return picks, coverage


# ---------------------------------------------------------------------- training
def evaluate_held_out(net, games: Games, ids, length, burn, ret_std, device, max_chunks=64):
    net.eval()
    sums = {}
    picks, coverage = balanced_held_out_chunks(games, ids, length)
    length = coverage["chunk_length"]
    if not picks:
        net.train()
        return {"loss": None, "policy_loss": None, "value_mse": None, "supervised_decisions": 0.0,
                "held_out_coverage": coverage}
    burn = min(burn, length)
    with torch.no_grad():
        for i in range(0, len(picks), 8):
            part = picks[i:i + 8]
            b, lab, starts, ret, supervised = games.batch(part, length, device)
            state = net.initial_state(len(part), device)
            logp, ent, value, belief, loss = net.evaluate(b, lab, state, seq_len=length, starts=starts)
            keep = keep_mask(starts, len(part), length, burn)
            policy_keep = keep * supervised
            for name, (hit, tot) in net.last_metrics.items():
                s = sums.setdefault(name, [0.0, 0.0])
                s[0] += float((hit * policy_keep).sum())
                s[1] += float((tot * policy_keep).sum())
            s = sums.setdefault("policy_loss", [0.0, 0.0])
            s[0] += float((loss * policy_keep).sum())
            s[1] += float(policy_keep.sum())
            s = sums.setdefault("supervised_decisions", [0.0, 0.0])
            s[0] += float(policy_keep.sum())
            s[1] += 1.0
            s = sums.setdefault("value_mse", [0.0, 0.0])
            s[0] += float((((value - ret / ret_std) ** 2) * keep).sum())
            s[1] += float(keep.sum())
    net.train()
    metrics = {k: (v[0] / v[1] if v[1] else None) for k, v in sums.items()}
    metrics["loss"] = metrics.get("policy_loss")      # compatibility with earlier reports
    metrics["held_out_coverage"] = coverage
    return metrics


def score_of(metrics: dict) -> float:
    keys = ("verb", "subject", "set_unit", "unit_none", "unit_ordered", "unit_verb", "blueprint", "enhancement",
            "option", "place", "anchor", "cell_near", "target")
    vals = [metrics[k] for k in keys if metrics.get(k) is not None]
    return float(np.mean(vals)) if vals else 0.0


def prefetch(make, n: int, depth: int = 3):
    """Build batches in a background thread while the GPU trains on the previous one."""
    import queue
    import threading
    q: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        for _ in range(n):
            q.put(make())
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


class _Batches(torch.utils.data.IterableDataset):
    """Training batches built in loader processes (memory-mapped games only): building a batch is Python and
    numpy work on one core (~1 s for 640 decisions), slower than a small network's step on the GPU."""

    def __init__(self, folders, units, ids, length, burn, batch_chunks, seed, weight=None):
        self.args = (list(folders), units, list(ids), length, burn, batch_chunks, seed, weight)

    def __iter__(self):
        folders, units, ids, length, burn, batch_chunks, seed, weight = self.args
        info = torch.utils.data.get_worker_info()
        games = Games.reopen(folders, units)
        rng = np.random.default_rng([seed, info.id if info else 0])
        while True:
            yield games.batch(chunks(games, ids, length, rng, batch_chunks, burn, weight), length, "cpu")


def loader_batches(games: "Games", ids, length, burn, batch_chunks, seed, n: int, workers: int, device,
                   weight=None):
    if not games.folders:
        raise ValueError("loader processes need memory-mapped games (--mapped)")
    data = torch.utils.data.DataLoader(_Batches(games.folders, games.units, ids, length, burn, batch_chunks, seed,
                                                weight),
                                       batch_size=None, num_workers=workers, prefetch_factor=2)
    to = lambda x: {k: v.to(device, non_blocking=True) for k, v in x.items()} if isinstance(x, dict) \
        else x.to(device, non_blocking=True)  # noqa: E731
    for i, item in enumerate(data):
        if i >= n:
            break
        yield tuple(to(x) for x in item)


def held_out_split(games: "Games", rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """(held-out game ids, training game ids): a tenth of the games, drawn among the games with supervised
    decisions for the trained decoder -- the others measure nothing (a draw without any lost a finished 2000-step
    round, 2026-10-06)."""
    order = rng.permutation(len(games.games))
    n_held = max(1, len(order) // 10)
    key = "unit_supervise" if games.units else "bc_supervise"
    measurable = [int(i) for i in order
                  if games.games[i].get(key) is None or bool(np.asarray(games.games[i][key]).any())]
    held = sorted(measurable[:n_held])
    return held, sorted(set(order.tolist()) - set(held))


def adapt_weights(weights: dict, expected: dict) -> dict:
    """A checkpoint from before the tokens carried the last order given to the unit: its token input layer gets
    zero weights for the new columns (they sit at the end), so it plays exactly as it did."""
    key = "tok_in.0.weight"
    if key in weights and key in expected and weights[key].shape != expected[key].shape \
            and weights[key].shape[0] == expected[key].shape[0] and weights[key].shape[1] < expected[key].shape[1]:
        grown = torch.zeros_like(expected[key])
        grown[:, : weights[key].shape[1]] = weights[key]
        weights = dict(weights, **{key: grown})
    return weights


def select_return_scale(dataset_ret_std: float, initial: dict | None = None) -> tuple[float, str]:
    """Keep initialized value targets in the checkpoint's units; legacy checkpoints use dataset units."""
    if initial is not None and "ret_std" in initial:
        supplied = initial["ret_std"]
        try:
            scale = float(supplied)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial checkpoint ret_std must be finite and positive") from exc
        if isinstance(supplied, bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("initial checkpoint ret_std must be finite and positive")
        return scale, "init_checkpoint"
    if not math.isfinite(dataset_ret_std) or dataset_ret_std <= 0:
        raise ValueError("dataset return standard deviation must be finite and positive")
    return dataset_ret_std, "legacy_checkpoint_dataset" if initial is not None else "dataset"


def train(cache: Path, out: Path, size: str = "M", steps: int = 4000, length: int = 40, burn: int = 8,
          batch_chunks: int = 16, lr: float = 3e-4, smoothing: float = 0.05, eval_every: int = 500,
          seed: int = 0, device: str | None = None, log=print, games: Games | None = None,
          init_checkpoint: Path | None = None, decoder: str = "slots", loader_workers: int = 0,
          recent_weight: float = 1.0, builder_weight: float = 1.0, accumulate: int = 1,
          grad_checkpoint: bool = False, stop_file: Path | None = None, memory_fraction: float | None = 0.9,
          verbs_after_attention: bool | None = None, fine_positions: bool | None = None,
          train_only: tuple | None = None) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    games = games or Games(cache)
    games.units = decoder == "units"
    rng = np.random.default_rng(seed)
    held, train_ids = held_out_split(games, rng)
    if not train_ids:
        raise ValueError("BC training needs at least two games so one can be held out")
    dataset_ret_std = float(np.concatenate([games.games[i]["returns"] for i in train_ids]).std() + 1e-6)
    initial = None
    if device.type == "cuda" and memory_fraction:
        # a batch that does not fit fails with an OutOfMemoryError instead of the driver thrashing at the limit
        # (the size-M attempt at the full batch: 24 GB used, ~100 nvlddmkm errors in half an hour, 2026-10-06)
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    net = ToolsNet(size, amp=device.type == "cuda", decoder=decoder).to(device)   # bf16 encoder, as PPO runs it
    net.build_budget = None                 # labels hold what M28 did, however many builds that was
    net.grad_checkpoint = bool(grad_checkpoint)
    if init_checkpoint is not None:
        if Path(init_checkpoint).resolve() == Path(out).resolve():
            raise ValueError("initial checkpoint and output checkpoint must be different paths")
        initial = torch.load(init_checkpoint, map_location=device, weights_only=False)
        if initial.get("size") != size:
            raise ValueError(f"checkpoint size {initial.get('size')!r} does not match requested size {size!r}")
        weights = initial.get("model")
        if not isinstance(weights, dict):
            raise ValueError("initial checkpoint has no model state")
        expected = net.state_dict()
        weights = adapt_weights(weights, expected)
        # a slots checkpoint may start a units network: everything but the per-unit input layer is shared
        fresh = {k for k in expected if k not in weights and k.startswith(("unit_in.", "unit_attn.", "intent_in.",
                                                                           "last_embed.") + ToolsNet.OPTIONAL_PREFIXES)}
        if (set(weights) | fresh) != set(expected) or any(weights[k].shape != expected[k].shape for k in weights):
            raise ValueError("initial checkpoint model parameters are incompatible with this ToolsNet")
        net.load_state_dict(weights, strict=not fresh)
        apply_model_options(net, initial)               # the network goes on as it was trained...
    if verbs_after_attention is not None:               # ...unless told otherwise
        net.verbs_after_attention = bool(verbs_after_attention)
    if fine_positions is not None:
        net.fine_positions = bool(fine_positions)
    log(f"model options: {model_options(net)}")
    ret_std, ret_std_source = select_return_scale(dataset_ret_std, initial)
    if train_only:                                       # the rest of the network stays as it is
        for name, p in net.named_parameters():
            p.requires_grad_(name.startswith(tuple(train_only)))
        log(f"training only {', '.join(train_only)}: "
            f"{sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e3:.0f} k weights")
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, eps=1e-5)
    log(f"size {size}: {parameter_count(net) / 1e6:.2f} M weights; {len(train_ids)} training games "
        f"({int(games.lengths[train_ids].sum())} decisions), {len(held)} held out; "
        f"dataset return std {dataset_ret_std:.2f}; selected return std {ret_std:.2f} ({ret_std_source})")
    weight = None
    origin = getattr(games, "origin", None)
    if recent_weight != 1.0 and origin is not None and (origin[train_ids] > 0).any():
        # the games from the caches after the first (DAgger: the policy's own games, where it actually gets to)
        # count `recent_weight` times
        weight = np.where(origin > 0, recent_weight, 1.0)
        d = games.lengths[train_ids] * weight[train_ids]
        log(f"games of the later caches count {recent_weight:g}x: "
            f"{d[origin[train_ids] > 0].sum() / d.sum():.0%} of the training decisions")
    t0 = time.monotonic()
    history, best, best_policy_loss = [], None, math.inf
    accumulate = max(1, int(accumulate))
    n_batches = steps * accumulate
    if loader_workers > 0:
        batches = loader_batches(games, train_ids, length, burn, batch_chunks, seed, n_batches, loader_workers,
                                 device, weight)
    else:
        batches = prefetch(lambda: games.batch(chunks(games, train_ids, length, rng, batch_chunks, burn, weight),
                                               length, device), n_batches)
    batches = iter(batches)
    net.train()
    for step in range(1, steps + 1):
        # `accumulate` batches of `batch_chunks` chunks make one update: the same lesson as one batch of
        # accumulate x batch_chunks, in a fraction of the memory
        opt.zero_grad()
        total = torch.zeros((), device=device)
        for _ in range(accumulate):
            b, lab, starts, ret, supervised = next(batches)
            keep = keep_mask(starts, batch_chunks, length, min(burn, max(0, length - 1)))
            if builder_weight != 1.0 and decoder == "units":
                # what the builders (engineers, the commander, factories) are told counts more: the economy
                # decides the game, and a small network spends itself on the army's moves (61 % of its verb
                # loss, builds given 8 % probability where M28 builds, 2026-10-06)
                builder = b["tokens"][..., 67] > 0                              # can build something now
                lab["uweight"] = 1.0 + (builder_weight - 1.0) * builder.float()
            state = net.initial_state(batch_chunks, device)
            logp, ent, value, belief, loss = net.evaluate(b, lab, state, seq_len=length, starts=starts,
                                                          smoothing=smoothing)
            bc_keep = keep * supervised
            bc_count = bc_keep.sum().clamp_min(1.0)
            bc_loss = (loss * bc_keep).sum() / bc_count
            aux_loss = ((0.5 * (value - ret / ret_std) ** 2
                         + 0.2 * ((belief - b["belief"]) ** 2).mean(-1)) * keep).sum() / keep.sum()
            part = (bc_loss + aux_loss) / accumulate
            part.backward()
            total = total + part.detach()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        stopping = stop_file is not None and step % 25 == 0 and step < steps and Path(stop_file).exists()
        if stopping:
            log(f"step {step}: {Path(stop_file).name} found; measuring and ending the training here")
        if step % eval_every == 0 or step == steps or stopping:
            # Keep completed optimizer work even if validation fails or a later phase is stopped.
            from .validation_checkpoint import checkpoint_then_validate
            _, m = checkpoint_then_validate(
                out.parent / f"{out.stem}_recovery_{step:06d}.pt", net, opt,
                {"size": size, "decoder": decoder, "model_options": model_options(net), "ret_std": ret_std,
                 "dataset_ret_std": dataset_ret_std, "ret_std_source": ret_std_source, "completed_steps": step,
                 "held_out_games": held},
                lambda: evaluate_held_out(net, games, held, length, burn, ret_std, device))
            m.update(step=step, train_loss=float(total.detach()), seconds=round(time.monotonic() - t0, 1),
                     score=score_of(m), selection_loss=m.get("policy_loss"))
            history.append(m)
            if m["selection_loss"] is not None and m["selection_loss"] < best_policy_loss:
                best_policy_loss = float(m["selection_loss"])
                best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            log(f"step {step}: loss {float(total.detach()):.3f}; held-out " + ", ".join(
                f"{k} {m[k]:.3f}" for k in (UNIT_METRIC_KEYS if decoder == "units" else METRIC_KEYS) + ("value_mse",)
                if m.get(k) is not None)
                + f" ({m['seconds']:.0f} s)")
            if stopping:
                break
    out.parent.mkdir(parents=True, exist_ok=True)
    if best is None:
        # nothing measured the held-out games (it should not happen now); the last weights are still the work
        log("WARNING: no held-out measurement selected a checkpoint; keeping the final weights")
        best = {k: v.detach().clone() for k, v in net.state_dict().items()}
        history.append({"step": steps, "selection_loss": math.inf, "policy_loss": None, "score": 0.0})
    state = {"model": best, "size": size, "decoder": decoder, "model_options": model_options(net),
             "ret_std": ret_std, "dataset_ret_std": dataset_ret_std,
             "ret_std_source": ret_std_source, "held_out_games": held, "history": history,
             "best": min((h for h in history if h.get("selection_loss") is not None),
                         key=lambda h: h["selection_loss"]), "params": parameter_count(net),
             "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None}
    torch.save(state, out)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", default=[],
                    help="folder of raw recordings (repeatable; none: train on what the cache holds)")
    ap.add_argument("--cache", required=True, action="append",
                    help="folder of encoded games (repeatable: all are trained on; --raw encodes into the first)")
    ap.add_argument("--out", required=True, help="checkpoint path (one per size in a study: _S, _M, _L)")
    ap.add_argument("--sizes", default="M")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--burn", type=int, default=8)
    ap.add_argument("--batch-chunks", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--init-checkpoint", type=Path, default=None,
                    help="initialize model weights only; optimizer state starts fresh")
    ap.add_argument("--decoder", default="slots", choices=("slots", "units"),
                    help="units: every unit decides for itself (per-unit labels) instead of a list of commands")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loader-workers", type=int, default=0,
                    help="build training batches in this many processes (needs --mapped)")
    ap.add_argument("--mapped", type=Path, default=None,
                    help="read the cached games memory-mapped from this folder (a corpus larger than the RAM)")
    ap.add_argument("--recent-weight", type=float, default=1.0,
                    help="decisions of the games in the caches after the first count this many times")
    ap.add_argument("--builder-weight", type=float, default=1.0,
                    help="units decoder: the lessons of units that can build count this many times")
    ap.add_argument("--accumulate", type=int, default=1,
                    help="batches per update (the memory of one, the lesson of this many)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute the encoder and decoder in the backward pass (about half the memory)")
    ap.add_argument("--stop-file", type=Path, help="end the training early (after a measurement) when this exists")
    ap.add_argument("--memory-fraction", type=float, default=0.9,
                    help="the share of the GPU's memory this process may use (0: no cap)")
    ap.add_argument("--verbs-after-attention", action="store_true",
                    help="units decoder: final verbs from the attended context (model option; kept once set)")
    ap.add_argument("--fine-positions", action="store_true",
                    help="units decoder: the fine pixel inside a chosen cell on large maps (model option)")
    ap.add_argument("--train-only", help="comma-separated parameter name prefixes: train only these (e.g. fine_)")
    a = ap.parse_args()
    caches = [Path(c) for c in a.cache]
    for raw in a.raw:
        build_cache(Path(raw), caches[0])
    games = Games(caches, mapped=a.mapped)
    results = {}
    for size in a.sizes.split(","):
        out = Path(a.out)
        if "," in a.sizes:
            out = out.with_name(f"{out.stem}_{size}{out.suffix}")
        state = train(caches[0], out, size=size, steps=a.steps, length=a.length, burn=a.burn,
                      batch_chunks=a.batch_chunks, eval_every=a.eval_every, games=games,
                      init_checkpoint=a.init_checkpoint, decoder=a.decoder, lr=a.lr,
                      loader_workers=a.loader_workers, recent_weight=a.recent_weight,
                      builder_weight=a.builder_weight, accumulate=a.accumulate, grad_checkpoint=a.grad_checkpoint,
                      stop_file=a.stop_file, memory_fraction=a.memory_fraction or None,
                      verbs_after_attention=True if a.verbs_after_attention else None,
                      fine_positions=True if a.fine_positions else None,
                      train_only=tuple(a.train_only.split(",")) if a.train_only else None)
        results[size] = {"params": state["params"], **state["best"]}
    report = Path(a.out).with_name("tools_bc_report.json")
    report.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
