"""Observation drift between two versions of the observer, on recorded games (CPU only, read-only).

    python tools/drift/obs_drift.py --old 3d3ef9f --new WORKTREE --net rl_089.pt  <raw folder or games...>

Each game (a raw recording, <uid>_<opponent>_<seed>.jsonl.gz, with blueprints.json.gz beside it) is replayed through
two ToolsObservers in lockstep -- `fa_env` as it is at git ref `--old` and at `--new` (WORKTREE: the files on disk)
-- with the bookkeeping fa_train.tools_bc.encode_game does (reclaim memory, claimed deposits, the acting side's
commands after each observation). Reported per observation key: how many values differ, and in how many decisions.
Tokens are matched by unit id (a token only one side has is counted apart), so a reordering is not a change.

With `--net` (a units-decoder checkpoint), the recorded choices of the network that played (step["policy"]) are
scored under both observations, the memory carried through the game: the old observer must reproduce the recorded
log-probabilities (a check of the replay), and the mean drop of the log-probability of the choices actually made is
how much the change moves that network's decisions. Choices that point at a token the new observation has not got
are counted, not scored.

The 2026-10-17 review fixes (recycled ids, civilian ghosts, token caps, silo mask, dropped commands) were measured
with it, one commit against the previous one (docs: the commit messages and the review's notes).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PY = REPO / "ai" / "python"
sys.path.insert(0, str(PY))

PER_STEP = ("map", "scalars", "mask_verb", "mask_build_cell", "mask_move_cell", "fine", "focus", "privileged", "belief")
PER_TOKEN = ("tokens", "token_bp", "token_order", "token_type", "token_last", "token_last_feat", "mask_anchor",
             "mask_silo", "mask_toggle")
PER_TOKEN_T = ("mask_subject", "mask_target")              # [rows, tokens]


def load_fa_env(ref: str, tmp: Path):
    """The fa_env package at git `ref` (WORKTREE: the working copy) as its own package, fa_env_<n>."""
    if ref == "WORKTREE":
        root = PY / "fa_env"
    else:
        root = tmp / f"fa_env_{ref}"
        root.mkdir(parents=True, exist_ok=True)
        names = subprocess.run(["git", "-C", str(REPO), "ls-tree", "--name-only", ref, "ai/python/fa_env/"],
                               capture_output=True, text=True, check=True).stdout.split()
        for name in names:
            data = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{name}"], capture_output=True,
                                  check=True).stdout
            (root / Path(name).name).write_bytes(data)
    pkg = f"fa_env_{abs(hash((ref, str(root)))) % 10 ** 8}"
    spec = importlib.util.spec_from_file_location(pkg, root / "__init__.py", submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    return (importlib.import_module(f"{pkg}.tools"), importlib.import_module(f"{pkg}.tools_obs"),
            importlib.import_module(f"{pkg}.reclaim_memory"))


class Side:
    """One observer version replaying a game."""

    def __init__(self, pkg, info: dict, bps: dict, me: int):
        tools, tools_obs, reclaim = pkg
        self.kit = tools.ToolKit(info, bps)
        self.ob = tools_obs.ToolsObserver(info, bps, me, self.kit)
        self.after_step = reclaim.after_step

    def observe(self, obs: dict) -> dict:
        return self.ob.observe(obs)

    def after(self, s: dict) -> None:
        """What tools_bc.encode_game does after a decision's observation (a policy game's bookkeeping)."""
        self.ob.set_wrecks(self.after_step(self.ob.wrecks, s.get("queries") or [], s.get("answers") or [],
                                           s.get("sent") or [], s.get("results") or []))
        for a in s.get("sent") or []:
            if a.get("type") == "build" and self.kit.role.get(a.get("bp")) in ("mex", "hydro"):
                self.ob.claim(a["pos"][0], a["pos"][1], s["tick"])
        self.ob.note_commands(s.get("commands") or [], s["tick"], s.get("results"))


def token_keys(ids: list) -> list:
    """A key per token: the unit id, else (None, k) for the k-th id-less token (deposit spots, in order)."""
    out, k = [], 0
    for i in ids:
        if i is None:
            out.append((None, k))
            k += 1
        else:
            out.append(i)
    return out


def compare(a: dict, b: dict, acc: dict) -> bool:
    """Count differing values of one decision into `acc`; True when anything differs."""
    any_diff = False

    def note(key, n_vals, n_diff):
        nonlocal any_diff
        e = acc.setdefault(key, [0, 0, 0])          # values compared, values changed, decisions with a change
        e[0] += n_vals
        e[1] += n_diff
        if n_diff:
            e[2] += 1
            any_diff = True

    for key in PER_STEP:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape:
            note(key, x.size, x.size)
        else:
            note(key, x.size, int(np.count_nonzero(~np.isclose(x, y, rtol=0, atol=1e-6))))
    ka, kb = token_keys(a["ids"]), token_keys(b["ids"])
    pos_b = {k: j for j, k in enumerate(kb)}
    common = [(i, pos_b[k]) for i, k in enumerate(ka) if k in pos_b]
    note("tokens_only_old", len(ka), len(ka) - len(common))
    note("tokens_only_new", len(kb), len(kb) - len(common))
    note("token_order", len(common), sum(1 for i, j in common if i != j))
    ia = np.array([i for i, _ in common], np.int64)
    ib = np.array([j for _, j in common], np.int64)
    for key in PER_TOKEN:
        x, y = np.asarray(a[key])[ia], np.asarray(b[key])[ib]
        note(key, x.size, int(np.count_nonzero(~np.isclose(x.astype(np.float64), y.astype(np.float64),
                                                             rtol=0, atol=1e-6))))
    for key in PER_TOKEN_T:
        x, y = np.asarray(a[key])[:, ia], np.asarray(b[key])[:, ib]
        note(key, x.size, int(np.count_nonzero(x != y)))
    for key in ("cand_ids", "enh_names"):
        note(key, 1, int(list(a[key]) != list(b[key])))
    return any_diff


# ---------------------------------------------------------------------- log-probabilities of the recorded choices
class Scorer:
    """rl_089-style scoring: memory carried through the game, chunks of `length` decisions."""

    def __init__(self, path: Path, threads: int, length: int = 32):
        import torch
        torch.set_num_threads(threads)
        from fa_train.learn_rl import load_net
        self.torch = torch
        self.net, _ = load_net(path, torch.device("cpu"))
        self.net.eval()
        self.length = length

    def initial(self):
        return self.net.initial_state(1, self.torch.device("cpu"))

    def chunk(self, xs: list, acts: list, temperature: float, state, first: bool):
        """Per-unit log-probabilities of `acts` (rows of UNIT_KEYS per token) under consecutive observations `xs`
        of one game, the memory carried in `state`: (list of [tokens] arrays, the state after them)."""
        from fa_env.observation_precision import widen_observation
        from fa_train.tools_model import collate, stack_actions
        torch = self.torch
        with torch.no_grad():
            b = collate([widen_observation(x) for x in xs], torch.device("cpu"))
            starts = torch.zeros(len(xs), dtype=torch.bool)
            starts[0] = first
            h, m, z, state = self.net.core(b, state, len(xs), starts, return_state=True)
            T = b["tokens"].shape[1]
            act = {k: torch.as_tensor(v) for k, v in stack_actions(acts, T, "units").items()}
            _, ent, _ = self.net.score(h, m, z, b, act, temperature=temperature)
            lp = ent["unit_logp"].float().numpy()
        return [lp[i, : len(xs[i]["ids"])] for i in range(len(xs))], state


def recorded_rows(policy: dict | None, n: int) -> dict | None:
    """The network's recorded choice as per-token rows (tools_bc.policy_arrays), or None."""
    from fa_train.tools_model import UNIT_KEYS
    if not policy or not isinstance(policy.get("action"), dict) or len(policy.get("unit_logp") or []) != n:
        return None
    row = {k: np.full(n, 0 if k == "uverb" else -1, np.int64) for k in UNIT_KEYS}
    for entry in policy["action"].get("units") or []:
        for k, v in zip(UNIT_KEYS, entry[1:]):
            row[k][entry[0]] = v
    return row


def remap(row: dict, ids_old: list, ids_new: list) -> dict | None:
    """The same choice on the new observation's tokens (the subject, anchor and target by unit id); None when a
    pointed-at token is missing there."""
    from fa_train.tools_model import UNIT_KEYS
    pos = {k: j for j, k in enumerate(token_keys(ids_new))}
    ko = token_keys(ids_old)
    out = {k: np.full(len(ids_new), 0 if k == "uverb" else -1, np.int64) for k in UNIT_KEYS}
    for i in np.flatnonzero(row["uverb"] > 0):
        j = pos.get(ko[i])
        if j is None:
            return None
        for k in UNIT_KEYS:
            v = int(row[k][i])
            if k in ("uanchor", "utarget") and v >= 0:
                t = pos.get(ko[v])
                if t is None:
                    return None
                v = t
            out[k][j] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("games", nargs="+", type=Path, help="raw folders or .jsonl.gz recordings")
    ap.add_argument("--old", default="3d3ef9f", help="git ref of the old observer")
    ap.add_argument("--new", default="WORKTREE", help="git ref of the new observer (WORKTREE: files on disk)")
    ap.add_argument("--net", type=Path, help="checkpoint whose recorded choices to score (the one that played)")
    ap.add_argument("--max-steps", type=int, default=0, help="only the first N decisions of each game (0: all)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", type=Path, help="write the totals here")
    a = ap.parse_args()
    from fa_env.record import read_blueprints, read_game
    from fa_train.tools_model import UNIT_KEYS

    paths = []
    for g in a.games:
        paths += sorted(g.glob("*.jsonl.gz")) if g.is_dir() else [g]
    tmp = Path(tempfile.mkdtemp(prefix="obs_drift_"))
    old_pkg, new_pkg = load_fa_env(a.old, tmp), load_fa_env(a.new, tmp)
    scorer = Scorer(a.net, a.threads) if a.net else None
    acc: dict = {}
    totals = {"games": 0, "decisions": 0, "decisions_changed": 0}
    lp = {"decisions": 0, "scored": 0, "unmapped": 0, "replay_err": [], "drop": [], "drop_ordered": [],
          "units_ordered": 0}
    for path in paths:
        t0 = time.monotonic()
        hello, steps, end = read_game(path)
        if not steps:
            continue
        if a.max_steps:
            steps = steps[: a.max_steps]
        bps = {b["id"]: b for b in read_blueprints(path.parent)}
        info, me = hello["hello"], hello["meta"]["me"]
        old, new = Side(old_pkg, info, bps, me), Side(new_pkg, info, bps, me)
        changed = 0
        buf: list = []                  # (old obs, new obs, old row, new row or None, recorded unit logp or None)
        states = [scorer.initial(), scorer.initial()] if scorer is not None else None
        first = [True]
        temp = [None]

        def flush():
            if not buf:
                return
            blank = lambda n: {k: np.full(n, 0 if k == "uverb" else -1, np.int64) for k in UNIT_KEYS}  # noqa: E731
            t = float(temp[0] or 1.0)
            lo, states[0] = scorer.chunk([e[0] for e in buf], [e[2] if e[2] is not None else blank(len(e[0]["ids"]))
                                                               for e in buf], t, states[0], first[0])
            ln, states[1] = scorer.chunk([e[1] for e in buf], [e[3] if e[3] is not None else blank(len(e[1]["ids"]))
                                                               for e in buf], t, states[1], first[0])
            first[0] = False
            for (xo, xn, ro, rn, rec), a_, b_ in zip(buf, lo, ln):
                if rec is None:
                    continue
                lp["decisions"] += 1
                lp["replay_err"].append(float(np.abs(a_ - rec).max()) if len(a_) else 0.0)
                if rn is None:
                    lp["unmapped"] += 1
                    continue
                # the decision's log-probability over the tokens both observations have (own tokens with a
                # choice, every other token's "nothing"; tokens only one side has are left out)
                kn = {k: j for j, k in enumerate(token_keys(xn["ids"]))}
                pairs = [(i, kn[k]) for i, k in enumerate(token_keys(xo["ids"])) if k in kn]
                to = np.array([p[0] for p in pairs], np.int64)
                tn = np.array([p[1] for p in pairs], np.int64)
                lp["scored"] += 1
                lp["drop"].append(float(a_[to].sum() - b_[tn].sum()))
                ordered = ro["uverb"][to] > 0
                lp["units_ordered"] += int(ordered.sum())
                lp["drop_ordered"] += list((a_[to] - b_[tn])[ordered])
            buf.clear()

        for st in steps:
            s, o = st["step"], st["msg"]["obs"]
            xo, xn = old.observe(o), new.observe(o)
            changed += compare(xo, xn, acc)
            if scorer is not None:
                pol = s.get("policy") or {}
                if buf and pol.get("temperature") != temp[0]:
                    flush()
                temp[0] = pol.get("temperature")
                row = recorded_rows(pol, len(xo["ids"]))
                rec = None if row is None else np.asarray(pol["unit_logp"], np.float32)
                buf.append((xo, xn, row, remap(row, xo["ids"], xn["ids"]) if row is not None else None, rec))
                if len(buf) >= scorer.length:
                    flush()
            old.after(s)
            new.after(s)
        if scorer is not None:
            flush()
        totals["games"] += 1
        totals["decisions"] += len(steps)
        totals["decisions_changed"] += changed
        msg = f"{path.name}: {len(steps)} decisions, {changed} with a changed observation"
        if scorer is not None and lp["replay_err"]:
            msg += f"; replay max |logp err| so far {max(lp['replay_err']):.5f}"
        print(f"{msg}  ({time.monotonic() - t0:.0f} s)", flush=True)

    print(f"\n{totals['games']} games, {totals['decisions']} decisions, {totals['decisions_changed']} "
          f"({100.0 * totals['decisions_changed'] / max(1, totals['decisions']):.1f} %) with any change")
    print(f"{'key':18s} {'values':>12s} {'changed':>10s} {'%':>8s} {'decisions':>9s}")
    for key, (n, d, k) in sorted(acc.items()):
        print(f"{key:18s} {n:12d} {d:10d} {100.0 * d / max(1, n):8.4f} {k:9d}")
    out = {"totals": totals, "keys": acc}
    if scorer is not None and lp["decisions"]:
        drop = np.asarray(lp["drop"], np.float64)
        per_unit = np.asarray(lp["drop_ordered"], np.float64)
        res = {"decisions": lp["decisions"], "scored": lp["scored"], "unmapped": lp["unmapped"],
               "replay_max_err": float(max(lp["replay_err"])),
               "replay_mean_err": float(np.mean(lp["replay_err"])),
               "mean_logp_drop_per_decision": float(drop.mean()) if len(drop) else 0.0,
               "mean_abs_logp_change_per_decision": float(np.abs(drop).mean()) if len(drop) else 0.0,
               "decisions_moved_over_0.01": int((np.abs(drop) > 0.01).sum()),
               "ordered_units": lp["units_ordered"],
               "mean_logp_drop_per_ordered_unit": float(per_unit.mean()) if len(per_unit) else 0.0,
               "ordered_units_moved_over_0.1": int((np.abs(per_unit) > 0.1).sum())}
        print("\nrecorded choices under old vs new observations:")
        for k, v in res.items():
            print(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")
        out["logp"] = res
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
