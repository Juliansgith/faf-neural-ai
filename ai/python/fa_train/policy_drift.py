"""How far a candidate's decisions moved from a base network, on recorded decisions (verify before any acting or
weights change reaches the games; a 'harmless' floor change once cut the win rate from 82 % to 49 %).

    python -m fa_train.policy_drift --base rl_258.pt --candidate arm.pt --cache DIR [--games 12]

For the base network's own recorded actions: the mean log-probability change per decision and per unit, and how the
recorded verbs split. About 0 means the same play. A drop of about 25 per decision was the regression of 2026-10-12.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa_env.tools import VERBS

from .learn_rl import chunk_batch, load_net
from .probability_accounting import cached_semantics
from .tools_bc import Games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=.5)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--candidate-bf16", action="store_true", help="the candidate's encoder in bfloat16: how far its scores are from float32")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    base, _ = load_net(a.base, device)
    cand, _ = load_net(a.candidate, device)
    base.eval(), cand.eval()
    cand.amp = bool(a.candidate_bf16)
    games = Games([Path(a.cache)], newest=a.games)
    picked = list(range(len(games.games)))[:a.games]
    drop, drop_units, decisions, units, largest = 0.0, 0.0, 0, 0, 0.0
    seen = {}
    for gi in picked:
        n = games.lengths[gi]
        state_b = base.initial_state(1, device)
        state_c = cand.initial_state(1, device)
        for start in range(0, n, a.length):
            row = [(gi, start)]
            c = chunk_batch(games, row, a.length, device)
            with torch.no_grad():
                hb, mb, zb, state_b = base.core(c["b"], state_b, a.length, c["starts"], return_state=True)
                hc, mc, zc, state_c = cand.core(c["b"], state_c, a.length, c["starts"], return_state=True)
                _, eb, _ = base.score(hb, mb, zb, c["b"], c["act"], temperature=a.temperature)
                _, ec, _ = cand.score(hc, mc, zc, c["b"], c["act"], temperature=a.temperature)
            known = eb["unit_known"]
            ok = c["valid"] > 0
            d = ((ec["unit_logp"] - eb["unit_logp"]) * known)
            drop += float((d.sum(1) * ok).sum())
            drop_units += float((d.abs().sum(1) * ok).sum())
            largest = max(largest, float((d.abs().max(1).values * ok).max()))
            decisions += int(ok.sum())
            units += int((known.sum(1) * ok).sum())
            verbs = c["act"]["uverb"]
            for v in verbs[known > 0].tolist():
                seen[v] = seen.get(v, 0) + 1
    report = {"decisions": decisions, "units": units,
              "mean_logp_change_per_decision": round(drop / max(decisions, 1), 4),
              "mean_abs_logp_change_per_unit": round(drop_units / max(units, 1), 5),
              "largest_abs_logp_change_of_a_unit": round(largest, 4),
              "verb_share": {VERBS[k]: round(v / max(units, 1), 4) for k, v in sorted(seen.items(), key=lambda x: -x[1])[:12]},
              "semantics": sorted({cached_semantics(games.games[i]) for i in picked})}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
