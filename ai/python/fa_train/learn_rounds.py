"""The learner's step for games played elsewhere (fa_train.player on the server, pulled with fa_train.server_sync):
encode the new rounds, clone on from the current network over everything, and (with --push) send the result back.
The DAgger loop (fa_train.dagger) without its playing step: the player's network may be hours old, which imitation
of a teacher's labels does not mind.

    python -m fa_train.learn_rounds --run RUN --init CKPT --rounds D:/faf-server/rounds \
        --base-cache CACHE ... --mapped F:/faf-nn-cache/m28_v10_mapped --steps 3000 [--pull] [--push]

RUN/learn.jsonl has one line per step (rounds and games used, held-out metrics, the checkpoint); RUN/STOP ends a
step early (tools_bc saves what it has).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .dagger import run_step, turn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path, help="the network to start from (before the first step)")
    ap.add_argument("--from-checkpoint", type=Path,
                    help="start this step from this network instead of the last one trained (a quality gate kept "
                         "an earlier one)")
    ap.add_argument("--rounds", type=Path, default=Path("D:/faf-server/rounds"))
    ap.add_argument("--base-cache", action="append", default=[])
    ap.add_argument("--mapped", type=Path)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--recent-weight", type=float, default=3.0)
    ap.add_argument("--batch-chunks", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=2)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--verbs-after-attention", action="store_true", help="tools_bc --verbs-after-attention")
    ap.add_argument("--loader-workers", type=int, default=8)
    ap.add_argument("--encode-workers", type=int, default=12)
    ap.add_argument("--turns", type=Path, help="share the GPU with the DAgger loops' turn locks")
    ap.add_argument("--pull", action="store_true", help="pull finished rounds from the player first")
    ap.add_argument("--push", action="store_true", help="push the new network to the player afterwards")
    ap.add_argument("--host", default="my-server")
    a = ap.parse_args()
    run = a.run.resolve()
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "STOP").unlink(missing_ok=True)
    journal, cache = run / "learn.jsonl", run / "cache"
    done = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    if a.pull:
        from .server_sync import pull
        pull(a.host, "~/fa/player", a.rounds, keep=False)
    encoded = {name for row in done for name in row.get("rounds", [])}
    new = sorted(p.name for p in a.rounds.glob("lane*_*") if (p / "DONE").exists() and p.name not in encoded)
    finished = [row for row in done if row.get("train_exit") == 0 and row.get("checkpoint")]
    checkpoint = Path(finished[-1]["checkpoint"]) if finished else a.init.resolve()
    if a.from_checkpoint:
        checkpoint = a.from_checkpoint.resolve()
    k = len(done) + 1
    row = {"step": k, "from": str(checkpoint), "rounds": new}
    log = run / f"learn_{k:02d}.log"
    if not new:
        print("no new rounds; training on what is there")
    t0 = time.monotonic()
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    size, decoder = saved.get("size", "M"), saved.get("decoder", "units")
    del saved
    with turn(a.turns, "train", run / "STOP") as mine:
        if not mine:
            return
        for name in new:                                  # 1. the new rounds become demonstrations
            code = run_step(["-m", "fa_train.encode_parallel", "--raw", str(a.rounds / name / "raw"), "--cache",
                             str(cache), "--workers", str(a.encode_workers)], log)
            if code != 0:
                print(f"encoding {name} failed (see {log.name}); stopping")
                return
        row["games"] = sum(len(list((a.rounds / n / "raw").glob("*.jsonl.gz"))) for n in new)
        out = run / "checkpoints" / f"step_{k:02d}.pt"   # 2. clone on, from everything
        train = ["-m", "fa_train.tools_bc", "--out", str(out), "--sizes", size, "--decoder", decoder,
                 "--steps", str(a.steps), "--eval-every", str(max(100, a.steps // 3)), "--lr", str(a.lr),
                 "--init-checkpoint", str(checkpoint), "--stop-file", str(run / "STOP"),
                 "--recent-weight", str(a.recent_weight), "--batch-chunks", str(a.batch_chunks),
                 "--accumulate", str(a.accumulate)] + (["--grad-checkpoint"] if a.grad_checkpoint else [])             + (["--verbs-after-attention"] if a.verbs_after_attention else [])
        for c in [*a.base_cache, str(cache)]:
            train += ["--cache", c]
        if a.mapped:
            train += ["--mapped", str(a.mapped), "--loader-workers", str(a.loader_workers)]
        row["train_exit"] = run_step(train, log)
    report = out.with_name("tools_bc_report.json")
    if out.exists():
        row["checkpoint"] = str(out)
        if report.exists():
            best = json.loads(report.read_text(encoding="utf-8")).get(size) or {}
            row["held_out"] = {key: round(v, 3) for key, v in best.items() if isinstance(v, float)}
        for old in out.parent.glob(f"{out.stem}_recovery_*.pt"):
            old.unlink(missing_ok=True)
    row["minutes"] = round((time.monotonic() - t0) / 60, 1)
    with journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"step {k}: {len(new)} rounds, {row.get('games', 0)} games -> {out.name if out.exists() else 'FAILED'} "
          f"in {row['minutes']} min (held out: {json.dumps(row.get('held_out'))})", flush=True)
    if a.push and out.exists():
        from .server_sync import push
        push(a.host, "~/fa/player", out)


if __name__ == "__main__":
    sys.exit(main())
