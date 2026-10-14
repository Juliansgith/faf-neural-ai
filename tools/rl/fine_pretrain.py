"""Teach a network's new fine head and close-ups from M28's exact positions before it plays with them.

    python fine_pretrain.py --code WORKTREE/ai/python --init rl2/checkpoints/rl_NNN.pt --out OUT.pt
        [--rounds rl2/rounds] [--since 20261012T1048] [--steps 1500] [--select-only]

Picks the large-map recordings with the shadow AI's raw orders (rounds started at or after --since, UTC as in
the round names), encodes them with --code's encoder into _runs/fine_cache, trains only the fine_
weights (tools_bc --train-only; the rest of the network stays as it is; never the close-ups: trained with the full
imitation loss they pulled every decision back to M28, rl_079) with fine positions on, and writes OUT:
the --init checkpoint with the trained weights and the option, everything else (learn_rl's history, return
scale, reward spec) kept."""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

FAF = Path(os.environ.get("FAF_ROOT", r"C:\faf"))   # the workspace (README: Setup)


def has_shadow(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if '"shadow"' in line:
                    return True
                if i > 400:
                    return False
    except (OSError, EOFError):
        return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=Path, action="append",
                    help="rounds folders (default: rl2's and the local player's)")
    ap.add_argument("--since", default="20261012T1048")
    ap.add_argument("--cache", type=Path, default=FAF / "_runs" / "fine_cache")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-chunks", type=int, default=4, help="chunks of 40 decisions per step (large-map battles "
                    "are big: 16 ran a 24 GB GPU out of memory)")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--max-games", type=int, default=80,
                    help="the newest this many recordings (all large-map games in memory at once: 411 would not fit)")
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    from fa_env.maps import MAP_POOL

    folders = a.rounds or [FAF / "_runs" / "server" / "rl2" / "rounds", FAF / "_runs" / "local_player" / "rounds"]
    picks = []
    for r in sorted(r for d in folders for r in d.glob("lane*_*")):
        stamp = r.name.split("_", 1)[1]
        if stamp < a.since or not (r / "DONE").exists() or not (r / "eval" / "episodes.jsonl").exists():
            continue
        for line in (r / "eval" / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line) if line.strip() else None
            if not e or MAP_POOL.get(e.get("map"), {}).get("km", 5) <= 10:
                continue
            for f in (r / "raw").glob(f"*_{e['ai']}_{e['seed']}.jsonl.gz"):
                if has_shadow(f):
                    picks.append(f)
    picks = sorted(picks, key=lambda f: f.parent.parent.name.split("_", 1)[1])[-a.max_games:]
    print(f"{len(picks)} large-map recordings with the shadow AI's orders (the newest)", flush=True)
    if a.select_only or not picks:
        return
    import torch
    from fa_train.tools_bc import Games, train
    by_dir = {}
    for f in picks:
        by_dir.setdefault(f.parent, []).append(f)
    import subprocess
    for d, files in by_dir.items():                     # encode_parallel: a game is ~40 s on one core
        subprocess.run([sys.executable, "-m", "fa_train.encode_parallel", "--raw", str(d), "--cache", str(a.cache),
                        "--workers", "8", "--glob", ",".join(f.name for f in files)], cwd=str(a.code), check=True)
    games = Games([a.cache], newest=len(picks))
    tmp = a.out.with_name(a.out.stem + "_bc.pt")
    state = train(a.cache, tmp, size=torch.load(a.init, map_location="cpu", weights_only=False).get("size", "M"),
                  steps=a.steps, games=games, init_checkpoint=a.init, decoder="units", lr=a.lr,
                  eval_every=max(100, a.steps // 6), fine_positions=True, train_only=("fine_",),
                  grad_checkpoint=True, batch_chunks=a.batch_chunks)
    trained = torch.load(tmp, map_location="cpu", weights_only=False)
    out = torch.load(a.init, map_location="cpu", weights_only=False)
    out["model"] = trained["model"]
    out["model_options"] = {**(out.get("model_options") or {}), "fine_positions": True}
    out["fine_pretrain"] = {"games": len(picks), "steps": a.steps, "best": state.get("best"), "from": str(a.init)}
    torch.save(out, a.out)
    print(json.dumps(out["fine_pretrain"], default=str)[:2000], flush=True)


if __name__ == "__main__":
    main()
