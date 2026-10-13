"""Encode raw recordings into the BC cache with several processes (tools_bc.build_cache does one game at a time).

    python -m fa_train.encode_parallel --raw RAW [--raw RAW2 ...] --cache CACHE [--workers 12] [--limit N] [--glob G]

Each recording is encoded once into CACHE (files are named by source and content, so folders with equal file
names do not collide); existing current files are skipped. Prints the summed drop diagnostics at the end."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

from .tools_bc import _add_game_stats, _new_command_diagnostics, build_cache


def _work(job: tuple) -> dict:
    raw, cache, files = job
    return build_cache(Path(raw), Path(cache), log=lambda *_: None, files=[Path(f) for f in files])


def encode(raws: list[Path], cache: Path, workers: int = 12, limit: int | None = None, seed: int = 0,
           pattern: str = "*.jsonl.gz", log=print) -> dict:
    jobs = []
    for raw in raws:
        files = sorted({f for part in pattern.split(",") for f in raw.glob(part)})
        if limit is not None:
            random.Random(seed).shuffle(files)
            files = files[:limit]
        jobs += [(str(raw), str(cache), [str(f)]) for f in files]
    totals = {"games": 0, "skipped": 0, "steps": 0, "commands": 0, "dropped": 0, "kept": 0,
              "command_diagnostics": _new_command_diagnostics("recording_bc")}
    t0 = time.monotonic()
    # one recording per task, largest first, so a long game does not finish last on its own
    jobs.sort(key=lambda j: -Path(j[2][0]).stat().st_size)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for i, part in enumerate(pool.imap_unordered(_work, jobs), 1):
            totals["games"] += part["games"]
            totals["skipped"] += part["skipped"]
            _add_game_stats(totals, part)
            if i % 50 == 0 or i == len(jobs):
                log(f"{i}/{len(jobs)} recordings, {time.monotonic() - t0:.0f} s; kept {totals['kept']} of "
                    f"{totals['commands']} commands")
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, help="a random sample of this many recordings per folder")
    ap.add_argument("--glob", default="*.jsonl.gz", help="which recordings: a pattern or a comma list, e.g. '*_m28ai_*.jsonl.gz'")
    a = ap.parse_args()
    totals = encode([Path(r) for r in a.raw], Path(a.cache), a.workers, a.limit, pattern=a.glob)
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
