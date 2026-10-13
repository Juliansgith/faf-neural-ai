"""DAgger: the policy plays, a teacher labels its games, the labelled games join the demonstrations, and the
policy is cloned again from all of them -- round after round.

    python -m fa_train.dagger --run RUN --init CKPT --teacher v3 --base-cache CACHE --mapped MAPPED \
        --rounds 8 --games 64 --steps 1500 --ai easy --maps small --workers 16

A clone only ever saw the states its teacher gets into; one missed order later it is somewhere the teacher never
was, and what it does there is a guess (the first clones: armies never sent, commanders walking to their death).
Here the teacher answers in exactly those states:

  v3   the rule teacher, asked at every decision what it would do (dry: nothing of it is executed)
  m28  M28AI in our slot in shadow mode (fa_env.env.FAEnv(shadow=True)): its orders are recorded, not given

Each round, every step its own process (one GPU process at a time):
  1. examples/tools_eval.py plays --games games per opponent with the current checkpoint (sampled), recorded with
     the teacher's labels, into RUN/raw/round_K. Its outcome table is the round's measurement of the policy.
  2. fa_train.encode_parallel encodes them into RUN/cache.
  3. fa_train.tools_bc clones on from the current checkpoint over the base demonstrations and all rounds' games.
RUN/dagger.jsonl has one line per round. Create RUN/STOP to end after the current step.

Two loops (two teachers, two runs) can share the computer with --turns DIR: only one of them has games running
and only one encodes and trains at a time, so one plays while the other trains.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == 259                                                 # STILL_ACTIVE


@contextmanager
def turn(folder: Path | None, name: str, stop: Path):
    """Hold `folder`/`name`.lock for the block: the loops sharing the folder do this part one at a time. Yields
    False without the turn when `stop` appears while waiting. A lock whose process is gone is taken over."""
    if folder is None:
        yield True
        return
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / f"{name}.lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                owner = int(lock.read_text() or 0)
            except (OSError, ValueError):
                owner = -1                                  # being written: look again
            if owner >= 0 and not _alive(owner):
                lock.unlink(missing_ok=True)
                continue
            if stop.exists():
                yield False
                return
            time.sleep(5)
    try:
        yield True
    finally:
        lock.unlink(missing_ok=True)


def tidy_round(checkpoint: Path, raw: Path, cold: Path | None, run: Path) -> int:
    """After a round's clone is saved: its recovery checkpoints go (they only matter while it trains; ~100 MB a
    round of size M), and with `cold` its raw recordings move to `cold`/<run name>/raw/<round>. Returns the bytes
    freed on this disk. A failure here never fails the round."""
    freed = 0
    for old in checkpoint.parent.glob(f"{checkpoint.stem}_recovery_*.pt"):
        try:
            size = old.stat().st_size
            old.unlink()
            freed += size
        except OSError:
            pass
    if cold is not None and raw.exists():
        target = Path(cold) / run.name / "raw" / raw.name
        try:
            size = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file())
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():                          # a retried round: the newer recordings win
                shutil.rmtree(target)
            shutil.move(str(raw), str(target))
            freed += size
        except OSError as exc:
            print(f"could not move {raw} to {target}: {exc}", flush=True)
    return freed


def run_step(args: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"\n$ {' '.join(args)}\n")
        out.flush()
        return subprocess.call([sys.executable, "-u", *args], cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path, help="the checkpoint to start from (a clone)")
    ap.add_argument("--teacher", required=True, choices=("v3", "m28"))
    ap.add_argument("--base-cache", action="append", default=[], help="encoded demonstrations to keep cloning from")
    ap.add_argument("--mapped", type=Path, help="memory-map the encoded games here (needed for --loader-workers)")
    ap.add_argument("--bin-dir", type=Path)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--games", type=int, default=64, help="games per opponent and round")
    ap.add_argument("--ai", default="easy")
    ap.add_argument("--maps", default="small")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7, help="the policy's arguments are sampled at this")
    ap.add_argument("--steps", type=int, default=1500, help="cloning steps per round")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loader-workers", type=int, default=4)
    ap.add_argument("--encode-workers", type=int, default=8)
    ap.add_argument("--recent-weight", type=float, default=3.0,
                    help="the rounds' games count this many times next to the base demonstrations")
    ap.add_argument("--turns", type=Path, help="a folder shared with other loops: games and training take turns")
    ap.add_argument("--builder-weight", type=float, default=1.0, help="tools_bc --builder-weight")
    ap.add_argument("--batch-chunks", type=int, default=16, help="tools_bc --batch-chunks")
    ap.add_argument("--accumulate", type=int, default=1, help="tools_bc --accumulate")
    ap.add_argument("--grad-checkpoint", action="store_true", help="tools_bc --grad-checkpoint")
    ap.add_argument("--cold", type=Path,
                    help="after a round is encoded and cloned, move its raw recordings here (another disk): they "
                         "are only read again to re-encode after a format change")
    a = ap.parse_args()
    run = a.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "STOP").unlink(missing_ok=True)
    journal, cache = run / "dagger.jsonl", run / "cache"
    done = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    done = [row for row in done if row.get("train_exit") == 0 and row.get("checkpoint")]    # the finished rounds
    checkpoint = Path(done[-1]["checkpoint"]) if done else a.init.resolve()
    saved = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    size, decoder = saved.get("size", "M"), saved.get("decoder", "slots")
    del saved
    for k in range(len(done) + 1, a.rounds + 1):
        if (run / "STOP").exists():
            break
        t0, row = time.monotonic(), {"round": k, "played_by": str(checkpoint)}
        raw, collect, log = run / "raw" / f"round_{k:02d}", run / f"collect_{k:02d}", run / f"round_{k:02d}.log"
        # 1. the current policy plays; the teacher labels
        play = ["examples/tools_eval.py", "--checkpoint", str(checkpoint), "--run", str(collect), "--ai", a.ai,
                "--games", str(a.games), "--maps", a.maps, "--factions", "1,2,3,4", "--ai-factions", "1,2,3,4",
                "--minutes", str(a.minutes), "--workers", str(a.workers), "--seed-base", str(700000 + 1000 * k),
                "--mode", "sample", "--temperature", str(a.temperature), "--record", str(raw), "--labels", a.teacher]
        if a.bin_dir:
            play += ["--bin-dir", str(a.bin_dir)]
        results = collect / "eval_results.json"
        if results.exists() and any(raw.glob("*.jsonl.gz")):
            print(f"round {k}: its games were played before (stopped ahead of cloning): continuing with them")
        else:
            with turn(a.turns, "games", run / "STOP") as mine:
                if not mine:
                    break
                row["play_exit"] = run_step(play, log)
        if results.exists():
            row["results"] = json.loads(results.read_text(encoding="utf-8")).get("summary")
        print(f"round {k}: {checkpoint.name} played: {json.dumps(row.get('results'))}", flush=True)
        if (run / "STOP").exists() or not any(raw.glob("*.jsonl.gz")):
            journal.open("a", encoding="utf-8").write(json.dumps(row) + "\n")
            break
        out = run / "checkpoints" / f"round_{k:02d}.pt"
        train = ["-m", "fa_train.tools_bc", "--out", str(out), "--sizes", size, "--decoder", decoder,
                 "--steps", str(a.steps), "--eval-every", str(max(100, a.steps // 3)), "--lr", str(a.lr),
                 "--init-checkpoint", str(checkpoint), "--stop-file", str(run / "STOP"),
                 "--recent-weight", str(a.recent_weight),
                 "--builder-weight", str(a.builder_weight), "--batch-chunks", str(a.batch_chunks),
                 "--accumulate", str(a.accumulate)] + (["--grad-checkpoint"] if a.grad_checkpoint else [])
        for c in [*a.base_cache, str(cache)]:
            train += ["--cache", c]
        if a.mapped:
            train += ["--mapped", str(a.mapped), "--loader-workers", str(a.loader_workers)]
        for old in out.parent.glob(f"{out.stem}_recovery_*.pt"):      # of an interrupted try at this round
            old.unlink()
        with turn(a.turns, "train", run / "STOP") as mine:
            if not mine:
                break
            # 2. its games become demonstrations
            row["encode_exit"] = run_step(["-m", "fa_train.encode_parallel", "--raw", str(raw), "--cache",
                                           str(cache), "--workers", str(a.encode_workers)], log)
            # 3. clone on, from everything
            row["train_exit"] = run_step(train, log)
        report = out.with_name("tools_bc_report.json")
        if out.exists():
            checkpoint = out
            if report.exists():
                best = json.loads(report.read_text(encoding="utf-8")).get(size) or {}
                row["held_out"] = {key: round(v, 3) for key, v in best.items() if isinstance(v, float)}
            row["freed_mb"] = round(tidy_round(out, raw, a.cold, run) / 2 ** 20)
        row.update(checkpoint=str(checkpoint), minutes=round((time.monotonic() - t0) / 60, 1))
        with journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"round {k}: cloned on to {checkpoint.name} in {row['minutes']} min "
              f"(held out: {json.dumps(row.get('held_out'))})", flush=True)
        if not out.exists():
            print(f"round {k}: cloning failed (see {log.name}); stopping", flush=True)
            break


if __name__ == "__main__":
    main()
