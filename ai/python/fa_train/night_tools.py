"""The unattended night pipeline for the full-toolbox network (docs/full-toolbox-design.md; the tools version
before it: docs/tools-design.md):

  1. collect  the teacher v3 plays (all four factions; easy, normal and the four personalities), every game
              recorded raw to <run>/raw (fa_train/collect_v3.py)
  2. clone    the network learns the teacher's commands from all recordings in --raw folders
  3. ppo      recurrent PPO from the clone on the opponent ladder (tools_trainer.LADDER), all four factions,
              online teacher distillation fading to --distill-end (0: the teacher is only a tutorial)
  4. final    greedy evaluation against normal and the rung reached
  5. report   <run>/report.md

    python -m fa_train.night_tools --run C:/faf/_runs/night-%Y%m%d --workers 20 --stop-at 07:00

Stops at --stop-at or when <run>/STOP exists; a failed phase is logged and the night goes on; Windows is
asked not to sleep. --clone skips collection and cloning; --resume continues PPO from a checkpoint;
--collect-games 0 skips collection (clone from --extra-raw folders; --reuse-cache copies their encoded games).
Before learning, the clone itself is evaluated (--baseline-evals), so the report shows where learning started.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train.night import (Tee, by_ai, keep_awake, parse_stop, raise_learner_priority,  # noqa: E402
                            write_report)
from fa_train.tools_bc import Games, build_cache, current_cache_paths, train as train_bc  # noqa: E402
from fa_train.collect_v3 import collect  # noqa: E402
from fa_train.tools_trainer import LADDERS, ToolsConfig, ToolsTrainer, stage_index  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder; strftime codes allowed (e.g. _runs/night-%%Y%%m%%d)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--stop-at", required=True, help="HH:MM (local time; tomorrow if already past)")
    ap.add_argument("--size", default="M", choices=("S", "M", "L"))
    ap.add_argument("--collect-games", type=int, default=200)
    ap.add_argument("--collect-hours", type=float, default=1.0)
    ap.add_argument("--collect-seed", type=int, default=7,
                    help="collection world-seed namespace; choose a new value for a fresh corpus")
    ap.add_argument("--m28-share", type=float, default=0.0,
                    help="share of collection games M28 plays our slot, its orders captured as labels")
    ap.add_argument("--extra-raw", action="append", default=[], help="more recording folders to clone from")
    ap.add_argument("--reuse-cache", action="append", default=[],
                    help="encoded games (tools_cache folders) to copy instead of encoding their recordings again")
    ap.add_argument("--temperature", type=float, default=0.5, help="training games act at this temperature")
    ap.add_argument("--distill-end", type=float, default=0.0, help="teacher-distillation weight it fades to")
    ap.add_argument("--distill-teacher", choices=("v3", "m28", "none"), default="v3",
                    help="online PPO teacher (m28: M28AI watches in shadow mode); none disables online distillation")
    ap.add_argument("--distill-start", type=float, default=1.0, help="initial teacher-distillation weight")
    ap.add_argument("--distill-fraction", type=float, default=1 / 3,
                    help="fraction of planned PPO updates over which the teacher weight fades")
    ap.add_argument("--entropy-bonus", type=float, default=0.0,
                    help="entropy bonus at the start (fades to a third of it); none by default: sampling at "
                         "--temperature explores already")
    ap.add_argument("--baseline-evals", default="easy:4,medium:4,adaptive:2,turtle:2",
                    help="greedy games of the clone before learning, as ai:games,... ('' for none)")
    ap.add_argument("--bc-steps", type=int, default=6000)
    ap.add_argument("--clone", help="skip collection and cloning; start PPO from this tools checkpoint")
    ap.add_argument("--resume", help="continue PPO from this tools checkpoint")
    ap.add_argument("--final-games", type=int, default=16)
    ap.add_argument("--batch", type=int, default=3072)
    ap.add_argument("--segment", type=int, default=32)
    ap.add_argument("--decision-ticks", type=int, default=20,
                    help="simulation ticks per tools decision in PPO and evaluations; match demonstrations explicitly")
    ap.add_argument("--lr", type=float, default=3e-5)   # a sharp policy: 1e-4 overshoots the KL target
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=45.0, help="time limit of games on the single --map")
    ap.add_argument("--single-map", action="store_true", help="Canis River only (default: the map pool, fa_env.maps, "
                                                                 "with time limits by map size)")
    ap.add_argument("--eval-minutes", type=float, default=45.0)
    ap.add_argument("--dagger-updates", type=int, default=0,
                    help="first updates: only imitate the online teacher on the policy's own games (no PPO yet)")
    ap.add_argument("--dagger-lr", type=float, default=2e-4)
    ap.add_argument("--decoder", default="slots", choices=("slots", "units"),
                    help="units: every unit decides for itself (taken from --clone / --resume when given)")
    ap.add_argument("--maps", default="", help="only these pool maps: a comma list, or 'small' (the 5 and 10 km maps)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="cap on the pool's time limits by map size")
    ap.add_argument("--scale", type=float, default=1.0, help="phase time budgets x this (short tests)")
    ap.add_argument("--fake-raw", help="machinery test: tools episodes replay these recordings, no games")
    # the RL phase after DAgger (2026-10-08): the ladder toward M28, a frozen anchor, factored PPO
    ap.add_argument("--ladder", default="aix", choices=sorted(LADDERS),
                    help="the opponent ladder: aix (FAF's cheating personalities) or m28 (M28, then M28 with multipliers)")
    ap.add_argument("--start-stage", help="start at this rung (its name, or index) instead of the first")
    ap.add_argument("--reference", help="frozen KL-anchor checkpoint (default: the --clone checkpoint when --ref-coef > 0)")
    ap.add_argument("--ref-coef", type=float, default=0.0, help="weight of the KL to the reference (0: no anchor)")
    ap.add_argument("--joint-ratio", action="store_true",
                    help="units decoder: PPO on each decision's joint ratio instead of one ratio per unit")
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute activations in the backward pass")
    ap.add_argument("--device", help="torch device for the learner (default: cuda when available)")
    ap.add_argument("--eval-mode", default="greedy", choices=("greedy", "sample"), help="how evaluation games are played")
    ap.add_argument("--eval-temperature", type=float, default=1.0, help="with --eval-mode sample")
    ap.add_argument("--micro-seqs", type=int, default=4, help="segments per forward pass (halved on out-of-memory)")
    ap.add_argument("--minibatch-seqs", type=int, default=16, help="segments per optimizer step")
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.7)
    return ap.parse_args(argv)


def map_pool(spec: str) -> tuple:
    from fa_env.maps import MAP_POOL
    if spec == "small":
        return tuple(sorted(k for k, m in MAP_POOL.items() if m["km"] <= 10))
    return tuple(m.strip() for m in spec.split(",") if m.strip())


def make_config(a, run):
    """Build and validate the runtime config before any run-folder or learner side effects."""
    start = a.resume or a.clone
    if start and Path(start).is_file():         # the checkpoint knows its network
        import torch
        saved = torch.load(start, map_location="cpu", weights_only=False)
        a.size, a.decoder = saved.get("size", a.size), saved.get("decoder", "slots")
    cfg = ToolsConfig(run_dir=str(run), workers=a.workers, size=a.size, batch=a.batch, segment=a.segment,
                      decoder=a.decoder, dagger_updates=a.dagger_updates, dagger_lr=a.dagger_lr,
                      decision_ticks=a.decision_ticks,
                      lr=a.lr, eval_every=a.eval_every, eval_games=a.eval_games, minutes=a.minutes,
                      eval_minutes=a.eval_minutes, curriculum=True, fake=a.fake_raw is not None, fake_raw=a.fake_raw,
                      minibatch_seqs=a.minibatch_seqs, micro_seqs=a.micro_seqs, act_temperature=a.temperature,
                      distill_teacher=a.distill_teacher,
                      distill_start=a.distill_start, distill_end=a.distill_end,
                      distill_fraction=a.distill_fraction, ladder=True, ladder_name=a.ladder, faction=0,
                      maps=not a.single_map, map_pool=map_pool(a.maps), max_minutes=a.max_minutes,
                      unit_ratio=not a.joint_ratio, ref_coef=a.ref_coef, grad_checkpoint=a.grad_checkpoint,
                      eval_mode=a.eval_mode, eval_temperature=a.eval_temperature,
                      gpu_memory_fraction=a.gpu_memory_fraction)
    cfg.ent_start, cfg.ent_end = a.entropy_bonus, a.entropy_bonus / 3
    if a.start_stage:
        stage_index(LADDERS[a.ladder], a.start_stage)          # fail early on a rung that does not exist
    return cfg


def main(argv=None):
    a = parse_args(argv)

    run = Path(datetime.now().strftime(a.run))
    cfg = make_config(a, run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "STOP").unlink(missing_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log = open(run / "night.log", "a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    stop_at = parse_stop(a.stop_at)
    minute = timedelta(minutes=1)
    fake = a.fake_raw is not None
    report = {"started": datetime.now().isoformat(timespec="seconds"), "stop_at": stop_at.isoformat(), "phases": [],
              "errors": [], "config": vars(a), "kind": "tools"}
    print(f"tools night run until {stop_at:%Y-%m-%d %H:%M}; create {run / 'STOP'} to stop early")
    keep_awake(True)
    raise_learner_priority()

    def stopped() -> bool:
        return (run / "STOP").exists() or datetime.now() >= stop_at

    def failed(phase: str, attempt: int = 1, trainer=None) -> None:
        text = traceback.format_exc()
        print(f"phase {phase} failed (attempt {attempt}):\n{text}")
        report["errors"].append({"phase": phase, "attempt": attempt, "error": text.strip().splitlines()[-1]})
        if trainer is not None:
            trainer._log_event({"event": "phase_error", "phase": phase, "attempt": attempt, "error": text[-4000:]})
            trainer.stop_workers()
        write_report(run, report)

    clone = a.clone
    trainer = None
    try:
        # 1. collect: teacher v3 games, recorded
        if not clone and not a.resume and not fake and a.collect_games > 0 and not stopped():
            t0 = time.monotonic()
            try:
                deadline = min(stop_at - 60 * minute * a.scale,
                               datetime.now() + timedelta(hours=a.collect_hours * a.scale))
                summary = collect(run, a.collect_games, a.workers,
                                  a.minutes if a.single_map else (a.max_minutes or None),
                                  stop_at=deadline, log=lambda s: print(s, flush=True),
                                  maps=["SCMP_016"] if a.single_map else (list(map_pool(a.maps)) or None),
                                  m28_share=a.m28_share, seed=a.collect_seed)
                report["phases"].append({"name": "collect: teacher v3 games, recorded", **summary,
                                         "minutes": round((time.monotonic() - t0) / 60, 1)})
            except Exception:
                failed("collect")
            write_report(run, report)

        # 2. clone the tools network from every recording
        if not clone and not a.resume and not stopped():
            t0 = time.monotonic()
            try:
                cache = run / "tools_cache"
                cache.mkdir(parents=True, exist_ok=True)
                for src in a.reuse_cache:                       # reuse only cache entries this encoder can read
                    for f in current_cache_paths(Path(src)):
                        if not (cache / f.name).exists():
                            shutil.copyfile(f, cache / f.name)
                folders = ([] if fake else [run / "raw"]) + [Path(p) for p in a.extra_raw] + \
                    ([Path(a.fake_raw)] if fake else [])
                for folder in folders:
                    if (folder / "blueprints.json.gz").exists():
                        build_cache(folder, cache)
                games = Games(cache)
                out = run / "checkpoints" / "tools_bc.pt"
                state = train_bc(cache, out, size=a.size, steps=a.bc_steps if not fake else 60,
                                 eval_every=max(20, min(500, a.bc_steps // 6)) if not fake else 30, games=games,
                                 decoder=a.decoder)
                best = state["best"]
                report["phases"].append({"name": "clone (tools network)", "size": a.size, "params": state["params"],
                                         "games": len(games.games), "decisions": int(games.lengths.sum()),
                                         "held_out": {k: round(v, 3) for k, v in best.items()
                                                      if isinstance(v, float)},
                                         "minutes": round((time.monotonic() - t0) / 60, 1)})
                clone = str(out)
                del games  # release the demonstration corpus before the long live-game phase
            except Exception:
                failed("clone")
                # A failed clone must not silently launch hours of PPO from random weights.
                raise
            write_report(run, report)

        # 3. PPO
        trainer = ToolsTrainer(cfg, device=a.device)
        final_budget = 25 * minute * a.scale
        ppo_until = stop_at - final_budget
        if not stopped() and datetime.now() < ppo_until:
            if a.resume:
                trainer.load(a.resume)
            elif clone:
                trainer.load(clone, optimizer=False)
            reference = a.reference or (clone if a.ref_coef > 0 else None)
            if reference:
                trainer.set_reference(reference)
            if a.start_stage:
                trainer.stage = stage_index(trainer.stages, a.start_stage)
                print(f"starting at rung {trainer.stages[trainer.stage]['name']}: "
                      f"opponents {trainer.stages[trainer.stage]['opponents']}")
            hours = max(0.1, (ppo_until - datetime.now()).total_seconds() / 3600)
            cfg.planned_updates = trainer.update + max(50, int(hours * 3600 * 60 * a.workers / 20 / cfg.batch))
            trainer.plan_deadline = ppo_until
            start_update, start_stage = trainer.update, trainer.stage
            if trainer.update == 0 and a.baseline_evals:        # where learning starts: the clone itself
                for item in a.baseline_evals.split(","):
                    ai, games = item.split(":")
                    trainer.schedule_eval(int(games), ai.strip())
            t0 = time.monotonic()
            summary = {"mode": "ppo", "stop": None}
            for attempt in range(1, 7):
                try:
                    summary = trainer.run("ppo", stop_at=ppo_until)
                    break
                except Exception:
                    failed("ppo", attempt, trainer)
                    latest = run / "checkpoints" / "latest.pt"
                    if attempt == 6 or stopped() or datetime.now() >= ppo_until:
                        summary["stop"] = "learner error"
                        break
                    if latest.exists():
                        trainer.load(latest)
            trainer.eval_queue.clear()
            curve = {}
            for r in trainer.eval_results:
                curve.setdefault(f"{r.get('tag')} {r.get('ai')}", []).append(1.0 if r.get("outcome") == "victory" else 0.0)
            train_games = [r for key, records in trainer.recent.items() if key.startswith("train_") for r in records]
            report["phases"].append({
                "name": "ppo (tools)", **summary, "updates": trainer.update - start_update,
                "planned_updates": cfg.planned_updates, "stage_start": trainer.stages[start_stage]["name"],
                "stage_end": trainer.stages[trainer.stage]["name"], "minutes": round((time.monotonic() - t0) / 60, 1),
                "eval_win_rate": {k: f"{int(sum(v))}/{len(v)}" for k, v in curve.items()},
                "last_training_games": by_ai(train_games)})
            trainer.eval_results.clear()
            write_report(run, report)

        # 4. final evaluation
        if datetime.now() < stop_at - 5 * minute * a.scale and not stopped():
            t0 = time.monotonic()
            try:
                trainer.schedule_eval(max(2, a.final_games // 2), "medium")
                for ai in trainer.stages[trainer.stage]["evals"]:              # the rung it reached
                    if ai != "medium":
                        trainer.schedule_eval(max(2, a.final_games // 4), ai)
                trainer.run("eval", stop_at=stop_at)
            except Exception:
                failed("final", trainer=trainer)
            report["phases"].append({"name": "final greedy evaluation", "minutes": round((time.monotonic() - t0) / 60, 1),
                                     "results": by_ai(trainer.eval_results)})
    finally:
        try:
            report["ended"] = datetime.now().isoformat(timespec="seconds")
            if trainer is not None:
                report["restarts"] = trainer.restarts
                trainer.close()
            write_report(run, report)
            keep_awake(False)
            print(f"report: {run / 'report.md'}")
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            log.close()


if __name__ == "__main__":
    main()
