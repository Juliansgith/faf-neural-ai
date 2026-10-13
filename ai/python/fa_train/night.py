"""The unattended night pipeline (docs/training-design.md, section 15):

  1. gate     the rule policy plays 30-minute games against easy (and medium): can this action space win?
  2. collect  the rule policy plays against easy and medium; its choices are saved for cloning
  3. clone    behaviour cloning of the actor, critic warm-started on the rule policy's returns
  4. ppo      PPO from the clone with a decaying KL pull towards it and periodic greedy evals; opponents
              follow the curriculum (stage A easy, promoted to stage B, mostly medium, at 90 % wins)
  5. final    greedy evaluation against easy and medium
  6. report   <run>/report.md

    python -m fa_train.night --run C:/faf/_runs/night1 --workers 20 --stop-at 07:00

Everything stops at --stop-at (games killed, checkpoint saved) or as soon as <run>/STOP exists, so the PC
is free in the morning. A phase that fails is logged and the night goes on (PPO resumes from its last
checkpoint). Windows is asked not to sleep while the run is going. --fake runs the whole pipeline without
games (a machinery test); --scale shrinks the phase budgets for a short real test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train.bc import train_bc  # noqa: E402
from fa_train.trainer import STAGES, TrainConfig, Trainer  # noqa: E402


def parse_stop(text: str) -> datetime:
    hour, minute = map(int, text.split(":"))
    now = datetime.now()
    stop = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return stop if stop > now else stop + timedelta(days=1)


class Tee:
    """Console output also goes to <run>/night.log."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while this process runs: a per-process request (what media players and
    downloads use) that ends with the process; no setting is changed. The display may still turn off."""
    if sys.platform != "win32":
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def raise_learner_priority() -> None:
    """Every game waits on this process for its next action; with 20 games oversubscribing the CPU it must
    not queue behind them. Above-normal priority for this process only (it is idle most of the time)."""
    if sys.platform != "win32":
        return
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass(ctypes.c_void_p(k32.GetCurrentProcess()), 0x00008000)   # ABOVE_NORMAL_PRIORITY_CLASS


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"games": 0}
    outcomes = {}
    for r in results:
        outcomes[r.get("outcome")] = outcomes.get(r.get("outcome"), 0) + 1
    return {"games": len(results), "outcomes": outcomes,
            "win_rate": round(outcomes.get("victory", 0) / len(results), 3),
            "mean_minutes": round(float(np.mean([r.get("minutes", 0) for r in results])), 1),
            "mean_value_advantage": round(float(np.mean([r.get("value_advantage", 0) for r in results])), 3),
            "mean_mass_destroyed": round(float(np.mean([r.get("mass_destroyed", 0) for r in results]))),
            "mean_mass_lost": round(float(np.mean([r.get("mass_lost", 0) for r in results])))}


def by_ai(results: list[dict]) -> dict:
    out = {}
    for ai in sorted({r.get("ai") for r in results}):
        out[ai] = summarize([r for r in results if r.get("ai") == ai])
    return out


def write_report(run: Path, report: dict) -> None:
    (run / "report.json").write_text(json.dumps(report, indent=1, default=str))
    lines = [f"# Night run {run.name}", "",
             f"Started {report['started']}, stop time {report['stop_at']}, ended {report.get('ended', '?')}.", ""]
    for phase in report["phases"]:
        lines.append(f"## {phase['name']}")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps({k: v for k, v in phase.items() if k != "name"}, indent=1, default=str))
        lines.append("```")
        lines.append("")
    if report.get("errors"):
        lines += ["## Errors", "", "```", json.dumps(report["errors"], indent=1, default=str), "```", ""]
    lines += ["TensorBoard: `tensorboard --logdir " + str(run / "tb") + "`",
              "Per-game records: episodes.jsonl; PPO updates: updates.jsonl; crashes and promotions: events.jsonl, "
              "game_logs/failed/."]
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder; strftime codes allowed (e.g. _runs/night-%%Y%%m%%d)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--stop-at", required=True, help="HH:MM (local time; tomorrow if already past)")
    ap.add_argument("--gate-games", type=int, default=15, help="rule-policy games against easy")
    ap.add_argument("--gate-medium", type=int, default=5, help="rule-policy games against medium")
    ap.add_argument("--bc-games", type=int, default=300)
    ap.add_argument("--collect-hours", type=float, default=1.25)
    ap.add_argument("--final-games", type=int, default=16)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--bc-checkpoint", help="skip collection and cloning; start PPO from this checkpoint")
    ap.add_argument("--resume", help="continue PPO from this checkpoint (optimizer, schedules, stage, teacher)")
    ap.add_argument("--fake", action="store_true", help="no games: exercise the whole pipeline quickly")
    ap.add_argument("--fake-delay", type=float, default=0.005)
    ap.add_argument("--scale", type=float, help="phase time budgets x this (default 1; 0.02 with --fake)")
    ap.add_argument("--bc-min-steps", type=int, default=10000, help="minimum gradient steps of cloning")
    ap.add_argument("--batch", type=int, default=3072)
    ap.add_argument("--segment", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=40, help="PPO updates between greedy evaluations")
    ap.add_argument("--eval-games", type=int, default=8, help="games per evaluation (8 = the stage's plan)")
    ap.add_argument("--minutes", type=float, default=20.0, help="game-time limit of training games")
    ap.add_argument("--eval-minutes", type=float, default=30.0, help="game-time limit of gate/eval games")
    a = ap.parse_args()

    run = Path(datetime.now().strftime(a.run))
    run.mkdir(parents=True, exist_ok=True)
    (run / "STOP").unlink(missing_ok=True)
    log = open(run / "night.log", "a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    stop_at = parse_stop(a.stop_at)
    minute = timedelta(minutes=1)
    scale = a.scale if a.scale is not None else (0.02 if a.fake else 1.0)
    cfg = TrainConfig(run_dir=str(run), workers=a.workers, fake=a.fake, fake_delay=a.fake_delay, batch=a.batch,
                      segment=a.segment, eval_every=a.eval_every, eval_games=a.eval_games,
                      minutes=a.minutes, eval_minutes=a.eval_minutes)
    trainer = Trainer(cfg)
    report = {"started": datetime.now().isoformat(timespec="seconds"), "stop_at": stop_at.isoformat(), "phases": [],
              "errors": [], "config": vars(a)}
    print(f"night run until {stop_at:%Y-%m-%d %H:%M}; create {run / 'STOP'} to stop early")
    keep_awake(True)
    raise_learner_priority()

    def left() -> timedelta:
        return stop_at - datetime.now()

    def stopped() -> bool:
        return (run / "STOP").exists() or datetime.now() >= stop_at

    def failed(phase: str, attempt: int = 1) -> None:
        text = traceback.format_exc()
        print(f"phase {phase} failed (attempt {attempt}):\n{text}")
        trainer._log_event({"event": "phase_error", "phase": phase, "attempt": attempt, "error": text[-4000:]})
        report["errors"].append({"phase": phase, "attempt": attempt, "error": text.strip().splitlines()[-1]})
        trainer.stop_workers()
        write_report(run, report)

    try:
        # 1. gate: the rule policy against easy (and medium, for information)
        if not a.skip_gate and not a.bc_checkpoint and not a.resume:
            t0 = time.monotonic()
            try:
                trainer.schedule_eval(a.gate_medium, "medium", rule=True, minutes=a.eval_minutes)
                trainer.schedule_eval(a.gate_games, "easy", rule=True, minutes=a.eval_minutes)
                trainer.run("eval", stop_at=min(stop_at, datetime.now() + 60 * minute * scale + 3 * minute))
            except Exception:
                failed("gate")
            report["phases"].append({"name": f"gate: rule policy, {a.eval_minutes:g}-minute games",
                                     "minutes": round((time.monotonic() - t0) / 60, 1),
                                     "results": by_ai(trainer.eval_results)})
            trainer.eval_results.clear()
            trainer.eval_queue.clear()
            write_report(run, report)

        # 2 + 3. behaviour cloning
        teacher = a.bc_checkpoint
        if not teacher and not a.resume and left() > 45 * minute * scale and not stopped():
            t0 = time.monotonic()
            cfg.opponents = {"easy": 0.5, "medium": 0.5}
            data = run / "bc_data"
            deadline = min(stop_at - 30 * minute * scale,
                           datetime.now() + timedelta(hours=a.collect_hours * max(scale, 0.05)))
            for attempt in (1, 2):
                try:
                    summary = trainer.run("collect", stop_at=deadline, max_episodes=a.bc_games, collect_dir=data)
                    report["phases"].append({"name": "collect: rule policy games for cloning", **summary,
                                             "minutes": round((time.monotonic() - t0) / 60, 1)})
                    break
                except Exception:
                    try:
                        trainer._flush_shard()     # keep the finished games collected so far
                    except Exception:
                        pass
                    failed("collect", attempt)
                    if any(data.glob("shard_*.npz")):
                        break                      # enough to clone from
            write_report(run, report)
            if any(data.glob("shard_*.npz")):
                t0 = time.monotonic()
                try:
                    bc = train_bc(data, run / "checkpoints" / "bc.pt", epochs=3 if a.fake else 8,
                                  min_steps=200 if a.fake else a.bc_min_steps)
                    last = bc["best"]
                    report["phases"].append({"name": "clone", "rows": bc["rows"], "held_out": bc["held_out"],
                                             "agreement": last["agreement"],
                                             "agreement_when_acting": last["agreement_when_acting"],
                                             "entropy": last["entropy"], "value_mse": last["value_mse"],
                                             "minutes": round((time.monotonic() - t0) / 60, 1)})
                    teacher = str(run / "checkpoints" / "bc.pt")
                except Exception:
                    failed("clone")
            write_report(run, report)

        # 4. PPO, with the curriculum; a learner error resumes from the last checkpoint
        final_budget = 25 * minute * scale
        ppo_until = stop_at - final_budget
        if not stopped() and datetime.now() < ppo_until:
            if a.resume:
                trainer.load(a.resume)
                bc_path = Path(a.bc_checkpoint) if a.bc_checkpoint else run / "checkpoints" / "bc.pt"
                if bc_path.exists():
                    trainer.set_teacher(bc_path)
            elif teacher:
                trainer.load(teacher, optimizer=False)
                trainer.set_teacher(teacher)
            cfg.curriculum = True
            hours = max(0.1, (ppo_until - datetime.now()).total_seconds() / 3600)
            # a first guess (~80 decisions/s with 20 games); re-estimated from the measured rate as it runs
            cfg.planned_updates = trainer.update + max(50, int(hours * 3600 * 80 * a.workers / 20 / cfg.batch))
            trainer.plan_deadline = ppo_until
            start_update, start_stage = trainer.update, trainer.stage
            t0 = time.monotonic()
            summary = {"mode": "ppo", "stop": None}
            for attempt in range(1, 7):
                try:
                    summary = trainer.run("ppo", stop_at=ppo_until)
                    break
                except Exception:
                    failed("ppo", attempt)
                    latest = run / "checkpoints" / "latest.pt"
                    if attempt == 6 or stopped() or datetime.now() >= ppo_until:
                        summary["stop"] = "learner error"
                        break
                    if latest.exists():
                        trainer.load(latest)
            trainer.eval_queue.clear()
            evals = trainer.eval_results
            curve = {}
            for r in evals:
                curve.setdefault(f"{r.get('tag')} {r.get('ai')}", []).append(1.0 if r.get("outcome") == "victory" else 0.0)
            train_games = [r for key, records in trainer.recent.items() if key.startswith("train_") for r in records]
            report["phases"].append({
                "name": "ppo", **summary, "updates": trainer.update - start_update,
                "planned_updates": cfg.planned_updates, "stage_start": STAGES[start_stage]["name"],
                "stage_end": STAGES[trainer.stage]["name"], "minutes": round((time.monotonic() - t0) / 60, 1),
                "skipped_minibatches": trainer.skipped_minibatches,
                "eval_win_rate": {k: f"{int(sum(v))}/{len(v)}" for k, v in curve.items()},
                "last_training_games": by_ai(train_games)})
            trainer.eval_results.clear()
            write_report(run, report)

        # 5. final evaluation
        if left() > 5 * minute * scale and not stopped():
            t0 = time.monotonic()
            try:
                trainer.schedule_eval(max(2, a.final_games // 2), "medium")
                trainer.schedule_eval(a.final_games, "easy")
                trainer.run("eval", stop_at=stop_at)
            except Exception:
                failed("final")
            report["phases"].append({"name": "final greedy evaluation", "minutes": round((time.monotonic() - t0) / 60, 1),
                                     "results": by_ai(trainer.eval_results)})
    finally:
        report["ended"] = datetime.now().isoformat(timespec="seconds")
        report["restarts"] = trainer.restarts
        write_report(run, report)
        trainer.close()
        keep_awake(False)
        print(f"report: {run / 'report.md'}")


if __name__ == "__main__":
    main()
