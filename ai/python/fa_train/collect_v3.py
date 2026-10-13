"""Collection for the full-toolbox network: the teacher v3 plays recorded games (docs/full-toolbox-design.md,
step 3). Each worker process plays whole games by itself (the teacher runs next to its game), so nothing waits
on a learner.

    python -m fa_train.collect_v3 --run _runs/v3-%Y%m%d --games 1000 --workers 20 --stop-at 08:00

Games rotate over our four factions and an opponent mix (FAF's easy, normal and the four personalities, by
default); every game is saved raw to <run>/raw (fa_env/record.py) and summarised in <run>/collect.jsonl. A STOP
file in the run folder ends the collection after the running games. Games run minimized at below-normal
priority and never take focus (GameConfig.minimized); one game per worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPPONENTS = {"easy": 0.15, "medium": 0.25, "adaptive": 0.15, "rush": 0.15, "turtle": 0.15, "tech": 0.15}


def play_one(job: dict) -> dict:
    """One recorded game in this worker process: the teacher v3 plays, or (job["player_ai"]) another AI plays
    our slot and the bridge captures its orders as the labels. A summary, or the error."""
    from fa_env import GameConfig
    from fa_env.record import Recorder
    from fa_env.teacher_v3 import TeacherV3
    from fa_env.tools_env import ToolsEnv
    t0 = time.monotonic()
    env = None
    try:
        log_dir = Path(job["run"]) / "game_logs" / f"w{mp.current_process().name.split('-')[-1]}"
        log_dir.mkdir(parents=True, exist_ok=True)
        player_ai = job.get("player_ai")
        env = ToolsEnv(GameConfig(map=job["map"], faction=job["faction"], ai=job["ai"], ai_faction=job.get("ai_faction"),
                                  seed=job["seed"], cheat=job.get("cheat"), slots=tuple(job["slots"]) if job.get("slots") else None,
                                  player_ai=player_ai, window=(640, 480), log_dir=log_dir, minimized=True),
                       time_limit_minutes=job["minutes"], capture=bool(player_ai))
        who = player_ai or "teacher"
        env.recorder = Recorder(Path(job["run"]) / "raw",
                                f"{job['index']:05d}_{who}_{job['map']}_{job['ai']}_f{job['faction']}_s{job['seed']}",
                                extra_meta={k: job[k] for k in ("engine_path", "engine_sha256") if k in job})
        teacher = None if player_ai else TeacherV3()
        env.reset()
        if teacher:
            teacher.reset(env)
        done, step, failed, captured = False, {}, 0, 0
        while not done:
            _, _, done, step = env.step_tools(env.teacher_commands(teacher, dry=False) if teacher else [])
            failed += step.get("action_errors") or 0
            captured += step.get("captured") or 0
        top = max([int(env.bps.get(u["bp"], {}).get("tech") or 1) for u in env.own_units().values()
                   if "FACTORY" in (env.bps.get(u["bp"], {}).get("categories") or [])] or [0])
        return {**job, **(step.get("episode") or {}), "factory_tech": top, "failed_orders": failed,
                "captured_orders": captured, "wall_s": round(time.monotonic() - t0)}
    except Exception as exc:                                  # a failed game must not stop the collection
        return {**job, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:],
                "wall_s": round(time.monotonic() - t0)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def jobs(run: Path, n: int, seed: int, minutes: float | None, opponents: dict, start_index: int,
         maps: list[str] | None = None, m28_share: float = 0.0,
         engine_provenance: dict | None = None):
    """Game settings: our faction rotates; opponent, map and start slots are drawn (the pool's sizes equally
    often); `minutes` None: the time limit by map size."""
    from fa_env.maps import MAP_POOL, sample_map
    rng = random.Random(seed * 7919 + start_index)
    names, weights = list(opponents), list(opponents.values())
    pool = {k: v for k, v in MAP_POOL.items() if not maps or k in maps}
    for i in range(start_index, start_index + n):
        map_id, slots, limit = sample_map(rng, pool)
        job = {"run": str(run), "index": i, "map": map_id, "slots": list(slots), "faction": i % 4 + 1,
                "ai": rng.choices(names, weights)[0], "ai_faction": None, "seed": seed * 100000 + i,
                "minutes": minutes or limit, **(engine_provenance or {})}
        if rng.random() < m28_share:      # M28 plays our slot (its orders are the labels), a fifth against itself
            job["player_ai"] = "m28ai"
            if rng.random() < 0.2:
                job["ai"] = "m28ai"
        yield job


def collect(run: Path, games: int, workers: int, minutes: float | None = None, stop_at: datetime | None = None,
            seed: int = 7, opponents: dict | None = None, log=print, maps: list[str] | None = None,
            m28_share: float = 0.0) -> dict:
    """Play `games` recorded teacher games (fewer if `stop_at` passes or <run>/STOP appears); a summary per
    opponent. Continues the numbering of an earlier collection in the same run."""
    opponents = opponents or OPPONENTS
    (run / "raw").mkdir(parents=True, exist_ok=True)
    out_path = run / "collect.jsonl"
    # Record the exact engine per game: a resumed collection may use a different binary.
    from fa_env.bridge import DEFAULT_BIN_DIR
    engine = (Path(DEFAULT_BIN_DIR) / "main.exe").resolve()
    digest = hashlib.sha256()
    with engine.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    engine_provenance = {"engine_path": str(engine), "engine_sha256": digest.hexdigest()}
    done_before = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0
    log(f"collecting {games} teacher v3 games in {run} with {workers} workers; opponents {opponents}")
    t0, results = time.monotonic(), []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=20) as pool, out_path.open("a", encoding="utf-8") as out:
        pending = jobs(run, games, seed, minutes, opponents, done_before, maps, m28_share,
                       engine_provenance)
        running, exhausted = [], False
        while True:
            stopping = (run / "STOP").exists() or (stop_at is not None and datetime.now() >= stop_at)
            while not (stopping or exhausted) and len(running) < workers:
                job = next(pending, None)
                if job is None:
                    exhausted = True
                    break
                running.append(pool.apply_async(play_one, (job,)))
            for r in [r for r in running if r.ready()]:
                running.remove(r)
                res = r.get()
                results.append(res)
                out.write(json.dumps(res) + "\n")
                out.flush()
                wins = sum(x.get("outcome") == "victory" for x in results)
                errors = sum("error" in x for x in results)
                log(f"[{datetime.now():%H:%M:%S}] game {res['index']} {res.get('player_ai', 'teacher')} {res['map']} "
                    f"vs {res['ai']} f{res['faction']}: "
                    f"{res.get('outcome', res.get('error'))} {res.get('minutes', '')} min, T{res.get('factory_tech', '?')} "
                    f"| {len(results)} done, {wins} wins, {errors} errors, "
                    f"{len(results) / max(1e-6, (time.monotonic() - t0) / 3600):.0f} games/h")
            if not running and (stopping or exhausted):
                break
            time.sleep(1.0)
    summary = {}
    for r in results:
        s = summary.setdefault(f"{r.get('player_ai', 'teacher')} vs {r['ai']}",
                               {"games": 0, "wins": 0, "t2": 0, "t3": 0, "errors": 0})
        s["games"] += 1
        s["wins"] += r.get("outcome") == "victory"
        s["t2"] += (r.get("factory_tech") or 0) >= 2
        s["t3"] += (r.get("factory_tech") or 0) >= 3
        s["errors"] += "error" in r
    return {"games": len(results), "hours": round((time.monotonic() - t0) / 3600, 2), "by_ai": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder; strftime codes allowed")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=None, help="time limit of a game (default: by map size)")
    ap.add_argument("--maps", help="comma list of pool maps (default: the whole pool, fa_env.maps)")
    ap.add_argument("--m28-share", type=float, default=0.0, help="share of games M28 plays our slot (captured)")
    ap.add_argument("--stop-at", help="HH:MM: no new games after this (running ones finish)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opponents", help="ai:weight,... (default: %s)" % ",".join(f"{k}:{v}" for k, v in OPPONENTS.items()))
    a = ap.parse_args()
    from fa_train.night import keep_awake, parse_stop, raise_learner_priority
    run = Path(datetime.now().strftime(a.run))
    run.mkdir(parents=True, exist_ok=True)
    (run / "STOP").unlink(missing_ok=True)
    opponents = dict((k, float(v)) for k, v in (p.split(":") for p in a.opponents.split(","))) if a.opponents else None
    keep_awake(True)
    raise_learner_priority()
    try:
        summary = collect(run, a.games, a.workers, a.minutes, parse_stop(a.stop_at) if a.stop_at else None, a.seed,
                          opponents, log=lambda s: print(s, flush=True), maps=a.maps.split(",") if a.maps else None,
                          m28_share=a.m28_share)
    finally:
        keep_awake(False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
