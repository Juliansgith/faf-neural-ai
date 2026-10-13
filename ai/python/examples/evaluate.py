"""Play G games of a reference agent against one FAF AI, P games at a time, and summarize.

    python evaluate.py --agent scripted --ai easy --games 8 --parallel 8 --minutes 20 --out <dir>

One game process per game (multiprocessing). Each game appends one JSON line to <out>/games.jsonl:
outcome (victory / defeat / timeout), game minutes, wall seconds, speed-up, final economy and trade
stats. The summary at the end gives win rate, mean length and aggregate throughput (game-seconds per
wall-second across all parallel games), which is also how to measure how many games this machine can
run at once.

Agents: scripted (fa_env.agents.ScriptedAgent), random (RandomAgent), noop (never acts).
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def play(job: dict) -> dict:
    from fa_env import FAEnv, GameConfig
    from fa_env.agents import RandomAgent, ScriptedAgent

    config = GameConfig(map=job["map"], faction=job["faction"], ai_faction=job["ai_faction"], ai=job["ai"],
                        seed=job["seed"], window=(640, 480))
    env = FAEnv(config, step_ticks=job["ticks"], max_ticks=int(job["minutes"] * 600))
    agent = {"scripted": ScriptedAgent, "random": lambda: RandomAgent(seed=job["seed"])}.get(job["agent"])
    agent = agent() if agent else None
    t0 = time.monotonic()
    record = dict(job)
    try:
        obs, info = env.reset()
        record["load_seconds"] = round(info.get("load_seconds", 0.0), 1)
        if agent:
            agent.reset(obs, info)
        step_info, done, errors = None, False, 0
        terminated = truncated = False
        while not done:
            actions, queries = agent.act(obs, step_info) if agent else ([], [])
            obs, _, terminated, truncated, step_info = env.step(actions, queries)
            errors += sum(1 for r in step_info["results"] if not r.get("ok"))
            done = terminated or truncated
        me = next(a for a in obs["armies"] if a["index"] == obs["me"])
        stats = me.get("stats", {})
        wall = time.monotonic() - t0 - record["load_seconds"]
        record.update(
            outcome=step_info["outcome"] or ("timeout" if truncated else "unknown"),
            game_minutes=round(obs["tick"] / 600, 2),
            wall_seconds=round(wall, 1),
            speedup=round(obs["tick"] / 10 / max(wall, 1e-6), 1),
            mass_income=round(me["eco"]["mass"]["income"], 1),
            energy_income=round(me["eco"]["energy"]["income"], 1),
            own_units=sum(1 for u in obs["units"] if u["army"] == obs["me"]),
            mass_destroyed=round(stats.get("Enemies_MassValue_Destroyed", 0.0)),
            mass_lost=round(stats.get("Units_MassValue_Lost", 0.0)),
            action_errors=errors,
        )
    except Exception as exc:  # a crashed or hung game is data too
        record.update(outcome="error", error=f"{type(exc).__name__}: {exc}"[:500],
                      trace=traceback.format_exc()[-1500:], wall_seconds=round(time.monotonic() - t0, 1))
    finally:
        env.close()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["scripted", "random", "noop"], default="scripted")
    ap.add_argument("--ai", default="easy", help="FAF AI personality key (easy, medium, adaptive, rush, ...)")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--ticks", type=int, default=10)
    ap.add_argument("--map", default="SCMP_016")
    ap.add_argument("--faction", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(agent=a.agent, ai=a.ai, map=a.map, faction=a.faction, ai_faction=1 + i % 4, seed=a.seed0 + i,
                 minutes=a.minutes, ticks=a.ticks) for i in range(a.games)]
    t0 = time.monotonic()
    results = []
    with mp.get_context("spawn").Pool(a.parallel, maxtasksperchild=1) as pool, \
            open(out / "games.jsonl", "a", encoding="utf-8") as log:
        for record in pool.imap_unordered(play, jobs):
            results.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"[{len(results)}/{len(jobs)}] seed {record['seed']} ai_faction {record['ai_faction']}: "
                  f"{record['outcome']} at {record.get('game_minutes', '?')} min, "
                  f"{record.get('speedup', '?')}x, {record.get('wall_seconds')} s", flush=True)
    wall = time.monotonic() - t0
    played = [r for r in results if r["outcome"] != "error"]
    game_seconds = sum(r["game_minutes"] * 60 for r in played)
    summary = {
        "agent": a.agent, "ai": a.ai, "games": len(results), "parallel": a.parallel,
        "outcomes": {k: sum(1 for r in results if r["outcome"] == k) for k in
                     sorted({r["outcome"] for r in results})},
        "mean_game_minutes": round(game_seconds / 60 / max(1, len(played)), 2),
        "wall_seconds": round(wall, 1),
        "aggregate_game_seconds_per_wall_second": round(game_seconds / wall, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
