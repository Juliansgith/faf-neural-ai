"""Gate F1 (docs/full-toolbox-design.md): the teacher v3 plays through the v3 executor.

It should win against normal (medium) at least 9 times in 10 and reach T3 in most long games. Games run one
after another (one game process at a time). Every two minutes a game prints its tech state: mass income,
factories and engineers per tech, power, shields, the commander's enhancements, the army per tech.

    python teacher_v3_game.py --ai medium --faction 1 --seed 3                  # one game
    python teacher_v3_game.py --ai medium --games 10 --quiet                    # gate F1: factions rotate
    python teacher_v3_game.py --ai turtle,adaptive --games 4 --record DIR       # long games, recorded
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env import GameConfig  # noqa: E402
from fa_env.teacher_v3 import TeacherV3  # noqa: E402
from fa_env.tools import cats  # noqa: E402
from fa_env.tools_env import ToolsEnv  # noqa: E402

FACTIONS = {1: "UEF", 2: "Aeon", 3: "Cybran", 4: "Seraphim"}


def tech_state(env, reached: dict) -> dict:
    count = collections.Counter()
    minute = round(env.state.tick / 600, 1)
    for u in env.own_units().values():
        if (u.get("frac") or 1) < 1:
            continue
        bp = env.bps.get(u["bp"], {})
        c, t = cats(bp), int(bp.get("tech") or 1)
        if "FACTORY" in c and "STRUCTURE" in c:
            count[f"fac_t{t}{'hq' if 'RESEARCH' in c else ''}"] += 1
            reached.setdefault(f"factory T{t}", minute)
        elif "ENGINEER" in c and "COMMAND" not in c:
            count[f"eng_t{t}"] += 1
        elif "ENERGYPRODUCTION" in c and "STRUCTURE" in c:
            count[f"power_t{t}"] += 1
        elif "SHIELD" in c and "STRUCTURE" in c:
            count["shield"] += 1
            reached.setdefault("shield", minute)
        elif "MASSEXTRACTION" in c:
            count[f"mex_t{t}"] += 1
        elif "MOBILE" in c and bp.get("weapons") and "COMMAND" not in c:
            count[f"army_t{t}"] += 1
            if t >= 2:
                reached.setdefault(f"army T{t}", minute)
        if "COMMAND" in c and u.get("enh"):
            count["acu:" + "+".join(sorted(u["enh"]))] = 1
            reached.setdefault("commander upgrade", minute)
    return dict(sorted(count.items()))


def play(ai: str, faction: int, seed: int, minutes: float = 45, ai_faction: int = 2, cheat: float | None = None,
         record: str | None = None, quiet: bool = False, teacher_version: str = "v3", map_id: str = "SCMP_016",
         slots: tuple | None = None) -> dict:
    env = ToolsEnv(GameConfig(map=map_id, faction=faction, ai=ai, ai_faction=ai_faction, seed=seed, cheat=cheat,
                              slots=slots, window=(640, 480), minimized=True), time_limit_minutes=minutes)
    if record:
        from fa_env.record import Recorder
        env.recorder = Recorder(Path(record), f"teacher_v3_{map_id}_{ai}_f{faction}_s{seed}")
    if teacher_version == "v2":             # the tools version's rule teacher (T1 only), for comparison
        from fa_env.rule_policy import RulePolicy
        teacher = RulePolicy()
    else:
        teacher = TeacherV3()
    t0 = time.monotonic()
    try:
        env.reset()
        teacher.reset(env)
        verbs, failures, done, last, reached = collections.Counter(), 0, False, -1, {}
        step = {}
        while not done:
            cmds = env.teacher_commands(teacher, dry=False)
            verbs.update(c["verb"] for c in cmds)
            _, _, done, step = env.step_tools(cmds)
            failures += step["action_errors"]
            minute = step["tick"] // 1200 * 2
            if minute != last:
                last = minute
                state = tech_state(env, reached)
                if not quiet:
                    income = env.state.eco.get("mass", {}).get("income", 0)
                    print(f"[{minute:2d} min] mass +{income:.1f} | {state} | failed {failures} | "
                          f"{time.monotonic() - t0:.0f} s", flush=True)
        tech_state(env, reached)
    finally:
        env.close()
    return {"ai": ai, "faction": FACTIONS.get(faction, faction), "seed": seed, "map": map_id, "slots": slots,
            **(step.get("episode") or {}),
            "reached": reached, "failed_orders": failures, "wall_s": round(time.monotonic() - t0),
            "commands": dict(verbs.most_common())}


def _play_job(job: tuple) -> dict:
    try:
        return play(*job)
    except Exception as exc:                  # one failed game must not end the series
        return {"ai": job[0], "faction": FACTIONS.get(job[1]), "seed": job[2], "map": job[9] if len(job) > 9 else None,
                "error": repr(exc), "reached": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", default="medium", help="one AI or a comma list (games alternate)")
    ap.add_argument("--faction", type=int, help="our faction (1-4); default: rotate over the four")
    ap.add_argument("--ai-faction", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1, help="first seed")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=45, help="time limit (T3 armies need longer than 30)")
    ap.add_argument("--cheat", type=float, help="AIx multiplier (cheating AIs only)")
    ap.add_argument("--record", help="save raw recordings to this folder")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--parallel", type=int, default=1, help="games at once (each its own process)")
    ap.add_argument("--teacher", default="v3", choices=("v2", "v3"), help="v2: the earlier rule teacher, as a baseline")
    ap.add_argument("--maps", default="SCMP_016", help="a map, a comma list (games rotate), or 'pool' (fa_env.maps)")
    ap.add_argument("--pool-minutes", action="store_true", help="each map's time limit by size (fa_env.maps)")
    a = ap.parse_args()
    from fa_env.maps import MAP_POOL, MINUTES_BY_KM
    ais = a.ai.split(",")
    maps = sorted(MAP_POOL) if a.maps == "pool" else a.maps.split(",")
    jobs = []
    for g in range(a.games):
        map_id = maps[g % len(maps)]
        entry = MAP_POOL.get(map_id)
        slots = tuple(entry["slots"][(g // len(maps)) % len(entry["slots"])]) if entry else None
        minutes = MINUTES_BY_KM[entry["km"]] if (entry and a.pool_minutes) else a.minutes
        jobs.append((ais[g % len(ais)], a.faction or (g % 4) + 1, a.seed + g, minutes, a.ai_faction, a.cheat, a.record,
                     a.quiet or a.parallel > 1, a.teacher, map_id, slots))
    results = []
    if a.parallel > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(a.parallel, len(jobs))) as pool:
            for r in pool.imap_unordered(_play_job, jobs):
                results.append(r)
                print(json.dumps({k: v for k, v in r.items() if k != "commands"}), flush=True)
    else:
        for job in jobs:
            r = play(*job)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "commands"}), flush=True)
    wins = sum(r.get("outcome") == "victory" for r in results)
    t3 = sum("factory T3" in r["reached"] for r in results)
    long_games = [r for r in results if (r.get("minutes") or 0) >= 25]
    print(f"\n{wins}/{len(results)} wins; T3 reached in {t3} games "
          f"({sum('factory T3' in r['reached'] for r in long_games)}/{len(long_games)} of games of 25+ minutes); "
          f"median length {sorted(r.get('minutes') or 0 for r in results)[len(results) // 2]} min")


if __name__ == "__main__":
    main()
