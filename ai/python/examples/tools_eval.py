"""Evaluate a tools checkpoint on a repeatable schedule of real games.

--games is per opponent. Reuse the same flags for a second checkpoint to pair
maps, starts, factions and seeds. The policy plays greedy unless --mode sample (--temperature, --top-p).
One process holds the network for all games (never one GPU context per game: 14 of them brought the
graphics driver, and Windows, down on 2026-10-05).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def csv(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated value")
    return items


def factions(value: str) -> list[int]:
    result = [int(part) for part in csv(value)]
    if any(f not in (1, 2, 3, 4) for f in result):
        raise ValueError("factions must be 1, 2, 3 or 4")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schedule_rows(ais: list[str], games, maps: list[str], ours: list[int],
                  theirs: list[int], minutes: float | None, seed_base: int,
                  minutes_large: float | None = None) -> list[dict]:
    """The same row index has identical conditions against every opponent. `games`: per opponent, an int for all
    or {opponent: count}."""
    from fa_env.maps import MAP_POOL, MINUTES_BY_KM

    rows = []
    for ai in ais:
        for i in range(games[ai] if isinstance(games, dict) else games):
            map_id = maps[i % len(maps)]
            entry = MAP_POOL[map_id]
            # Cover every map/faction pair before repeating one. A later pass
            # reverses the start, while a checkerboard gives both starts a
            # chance even in a small smoke test.
            round_idx, map_idx = divmod(i, len(maps))
            pass_idx, faction_idx = divmod(round_idx, len(ours))
            slots = entry["slots"][(pass_idx + faction_idx + map_idx) % len(entry["slots"])]
            rows.append({"ai": ai, "map": map_id, "slots": tuple(slots),
                         "faction": ours[faction_idx],
                         "ai_faction": theirs[(round_idx + map_idx) % len(theirs)],
                         "seed": seed_base + i,
                         "minutes": minutes_large if minutes_large is not None and entry["km"] > 10
                         else minutes if minutes is not None else MINUTES_BY_KM[entry["km"]]})
    return rows


def telemetry_line(episodes: Path) -> str:
    """What the policy did, over all games: commands per decision by verb and the most produced blueprints."""
    import collections
    verbs, produced, decisions = collections.Counter(), collections.Counter(), 0
    for line in episodes.read_text(encoding="utf-8").splitlines():
        t = json.loads(line).get("command_telemetry") or {}
        verbs.update(t.get("verbs") or {})
        produced.update(t.get("productions") or {})
        decisions += t.get("policy_decisions") or 0
    n, total = max(1, decisions), max(1, sum(produced.values()))
    return ("per decision: " + ", ".join(f"{v} {c / n:.2f}" for v, c in verbs.most_common(7))
            + " | produced: " + ", ".join(f"{b} {c / total:.0%}" for b, c in produced.most_common(5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--bin-dir", type=Path, help="FAF bin folder containing main.exe")
    ap.add_argument("--ai", default="easy", help="comma-separated opponents; name:count sets one opponent's games")
    ap.add_argument("--games", type=int, default=3, help="games per opponent")
    ap.add_argument("--maps", help="comma-separated map IDs")
    ap.add_argument("--map-pool", action="store_true")
    ap.add_argument("--factions", default="1", help="comma-separated player factions")
    ap.add_argument("--ai-factions", default="1", help="comma-separated opponent factions")
    ap.add_argument("--minutes", help="override the map-size time limit: MINUTES, or MINUTES,LARGE with LARGE the "
                    "limit on maps above 10 km (a 20 km game needs longer to be won; the player passes its config's "
                    "'minutes' as it is, so \"40,55\" there needs no player restart)")
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--record", type=Path, help="save raw recordings of the games to this folder")
    ap.add_argument("--labels", choices=("v3", "m28"), help="with --record: a teacher's labels for every decision "
                    "(v3: the rule teacher asked dry; m28: M28AI in our slot in shadow mode)")
    ap.add_argument("--mode", default="greedy", choices=("greedy", "sample"), help="how the policy picks")
    ap.add_argument("--temperature", type=float, default=1.0, help="with --mode sample")
    ap.add_argument("--top-p", type=float, help="with --mode sample: nucleus sampling")
    ap.add_argument("--league-dir", type=Path,
                    help="self-play opponents ('self@<name>' in --ai): <name>.pt here (default: the checkpoint's "
                         "player home, <home>/rounds/<round>/network.pt -> <home>/league)")
    ap.add_argument("--build-budget", type=int, help="builds per decision (-1: no limit; default: the network's)")
    ap.add_argument("--act-gather-ms", type=float, default=0.0,
                    help="decide for several games at once: gather observations this long (CPU players)")
    ap.add_argument("--act-min-batch", type=int, default=1, help="with --act-gather-ms: stop gathering at this many")
    ap.add_argument("--watch", action="store_true",
                    help="a visible window to watch the games in (use --workers 1): not minimized, it takes focus")
    ap.add_argument("--device", help="where the network runs (cpu, cuda); default: the GPU when there is one")
    ap.add_argument("--debug", action="store_true",
                    help="record what each decision was made of for tools/watch_viewer.py (RUN/watch; --watch too)")
    ap.add_argument("--speed", type=int, default=50,
                    help="the game's pace (/agentspeed, FA sim rate: 0 normal, 3 about 2x, 50 as fast as it goes)")
    ap.add_argument("--commander-greedy", action="store_true",
                    help="the commander's destination is its most likely cell (its other choices as --mode)")
    args = ap.parse_args()
    minutes_large = None
    if args.minutes is not None:
        parts = [float(x) for x in str(args.minutes).split(",")]
        args.minutes, minutes_large = parts[0], (parts[1] if len(parts) > 1 else None)
    if args.games < 1 or args.workers < 1 or (args.minutes is not None and args.minutes <= 0):
        ap.error("games and workers must be positive; minutes must be positive when supplied")
    if args.maps and args.map_pool:
        ap.error("choose either --maps or --map-pool")
    # bridge.py captures FA_ENV_BIN_DIR during import, so set it before importing fa_env.
    if args.bin_dir:
        os.environ["FA_ENV_BIN_DIR"] = str(args.bin_dir.resolve())

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fa_env.bridge import DEFAULT_BIN_DIR
    from fa_env.maps import MAP_POOL

    try:
        specs, ours, theirs = csv(args.ai), factions(args.factions), factions(args.ai_factions)
        ais = [s.partition(":")[0] for s in specs]
        counts = {s.partition(":")[0]: int(s.partition(":")[2]) if s.partition(":")[2] else args.games for s in specs}
        maps = sorted(MAP_POOL) if args.map_pool else csv(args.maps) if args.maps else ["SCMP_016"]
        if args.maps == "small":
            maps = sorted(k for k, m in MAP_POOL.items() if m["km"] <= 10)
    except ValueError as exc:
        ap.error(str(exc))
    unknown = [name for name in maps if name not in MAP_POOL]
    if unknown:
        ap.error(f"unknown map IDs: {', '.join(unknown)}")
    checkpoint, bin_dir = args.checkpoint.resolve(), Path(DEFAULT_BIN_DIR).resolve()
    engine = bin_dir / "main.exe"
    if not checkpoint.is_file() or not engine.is_file():
        ap.error(f"checkpoint or engine missing: {checkpoint}, {engine}")
    rows = schedule_rows(ais, counts, maps, ours, theirs, args.minutes, args.seed_base, minutes_large)
    run = args.run or Path(__file__).resolve().parents[4] / "_runs" / f"tools_eval-{datetime.now():%Y%m%d-%H%M%S}"
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "engine": str(engine), "engine_sha256": sha256(engine),
                "policy": args.mode if args.mode == "greedy" else
                f"sample T={args.temperature} top_p={args.top_p}", "build_budget": args.build_budget,
                "commander_greedy": bool(args.commander_greedy), "rows": rows}
    python_root = Path(__file__).resolve().parents[1]
    policy_sources = ("fa_train/tools_model.py", "fa_train/tools_trainer.py", "fa_train/worker.py", "fa_env/tools.py",
                      "fa_env/tools_obs.py", "fa_env/tools_env.py")
    manifest["policy_source_sha256"] = {name: sha256(python_root / name) for name in policy_sources}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    import torch
    from fa_train.night import by_ai
    from fa_train.tools_trainer import ToolsConfig, ToolsTrainer

    run.mkdir(parents=True, exist_ok=True)
    (run / "eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    size = saved.get("size", "M")
    cfg = ToolsConfig(run_dir=str(run), workers=args.workers, size=size, decoder=saved.get("decoder", "slots"),
                      eval_minutes=args.minutes or 30.0, eval_mode=args.mode, eval_temperature=args.temperature,
                      eval_top_p=args.top_p, act_gather_ms=args.act_gather_ms, act_min_batch=args.act_min_batch)
    trainer = ToolsTrainer(cfg, device=args.device)
    trainer.load(checkpoint, optimizer=False)
    if args.build_budget is not None:
        trainer.acting.build_budget = None if args.build_budget < 0 else args.build_budget
    trainer.acting.commander_greedy = bool(args.commander_greedy)
    try:
        for ai in ais:
            selection = [row for row in rows if row["ai"] == ai]
            offset = len(trainer.eval_queue)
            trainer.schedule_eval(len(selection), ai)
            for episode, row in zip(trainer.eval_queue[offset:], selection):
                episode.update(row)
                episode["tag"] = "paired-greedy-eval"
                if args.watch:
                    episode["watch"] = True
                if args.watch or args.debug:
                    episode["debug_dir"] = str(run / "watch")
                episode["speed"] = args.speed
                if str(ai).startswith("self@"):         # self-play: the opponent network plays as we do
                    episode["league_dir"] = str((args.league_dir or checkpoint.resolve().parents[2] / "league"))
                    episode["opponent_mode"], episode["opponent_temperature"] = args.mode, args.temperature
                    episode["opponent_top_p"] = args.top_p
                if args.record:
                    episode["record_dir"] = str(args.record.resolve())
                if args.labels:
                    episode["labels"] = args.labels
        trainer.run("eval")
        result = {"summary": by_ai(trainer.eval_results), "games": trainer.eval_results}
        (run / "eval_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
        print(telemetry_line(run / "episodes.jsonl"))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
