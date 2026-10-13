"""Does a teacher-driven game replay identically? Plays the same game (same seed) twice with TeacherV3 acting and
compares, step by step, the sim's own per-beat checksums (the bridge's "sums": the desync check's MD5 of every
beat), the observations, the teacher's commands and anything the command stream carried besides the agent's own
orders ("stream", e.g. commands from the game's UI).

    python teacher_determinism.py --exe C:/faf/_enginetest/bin_det/main.exe --minutes 30
    python teacher_determinism.py --compare run_a.jsonl run_b.jsonl

Each run is its own Python process with a different PYTHONHASHSEED, so a teacher that depended on the iteration
order of a set (or anything else process-specific) would show up as different commands for identical
observations. One game at a time. Exit code 0 when both runs agree on every beat.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai" / "python"))


def digest(value) -> str:
    return hashlib.blake2b(json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), digest_size=8).hexdigest()


def play(args) -> None:
    """One game: TeacherV3 plays through ToolsEnv; every bridge message is logged to args.out (JSON lines)."""
    from fa_env import GameConfig
    from fa_env.teacher_v3 import TeacherV3
    from fa_env.tools_env import ToolsEnv

    exe = Path(args.exe)
    config = GameConfig(map=args.map, faction=args.faction, ai=args.ai, ai_faction=args.ai_faction, seed=args.seed,
                        cheat=args.cheat,
                        window=(640, 480), minimized=True, exe=exe, bin_dir=exe.parent,
                        script=Path(args.script) if args.script else GameConfig.script,
                        extra_args=(args.extra_str or "").split())
    env = ToolsEnv(config, time_limit_minutes=args.minutes)
    out = open(args.out, "w", encoding="utf-8")
    obs_out = gzip.open(str(args.out) + ".obs.gz", "wt", encoding="utf-8") if args.obs_log else None
    fa_step = env.fa.step

    def logged_step(actions=(), queries=(), wait=None):
        actions, queries = list(actions), list(queries)
        result = fa_step(actions, queries, wait)
        m = env.fa.last_message or {}
        obs = m.get("obs") or {}
        record = {"tick": m.get("tick"), "beat": m.get("beat"), "sums": m.get("sums"), "stream": m.get("stream"),
                  "sent": digest([actions, queries, wait]), "obs": digest(obs),
                  "result": digest(m.get("result")), "errors": m.get("errors")}
        if args.full:
            record["actions"], record["queries"] = actions, queries
        out.write(json.dumps(record) + "\n")
        if obs_out is not None:
            obs_out.write(json.dumps({"tick": m.get("tick"), "obs": obs}, sort_keys=True) + "\n")
        return result

    env.fa.step = logged_step
    teacher = TeacherV3()
    t0 = time.monotonic()
    try:
        env.reset()
        hello = env.fa.last_message or {}
        out.write(json.dumps({"hello": True, "tick": hello.get("tick"), "beat": hello.get("beat"),
                              "sums": hello.get("sums"), "stream": hello.get("stream"),
                              "log": str(env.fa.game.log_path) if env.fa.game else None,
                              "args": env.fa.game.args if env.fa.game else None}) + "\n")
        teacher.reset(env)
        done = False
        while not done:
            commands = env.teacher_commands(teacher, dry=False)
            out.write(json.dumps({"commands": digest(commands), "n": len(commands),
                                  **({"cmds": commands} if args.full else {})}) + "\n")
            _, _, done, step = env.step_tools(commands)
        out.write(json.dumps({"end": step.get("episode"), "wall_s": round(time.monotonic() - t0, 1)}) + "\n")
    finally:
        env.close()
        out.close()
        if obs_out is not None:
            obs_out.close()


def load(path: Path) -> dict:
    """Beat -> checksum, the per-step records in order, and every stream entry."""
    sums, steps, stream, commands, meta = {}, [], [], [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "commands" in r:
            commands.append(r)
            continue
        if "end" in r or "hello" in r:
            meta.update(r)
            continue
        steps.append(r)
        for beat, value in r.get("sums") or []:
            sums[int(beat)] = value
        for entry in r.get("stream") or []:
            stream.append(tuple(entry))
    return {"sums": sums, "steps": steps, "stream": stream, "commands": commands, "meta": meta}


def obs_at(path: Path, tick) -> dict | None:
    obs_path = Path(str(path) + ".obs.gz")
    if not obs_path.exists():
        return None
    with gzip.open(obs_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["tick"] == tick:
                return r["obs"]
    return None


def show_obs_diff(a_path: Path, b_path: Path, tick, limit: int = 12) -> None:
    """The units (and army entries) that differ in the first differing observation, field by field."""
    oa, ob = obs_at(a_path, tick), obs_at(b_path, tick)
    if oa is None or ob is None:
        return
    ua = {(u.get("id"), bool(u.get("radar")), bool(u.get("ghost"))): u for u in oa.get("units") or []}
    ub = {(u.get("id"), bool(u.get("radar")), bool(u.get("ghost"))): u for u in ob.get("units") or []}
    shown = 0
    print(f"observation at tick {tick}: units {len(ua)} / {len(ub)}")
    for key in sorted(ua.keys() | ub.keys(), key=str):
        x, y = ua.get(key), ub.get(key)
        if x == y:
            continue
        if x is None or y is None:
            print("  only in", "b" if x is None else "a", key, (x or y).get("bp"))
        else:
            fields = {f: (x.get(f), y.get(f)) for f in x.keys() | y.keys() if x.get(f) != y.get(f)}
            print("  unit", key[0], x.get("bp"), "army", x.get("army"), fields)
        shown += 1
        if shown >= limit:
            break
    for aa, ab in zip(oa.get("armies") or [], ob.get("armies") or []):
        if aa != ab:
            print("  army", aa.get("index"), "differs:",
                  json.dumps({k: (aa.get(k), ab.get(k)) for k in aa if aa.get(k) != ab.get(k)})[:600])


def compare(a_path: Path, b_path: Path) -> int:
    a, b = load(a_path), load(b_path)
    common = sorted(set(a["sums"]) & set(b["sums"]))
    first_beat = next((beat for beat in common if a["sums"][beat] != b["sums"][beat]), None)
    first_step = next((i for i, (x, y) in enumerate(zip(a["steps"], b["steps"]))
                       if (x["tick"], x["obs"]) != (y["tick"], y["obs"])), None)
    first_cmd = next((i for i, (x, y) in enumerate(zip(a["commands"], b["commands"])) if x["commands"] != y["commands"]),
                     None)
    first_sent = next((i for i, (x, y) in enumerate(zip(a["steps"], b["steps"])) if x["sent"] != y["sent"]), None)
    report = {
        "beats_compared": len(common), "last_common_beat": common[-1] if common else None,
        "first_differing_beat": first_beat,
        "first_differing_observation": None if first_step is None else a["steps"][first_step]["tick"],
        "first_differing_teacher_decision": first_cmd,
        "first_differing_sent_step_tick": None if first_sent is None else a["steps"][first_sent]["tick"],
        "stream_entries": [len(a["stream"]), len(b["stream"])],
        "stream_identical": a["stream"] == b["stream"],
        "ends": [a["meta"].get("end"), b["meta"].get("end")],
    }
    print(json.dumps(report, indent=1))
    if first_step is not None:
        show_obs_diff(a_path, b_path, a["steps"][first_step]["tick"])
    if a["stream"] or b["stream"]:
        print("stream (first 20 of run a):", a["stream"][:20])
        if a["stream"] != b["stream"]:
            print("stream (first 20 of run b):", b["stream"][:20])
    return 0 if first_beat is None and common else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=r"C:\faf\_enginetest\bin_det\main.exe")
    ap.add_argument("--script", help="bridge script (default: the fa_env one)")
    ap.add_argument("--map", default="SCMP_016")
    ap.add_argument("--faction", type=int, default=1)
    ap.add_argument("--ai", default="medium")
    ap.add_argument("--ai-faction", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--cheat", type=float, help="AIx multiplier (the cheating AIs)")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--out-dir", default=str(Path(os.environ.get("TEMP", ".")) / "teacher_determinism"))
    ap.add_argument("--full", action="store_true", help="also log the actions and commands themselves")
    ap.add_argument("--extra", nargs="*", help="extra game arguments")
    ap.add_argument("--extra-a", nargs="*", default=[], help="extra game arguments of run a only (e.g. /agentpoison 0)")
    ap.add_argument("--extra-b", nargs="*", default=[], help="extra game arguments of run b only")
    ap.add_argument("--obs-log", action="store_true", help="also keep every observation (gzip), to show what differs")
    ap.add_argument("--extra-str", help=argparse.SUPPRESS)  # internal: the child's game arguments
    ap.add_argument("--out", help=argparse.SUPPRESS)          # internal: play one game into this file
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(Path(a.compare[0]), Path(a.compare[1])))
    if a.out:
        play(a)
        return
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths = []
    for run, hash_seed, own in (("a", "1", a.extra_a), ("b", "2", a.extra_b)):
        path = out_dir / f"{stamp}_{run}.jsonl"
        paths.append(path)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--out", str(path), "--exe", a.exe, "--map", a.map,
               "--faction", str(a.faction), "--ai", a.ai, "--ai-faction", str(a.ai_faction), "--seed", str(a.seed),
               "--minutes", str(a.minutes), "--extra-str", " ".join((a.extra or []) + list(own))] + \
            (["--script", a.script] if a.script else []) + (["--full"] if a.full else []) + \
            (["--cheat", str(a.cheat)] if a.cheat is not None else []) + \
            (["--obs-log"] if a.obs_log else [])
        t0 = time.monotonic()
        subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONHASHSEED=hash_seed))
        print(f"run {run}: {time.monotonic() - t0:.0f} s -> {path}", flush=True)
    sys.exit(compare(*paths))


if __name__ == "__main__":
    main()
