"""Where does a training game's wall time go? Plays one teacher-v3 game and records, per agent step, the engine's
timing report (sim, observation, act), the Python side's time (JSON parsing, the teacher and the observer), the
observation's size and unit counts; optionally samples the sim thread with thread_profile.py at given game minutes,
and time-stamps the loading phases from the game log.

    python perf_game.py --exe C:/faf/_enginetest/bin/main.exe --out DIR [--minutes 45]
                        [--ai turtle] [--seed 31] [--profile-at 20,35,44] [--laps 1] [--fine-laps-at 30]
                        [--native 0|1|ab|check]

--laps 1 asks the bridge script to time its sections (AgentBridge_Lap; needs an engine with it). --fine-laps-at M
switches to per-unit section timing for one game minute from minute M (costly: those steps are marked). --native
selects the observation encoder: 0 the Lua one, 1 the native one, ab alternates per decision (a fair comparison in
one game), check runs both every decision and diffs them (equivalence test; see --check-out).

Writes DIR/steps.jsonl (one record per step), DIR/summary.md (tables per 5 game minutes), DIR/load.json and, with
--profile-at, DIR/profile_<min>.txt.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai" / "python"))
sys.path.insert(0, str(HERE.parent / "ai" / "python" / "examples"))
from fa_env import GameConfig  # noqa: E402
from fa_env import bridge as bridge_mod  # noqa: E402
from fa_env.env import FAEnv  # noqa: E402
from fa_env.teacher_v3 import TeacherV3  # noqa: E402
from fa_env.tools_env import ToolsEnv  # noqa: E402

LOAD_MARKERS = [
    ("init_lua", "Game version:"),
    ("user_lua", "Lua user environment initialized"),
    ("shaders", "SHADERS COMPILED"),
    ("launch", "Hooked /lua/singleplayerlaunch.lua"),
    ("map_loader", "Starting background task \"Map loader"),
    ("bp_loading", "Blueprints Loading... '*.bp' files"),
    ("bp_registering", "Blueprints Registering"),
    ("session_lua", "Lua session environment initialized"),
    ("sim_lua", "Lua sim environment initialized"),
    ("map_script", "Loading script file"),
    ("armies", "CreateArmy group ARMY_1"),
    ("nav_done", "Generated in"),
    ("bridge_enabled", "AgentBridge: enabled"),
    ("preload", "Preloading"),
    ("terrain", "creating medium fidelity terrain"),
    ("bridge_connect", "AgentBridge: connecting"),
    ("bridge_loaded", "AgentBridge: loaded"),
]


class LogWatcher(threading.Thread):
    """Time-stamps the first appearance of each load marker in the game log (polls every 20 ms)."""

    def __init__(self, path: Path, t0: float):
        super().__init__(daemon=True)
        self.path, self.t0 = path, t0
        self.seen: dict[str, float] = {}
        self.stop = False

    def run(self):
        pos, buf = 0, ""
        while not self.stop and len(self.seen) < len(LOAD_MARKERS):
            try:
                with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except OSError:
                chunk = ""
            now = time.monotonic() - self.t0
            if chunk:
                buf += chunk
                lines = buf.split("\n")
                buf = lines.pop()
                for line in lines:
                    for name, marker in LOAD_MARKERS:
                        if name not in self.seen and marker in line:
                            self.seen[name] = round(now, 3)
            time.sleep(0.02)


def install_hooks(rec: dict):
    """Time JSON parsing inside BridgeServer.recv, and let the harness add a config to the next reply."""
    orig_recv = bridge_mod.BridgeServer.recv
    orig_send = bridge_mod.BridgeServer.send

    def recv(self, timeout=None):
        if self._conn is None:
            raise bridge_mod.BridgeClosed("not connected")
        self._conn.settimeout(timeout)
        t0 = time.perf_counter()
        try:
            line = self._reader.readline()
        except OSError as exc:
            raise bridge_mod.BridgeClosed(str(exc)) from exc
        t1 = time.perf_counter()
        if not line:
            raise bridge_mod.BridgeClosed("the game closed the connection")
        self.bytes_received += len(line)
        self.last_line = line
        msg = json.loads(line.decode("utf-8", errors="replace"))
        t2 = time.perf_counter()
        rec["wait_ms"] = (t1 - t0) * 1000.0
        rec["parse_ms"] = (t2 - t1) * 1000.0
        rec["bytes"] = len(line)
        rec["recv_at"] = t2
        if msg.get("type") == "hello":
            rec["hello_load"] = msg.get("load")
        return msg

    def send(self, message):
        extra = rec.pop("next_config", None)
        if extra:
            message = dict(message)
            message["config"] = {**(message.get("config") or {}), **extra}
        rec["send_at"] = time.perf_counter()
        return orig_send(self, message)

    bridge_mod.BridgeServer.recv = recv
    bridge_mod.BridgeServer.send = send
    return orig_recv, orig_send


def summarize(steps: list[dict], out: Path, bucket_min: float = 5.0):
    """Tables per `bucket_min` game minutes."""
    buckets = collections.OrderedDict()
    for s in steps:
        if s.get("profiling") or s.get("fine") or s.get("first"):
            continue
        b = int(s["tick"] // (bucket_min * 600))
        buckets.setdefault(b, []).append(s)
    lines = []
    head = ("| game min | steps | ticks | own units | other units | obs KB | wall/step ms | sim wall/tick ms | "
            "sim CPU/tick ms | observe ms (lua+encode) | encode ms | act ms | recv wait ms | parse ms | "
            "python ms | game-s per wall-s |")
    lines.append(head)
    lines.append("|" + "|".join(["---"] * (head.count("|") - 1)) + "|")
    for b, group in buckets.items():
        def mean(key, g=group):
            vals = [x[key] for x in g if x.get(key) is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        ticks = sum(x.get("ticks", 0) for x in group)
        wall = sum(x.get("step_wall_ms", 0) for x in group)
        sim_wall = sum(x.get("simMs", 0) for x in group)
        sim_cpu = sum(x.get("simCpuMs", 0) for x in group)
        lines.append(
            f"| {b * bucket_min:.0f}-{(b + 1) * bucket_min:.0f} | {len(group)} | {ticks} | {mean('own'):.0f} | "
            f"{mean('other'):.0f} | {mean('bytes') / 1024:.0f} | {wall / len(group):.1f} | "
            f"{sim_wall / max(1, ticks):.2f} | {sim_cpu / max(1, ticks):.2f} | {mean('observeMs'):.2f} | "
            f"{mean('observeMs') - mean('observeLuaMs'):.2f} | {mean('actMs'):.2f} | {mean('wait_ms'):.1f} | "
            f"{mean('parse_ms'):.2f} | {mean('python_ms'):.1f} | {ticks / 10 / max(1e-9, wall / 1000):.1f} |")
    # share of wall time
    lines.append("")
    lines.append("Share of wall time per bucket (sim ticks / observation build / act / transfer+parse / Python):")
    lines.append("")
    lines.append("| game min | sim | observe | act | transport+parse | python |")
    lines.append("|---|---|---|---|---|---|")
    for b, group in buckets.items():
        wall = sum(x.get("step_wall_ms", 0) for x in group)
        sim = sum(x.get("simMs", 0) for x in group)
        obs = sum(x.get("observeMs", 0) for x in group)
        act = sum(x.get("actMs", 0) for x in group)
        py = sum(x.get("python_ms", 0) for x in group)
        rest = wall - sim - obs - act - py
        lines.append(f"| {b * bucket_min:.0f}-{(b + 1) * bucket_min:.0f} | {100 * sim / wall:.0f} % | "
                     f"{100 * obs / wall:.0f} % | {100 * act / wall:.0f} % | {100 * rest / wall:.0f} % | "
                     f"{100 * py / wall:.0f} % |")
    lap_rows = collections.OrderedDict()
    for b, group in buckets.items():
        laps = collections.Counter()
        n = 0
        for x in group:
            if x.get("laps"):
                laps.update(x["laps"])
                n += 1
        if n:
            lap_rows[b] = {k: v / n for k, v in laps.items()}
    if lap_rows:
        names = sorted({k for r in lap_rows.values() for k in r})
        lines.append("")
        lines.append("Bridge script sections (ms per observation, AgentBridge_Lap):")
        lines.append("")
        lines.append("| game min | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for b, r in lap_rows.items():
            lines.append(f"| {b * bucket_min:.0f}-{(b + 1) * bucket_min:.0f} | " +
                         " | ".join(f"{r.get(k, 0):.2f}" for k in names) + " |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def diff_units(lua_units: list, native_units: list, tol: float = 1e-4) -> list[str]:
    """Semantic differences between two unit lists (floats compared with a relative tolerance)."""
    problems = []
    if len(lua_units) != len(native_units):
        problems.append(f"count {len(lua_units)} != {len(native_units)}")
    by_id_l = {(u.get("id"), bool(u.get("radar")), bool(u.get("ghost"))): u for u in lua_units}
    by_id_n = {(u.get("id"), bool(u.get("radar")), bool(u.get("ghost"))): u for u in native_units}
    order_l = [u.get("id") for u in lua_units]
    order_n = [u.get("id") for u in native_units]
    if order_l != order_n and set(order_l) == set(order_n):
        problems.append("order differs")
    for key in by_id_l.keys() | by_id_n.keys():
        a, b = by_id_l.get(key), by_id_n.get(key)
        if a is None or b is None:
            problems.append(f"{key}: only in {'native' if a is None else 'lua'}")
            continue
        for f in a.keys() | b.keys():
            va, vb = a.get(f), b.get(f)
            if not same(va, vb, tol):
                problems.append(f"{key[0]} {f}: lua={va!r} native={vb!r}")
    return problems


def same(a, b, tol) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return a == b or abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=r"C:\faf\_enginetest\bin\main.exe")
    ap.add_argument("--script", default=str(HERE.parent / "ai" / "agentbridge.lua"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--ai", default="turtle")
    ap.add_argument("--faction", type=int, default=1)
    ap.add_argument("--ai-faction", type=int, default=2)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--map", default="SCMP_016")
    ap.add_argument("--profile-at", default="", help="game minutes at which to sample the sim thread")
    ap.add_argument("--profile-seconds", type=float, default=8.0)
    ap.add_argument("--laps", type=int, default=0)
    ap.add_argument("--fine-laps-at", type=float, default=-1)
    ap.add_argument("--native", default="", help="0, 1, ab or check (default: the script's default)")
    ap.add_argument("--check-every", type=int, default=1, help="with --native check: diff every Nth decision")
    ap.add_argument("--load-only", action="store_true", help="measure loading, then quit")
    ap.add_argument("--profile-load-at", type=float, default=-1,
                    help="seconds after launch at which to sample the busiest thread for --profile-seconds")
    ap.add_argument("--profile-load-rank", type=int, default=1)
    ap.add_argument("--extra", default="", help="extra game arguments, comma separated")
    ap.add_argument("--no-lean", action="store_true", help="load everything (no /agentlean)")
    ap.add_argument("--digest", action="store_true", help="record a fingerprint of every observation")
    ap.add_argument("--mount-root", action="store_true", help="mount the overrides folder at / (the old init file)")
    ap.add_argument("--dump-ticks", type=lambda t: [int(x) for x in t.split("-")], default=None,
                    help="A-B: save observations and orders of these ticks to dump.jsonl")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mount_root:      # the old init file: the overrides folder mounted at / (for load-time comparisons)
        orig_write = bridge_mod.write_init_file

        def write_init_file(path, *args, **kwargs):
            p = orig_write(path, *args, **kwargs)
            text = p.read_text(encoding="utf-8")
            schook = (bridge_mod.OVERRIDES_DIR / "schook").as_posix()
            text = text.replace(f"dir = [[{schook}]], mountpoint = '/schook'",
                                f"dir = [[{bridge_mod.OVERRIDES_DIR.as_posix()}]], mountpoint = '/'")
            p.write_text(text, encoding="utf-8")
            return p

        bridge_mod.write_init_file = write_init_file
    exe = Path(a.exe)
    rec: dict = {}
    install_hooks(rec)

    config = GameConfig(map=a.map, faction=a.faction, ai=a.ai, ai_faction=a.ai_faction, seed=a.seed,
                        window=(640, 480), minimized=True, exe=exe, bin_dir=exe.parent, script=Path(a.script),
                        log_dir=out / "logs", extra_args=[x for x in a.extra.split(",") if x], lean=not a.no_lean)
    env = ToolsEnv(config, time_limit_minutes=a.minutes)
    teacher = TeacherV3()

    # ---- loading
    t0 = time.monotonic()
    watcher_holder = {}
    orig_process_init = bridge_mod.GameProcess.__init__

    def process_init(self, cfg, port):
        orig_process_init(self, cfg, port)
        w = LogWatcher(self.log_path, t0)
        w.start()
        watcher_holder["w"] = w
        watcher_holder["pid"] = self.process.pid
        if a.profile_load_at >= 0:        # sample the busiest thread while the game loads
            pid_ = self.process.pid

            def later():
                time.sleep(a.profile_load_at)
                with open(out / f"profile_load_{a.profile_load_at:.0f}s.txt", "w", encoding="utf-8") as pf:
                    subprocess.run([sys.executable, str(HERE / "thread_profile.py"), "--pid", str(pid_), "--seconds",
                                    str(a.profile_seconds), "--top", "400", "--exe", str(exe), "--rank",
                                    str(a.profile_load_rank)], stdout=pf, stderr=subprocess.STDOUT)
            threading.Thread(target=later, daemon=True).start()

    bridge_mod.GameProcess.__init__ = process_init
    env.reset()
    t_reset = time.monotonic() - t0
    w = watcher_holder.get("w")
    if w:
        w.stop = True
    load = {"reset_s": round(t_reset, 3), "load_seconds": env.fa.info.get("load_seconds"),
            "engine_marks_ms": rec.get("hello_load"), "log_markers": w.seen if w else {}, "exe": str(exe)}
    (out / "load.json").write_text(json.dumps(load, indent=1), encoding="utf-8")
    print("load:", json.dumps(load), flush=True)
    pid = watcher_holder.get("pid")
    if a.load_only:
        env.close()
        return

    teacher.reset(env)
    base_config = {}
    if a.laps:
        base_config["profile"] = a.laps
    native_mode = a.native
    if native_mode in ("0", "1"):
        base_config["nativeObs"] = native_mode == "1"
    if native_mode == "check":
        base_config["obsCheck"] = True
    if base_config:
        rec["next_config"] = dict(base_config)

    profile_at = [float(x) for x in a.profile_at.split(",") if x.strip()]
    profiling = {"proc": None, "until": 0.0}
    steps_f = open(out / "steps.jsonl", "w", encoding="utf-8")
    check_f = open(out / "check.jsonl", "w", encoding="utf-8") if native_mode == "check" else None
    steps = []
    check_stats = collections.Counter()
    coverage = collections.Counter()

    # wrap FAEnv.step: one record per bridge step
    orig_step = FAEnv.step
    state = {"last_recv": None, "tick": 0, "decision": 0, "fine": False}

    def step(self, actions=(), queries=(), wait=None):
        send_t = time.perf_counter()
        python_ms = (send_t - state["last_recv"]) * 1000.0 if state["last_recv"] else None
        prev_tick = state["tick"]
        result = orig_step(self, actions, queries, wait)
        state["last_recv"] = rec.get("recv_at")
        msg = self.last_message or {}
        obs = msg.get("obs") or {}
        timing = msg.get("timing") or {}
        if msg.get("errors"):
            print("bridge errors:", msg["errors"][:3], flush=True)
        me = obs.get("me", 1)
        units = obs.get("units") or []
        own = sum(1 for u in units if u.get("army") == me)
        r = {"tick": msg.get("tick", 0), "ticks": msg.get("tick", 0) - prev_tick, "own": own,
             "other": len(units) - own, "bytes": rec.get("bytes"), "wait_ms": rec.get("wait_ms"),
             "parse_ms": rec.get("parse_ms"), "python_ms": python_ms,
             "step_wall_ms": (rec.get("recv_at", send_t) - (rec.get("send_at") or send_t)) * 1000.0
             + (python_ms or 0.0),
             "profiling": bool(profiling["proc"] and profiling["proc"].poll() is None),
             "fine": state["fine"], "native": state.get("native_now"), "first": prev_tick == 0}
        for k, v in timing.items():
            r[k] = v
        if a.dump_ticks and a.dump_ticks[0] <= r["tick"] <= a.dump_ticks[1]:
            with open(out / "dump.jsonl", "a", encoding="utf-8") as df:
                df.write(json.dumps({"tick": r["tick"], "obs": obs, "sent": list(actions), "queries": list(queries),
                                     "result": msg.get("result")}, sort_keys=True) + "\n")
        if a.digest:        # a fingerprint of the observation: equal runs play the same game
            import hashlib
            r["digest"] = hashlib.blake2b(json.dumps(obs, sort_keys=True).encode(), digest_size=8).hexdigest()
        state["tick"] = r["tick"]
        steps.append(r)
        steps_f.write(json.dumps(r) + "\n")
        if native_mode == "check" and "units_native" in obs:
            problems = diff_units(obs.get("units") or [], obs.get("units_native") or [])
            if not same(obs.get("privileged"), obs.get("privileged_native"), 1e-4):
                problems.append(f"privileged: lua={obs.get('privileged')!r} native={obs.get('privileged_native')!r}")
            check_stats["checked"] += 1
            check_stats["units"] += len(obs.get("units") or [])
            check_stats["exact_json"] += int(json.dumps(obs.get("units"), sort_keys=True)
                                             == json.dumps(obs.get("units_native"), sort_keys=True))
            for u in obs.get("units_native") or []:      # which fields the comparison covered
                coverage.update(k for k in u if k not in ("id", "army", "x", "y", "z"))
            if problems:
                check_stats["with_diffs"] += 1
                check_f.write(json.dumps({"tick": r["tick"], "problems": problems[:50], "n": len(problems)}) + "\n")
                check_f.flush()
        return result

    FAEnv.step = step
    done = False
    t_game = time.monotonic()
    last_print = -1
    fine_on = False
    try:
        while not done:
            minute = state["tick"] / 600.0
            # sim-thread sampling at the requested minutes
            if profile_at and minute >= profile_at[0] and pid:
                m = profile_at.pop(0)
                pf = open(out / f"profile_{m:.0f}.txt", "w", encoding="utf-8")
                profiling["proc"] = subprocess.Popen(
                    [sys.executable, str(HERE / "thread_profile.py"), "--pid", str(pid), "--seconds",
                     str(a.profile_seconds), "--top", "400", "--exe", str(exe)], stdout=pf, stderr=subprocess.STDOUT)
            if a.fine_laps_at >= 0 and not fine_on and minute >= a.fine_laps_at:
                fine_on = True
                state["fine"] = True
                rec["next_config"] = {**rec.get("next_config", {}), "profile": 2}
            if fine_on and state["fine"] and minute >= a.fine_laps_at + 1:
                state["fine"] = False
                rec["next_config"] = {**rec.get("next_config", {}), "profile": a.laps}
            if native_mode == "ab":
                state["decision"] += 1
                now_native = state["decision"] % 2 == 0
                state["native_now"] = now_native
                rec["next_config"] = {**rec.get("next_config", {}), "nativeObs": now_native}
            cmds = env.teacher_commands(teacher, dry=False)
            _, _, done, info = env.step_tools(cmds)
            if int(minute) // 5 != last_print:
                last_print = int(minute) // 5
                print(f"[{minute:5.1f} min] units {steps[-1]['own']}+{steps[-1]['other']} | "
                      f"{time.monotonic() - t_game:.0f} s | {dict(check_stats)}", flush=True)
    finally:
        FAEnv.step = orig_step
        steps_f.close()
        if check_f:
            check_f.close()
        env.close()
    total = time.monotonic() - t_game
    print(f"game: {state['tick'] / 600:.1f} game-min in {total:.0f} s wall; outcome {info.get('outcome')}; "
          f"check {dict(check_stats)}", flush=True)
    if coverage:
        print("fields compared (unit records carrying each):", dict(coverage.most_common()), flush=True)
    text = summarize(steps, out)
    if native_mode == "ab":
        for flag in (False, True):
            sub = [s for s in steps if s.get("native") is flag]
            (out / f"steps_native{int(flag)}.jsonl").write_text("\n".join(json.dumps(s) for s in sub),
                                                               encoding="utf-8")
            sub_out = out / f"native{int(flag)}"
            sub_out.mkdir(exist_ok=True)
            summarize(sub, sub_out)
    print(text)
    (out / "result.json").write_text(json.dumps({"wall_s": total, "ticks": state["tick"],
                                                 "outcome": info.get("outcome"), "check": dict(check_stats),
                                                 "coverage": dict(coverage),
                                                 "load": load}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
