"""Game worker: one process, one game at a time, driven by the learner over a pipe.

Protocol (learner -> worker):  ("reset", episode) | ("act", choice[, policy]) | ("close",) | ("stop",)
            (worker -> learner):  ("obs", obs, rewards | None, done, info) | ("error", text, uid)

"close" ends the running game without a reply (the learner no longer needs it, e.g. between phases).

`episode` is a dict: map, faction, ai, ai_faction, seed, minutes, mode ("policy" | "rule"), eval, fake.
In "rule" mode every observation carries `info["rule_action"]`, the rule policy's choice for it, which the
learner sends back (behaviour-cloning collection and the rule baseline). Any exception in the game -- a
crash, a hang caught by the bridge watchdog, a closed connection -- becomes an ("error", ...) message and
the game is torn down; the learner cuts the episode and sends a new reset.
"""
from __future__ import annotations

import os
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fa_env.observation_precision import compact_observation


class FakeEnv:
    """Same interface and shapes as FactoredEnv, no game: for testing the training machinery at scale."""

    def __init__(self, episode: dict):
        from fa_env.factored import HEADS, MAX_UNITS, observation_sizes
        self.heads, self.max_units = HEADS, MAX_UNITS
        self.sizes = observation_sizes()
        self.rng = np.random.default_rng(episode.get("seed", 0))
        self.length = int(self.rng.integers(40, 120))
        self.t = 0
        self.delay = float(episode.get("fake_delay", 0.005))

    def reset(self):
        self.t = 0
        return self._obs(), {"tick": 0, "load_seconds": 0.0}

    def _obs(self):
        rng = self.rng
        n = int(rng.integers(5, 60))
        mask = np.zeros(self.max_units, bool)
        mask[:n] = True
        out = {
            "scalars": rng.normal(size=self.sizes["scalars"]).astype(np.float32),
            "units": rng.normal(size=(self.max_units, self.sizes["units"])).astype(np.float32),
            "unit_bp": rng.integers(0, 700, self.max_units).astype(np.int64),
            "unit_group": rng.integers(0, 4, self.max_units).astype(np.int64),
            "unit_mask": mask,
            "privileged": rng.normal(size=self.sizes["privileged"]).astype(np.float32),
        }
        for h, opts in self.heads.items():
            m = rng.random(len(opts)) < 0.5
            m[0] = True
            out[f"mask_{h}"] = m
        return out

    def rule_action(self, obs):
        return {h: int(np.flatnonzero(obs[f"mask_{h}"])[-1]) for h in self.heads}

    def step(self, choice):
        time.sleep(self.delay)
        self.t += 1
        done = self.t >= self.length
        good = sum(choice.values())
        rewards = {"trade": 0.01 * good, "economy": 0.0, "terminal": (10.0 if good > 8 else -10.0) if done else 0.0}
        info = {"tick": self.t * 20, "outcome": ("victory" if good > 8 else "defeat") if done else None,
                "action_errors": 0, "errors": [], "decisions": self.t}
        if done:
            info["episode"] = {"outcome": info["outcome"], "minutes": self.t / 30, "mass_income": 0.0,
                               "mass_destroyed": 0, "mass_lost": 0, "value_advantage": 0.0, "own_units": 0}
        return self._obs(), rewards, done, info

    def close(self):
        pass


def opponent(key: str | None) -> tuple[str | None, float | None]:
    """An opponent key -> (AI personality, AIx multiplier): 'medium' -> ('medium', None), 'rushcheat@1.2' ->
    ('rushcheat', 1.2). Keys stay whole everywhere else (logs, win rates), so each multiplier counts apart."""
    if key and "@" in key:
        ai, cheat = key.split("@", 1)
        return ai, float(cheat)
    return key, None


def compact(obs: dict) -> dict:
    """Tools observations cross the pipe at half size (float16); the learner widens them again."""
    return compact_observation(obs)


def preserve_eval_logs(log_dir: Path, log_root: Path, worker_id: int, uid) -> list[dict]:
    """Archive closed evaluation logs with durable identities before worker cleanup."""
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return []
    # Keep the episode identity beside the source until its archive succeeds. A restarted
    # worker must recover this episode before its normal cleanup or next game reuses the folder.
    pending = log_dir / "pending_eval.json"
    pending_temp = log_dir / "pending_eval.json.tmp"
    pending_temp.write_text(
        json.dumps({"worker_id": worker_id, "uid": uid}), encoding="utf-8")
    pending_temp.replace(pending)
    # UID can be absent or reused after resuming a run. Never overwrite an earlier episode.
    destination = log_root / "episodes" / f"worker{worker_id:02d}_{uuid.uuid4().hex}"
    destination.mkdir(parents=True)
    artifacts = []
    for source in logs:
        target = destination / source.name
        shutil.copy2(source, target)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        artifacts.append({"path": str(target.resolve()), "sha256": digest.hexdigest(),
                          "bytes": target.stat().st_size})
    # Failed/interrupted episodes have no terminal result, so retain their UID mapping on disk too.
    (destination / "manifest.json").write_text(
        json.dumps({"worker_id": worker_id, "uid": uid, "game_logs": artifacts}, indent=2), encoding="utf-8")
    return artifacts


class FakeToolsEnv:
    """Tools-mode stand-in for a game: replays a recorded teacher game (real observations, masks and teacher
    labels), ignores the commands it gets, and ends the game with a random result. For machinery tests."""

    def __init__(self, episode: dict):
        from fa_env.record import read_blueprints, read_game
        from fa_env.tools import ToolKit
        from fa_env.tools_obs import ToolsObserver
        folder = Path(episode["fake_raw"])
        paths = sorted(folder.glob("*.jsonl.gz"))
        rng = np.random.default_rng(episode.get("seed", 0))
        bps = {b["id"]: b for b in read_blueprints(folder)}
        hello, self.steps, _ = read_game(paths[int(rng.integers(len(paths)))])
        info, self.me = hello["hello"], hello["meta"]["me"]
        self.kit = ToolKit(info, bps)
        self.observer = ToolsObserver(info, bps, self.me, self.kit)
        self.win = bool(rng.random() < 0.5)
        self.length = min(len(self.steps), int(rng.integers(60, 200)))
        self.t = 0
        self.delay = float(episode.get("fake_delay", 0.005))

    def _obs(self):
        return self.observer.observe(self.steps[self.t]["msg"]["obs"])

    def reset(self):
        self.t = 0
        return self._obs(), {"tick": 0, "load_seconds": 0.0}

    def teacher_commands(self, policy=None, dry=True):
        st = self.steps[self.t]
        units = {str(u["id"]): u for u in st["msg"]["obs"]["units"] if u.get("army") == self.me}
        return self.kit.commands_from_actions(st["step"].get("sent") or [], units)

    def step_tools(self, commands):
        time.sleep(self.delay)
        self.observer.note_commands(commands, self.t * 20)
        self.t += 1
        done = self.t >= self.length - 1
        rewards = {"trade": 0.002 * len(commands), "economy": 0.0,
                   "terminal": (10.0 if self.win else -10.0) if done else 0.0}
        info = {"tick": self.t * 20, "outcome": ("victory" if self.win else "defeat") if done else None,
                "action_errors": 0, "errors": [], "decisions": self.t}
        if done:
            info["episode"] = {"outcome": info["outcome"], "minutes": self.t / 30, "mass_income": 0.0,
                               "mass_destroyed": 0, "mass_lost": 0, "value_advantage": 0.0, "own_units": 0}
        return self._obs(), rewards, done, info

    def close(self):
        pass


def worker_main(worker_id: int, conn, log_root: str) -> None:
    from fa_env.bridge import GameConfig
    from fa_env.factored import FactoredEnv
    from fa_env.rule_policy import RulePolicy

    env = None
    rule = None
    tools = False       # tools episodes: commands in, ToolsObserver observations out
    teacher = None      # tools episodes with teacher labels: the teacher v3, asked every decision
    shadow = False      # ... or M28 in shadow mode: its labels arrive with the next observation
    selfplay = None     # a self-play game (opponent "self@<network>"): fa_env.selfplay.SelfPlayGame
    opponent_player = None      # ... its other side: a frozen network (fa_train.selfplay_opponent)
    opponent_obs = None
    uid = None          # the learner's id of the running episode, echoed in every message
    failures = 0        # consecutive failed games: a systemic problem must not become a relaunch storm
    eval_logs = False   # only real evaluation episodes are retained without the training failure cap
    log_dir = Path(log_root) / f"worker{worker_id:02d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = log_dir / "pending_eval.json"
    if pending.exists():
        identity = json.loads(pending.read_text(encoding="utf-8"))
        preserve_eval_logs(log_dir, Path(log_root), identity["worker_id"], identity["uid"])
        # Delete only after successful recovery; any error leaves the source and identity intact.
        for source in log_dir.glob("*.log"):
            source.unlink()
        pending.unlink()

    def close_env(keep_logs: bool):
        nonlocal env, eval_logs, selfplay, opponent_player, opponent_obs
        selfplay = opponent_player = opponent_obs = None
        game = getattr(getattr(env, "fa", None), "game", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                # A recorder close failure must not leave the engine writing while we archive its log.
                if eval_logs and game is not None:
                    game.stop()
            env = None
        artifacts = []
        if eval_logs:
            # If preservation fails, propagate before deleting the source logs. The exception handler
            # retries preservation and the worker never silently discards an evaluation log.
            artifacts = preserve_eval_logs(log_dir, Path(log_root), worker_id, uid)
            eval_logs = False
            keep_logs = False  # already durable; reserve the bounded failure directory for training
        # Thousands of games a night: keep only the logs of games that failed, moved aside (a bounded number).
        failed_root = Path(log_root) / "failed"
        if keep_logs and (not failed_root.exists() or sum(1 for _ in failed_root.iterdir()) < 200):
            failed = failed_root / f"worker{worker_id:02d}_{time.strftime('%H%M%S')}"
            failed.mkdir(parents=True, exist_ok=True)
            for f in log_dir.glob("*.log"):
                try:
                    shutil.move(str(f), str(failed / f.name))
                except OSError:
                    pass
        for f in log_dir.glob("*"):
            try:
                f.unlink() if f.is_file() else shutil.rmtree(f, ignore_errors=True)
            except OSError:
                pass
        return artifacts

    def reset_tools(episode: dict):
        """Start a tools episode (commands in, ToolsObserver observations out), optionally with teacher labels."""
        nonlocal env, teacher, shadow, selfplay, opponent_player, opponent_obs
        from fa_train.tools_distillation import episode_teacher
        from fa_train.tools_cadence import LEGACY_DECISION_TICKS, decision_ticks
        source = episode_teacher(episode)  # validate before constructing any environment/game
        cadence = decision_ticks(episode.get("decision_ticks", LEGACY_DECISION_TICKS))
        teacher = None
        shadow = source == "m28" and not episode.get("fake")
        selfplay = opponent_player = opponent_obs = None
        if episode.get("fake"):
            env = FakeToolsEnv(episode)
        elif str(episode.get("ai") or "").startswith("self@"):
            # self-play: a past network of ours on army 2, played here on the CPU; both armies M28 brains in shadow
            # mode, so this side's labels are M28's as in every game
            from fa_env.selfplay import SelfPlayGame
            from fa_train.selfplay_opponent import OpponentPlayer
            name = str(episode["ai"]).split("@", 1)[1]
            opponent_player = OpponentPlayer(Path(episode.get("league_dir") or ".") / f"{name}.pt",
                                             mode=episode.get("opponent_mode", "sample"),
                                             temperature=float(episode.get("opponent_temperature", 0.5)),
                                             top_p=episode.get("opponent_top_p"))
            config = GameConfig(map=episode.get("map", "SCMP_016"), faction=episode.get("faction", 1),
                                ai="m28ai", ai_faction=episode.get("ai_faction"), seed=episode.get("seed"),
                                slots=tuple(episode["slots"]) if episode.get("slots") else None,
                                window=(1600, 900) if episode.get("watch") else (640, 480), log_dir=log_dir,
                                minimized=not episode.get("watch"), speed=int(episode.get("speed", 50)),
                                player_ai="m28ai")
            selfplay = SelfPlayGame(config, decision_ticks=cadence, time_limit_minutes=episode.get("minutes", 20.0),
                                    deduplicate_navigation=episode.get("deduplicate_navigation", False))
            env = selfplay.sides[0]
            shadow = source == "m28"
            if episode.get("record_dir"):
                from fa_env.record import Recorder
                env.recorder = Recorder(Path(episode["record_dir"]),
                                        f"{uid or 0:07d}_{episode.get('ai')}_{episode.get('seed')}",
                                        extra_meta={"opponent": episode.get("ai"),
                                                    **({"label_source": source} if source != "none" else {})})
        else:
            from fa_env.tools_env import ToolsEnv
            ai, cheat = opponent(episode.get("ai"))
            config = GameConfig(map=episode.get("map", "SCMP_016"), faction=episode.get("faction", 1),
                                ai=ai, cheat=cheat, ai_faction=episode.get("ai_faction"), seed=episode.get("seed"),
                                slots=tuple(episode["slots"]) if episode.get("slots") else None,
                                window=(1600, 900) if episode.get("watch") else (640, 480), log_dir=log_dir,
                                minimized=not episode.get("watch"), speed=int(episode.get("speed", 50)),
                                player_ai="m28ai" if shadow else None)
            env = ToolsEnv(config, time_limit_minutes=episode.get("minutes", 20.0),
                           decision_ticks=cadence,
                           deduplicate_navigation=episode.get("deduplicate_navigation", False),
                           **({"capture": True, "shadow": True} if shadow else {}))
            if episode.get("record_dir"):
                from fa_env.record import Recorder
                env.recorder = Recorder(Path(episode["record_dir"]),
                                        f"{uid or 0:07d}_{episode.get('ai')}_{episode.get('seed')}",
                                        extra_meta={"label_source": source} if source != "none" else None)
        if selfplay is not None:
            (obs, info), (opponent_obs, _) = selfplay.reset()
        else:
            obs, info = env.reset()
        if source == "v3":
            from fa_env.teacher_v3 import TeacherV3
            teacher = True if episode.get("fake") else TeacherV3()
            if teacher is not True:
                teacher.reset(env)
        game = getattr(getattr(env, "fa", None), "game", None)
        info = dict(info, first=True, uid=uid, pid=game.process.pid if game else None,
                    decision_ticks=cadence,
                    distill_teacher=episode.get("distill_teacher", "v3"),
                    teacher_labels=teacher is not None or shadow)
        if teacher is not None:
            info["teacher"] = env.teacher_commands(teacher)
            if teacher is not True:
                env.step_labels = info["teacher"]
        return obs, info

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "close":
            close_env(keep_logs=False)
            uid = None
            continue
        try:
            if kind == "reset":
                episode = msg[1]
                close_env(keep_logs=False)
                uid = episode.get("uid")
                eval_logs = bool(episode.get("eval") and not episode.get("fake"))
                if failures:
                    time.sleep(min(120.0, 5.0 * 2 ** (failures - 1)))
                tools = episode.get("kind") == "tools"
                if tools:
                    obs, info = reset_tools(episode)
                    failures = 0
                    last_obs = obs                  # the observation the next decision is made on (watch debugger)
                    conn.send(("obs", compact(obs), None, False, info))
                    continue
                if episode.get("fake"):
                    env = FakeEnv(episode)
                else:
                    # minimized: a game then draws nothing (1.3 instead of 1.7 cores at full speed)
                    ai, cheat = opponent(episode.get("ai"))
                    config = GameConfig(map=episode.get("map", "SCMP_016"), faction=episode.get("faction", 1),
                                        ai=ai, cheat=cheat, ai_faction=episode.get("ai_faction"),
                                        seed=episode.get("seed"), window=(640, 480), log_dir=log_dir,
                                        slots=tuple(episode["slots"]) if episode.get("slots") else None,
                                        minimized=True)
                    env = FactoredEnv(config, time_limit_minutes=episode.get("minutes", 20.0))
                    if episode.get("record_dir"):       # raw recording (fa_env/record.py)
                        from fa_env.record import Recorder
                        env.recorder = Recorder(Path(episode["record_dir"]),
                                                f"{uid or 0:07d}_{episode.get('ai')}_{episode.get('seed')}")
                obs, info = env.reset()
                rule = None
                if episode.get("mode") == "rule":
                    if episode.get("fake"):
                        rule = env
                    else:
                        rule = RulePolicy()
                        rule.reset(env)
                game = getattr(getattr(env, "fa", None), "game", None)
                info = dict(info, first=True, uid=uid, pid=game.process.pid if game else None)
                if rule is not None:
                    info["rule_action"] = rule.rule_action(obs) if episode.get("fake") else rule.act(env)
                failures = 0
                conn.send(("obs", obs, None, False, info))
            elif kind == "act":
                if env is None:
                    conn.send(("error", "act without a running game", uid))
                    continue
                if tools:
                    t_step = time.monotonic()
                    env.policy_record = msg[2] if len(msg) > 2 else None    # the network's own choice, recorded
                    if len(msg) > 3 and msg[3] is not None and episode.get("debug_dir"):
                        try:                        # the watch debugger: what this decision was made of
                            from fa_train.watch_debug import write_decision
                            write_decision(Path(episode["debug_dir"]), env, last_obs, list(msg[1]), msg[3])
                        except Exception as exc:    # noqa: BLE001 -- a debugger must never stop the game
                            print(f"watch debugger: {type(exc).__name__}: {exc}", flush=True)
                    if selfplay is not None:
                        (obs, rewards, done, info), (opponent_obs, _, _, _) = selfplay.step(
                            list(msg[1]), opponent_player.act(opponent_obs))
                    else:
                        obs, rewards, done, info = env.step_tools(list(msg[1]))
                    info["step_ms"] = (time.monotonic() - t_step) * 1000.0     # the game's side of a decision
                    info["uid"] = uid
                    if shadow:              # what M28 would have ordered in the decision just played
                        info["teacher_prev"] = info.pop("shadow_commands", [])
                        info.pop("shadow", None)
                    if teacher is not None and not done:
                        info["teacher"] = env.teacher_commands(teacher)
                        if teacher is not True:
                            env.step_labels = info["teacher"]      # recorded with the step they are for
                    if done and eval_logs:
                        info.setdefault("episode", {})["game_logs"] = close_env(keep_logs=False)
                    last_obs = obs
                    conn.send(("obs", compact(obs), rewards, done, info))
                    if done:
                        close_env(keep_logs=False)
                    continue
                obs, rewards, done, info = env.step(dict(msg[1]))
                info["uid"] = uid
                if rule is not None and not done:
                    info["rule_action"] = rule.rule_action(obs) if isinstance(rule, FakeEnv) else rule.act(env)
                if done and eval_logs:
                    info.setdefault("episode", {})["game_logs"] = close_env(keep_logs=False)
                conn.send(("obs", obs, rewards, done, info))
                if done:
                    close_env(keep_logs=False)
        except Exception as exc:
            text = f"worker {worker_id}: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
            failures += 1
            close_env(keep_logs=True)
            try:
                conn.send(("error", text, uid))
            except (BrokenPipeError, OSError):
                break
    close_env(keep_logs=False)
