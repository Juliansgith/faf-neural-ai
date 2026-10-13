"""Trainer: asynchronous game workers with batched GPU inference, for PPO, rule-policy data collection and
evaluation (docs/training-design.md, sections 8-12).

Workers step independently: whichever games have an observation ready are batched onto the GPU, so a
game that is loading (15-65 s) or slow never stalls the others. Each training worker's transitions form
segments of `segment` decisions; a segment is closed with GAE when it is full (bootstrapped from the next
value) or when the episode ends. A game that crashes or hangs loses its open segment and is relaunched.

Modes:
  ppo      the actor samples, transitions are learned from (PPO, KL pull to a cloned policy if given)
  collect  the rule policy plays; (observation, action) pairs and per-step rewards are saved for cloning
  eval     the actor plays greedily (or the rule policy, rule=True); only results are kept

Stop: at `stop_at` (a datetime), after `max_updates` / `max_episodes`, or when <run_dir>/STOP exists.
"""
from __future__ import annotations

import json
import os
import math
import multiprocessing as mp
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fa_env.factored import HEAD_NAMES, HEADS

from .model import Actor, Critic, distributions, masked_logits, parameter_count, stack, to_torch
from .worker import worker_main

# Curriculum (docs/training-design.md, section 10). Promotion is judged on the training games, which the
# sampling policy plays: a harder test than greedy evaluation, and there are many more of them. Easy stays
# in stage B's mix against forgetting. Stage C (more maps, all factions, more AIs) is step 3.
STAGES = (
    {"name": "A", "opponents": {"easy": 1.0}, "evals": {"easy": 8}, "promote": ("easy", 0.90, 100)},
    {"name": "B", "opponents": {"easy": 0.3, "medium": 0.7}, "evals": {"medium": 6, "easy": 2},
     "promote": ("medium", 0.75, 100)},
)


@dataclass
class TrainConfig:
    run_dir: str
    workers: int = 20
    fake: bool = False
    fake_delay: float = 0.005
    map: str = "SCMP_016"
    maps: bool = False                # True: every game a pool map and start (fa_env.maps), time limit by map size
    map_pool: tuple = ()              # with `maps`: only these pool maps (empty: the whole pool)
    max_minutes: float = 0.0          # with `maps`: a cap on the time limit by map size (0: none)
    faction: int = 1                  # 0: every game a random faction (evaluations rotate over the four)
    minutes: float = 20.0
    opponents: dict = field(default_factory=lambda: {"easy": 1.0})
    curriculum: bool = False          # PPO opponents and evaluations follow STAGES instead of `opponents`
    record_collect: bool = True       # save the raw teacher games of collect mode to <run>/raw
    # PPO
    gamma: float = 0.997
    gae_lambda: float = 0.95
    segment: int = 128
    batch: int = 3072
    minibatches: int = 4
    epochs: int = 3
    lr: float = 2.5e-4
    clip: float = 0.2
    vf_clip: float = 0.2
    vf_coef: float = 0.5
    ent_start: float = 0.01
    ent_end: float = 0.003
    max_grad: float = 0.5
    target_kl: float = 0.02
    shaping_start: float = 1.0
    shaping_end: float = 0.2
    shaping_fraction: float = 0.6
    bc_kl_start: float = 0.5
    bc_kl_fraction: float = 0.3
    planned_updates: int = 600
    # evaluation and bookkeeping
    eval_every: int = 25
    eval_games: int = 8
    eval_minutes: float = 30.0
    eval_ai: str = "easy"
    checkpoint_every: int = 10
    # decide for several games at once: after the first observation arrives, keep gathering for up to
    # act_gather_ms until act_min_batch are waiting. A decision costs ~15 ms alone and ~6 ms each in a batch of 14
    # on 4 CPU threads (the server, 2026-10-09); on a CPU player that is most of its decision cost. 0: off.
    act_gather_ms: float = 0.0
    act_min_batch: int = 1


class RunningStd:
    """Running variance of discounted returns, for reward scaling."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-8))


class Segment:
    def __init__(self):
        self.obs, self.actions, self.logp, self.values, self.rewards, self.dones = [], [], [], [], [], []

    def __len__(self):
        return len(self.rewards)


class WorkerState:
    def __init__(self, conn, process):
        self.conn, self.process = conn, process
        self.episode: dict | None = None
        self.segment = Segment()
        self.ret_acc = 0.0
        self.ep_return = 0.0
        self.ep_components = {"terminal": 0.0, "trade": 0.0, "economy": 0.0, "commander": 0.0}
        self.started = time.monotonic()
        self.decisions = 0
        self.last_tick: int | None = None
        self.collect_rows: list = []     # collect mode: (obs, action, rewards) of the running episode
        self.game_pid: int | None = None  # the worker's game, killed if the worker itself has to be replaced


class Trainer:
    def __init__(self, config: TrainConfig, device: str | None = None):
        self.cfg = config
        self.run_dir = Path(config.run_dir).resolve()     # absolute: game command lines are matched against it
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=config.lr, eps=1e-5)
        self.bc_actor: Actor | None = None
        self.rms = RunningStd()
        self.update = 0
        self.episodes = 0
        self.stage = 0
        self.stages = STAGES              # the curriculum (the tools trainer may use the full ladder)
        self.skipped_minibatches = 0      # non-finite loss or gradient: skipped, counted, logged
        # When set, planned_updates (the length of the schedules) is re-estimated from the measured update
        # rate so the annealing finishes by this time, whatever the throughput turns out to be.
        self.plan_deadline: datetime | None = None
        self.rng = random.Random(time.time_ns())    # new games every run; every episode's seed is logged
        self.workers: dict[int, WorkerState] = {}
        self.ctx = mp.get_context("spawn")
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir / "tb"))
        except Exception as exc:  # tensorboard missing: JSONL logs still work
            print(f"(no TensorBoard: {exc})")
        self.episode_log = open(self.run_dir / "episodes.jsonl", "a", encoding="utf-8")
        self.update_log = open(self.run_dir / "updates.jsonl", "a", encoding="utf-8")
        self.recent: dict[str, list] = {}
        self.eval_queue: list[dict] = []
        self.eval_results: list[dict] = []
        self.restarts = 0
        self.game_seconds = 0.0
        self.decisions_total = 0
        print(f"actor {parameter_count(self.actor) / 1e3:.0f} k parameters, critic "
              f"{parameter_count(self.critic) / 1e3:.0f} k, device {self.device}")

    # ------------------------------------------------------------------ workers
    def start_workers(self, n: int | None = None) -> None:
        n = n or self.cfg.workers
        log_root = str(self.run_dir / "game_logs")
        for i in range(n):
            parent, child = self.ctx.Pipe()
            p = self.ctx.Process(target=worker_main, args=(i, child, log_root), daemon=True)
            p.start()
            self.workers[i] = WorkerState(parent, p)

    def stop_workers(self) -> None:
        for w in self.workers.values():
            try:
                w.conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        deadline = time.monotonic() + 30
        for w in self.workers.values():
            w.process.join(timeout=max(0.1, deadline - time.monotonic()))
            if w.process.is_alive():
                w.process.kill()
        self.workers.clear()
        self.kill_games()

    def kill_games(self) -> None:
        """Games are started by the workers; make sure none of this run's outlive it. Only games whose
        command line points into this run's folder (their /log path) -- never any other main.exe."""
        if self.cfg.fake:
            return
        if os.name != "nt":                  # under Wine: the game's command line holds its Z: drive log path
            from fa_env.bridge import game_path
            marker = game_path(self.run_dir / "game_logs").lower()
            for proc in Path("/proc").iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").lower()
                    if "main.exe" in cmd and marker in cmd:
                        os.kill(int(proc.name), 9)
                except (OSError, ValueError):
                    pass
            return
        marker = str(self.run_dir / "game_logs").lower().replace("'", "''")
        script = ("Get-CimInstance Win32_Process -Filter \"Name='main.exe'\" | Where-Object { $_.CommandLine -and "
                  f"$_.CommandLine.ToLower().Contains('{marker}') }} | "
                  "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass

    def opponents(self) -> dict:
        return self.stages[self.stage]["opponents"] if self.cfg.curriculum else self.cfg.opponents

    def _episode(self, mode: str) -> dict:
        cfg = self.cfg
        if self.eval_queue:
            return self.eval_queue.pop(0)
        names, weights = zip(*self.opponents().items())
        map_id, slots, minutes = cfg.map, None, cfg.minutes
        if cfg.maps:
            from fa_env.maps import sample_map
            map_id, slots, minutes = sample_map(self.rng, self._map_pool())
            minutes = min(minutes, cfg.max_minutes) if cfg.max_minutes else minutes
        episode = {"map": map_id, "slots": slots, "faction": cfg.faction or self.rng.randint(1, 4),
                   "ai": self.rng.choices(names, weights)[0],
                   "ai_faction": self.rng.randint(1, 4), "seed": self.rng.randint(1, 10 ** 6), "minutes": minutes,
                   "mode": mode, "eval": False, "fake": cfg.fake, "fake_delay": cfg.fake_delay, "stage": self.stage}
        if self.mode == "collect" and cfg.record_collect and not cfg.fake:
            episode["record_dir"] = str(self.run_dir / "raw")
        return episode

    def schedule_periodic_eval(self) -> None:
        plan = self.stages[self.stage]["evals"] if self.cfg.curriculum else {self.cfg.eval_ai: 8}
        for ai, games in plan.items():             # eval_games scales the plan (8 = as written)
            self.schedule_eval(max(1, round(games * self.cfg.eval_games / 8)), ai)

    def _check_promotion(self) -> None:
        if not self.cfg.curriculum or self.stage + 1 >= len(self.stages):
            return
        # one condition (ai, rate, window) or several, all needed; `ai` may be a tuple of opponents counted
        # together (e.g. the four AIx personalities at one multiplier)
        rule = self.stages[self.stage]["promote"]
        rates = []
        for ai, need, window in ([rule] if isinstance(rule[0], str) else rule):
            pool = (ai,) if isinstance(ai, str) else tuple(ai)
            games = sorted((r for a in pool for r in self.recent.get(f"train_{a}", [])
                            if r.get("stage") == self.stage and r.get("mode") == "ppo"),
                           key=lambda r: r.get("time") or "")[-window:]
            if len(games) < window:
                return
            rate = float(np.mean([1.0 if r.get("outcome") == "victory" else 0.0 for r in games]))
            if rate < need:
                return
            rates.append(rate)
        first = rule if isinstance(rule[0], str) else rule[0]
        ai, window, rate = first[0], first[2], min(rates)
        self.stage += 1
        name = self.stages[self.stage]["name"]
        self._log_event({"event": "promotion", "stage": name, "update": self.update, "episodes": self.episodes,
                         "win_rate": rate, "against": ai, "window": window})
        self.save(f"stage{name}_start.pt")
        print(f"*** promoted to stage {name} at update {self.update}: {rate:.0%} of the last {window} "
              f"training games against {ai} won; opponents now {self.stages[self.stage]['opponents']}")

    def _map_pool(self) -> dict:
        from fa_env.maps import MAP_POOL
        unknown = [m for m in self.cfg.map_pool if m not in MAP_POOL]
        if unknown:
            raise ValueError(f"not in the map pool: {unknown}")
        return {k: v for k, v in MAP_POOL.items() if not self.cfg.map_pool or k in self.cfg.map_pool}

    def schedule_eval(self, games: int, ai: str, rule: bool = False, minutes: float | None = None) -> None:
        from fa_env.maps import MAP_POOL, MINUTES_BY_KM
        pool = sorted(self._map_pool())
        for i in range(games):
            map_id, slots, limit = self.cfg.map, None, self.cfg.eval_minutes
            if self.cfg.maps:                 # evaluations walk through the pool: map i, its slot pairs in turn
                map_id = pool[i % len(pool)]
                entry = MAP_POOL[map_id]
                slots = tuple(entry["slots"][(i // len(pool)) % len(entry["slots"])])
                limit = MINUTES_BY_KM[entry["km"]]
                limit = min(limit, self.cfg.max_minutes) if self.cfg.max_minutes else limit
            self.eval_queue.append({"map": map_id, "slots": slots, "faction": self.cfg.faction or 1 + i % 4, "ai": ai,
                                    "ai_faction": 1 + i % 4, "seed": 900000 + i,
                                    "minutes": minutes or limit, "mode": "rule" if rule else "policy",
                                    "eval": True, "fake": self.cfg.fake, "fake_delay": self.cfg.fake_delay,
                                    "tag": f"eval@{self.update}"})

    def _send_reset(self, wid: int, mode: str) -> None:
        w = self.workers[wid]
        w.episode = dict(self._episode(mode))
        self.next_uid = getattr(self, "next_uid", 0) + 1
        w.episode["uid"] = self.next_uid
        w.segment = Segment()
        w.ret_acc, w.ep_return, w.decisions = 0.0, 0.0, 0
        w.last_tick = None
        w.ep_components = {"terminal": 0.0, "trade": 0.0, "economy": 0.0, "commander": 0.0}
        w.collect_rows = []
        w.started = time.monotonic()
        w.conn.send(("reset", w.episode))

    # ------------------------------------------------------------------ schedules
    def _frac(self, fraction: float) -> float:
        return min(1.0, self.update / max(1.0, fraction * self.cfg.planned_updates))

    def shaping(self) -> float:
        c = self.cfg
        return c.shaping_start + (c.shaping_end - c.shaping_start) * self._frac(c.shaping_fraction)

    def ent_coef(self) -> float:
        c = self.cfg
        return c.ent_start + (c.ent_end - c.ent_start) * self._frac(1.0)

    def learning_rate(self) -> float:
        """Linear decay to 10 % of the start by the planned end (a resumed run warms back up a little)."""
        return self.cfg.lr * (1.0 - 0.9 * self._frac(1.0))

    def bc_beta(self) -> float:
        if self.bc_actor is None:
            return 0.0
        return self.cfg.bc_kl_start * (1.0 - self._frac(self.cfg.bc_kl_fraction))

    def reward(self, components: dict) -> float:
        return components["terminal"] + self.shaping() * (components["trade"] + components["economy"]
                                                          + components.get("commander", 0.0))

    # ------------------------------------------------------------------ main loop
    def run(self, mode: str, stop_at: datetime | None = None, max_updates: int | None = None,
            max_episodes: int | None = None, collect_dir: Path | None = None, shard_rows: int = 20000) -> dict:
        """Drive all workers in `mode` until a stop condition; returns a summary."""
        assert mode in ("ppo", "collect", "eval")
        self.mode = mode
        self.collect_dir = collect_dir
        self.shard_rows = shard_rows
        self.collect_buffer: list = []
        # continue the numbering of shards already there (a retried collection must not overwrite them)
        self.collect_shards = len(list(collect_dir.glob("shard_*.npz"))) if collect_dir and collect_dir.exists() else 0
        self.ready: list[Segment] = []
        start_update, start_episodes = self.update, self.episodes
        episode_mode = "rule" if mode == "collect" else "policy"
        if not self.workers:
            self.start_workers()
        for wid in self.workers:
            if mode == "eval" and not self.eval_queue:
                self.workers[wid].episode = None      # more workers than evaluation games: stay idle
                continue
            self._send_reset(wid, episode_mode)
        t0, last_log = time.monotonic(), time.monotonic()
        self.throughput_t0, self.throughput_decisions, self.throughput_game_s = t0, 0, 0.0
        plan_mark = (t0, self.update)
        stop_file = self.run_dir / "STOP"
        stop_reason = None
        while True:
            if stop_at and datetime.now() >= stop_at:
                stop_reason = "stop time"
            elif stop_file.exists():
                stop_reason = "STOP file"
            elif max_updates is not None and self.update - start_update >= max_updates:
                stop_reason = "updates"
            elif max_episodes is not None and self.episodes - start_episodes >= max_episodes:
                stop_reason = "episodes"
            elif mode == "eval" and not self.eval_queue and all(w.episode is None for w in self.workers.values()):
                stop_reason = "eval done"
            if stop_reason:
                break
            conns = {w.conn: wid for wid, w in self.workers.items()}
            pending = []
            gather_until = None
            t_wait = time.monotonic()
            ready = wait(list(conns), timeout=0.05)
            self.idle_s = getattr(self, "idle_s", 0.0) + time.monotonic() - t_wait
            while ready:
                for conn in ready:
                    wid = conns.pop(conn)
                    try:
                        msg = conn.recv()
                    except (EOFError, OSError):
                        self._restart_worker(wid, episode_mode, "pipe closed")
                        continue
                    item = self._handle(wid, msg, episode_mode)
                    if item is not None:
                        self.workers[wid].t_recv = time.monotonic()
                        pending.append(item)
                if not pending or len(pending) >= self.cfg.act_min_batch or self.cfg.act_gather_ms <= 0:
                    break
                if gather_until is None:
                    gather_until = time.monotonic() + self.cfg.act_gather_ms / 1000.0
                left = gather_until - time.monotonic()
                if left <= 0 or not conns:
                    break
                t_wait = time.monotonic()
                ready = wait(list(conns), timeout=left)
                self.idle_s = getattr(self, "idle_s", 0.0) + time.monotonic() - t_wait
            if pending:
                self._act(pending)
            if mode == "ppo" and self._maybe_update():
                if self.update % self.cfg.checkpoint_every == 0:
                    self.save("latest.pt")
                if self.update % 50 == 0:
                    self.save(f"ckpt_{self.update:05d}.pt")
                if self.cfg.eval_every and self.update % self.cfg.eval_every == 0:
                    self.schedule_periodic_eval()
                done_here = self.update - plan_mark[1]
                if self.plan_deadline and done_here and done_here % 10 == 0:
                    rate = done_here / (time.monotonic() - plan_mark[0])
                    remaining = max(0.0, (self.plan_deadline - datetime.now()).total_seconds())
                    self.cfg.planned_updates = max(self.update + 1, int(self.update + rate * remaining))
            if time.monotonic() - last_log > 60:
                last_log = time.monotonic()
                self._log_throughput()
        if mode == "collect":
            self._flush_shard()
        if mode == "ppo":
            self.save("latest.pt")
        # The games of this phase are not needed any more; the next phase starts its own.
        for w in self.workers.values():
            if w.episode is not None:
                try:
                    w.conn.send(("close",))
                except (BrokenPipeError, OSError):
                    pass
                w.episode = None
        summary = {"mode": mode, "stop": stop_reason, "updates": self.update - start_update,
                   "episodes": self.episodes - start_episodes, "wall_minutes": round((time.monotonic() - t0) / 60, 1),
                   "restarts": self.restarts}
        print("run finished:", summary)
        return summary

    def _restart_worker(self, wid: int, episode_mode: str, reason: str) -> None:
        """Replace a worker process that died."""
        self.restarts += 1
        old = self.workers[wid]
        try:
            old.process.kill()
        except Exception:
            pass
        if old.game_pid:                     # its game would otherwise run on, orphaned, next to the new one
            subprocess.run(["taskkill", "/F", "/PID", str(old.game_pid)], capture_output=True)
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=worker_main, args=(wid, child, str(self.run_dir / "game_logs")), daemon=True)
        p.start()
        self.workers[wid] = WorkerState(parent, p)
        self._log_event({"event": "worker_restart", "worker": wid, "reason": reason})
        if self.mode != "eval" or self.eval_queue:
            self._send_reset(wid, episode_mode)

    def _handle(self, wid: int, msg, episode_mode: str):
        w = self.workers[wid]
        # Messages of an episode the learner no longer runs (e.g. from before a phase change) are stale.
        uid = msg[2] if msg[0] == "error" and len(msg) > 2 else (msg[4].get("uid") if msg[0] == "obs" else None)
        if w.episode is None or uid != w.episode.get("uid"):
            return None
        if msg[0] == "error":
            self.restarts += 1
            self._log_event({"event": "game_error", "worker": wid, "episode": w.episode, "error": msg[1][-1500:]})
            print(f"worker {wid}: game error, relaunching ({msg[1].splitlines()[0][:160]})")
            w.segment = Segment()          # the open segment has no proper end: drop it
            w.collect_rows = []
            if w.episode and w.episode.get("eval"):
                self.eval_queue.insert(0, w.episode)
            w.episode = None
            if self.mode != "eval" or self.eval_queue:
                self._send_reset(wid, episode_mode)
            return None
        _, obs, rewards, done, info = msg
        ep = w.episode or {}
        if info.get("first"):
            w.game_pid = info.get("pid")
        if rewards is not None:
            r = self.reward(rewards)
            w.ep_return += r
            for k in w.ep_components:
                w.ep_components[k] += rewards.get(k, 0.0)
            w.decisions += 1
            self.decisions_total += 1
            self.throughput_decisions += 1
            tick = info.get("tick")
            previous_tick = w.last_tick
            elapsed_ticks = (max(0, tick - previous_tick)
                             if tick is not None and previous_tick is not None
                             else ep.get("decision_ticks", 20))
            self.throughput_game_s += elapsed_ticks / 10.0
            if self.mode == "ppo" and not ep.get("eval"):
                w.ret_acc = w.ret_acc * self.cfg.gamma + r
                self.rms.update(w.ret_acc)
                w.segment.rewards.append(r / self.rms.std)
                w.segment.dones.append(float(done))
                if "teacher_prev" in info and getattr(w.segment, "teacher", None):
                    # a teacher that answers after the step (M28 in shadow mode): labels of the decision just played
                    w.segment.teacher[len(w.segment.rewards) - 1] = info["teacher_prev"]
            elif self.mode == "collect" and w.collect_rows:
                w.collect_rows[-1][2] = rewards
        w.last_tick = info.get("tick")
        if done:
            self._episode_end(wid, info)
            if self.mode == "ppo" and not ep.get("eval") and len(w.segment):
                self._close_segment(w.segment, bootstrap=0.0)
            if self.mode == "collect":
                self._store_episode(w)
            w.episode = None
            if self.mode != "eval" or self.eval_queue:
                self._send_reset(wid, episode_mode)
            return None
        return wid, obs, info

    def _act(self, pending: list) -> None:
        learn, greedy, rule = [], [], []
        for wid, obs, info in pending:
            ep = self.workers[wid].episode or {}
            if ep.get("mode") == "rule":
                rule.append((wid, obs, info))
            elif ep.get("eval"):
                greedy.append((wid, obs, info))
            else:
                learn.append((wid, obs, info))
        for wid, obs, info in rule:
            choice = info.get("rule_action") or {h: 0 for h in HEAD_NAMES}
            w = self.workers[wid]
            if self.mode == "collect":
                w.collect_rows.append([obs, np.array([choice[h] for h in HEAD_NAMES], np.int64), None])
            w.conn.send(("act", choice))
        for group, sample in ((learn, True), (greedy, False)):
            if not group:
                continue
            batch = to_torch(stack([o for _, o, _ in group]), self.device)
            with torch.no_grad():
                dists = distributions(self.actor(batch), batch)
                if sample:
                    actions = {h: d.sample() for h, d in dists.items()}
                    logp = sum(dists[h].log_prob(actions[h]) for h in HEAD_NAMES)
                    values = self.critic(batch)
                else:
                    actions = {h: d.probs.argmax(-1) for h, d in dists.items()}
            for i, (wid, obs, _) in enumerate(group):
                choice = {h: int(actions[h][i]) for h in HEAD_NAMES}
                w = self.workers[wid]
                if sample:
                    seg = w.segment
                    if len(seg.actions) >= self.cfg.segment and len(seg) == len(seg.actions):
                        self._close_segment(seg, bootstrap=float(values[i]))
                        w.segment = seg = Segment()
                    seg.obs.append(obs)
                    seg.actions.append(np.array([choice[h] for h in HEAD_NAMES], np.int64))
                    seg.logp.append(float(logp[i]))
                    seg.values.append(float(values[i]))
                w.conn.send(("act", choice))

    # ------------------------------------------------------------------ PPO
    def _close_segment(self, seg: Segment, bootstrap: float) -> None:
        n = len(seg)
        if n == 0:
            return
        seg.obs, seg.actions, seg.logp, seg.values = seg.obs[:n], seg.actions[:n], seg.logp[:n], seg.values[:n]
        values = np.asarray(seg.values, np.float32)
        rewards = np.asarray(seg.rewards, np.float32)
        dones = np.asarray(seg.dones, np.float32)
        adv = np.zeros(n, np.float32)
        last = 0.0
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        for t in reversed(range(n)):
            next_value = bootstrap if t == n - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + g * next_value * nonterminal - values[t]
            last = delta + g * lam * nonterminal * last
            adv[t] = last
        seg.advantages, seg.returns = adv, adv + values
        self.ready.append(seg)

    def _maybe_update(self) -> bool:
        """Run a PPO update when a batch is ready; True when an update has completed (a subclass may run
        updates in the background and report them when they finish)."""
        if sum(len(s) for s in self.ready) < self.cfg.batch:
            return False
        self._ppo_update()
        return True

    def _ppo_update(self) -> None:
        t0 = time.monotonic()
        segs, self.ready = self.ready, []
        obs = to_torch(stack([o for s in segs for o in s.obs]), self.device)
        actions = torch.as_tensor(np.stack([a for s in segs for a in s.actions]), device=self.device)
        old_logp = torch.as_tensor(np.array([x for s in segs for x in s.logp], np.float32), device=self.device)
        old_values = torch.as_tensor(np.array([x for s in segs for x in s.values], np.float32), device=self.device)
        adv = torch.as_tensor(np.concatenate([s.advantages for s in segs]), device=self.device)
        returns = torch.as_tensor(np.concatenate([s.returns for s in segs]), device=self.device)
        n = adv.shape[0]
        cfg = self.cfg
        ent_coef, beta, lr = self.ent_coef(), self.bc_beta(), self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        stats = {k: [] for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "bc_kl")}
        head_entropy = {h: [] for h in HEAD_NAMES}
        stop = False
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for mb in perm.chunk(cfg.minibatches):
                mb_obs = {k: v[mb] for k, v in obs.items()}
                logits = self.actor(mb_obs)
                dists = distributions(logits, mb_obs)
                logp = sum(dists[h].log_prob(actions[mb, i]) for i, h in enumerate(HEAD_NAMES))
                ent = {h: dists[h].entropy() for h in HEAD_NAMES}
                entropy = sum(ent.values())
                ratio = (logp - old_logp[mb]).exp()
                a = adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v = self.critic(mb_obs)
                v_clip = old_values[mb] + (v - old_values[mb]).clamp(-cfg.vf_clip, cfg.vf_clip)
                v_loss = 0.5 * torch.max((v - returns[mb]) ** 2, (v_clip - returns[mb]) ** 2).mean()
                loss = pg + cfg.vf_coef * v_loss - ent_coef * entropy.mean()
                bc_kl = torch.zeros((), device=self.device)
                if beta > 0 and self.bc_actor is not None:
                    with torch.no_grad():
                        teacher = masked_logits(self.bc_actor(mb_obs), mb_obs)
                    student = masked_logits(logits, mb_obs)
                    for h in HEAD_NAMES:
                        p = F.softmax(teacher[h], -1)
                        bc_kl = bc_kl + (p * (F.log_softmax(teacher[h], -1) - F.log_softmax(student[h], -1))).sum(-1).mean()
                    loss = loss + beta * bc_kl
                self.optimizer.zero_grad()
                if not torch.isfinite(loss):
                    self.skipped_minibatches += 1          # one bad minibatch must not poison the weights
                    continue
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                                      cfg.max_grad)
                if not torch.isfinite(norm):
                    self.optimizer.zero_grad()
                    self.skipped_minibatches += 1
                    continue
                self.optimizer.step()
                with torch.no_grad():
                    log_ratio = logp - old_logp[mb]
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    stats["pg_loss"].append(pg.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append(approx_kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    stats["bc_kl"].append(bc_kl.item())
                    for h in HEAD_NAMES:
                        head_entropy[h].append(ent[h].mean().item())
                if approx_kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.update += 1
        with torch.no_grad():
            var = returns.var()
            explained = (1 - (returns - old_values).var() / var).item() if var > 0 else 0.0
        mean = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731
        record = {k: mean(v) for k, v in stats.items()}
        record.update(update=self.update, samples=n, epochs_run=epoch + 1, explained_variance=explained,
                      ent_coef=ent_coef, bc_beta=beta, lr=lr, shaping=self.shaping(), return_std=self.rms.std,
                      stage=self.stages[self.stage]["name"] if self.cfg.curriculum else None,
                      planned_updates=cfg.planned_updates, skipped_minibatches=self.skipped_minibatches,
                      seconds=time.monotonic() - t0, time=datetime.now().isoformat(timespec="seconds"),
                      **{f"entropy_{h}": mean(v) for h, v in head_entropy.items()})
        self.update_log.write(json.dumps(record) + "\n")
        self.update_log.flush()
        if self.writer:
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"ppo/{k}", v, self.update)
        print(f"update {self.update}: {n} samples, pg {record['pg_loss']:+.3f} v {record['v_loss']:.3f} "
              f"ent {record['entropy']:.2f} kl {record['approx_kl']:.4f} bc_kl {record['bc_kl']:.3f} "
              f"ev {explained:+.2f} ({record['seconds']:.1f} s)")

    # ------------------------------------------------------------------ episodes and logging
    def _episode_end(self, wid: int, info: dict) -> None:
        w = self.workers[wid]
        ep = w.episode or {}
        summary = info.get("episode") or {}
        self.episodes += 1
        record = {"time": datetime.now().isoformat(timespec="seconds"), "worker": wid, "update": self.update,
                  "mode": self.mode, "eval": ep.get("eval", False), "rule": ep.get("mode") == "rule",
                  "stage": ep.get("stage"), "tag": ep.get("tag"),
                  "ai": ep.get("ai"), "ai_faction": ep.get("ai_faction"), "faction": ep.get("faction"),
                  "map": ep.get("map"), "slots": ep.get("slots"),
                  "seed": ep.get("seed"),
                  "return": round(w.ep_return, 3), "components": {k: round(v, 3) for k, v in w.ep_components.items()},
                  "decisions": w.decisions, "wall_seconds": round(time.monotonic() - w.started, 1), **summary}
        self.episode_log.write(json.dumps(record) + "\n")
        self.episode_log.flush()
        # gate: the rule policy evaluated; collect: its games for cloning; eval: the actor, greedy;
        # train: the actor's own PPO games (the only ones promotion counts)
        kind = {(True, True): "gate", (True, False): "eval", (False, True): "collect", (False, False): "train"}[
            (bool(ep.get("eval")), ep.get("mode") == "rule")]
        key = f"{kind}_{ep.get('ai')}"
        self.recent.setdefault(key, []).append(record)
        self.recent[key] = self.recent[key][-200:]
        if ep.get("eval"):
            self.eval_results.append(record)
        if self.writer:
            wins = [1.0 if r.get("outcome") == "victory" else 0.0 for r in self.recent[key][-50:]]
            self.writer.add_scalar(f"win_rate/{key}", float(np.mean(wins)), self.update)
            self.writer.add_scalar(f"return/{key}", record["return"], self.update)
            self.writer.add_scalar(f"value_advantage/{key}", summary.get("value_advantage", 0.0), self.update)
            self.writer.add_scalar(f"minutes/{key}", summary.get("minutes", 0.0), self.update)
            if self.cfg.curriculum:
                self.writer.add_scalar("curriculum/stage", self.stage, self.update)
        print(f"episode {self.episodes} [{key}] {summary.get('outcome')} at {summary.get('minutes')} min, "
              f"return {w.ep_return:+.2f}, adv {summary.get('value_advantage')}, worker {wid}")
        if self.mode == "ppo" and not ep.get("eval"):
            self._check_promotion()

    def _log_event(self, event: dict) -> None:
        event["time"] = datetime.now().isoformat(timespec="seconds")
        with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _log_throughput(self) -> None:
        wall = time.monotonic() - self.throughput_t0
        dps, gps = self.throughput_decisions / wall, self.throughput_game_s / wall
        print(f"[{datetime.now():%H:%M:%S}] {dps:.1f} decisions/s, {gps:.0f} game-s/s, update {self.update}, "
              f"episodes {self.episodes}, restarts {self.restarts}")
        if self.writer:
            self.writer.add_scalar("throughput/decisions_per_s", dps, self.update)
            self.writer.add_scalar("throughput/game_seconds_per_s", gps, self.update)
            self.writer.add_scalar("throughput/restarts", self.restarts, self.update)
        self.throughput_t0, self.throughput_decisions, self.throughput_game_s = time.monotonic(), 0, 0.0

    # ------------------------------------------------------------------ collection (behaviour cloning)
    def _store_episode(self, w: WorkerState) -> None:
        rows = [r for r in w.collect_rows if r[2] is not None]
        if not rows:
            return
        g, c = self.cfg.gamma, self.cfg.shaping_start
        rewards = np.array([r[2]["terminal"] + c * (r[2]["trade"] + r[2]["economy"] + r[2].get("commander", 0.0))
                            for r in rows], np.float32)
        returns = np.zeros_like(rewards)
        acc = 0.0
        for t in reversed(range(len(rewards))):
            acc = rewards[t] + g * acc
            returns[t] = acc
        uid = (w.episode or {}).get("uid", 0)
        for (obs, action, _), ret in zip(rows, returns):
            self.collect_buffer.append((obs, action, float(ret), uid))
        if len(self.collect_buffer) >= self.shard_rows:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self.collect_buffer or self.collect_dir is None:
            return
        self.collect_dir.mkdir(parents=True, exist_ok=True)
        obs = stack([row[0] for row in self.collect_buffer])
        path = self.collect_dir / f"shard_{self.collect_shards:03d}.npz"
        # episode ids let cloning hold out whole games; the run-time prefix keeps them unique across runs
        prefix = int(self.throughput_t0 * 1000) % 10 ** 9 * 10 ** 6
        np.savez_compressed(path, actions=np.stack([row[1] for row in self.collect_buffer]),
                            returns=np.array([row[2] for row in self.collect_buffer], np.float32),
                            episodes=np.array([prefix + row[3] for row in self.collect_buffer], np.int64), **obs)
        print(f"saved {len(self.collect_buffer)} rows to {path}")
        self.collect_shards += 1
        self.collect_buffer = []

    # ------------------------------------------------------------------ checkpoints
    def save(self, name: str) -> Path:
        path = self.run_dir / "checkpoints" / name
        state = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "update": self.update, "episodes": self.episodes,
                 "rms": (self.rms.mean, self.rms.var, self.rms.count), "config": asdict(self.cfg),
                 "stage": self.stage, "heads": HEADS}
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)              # a kill mid-write must not leave a broken latest.pt
        tmp.replace(path)
        return path

    def load(self, path: str | Path, optimizer: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state["actor"])
        if state.get("critic"):
            self.critic.load_state_dict(state["critic"])
        if optimizer and state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
        self.update = state.get("update", 0)
        self.episodes = max(self.episodes, state.get("episodes", 0))   # a clone (0 episodes) keeps the count
        self.stage = min(state.get("stage", 0), len(self.stages) - 1)
        if state.get("rms"):
            self.rms.mean, self.rms.var, self.rms.count = state["rms"]

    def set_teacher(self, path: str | Path) -> None:
        """Freeze a (cloned) actor as the KL target of the first part of PPO."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.bc_actor = Actor().to(self.device)
        self.bc_actor.load_state_dict(state["actor"])
        self.bc_actor.eval()
        for p in self.bc_actor.parameters():
            p.requires_grad_(False)

    def close(self) -> None:
        self.stop_workers()
        self.episode_log.close()
        self.update_log.close()
        if self.writer:
            self.writer.close()
