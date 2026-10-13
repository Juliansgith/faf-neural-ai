"""PPO for the tools network (docs/tools-design.md, section 5).

What differs from the menu trainer (fa_train/trainer.py, whose workers, episodes, curriculum, evaluations and
logs it reuses):
- recurrent: each game carries the LSTM memory; training sequences are segments of `segment` decisions that
  store the memory they started with;
- commands: the network's choices are turned into commands with unit ids for the worker;
- online teacher distillation: training games ask the rule teacher, every decision, what it would order;
  an imitation loss on those labels starts at `distill_start` and fades over `distill_fraction` to
  `distill_end`, which stays: an anchor that keeps the policy from drifting into noise (night 1);
- training games act at `act_temperature` and PPO scores them at the same temperature: an order stays in force
  until it is replaced, so sampling at temperature 1 re-orders units every few decisions and nothing gets done;
- entropy brake: the entropy bonus is switched off while the entropy is above `entropy_ceiling` times its
  value at the first update;
- belief: an auxiliary loss makes the memory predict the enemy's hidden totals;
- updates run in a background thread on their own CUDA stream with a separate copy of the weights for
  acting, so twenty games keep playing while the network trains. Minibatches go through the network in
  micro-batches of `micro_seqs` segments (gradients accumulated), halved automatically on an out-of-memory
  error, and the process may use only `gpu_memory_fraction` of the GPU: on Windows a full GPU pages to system
  memory and an update then takes ten minutes instead of seconds (night 1).
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .tools_model import (action_row, actions_to_commands, apply_model_options, empty_unit_labels, model_options,
                          stack_actions,
                          ACTION_KEYS, K, ToolsNet, collate, commands_to_indices, empty_labels, indices_to_commands,
                          parameter_count)
from fa_env.tools import SET_SUBJECT, VERBS
from fa_env.observation_precision import widen_observation
from .trainer import STAGES, TrainConfig, Trainer, WorkerState
from .tools_distillation import checkpoint_distillation, distillation_metadata, validate_distillation
from .checkpoint_contract import runtime_set_distribution, validate_checkpoint_set_distribution
from .tools_cadence import LEGACY_DECISION_TICKS, checkpoint_cadence, decision_ticks

# The opponent ladder of the full-toolbox design (docs/full-toolbox-design.md, section 6): FAF's easy, normal,
# its four personalities, then the cheating AIx at rising multipliers ('<ai>@<multiplier>'). Every rung keeps
# about 20 % of its games against the rungs below, against forgetting. M28 (rung L4) joins only with the user's go.
PERSONALITIES = ("adaptive", "rush", "turtle", "tech")


def _aix_rung(name: str, mult: str, lower: dict) -> dict:
    aix = tuple(f"{p}cheat@{mult}" for p in PERSONALITIES)
    return {"name": name, "opponents": {**{a: 0.2 for a in aix}, **lower},
            "evals": {a: 3 for a in aix}, "promote": [(aix, 0.75, 100)]}


_LOWER = (
    {"name": "L0", "opponents": {"easy": 1.0}, "evals": {"easy": 8}, "promote": ("easy", 0.90, 100)},
    {"name": "L1", "opponents": {"medium": 0.8, "easy": 0.2}, "evals": {"medium": 8, "easy": 2},
     "promote": ("medium", 0.85, 100)},
    {"name": "L2", "opponents": {**{p: 0.2 for p in PERSONALITIES}, "medium": 0.1, "easy": 0.1},
     "evals": {p: 4 for p in PERSONALITIES}, "promote": [(p, 0.80, 40) for p in PERSONALITIES]},
)
LADDER = (
    *_LOWER,
    _aix_rung("L3a", "1.0", {p: 0.05 for p in PERSONALITIES}),
    _aix_rung("L3b", "1.2", {f"{p}cheat@1.0": 0.05 for p in PERSONALITIES}),
    _aix_rung("L3c", "1.5", {f"{p}cheat@1.2": 0.05 for p in PERSONALITIES}),
)


def _m28_rung(name: str, key: str, lower: dict) -> dict:
    """A rung against M28 ('m28ai', or its cheating variant 'm28aicheat@<multiplier>', M28AIData.lua);
    `lower` keeps a share of games against what came before."""
    return {"name": name, "opponents": {key: 0.6, **lower}, "evals": {key: 8}, "promote": (key, 0.75, 100)}


# The ladder toward the goal (beat M28, then M28 with cheat multipliers): the lower rungs as above, then M28
# itself, then AIx M28 at rising resource/build multipliers.
LADDER_M28 = (
    *_LOWER,
    _m28_rung("M28", "m28ai", {**{p: 0.075 for p in PERSONALITIES}, "medium": 0.1}),
    _m28_rung("M28x1.1", "m28aicheat@1.1", {"m28ai": 0.3, "adaptive": 0.1}),
    _m28_rung("M28x1.25", "m28aicheat@1.25", {"m28aicheat@1.1": 0.3, "m28ai": 0.1}),
    _m28_rung("M28x1.5", "m28aicheat@1.5", {"m28aicheat@1.25": 0.3, "m28ai": 0.1}),
    _m28_rung("M28x2", "m28aicheat@2.0", {"m28aicheat@1.5": 0.3, "m28aicheat@1.25": 0.1}),
)
LADDERS = {"aix": LADDER, "m28": LADDER_M28}


def stage_index(stages, name: str) -> int:
    """The index of the rung called `name` (or a numeric index as a string)."""
    for i, stage in enumerate(stages):
        if stage["name"] == name:
            return i
    if name.isdigit() and int(name) < len(stages):
        return int(name)
    raise ValueError(f"no rung {name!r}; the ladder has {[s['name'] for s in stages]}")


@dataclass
class ToolsConfig(TrainConfig):
    size: str = "M"
    decision_ticks: int = LEGACY_DECISION_TICKS  # set 2 explicitly for the weekend demonstration cadence
    segment: int = 32
    minibatch_seqs: int = 16
    distill_teacher: str = "v3"
    distill_start: float = 1.0
    distill_fraction: float = 0.3
    distill_end: float = 0.1
    act_temperature: float = 0.5
    decoder: str = "slots"              # 'units': every unit decides for itself (ToolsNet)
    # The first `dagger_updates` updates only imitate the online teacher on the policy's own games (DAgger): no
    # policy gradient, no KL stop, teacher weight 1, learning rate `dagger_lr`. PPO's small steps and KL stop
    # made the same lesson crawl: 25 updates in, the policy still gave a quarter of the teacher's orders
    # (2026-10-05). The value head learns throughout, so PPO starts with a critic that knows the policy.
    dagger_updates: int = 0
    dagger_lr: float = 2e-4
    # how evaluation games are played: 'greedy', or 'sample' at eval_temperature (with eval_top_p: nucleus)
    eval_mode: str = "greedy"
    eval_temperature: float = 1.0
    eval_top_p: float | None = None
    entropy_ceiling: float = 2.0
    micro_seqs: int = 4
    gpu_memory_fraction: float = 0.7
    warmup_updates: int = 10
    # bf16 encoder (acting and training alike) plus TF32 matmuls: training micro-batches 41 % faster, but acting and
    # scoring then disagree by a median 0.019 in log-probability (max 0.13; float32: 0.0006) -- noise in every PPO
    # ratio. Off while updates are not the bottleneck; cloning, which has no ratios, runs in bf16 regardless.
    amp_encoder: bool = False
    act_priority: bool = True           # choosing moves on a high-priority CUDA stream, ahead of updates
    # a small entropy bonus: sampling at act_temperature explores already, and in the pipeline test (and on
    # night 1) the bonus pushed entropy up whenever the teacher's weight fell
    ent_start: float = 0.003
    ent_end: float = 0.001
    lr: float = 3e-5                    # a sharp policy: 1e-4 overshot the KL target within six minibatches
    belief_coef: float = 0.2
    ent_weights: dict = field(default_factory=lambda: {"verb": 1.0, "subject": 1.0, "blueprint": 0.5,
                                                       "enhancement": 0.25, "option": 0.25, "count": 0.5,
                                                       "place": 0.5, "cell": 1.0, "target": 0.5})
    fake_raw: str | None = None       # recordings the fake tools episodes replay
    ladder: bool = False              # opponents follow a ladder (LADDERS[ladder_name]) instead of the two-stage STAGES
    ladder_name: str = "aix"          # 'aix': FAF's cheating personalities; 'm28': M28, then M28 with multipliers
    # Factored PPO for the units decoder: every unit's choice is its own action, with its decision's advantage.
    # A whole decision's ratio is a product over hundreds of units (one per unit that can act), so the clip
    # triggers on nearly every decision and the KL stop after a few minibatches, and nothing is learned
    # from the rest. False: the joint ratio (what the slots decoder always uses).
    unit_ratio: bool = True
    # KL to a frozen reference policy (the DAgger network), estimated per sampled action (per unit with
    # unit_ratio) as exp(d) - d - 1 with d = log ref - log policy (unbiased for KL(policy || ref)), weighted by
    # ref_coef: the policy may move where the reward says so, and is pulled back everywhere else. The online
    # teacher only sees the states the policy visits; this anchor holds the whole distribution.
    reference: str | None = None
    ref_coef: float = 0.0
    grad_checkpoint: bool = False     # recompute the encoder and decoder in the backward pass (half the memory)

    def __post_init__(self):
        validate_distillation(self)
        decision_ticks(self.decision_ticks)
        if self.ladder_name not in LADDERS:
            raise ValueError(f"unknown ladder {self.ladder_name!r}; choose one of {sorted(LADDERS)}")


class ToolSegment:
    def __init__(self, state, first_is_start: bool):
        self.state = state                    # (h, c) [1,1,core] on the CPU: the memory at the first decision
        self.first_is_start = first_is_start
        self.obs, self.actions, self.logp, self.values, self.rewards, self.dones, self.teacher = [], [], [], [], [], [], []
        self.unit_logp = []                   # units decoder: each decision's per-unit log-probabilities [n_tokens]

    def __len__(self):
        return len(self.rewards)


def compact_action(acts: dict, i: int, n_tokens: int, decoder: str) -> dict:
    """Row `i` of a batch of chosen actions, small enough for every recorded step: the units decoder as
    [token, uverb, ubp, uenh, uoption, ucount, uplace, uanchor, ucell, uspot, utarget] for the units given a verb
    (every other unit chose nothing); the slots decoder as its per-slot lists (sets as token lists)."""
    row = action_row(acts, i, n_tokens, decoder)
    if decoder == "units":
        from .tools_model import UNIT_KEYS
        given = np.flatnonzero(np.asarray(row["uverb"]) > 0)
        return {"units": [[int(t)] + [int(row[k][t]) for k in UNIT_KEYS] for t in given]}
    out = {}
    for k, v in row.items():
        v = np.asarray(v)
        out[k] = [np.flatnonzero(slot).tolist() for slot in v] if k == "set" else v.astype(int).tolist()
    return out


def expand_action(compact: dict, n_tokens: int, decoder: str) -> dict:
    """compact_action back to an action row (what `stack_actions` takes)."""
    from .tools_model import ACTION_KEYS, K, UNIT_KEYS
    if decoder == "units":
        row = {k: np.full(n_tokens, 0 if k == "uverb" else -1, np.int64) for k in UNIT_KEYS}
        for entry in compact.get("units") or []:
            t = entry[0]
            for k, v in zip(UNIT_KEYS, entry[1:]):
                row[k][t] = v
        return row
    row = {}
    for k in ACTION_KEYS:
        if k == "set":
            arr = np.zeros((K, n_tokens), bool)
            for slot, members in enumerate(compact["set"]):
                arr[slot, members] = True
            row[k] = arr
        else:
            row[k] = np.asarray(compact[k], np.int64)
    return row


def widen(obs: dict) -> dict:
    return widen_observation(obs)


def normalized_distillation_loss(bc: torch.Tensor, valid: torch.Tensor, supervised: torch.Tensor,
                                supervised_rows: float) -> torch.Tensor:
    """Normalize imitation loss by rows that have a usable teacher label, not merely a teacher response."""
    return (bc * valid * supervised).sum() / max(float(supervised_rows), 1.0)


def decoded_set_counts(actions: dict, row: int) -> tuple[dict, dict]:
    """Count selected group verbs before command conversion drops empty subject sets."""
    selected, empty = {}, {}
    for slot, index in enumerate(actions["verb"][row]):
        verb = VERBS[int(index)]
        if verb == "end":
            break
        if verb in SET_SUBJECT:
            selected[verb] = selected.get(verb, 0) + 1
            if not bool(actions["set"][row, slot].any()):
                empty[verb] = empty.get(verb, 0) + 1
    return selected, empty


class ToolsTrainer(Trainer):
    def __init__(self, config: ToolsConfig, device: str | None = None):
        validate_distillation(config)  # configs are mutable; check again before base allocation or filesystem work
        decision_ticks(config.decision_ticks)
        super().__init__(config, device)
        self.actor = self.critic = None          # the menu networks are not used here
        if self.device.type == "cuda":
            # set either way: cloning earlier in the same process turns TF32 on, and exact ratios need it off
            torch.backends.cuda.matmul.allow_tf32 = config.amp_encoder
        self.net = ToolsNet(config.size, amp=config.amp_encoder, decoder=config.decoder).to(self.device)
        self.acting = ToolsNet(config.size, amp=config.amp_encoder, decoder=config.decoder).to(self.device)
        self.acting.load_state_dict(self.net.state_dict())
        self.acting.eval()
        self.net.grad_checkpoint = bool(config.grad_checkpoint)
        self.factored = config.decoder == "units" and bool(config.unit_ratio)
        self.reference: ToolsNet | None = None
        if config.ladder:
            self.stages = LADDERS[config.ladder_name]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.memory: dict[int, tuple] = {}
        self.thread: threading.Thread | None = None
        self.thread_error: BaseException | None = None
        self.thread_record: dict | None = None
        # Updates run on a low-priority stream and the games' decisions on a high-priority one: a decision is
        # hundreds of small kernels, and behind an update's large ones each waited its turn (measured: ~150 ms
        # per decision batch for ~15 ms of encoder work), with twenty games idle meanwhile.
        cuda = self.device.type == "cuda"
        self.stream = torch.cuda.Stream(priority=0) if cuda else None
        self.act_stream = torch.cuda.Stream(priority=-10) if cuda and config.act_priority else None
        if self.device.type == "cuda" and config.gpu_memory_fraction:
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction, index)
        self.micro = max(1, config.micro_seqs)
        self.entropy_ref: float | None = None
        self.last_entropy: float | None = None
        print(f"tools network {config.size}: {parameter_count(self.net) / 1e6:.2f} M weights; training games act at "
              f"temperature {config.act_temperature}")

    # ------------------------------------------------------------------ episodes
    def _episode(self, mode: str) -> dict:
        cadence = decision_ticks(self.cfg.decision_ticks)
        ep = super()._episode(mode)
        ep["decision_ticks"] = decision_ticks(ep.get("decision_ticks", cadence))
        ep["kind"] = "tools"
        ep["distill_teacher"] = self.cfg.distill_teacher
        ep.pop("teacher_labels", None)
        if self.cfg.fake:
            ep["fake_raw"] = self.cfg.fake_raw
        if not ep.get("eval") and self.mode == "ppo" and self.distill_beta() > 0:
            ep["teacher_labels"] = True
        if ep.get("labels"):            # a scheduled game that asks for a teacher's labels itself (tools_eval --labels)
            ep["distill_teacher"], ep["teacher_labels"] = ep["labels"], True
        return ep

    def schedule_eval(self, games: int, ai: str, rule: bool = False, minutes: float | None = None) -> None:
        cadence = decision_ticks(self.cfg.decision_ticks)
        before = len(self.eval_queue)
        super().schedule_eval(games, ai, rule, minutes)
        for ep in self.eval_queue[before:]:
            ep["decision_ticks"] = cadence
            ep["kind"] = "tools"
            ep["distill_teacher"] = self.cfg.distill_teacher
            ep.pop("teacher_labels", None)
            if self.cfg.fake:
                ep["fake_raw"] = self.cfg.fake_raw

    def _send_reset(self, wid: int, mode: str) -> None:
        super()._send_reset(wid, mode)
        zero = torch.zeros(1, 1, self.net.core_dim)
        self.memory[wid] = (zero.to(self.device), zero.clone().to(self.device))
        worker = self.workers[wid]
        worker.segment = ToolSegment((zero, zero.clone()), first_is_start=True)
        worker.command_telemetry = {"policy_decisions": 0, "empty_command_decisions": 0, "issued_commands": 0,
                                    "sent_actions": 0, "action_errors": 0, "verbs": {}, "builds": {},
                                    "productions": {}, "action_error_details": {},
                                    "decoded_set_verbs": {}, "empty_set_verbs": {}}

    def _handle(self, wid: int, msg, episode_mode: str):
        """Attach compact command/action diagnostics to each finished tools episode."""
        worker = self.workers[wid]
        uid = msg[2] if msg[0] == "error" and len(msg) > 2 else (msg[4].get("uid") if msg[0] == "obs" else None)
        episode = worker.episode or {}
        if msg[0] == "obs" and uid == episode.get("uid"):
            _, _, _, done, info = msg
            telemetry = getattr(worker, "command_telemetry", None)
            if telemetry is not None and not info.get("first"):
                try:
                    telemetry["sent_actions"] += int(info.get("sent") or 0)
                    telemetry["action_errors"] += int(info.get("action_errors") or 0)
                except (TypeError, ValueError, OverflowError):
                    pass
                for detail in info.get("errors") or []:
                    key = str(detail).splitlines()[0][:120] if detail else "unknown"
                    telemetry["action_error_details"][key] = telemetry["action_error_details"].get(key, 0) + 1
            if done and telemetry is not None and isinstance(info.get("episode"), dict):
                info["episode"]["command_telemetry"] = {
                    **telemetry,
                    "verbs": dict(telemetry["verbs"]),
                    "builds": dict(telemetry["builds"]),
                    "productions": dict(telemetry["productions"]),
                    "action_error_details": dict(telemetry["action_error_details"]),
                    "decoded_set_verbs": dict(telemetry["decoded_set_verbs"]),
                    "empty_set_verbs": dict(telemetry["empty_set_verbs"]),
                }
        return super()._handle(wid, msg, episode_mode)

    def _restart_worker(self, wid: int, episode_mode: str, reason: str) -> None:
        super()._restart_worker(wid, episode_mode, reason)

    # ------------------------------------------------------------------ schedules
    def distill_beta(self) -> float:
        c = self.cfg
        validate_distillation(c)
        if c.distill_teacher == "none":
            return 0.0
        if self.in_dagger():
            return 1.0
        done = max(0, self.update - c.dagger_updates)           # the fade starts when PPO does
        frac = min(1.0, done / max(1.0, c.distill_fraction * max(1, c.planned_updates - c.dagger_updates)))
        return c.distill_end + (c.distill_start - c.distill_end) * (1.0 - frac)

    def in_dagger(self) -> bool:
        return self.cfg.distill_teacher != "none" and self.update < self.cfg.dagger_updates

    def learning_rate(self) -> float:
        """The base schedule, warmed up over the first updates: a fresh Adam moves every weight by about the
        learning rate on its first steps, and a sharp (low-temperature) policy then drifts far past the KL
        target (measured: KL 0.08 -> 1.0 within six minibatches at 1e-4)."""
        if self.in_dagger():
            return self.cfg.dagger_lr * min(1.0, (self.update + 1) / 3)
        since = self.update - (self.cfg.dagger_updates if self.cfg.distill_teacher != "none" else 0)
        return super().learning_rate() * min(1.0, (since + 1) / max(1, self.cfg.warmup_updates))

    def ent_coef(self) -> float:
        c = self.cfg
        if self.entropy_ref and self.last_entropy is not None and self.last_entropy > c.entropy_ceiling * self.entropy_ref:
            return 0.0                              # the brake: no push towards randomness while too random
        return super().ent_coef()

    # ------------------------------------------------------------------ acting
    def _act_stat(self, key: str, value: float) -> None:
        stats = self.__dict__.setdefault("act_stats", {})
        stats[key] = stats.get(key, 0.0) + value

    def _log_throughput(self) -> None:
        """The throughput line, then where a decision's time goes: the game's step (simulation, bridge, observing:
        step_ms), the wait from an observation's arrival to its orders, the batches, and the deciding process's
        own time (preparing, the network, turning outputs into orders) against its idle time."""
        wall = time.monotonic() - self.throughput_t0
        super()._log_throughput()
        st = self.__dict__.pop("act_stats", {})
        idle, self.idle_s = getattr(self, "idle_s", 0.0), 0.0
        n = st.get("decisions", 0.0)
        if n:
            k = max(1.0, st.get("steps", 0.0))
            print(f"   game step: bridge {st.get('bridge_ms', 0) / k:.0f} ms over {st.get('bridge_trips', 0) / k:.2f} "
                  f"round trips (the game simulating: {st.get('sim_ms', 0) / k:.0f} ms of the last), "
                  f"observing {st.get('observe_ms', 0) / k:.1f} ms, rest of the step "
                  f"{(st.get('step_ms', 0) - st.get('bridge_ms', 0) - st.get('observe_ms', 0)) / k:.0f} ms", flush=True)
            print(f"   per decision: game step {st.get('step_ms', 0) / max(1, st.get('steps', 0)):.0f} ms, "
                  f"waited for orders {1000 * st.get('latency', 0) / n:.0f} ms; batches of {n / st['batches']:.1f}; "
                  f"decider: prepare {1000 * st.get('prep', 0) / n:.1f}, network {1000 * st.get('net', 0) / n:.1f}, "
                  f"orders {1000 * st.get('post', 0) / n:.1f} ms per decision; idle {100 * idle / wall:.0f} %",
                  flush=True)

    def _act(self, pending: list) -> None:
        from fa_train.watch_debug import watch_payload
        learn, greedy = [], []
        for wid, obs, info in pending:
            ep = self.workers[wid].episode or {}
            (greedy if ep.get("eval") else learn).append((wid, obs, info))
        for group, sample in ((learn, True), (greedy, False)):
            if not group:
                continue
            t0 = time.monotonic()
            for _, _, info in group:
                if info.get("step_ms") is not None:
                    self._act_stat("step_ms", float(info["step_ms"]))
                    self._act_stat("steps", 1.0)
                    for key in ("bridge_ms", "observe_ms", "sim_ms", "bridge_trips"):
                        self._act_stat(key, float(info.get(key) or 0.0))
            with self._acting_stream():
                obs_list = [widen(o) for _, o, _ in group]
                b = collate(obs_list, self.device)
                t1 = time.monotonic()
                h_in = torch.cat([self.memory[wid][0] for wid, _, _ in group], 1)
                c_in = torch.cat([self.memory[wid][1] for wid, _, _ in group], 1)
                watching = [bool((self.workers[wid].episode or {}).get("debug_dir")) for wid, _, _ in group]
                self.acting.debug = any(watching)
                with torch.no_grad():
                    if sample:
                        mode, temperature = "sample", self.cfg.act_temperature
                    else:
                        mode, temperature = self.cfg.eval_mode, self.cfg.eval_temperature
                        self.acting.act_top_p = self.cfg.eval_top_p
                    try:
                        acts, logp, value, (h, c) = self.acting.forward_step(b, (h_in, c_in), mode,
                                                                           temperature=temperature)
                    finally:
                        self.acting.act_top_p = None
                acts_np = {k: v.cpu().numpy() for k, v in acts.items()}
                logp, value = logp.cpu().numpy(), value.cpu().numpy()
                t2 = time.monotonic()
                unit_logp = getattr(self.acting, "last_unit_logp", None)
                unit_logp = unit_logp.cpu().numpy() if unit_logp is not None else None
                dbg = self.acting.last_debug if self.acting.debug else None
                self.acting.debug = False
            for i, (wid, obs, info) in enumerate(group):
                w = self.workers[wid]
                self.memory[wid] = (h[:, i:i + 1].clone(), c[:, i:i + 1].clone())
                n_tok = len(obs["ids"])
                commands = actions_to_commands(acts_np, i, obs, self.cfg.decoder)
                telemetry = getattr(w, "command_telemetry", None)
                if telemetry is not None:
                    selected_sets, empty_sets = decoded_set_counts(acts_np, i) if self.cfg.decoder == "slots" \
                        else ({}, {})
                    for name, counts in (("decoded_set_verbs", selected_sets), ("empty_set_verbs", empty_sets)):
                        for verb, count in counts.items():
                            telemetry[name][verb] = telemetry[name].get(verb, 0) + count
                    telemetry["policy_decisions"] += 1
                    telemetry["issued_commands"] += len(commands)
                    if not commands:
                        telemetry["empty_command_decisions"] += 1
                    for command in commands:
                        verb = str(command.get("verb", "unknown"))
                        telemetry["verbs"][verb] = telemetry["verbs"].get(verb, 0) + 1
                        if verb == "build":
                            bp = str(command.get("bp", "unknown"))
                            telemetry["builds"][bp] = telemetry["builds"].get(bp, 0) + 1
                        elif verb == "produce":
                            bp = str(command.get("bp", "unknown"))
                            telemetry["productions"][bp] = telemetry["productions"].get(bp, 0) + 1
                if sample:
                    seg = w.segment
                    if len(seg.obs) >= self.cfg.segment and len(seg) == len(seg.obs):
                        self._close_segment(seg, bootstrap=float(value[i]))
                        state = (h_in[:, i:i + 1].detach().cpu(), c_in[:, i:i + 1].detach().cpu())
                        w.segment = seg = ToolSegment(state, first_is_start=False)
                    seg.obs.append(obs)
                    seg.actions.append(action_row(acts_np, i, n_tok, self.cfg.decoder))
                    seg.logp.append(float(logp[i]))
                    seg.values.append(float(value[i]))
                    seg.teacher.append(info.get("teacher"))
                    seg.unit_logp.append(unit_logp[i][:n_tok].copy() if unit_logp is not None else None)
                debug = watch_payload(dbg, i, obs) if dbg is not None and watching[i] else None
                if debug is not None:
                    w.conn.send(("act", commands, None, debug))
                elif (w.episode or {}).get("record_dir"):
                    # what the network itself chose, for learning from the recording later (offline PPO): the
                    # log-probabilities it chose with, its value estimate, and the chosen indices
                    policy = {"logp": round(float(logp[i]), 5), "value": round(float(value[i]), 5),
                              "mode": mode, "temperature": temperature,
                              "action": compact_action(acts_np, i, n_tok, self.cfg.decoder)}
                    if unit_logp is not None:
                        policy["unit_logp"] = [round(float(x), 5) for x in unit_logp[i][:n_tok]]
                    w.conn.send(("act", commands, policy))
                else:
                    w.conn.send(("act", commands))
                if getattr(w, "t_recv", None) is not None:
                    self._act_stat("latency", time.monotonic() - w.t_recv)
            t3 = time.monotonic()
            self._act_stat("prep", t1 - t0)
            self._act_stat("net", t2 - t1)
            self._act_stat("post", t3 - t2)
            self._act_stat("decisions", float(len(group)))
            self._act_stat("batches", 1.0)

    def _acting_stream(self):
        """The high-priority stream for a decision, after everything queued on the default stream so far
        (fresh memories of reset games, weights loaded after an update)."""
        import contextlib
        if self.act_stream is None:
            return contextlib.nullcontext()
        self.act_stream.wait_stream(torch.cuda.default_stream(self.device))
        return torch.cuda.stream(self.act_stream)

    def _close_segment(self, seg, bootstrap: float) -> None:
        n = len(seg)
        if n == 0:
            return
        seg.obs, seg.actions, seg.logp, seg.values, seg.teacher = \
            seg.obs[:n], seg.actions[:n], seg.logp[:n], seg.values[:n], seg.teacher[:n]
        seg.unit_logp = seg.unit_logp[:n]
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

    # ------------------------------------------------------------------ updates (background)
    def _maybe_update(self) -> bool:
        if self.thread is not None:
            if self.thread.is_alive():
                return False
            self._finish_update()
            return True
        if sum(len(s) for s in self.ready) >= self.cfg.batch:
            segs, self.ready = self.ready, []
            self.thread_error, self.thread_record = None, None
            self.thread = threading.Thread(target=self._update_thread, args=(segs,), daemon=True)
            self.thread.start()
        return False

    def _finish_update(self) -> None:
        self.thread.join()
        self.thread = None
        if self.thread_error is not None:
            raise RuntimeError("PPO update failed") from self.thread_error
        self.acting.load_state_dict(self.net.state_dict())
        self.update += 1
        record = self.thread_record or {}
        if record.get("entropy") is not None:
            self.last_entropy = float(record["entropy"])
            if self.entropy_ref is None and self.last_entropy > 0:
                self.entropy_ref = self.last_entropy
        if self.device.type == "cuda":
            torch.cuda.empty_cache()                # give the cache back between updates
        record.update(update=self.update, time=datetime.now().isoformat(timespec="seconds"),
                      stage=self.stages[self.stage]["name"] if self.cfg.curriculum else None,
                      planned_updates=self.cfg.planned_updates)
        import json
        self.update_log.write(json.dumps(record) + "\n")
        self.update_log.flush()
        if self.writer:
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"ppo/{k}", v, self.update)
        print(f"update {self.update}: {record.get('samples')} samples, pg {record.get('pg_loss', 0):+.3f} "
              f"v {record.get('v_loss', 0):.3f} ent {record.get('entropy', 0):.2f} kl {record.get('approx_kl', 0):.4f} "
              f"distill {record.get('distill_loss', 0):.3f} (x{record.get('distill_beta', 0):.2f}) "
              f"ref {record.get('ref_kl', 0):.4f} (x{record.get('ref_coef', 0):.2f}) "
              f"belief {record.get('belief_loss', 0):.3f} ev {record.get('explained_variance', 0):+.2f} "
              f"micro {record.get('micro_seqs')} peak {record.get('peak_gb', 0):.1f} GB ({record.get('seconds', 0):.1f} s)")

    def _update_thread(self, segs) -> None:
        try:
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    self.thread_record = self._ppo_update_tools(segs)
                self.stream.synchronize()
            else:
                self.thread_record = self._ppo_update_tools(segs)
        except BaseException as exc:          # reported (and re-raised) by the main thread
            self.thread_error = exc

    def _sequences(self, mb: list, L: int, teacher_rows: dict | None = None):
        """A minibatch of segments as one padded, sequence-major batch."""
        dev = self.device
        obs, acts, valid, starts, old_logp, old_v, adv, ret, teach, teach_has, old_ulp = \
            [], [], [], [], [], [], [], [], [], [], []
        for seg in mb:
            n = len(seg)
            for t in range(L):
                j = min(t, n - 1)
                obs.append(widen(seg.obs[j]))
                acts.append(seg.actions[j])
                valid.append(t < n)
                starts.append(t == 0 and seg.first_is_start)
                old_logp.append(seg.logp[j])
                old_ulp.append(seg.unit_logp[j] if j < len(seg.unit_logp) else None)
                old_v.append(seg.values[j])
                adv.append(seg.adv_n[j])
                ret.append(seg.returns[j])
                if teacher_rows is None:
                    teach.append(seg.teacher[j] if t < n else None)
                    teach_has.append(False)
                else:
                    commands, masks = teacher_rows[id(seg)]
                    teach.append(commands[t] if t < n else None)
                    teach_has.append(masks[t] if t < n else False)
        b = collate(obs, dev)
        T = b["tokens"].shape[1]
        act = {k: torch.as_tensor(v, device=dev) for k, v in stack_actions(acts, T, self.cfg.decoder).items()}
        state = (torch.cat([s.state[0] for s in mb], 1).to(dev), torch.cat([s.state[1] for s in mb], 1).to(dev))
        f = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)  # noqa: E731
        ulp = np.zeros((len(obs), T), np.float32)           # the per-unit log-probabilities when acting, padded
        for i, row in enumerate(old_ulp):
            if row is not None:
                ulp[i, : len(row)] = row
        return (b, act, state, f(starts, torch.bool), f(valid), f(old_logp), f(old_v), f(adv), f(ret),
                teach, teach_has, obs, f(ulp))

    def _validated_teacher_row(self, cmds: list | None, obs: dict) -> tuple[list | None, bool]:
        if cmds is None:
            return None, False
        from .tools_bc import filter_commands
        try:
            accepted, _ = filter_commands(cmds, obs, source="online_distillation")
        except (KeyError, IndexError, TypeError, ValueError):
            accepted = []
        # [] is an intentional END label; a nonempty but fully masked list is unknown behavior.
        return accepted, (not cmds or bool(accepted))

    def _prepare_teacher_rows(self, mb: list, L: int) -> tuple[dict, float]:
        """Filter labels once and count only rows that can contribute distillation loss in the whole minibatch."""
        rows = {}
        count = 0
        for seg in mb:
            commands, supervised = [], []
            for t in range(min(len(seg), L)):
                accepted, has = self._validated_teacher_row(seg.teacher[t], seg.obs[t])
                commands.append(accepted)
                supervised.append(has)
                count += int(has)
            rows[id(seg)] = (commands, supervised)
        return rows, float(count)

    def _teacher_labels(self, labels: list, supervised: list[bool], T: int):
        """Pack validated command labels into padded tensors for the policy's teacher-forced score pass."""
        dev = self.device
        rows = [lab if lab is not None else self._no_labels(1) for lab in labels]
        act = {k: torch.as_tensor(v, device=dev) for k, v in stack_actions(rows, T, self._decoder).items()}
        return act, torch.as_tensor(np.asarray(supervised, np.float32), device=dev)

    @property
    def _decoder(self) -> str:
        return getattr(getattr(self, "cfg", None), "decoder", "slots")

    def _no_labels(self, n_tokens: int) -> dict:
        return empty_unit_labels(n_tokens) if self._decoder == "units" else empty_labels(n_tokens)

    def _labels(self, commands: list, obs: dict) -> dict:
        """Commands (already filtered for this observation) as the decoder's labels."""
        if self._decoder == "units":
            from .tools_bc import lookups, unit_labels
            return unit_labels(commands, obs, lookups(obs))
        return commands_to_indices(commands, obs)

    def _teacher_actions(self, teach: list, obs: list, T: int, supervised_override: list[bool] | None = None,
                         prevalidated: bool = False):
        """Teacher commands -> index labels (and which rows have any). Commands the student's masks do not
        allow are dropped, as cloning drops them: a label on a masked option scores about 1e9 (the day run of
        2026-09-30 logged distillation losses in the millions from ~0.2 % of its labels)."""
        labels = []
        has = []
        for i, (cmds, o) in enumerate(zip(teach, obs)):
            if prevalidated:
                accepted = cmds
                supervised = bool(supervised_override[i]) if supervised_override is not None else False
            else:
                accepted, supervised = self._validated_teacher_row(cmds, o)
            if accepted is None:
                labels.append(self._no_labels(len(o["ids"])))
                has.append(False)
            else:
                labels.append(self._labels(accepted, o))
                has.append(supervised)
        return self._teacher_labels(labels, has, T)

    def _ppo_update_tools(self, segs: list) -> dict:
        t0 = time.monotonic()
        cfg = self.cfg
        L = cfg.segment
        tau = cfg.act_temperature
        all_adv = np.concatenate([s.advantages for s in segs])
        mean, std = float(all_adv.mean()), float(all_adv.std() + 1e-8)
        for s in segs:
            s.adv_n = (s.advantages - mean) / std
        ent_coef, beta, lr = self.ent_coef(), self.distill_beta(), self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        stats = {k: [] for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "distill_loss", "belief_loss",
                                 "ref_kl", "decision_kl")}
        head_ent = {k: [] for k in cfg.ent_weights}
        rng = np.random.default_rng()
        stop = False
        epoch = 0
        for epoch in range(cfg.epochs):
            order = rng.permutation(len(segs))
            for start in range(0, len(order), cfg.minibatch_seqs):
                mb = [segs[i] for i in order[start:start + cfg.minibatch_seqs]]
                while True:
                    try:
                        sums = self._minibatch(mb, L, tau, ent_coef, beta)
                        break
                    except torch.cuda.OutOfMemoryError:
                        self.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        if self.micro == 1:
                            sums = None
                            break
                        self.micro = max(1, self.micro // 2)
                        print(f"out of GPU memory in an update: micro-batches of {self.micro} segments from now on")
                if sums is None or not all(np.isfinite(v) for v in (sums["loss"],)):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.skipped_minibatches += 1
                    continue
                norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad)
                if not torch.isfinite(norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.skipped_minibatches += 1
                    continue
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                for k in stats:
                    stats[k].append(sums[k])
                for k in head_ent:
                    head_ent[k].append(sums["head_" + k])
                if sums["approx_kl"] > 1.5 * cfg.target_kl and not self.in_dagger():
                    stop = True
                    break
            if stop:
                break
        returns = np.concatenate([s.returns for s in segs])
        values = np.concatenate([np.asarray(s.values, np.float32) for s in segs])
        var = returns.var()
        explained = float(1 - (returns - values).var() / var) if var > 0 else 0.0
        mean_of = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731
        record = {k: mean_of(v) for k, v in stats.items()}
        peak = torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if self.device.type == "cuda" else 0.0
        record.update(samples=int(sum(len(s) for s in segs)), epochs_run=epoch + 1, explained_variance=explained,
                      ent_coef=ent_coef, distill_beta=beta, lr=lr, shaping=self.shaping(), return_std=self.rms.std,
                      skipped_minibatches=self.skipped_minibatches, seconds=time.monotonic() - t0,
                      micro_seqs=self.micro, peak_gb=peak, temperature=tau, ref_coef=cfg.ref_coef,
                      factored=self.factored, **{f"entropy_{k}": mean_of(v) for k, v in head_ent.items()})
        return record

    def _minibatch(self, mb: list, L: int, tau: float, ent_coef: float, beta: float) -> dict:
        """Forward and backward for one minibatch in micro-batches (gradients accumulate; the optimizer step is the
        caller's). Every sum is normalised by the whole minibatch, so the result does not depend on the split.
        Returns the minibatch's statistics."""
        cfg = self.cfg
        n_valid = float(sum(min(len(s), L) for s in mb)) or 1.0
        teacher_rows, n_teach = self._prepare_teacher_rows(mb, L) if beta > 0 else (None, 1.0)
        n_teach = max(1.0, n_teach)
        sums = {k: 0.0 for k in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "distill_loss",
                                 "belief_loss", "ref_kl", "decision_kl")}
        sums.update({"head_" + k: 0.0 for k in cfg.ent_weights})
        self.optimizer.zero_grad(set_to_none=True)
        for i in range(0, len(mb), self.micro):
            chunk = mb[i:i + self.micro]
            b, act, state, starts, valid, old_logp, old_v, adv, ret, teach, teach_has, obs, old_ulp = \
                self._sequences(chunk, L, teacher_rows)
            h, m, z = self.net.core(b, state, seq_len=L, starts=starts)
            logp, ent, _ = self.net.score(h, m, z, b, act, temperature=tau)
            value, belief = self.net.heads(z, b)
            w = valid / n_valid
            ratio = (logp - old_logp).exp()
            log_ratio = logp - old_logp
            decision_kl = (((ratio - 1) - log_ratio) * w).sum()          # the joint ratio's KL, for comparison
            if self.factored:
                # every unit its own action: its ratio against its own log-probability when acting, the decision's
                # advantage; a decision's loss is the mean over its scored units
                ulp, known = ent["unit_logp"], ent["unit_known"]                                   # [N,T]
                n_units = known.sum(1).clamp(min=1.0)
                per_unit = lambda x: ((x * known).sum(1) / n_units * w).sum()  # noqa: E731
                u_log_ratio = (ulp - old_ulp) * known
                u_ratio = u_log_ratio.exp()
                pg = per_unit(torch.max(-adv.unsqueeze(1) * u_ratio,
                                        -adv.unsqueeze(1) * u_ratio.clamp(1 - cfg.clip, 1 + cfg.clip)))
                approx_kl = per_unit((u_ratio - 1) - u_log_ratio)
                clipfrac = per_unit(((u_ratio - 1).abs() > cfg.clip).float())
            else:
                pg = (torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)) * w).sum()
                approx_kl = decision_kl
                clipfrac = (((ratio - 1).abs() > cfg.clip).float() * w).sum()
            v_clip = old_v + (value - old_v).clamp(-cfg.vf_clip, cfg.vf_clip)
            v_loss = 0.5 * (torch.max((value - ret) ** 2, (v_clip - ret) ** 2) * w).sum()
            entropy = sum(cfg.ent_weights[k] * ent[k] for k in cfg.ent_weights)
            if self.factored:
                # per unit, like the factored policy term (ent[k] is a decision's sum over its units)
                entropy = entropy / ent["unit_known"].sum(1).clamp(min=1.0)
            belief_loss = ((((belief - b["belief"]) ** 2).mean(-1)) * w).sum()
            loss = (0.0 if self.in_dagger() else 1.0) * pg + cfg.vf_coef * v_loss \
                - ent_coef * (entropy * w).sum() + cfg.belief_coef * belief_loss
            ref_kl = torch.zeros((), device=self.device)
            if self.reference is not None and cfg.ref_coef > 0:
                # the reference scores the same actions from the same stored memory (it started as this network;
                # the anchor keeps them close, so its own memory would differ little)
                with torch.no_grad():
                    hr, mr, zr = self.reference.core(b, state, seq_len=L, starts=starts)
                    lp_ref, ent_ref, _ = self.reference.score(hr, mr, zr, b, act, temperature=tau)
                if self.factored:
                    d = (ent_ref["unit_logp"] - ulp) * known
                    ref_kl = per_unit(d.exp() - d - 1)
                else:
                    d = lp_ref - logp
                    ref_kl = ((d.exp() - d - 1) * w).sum()
                loss = loss + cfg.ref_coef * ref_kl
            distill = torch.zeros((), device=self.device)
            if beta > 0 and any(teach_has):
                t_act, has = self._teacher_actions(teach, obs, b["tokens"].shape[1],
                                                   supervised_override=teach_has, prevalidated=True)
                # scored at temperature 1, like cloning: at the acting temperature, the smoothed labels pull the
                # sharpened policy flat again (measured: entropy 0.62 -> 0.94 in 18 distillation-only steps at
                # 0.5, unchanged at 1; the day run's entropy went 0.54 -> 2.4 in 18 updates)
                _, _, bc = self.net.score(h, m, z, b, t_act, smoothing=0.05, temperature=1.0)
                if self.factored:
                    # per unit, like the factored policy term (bc is a decision's sum over its units)
                    bc = bc / ent["unit_known"].sum(1).clamp(min=1.0)
                distill = normalized_distillation_loss(bc, valid, has, n_teach)
                loss = loss + beta * distill
            if not torch.isfinite(loss):
                sums["loss"] = float("nan")
                return sums
            loss.backward()
            with torch.no_grad():
                sums["loss"] += float(loss)
                sums["pg_loss"] += float(pg)
                sums["v_loss"] += float(v_loss)
                sums["entropy"] += float((entropy * w).sum())
                sums["approx_kl"] += float(approx_kl)
                sums["decision_kl"] += float(decision_kl)
                sums["clipfrac"] += float(clipfrac)
                sums["distill_loss"] += float(distill)
                sums["belief_loss"] += float(belief_loss)
                sums["ref_kl"] += float(ref_kl)
                for k in cfg.ent_weights:
                    sums["head_" + k] += float((ent[k] * w).sum())
        return sums

    def bc_beta(self) -> float:          # the menu trainer's KL pull is replaced by teacher distillation
        return 0.0

    # ------------------------------------------------------------------ checkpoints
    def save(self, name: str) -> Path:
        if self.thread is not None:
            self._finish_update()
        contract = runtime_set_distribution(self.net, self.acting)
        path = self.run_dir / "checkpoints" / name
        state = {"model": self.net.state_dict(), "optimizer": self.optimizer.state_dict(), "size": self.cfg.size,
                 "decoder": self.cfg.decoder, "model_options": model_options(self.net),
                 "update": self.update, "episodes": self.episodes, "rms": (self.rms.mean, self.rms.var, self.rms.count),
                 "config": asdict(self.cfg), "stage": self.stage, "kind": "tools",
                 "entropy_ref": self.entropy_ref}
        if contract is not None:
            state["set_distribution"] = contract
        state["distillation"] = distillation_metadata(self.cfg, self.distill_beta())
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        return path

    def load(self, path, optimizer: bool = True) -> None:
        validate_distillation(self.cfg)
        decision_ticks(self.cfg.decision_ticks)
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("decoder", "slots") != self.cfg.decoder:
            raise ValueError(f"checkpoint decoder {state.get('decoder', 'slots')!r} differs from the trainer's "
                             f"{self.cfg.decoder!r}")
        validate_checkpoint_set_distribution(state, self.net, self.acting)
        cadence = checkpoint_cadence(state.get("config"), self.cfg.decision_ticks)
        self.net.load_state_dict(state["model"])
        self.acting.load_state_dict(state["model"])
        options = apply_model_options(self.net, state)       # the network runs as it was trained
        apply_model_options(self.acting, state)
        print(f"model options: {options}")
        if optimizer and state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
        self.update = state.get("update", 0)
        self.episodes = max(self.episodes, state.get("episodes", 0))
        self.stage = min(state.get("stage", 0), len(self.stages) - 1)
        if state.get("entropy_ref"):           # the brake keeps its reference across a resume
            self.entropy_ref = float(state["entropy_ref"])
        if state.get("rms"):
            self.rms.mean, self.rms.var, self.rms.count = state["rms"]
        elif state.get("ret_std"):                       # a cloned network: returns were scaled by this
            self.rms.mean, self.rms.var, self.rms.count = 0.0, state["ret_std"] ** 2, 1e4
        self._log_event({"event": "checkpoint_cadence", "path": str(path), **cadence})
        if cadence["mismatch"]:
            print(f"WARNING: checkpoint decision cadence differs: saved={cadence['saved_decision_ticks']}, "
                  f"active={cadence['active_decision_ticks']}. Runtime config remains authoritative.")
        elif cadence["saved_provenance"] == "unavailable":
            print(f"checkpoint cadence unavailable; active decision_ticks={cadence['active_decision_ticks']}. "
                  "Verify demonstration cadence before training; it is not inferred from weights.")
        # The active schedule uses the resumed update, while runtime config remains authoritative.
        provenance = checkpoint_distillation(state.get("config"), self.cfg, self.distill_beta())
        event = {"event": "checkpoint_distillation", "path": str(path), **provenance,
                 "saved_config": state.get("config"), "active_config": asdict(self.cfg)}
        self._log_event(event)
        print(f"checkpoint distillation: saved={provenance['saved']}; active={provenance['active']}; "
              "runtime config remains authoritative")
        if provenance["mismatches"]:
            print("WARNING: checkpoint distillation differs from active runtime config: "
                  + ", ".join(provenance["mismatches"]) + ". Resume/clone does not restore these options.")
        for diagnostic in provenance["diagnostics"]:
            print(f"checkpoint distillation: {diagnostic}")

    def set_teacher(self, path) -> None:
        raise NotImplementedError("unsupported checkpoint teacher for ToolsTrainer; select distill_teacher='v3' "
                                  "or 'none' via ToolsConfig. load()/clone/resume initialize the model only.")

    def set_reference(self, path) -> None:
        """Freeze the network saved at `path` (the DAgger clone PPO starts from) as the KL anchor (`ref_coef`)."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("decoder", "slots") != self.cfg.decoder or state.get("size", "M") != self.cfg.size:
            raise ValueError(f"reference {path} is a {state.get('size')} {state.get('decoder', 'slots')} network; "
                             f"the trainer's is {self.cfg.size} {self.cfg.decoder}")
        self.reference = ToolsNet(self.cfg.size, amp=self.cfg.amp_encoder, decoder=self.cfg.decoder).to(self.device)
        self.reference.load_state_dict(state["model"])
        apply_model_options(self.reference, state)
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.cfg.reference = str(path)
        self._log_event({"event": "reference", "path": str(path), "ref_coef": self.cfg.ref_coef})

    def close(self) -> None:
        if self.thread is not None:
            try:
                self._finish_update()
            except Exception:
                pass
        super().close()


__all__ = ["ToolsConfig", "ToolsTrainer", "WorkerState"]
