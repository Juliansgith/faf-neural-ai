"""ToolsEnv: a live game driven by commands with arguments (docs/tools-design.md).

    env = ToolsEnv(GameConfig(map="SCMP_016", ai="easy"), time_limit_minutes=20)
    obs, info = env.reset()                         # ToolsObserver output (numpy arrays + "ids")
    obs, rewards, done, info = env.step_tools([{"verb": "build", "subject": [id], "bp": "ueb1101", "cell": (9, 3)}])

It inherits the game loop, rewards and episode bookkeeping from FactoredEnv. `teacher_commands(policy)`
says what the rule teacher would order in the current state, as commands, without executing anything --
the labels of online teacher distillation.
"""
from __future__ import annotations

import copy
import math
import time
from collections import deque

from .factored import HEAD_NAMES, FactoredEnv
from .tools import ToolKit
from .tools_obs import ToolsObserver
from .tools_diagnostics import action_failures, transition_record
from .reclaim_memory import after_step as reclaim_after_step

NO_CHOICE = {h: 0 for h in HEAD_NAMES}
MOVING_ORDERS = frozenset({2, 35})     # a unit's current order as the bridge reports it: Move, AggressiveMove
RECLAIM_SCAN_EVERY = 5                 # decisions between reclaim scans (the observation's reclaim tokens)


class ToolsEnv(FactoredEnv):
    def __init__(self, *args, deduplicate_navigation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.deduplicate_navigation = bool(deduplicate_navigation)
        self.kit: ToolKit | None = None
        self.observer: ToolsObserver | None = None
        self.rallied_by_player: set = set()     # factories this game's player has given a rally point
        # what a teacher wants in the decision about to be played (set by the caller before step_tools; a shadow
        # teacher's come with the step itself): recorded as the step's `labels` -- demonstrations for the states
        # the playing policy gets into (DAgger)
        self.step_labels: list | None = None
        # self-play (fa_env.selfplay): every decision makes both bridge trips, queries or not, so that the two
        # sides of one game meet at each trip
        self.lockstep = False

    def encode(self):          # the menu observation is not needed here
        return None

    def reset(self, config=None):
        _, info = super().reset(config)
        return self._tools_begin(), info

    def attach(self, obs: dict, info: dict):
        """Begin an episode on a game something else runs and steps (a self-play side, fa_env.selfplay)."""
        _, out = FactoredEnv._begin(self, obs, info)
        return self._tools_begin(), out

    def _tools_begin(self):
        self.kit = ToolKit(self.info, self.bps)
        self.observer = ToolsObserver(self.info, self.bps, self.me, self.kit)
        self.rallied_by_player = set()
        self.economy_snapshots = []
        self.recent_state_snapshots = deque(maxlen=31)
        self.navigation_repeats_suppressed = 0
        return self.observer.observe(self.obs)

    def own_units(self) -> dict[str, dict]:
        return {str(u["id"]): u for u in self.obs.get("units", []) if u.get("army") == self.me}

    def step_tools(self, commands: list[dict]):
        t_start = time.monotonic()
        self.bridge_s = 0.0                 # time inside the bridge round trips (the game simulating and observing)
        tick = self.state.tick
        self.state_tick_before = tick
        self.units_before = self.own_units()
        actions, queries, pending = self.kit.plan(commands, self.units_before, set(self.observer.taken))
        if getattr(self, "deduplicate_navigation", False):
            from .identical_navigation import identical_navigation
            before = sum(len(a.get("units", [])) for a in actions)
            actions = identical_navigation(actions, self.own_units())
            # Each suppressed unit removes exactly one clear and one navigation.
            self.navigation_repeats_suppressed += (
                before - sum(len(a.get("units", [])) for a in actions)) // 2
        for a in actions:
            if a["type"] == "rally":
                self.rallied_by_player.update(str(u) for u in a["units"])
        for a in actions:           # a spot ordered now is not free for the next few decisions
            if a["type"] == "build" and self.kit.role.get(a["bp"]) in ("mex", "hydro"):
                self.observer.claim(a["pos"][0], a["pos"][1], tick)
                self.claimed[(round(a["pos"][0]), round(a["pos"][1]))] = tick
        n_sites = len(queries)
        if self.decisions % RECLAIM_SCAN_EVERY == 0:
            queries += self._reclaim_scans()
        raw = self.fa.server.last_line if self.fa.server is not None else None
        sent, answers, results, captured_orders = [], [], [], []
        wait = self.decision_ticks
        if queries or getattr(self, "lockstep", False):
            sent += actions
            t_b = time.monotonic()
            obs, _, terminated, truncated, info = self.fa.step(actions, queries, wait=1)
            self.bridge_s += time.monotonic() - t_b
            self.bridge_trips = 2
            if self.fa.capture:
                captured_orders.extend(obs.get("orders") or [])
            wait -= 1
            answers = info.get("queries") or []
            results += info.get("results") or []
            self._refresh_reclaim(queries, answers)
            actions = self.kit.complete(pending, answers[:n_sites])
            if terminated or truncated:
                return self._tools_result(raw, commands, sent, queries, answers, results,
                                          self._finish(obs, info, terminated, truncated, dict(NO_CHOICE)),
                                          captured_orders)
        sent += actions
        if not queries:
            self.bridge_trips = 1
        t_b = time.monotonic()
        obs, _, terminated, truncated, info = self.fa.step(actions, wait=max(1, wait))
        self.bridge_s += time.monotonic() - t_b
        self.step_t0 = t_start
        if self.fa.capture:
            captured_orders.extend(obs.get("orders") or [])
        results += info.get("results") or []
        return self._tools_result(raw, commands, sent, queries, answers, results,
                                  self._finish(obs, info, terminated, truncated, dict(NO_CHOICE)),
                                  captured_orders)

    def _reclaim_scans(self) -> list[dict]:
        """Reclaim queries: around the base, and around up to three groups of our engineers working away from it
        (the reclaim an army actually walks past; M28 reclaims all over the map)."""
        scans = [{"type": "reclaim", "pos": list(self.start), "radius": 50, "energy": True, "limit": 40}]
        cells = {}
        for u in self.own_units().values():
            if "ENGINEER" in (self.bps.get(u.get("bp"), {}).get("categories") or []) and \
                    math.hypot(u["x"] - self.start[0], u["z"] - self.start[1]) > 40:
                cells.setdefault(self.kit.cell(u["x"], u["z"]), []).append(u)
        for group in sorted(cells.values(), key=len, reverse=True)[:3]:
            cx = sum(u["x"] for u in group) / len(group)
            cz = sum(u["z"] for u in group) / len(group)
            scans.append({"type": "reclaim", "pos": [cx, cz], "radius": 25, "energy": True, "limit": 20})
        return scans

    def _tools_result(self, raw, commands, sent, queries, answers, results, finished, captured_orders=None):
        _, rewards, done, step_info = finished
        tick = self.state_tick_before
        # A tools decision may have two bridge steps (query, then placement).
        # Count/report both batches; the base _finish only sees the final batch.
        failed, details = action_failures(sent, results)
        step_info["action_errors"] = failed
        step_info["action_failures"] = details
        step_info["errors"] = list(step_info.get("errors") or []) + [
            f"action[{d['index']}] {d['type']}: {d['error']}" for d in details]
        self._refresh_reclaim([], [], sent, results)
        self._note_economy_snapshot(done)
        self._note_recent_state()
        if done and isinstance(step_info.get("episode"), dict):
            step_info["episode"]["economy_snapshots"] = list(self.economy_snapshots)
            step_info["episode"]["recent_state_snapshots"] = list(self.recent_state_snapshots)
            step_info["episode"]["navigation_repeats_suppressed"] = getattr(self, "navigation_repeats_suppressed", 0)
        self.observer.note_commands(commands, tick, results)
        if self.fa.capture:
            # another AI plays our army: the orders it gave since the decision's observation are its "actions"
            # Each bridge observation drains capture. A query and its placement
            # produce two observations, so retain orders from both substeps.
            orders = self.obs.get("orders") or [] if captured_orders is None else captured_orders
            orders = [a for a in orders if a.get("type") not in ("attack_ground",)]
            step_info["captured"] = len(orders)
            if getattr(self.fa, "shadow", False):
                # the other AI only watches: what it would have ordered in this decision, as commands for the
                # observation the decision was made from (labels); the agent's own actions stay the game's
                step_info["shadow"] = orders
                step_info["shadow_commands"] = self.kit.commands_from_actions(
                    orders, getattr(self, "units_before", None) or self.own_units())
            else:
                sent = orders
        labels = step_info["shadow_commands"] if "shadow_commands" in step_info else getattr(self, "step_labels", None)
        self.step_labels = None
        if self.recorder is not None:
            self.recorder.step(raw, {"tick": tick, "commands": commands, "sent": sent, "queries": queries,
                                     "answers": answers, "results": results, "labels": labels,
                                     "deduplicate_navigation": getattr(self, "deduplicate_navigation", False),
                                     "policy": getattr(self, "policy_record", None),
                                     "transition": transition_record(tick, self.obs.get("tick", tick),
                                         rewards, done, step_info.get("outcome"), self.time_limit_ticks),
                                     # the shadow AI's raw orders, exact positions included: the labels
                                     # above keep cells only, fine positions are derived from these
                                     **({"shadow": step_info["shadow"]} if "shadow" in step_info else {})})
            if done:
                self.recorder.end(step_info.get("episode"))
        step_info["sent"] = len(sent)
        t_obs = time.monotonic()
        observation = self.observer.observe(self.obs)
        # where a decision's time went on the game's side (fa_train.tools_trainer logs it)
        step_info["bridge_ms"] = 1000.0 * getattr(self, "bridge_s", 0.0)
        step_info["bridge_trips"] = getattr(self, "bridge_trips", 1)
        step_info["observe_ms"] = 1000.0 * (time.monotonic() - t_obs)
        timing = step_info.get("timing") or {}
        step_info["sim_ms"] = float(timing.get("simMs") or 0.0)
        return observation, rewards, done, step_info

    def _refresh_reclaim(self, queries, answers, sent=(), results=()):
        props = reclaim_after_step(self.observer.wrecks, queries, answers, sent, results)
        self.observer.set_wrecks(props)
        self.reclaim_cache = sorted((r for r in props if r.get("mass", 0) >= 8),
                                    key=lambda r: -r["mass"])[:40]

    def _note_economy_snapshot(self, done: bool) -> None:
        """Measure completed economy at fixed early times, before defeat empties army economy data."""
        tick = int(self.obs.get("tick") or 0)
        if done and self.state.acu is None:
            return
        for minute in (3, 5, 10):
            if tick < minute * 600 or any(s["minute"] == minute for s in self.economy_snapshots):
                continue
            # A step advances two seconds: record its actual tick, never imply an earlier observation.
            complete, building = {}, {}
            for unit in self.own_units().values():
                bp = unit.get("bp")
                counts = complete if float(unit.get("frac", 1.0)) >= 1.0 else building
                counts[bp] = counts.get(bp, 0) + 1
            roles = {}
            for bp, count in complete.items():
                role = self.kit.role.get(bp, "other")
                roles[role] = roles.get(role, 0) + count
            self.economy_snapshots.append({"minute": minute, "tick": tick,
                                           "mass_income": self._mass_income(self.obs),
                                           "completed_roles": roles, "completed_blueprints": complete,
                                           "building_blueprints": building})

    def _note_recent_state(self) -> None:
        """Keep roughly the last minute of commander and completed-unit activity for loss diagnosis."""
        roles = {}
        for unit in self.own_units().values():
            if float(unit.get("frac", 1.0)) < 1.0:
                continue
            role = self.kit.role.get(unit.get("bp"), "other")
            counts = roles.setdefault(role, {"units": 0, "idle": 0, "orders": {}})
            counts["units"] += 1
            counts["idle"] += int(bool(unit.get("idle")))
            order = str(unit.get("order", 0))
            counts["orders"][order] = counts["orders"].get(order, 0) + 1
        acu = self.state.acu
        self.recent_state_snapshots.append({"tick": int(self.obs.get("tick") or 0),
                                            "mass_income": self._mass_income(self.obs),
                                            "commander": {key: acu.get(key) for key in
                                                          ("id", "x", "z", "hp", "maxhp", "idle", "order", "dest")}
                                                         if acu else None,
                                            "completed_role_activity": roles})

    # ------------------------------------------------------------------ the teacher, without acting
    def teacher_commands(self, policy, dry: bool = True) -> list[dict]:
        """The rule teacher's commands for the current state. With `dry` (labels while the student acts) the
        skills' bookkeeping is restored afterwards, so nothing changes but the policy's own memory; without it
        (the teacher acting) the bookkeeping advances as in its own games. Builds away from deposits carry no
        cell (finding the exact spot needs a game query): cloning scores their verb, builder and kind only."""
        saved = (copy.copy(self.claimed), self.army_order, set(self.ordered_units), dict(self.visited),
                 set(self.rallied), list(self.reclaim_cache), self.army_target, self.army_order_since)
        try:
            if hasattr(policy, "commands"):            # a teacher that speaks commands (TeacherV3)
                commands = policy.commands(self)
            else:
                commands = self.macro_commands(policy.act(self))
        finally:
            if dry:
                (self.claimed, self.army_order, self.ordered_units, self.visited, self.rallied,
                 self.reclaim_cache, self.army_target, self.army_order_since) = saved
        if dry:
            commands = self._orders_not_in_effect(commands)
        return commands

    def macro_commands(self, choice: dict) -> list[dict]:
        """One option per menu head (FactoredEnv's skills) as commands. Updates the skills' bookkeeping."""
        _, actions, _, pending = self.plan_macro(choice)
        commands = self.kit.commands_from_actions(actions, self.own_units())
        if pending is not None:
            unit_id, bp_id, centre = pending[:3]
            adjacent_to = pending[4] if len(pending) > 4 else []
            if bp_id in self.bps and adjacent_to:
                commands.insert(0, {"verb": "build", "subject": [str(unit_id)], "bp": bp_id, "place": "next",
                                    "anchor": str(adjacent_to[0])})
            elif bp_id in self.bps:
                commands.insert(0, {"verb": "build", "subject": [str(unit_id)], "bp": bp_id, "place": "cell",
                                    "cell": None, "spot": None, "aim": tuple(centre)})
        return commands

    def _orders_not_in_effect(self, commands: list[dict]) -> list[dict]:
        """The labels of the student's games keep only orders that would change something. The teacher's own
        bookkeeping (which factories it rallied, which units have the army's order) is restored after every dry
        call, so without this it re-issues every rally point and every army order every decision -- 686 rallies
        in one 12-minute game where its own games send a handful -- and distillation taught command spam."""
        units = self.own_units()
        reach = 1.5 * max(self.kit.cell_w, self.kit.cell_h)
        out = []
        for c in commands:
            verb = c["verb"]
            if verb == "rally" and all(str(u) in self.rallied_by_player for u in c.get("subject") or []):
                continue
            if verb in ("attack_move", "move") and c.get("cell") is not None:
                tx, tz = self.kit.where(c["cell"], c.get("sub"))
                near = 1.5 * self.kit.reach_of(c) if c.get("sub") is not None else reach
                keep = [u for u in c.get("subject") or [] if not self._heading_to(units.get(str(u)), tx, tz, near)]
                if not keep:
                    continue
                c = dict(c, subject=keep)
            out.append(c)
        return out

    @staticmethod
    def _heading_to(unit: dict | None, x: float, z: float, reach: float) -> bool:
        """Whether a unit is at (x, z) or already moving there."""
        if unit is None:
            return False
        if math.hypot(unit["x"] - x, unit["z"] - z) <= reach:
            return True
        dest = unit.get("dest")
        return unit.get("order") in MOVING_ORDERS and dest is not None and \
            math.hypot(dest[0] - x, dest[1] - z) <= reach

    def free_cell_near(self, x: float, z: float, bp_id: str) -> tuple[int, int]:
        """The cell nearest to (x, z) where blueprint `bp_id` may be built and no own structure stands yet -- how
        a player picks an open spot near where they want a building."""
        import numpy as np
        from .tools import placement_class
        from .tools_obs import MAP_CHANNELS
        obs = self.observer.observe(self.obs)
        cls = placement_class(self.bps[bp_id])
        allowed = obs["mask_build_cell"][cls] & (obs["map"][MAP_CHANNELS.index("own_structures")] == 0)
        cx, cz = self.kit.cell(x, z)
        ys, xs = np.nonzero(allowed)
        if len(xs) == 0:
            return (cx, cz)
        i = int(np.argmin((xs - cx) ** 2 + (ys - cz) ** 2))
        return (int(xs[i]), int(ys[i]))
