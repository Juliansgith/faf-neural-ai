"""FactoredEnv: the macro action space of docs/training-design.md (v1) over FAEnv.

Every decision (default 20 ticks = 2 s) the policy picks one option per head; scripted skills turn the
choices into bridge actions. Options that cannot act right now are masked.

    env = FactoredEnv(GameConfig(map="SCMP_016", ai="easy"), time_limit_minutes=20)
    obs, info = env.reset()
    obs, rewards, done, info = env.step({"build": 1, "produce": 0, "army": 0, "commander": 0, "tech": 0})

`obs` is a dict of numpy arrays: actor inputs (`scalars`, `units`, `unit_bp`, `unit_group`,
`unit_mask`), critic-only inputs (`privileged`) and one boolean mask per head (`mask_<head>`).
`rewards` holds the reward components separately (terminal, trade, economy); the trainer combines them
(the shaping weight is annealed there). `env.state` is the decoded game state the skills and the rule
policy work from.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bridge import GameConfig
from .env import FAEnv, army, outcome_of

HEADS: dict[str, tuple[str, ...]] = {
    "build": ("none", "mex", "power", "land_factory", "air_factory", "radar", "point_defense", "aa_defense",
              "hydro", "reclaim", "assist", "mass_storage"),
    "produce": ("none", "engineer", "tank", "artillery", "mobile_aa", "land_scout", "interceptor", "bomber",
                "air_scout"),
    "army": ("none", "attack_nearest", "attack_base", "raid", "defend", "retreat", "hold", "hunt"),
    "commander": ("none", "home", "join_army"),
    "tech": ("none", "upgrade_mex", "upgrade_factory"),
}
HEAD_NAMES = tuple(HEADS)
BUILD_ORDER = 8          # EUnitCommandType BuildMobile, as the bridge reports a unit's current order
ACU_DEPOSIT_RANGE = 35.0  # the commander takes deposits this near its start; engineers take the rest

# Caps that keep early, random-ish policies from absurd spam; generous enough not to bind good play.
CAPS = {"land_factory": 8, "air_factory": 4, "radar": 1, "point_defense": 8, "aa_defense": 6}

GROUPS = ("own", "seen", "ghost", "radar")
MAX_UNITS = 256
UNIT_FEATURES = (
    "x", "z", "dx_home", "dz_home", "hp_frac", "build_frac", "idle", "work", "queue",
    "log_mass", "log_health", "dps", "range", "speed", "tech",
    "is_engineer", "is_factory", "is_mex", "is_power", "is_structure", "is_air", "is_naval", "is_land",
    "is_command", "is_defense", "is_scout",
)
PRIVILEGED_FEATURES = (
    "mass_income", "energy_income", "army_mass", "structure_mass", "engineer_mass", "combat", "engineers",
    "factories", "mexes", "commander_health",
)


def cats(bp: dict) -> set:
    return set(bp.get("categories") or [])


def has(bp: dict, *wanted: str, without: tuple[str, ...] = ()) -> bool:
    c = cats(bp)
    return all(w in c for w in wanted) and not any(w in c for w in without)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def weapon_dps(bp: dict) -> float:
    total = 0.0
    for w in bp.get("weapons") or []:
        damage, rof = w.get("damage") or 0.0, w.get("rof") or 0.0
        total += float(damage) * float(rof)
    return total


def weapon_range(bp: dict) -> float:
    return max([float(w.get("range") or 0.0) for w in bp.get("weapons") or []] or [0.0])


@dataclass
class State:
    """The decoded game state for skills, masks and the rule policy."""
    tick: int = 0
    me: int = 1
    eco: dict = field(default_factory=dict)
    unit_count: float = 0.0
    unit_cap: float = 1.0
    own: dict = field(default_factory=dict)          # role -> list of unit dicts
    enemies_seen: list = field(default_factory=list)
    ghosts: list = field(default_factory=list)
    radar: list = field(default_factory=list)
    free_mass: list = field(default_factory=list)     # (x, z) deposits we may claim
    free_hydro: list = field(default_factory=list)
    idle_builders: list = field(default_factory=list)
    idle_land_factories: list = field(default_factory=list)
    idle_air_factories: list = field(default_factory=list)
    army: list = field(default_factory=list)         # complete combat units
    acu: dict | None = None


class FactoredEnv:
    def __init__(self, config: GameConfig | None = None, decision_ticks: int = 20, time_limit_minutes: float = 20.0,
                 max_units: int = MAX_UNITS, reclaim_scan_every: int = 15, **fa_kwargs):
        self.config = config or GameConfig()
        self.decision_ticks = decision_ticks
        self.time_limit_ticks = int(time_limit_minutes * 600)
        self.max_units = max_units
        self.reclaim_scan_every = reclaim_scan_every
        self.fa = FAEnv(self.config, step_ticks=decision_ticks, max_ticks=self.time_limit_ticks, privileged=True,
                        **fa_kwargs)
        self.state = State()
        self.recorder = None      # an fa_env.record.Recorder saves the raw game when set

    # ------------------------------------------------------------------ episode
    def reset(self, config: GameConfig | None = None):
        if config is not None:
            self.config = config
            self.fa.config = config
        obs, info = self.fa.reset()
        return self._begin(obs, info)

    def _begin(self, obs: dict, info: dict):
        """An episode's bookkeeping from the game's first observation and hello (reset; a self-play side)."""
        self.info = info
        self.bps = {bp["id"]: bp for bp in info["blueprints"]}
        self._init_blueprint_tables()
        self.me = obs["me"]
        armies = {a["index"]: a for a in info["armies"]}
        self.start = tuple(armies[self.me]["start"])
        enemies = [a for a in info["armies"] if a["relation"] == "enemy" and not a["civilian"]]
        self.enemy_start = tuple(enemies[0]["start"]) if enemies else self.start
        self.enemy_indices = [a["index"] for a in enemies]
        m = info["map"]
        self.width, self.height = float(m["width"]), float(m["height"])
        self.diag = math.hypot(self.width, self.height)
        self.front = self._toward_enemy(0.22, cap=60.0)
        self.deposits = [(d["x"], d["z"], d["type"]) for d in m["deposits"]]
        self.claimed: dict[tuple, int] = {}
        self.reclaim_cache: list[dict] = []
        self.decisions = 0
        self.last_choice = {h: 0 for h in HEAD_NAMES}
        self.last_ok = {h: 1.0 for h in HEAD_NAMES}
        self.army_order: tuple | None = None
        self.army_target: tuple | None = None       # where the current attack order leads, and since when
        self.army_order_since = 0
        self.ordered_units: set = set()
        self.visited: dict = {}
        self.rallied: set = set()
        self.chosen = None
        self.obs = obs
        self.prev_trade = self._trade(obs)
        self.prev_income = self._mass_income(obs)
        self._decode(obs)
        self.prev_acu_health = self._acu_health()
        if self.recorder is not None:
            c = self.config
            self.recorder.start(info, {"map": c.map, "faction": c.faction, "ai": c.ai, "ai_faction": c.ai_faction,
                                       "seed": c.seed, "victory": c.victory, "decision_ticks": self.decision_ticks,
                                       "time_limit_ticks": self.time_limit_ticks, "me": self.me})
        return self.encode(), {"tick": obs.get("tick", 0), "load_seconds": info.get("load_seconds", 0.0)}

    def close(self):
        if self.recorder is not None:
            self.recorder.close()           # a game that ended early keeps what it recorded
        self.fa.close()

    # ------------------------------------------------------------------ blueprints
    def _init_blueprint_tables(self):
        ids = sorted(self.bps)
        self.bp_index = {bp_id: i + 1 for i, bp_id in enumerate(ids)}   # 0 = padding / unknown
        self.vocab_size = len(ids) + 1
        self.bp_static: dict[str, np.ndarray] = {}
        for bp_id, bp in self.bps.items():
            self.bp_static[bp_id] = np.array([
                math.log1p(float(bp.get("mass") or 0.0)) / 10.0,
                math.log1p(float(bp.get("health") or 0.0)) / 12.0,
                min(weapon_dps(bp), 2000.0) / 200.0,
                weapon_range(bp) / 100.0,
                float(bp.get("speed") or 0.0) / 10.0,
                float(bp.get("tech") or 0) / 4.0,
                has(bp, "ENGINEER"), has(bp, "FACTORY", "STRUCTURE"), has(bp, "MASSEXTRACTION"),
                has(bp, "ENERGYPRODUCTION", "STRUCTURE"), has(bp, "STRUCTURE"), has(bp, "AIR"), has(bp, "NAVAL"),
                has(bp, "LAND"), has(bp, "COMMAND"), has(bp, "DEFENSE"), has(bp, "SCOUT"),
            ], np.float32)
        self.n_static = len(next(iter(self.bp_static.values())))

    def _resolve_choices(self, acu_bp: str) -> dict:
        """Tech-1 blueprints of the commander's faction for every skill."""
        bps = self.bps

        def pick(pool, *wanted, without=(), key=None):
            found = [bps[i] for i in pool if i in bps and has(bps[i], *wanted, without=without)]
            if key:
                found.sort(key=key, reverse=True)
            return found[0]["id"] if found else None

        acu_can = bps[acu_bp].get("buildable") or []
        c = {
            "mex": pick(acu_can, "MASSEXTRACTION", "TECH1", "STRUCTURE"),
            "power": pick(acu_can, "ENERGYPRODUCTION", "TECH1", "STRUCTURE", without=("HYDROCARBON",)),
            "hydro": pick(acu_can, "HYDROCARBON", "STRUCTURE"),
            "land_factory": pick(acu_can, "FACTORY", "LAND", "TECH1", "STRUCTURE", without=("SUPPORTFACTORY",)),
            "air_factory": pick(acu_can, "FACTORY", "AIR", "TECH1", "STRUCTURE", without=("SUPPORTFACTORY",)),
            "radar": pick(acu_can, "RADAR", "STRUCTURE", "TECH1"),
            "point_defense": pick(acu_can, "DEFENSE", "DIRECTFIRE", "STRUCTURE", "TECH1"),
            "aa_defense": pick(acu_can, "DEFENSE", "ANTIAIR", "STRUCTURE", "TECH1"),
            "mass_storage": pick(acu_can, "MASSSTORAGE", "STRUCTURE", "TECH1"),
        }
        land = bps.get(c["land_factory"] or "", {}).get("buildable") or []
        air = bps.get(c["air_factory"] or "", {}).get("buildable") or []
        health = lambda bp: bp.get("health") or 0  # noqa: E731
        c.update({
            "engineer": pick(land, "ENGINEER", "TECH1", "MOBILE", "LAND"),
            "tank": pick(land, "DIRECTFIRE", "TECH1", "MOBILE", "LAND", without=("ENGINEER", "SCOUT"), key=health),
            "artillery": pick(land, "INDIRECTFIRE", "TECH1", "MOBILE", "LAND", without=("ENGINEER",)),
            "mobile_aa": pick(land, "ANTIAIR", "TECH1", "MOBILE", "LAND", without=("ENGINEER",)),
            "land_scout": pick(land, "SCOUT", "TECH1", "MOBILE", "LAND"),
            "interceptor": pick(air, "ANTIAIR", "AIR", "TECH1", "MOBILE", without=("SCOUT", "BOMBER")),
            "bomber": pick(air, "BOMBER", "AIR", "TECH1", "MOBILE"),
            "air_scout": pick(air, "SCOUT", "AIR", "TECH1", "MOBILE"),
        })
        return c

    # ------------------------------------------------------------------ decoding
    def _role(self, bp: dict) -> str | None:
        if has(bp, "COMMAND"):
            return "acu"
        if has(bp, "ENGINEER", "MOBILE"):
            return "engineer"
        if has(bp, "FACTORY", "STRUCTURE"):
            return "air_factory" if has(bp, "AIR") else "land_factory"
        if has(bp, "MASSEXTRACTION"):
            return "mex"
        if has(bp, "MASSSTORAGE", "STRUCTURE"):
            return "mass_storage"
        if has(bp, "HYDROCARBON"):
            return "hydro"
        if has(bp, "ENERGYPRODUCTION", "STRUCTURE"):
            return "power"
        if has(bp, "RADAR", "STRUCTURE"):
            return "radar"
        if has(bp, "DEFENSE", "STRUCTURE"):
            return "aa_defense" if has(bp, "ANTIAIR") else "point_defense"
        if has(bp, "SCOUT", "MOBILE"):
            return "scout"
        if has(bp, "MOBILE") and bp.get("weapons"):
            if has(bp, "AIR"):
                return "bomber" if has(bp, "BOMBER") else "interceptor"
            if has(bp, "INDIRECTFIRE"):
                return "artillery"
            if has(bp, "ANTIAIR"):
                return "mobile_aa"
            return "tank"
        return "other"

    def _decode(self, obs: dict) -> None:
        s = State(tick=obs.get("tick", 0), me=self.me)
        me = army(obs, self.me) or {}
        s.eco = me.get("eco", {})
        s.unit_count = float(me.get("unitCount") or 0.0)
        s.unit_cap = float(me.get("unitCap") or 1.0)
        own: dict[str, list] = {}
        for u in obs.get("units", []):
            if u.get("army") == self.me:
                role = self._role(self.bps.get(u.get("bp"), {}))
                own.setdefault(role, []).append(u)
            elif u.get("radar"):
                s.radar.append(u)
            elif u.get("ghost"):
                s.ghosts.append(u)
            elif self._relation(obs, u.get("army")) == "enemy":
                s.enemies_seen.append(u)
        s.own = own
        s.acu = (own.get("acu") or [None])[0]
        if s.acu and self.chosen is None:
            self.chosen = self._resolve_choices(s.acu["bp"])
        complete = lambda u: u.get("frac", 1.0) >= 1.0  # noqa: E731
        s.idle_builders = [u for u in own.get("engineer", []) if u.get("idle") and complete(u)]
        if s.acu and s.acu.get("idle"):
            s.idle_builders.append(s.acu)
        s.idle_land_factories = [u for u in own.get("land_factory", []) if u.get("idle") and complete(u)]
        s.idle_air_factories = [u for u in own.get("air_factory", []) if u.get("idle") and complete(u)]
        s.army = [u for role in ("tank", "artillery", "mobile_aa", "interceptor", "bomber")
                  for u in own.get(role, []) if complete(u)]
        # Extractors snap to the build grid, up to ~1 unit from the deposit marker: match by distance, not by
        # rounded coordinates (which sent engineers to build on spots that already had our extractor).
        occupied = [(u["x"], u["z"]) for role in ("mex", "hydro") for u in own.get(role, [])]
        occupied += [(u["x"], u["z"]) for u in s.enemies_seen + s.ghosts
                     if self._role(self.bps.get(u.get("bp"), {})) in ("mex", "hydro")]
        for x, z, kind in self.deposits:
            key = (round(x), round(z))
            if any(abs(ox - x) <= 2.5 and abs(oz - z) <= 2.5 for ox, oz in occupied):
                continue
            if s.tick - self.claimed.get(key, -10 ** 9) < 900:
                continue
            if dist((x, z), self.start) > 1.15 * dist((x, z), self.enemy_start):
                continue   # deep in the enemy half
            (s.free_mass if kind == "mass" else s.free_hydro).append((x, z))
        self.state = s

    @staticmethod
    def _relation(obs, index):
        a = army(obs, index)
        return a["relation"] if a else "enemy"

    def _toward_enemy(self, fraction: float, cap: float | None = None) -> tuple[float, float]:
        """The point `fraction` of the way to the enemy's start, at most `cap` units out (fractions were tuned
        on a 5 km map; on 20 km, 22 % put the defence line 230 units from the base)."""
        d = dist(self.start, self.enemy_start)
        if cap is not None and d > 0:
            fraction = min(fraction, cap / d)
        return (self.start[0] + fraction * (self.enemy_start[0] - self.start[0]),
                self.start[1] + fraction * (self.enemy_start[1] - self.start[1]))

    # ------------------------------------------------------------------ masks
    def masks(self) -> dict[str, np.ndarray]:
        s, c = self.state, self.chosen or {}
        own = s.own
        count = lambda role: len(own.get(role, []))  # noqa: E731
        cap_ok = s.unit_count < s.unit_cap - 1
        m = {h: np.zeros(len(opts), bool) for h, opts in HEADS.items()}
        for h in m:
            m[h][0] = True
        builders = bool(s.idle_builders)
        b = m["build"]
        b[1] = bool(c.get("mex")) and self._deposit_builders(s.free_mass)
        b[2] = builders and bool(c.get("power"))
        b[3] = builders and bool(c.get("land_factory")) and count("land_factory") < CAPS["land_factory"]
        b[4] = builders and bool(c.get("air_factory")) and count("air_factory") < CAPS["air_factory"]
        b[5] = builders and bool(c.get("radar")) and count("radar") < CAPS["radar"]
        b[6] = builders and bool(c.get("point_defense")) and count("point_defense") < CAPS["point_defense"]
        b[7] = builders and bool(c.get("aa_defense")) and count("aa_defense") < CAPS["aa_defense"]
        b[8] = bool(c.get("hydro")) and self._deposit_builders(s.free_hydro)
        b[9] = any(u is not s.acu for u in s.idle_builders) and bool(self.reclaim_cache)
        b[10] = any(u is not s.acu for u in s.idle_builders) and bool(own.get("land_factory") or own.get("air_factory"))
        b[11] = builders and bool(c.get("mass_storage")) and count("mex") > 0 and \
            count("mass_storage") < 4 * count("mex")
        p = m["produce"]
        land, air = bool(s.idle_land_factories) and cap_ok, bool(s.idle_air_factories) and cap_ok
        for i, key in enumerate(HEADS["produce"][1:], start=1):
            is_air = key in ("interceptor", "bomber", "air_scout")
            p[i] = (air if is_air else land) and bool(c.get(key))
        a = m["army"]
        have_army = bool(s.army)
        known = bool(s.enemies_seen or s.ghosts or s.radar)
        known_mex = any(has(self.bps.get(u.get("bp"), {}), "MASSEXTRACTION") for u in s.enemies_seen + s.ghosts)
        a[1] = have_army and known
        a[2] = a[4] = a[5] = a[6] = a[7] = have_army
        a[3] = have_army and known_mex
        cm = m["commander"]
        if s.acu:
            cm[1] = dist((s.acu["x"], s.acu["z"]), self.start) > 15
            cm[2] = have_army
        t = m["tech"]
        t[1] = any(self._upgrade_of(u) for u in own.get("mex", []) if u.get("idle") and u.get("frac", 1) >= 1)
        t[2] = any(self._upgrade_of(u) for u in s.idle_land_factories + s.idle_air_factories)
        return m

    def _deposit_builders(self, pool) -> bool:
        """Whether an idle builder can take a deposit of `pool`: an engineer any, the commander only one near
        the base. Sending the commander to every free deposit kept it walking out and home again on maps whose
        deposits are not next to the start, and it never built a factory (Theta Passage)."""
        s = self.state
        if any(u is not s.acu for u in s.idle_builders):
            return bool(pool)
        return s.acu in s.idle_builders and any(dist(d, self.start) <= ACU_DEPOSIT_RANGE for d in pool)

    def _upgrade_of(self, unit: dict) -> str | None:
        target = self.bps.get(unit.get("bp"), {}).get("upgradesTo")
        return target if target in self.bps else None

    # ------------------------------------------------------------------ step
    def step(self, choice: dict[str, int]):
        """Apply one option per head, advance one decision interval; returns obs, rewards, done, info."""
        choice, actions, queries, pending_site = self.plan_macro(choice)
        self.state_tick_before = self.state.tick
        return self._send(choice, actions, queries, pending_site)

    def plan_macro(self, choice: dict[str, int]):
        """The bridge actions (and queries) the skills make of one option per head, without sending them.
        Updates the skills' bookkeeping (claimed deposits, army order, rallied factories, ...)."""
        masks = self.masks()
        choice = dict(choice)
        for h in HEAD_NAMES:
            i = int(choice.get(h, 0))
            if not (0 <= i < len(HEADS[h])) or not masks[h][i]:
                i = 0                                        # a masked option is treated as none
            choice[h] = i
        s = self.state
        actions: list[dict] = []
        queries: list[dict] = []
        pending_site: tuple | None = None

        # commander first, so a moving commander is not also handed a build order
        acu_busy = False
        cmd = HEADS["commander"][choice["commander"]]
        if s.acu:
            hurt = s.acu.get("maxhp") and s.acu.get("hp", 0) / s.acu["maxhp"] < 0.4
            if hurt and dist((s.acu["x"], s.acu["z"]), self.start) > 15:
                cmd = "home"                                  # scripted safety rule
            elif cmd == "home" and s.acu.get("order") == BUILD_ORDER:
                # on its way to a build (an extractor 20 units out): orders replace what a unit does, so "home"
                # would cancel it, and the next build send it out again -- on every map but Canis River, whose
                # first extractors lie within 15 units of the start, the commander built almost nothing
                cmd = "none"
            if cmd == "home":
                actions.append({"type": "move", "units": [s.acu["id"]], "pos": list(self.start)})
                acu_busy = True
            elif cmd == "join_army" and s.army:
                cx, cz = self._centroid(s.army)
                actions.append({"type": "attack_move", "units": [s.acu["id"]], "pos": [cx, cz]})
                acu_busy = True
        builders = [u for u in s.idle_builders if not (acu_busy and u is s.acu)]

        build = HEADS["build"][choice["build"]]
        if build != "none" and builders:
            actions_b, site = self._skill_build(build, builders)
            actions += actions_b
            pending_site = site
        produce = HEADS["produce"][choice["produce"]]
        if produce != "none":
            actions += self._skill_produce(produce)
        army_choice = HEADS["army"][choice["army"]]
        if army_choice != "none":
            actions += self._skill_army(army_choice)
        tech = HEADS["tech"][choice["tech"]]
        if tech != "none":
            actions += self._skill_tech(tech)
        actions += self._rally_new_factories()
        actions += self._scout_patrols()

        if self.decisions % self.reclaim_scan_every == 0:
            queries.append({"type": "reclaim", "pos": list(self.start), "radius": 45})
        if pending_site is not None:
            unit_id, bp_id, centre, radius, adjacent_to, adjacent_only = pending_site
            q = {"type": "find_build_site", "bp": bp_id, "pos": list(centre), "radius": radius}
            if adjacent_to:
                q.update(adjacent_to=[str(i) for i in adjacent_to], adjacent_only=adjacent_only)
            queries.append(q)
        return choice, actions, queries, pending_site

    def _send(self, choice, actions, queries, pending_site):
        wait = self.decision_ticks
        raw = self.fa.server.last_line if self.fa.server is not None else None   # the message decided on
        sent: list[dict] = []
        answers: list = []
        if queries:
            sent += actions
            obs, _, terminated, truncated, info = self.fa.step(actions, queries, wait=1)
            wait -= 1
            actions = []
            answers = info.get("queries") or []
            for q, ans in zip(queries, answers):
                if q["type"] == "reclaim" and ans.get("ok"):
                    self.reclaim_cache = sorted((r for r in ans.get("value") or [] if r.get("mass", 0) >= 8),
                                                key=lambda r: -r["mass"])[:40]
                elif q["type"] == "find_build_site" and ans.get("ok") and ans.get("value"):
                    unit_id, bp_id = pending_site[0], pending_site[1]
                    actions.append({"type": "build", "units": [unit_id], "bp": bp_id, "pos": ans["value"]})
            if terminated or truncated:
                return self._record(raw, choice, sent, queries, answers,
                                    self._finish(obs, info, terminated, truncated, choice))
        sent += actions
        obs, _, terminated, truncated, info = self.fa.step(actions, wait=max(1, wait))
        return self._record(raw, choice, sent, queries, answers, self._finish(obs, info, terminated, truncated, choice))

    def _record(self, raw, choice, sent, queries, answers, result):
        if self.recorder is not None:
            self.recorder.step(raw, {"tick": self.state_tick_before, "choice": dict(choice), "sent": sent,
                                     "queries": queries, "answers": answers})
            if result[2]:
                self.recorder.end(result[3].get("episode"))
        return result

    def _finish(self, obs, info, terminated, truncated, choice):
        self.decisions += 1
        results = info.get("results") or []
        failed = sum(1 for r in results if not r.get("ok"))
        for h in HEAD_NAMES:
            self.last_choice[h] = choice[h]
            self.last_ok[h] = 0.0 if failed else 1.0
        self.obs = obs
        self._decode(obs)
        tick = obs.get("tick", 0)
        outcome = info.get("outcome")
        time_up = tick >= self.time_limit_ticks
        done = bool(terminated or truncated or time_up)
        trade = self._trade(obs)
        income = self._mass_income(obs)
        # the commander's lost health, as a fraction: losing it is losing the game (demoralization), and a
        # penalty at the moment of damage teaches caution long before the -10 of a dead commander does
        acu_health = self._acu_health()
        lost = 0.0 if acu_health is None or self.prev_acu_health is None else max(0.0, self.prev_acu_health - acu_health)
        rewards = {
            "trade": (trade - self.prev_trade) / 1000.0,
            "economy": (income - self.prev_income) / 20.0,
            "commander": -2.0 * lost,
            "terminal": 0.0,
        }
        self.prev_trade, self.prev_income, self.prev_acu_health = trade, income, acu_health
        if outcome == "victory":
            # a faster win is worth a little more (up to +5, nothing at the time limit): a long game full of
            # kills and economy was worth nearly as much as a win, and a third of the day run's games against
            # medium ended on the clock while ahead. Only wins get it, so losing fast earns nothing.
            rewards["terminal"] = 10.0 + 5.0 * max(0.0, 1.0 - tick / max(1, self.time_limit_ticks))
        elif outcome == "defeat":
            rewards["terminal"] = -10.0
        elif done:
            rewards["terminal"] = 5.0 * self.value_advantage(obs)
            outcome = "timeout"
        step_info = {
            "tick": tick, "outcome": outcome if done else None, "action_errors": failed,
            "errors": info.get("errors") or [], "decisions": self.decisions,
        }
        if done:
            step_info["episode"] = self.episode_summary(obs, outcome)
        return self.encode(), rewards, done, step_info

    # ------------------------------------------------------------------ skills
    def _nearest(self, units, point):
        return min(units, key=lambda u: dist((u["x"], u["z"]), point))

    def _centroid(self, units):
        return (sum(u["x"] for u in units) / len(units), sum(u["z"] for u in units) / len(units))

    def _skill_build(self, build: str, builders: list):
        s, c = self.state, self.chosen or {}
        if build in ("mex", "hydro"):
            pool = s.free_mass if build == "mex" else s.free_hydro
            engineers = [b for b in builders if b is not s.acu]
            if not engineers:                   # the commander builds only near the base (ACU_DEPOSIT_RANGE)
                pool = [d for d in pool if dist(d, self.start) <= ACU_DEPOSIT_RANGE]
                if not pool:
                    return [], None
            else:
                builders = engineers
            target = min(pool, key=lambda d: dist(d, self.start) + 0.5 * min(dist(d, (b["x"], b["z"])) for b in builders))
            unit = self._nearest(builders, target)
            self.claimed[(round(target[0]), round(target[1]))] = s.tick
            return [{"type": "build", "units": [unit["id"]], "bp": c[build], "pos": [target[0], target[1]]}], None
        if build == "reclaim":
            engineers = [u for u in builders if u is not s.acu]
            field_ = self.reclaim_cache.pop(0)
            unit = self._nearest(engineers, (field_["x"], field_["z"]))
            return [{"type": "reclaim", "units": [unit["id"]], "target": field_["id"],
                     "pos": [field_["x"], field_["z"]]}], None
        if build == "assist":
            engineers = [u for u in builders if u is not s.acu]
            factories = s.own.get("land_factory", []) + s.own.get("air_factory", [])
            unit = engineers[0]
            target = self._nearest(factories, (unit["x"], unit["z"]))
            return [{"type": "guard", "units": [u["id"] for u in engineers[:3]], "target": target["id"]}], None
        centre, radius = self.start, 40
        adjacent_to, adjacent_only = [], False
        if build in ("point_defense", "aa_defense"):
            centre, radius = self._toward_enemy(0.12, cap=35.0), 25
        elif build == "power":
            # power against factories when there is room (adjacency lowers what they cost to run)
            factories = s.own.get("land_factory", []) + s.own.get("air_factory", [])
            adjacent_to = [u["id"] for u in sorted(factories, key=lambda u: dist((u["x"], u["z"]), self.start))]
        elif build == "mass_storage":
            # storages around extractors, upgraded ones and those nearest home first (adjacency raises output)
            mexes = [u for u in s.own.get("mex", []) if u.get("frac", 1) >= 1]
            mexes.sort(key=lambda u: (-(self.bps.get(u.get("bp"), {}).get("tech") or 1),
                                      dist((u["x"], u["z"]), self.start)))
            if not mexes:
                return [], None
            adjacent_to, adjacent_only = [u["id"] for u in mexes], True
            centre = (mexes[0]["x"], mexes[0]["z"])
        unit = self._nearest(builders, centre)
        return [], (unit["id"], c[build], centre, radius, adjacent_to, adjacent_only)

    def _skill_produce(self, produce: str) -> list[dict]:
        s, c = self.state, self.chosen or {}
        is_air = produce in ("interceptor", "bomber", "air_scout")
        factories = s.idle_air_factories if is_air else s.idle_land_factories
        if not factories or not c.get(produce):
            return []
        count = 1 if produce in ("engineer", "land_scout", "air_scout") else 2
        return [{"type": "factory_build", "units": [factories[0]["id"]], "bp": c[produce], "count": count}]

    def _skill_army(self, order: str) -> list[dict]:
        s = self.state
        if not s.army:
            return []
        ids = [u["id"] for u in s.army]
        if order == "hold":
            key = ("hold",)
            action = {"type": "stop", "units": ids}
        elif order == "retreat":
            key = ("retreat",)
            action = {"type": "move", "units": ids, "pos": list(self.start)}
        elif order == "defend":
            key = ("defend",)
            action = {"type": "attack_move", "units": ids, "pos": list(self.front)}
        elif order == "attack_base":
            key = ("attack_base",)
            action = {"type": "attack_move", "units": ids, "pos": list(self.enemy_start)}
        elif order == "hunt":
            target = self._hunt_target(self._centroid(s.army))
            key = ("hunt", round(target[0] / 8), round(target[1] / 8))
            action = {"type": "attack_move", "units": ids, "pos": [target[0], target[1]]}
        else:
            centre = self._centroid(s.army)
            pool = s.enemies_seen + s.ghosts + s.radar
            if order == "raid":
                pool = [u for u in s.enemies_seen + s.ghosts
                        if has(self.bps.get(u.get("bp"), {}), "MASSEXTRACTION")]
            if not pool:
                return []
            target = self._nearest(pool, centre)
            key = (order, round(target["x"] / 8), round(target["z"] / 8))
            action = {"type": "attack_move", "units": ids, "pos": [target["x"], target["z"]]}
        # An attack keeps its target until the army gets there or 45 decisions (90 s) pass: orders now replace
        # what units do (a player's click), and re-targeting at every change of the nearest enemy kept the army
        # turning instead of pushing in (draws at +0.7 value against normal, 2026-09-30). Defend and retreat
        # switch at once.
        attack = ("attack_nearest", "raid", "hunt", "attack_base")
        if order in attack and self.army_order and self.army_order[0] in attack and key != self.army_order \
                and self.army_target is not None and dist(self._centroid(s.army), self.army_target) > 20 \
                and self.decisions - self.army_order_since < 45:
            key = self.army_order
            action = dict(action, pos=list(self.army_target))
        # Re-issuing the same order every 2 s would reset every unit's path; send it only to units that
        # have not had it yet, and only once enough of them have gathered to move as a group (a trickle
        # of single units into the enemy is how the old scripted bot lost 3.5 times what it killed).
        if key == self.army_order:
            fresh = [i for i in ids if i not in self.ordered_units]
            if len(fresh) < (1 if order in ("hold", "retreat", "defend") else 5):
                return []
            action = dict(action, units=fresh)
            self.ordered_units.update(fresh)
            return [action]
        self.army_order = key
        self.army_target = tuple(action["pos"]) if "pos" in action else None
        self.army_order_since = self.decisions
        self.ordered_units = set(ids)
        return [action]

    def _hunt_points(self) -> list[tuple[float, float]]:
        """Places an enemy tends to be: its start and the deposits of its half of the map."""
        points = [self.enemy_start]
        for x, z, _ in self.deposits:
            if dist((x, z), self.enemy_start) < dist((x, z), self.start):
                points.append((x, z))
        return points

    def _hunt_target(self, centre) -> tuple[float, float]:
        """The nearest search point the army has not visited in the last three minutes."""
        tick = self.state.tick
        for p in self._hunt_points():
            if dist(p, centre) < 18:
                self.visited[(round(p[0]), round(p[1]))] = tick
        fresh = [p for p in self._hunt_points() if tick - self.visited.get((round(p[0]), round(p[1])), -10 ** 9) > 1800]
        pool = fresh or self._hunt_points()
        return min(pool, key=lambda p: dist(p, centre))

    def _scout_patrols(self) -> list[dict]:
        """Scouts are not the policy's business: idle ones patrol the enemy half's deposits."""
        out = []
        points = sorted(self._hunt_points(), key=lambda p: dist(p, self.start))
        for u in self.state.own.get("scout", []):
            if u.get("idle") and u.get("frac", 1) >= 1:
                route = points[:: max(1, len(points) // 5)][:5] + [self.start]
                out += [{"type": "patrol", "units": [u["id"]], "pos": [x, z]} for x, z in route]
        return out

    def _skill_tech(self, tech: str) -> list[dict]:
        s = self.state
        if tech == "upgrade_mex":
            pool = [u for u in s.own.get("mex", []) if u.get("idle") and u.get("frac", 1) >= 1 and self._upgrade_of(u)]
        else:
            pool = [u for u in s.idle_land_factories + s.idle_air_factories if self._upgrade_of(u)]
        if not pool:
            return []
        unit = self._nearest(pool, self.start)
        return [{"type": "upgrade", "units": [unit["id"]], "bp": self._upgrade_of(unit)}]

    def _rally_new_factories(self) -> list[dict]:
        out = []
        for u in self.state.own.get("land_factory", []) + self.state.own.get("air_factory", []):
            if u["id"] not in self.rallied and u.get("frac", 1) >= 1:
                self.rallied.add(u["id"])
                out.append({"type": "rally", "units": [u["id"]], "pos": list(self._toward_enemy(0.1, cap=30.0))})
        return out

    # ------------------------------------------------------------------ rewards and summaries
    def _trade(self, obs) -> float:
        a = army(obs, self.me) or {}
        st = a.get("stats") or {}
        return float(st.get("Enemies_MassValue_Destroyed") or 0.0) - float(st.get("Units_MassValue_Lost") or 0.0)

    def _acu_health(self) -> float | None:
        acu = self.state.acu
        return (float(acu.get("hp") or 0.0) / acu["maxhp"]) if acu and acu.get("maxhp") else None

    def _mass_income(self, obs) -> float:
        a = army(obs, self.me) or {}
        return float(((a.get("eco") or {}).get("mass") or {}).get("income") or 0.0)   # empty once defeated

    def _value(self, p: dict) -> float:
        return float((p.get("armyMass") or 0) + (p.get("structureMass") or 0) + (p.get("engineerMass") or 0)
                     + 60.0 * (p.get("massIncome") or 0))

    def value_advantage(self, obs) -> float:
        """(own - enemy) total value over their sum, in [-1, 1]; privileged (reward and critic only)."""
        priv = obs.get("privileged") or {}
        mine = self._value(priv.get(str(self.me), {}))
        theirs = sum(self._value(priv.get(str(i), {})) for i in self.enemy_indices)
        return (mine - theirs) / max(1000.0, mine + theirs)

    def episode_summary(self, obs, outcome) -> dict:
        me = army(obs, self.me) or {}
        st, eco = me.get("stats") or {}, me.get("eco") or {}
        return {
            "outcome": outcome, "minutes": round(obs.get("tick", 0) / 600, 2),
            "mass_income": round(float((eco.get("mass") or {}).get("income") or 0.0), 1),
            "mass_destroyed": round(float(st.get("Enemies_MassValue_Destroyed") or 0.0)),
            "mass_lost": round(float(st.get("Units_MassValue_Lost") or 0.0)),
            "value_advantage": round(self.value_advantage(obs), 3),
            "own_units": sum(len(v) for v in self.state.own.values()),
        }

    # ------------------------------------------------------------------ encoding
    def encode(self) -> dict[str, np.ndarray]:
        s, obs = self.state, self.obs
        own = s.own
        n = lambda role: len(own.get(role, []))  # noqa: E731
        mass, energy = s.eco.get("mass", {}), s.eco.get("energy", {})
        acu = s.acu
        acu_hp = (acu.get("hp", 0) / acu["maxhp"]) if acu and acu.get("maxhp") else 0.0
        acu_home = dist((acu["x"], acu["z"]), self.start) / self.diag if acu else 1.0
        army_mass_seen = sum(float(self.bps.get(u.get("bp"), {}).get("mass") or 0.0)
                             for u in s.enemies_seen if self._role(self.bps.get(u.get("bp"), {})) in
                             ("tank", "artillery", "mobile_aa", "interceptor", "bomber"))
        own_army_mass = sum(float(self.bps.get(u.get("bp"), {}).get("mass") or 0.0) for u in s.army)
        centre = self._centroid(s.army) if s.army else self.start
        under_construction = sum(1 for v in own.values() for u in v if u.get("frac", 1) < 1)
        lg = math.log1p
        scalars = [
            s.tick / 18000.0,
            mass.get("stored", 0.0) / 1000.0, (mass.get("max") or 0.0) / 1000.0, mass.get("income", 0.0) / 50.0,
            mass.get("requested", 0.0) / 50.0, mass.get("trend", 0.0) / 20.0,
            energy.get("stored", 0.0) / 10000.0, (energy.get("max") or 0.0) / 10000.0,
            energy.get("income", 0.0) / 500.0, energy.get("requested", 0.0) / 500.0, energy.get("trend", 0.0) / 200.0,
            float((mass.get("stored") or 0.0) < 20 and (mass.get("requested") or 0.0) > (mass.get("income") or 0.0)),
            float((energy.get("stored") or 0.0) < 100
                  and (energy.get("requested") or 0.0) > (energy.get("income") or 0.0)),
            s.unit_count / max(1.0, s.unit_cap),
            float(acu is not None), acu_hp, acu_home,
            lg(n("engineer")), lg(len([u for u in s.idle_builders if u is not acu])), float(bool(acu and acu.get("idle"))),
            lg(n("land_factory")), lg(n("air_factory")), lg(len(s.idle_land_factories)), lg(len(s.idle_air_factories)),
            lg(n("mex")), lg(n("power")), lg(n("hydro")), lg(n("radar")), lg(n("point_defense")), lg(n("aa_defense")),
            lg(n("tank")), lg(n("artillery")), lg(n("mobile_aa")), lg(n("interceptor")), lg(n("bomber")),
            lg(n("scout")), lg(under_construction), lg(len(s.army)), own_army_mass / 2000.0,
            dist(centre, self.enemy_start) / self.diag, dist(centre, self.start) / self.diag,
            lg(len(s.enemies_seen)), lg(len(s.ghosts)), lg(len(s.radar)), army_mass_seen / 2000.0,
            lg(len([u for u in s.enemies_seen if dist((u["x"], u["z"]), self.start) < 60])),
            lg(len(s.free_mass)), lg(len(s.free_hydro)), lg(len(self.reclaim_cache)),
        ]
        for h in HEAD_NAMES:
            onehot = [0.0] * len(HEADS[h])
            onehot[self.last_choice[h]] = 1.0
            scalars += onehot
        scalars += [self.last_ok[h] for h in HEAD_NAMES]

        rows = []
        for group, units in ((0, [u for v in own.values() for u in v]), (1, s.enemies_seen), (2, s.ghosts), (3, s.radar)):
            for u in units:
                rows.append((group, u))
        rows = rows[: self.max_units]
        feats = np.zeros((self.max_units, len(UNIT_FEATURES)), np.float32)
        bp_ids = np.zeros(self.max_units, np.int64)
        groups = np.zeros(self.max_units, np.int64)
        mask = np.zeros(self.max_units, bool)
        zero_static = np.zeros(self.n_static, np.float32)
        for i, (group, u) in enumerate(rows):
            bp_id = u.get("bp")
            static = self.bp_static.get(bp_id, zero_static) if bp_id else zero_static
            maxhp = u.get("maxhp") or 0.0
            feats[i, :9] = (
                u["x"] / self.width, u["z"] / self.height,
                (u["x"] - self.start[0]) / self.width, (u["z"] - self.start[1]) / self.height,
                (u.get("hp", 0.0) / maxhp) if maxhp else 0.0, u.get("frac", 1.0),
                float(bool(u.get("idle"))), float(u.get("work") or 0.0), min(float(u.get("queue") or 0), 20.0) / 20.0,
            )
            feats[i, 9:] = static
            bp_ids[i] = self.bp_index.get(bp_id, 0) if bp_id else 0
            groups[i] = group
            mask[i] = True

        priv = obs.get("privileged") or {}
        p_me = priv.get(str(self.me), {})
        p_enemy: dict = {}
        for idx in self.enemy_indices:
            for k, v in (priv.get(str(idx)) or {}).items():
                p_enemy[k] = p_enemy.get(k, 0.0) + float(v)
        privileged = []
        for p in (p_me, p_enemy):
            privileged += [
                p.get("massIncome", 0.0) / 50.0, p.get("energyIncome", 0.0) / 500.0, p.get("armyMass", 0.0) / 2000.0,
                p.get("structureMass", 0.0) / 5000.0, p.get("engineerMass", 0.0) / 1000.0, lg(p.get("combat", 0.0)),
                lg(p.get("engineers", 0.0)), lg(p.get("factories", 0.0)), lg(p.get("mexes", 0.0)),
                p.get("commanderHealth", 0.0),
            ]
        privileged.append(self.value_advantage(obs))

        # A NaN or inf from the game (a bad stat, a divide in Lua) would poison the network and stop training;
        # the clip is only a safety bound, every feature is scaled to a few units at most.
        clean = lambda a: np.clip(np.nan_to_num(np.asarray(a, np.float32), nan=0.0, posinf=50.0, neginf=-50.0),  # noqa: E731
                                  -50.0, 50.0)
        out = {
            "scalars": clean(scalars),
            "units": clean(feats), "unit_bp": bp_ids, "unit_group": groups, "unit_mask": mask,
            "privileged": clean(privileged),
        }
        for h, m in self.masks().items():
            out[f"mask_{h}"] = m
        return out


def observation_sizes() -> dict[str, int]:
    """Vector sizes the network needs (scalars depend on the head sizes)."""
    n_scalars = 49 + sum(len(v) for v in HEADS.values()) + len(HEADS)
    return {"scalars": n_scalars, "units": len(UNIT_FEATURES), "privileged": 2 * len(PRIVILEGED_FEATURES) + 1,
            "max_units": MAX_UNITS}
