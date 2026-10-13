"""TeacherV3: the rule teacher that techs up, in the full-toolbox commands (docs/full-toolbox-design.md, section 4).

The tutorial for the full-toolbox network. It keeps RulePolicy's rules for the economy, the army, the commander,
storages and adjacency (through FactoredEnv's skills) and adds a tech layer that gives v3 commands directly:

- factories: a T2 land headquarters once mass income passes `T2_HQ`, support factories after it, T3 at `T3_HQ`;
  every idle factory produces (engineers of each tech up to a target, then the army mix at the factory's best
  tech: the strongest unit per mass of each role);
- T2/T3 engineers build T2/T3 power when energy runs short, shields over the base, T2 defences;
- extractors upgrade T1 -> T2 -> T3 as mass and energy allow;
- the commander gets its gun upgrade once the economy can pay for it.

Every rule reads only what the network observes (the game state of `env`), and every command goes through the
same executor as the network's, so the recorded commands are labels as they are.

    env = ToolsEnv(config); teacher = TeacherV3()
    obs, info = env.reset(); teacher.reset(env)
    obs, rewards, done, info = env.step_tools(env.teacher_commands(teacher, dry=False))
"""
from __future__ import annotations

import math

from .factored import HEADS, dist
from .rule_policy import RulePolicy
from .tools import available_enhancements, cats, combat_weapons, is_suicide_unit

T2_HQ, T3_HQ = 20.0, 35.0             # mass income for the T2 and T3 land headquarters (at 50, T3 came in 1 of 5
                                      # 30-minute games against the personalities: the upgrade takes minutes)
T2_POWER, T3_POWER = 35.0, 60.0       # mass income from which new power is T2, then T3
T3_MEX_INCOME = 25.0                  # extractors go T2 -> T3 from here (with 400 mass banked)
SHIELD_INCOME = 30.0
GUN = {"UEF": "HeavyAntiMatterCannon", "Aeon": "CrysalisBeam", "Cybran": "CoolingUpgrade", "Seraphim": "RateOfFire"}
ENHANCE_ORDER = 28                    # EUnitCommandType Script: an enhancement in progress
ROLE_MIX = {"tank": 6, "artillery": 2, "mobile_aa": 1}
LATE_ROLE_MIX = {"tank": 5, "artillery": 4, "mobile_aa": 1}      # from minute 15: artillery against defences
ATTACK_ORDER = 10                     # EUnitCommandType Attack
T1_TANK_MASS = 55.0                   # about a T1 tank's cost in every faction
PUSH_SIZE = 40.0                      # T1-tank equivalents: from here the army goes for the base


class TeacherV3(RulePolicy):
    def reset(self, env) -> None:
        super().reset(env)

    # ------------------------------------------------------------------ helpers
    def _bp(self, env, u) -> dict:
        return env.bps.get(u.get("bp"), {})

    def _tech(self, env, bp_id) -> int:
        return int(env.bps.get(bp_id, {}).get("tech") or 1)

    def _is(self, env, bp_id, *wanted, without=()) -> bool:
        c = cats(env.bps.get(bp_id, {}))
        return all(w in c for w in wanted) and not any(w in c for w in without)

    @staticmethod
    def _complete(u) -> bool:
        return (u.get("frac") or 1.0) >= 1.0

    def _options(self, env, unit, *wanted, without=()) -> list[str]:
        return [b for b in unit.get("can_build") or [] if self._is(env, b, *wanted, without=without)]

    def _best_tech(self, env, options: list[str], key=None) -> str | None:
        """The highest-tech option; among those, the best by `key` (default: the cheapest)."""
        if not options:
            return None
        top = max(self._tech(env, b) for b in options)
        pool = [b for b in options if self._tech(env, b) == top]
        return max(pool, key=key) if key else min(pool, key=lambda b: float(env.bps[b].get("mass") or 0))

    def _combat_value(self, env, bp_id) -> float:
        """Strength per mass: sqrt(damage per second x health) / mass (a crude Lanchester value)."""
        bp = env.bps.get(bp_id, {})
        dps = sum(float(w.get("damage") or 0) * float(w.get("rof") or 0) for w in combat_weapons(bp))
        return math.sqrt(max(dps, 0.0) * float(bp.get("health") or 0)) / max(1.0, float(bp.get("mass") or 1))

    def _role_options(self, env, factory, role) -> list[str]:
        mobile_land = ("MOBILE", "LAND")
        not_army = ("ENGINEER", "SCOUT", "COMMAND", "SUBCOMMANDER")
        if role == "tank":
            opts = self._options(env, factory, *mobile_land, "DIRECTFIRE", without=not_army + ("ANTIAIR",))
        elif role == "artillery":
            opts = self._options(env, factory, *mobile_land, "INDIRECTFIRE", without=not_army)
        elif role == "mobile_aa":
            opts = self._options(env, factory, *mobile_land, "ANTIAIR", without=not_army)
        else:
            return []
        return [b for b in opts if not is_suicide_unit(env.bps.get(b, {}))]

    def _army_size(self, env, army) -> float:
        """The army in T1-tank equivalents (by mass): a T2 or T3 army is fewer, heavier units, and counting heads
        held it at home while far ahead (a 30-minute draw against normal with +0.54 value)."""
        return sum(float(self._bp(env, u).get("mass") or 0.0) for u in army) / T1_TANK_MASS

    # ------------------------------------------------------------------ the teacher
    def commands(self, env) -> list[dict]:
        s = env.state
        own = env.own_units()
        units = list(own.values())
        minute = s.tick / 600.0
        mass, energy = s.eco.get("mass", {}), s.eco.get("energy", {})
        m_inc, m_stored = mass.get("income", 0.0), mass.get("stored", 0.0)
        e_inc, e_req = energy.get("income", 0.0), energy.get("requested", 0.0)
        e_stored, e_max = energy.get("stored", 0.0), (energy.get("max") or 1.0)
        energy_ok = e_inc > 1.1 * e_req and e_stored > 0.3 * e_max
        reserved: set = set()
        out: list[dict] = []
        factories = [u for u in units if self._is(env, u["bp"], "FACTORY", "STRUCTURE")]
        land = [u for u in factories if self._is(env, u["bp"], "LAND")]
        hq = lambda tech: [u for u in land if self._complete(u) and self._is(env, u["bp"], "RESEARCH")  # noqa: E731
                           and self._tech(env, u["bp"]) >= tech]
        building_factory = any(not self._complete(u) and self._tech(env, u["bp"]) >= 2 for u in factories)
        base_threat = self._threat_near(env, env.start, 70)

        # ---- the commander's gun, at home and with the economy to pay for it
        acu = s.acu
        if acu and minute >= 8 and m_inc >= 15 and energy_ok and base_threat == 0 and \
                dist((acu["x"], acu["z"]), env.start) < 30 and acu.get("order") != ENHANCE_ORDER:
            gun = GUN.get(self._bp(env, acu).get("faction"))
            if gun and gun in available_enhancements(acu, self._bp(env, acu)):
                out.append({"verb": "enhance", "subject": [str(acu["id"])], "enhancement": gun})
                reserved.add(str(acu["id"]))
        if acu and acu.get("order") == ENHANCE_ORDER:
            reserved.add(str(acu["id"]))                   # no new order while it upgrades

        # ---- factory upgrades: T2 headquarters, support factories, T3
        idle_land = [u for u in land if u.get("idle") and self._complete(u)]
        if not building_factory:
            upgrade = None
            if m_inc >= T3_HQ and not hq(3):
                pool = [(u, self._options(env, u, "FACTORY", "LAND", "RESEARCH", "TECH3")) for u in idle_land]
                upgrade = next(((u, o[0]) for u, o in pool if o), None)
            if upgrade is None and m_inc >= T2_HQ and not hq(2):
                pool = [(u, self._options(env, u, "FACTORY", "LAND", "RESEARCH", "TECH2")) for u in idle_land]
                upgrade = next(((u, o[0]) for u, o in pool if o), None)
            if upgrade is None and hq(2) and m_inc >= T2_HQ + 5 and m_stored > 300:
                # support factories: the lowest-tech factory below the best headquarters
                top = max(self._tech(env, u["bp"]) for u in hq(2))
                pool = sorted(((u, self._options(env, u, "FACTORY", "LAND", "SUPPORTFACTORY")) for u in idle_land
                               if self._tech(env, u["bp"]) < top), key=lambda p: self._tech(env, p[0]["bp"]))
                upgrade = next(((u, self._best_tech(env, o)) for u, o in pool if o), None)
            if upgrade is not None:
                u, bp = upgrade
                out.append({"verb": "upgrade", "subject": [str(u["id"])], "bp": bp})
                reserved.add(str(u["id"]))

        # ---- production: every idle factory
        n = lambda role: self._count(s, role)  # noqa: E731
        engineers_by_tech = {t: sum(1 for u in units if self._is(env, u["bp"], "ENGINEER", "MOBILE", without=("COMMAND",))
                                    and self._tech(env, u["bp"]) == t) for t in (1, 2, 3)}
        want_engineers = {1: min(2 + int(minute * 0.9), 10), 2: 2 + int(m_inc >= 40), 3: 2}
        enemy_air = self._enemy_air(env)
        mix = dict(LATE_ROLE_MIX if minute >= 15 else ROLE_MIX, mobile_aa=1 if (enemy_air or minute > 8) else 0)
        planned: dict[str, int] = {}
        for f in sorted((u for u in factories if u.get("idle") and self._complete(u) and str(u["id"]) not in reserved),
                        key=lambda u: -self._tech(env, u["bp"])):
            ftech = self._tech(env, f["bp"])
            if self._is(env, f["bp"], "AIR"):
                role = "interceptor" if enemy_air or n("interceptor") < 2 else "bomber"
                opts = self._options(env, f, "AIR", "MOBILE", "ANTIAIR", without=("BOMBER", "SCOUT")) \
                    if role == "interceptor" else self._options(env, f, "AIR", "MOBILE", "BOMBER")
                bp = self._best_tech(env, [b for b in opts if self._tech(env, b) == 1])
                if bp:
                    out.append({"verb": "produce", "subject": [str(f["id"])], "bp": bp, "count": 2})
                continue
            bp, count = None, 2
            # engineers: this factory's own tech first (T2/T3 engineers build T2/T3 structures)
            for t in range(ftech, 0, -1):
                if engineers_by_tech[t] + planned.get(f"eng{t}", 0) < want_engineers[t]:
                    opts = [b for b in self._options(env, f, "ENGINEER", "MOBILE", without=("COMMAND", "FIELDENGINEER"))
                            if self._tech(env, b) == t]
                    if opts:
                        bp, count = min(opts, key=lambda b: float(env.bps[b].get("mass") or 0)), 1
                        planned[f"eng{t}"] = planned.get(f"eng{t}", 0) + 1
                        break
            if bp is None and n("scout") == 0 and minute > 1.5 and ftech == 1 and not planned.get("scout"):
                opts = self._options(env, f, "MOBILE", "LAND", "SCOUT")
                if opts:
                    bp, count = opts[0], 1
                    planned["scout"] = 1
            if bp is None:
                # the land role furthest below its share of the army we have now, at this factory's best tech
                roles = sorted((r for r in mix if mix[r]), key=lambda r: (n(r) + planned.get(r, 0)) / mix[r])
                for r in roles:
                    bp = self._best_tech(env, self._role_options(env, f, r), key=lambda b: self._combat_value(env, b))
                    if bp:
                        planned[r] = planned.get(r, 0) + 2
                        break
            if bp:
                out.append({"verb": "produce", "subject": [str(f["id"])], "bp": bp, "count": count})

        # ---- tech builds by T2/T3 engineers: power, shields
        tech_engineers = sorted((u for u in s.idle_builders if u is not acu and self._tech(env, u["bp"]) >= 2
                                 and str(u["id"]) not in reserved), key=lambda u: -self._tech(env, u["bp"]))
        energy_short = e_inc < 1.15 * e_req + 20 and e_stored < 0.6 * e_max
        shields = [u for u in units if self._is(env, u["bp"], "SHIELD", "STRUCTURE")]
        for eng in tech_engineers:
            order = None
            if energy_short and m_inc >= T2_POWER:
                opts = self._options(env, eng, "ENERGYPRODUCTION", "STRUCTURE", without=("HYDROCARBON",))
                opts = [b for b in opts if self._tech(env, b) >= 2 and (m_inc >= T3_POWER or self._tech(env, b) == 2)]
                power = self._best_tech(env, opts)
                if power:
                    order = ("power", power)
                    energy_short = False
            if order is None and m_inc >= SHIELD_INCOME and len(shields) < 1 + int(m_inc >= 60):
                shield = self._best_tech(env, [b for b in self._options(env, eng, "SHIELD", "STRUCTURE")
                                               if self._tech(env, b) == 2])
                if shield:
                    order = ("shield", shield)
                    shields.append({"bp": shield})
            if order is None:
                continue
            kind, bp = order
            anchor = self._anchor(env, factories if kind == "power" else units, kind)
            if anchor is not None:
                out.append({"verb": "build", "subject": [str(eng["id"])], "bp": bp, "place": "next",
                            "anchor": str(anchor["id"])})
            else:
                out.append({"verb": "build", "subject": [str(eng["id"])], "bp": bp, "place": "cell", "cell": None,
                            "spot": None, "aim": tuple(env._toward_enemy(0.05, cap=15.0))})
            reserved.add(str(eng["id"]))

        # ---- extractor upgrades: T1 -> T2 from minute 8 with some mass banked (earlier, they starved the army
        # that has to hold the first attacks); T2 -> T3 once the income carries it. Nearest home first.
        mexes = [u for u in units if self._is(env, u["bp"], "MASSEXTRACTION")]
        upgrading = sum(1 for u in mexes if u.get("order") == 27 or u.get("work"))
        if minute >= 8 and e_inc > e_req and upgrading < 1 + int(m_inc / 20):
            for u in sorted((u for u in mexes if self._complete(u) and u.get("idle") and not u.get("work")),
                            key=lambda u: (self._tech(env, u["bp"]), dist((u["x"], u["z"]), env.start))):
                target = self._bp(env, u).get("upgradesTo")
                if target not in env.bps:
                    continue
                # at least one always upgrading from minute 8, even with nothing banked: production spends all
                # the mass, and waiting for a bank held Cybran at 21 mass/s for a whole game
                if (self._tech(env, target) == 2 and (upgrading == 0 or m_stored >= 200)) or \
                        (m_inc >= T3_MEX_INCOME and m_stored >= 400):
                    out.append({"verb": "upgrade", "subject": [str(u["id"])], "bp": target})
                break

        # ---- the rest: RulePolicy's economy, army and commander through FactoredEnv's skills
        choice = self.act(env)
        # a big army goes for the base (and then hunts): RulePolicy attacks the nearest enemy whenever one is
        # known, so a winning army kept fighting what came at it and never reached the commander (draws on the
        # clock at +0.6 value against normal)
        army_option = HEADS["army"][choice["army"]]
        push_size = max(PUSH_SIZE, 3.0 * minute)
        if army_option in ("attack_nearest", "raid") and minute >= 14 and base_threat == 0 and \
                self._army_size(env, s.army) >= push_size:
            choice["army"] = self._pick(env.masks(), "army", "hunt" if self.base_reached else "attack_base")
        # the enemy commander in sight: the army attacks it (a game against an AI ends with its commander)
        enemy_acus = [u for u in s.enemies_seen if self._is(env, u.get("bp"), "COMMAND")]
        if enemy_acus and s.army and minute >= 10 and base_threat == 0 and \
                self._army_size(env, s.army) >= 0.75 * PUSH_SIZE:
            centre = env._centroid(s.army)
            target = min(enemy_acus, key=lambda u: dist((u["x"], u["z"]), centre))
            fresh = [str(u["id"]) for u in s.army if str(u["id"]) not in reserved
                     and not (u.get("order") == ATTACK_ORDER and str(u.get("focus")) == str(target["id"]))]
            if fresh:
                out.append({"verb": "attack", "subject": fresh, "target": str(target["id"])})
            choice["army"] = 0
        choice["produce"] = 0                        # production is decided above, for every factory
        choice["tech"] = 0                           # so are factory and extractor upgrades
        saved = s.idle_builders
        s.idle_builders = [u for u in saved if str(u["id"]) not in reserved]
        if acu and str(acu["id"]) in reserved:
            choice["commander"] = 0 if base_threat == 0 else choice["commander"]
        try:
            macro = env.macro_commands(choice)
        finally:
            s.idle_builders = saved
        commander_verbs = ("commander_move", "commander_attack_move")      # defending the base comes first
        return out + [self._tech_up(env, c, m_inc) for c in macro
                      if not (set(map(str, c.get("subject") or [])) & reserved and c["verb"] not in commander_verbs)]

    def _anchor(self, env, candidates, kind):
        """What a new power plant or shield goes against: power against factories (adjacency lowers their
        energy cost), a shield against the building nearest home."""
        pool = [u for u in candidates if self._complete(u) and "STRUCTURE" in cats(self._bp(env, u))
                and dist((u["x"], u["z"]), env.start) < 45]
        if kind == "shield":
            pool = [u for u in pool if self._is(env, u["bp"], "FACTORY") or self._is(env, u["bp"], "ENERGYPRODUCTION")]
        return min(pool, key=lambda u: dist((u["x"], u["z"]), env.start)) if pool else None

    def _tech_up(self, env, c: dict, m_inc: float) -> dict:
        """RulePolicy's T1 defences and power, at the best tech the builder has (power by income)."""
        if c["verb"] != "build" or not c.get("bp"):
            return c
        role = env.kit.role.get(c["bp"])
        if role not in ("point_defense", "aa_defense", "power"):
            return c
        builder = env.own_units().get(str(c["subject"][0])) or {}
        opts = [b for b in builder.get("can_build") or [] if env.kit.role.get(b) == role]
        if role == "power":
            opts = [b for b in opts if self._tech(env, b) == 1 or (m_inc >= T2_POWER and self._tech(env, b) == 2)
                    or (m_inc >= T3_POWER and self._tech(env, b) == 3)]
        elif m_inc < 25:
            opts = [b for b in opts if self._tech(env, b) == 1]
        best = self._best_tech(env, opts)
        return dict(c, bp=best) if best else c


__all__ = ["TeacherV3", "HEADS"]
