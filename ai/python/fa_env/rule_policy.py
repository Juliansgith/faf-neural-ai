"""RulePolicy: a hand-written macro policy in FactoredEnv's action space.

It is the teacher for behaviour cloning and the baseline the learned policy has to beat, and it answers
the first question of the training design: can this action space win at all (gate 1: >= 90 % of
30-minute games against FAF's easy AI)?

    env, policy = FactoredEnv(config), RulePolicy()
    obs, _ = env.reset(); policy.reset(env)
    while True:
        obs, rewards, done, info = env.step(policy.act(env))

The rules only read `env.state` (the decoded game state) and `env.masks()`, and only choose options
the masks allow, so every choice is also a valid target for the network.
"""
from __future__ import annotations

from .factored import HEADS, dist, has

INDEX = {h: {name: i for i, name in enumerate(opts)} for h, opts in HEADS.items()}
COMBAT = ("tank", "artillery", "mobile_aa", "interceptor", "bomber")


class RulePolicy:
    def reset(self, env) -> None:
        self.last_upgrade_decision = -10 ** 6
        self.base_reached = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pick(masks, head, *names) -> int:
        for name in names:
            i = INDEX[head][name]
            if masks[head][i]:
                return i
        return 0

    @staticmethod
    def _count(s, role) -> int:
        return len(s.own.get(role, []))

    def _enemy_air(self, env) -> bool:
        return any(has(env.bps.get(u.get("bp"), {}), "AIR", "MOBILE") for u in env.state.enemies_seen)

    def _army_size(self, env, army) -> float:
        """How big the army counts as for attack decisions (TeacherV3 counts by mass: tech units are fewer)."""
        return len(army)

    def _threat_near(self, env, point, radius) -> float:
        """Mass value of seen enemy combat units within `radius` of `point`."""
        total = 0.0
        for u in env.state.enemies_seen:
            bp = env.bps.get(u.get("bp"), {})
            if bp.get("weapons") and has(bp, "MOBILE") and dist((u["x"], u["z"]), point) < radius:
                total += float(bp.get("mass") or 0.0)
        return total

    # ------------------------------------------------------------------ policy
    def act(self, env) -> dict[str, int]:
        s, masks = env.state, env.masks()
        minute = s.tick / 600.0
        mass, energy = s.eco.get("mass", {}), s.eco.get("energy", {})
        m_inc, m_req, m_stored = mass.get("income", 0.0), mass.get("requested", 0.0), mass.get("stored", 0.0)
        e_inc, e_req = energy.get("income", 0.0), energy.get("requested", 0.0)
        e_stored, e_max = energy.get("stored", 0.0), (energy.get("max") or 1.0)
        n = lambda role: self._count(s, role)  # noqa: E731
        choice = {h: 0 for h in HEADS}

        # ---- build: energy ahead of spend, mass everywhere reachable, factories by income
        energy_short = n("power") + n("hydro") == 0 or (e_inc < 1.15 * e_req + 15 and e_stored < 0.6 * e_max)
        wanted_factories = 1 + int(m_inc / 7)
        build_order = []
        if energy_short:
            build_order.append("power")
        if n("mex") < 3 + minute:
            build_order.append("mex")
        if n("land_factory") < 1:
            build_order.append("land_factory")
        if minute > 2 and s.free_hydro:
            build_order.append("hydro")
        if n("land_factory") < wanted_factories and m_stored > 120:
            build_order.append("land_factory")
        if minute > 5 and n("radar") == 0:
            build_order.append("radar")
        if minute > 7 and n("air_factory") == 0 and m_inc > 14:
            build_order.append("air_factory")
        if self._enemy_air(env) and n("aa_defense") < 2:
            build_order.append("aa_defense")
        # mass piling up: storages around extractors, whose output adjacency raises
        if minute > 9 and m_stored > 0.7 * (mass.get("max") or 1e9) and n("mass_storage") < 4 * n("mex"):
            build_order.append("mass_storage")
        if m_stored < 150:
            build_order.append("reclaim")
        build_order += ["mex", "power" if e_stored < 0.9 * e_max else "assist", "assist"]
        choice["build"] = self._pick(masks, "build", *build_order)

        # ---- produce: engineers first, then a tank-heavy mix; interceptors when the enemy flies
        target_engineers = min(2 + int(minute * 0.9), 10)
        produce_order = []
        if n("engineer") < target_engineers:
            produce_order.append("engineer")
        if self._count(s, "scout") == 0 and minute > 1.5:
            produce_order.append("land_scout")
        if s.idle_air_factories:
            produce_order += ["interceptor"] if self._enemy_air(env) or n("interceptor") < 2 else ["bomber"]
        # The land type furthest below its share of the army we have now. Decided from visible counts, not
        # a private tally of past orders: a network cloning this policy sees only the game (with a tally
        # it matched just ~55 % of these choices), and replacing losses is the better rule anyway.
        mix = {"tank": 6, "artillery": 2, "mobile_aa": 1 if (self._enemy_air(env) or minute > 8) else 0}
        land_units = sorted((k for k in mix if mix[k]), key=lambda k: n(k) / mix[k])
        produce_order += land_units + ["engineer"]
        choice["produce"] = self._pick(masks, "produce", *produce_order)

        # ---- army: defend the base, otherwise gather and attack as a group
        army = s.army
        army_mass = sum(float(env.bps.get(u.get("bp"), {}).get("mass") or 0.0) for u in army)
        base_threat = self._threat_near(env, env.start, 70)
        attack_size = min(10 + int(minute / 3) * 2, 30)
        if army:
            centre = env._centroid(army)
            local_threat = self._threat_near(env, centre, 40)
            if base_threat > 0:
                choice["army"] = self._pick(masks, "army", "defend")
            elif local_threat > 1.6 * army_mass and dist(centre, env.start) > 40:
                choice["army"] = self._pick(masks, "army", "retreat")
            elif self._army_size(env, army) >= attack_size or (
                    env.army_order and env.army_order[0] in ("attack_nearest", "raid", "attack_base", "hunt")
                    and self._army_size(env, army) >= attack_size // 2):
                if dist(centre, env.enemy_start) < 30:
                    self.base_reached = True
                preferred = ["raid" if minute < 12 else "attack_nearest", "attack_nearest"]
                preferred.append("hunt" if self.base_reached else "attack_base")
                choice["army"] = self._pick(masks, "army", *preferred)
            elif not env.army_order or env.army_order[0] != "defend":
                choice["army"] = self._pick(masks, "army", "defend")

        # ---- commander: stay home and build; help defend when the base is attacked
        if s.acu:
            if base_threat > 0 and army:
                choice["commander"] = self._pick(masks, "commander", "join_army")
            else:
                choice["commander"] = self._pick(masks, "commander", "home")

        # ---- tech: upgrade mexes once the economy is healthy, then one factory
        if minute > 9 and m_stored > 350 and e_inc > 1.2 * e_req and env.decisions - self.last_upgrade_decision > 40:
            choice["tech"] = self._pick(masks, "tech", "upgrade_mex")
            if choice["tech"]:
                self.last_upgrade_decision = env.decisions
        elif minute > 13 and n("land_factory") >= 3 and m_stored > 500:
            choice["tech"] = self._pick(masks, "tech", "upgrade_factory")
        return choice
