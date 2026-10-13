"""Reference agents for the bridge: a scripted build-order bot and a random one.

Neither is meant to play well. ScriptedAgent exercises most of the action API (mobile builds on
deposits and on free sites, factory queues, assist, attack-move) so it doubles as an end-to-end test
and as a baseline to measure learned agents against; RandomAgent is a fuzzer.
"""
from __future__ import annotations

import math
import random

FACTION_NAMES = {1: "UEF", 2: "Aeon", 3: "Cybran", 4: "Seraphim"}


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def has(bp: dict, *wanted: str, without: tuple[str, ...] = ()) -> bool:
    cats = set(bp.get("categories") or [])
    return all(c in cats for c in wanted) and not any(c in cats for c in without)


ROLES = ("acu", "engineer", "factory", "tank", "mex", "pgen")


def classify(units: list[dict], bps: dict) -> dict[str, list[dict]]:
    """Group the agent's own units by role (anything else is left out)."""
    by_role = {role: [] for role in ROLES}
    for u in units:
        bp = bps.get(u.get("bp"), {})
        if has(bp, "COMMAND"):
            by_role["acu"].append(u)
        elif has(bp, "ENGINEER", "MOBILE"):
            by_role["engineer"].append(u)
        elif has(bp, "FACTORY", "STRUCTURE"):
            by_role["factory"].append(u)
        elif has(bp, "MASSEXTRACTION"):
            by_role["mex"].append(u)
        elif has(bp, "ENERGYPRODUCTION", "STRUCTURE"):
            by_role["pgen"].append(u)
        elif has(bp, "MOBILE", "LAND") and bp.get("weapons"):
            by_role["tank"].append(u)
    return by_role


def resolve_blueprints(bps: dict, acu_bp: str) -> dict[str, str | None]:
    """Tech-1 blueprints for the ACU's faction: mex, pgen, land factory, engineer, main tank."""
    def pick(candidates, *wanted, without=()):
        return [bps[i] for i in candidates if i in bps and has(bps[i], *wanted, without=without)]

    acu_can = bps[acu_bp].get("buildable") or []
    mex = pick(acu_can, "MASSEXTRACTION", "TECH1", "STRUCTURE")
    pgen = pick(acu_can, "ENERGYPRODUCTION", "TECH1", "STRUCTURE", without=("HYDROCARBON",))
    factory = pick(acu_can, "FACTORY", "LAND", "TECH1", "STRUCTURE", without=("SUPPORTFACTORY",))
    chosen = {
        "mex": mex[0]["id"] if mex else None,
        "pgen": pgen[0]["id"] if pgen else None,
        "factory": factory[0]["id"] if factory else None,
        "engineer": None,
        "tank": None,
    }
    if factory:
        fac_can = factory[0].get("buildable") or []
        engineers = pick(fac_can, "ENGINEER", "TECH1", "MOBILE", "LAND")
        tanks = pick(fac_can, "DIRECTFIRE", "TECH1", "MOBILE", "LAND", without=("ENGINEER", "SCOUT"))
        tanks.sort(key=lambda bp: bp.get("health") or 0, reverse=True)
        chosen["engineer"] = engineers[0]["id"] if engineers else None
        chosen["tank"] = tanks[0]["id"] if tanks else None
    return chosen


class ScriptedAgent:
    """Opening build order, then a stream of tanks sent at the enemy start position."""

    ATTACK_GROUP = 8

    def reset(self, obs: dict, info: dict) -> None:
        self.info = info
        self.me = obs["me"]
        self.bps = {bp["id"]: bp for bp in info["blueprints"]}
        armies = {a["index"]: a for a in info["armies"]}
        mine = armies[self.me]
        self.start = tuple(mine["start"])
        enemies = [a for a in info["armies"] if a["relation"] == "enemy" and not a["civilian"]]
        self.targets = [tuple(a["start"]) for a in enemies] or [self.start]
        self.faction = FACTION_NAMES.get(mine["faction"], "UEF")
        self.deposits = sorted(
            ((d["x"], d["z"]) for d in info["map"]["deposits"] if d["type"] == "mass"),
            key=lambda p: distance(p, self.start),
        )
        self.claimed: dict[tuple, int] = {}      # deposit -> tick it was ordered
        self.pending_sites: dict[int, tuple] = {}  # query index -> (unit id, bp)
        self.chosen = None                      # role -> blueprint id, resolved from the ACU/factory lists
        self.attack_wave = 0

    # ---------------------------------------------------------------- policy
    def act(self, obs: dict, step_info: dict | None = None):
        actions, queries = [], []
        tick = obs.get("tick", 0)
        units = [u for u in obs.get("units", []) if u.get("army") == self.me and not u.get("radar")]
        by_role = classify(units, self.bps)

        if not by_role["acu"]:
            return actions, queries
        acu = by_role["acu"][0]
        if self.chosen is None:
            self.chosen = resolve_blueprints(self.bps, acu["bp"])

        # answers to last step's site queries -> build orders
        ordered = set()
        if step_info:
            for index, answer in enumerate(step_info.get("queries") or []):
                request = self.pending_sites.get(index)
                if request and answer.get("ok") and answer.get("value"):
                    unit_id, bp_id = request
                    actions.append({"type": "build", "units": [unit_id], "bp": bp_id, "pos": answer["value"]})
                    ordered.add(unit_id)
        self.pending_sites = {}

        me = next((a for a in obs["armies"] if a["index"] == self.me), {})
        energy_income = me.get("eco", {}).get("energy", {}).get("income", 0.0)
        mass_built = {(round(m["x"]), round(m["z"])) for m in by_role["mex"]}

        def free_deposit(near):
            for d in sorted(self.deposits, key=lambda p: distance(p, near)):
                key = (round(d[0]), round(d[1]))
                if key in mass_built:
                    continue
                ordered = self.claimed.get(d)
                if ordered is not None and tick - ordered < 600:
                    continue
                return d
            return None

        def site_query(unit, bp_id, near):
            self.pending_sites[len(queries)] = (unit["id"], bp_id)
            queries.append({"type": "find_build_site", "bp": bp_id, "pos": [near[0], near[1]], "radius": 40})

        def builder_task(unit, is_acu):
            pos = (unit["x"], unit["z"])
            factories = len(by_role["factory"])
            if self.chosen["pgen"] and (energy_income < 25 + 10 * factories or len(by_role["pgen"]) < 2):
                site_query(unit, self.chosen["pgen"], self.start)
                return
            if is_acu and self.chosen["factory"] and factories < 2:
                site_query(unit, self.chosen["factory"], self.start)
                return
            deposit = free_deposit(pos)
            if deposit and self.chosen["mex"] and (not is_acu or distance(deposit, self.start) < 50):
                self.claimed[deposit] = tick
                actions.append({"type": "build", "units": [unit["id"]], "bp": self.chosen["mex"], "pos": list(deposit)})
                return
            if by_role["factory"]:
                actions.append({"type": "guard", "units": [unit["id"]], "target": by_role["factory"][0]["id"]})

        if acu.get("idle") and acu["id"] not in ordered:
            builder_task(acu, True)
        for engineer in by_role["engineer"]:
            if engineer.get("idle") and engineer.get("frac", 1) >= 1 and engineer["id"] not in ordered:
                builder_task(engineer, False)

        for factory in by_role["factory"]:
            if factory.get("idle") and factory.get("frac", 1) >= 1:
                if len(by_role["engineer"]) < 4 and self.chosen.get("engineer"):
                    actions.append({"type": "factory_build", "units": [factory["id"]], "bp": self.chosen["engineer"], "count": 1})
                elif self.chosen.get("tank"):
                    actions.append({"type": "factory_build", "units": [factory["id"]], "bp": self.chosen["tank"], "count": 5})

        idle_tanks = [t["id"] for t in by_role["tank"] if t.get("idle") and t.get("frac", 1) >= 1]
        if len(idle_tanks) >= self.ATTACK_GROUP:
            target = self.targets[self.attack_wave % len(self.targets)]
            self.attack_wave += 1
            actions.append({"type": "attack_move", "units": idle_tanks, "pos": [target[0], target[1]]})
        return actions, queries


class RandomAgent:
    """Random move / attack-move orders for random own units; a fuzzer for the bridge."""

    def __init__(self, seed: int = 0, orders_per_step: int = 3):
        self.rng = random.Random(seed)
        self.orders_per_step = orders_per_step

    def reset(self, obs: dict, info: dict) -> None:
        self.me = obs["me"]
        m = info["map"]
        self.area = m.get("playable") or [0, 0, m["width"], m["height"]]

    def act(self, obs: dict, step_info: dict | None = None):
        own = [u["id"] for u in obs.get("units", []) if u.get("army") == self.me and u.get("frac", 1) >= 1]
        actions = []
        for _ in range(self.orders_per_step if own else 0):
            x = self.rng.uniform(self.area[0], self.area[2])
            z = self.rng.uniform(self.area[1], self.area[3])
            kind = self.rng.choice(["move", "attack_move", "patrol", "stop"])
            chosen = self.rng.sample(own, k=min(len(own), self.rng.randint(1, 4)))
            action = {"type": kind, "units": chosen}
            if kind != "stop":
                action["pos"] = [x, z]
            actions.append(action)
        return actions, []
