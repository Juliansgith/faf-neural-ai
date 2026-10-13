"""The full-toolbox action space: every command a player can give, every argument chosen by the policy
(docs/full-toolbox-design.md, section 2; the earlier "kinds" version is docs/tools-design.md).

A command is a dict:

    {"verb": "build", "subject": [builder_id], "bp": "ueb1101", "place": "next", "anchor": factory_id}
    {"verb": "build", "subject": [builder_id], "bp": "ueb3101", "place": "cell", "cell": (gx, gz), "spot": (sx, sz)}
    {"verb": "produce", "subject": [factory_id], "bp": "uel0201", "count": 5}
    {"verb": "enhance", "subject": [commander_id], "enhancement": "HeavyAntiMatterCannon"}
    {"verb": "attack_move", "subject": [id, id, ...], "cell": (gx, gz)}
    {"verb": "attack", "subject": [...], "target": enemy_id}
    {"verb": "fire_state", "subject": [...], "option": 1}

Blueprints are exact ids: a builder builds any blueprint it can build right now (the bridge's `can_build`,
which applies tech, FAF's HQ rule, restrictions and the unit cap), from any tech level.

`ToolKit` (one per game) turns commands into bridge actions -- `plan()` returns actions plus the
find_build_site queries a build needs, `complete()` turns the answers into the build actions -- and turns
recorded bridge actions back into commands (`commands_from_actions`), which is how the teacher's recorded
games become training labels.
"""
from __future__ import annotations

import math
from .command_grouping import group_independent

VERBS = ("end", "build", "produce", "upgrade", "enhance", "move", "attack_move", "patrol", "attack", "assist",
         "repair", "reclaim", "capture", "overcharge", "load", "unload", "tactical", "nuke", "silo_build", "rally",
         "fire_state", "toggle", "pause", "dive", "stop", "commander_move", "commander_attack_move")
COUNTS = (1, 2, 3, 5, 10)
GRID = 32
SPOTS = 4                 # a cell's 4 x 4 spots: placement to a quarter of a cell (2 units on Canis River)
# Fine positions: the same precision on every map. A cell of a 5 km map is 8 game units (~160 m); a larger map's
# cell splits into fine pixels of FINE_UNITS, up to FINE x FINE (an 81 km map: 128-unit cells, 16 x 16 pixels).
# A command's "sub" (fx, fz) is the pixel in its cell; a build's spot is then a spot of that pixel. Without
# "sub" (a 5 km map, or a network without the fine head) positions are the cell's, as before.
FINE = 16
FINE_UNITS = 8.0
PLACES = ("cell", "next")  # where a building goes: a spot in a cell, or against another building (adjacency)
# commands per decision: M28 gives 14 per 2 s on average (31 at the 90th percentile); 8 kept 42 % of its commands,
# and the canonical order put army movement last, so mostly that was lost. 24 keeps 86 % (2026-10-05, 30 games)
MAX_COMMANDS = 24
# reclaim is an area command: the most valuable props within a cell's reach, found when it executes (the
# observation holds 48 props; M28 reclaimed props outside them 78 % of the time)
RECLAIM_TARGETS = 5
# per-verb choices of the option head
FIRE_STATES = ("return_fire", "hold_fire", "ground_fire")
SILO_KINDS = ("tactical", "nuke")
N_TOGGLES = 9             # script bits 0-8: shield, weapon, jamming, intel, production, stealth, generic, special, cloak
N_OPTIONS = max(len(FIRE_STATES), len(SILO_KINDS), N_TOGGLES)
OPTION_VERBS = frozenset({"fire_state", "silo_build", "toggle"})
# where a building can go: land, water, or on a deposit (mass or hydrocarbon)
PLACEMENT_CLASSES = ("land", "water", "mass", "hydro")

# The commander is never part of a set: losing it loses the game, and a small per-unit chance of joining an
# attack order, sampled every decision, adds up. It moves only by its own commands (as players treat it).
SINGLE_SUBJECT = frozenset({"build", "produce", "upgrade", "enhance", "overcharge", "unload", "tactical", "nuke",
                            "silo_build", "rally", "toggle", "pause", "commander_move", "commander_attack_move"})
SET_SUBJECT = frozenset({"move", "attack_move", "patrol", "attack", "assist", "repair", "reclaim", "capture", "load",
                         "fire_state", "dive", "stop"})
CELL_VERBS = frozenset({"build", "move", "attack_move", "patrol", "reclaim", "unload", "nuke", "rally",
                        "commander_move", "commander_attack_move"})
TARGET_ROWS = ("enemy", "own", "damaged", "wreck", "transport")     # wreck: no verb points at it since reclaim
TARGET_VERBS = {"attack": "enemy", "overcharge": "enemy", "tactical": "enemy", "capture": "enemy", "assist": "own",
                "repair": "damaged", "load": "transport"}            # became an area command (cell)
# set verbs the commander may join: base work, where a sampled per-unit choice cannot walk it into a fight
COMMANDER_SET_VERBS = frozenset({"assist", "repair", "stop"})
BLUEPRINT_VERBS = frozenset({"build", "produce", "upgrade"})     # upgrade: when the subject offers a choice
# canonical order of commands within a decision (labels are sorted by it, so every slot is predictable)
VERB_ORDER = {v: i for i, v in enumerate((
    "commander_move", "commander_attack_move", "enhance", "overcharge", "build", "produce", "upgrade", "silo_build",
    "tactical", "nuke", "assist", "repair", "reclaim", "capture", "load", "unload", "attack_move", "move", "attack",
    "patrol", "dive", "fire_state", "toggle", "pause", "rally", "stop"))}
# The engine's Issue* orders queue behind what a unit is doing (a shift-click); a player's plain click replaces it.
# These verbs replace (the executor clears the queue first): an attack-move given to a patrolling unit otherwise
# never runs, patrol being endless. A build replaces too, unless the builder is building (it then queues, so a
# sampled build does not abandon a half-finished structure); produce, upgrade and silo_build add to a factory's
# or silo's queue, as in the game.
REPLACE_VERBS = frozenset({"move", "attack_move", "patrol", "attack", "assist", "repair", "reclaim", "capture", "load",
                           "unload", "overcharge", "tactical", "nuke", "enhance", "commander_move",
                           "commander_attack_move"})
BUILDING_ORDERS = frozenset({8})      # EUnitCommandType BuildMobile, as the bridge reports a unit's current order
BUILD_QUEUE_CAP = 3                  # orders a builder at work may hold before more builds are left out
# which building a new one should touch for its adjacency bonus, best first (by role, see `role_of`)
ANCHOR_PREFERENCE = {"mass_storage": ("mex",), "energy_storage": ("power", "hydro"),
                     "power": ("land_factory", "air_factory", "naval_factory", "point_defense", "aa_defense", "radar")}


def cats(bp: dict) -> set:
    return set(bp.get("categories") or [])


def build_options(unit: dict, blueprints: dict) -> list[str]:
    """Shared observer/planner choices, including near-complete factory queues.

    Native observations and legacy recordings can omit can_build before completion.
    Explicit engine choices, including an empty list, always override the fallback.
    """
    fraction = 1.0 if unit.get("frac") is None else float(unit["frac"])
    bp = blueprints.get(unit.get("bp"), {})
    if fraction < 1 and not (fraction >= .9 and "FACTORY" in cats(bp)):
        return []
    if unit.get("can_build") is not None:
        choices = list(unit["can_build"])
    else:
        tech = float(bp.get("tech") or 1)
        choices = [b for b in bp.get("buildable") or []
                   if float(blueprints.get(b, {}).get("tech") or 1) <= tech]
    return choices if fraction >= 1 else [
        b for b in choices if "STRUCTURE" not in cats(blueprints.get(b, {}))]


NOT_COMBAT_WEAPONS = ("death", "suicide", "dummy", "collision", "overcharge", "teleport")   # by label: not its gun


def combat_weapons(bp: dict) -> list[dict]:
    """The weapons a unit fights with: not its death blast (a commander's, a Fire Beetle's 1100 damage) or
    placeholder weapons, which a damage-per-second figure would count as if they fired every second."""
    return [w for w in bp.get("weapons") or []
            if not any(word in str(w.get("label") or "").lower() for word in NOT_COMBAT_WEAPONS)]


def is_suicide_unit(bp: dict) -> bool:
    return any("suicide" in str(w.get("label") or "").lower() for w in bp.get("weapons") or [])


def role_of(bp: dict) -> str | None:
    """A coarse, faction-independent role (for adjacency preferences, the teacher and statistics; the policy
    itself chooses exact blueprints)."""
    c = cats(bp)
    if "STRUCTURE" in c:
        for cat, role in (("MASSEXTRACTION", "mex"), ("HYDROCARBON", "hydro"), ("MASSSTORAGE", "mass_storage"),
                          ("ENERGYSTORAGE", "energy_storage"), ("MASSFABRICATION", "mass_fabricator"),
                          ("SHIELD", "shield"), ("RADAR", "radar"), ("SONAR", "sonar"), ("WALL", "wall"),
                          ("NUKE", "nuke_launcher"), ("TACTICALMISSILEPLATFORM", "tml")):
            if cat in c:
                return role
        if "ENERGYPRODUCTION" in c:
            return "power"
        if "FACTORY" in c:
            return "air_factory" if "AIR" in c else "naval_factory" if "NAVAL" in c else "land_factory"
        if "DEFENSE" in c and "ANTIAIR" in c:
            return "aa_defense"
        if "DEFENSE" in c and "ANTIMISSILE" in c:
            return "anti_missile"
        if "DEFENSE" in c:
            return "point_defense"
        if "ARTILLERY" in c:
            return "artillery_structure"
        return "structure"
    if "MOBILE" not in c:
        return None
    if "COMMAND" in c:
        return "commander"
    if "SUBCOMMANDER" in c:
        return "support_commander"
    if "ENGINEER" in c:
        return "engineer"
    if "EXPERIMENTAL" in c:
        return "experimental"
    if "AIR" in c:
        for cat, role in (("TRANSPORTATION", "air_transport"), ("SCOUT", "air_scout"), ("BOMBER", "bomber"),
                          ("GROUNDATTACK", "gunship"), ("ANTIAIR", "interceptor")):
            if cat in c:
                return role
        return "air"
    if "NAVAL" in c:
        return "submarine" if "SUBMERSIBLE" in c else "ship"
    if "SCOUT" in c:
        return "land_scout"
    if "INDIRECTFIRE" in c:
        return "artillery"
    if "ANTIAIR" in c:
        return "mobile_aa"
    if "DIRECTFIRE" in c:
        return "tank"
    return "mobile"


def placement_class(bp: dict) -> int:
    """Where a structure of this blueprint goes: index into PLACEMENT_CLASSES."""
    c = cats(bp)
    if "MASSEXTRACTION" in c:
        return 2
    if "HYDROCARBON" in c:
        return 3
    if "NAVAL" in c or "SONAR" in c:
        return 1
    return 0


def skirt_rect(bp: dict, x: float, z: float) -> tuple[float, float, float, float]:
    """The skirt a structure occupies at (x, z): x0, z0, x1, z1 (RUnitBlueprint::GetSkirtRect)."""
    fx, fz = (bp.get("footprint") or [1, 1])[:2]
    sk = bp.get("skirt") or [0, 0, 0, 0]
    xl, zl = math.floor(x - fx * 0.5 + 0.5), math.floor(z - fz * 0.5 + 0.5)
    x0, x1 = (xl + sk[2], xl + sk[2] + sk[0]) if sk[0] else (xl, xl + fx)
    z0, z1 = (zl + sk[3], zl + sk[3] + sk[1]) if sk[1] else (zl, zl + fz)
    return x0, z0, x1, z1


def adjacent(a: tuple, b: tuple) -> bool:
    """The engine's adjacency test (Unit::OverlapsWith): the skirts share an edge (within 1) and one's side
    lies within the other's. Corners do not count."""
    ax0, az0, ax1, az1 = a
    bx0, bz0, bx1, bz1 = b
    if abs(ax0 - bx1) < 1 or abs(ax1 - bx0) < 1:
        return (az0 >= bz0 and bz1 >= az1) or (bz0 >= az0 and az1 >= bz1)
    if abs(az0 - bz1) < 1 or abs(az1 - bz0) < 1:
        return (ax0 >= bx0 and bx1 >= ax1) or (bx0 >= ax0 and ax1 >= bx1)
    return False


def silo_kinds(unit: dict, bp: dict) -> set:
    """Which missiles a unit can build: 'tactical' and/or 'nuke' (launchers, and commanders with the missile
    enhancements)."""
    c, enh = cats(bp), set(unit.get("enh") or [])
    out = set()
    if "TACTICALMISSILEPLATFORM" in c or ("SILO" in c and "NUKE" not in c) or enh & {"TacticalMissile", "Missile"}:
        out.add("tactical")
    if ("NUKE" in c and "SILO" in c) or "TacticalNukeMissile" in enh:
        out.add("nuke")
    return out


def available_enhancements(unit: dict, bp: dict) -> list[str]:
    """Enhancements a commander or support commander can start now: not installed, prerequisite installed, and
    its slot free or taken by that prerequisite (FAF upgrades an enhancement in place, e.g. T2 -> T3
    engineering)."""
    installed = set(unit.get("enh") or [])
    by_name = {e["name"]: e for e in bp.get("enhancements") or []}
    used_slots = {by_name[n]["slot"]: n for n in installed if n in by_name}
    out = []
    for e in bp.get("enhancements") or []:
        name, req = e["name"], e.get("requires")
        if name in installed or (req and req not in installed):
            continue
        holder = used_slots.get(e["slot"])
        if holder is not None and holder != req:
            continue
        out.append(name)
    return out


class ToolKit:
    """Per-game translation between commands and bridge actions."""

    def __init__(self, info: dict, blueprints: dict[str, dict]):
        m = info["map"]
        self.width, self.height = float(m["width"]), float(m["height"])
        self.cell_w, self.cell_h = self.width / GRID, self.height / GRID
        self.fine_x, self.fine_z = fine_split(self.cell_w), fine_split(self.cell_h)
        self.has_fine = self.fine_x > 1 or self.fine_z > 1
        self.bps = blueprints
        self.deposits = [(float(d["x"]), float(d["z"]), d["type"]) for d in m.get("deposits", [])]
        self.role = {bp_id: role_of(bp) for bp_id, bp in blueprints.items()}

    # ------------------------------------------------------------------ geometry
    def cell(self, x: float, z: float) -> tuple[int, int]:
        return (min(GRID - 1, max(0, int(x / self.cell_w))), min(GRID - 1, max(0, int(z / self.cell_h))))

    def centre(self, cell) -> tuple[float, float]:
        return ((cell[0] + 0.5) * self.cell_w, (cell[1] + 0.5) * self.cell_h)

    def spot(self, x: float, z: float) -> tuple[int, int]:
        """Which of the 4 x 4 spots of its cell (x, z) lies in -- of its fine pixel on a map with them."""
        gx, gz = self.cell(x, z)
        fx, fz = self.sub(x, z) if self.has_fine else (0, 0)
        sx = int(((x / self.cell_w - gx) * self.fine_x - fx) * SPOTS)
        sz = int(((z / self.cell_h - gz) * self.fine_z - fz) * SPOTS)
        return (min(SPOTS - 1, max(0, sx)), min(SPOTS - 1, max(0, sz)))

    def spot_centre(self, cell, spot) -> tuple[float, float]:
        return ((cell[0] + (spot[0] + 0.5) / SPOTS) * self.cell_w, (cell[1] + (spot[1] + 0.5) / SPOTS) * self.cell_h)

    def sub(self, x: float, z: float) -> tuple[int, int]:
        """The fine pixel of its cell that (x, z) lies in."""
        gx, gz = self.cell(x, z)
        fx = int((x / self.cell_w - gx) * self.fine_x)
        fz = int((z / self.cell_h - gz) * self.fine_z)
        return (min(self.fine_x - 1, max(0, fx)), min(self.fine_z - 1, max(0, fz)))

    def where(self, cell, sub=None, spot=None) -> tuple[float, float]:
        """The world position of a command's cell, fine pixel (when given, on a map with them) and spot."""
        if sub is None or not self.has_fine or not _fine_pair(sub, self.fine_x, self.fine_z):
            if spot is None:
                return self.centre(cell)
            return self.spot_centre(cell, spot)
        sx, sz = spot if spot is not None else (SPOTS / 2 - 0.5, SPOTS / 2 - 0.5)
        return ((cell[0] + (sub[0] + (sx + 0.5) / SPOTS) / self.fine_x) * self.cell_w,
                (cell[1] + (sub[1] + (sz + 0.5) / SPOTS) / self.fine_z) * self.cell_h)

    def reach_of(self, c: dict) -> float:
        """The size of a command's chosen area: its fine pixel's, else its cell's."""
        if c.get("sub") is not None and self.has_fine:
            return max(self.cell_w / self.fine_x, self.cell_h / self.fine_z)
        return max(self.cell_w, self.cell_h)

    def place(self, x: float, z: float) -> dict:
        """A position's command arguments: its cell, and its fine pixel on a map with them."""
        out = {"cell": self.cell(x, z)}
        if self.has_fine:
            out["sub"] = self.sub(x, z)
        return out

    def adjacency_partners(self, bp_id: str, x: float, z: float, units: dict, reach: float = 6.0) -> list[str]:
        """Own structures near (x, z) that a new `bp_id` gives or takes an adjacency bonus with, nearest first:
        power generators beside factories and energy users, storages beside extractors and generators."""
        c = cats(self.bps.get(bp_id, {}))
        if "ENERGYPRODUCTION" in c:
            wanted = ("FACTORY", "MASSFABRICATION", "SHIELD", "ENERGYSTORAGE")
        elif "MASSSTORAGE" in c:
            wanted = ("MASSEXTRACTION", "MASSFABRICATION")
        elif "ENERGYSTORAGE" in c:
            wanted = ("ENERGYPRODUCTION",)
        elif "MASSFABRICATION" in c:
            wanted = ("ENERGYPRODUCTION", "MASSSTORAGE")
        else:
            return []
        near = []
        for uid, u in units.items():
            uc = cats(self.bps.get(u.get("bp"), {}))
            if "STRUCTURE" not in uc or not any(w in uc for w in wanted):
                continue
            d = math.hypot(float(u.get("x", 0.0)) - x, float(u.get("z", 0.0)) - z)
            if d <= reach:
                near.append((d, str(uid)))
        return [uid for _, uid in sorted(near)[:4]]

    def anchor_for(self, bp_id: str, x: float, z: float, units: dict) -> str | None:
        """The own structure a building of `bp_id` at (x, z) touches (engine adjacency rule), preferring the
        one its adjacency bonus is for; None when it touches nothing."""
        bp = self.bps.get(bp_id)
        if bp is None:
            return None
        rect = skirt_rect(bp, x, z)
        touching = []
        for uid, u in units.items():
            ubp = self.bps.get(u.get("bp"))
            if ubp is None or "STRUCTURE" not in cats(ubp) or (u.get("frac") or 1.0) < 1.0:
                continue
            if adjacent(rect, skirt_rect(ubp, u["x"], u["z"])):
                touching.append((uid, self.role.get(u.get("bp"))))
        if not touching:
            return None
        prefer = ANCHOR_PREFERENCE.get(self.role.get(bp_id), ())
        touching.sort(key=lambda t: prefer.index(t[1]) if t[1] in prefer else len(prefer))
        return touching[0][0]

    def deposit_in(self, cell, kind: str, taken: set) -> tuple[float, float] | None:
        """The free deposit of `kind` ('mass'/'hydro') in `cell`, else the nearest free one within 1.5 cells."""
        cx, cz = self.centre(cell)
        best, best_d = None, 1.5 * max(self.cell_w, self.cell_h)
        for x, z, t in self.deposits:
            if t != kind or (round(x), round(z)) in taken:
                continue
            if self.cell(x, z) == tuple(cell):
                return (x, z)
            d = math.hypot(x - cx, z - cz)
            if d < best_d:
                best, best_d = (x, z), d
        return best

    def is_commander(self, unit: dict | None) -> bool:
        return bool(unit) and "COMMAND" in cats(self.bps.get(unit.get("bp"), {}))

    # ------------------------------------------------------------------ commands -> actions
    def plan(self, commands: list[dict], units: dict[str, dict], taken: set) -> tuple[list, list, list]:
        """Bridge actions for `commands`. Builds away from deposits need a legal spot first: they come back
        as `queries` (find_build_site) plus `pending` entries for `complete()`.

        `units` maps id -> own unit (from the observation); `taken` holds rounded (x, z) of deposits that are
        occupied or already claimed."""
        actions, queries, pending = [], [], []
        for c in commands:
            verb, subject = c["verb"], [str(u) for u in c.get("subject") or []]
            if verb == "end" or not subject:
                continue
            if verb in REPLACE_VERBS or verb == "stop":
                # A unit at a build finishes it: replacing orders and STOP skip it. M28 queued most of its moves
                # behind builds (no clear) and reassigned builders whose work others finished; replayed, they
                # cancelled every build the clone started (2026-10-05: 95 commander moves, 60 builds, nothing done
                # in 9 min; then its first factory left at 94 % by STOP + a new build)
                subject = [u for u in subject if (units.get(u) or {}).get("order") not in BUILDING_ORDERS]
                if not subject:
                    continue
            one = units.get(subject[0])
            if verb in REPLACE_VERBS or verb == "stop":
                # Later replacement wins over a build awaiting placement in phase two.
                pending = [(*entry[:4], entry[4] and entry[0] not in subject) for entry in pending]
                # IssueStop is itself queued by the engine. Clear first so STOP
                # actually cancels an active build instead of waiting behind it.
                actions.append({"type": "clear", "units": subject if verb in SET_SUBJECT else subject[:1]})
            if verb == "build":
                bp = c.get("bp")
                if not one or bp not in (one.get("can_build") or []) or bp not in self.bps:
                    continue
                if c.get("place") not in (None, "cell", "next"):
                    continue
                clear = one.get("order") not in BUILDING_ORDERS
                if not clear and (one.get("queue") or 0) >= BUILD_QUEUE_CAP:
                    continue            # a builder at work takes a few more builds, not one every decision
                cell, spot_arg = c.get("cell"), c.get("spot")
                if cell is not None and not _grid_pair(cell, GRID):
                    continue
                if spot_arg is not None and not _grid_pair(spot_arg, SPOTS):
                    continue
                pc = placement_class(self.bps[bp])
                if pc >= 2:
                    spot = self.deposit_in(c["cell"], "mass" if pc == 2 else "hydro", taken) if c.get("cell") else None
                    if spot:
                        taken.add((round(spot[0]), round(spot[1])))
                        if clear:
                            # This successful immediate build replaces older delayed work. A failed
                            # deposit lookup must leave that work intact; BUILDING builders append.
                            pending = [(*entry[:4], entry[4] and entry[0] != subject[0]) for entry in pending]
                            actions.append({"type": "clear", "units": subject[:1]})
                        actions.append({"type": "build", "units": subject[:1], "bp": bp, "pos": list(spot)})
                    continue
                if c.get("place") == "next":
                    if str(c.get("anchor")) not in units:
                        continue
                    # against the anchor; when nothing against it is free, the nearest legal spot around it
                    anchor = units.get(str(c["anchor"])) or {}
                    pos = [anchor.get("x", 0.0), anchor.get("z", 0.0)]
                    queries.append({"type": "find_build_site", "bp": bp, "pos": pos, "radius": 0,
                                    "adjacent_to": [str(c["anchor"])], "adjacent_only": True})
                    queries.append({"type": "find_build_site", "bp": bp, "pos": pos,
                                    "radius": 1.5 * max(self.cell_w, self.cell_h)})
                    pending.append((subject[0], bp, 2, clear, True))
                    continue
                if c.get("cell") is None and c.get("aim") is not None:         # a teacher label's exact aim
                    aim = c["aim"]
                    if not _position_pair(aim):
                        continue
                    cx, cz = aim
                elif c.get("cell") is not None:
                    cx, cz = self.where(c["cell"], c.get("sub"), c.get("spot") or (SPOTS // 2, SPOTS // 2))
                else:
                    continue
                # a building with an adjacency bonus placed by spot snaps against a structure it benefits from
                # when one is right there (the spot grid is coarse: power generators "beside" a factory landed a
                # tile off, 2026-10-12), as a player's build preview does; else the spot as chosen
                partners = self.adjacency_partners(bp, cx, cz, units)
                if partners:
                    queries.append({"type": "find_build_site", "bp": bp, "pos": [cx, cz], "radius": 0,
                                    "adjacent_to": partners, "adjacent_only": True})
                queries.append({"type": "find_build_site", "bp": bp, "pos": [cx, cz],
                                "radius": 0.75 * self.reach_of(c)})
                pending.append((subject[0], bp, 2 if partners else 1, clear, True))
            elif verb == "produce":
                bp = c.get("bp")
                if one and bp in build_options(one, self.bps):
                    actions.append({"type": "factory_build", "units": subject[:1], "bp": bp,
                                    "count": int(c.get("count") or 1)})
            elif verb == "upgrade":         # a factory's chosen upgrade, else the blueprint's one
                target = c.get("bp") if one and c.get("bp") in (one.get("can_build") or []) else \
                    one and self.bps.get(one["bp"], {}).get("upgradesTo")
                if target in self.bps:
                    actions.append({"type": "upgrade", "units": subject[:1], "bp": target})
            elif verb == "enhance":
                if c.get("enhancement"):
                    actions.append({"type": "enhance", "units": subject[:1], "name": c["enhancement"]})
            elif verb in ("move", "attack_move", "patrol", "rally"):
                actions.append({"type": verb, "units": subject if verb != "rally" else subject[:1],
                                "pos": list(self.where(c["cell"], c.get("sub")))})
            elif verb == "reclaim":
                if not _grid_pair(c.get("cell"), GRID):
                    continue
                # the props come back with the query's answer; complete() queues the best of them (this
                # decision's clear went out above)
                centre = list(self.where(c["cell"], c.get("sub")))
                queries.append({"type": "reclaim", "pos": centre, "energy": True,
                                "radius": max(12.0, self.reach_of(c)), "limit": 4 * RECLAIM_TARGETS})
                pending.append((tuple(subject), (RECLAIM, centre), 1, False, True))
            elif verb in ("commander_move", "commander_attack_move"):
                actions.append({"type": verb.replace("commander_", ""), "units": subject[:1],
                                "pos": list(self.where(c["cell"], c.get("sub")))})
            elif verb == "unload":
                actions.append({"type": "transport_unload", "units": subject[:1],
                                "pos": list(self.where(c["cell"], c.get("sub")))})
            elif verb == "nuke":
                actions.append({"type": "nuke", "units": subject[:1], "pos": list(self.where(c["cell"], c.get("sub")))})
            elif verb in ("attack", "tactical"):
                actions.append({"type": verb, "units": subject, "target": str(c["target"])})
            elif verb in ("overcharge", "capture", "repair"):
                actions.append({"type": verb, "units": subject, "target": str(c["target"])})
            elif verb == "assist":
                actions.append({"type": "guard", "units": subject, "target": str(c["target"])})
            elif verb == "load":
                actions.append({"type": "transport_load", "units": subject, "target": str(c["target"])})
            elif verb == "silo_build":
                actions.append({"type": "silo_build", "units": subject[:1],
                                "nuke": SILO_KINDS[int(c.get("option") or 0)] == "nuke"})
            elif verb == "fire_state":
                actions.append({"type": "fire_state", "units": subject, "state": int(c.get("option") or 0)})
            elif verb == "toggle":
                bit = int(c.get("option") or 0)
                on = bit not in ((one or {}).get("toggles_on") or [])
                actions.append({"type": "toggle", "units": subject[:1], "bit": bit, "on": on})
            elif verb == "pause":
                actions.append({"type": "pause", "units": subject[:1], "on": not (one or {}).get("paused")})
            elif verb == "dive":
                actions.append({"type": "dive", "units": subject})
            elif verb == "stop":
                actions.append({"type": "stop", "units": subject})
        return actions, queries, pending

    @staticmethod
    def complete(pending: list, answers: list) -> list[dict]:
        """Build actions from the find_build_site answers: each pending build owns its next `n` answers and
        takes the first that found a site. Clear and build execute together only after success.
        Later replacing commands deactivate pending entries but retain their query answer offsets.
        Other delayed builds run in phase two; arbitrary mixed append queues are not exact replay."""
        out, k = [], 0
        for entry in pending:
            unit_id, bp, n = entry[:3]
            clear, active = entry[3:] if len(entry) == 5 else (False, True)
            if isinstance(bp, tuple) and bp[0] == RECLAIM:
                ans = answers[k] if k < len(answers) else None
                if active and ans and ans.get("ok"):
                    # `pos`: the commanded cell's centre, so the recorded actions give back the command (labels)
                    out += [{"type": "reclaim", "units": list(unit_id), "target": str(p["id"]), "pos": bp[1]}
                            for p in reclaim_route(ans.get("value") or [])]
                k += n
                continue
            for ans in answers[k:k + n]:
                if active and ans and ans.get("ok") and _position_pair(ans.get("value")):
                    if clear:
                        out.append({"type": "clear", "units": [unit_id]})
                    out.append({"type": "build", "units": [unit_id], "bp": bp, "pos": ans["value"]})
                    break
            k += n
        return out

    # ------------------------------------------------------------------ actions -> commands (labels)
    def commands_from_actions(self, actions: list[dict], units: dict[str, dict],
                              prior_commands=()) -> list[dict]:
        """Reduce events in recording order, consuming clear internally (also against BC carry).

        This compresses unmarked navigation to its last order and patrol runs to their farthest point;
        it cannot replay arbitrary raw queues, toggle on-state, or more than the policy slot budget.
        Independent commands are canonicalized; surviving dependent queue orders retain chronology.
        """
        out = [dict(c, subject=list(c.get("subject") or [])) for c in prior_commands]
        patrols = {}
        for a in actions:
            subject = [str(u) for u in a.get("units") or []]
            t = a.get("type")
            if t in ("clear", "stop"):
                canceled = set(subject)
                out = [dict(c, subject=[u for u in c["subject"] if u not in canceled])
                       if c["verb"] in QUEUE_VERBS and canceled.intersection(c["subject"]) else c for c in out]
                out = [c for c in out if c["subject"]]
                for u in subject:
                    patrols.pop(u, None)
                if t == "clear":
                    continue
            if t == "patrol":
                for u in subject:
                    if self.is_commander(units.get(u)):
                        continue
                    pos = a["pos"]
                    if u not in patrols:
                        c = {"verb": "patrol", "subject": [u], **self.place(*pos)}
                        out.append(c)
                        patrols[u] = (c, pos)
                    else:
                        c, old = patrols[u]
                        here = units.get(u) or {}
                        origin = (here.get("x", old[0]), here.get("z", old[1]))
                        if math.dist(pos, origin) > math.dist(old, origin):
                            c.update(self.place(*pos))
                            patrols[u] = (c, pos)
                continue
            commands = self._commands_from_actions([a], units)
            for c in commands:
                if c["verb"] in QUEUE_VERBS:
                    for u in c["subject"]:
                        patrols.pop(u, None)
            out.extend(commands)
        grouped = group_independent(merge_commands(out), SET_SUBJECT, VERB_ORDER)
        # A unit at a build finishes it (plan): an AI's clear + build there is a build queued behind it, and a STOP
        # to it is no command we execute. As STOP + build labels they taught the clone to re-order every busy
        # engineer each decision -- 12 STOP + 12 builds filled all 24 slots from minute 4 (2026-10-05)
        busy = {u for u, unit in units.items() if unit.get("order") in BUILDING_ORDERS}
        out = []
        for c in grouped:
            if c["verb"] == "stop":
                c = dict(c, subject=[u for u in c["subject"] if u not in busy])
                if not c["subject"]:
                    continue
            out.append(c)
        return out

    def _commands_from_actions(self, actions: list[dict], units: dict[str, dict]) -> list[dict]:
        """The commands that produce `actions` (the teacher's recorded game commands), in canonical order.
        A unit's run of patrol waypoints becomes one patrol to its farthest waypoint. The commander moves by
        its own commands; it is taken out of any set."""
        out, patrols = [], {}
        for a in actions:
            t, subject = a.get("type"), [str(u) for u in a.get("units") or []]
            if not subject or t == "clear":          # the executor's own queue clearing, not a command
                continue
            acu = [u for u in subject if self.is_commander(units.get(u))]
            if acu and t in ("move", "attack_move"):
                out.append({"verb": f"commander_{t}", "subject": acu[:1], **self.place(*a["pos"])})
            if acu and t not in ("build", "upgrade", "enhance", "overcharge", "silo_build", "tactical", "toggle",
                                 "pause", "guard", "repair", "stop"):     # its own verbs and COMMANDER_SET_VERBS
                subject = [u for u in subject if u not in acu]
                if not subject:
                    continue
            if t == "build":
                bp_id = a.get("bp")
                bp = self.bps.get(bp_id)
                if bp is None:
                    continue
                x, z = a["pos"]
                if placement_class(bp) >= 2:
                    out.append({"verb": "build", "subject": subject[:1], "bp": bp_id, "place": "cell",
                                "cell": self.cell(x, z), "spot": None})
                    continue
                anchor = self.anchor_for(bp_id, x, z, units)
                if anchor is not None:
                    out.append({"verb": "build", "subject": subject[:1], "bp": bp_id, "place": "next", "anchor": anchor})
                else:
                    out.append({"verb": "build", "subject": subject[:1], "bp": bp_id, "place": "cell",
                                **self.place(x, z), "spot": self.spot(x, z)})
            elif t == "factory_build":
                count = min(COUNTS, key=lambda n: abs(n - int(a.get("count") or 1)))
                out.append({"verb": "produce", "subject": subject[:1], "bp": a.get("bp"), "count": count})
            elif t in ("move", "attack_move", "rally"):
                out.append({"verb": t, "subject": subject, **self.place(*a["pos"])})
            elif t == "patrol":
                for u in subject:
                    patrols.setdefault(u, []).append(a["pos"])
            elif t == "guard" and a.get("target") is not None:
                out.append({"verb": "assist", "subject": subject, "target": str(a["target"])})
            elif t in ("attack", "tactical", "overcharge", "capture", "repair") and a.get("target") is not None:
                out.append({"verb": t, "subject": subject, "target": str(a["target"])})
            elif t == "reclaim":
                # where the prop is (captured since 2026-10-05, the teacher's own), else where the engineers are
                # (earlier captures kept only the prop's id; AIs reclaim around their engineers)
                pos = a.get("target_pos") or a.get("pos")
                if not _position_pair(pos):
                    here = [units[u] for u in subject if u in units]
                    pos = (sum(u["x"] for u in here) / len(here), sum(u["z"] for u in here) / len(here)) \
                        if here else None
                if pos is not None:
                    out.append({"verb": "reclaim", "subject": subject, **self.place(*pos)})
            elif t == "transport_load" and a.get("target") is not None:
                out.append({"verb": "load", "subject": subject, "target": str(a["target"])})
            elif t == "transport_unload":
                out.append({"verb": "unload", "subject": subject[:1], **self.place(*a["pos"])})
            elif t == "nuke":
                out.append({"verb": "nuke", "subject": subject[:1], **self.place(*a["pos"])})
            elif t == "upgrade":
                out.append({"verb": "upgrade", "subject": subject[:1], "bp": a.get("bp")})
            elif t == "enhance" and a.get("name"):
                out.append({"verb": "enhance", "subject": subject[:1], "enhancement": a["name"]})
            elif t == "silo_build":
                out.append({"verb": "silo_build", "subject": subject[:1], "option": 1 if a.get("nuke") else 0})
            elif t == "fire_state":
                out.append({"verb": "fire_state", "subject": subject, "option": int(a.get("state") or 0)})
            elif t == "toggle":
                out.append({"verb": "toggle", "subject": subject[:1], "option": int(a.get("bit") or 0)})
            elif t in ("pause", "dive", "stop"):
                out.append({"verb": t, "subject": subject if t != "pause" else subject[:1]})
        for u, points in patrols.items():
            here = units.get(u)
            origin = (here["x"], here["z"]) if here else points[0]
            far = max(points, key=lambda p: math.hypot(p[0] - origin[0], p[1] - origin[1]))
            out.append({"verb": "patrol", "subject": [u], **self.place(*far)})
        out.sort(key=lambda c: VERB_ORDER.get(c["verb"], 99))
        return out


# Orthogonal controls never compete with navigation or disappear at a queue clear.
CONTROL_VERBS = frozenset({"fire_state", "dive", "toggle", "pause", "rally"})
QUEUE_VERBS = set(VERBS) - CONTROL_VERBS - {"end"}
NAVIGATION_VERBS = (SET_SUBJECT - CONTROL_VERBS) | {"commander_move", "commander_attack_move"}


RECLAIM = "<reclaim>"       # a pending entry's blueprint slot: an area reclaim waiting for its props


def reclaim_route(props: list[dict], n: int = RECLAIM_TARGETS) -> list[dict]:
    """The `n` most valuable props (mass, energy at a tenth), visited nearest-first from the richest."""
    best = sorted((p for p in props if _position_pair((p.get("x"), p.get("z"))) and p.get("id") is not None),
                  key=lambda p: -(p.get("mass", 0) + p.get("energy", 0) / 10))[:n]
    route = best[:1]
    rest = best[1:]
    while rest:
        last = route[-1]
        nxt = min(rest, key=lambda p: math.hypot(p["x"] - last["x"], p["z"] - last["z"]))
        rest.remove(nxt)
        route.append(nxt)
    return route


def _position_pair(value):
    return (isinstance(value, (tuple, list)) and len(value) == 2 and
            all(isinstance(n, (int, float)) and math.isfinite(n) for n in value))


def fine_split(cell_size: float) -> int:
    """How many fine pixels a cell of `cell_size` game units splits into per side (1 on a 5 km map)."""
    return max(1, min(FINE, int(round(cell_size / FINE_UNITS))))


def _fine_pair(value, nx, nz):
    return (isinstance(value, (tuple, list)) and len(value) == 2 and
            all(isinstance(n, int) and not isinstance(n, bool) for n in value) and
            0 <= value[0] < nx and 0 <= value[1] < nz)


def _grid_pair(value, size):
    return (isinstance(value, (tuple, list)) and len(value) == 2 and
            all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n < size for n in value))


def merge_commands(commands: list[dict]) -> list[dict]:
    """Chronological plain commands: compress navigation only, then canonicalize independent orders.

    No raw clears are inferred here. Build/production accumulation survives. Grouping is adjacent only:
    grouping across an intervening same-unit queue order would move a dependent order in time.
    """
    last = {}
    for i, c in enumerate(commands):
        if c["verb"] in NAVIGATION_VERBS:
            for u in c.get("subject") or []:
                last[u] = i
    out = []
    for i, c in enumerate(commands):
        keep = [u for u in c.get("subject") or []
                if c["verb"] not in NAVIGATION_VERBS or last.get(u) == i]
        if keep:
            out.append(dict(c, subject=keep))
    # A stable topological canonical sort: earlier same-unit queue commands are prerequisites.
    result = []
    while out:
        ready = []
        blocked = set()
        for i, c in enumerate(out):
            subjects = set(c["subject"]) if c["verb"] in QUEUE_VERBS else set()
            if not subjects & blocked:
                ready.append(i)
            blocked.update(subjects)
        i = min(ready, key=lambda j: (VERB_ORDER.get(out[j]["verb"], 99), j))
        c = out.pop(i)
        if (result and c["verb"] in SET_SUBJECT and
                {k: v for k, v in result[-1].items() if k != "subject"} ==
                {k: v for k, v in c.items() if k != "subject"}):
            result[-1]["subject"] += [u for u in c["subject"] if u not in result[-1]["subject"]]
        else:
            result.append(c)
    return result


assert SINGLE_SUBJECT | SET_SUBJECT == set(VERBS) - {"end"}
assert set(VERB_ORDER) == set(VERBS) - {"end"}
