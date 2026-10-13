"""Observation v3 for the full-toolbox action space (docs/full-toolbox-design.md, section 2; v2 was
docs/tools-design.md, section 3).

`ToolsObserver` is stateful (memory maps, when each unit was last commanded, per-verb success), so it is fed
every decision of a game in order -- live from ToolsEnv, or offline from a raw recording:

    ob = ToolsObserver(info, blueprints, me, kit)
    for each decision:  x = ob.observe(raw_obs);  ...;  ob.note_commands(commands, tick, results)

`observe()` returns numpy arrays plus three id lists that map the network's pointers back to the game:
- `ids`: the unit id of every token (subjects, targets, anchors);
- `cand_ids`: the blueprint of every candidate -- everything our builders can build right now (the bridge's
  `can_build`), which is what `build` and `produce` point at; `can_build` pairs (token, candidate) say who can
  build what;
- `enh_names`: the enhancement of every enhancement candidate (what our commanders can start now);
  `can_enh` pairs (token, candidate) say whose it is.
"""
from __future__ import annotations

import math

import numpy as np
from .blueprint_vocab import blueprint_indices

from .tools import (GRID, fine_split, N_OPTIONS, N_TOGGLES, PLACEMENT_CLASSES, SET_SUBJECT, SILO_KINDS, SINGLE_SUBJECT,
                    TARGET_ROWS, TARGET_VERBS, VERBS, ToolKit, available_enhancements, build_options, cats, combat_weapons,
                    placement_class, silo_kinds, skirt_rect)

assert TARGET_ROWS == ("enemy", "own", "damaged", "wreck", "transport")      # the rows `_masks` fills

TOKEN_TYPES = ("own", "enemy", "ghost", "radar", "mass_spot", "hydro_spot", "wreck")
T_OWN, T_ENEMY, T_GHOST, T_RADAR, T_MASS, T_HYDRO, T_WRECK = range(7)
MAX_TOKENS = 768
CAPS = {T_OWN: 480, T_ENEMY: 128, T_GHOST: 48, T_RADAR: 48, T_MASS: 40, T_HYDRO: 8, T_WRECK: 48}
N_FEATURES = 70        # 0-6 type, 7-12 place, 13-22 state, 23-48 blueprint stats, 49-50 heading, 51-53 velocity,
                       # 54 buildings touching it, 55-57 its actual mass and energy output and energy use,
                       # 58-69 upgrades, missiles, cargo, fire state, abilities, what it can build, pause, shield
N_ORDERS = 48                       # command-type ids 0..46 (+1 so 0 means no order)
MAP_CHANNELS = ("height", "water", "slope", "mass_spots", "hydro_spots", "own_mobile", "own_structures",
                "enemy_mobile", "enemy_structures", "remembered", "radar", "last_looked", "enemy_strength",
                "own_start", "enemy_start", "buildable",
                # danger maps (2026-10-11): firepower reaching each cell (weapon ranges x damage per second, log),
                # and shield cover; enemy ones from seen units and remembered structures
                "enemy_ground_fire", "enemy_air_fire", "own_ground_fire", "enemy_shield", "own_shield")
N_SCALARS = 34 + len(VERBS)            # 26 game numbers, 6 of tech, 2 of the map + the success rate of each verb
PRIVILEGED = ("massIncome", "energyIncome", "armyMass", "structureMass", "engineerMass", "combat", "engineers",
              "factories", "mexes", "commanderHealth")
PRIV_SCALE = (50.0, 500.0, 2000.0, 5000.0, 1000.0, None, None, None, None, 1.0)   # None: log1p
V_INDEX = {v: i for i, v in enumerate(VERBS)}
N_STATIC = 26
# blueprint candidates: stats + build time and energy + how many we have + where it goes (4 classes or mobile)
MAX_CANDIDATES = 128
N_CAND_FEATURES = N_STATIC + 3 + len(PLACEMENT_CLASSES) + 1
CAND_MOBILE = len(PLACEMENT_CLASSES)   # cand_kind of a mobile unit (produced by a factory, not placed)
# enhancement candidates: cost, time, what it gives, slot, whether it replaces the one in its slot
MAX_ENHANCEMENTS = 48
ENH_SLOTS = ("LCH", "RCH", "Back")
N_ENH_FEATURES = 5 + len(ENH_SLOTS) + 1 + 2
# which arrays are per token, per candidate, per step (the BC cache and collate use these)
# token_last: the verb of the last order the acting side gave the unit (0: none yet); token_last_feat: where
# that order pointed (relative to the unit now) and whether it pointed anywhere. The engine reports a unit's
# current command type, not what it is building, reclaiming or heading for or who gave it: M28 keeps that in its
# assignment tables, and a clone deciding per unit must see it too (2026-10-06).
N_LAST_FEATURES = 3
TOKEN_KEYS = ("tokens", "token_bp", "token_order", "token_type", "token_last", "token_last_feat", "mask_anchor",
              "mask_silo", "mask_toggle")
TOKEN_T_KEYS = ("mask_subject", "mask_target")          # [rows, T]
CAND_KEYS = ("cand_bp", "cand_feat", "cand_kind")
ENH_KEYS = ("enh_idx", "enh_feat")
PAIR_KEYS = ("can_build", "can_enh")                     # [P, 2] (token, candidate)
STEP_KEYS = ("map", "scalars", "privileged", "belief", "mask_verb", "mask_build_cell", "mask_move_cell", "fine",
             "focus")
# the close-ups' centres (focus [N_FOCUS, 3]: x / width, z / height, present): our commander, our army, the fight
N_FOCUS = 3


def _weapon_dps(bp: dict) -> float:
    return sum(float(w.get("damage") or 0) * float(w.get("rof") or 0) for w in combat_weapons(bp))


GROUND_FIRE = ("UWRC_DirectFire", "UWRC_IndirectFire")


def _threat_profile(bp: dict) -> tuple[tuple, tuple, float]:
    """((damage per second, range) of each ground weapon, the same of each anti-air weapon, shield radius) of a
    blueprint, for the danger maps. Each weapon covers its own range (summed at the longest range, a commander's
    missile made it a 5000-dps threat across the map). A commander (or support commander) counts with its first
    direct-fire weapon only: its others (tactical missile, microwave laser, ...) come with upgrades."""
    weapons = combat_weapons(bp)
    if "COMMAND" in cats(bp) or "SUBCOMMANDER" in cats(bp):
        weapons = [w for w in weapons if w.get("rangeCategory") == "UWRC_DirectFire"][:1]

    def each(ws):
        return tuple((min(float(w.get("damage") or 0) * float(w.get("rof") or 0), 5000.0), float(w.get("range") or 0))
                     for w in ws if (w.get("damage") or 0) and (w.get("rof") or 0) and (w.get("range") or 0))

    return (each([w for w in weapons if w.get("rangeCategory") in GROUND_FIRE]),
            each([w for w in weapons if w.get("rangeCategory") == "UWRC_AntiAir"]),
            float(bp.get("shieldSize") or 0.0) / 2.0)


def _weapon_range(bp: dict) -> float:
    return max([float(w.get("range") or 0) for w in combat_weapons(bp)] or [0.0])


def enhancement_vocabulary(blueprints: dict[str, dict]) -> dict[str, int]:
    """Every enhancement name of the game's blueprints -> 1.. (0: unknown), the same in every game of a mod
    version (sorted)."""
    names = sorted({e["name"] for bp in blueprints.values() for e in bp.get("enhancements") or []})
    return {n: i + 1 for i, n in enumerate(names)}


class ToolsObserver:
    def __init__(self, info: dict, blueprints: dict[str, dict], me: int, kit: ToolKit):
        self.kit, self.me, self.bps = kit, me, blueprints
        m = info["map"]
        self.width, self.height = float(m["width"]), float(m["height"])
        self.diag = math.hypot(self.width, self.height)
        armies = {a["index"]: a for a in info["armies"]}
        self.start = tuple(armies[me]["start"])
        enemies = [a for a in info["armies"] if a.get("relation") == "enemy" and not a.get("civilian")]
        self.enemy_start = tuple(enemies[0]["start"]) if enemies else self.start
        self.enemy_indices = [a["index"] for a in enemies]
        # Modded teachers must not shift the identities of every ordinary unit.
        self.bp_index = blueprint_indices(blueprints)
        self.vocab = max(self.bp_index.values(), default=0) + 1
        self.enh_vocab = enhancement_vocabulary(blueprints)
        self.static = {bp_id: self._static(bp) for bp_id, bp in blueprints.items()}
        self.zero_static = np.zeros(N_STATIC, np.float32)
        self.is_mobile = {i: "MOBILE" in cats(bp) for i, bp in blueprints.items()}
        self.is_structure = {i: "STRUCTURE" in cats(bp) for i, bp in blueprints.items()}
        self.place_class = {i: placement_class(bp) for i, bp in blueprints.items() if "STRUCTURE" in cats(bp)}
        # what the mask builder asks of a blueprint, once per blueprint instead of once per unit per decision
        self.flags = {i: self._flags(bp) for i, bp in blueprints.items()}
        self.threat = {i: _threat_profile(bp) for i, bp in blueprints.items()}
        self.zero_flags = self._flags({})
        cw, ch = self.width / GRID, self.height / GRID
        self.cell_cx = (np.arange(GRID) + 0.5) * cw          # the cells' centres, for the vision marks
        self.cell_cz = (np.arange(GRID) + 0.5) * ch
        # static map planes
        self.static_map = np.zeros((5, GRID, GRID), np.float32)
        hm = m.get("heightmap") or {}
        n = hm.get("n")
        buildable = np.ones((GRID, GRID), bool)
        water = np.zeros((GRID, GRID), bool)
        if n:
            terrain = np.asarray(hm["terrain"], np.float32).reshape(n, n)
            surface = np.asarray(hm["surface"], np.float32).reshape(n, n)
            t = _resize(terrain, GRID)
            wet = _resize((surface > terrain + 0.01).astype(np.float32), GRID)
            gy, gx = np.gradient(t)
            slope = np.hypot(gx, gy) / max(self.width / GRID, 1.0)
            self.static_map[0] = t / max(1.0, float(t.max()))
            self.static_map[1] = wet
            self.static_map[2] = np.clip(slope * 4.0, 0.0, 1.0)
            buildable = (wet < 0.5) & (slope < 0.35)
            water = wet >= 0.5
        for x, z, kind in kit.deposits:
            gx_, gz_ = kit.cell(x, z)
            self.static_map[3 if kind == "mass" else 4, gz_, gx_] = 1.0
        self.buildable = buildable
        self.water = water
        self.start_maps = np.zeros((2, GRID, GRID), np.float32)
        for k, (sx, sz) in enumerate((self.start, self.enemy_start)):
            cx, cz = kit.cell(sx, sz)
            yy, xx = np.mgrid[0:GRID, 0:GRID]
            self.start_maps[k] = np.exp(-((xx - cx) ** 2 + (yy - cz) ** 2) / 4.0)
        self.move_cells = np.ones((GRID, GRID), bool)
        pa = m.get("playable")
        if pa:
            x0, z0 = kit.cell(pa[0], pa[1])
            x1, z1 = kit.cell(max(pa[0], pa[2] - 0.01), max(pa[1], pa[3] - 0.01))
            self.move_cells[:] = False
            self.move_cells[z0:z1 + 1, x0:x1 + 1] = True
        self.reset()

    def _static(self, bp: dict) -> np.ndarray:
        c = cats(bp)
        return np.array([
            math.log1p(float(bp.get("mass") or 0)) / 10.0, math.log1p(float(bp.get("health") or 0)) / 12.0,
            min(_weapon_dps(bp), 2000.0) / 200.0, _weapon_range(bp) / 100.0, float(bp.get("speed") or 0) / 10.0,
            float(bp.get("tech") or 0) / 4.0, float(bp.get("vision") or 0) / 100.0,
            math.log1p(float(bp.get("buildRate") or 0)) / 5.0,
            "ENGINEER" in c, "FACTORY" in c and "STRUCTURE" in c, "MASSEXTRACTION" in c,
            "ENERGYPRODUCTION" in c and "STRUCTURE" in c, "STRUCTURE" in c, "AIR" in c, "NAVAL" in c, "LAND" in c,
            "COMMAND" in c, "DEFENSE" in c, "SCOUT" in c, "MOBILE" in c,
            float(bp.get("massProduction") or 0) / 10.0, float(bp.get("energyProduction") or 0) / 100.0,
            math.log1p(float(bp.get("massStorage") or 0)) / 10.0, math.log1p(float(bp.get("energyStorage") or 0)) / 10.0,
            float(bp.get("radar") or 0) / 200.0, float(bool(bp.get("upgradesTo"))),
        ], np.float32)

    def reset(self) -> None:
        self.last_looked = np.full((GRID, GRID), -10 ** 6, np.float64)
        self.enemy_memory = np.zeros((GRID, GRID), np.float32)
        self.last_tick = 0
        self.commanded: dict[str, int] = {}
        self.last_order: dict[str, tuple] = {}      # unit id -> (verb index, x, z or None, None)
        self.claimed: dict[tuple, int] = {}
        self.success = np.ones(len(VERBS), np.float32)
        self.wrecks: list[dict] = []
        self.last_pos: dict[str, tuple] = {}
        self.taken: set = set()

    # ------------------------------------------------------------------ updates from the acting side
    def note_commands(self, commands: list[dict], tick: int, results: list | None = None) -> None:
        for c in commands:
            v = V_INDEX.get(c.get("verb"))
            at = None
            cell = c.get("cell")
            if cell is not None and len(cell) == 2:
                at = ((cell[0] + 0.5) * self.width / GRID, (cell[1] + 0.5) * self.height / GRID)
            elif c.get("target") is not None:
                p = self.last_pos.get(str(c["target"]))
                at = (p[0], p[1]) if p else None
            elif c.get("anchor") is not None:
                p = self.last_pos.get(str(c["anchor"]))
                at = (p[0], p[1]) if p else None
            for u in c.get("subject") or []:
                self.commanded[str(u)] = tick
                if v is not None:
                    self.last_order[str(u)] = (v, at)
        if results is not None:
            ok = [bool(r.get("ok")) for r in results]
            for c in commands:
                v = V_INDEX.get(c["verb"])
                if v is not None and ok:
                    self.success[v] = 0.8 * self.success[v] + 0.2 * (sum(ok) / len(ok))

    def claim(self, x: float, z: float, tick: int) -> None:
        self.claimed[(round(x), round(z))] = tick

    def set_wrecks(self, props: list[dict]) -> None:
        """The reclaim tokens: the most valuable props of the last scans (wrecks, rocks, trees)."""
        worth = lambda p: (p.get("mass") or 0) + (p.get("energy") or 0) / 10.0  # noqa: E731
        self.wrecks = sorted((p for p in props if (p.get("mass") or 0) >= 2 or (p.get("energy") or 0) >= 25),
                             key=lambda p: -worth(p))[:CAPS[T_WRECK]]

    def can_build_of(self, u: dict) -> list[str]:
        """Legal builds, including production queues on nearly completed factories.

        Legacy recordings fall back to the blueprint list; unfinished factories
        expose mobile production only, never upgrades or construction commands.
        """
        return build_options(u, self.bps)

    # ------------------------------------------------------------------ observation
    def observe(self, obs: dict) -> dict:
        tick = int(obs.get("tick", 0))
        relation = {a["index"]: a["relation"] for a in obs.get("armies", [])}
        own, enemy, ghost, radar = [], [], [], []
        for u in obs.get("units", []):
            if u.get("army") == self.me:
                own.append(u)
            elif u.get("radar"):
                radar.append(u)
            elif u.get("ghost"):
                ghost.append(u)
            elif relation.get(u.get("army"), "enemy") == "enemy":
                enemy.append(u)
        # deposits and who holds them
        holders = {}
        for group, who in ((own, 1), (enemy, 2), (ghost, 2)):
            for u in group:
                if self.kit.role.get(u.get("bp")) in ("mex", "hydro"):
                    holders[(round(u["x"]), round(u["z"]))] = who
        taken = set()
        spots = []
        for x, z, t in self.kit.deposits:
            who = holders.get((round(x), round(z)), 0)
            if not who:
                for (hx, hz), h in holders.items():
                    if abs(hx - x) <= 2.5 and abs(hz - z) <= 2.5:     # extractors snap up to ~1 unit away
                        who = h
                        break
            claimed = tick - self.claimed.get((round(x), round(z)), -10 ** 9) < 900
            if who or claimed:
                taken.add((round(x), round(z)))
            spots.append({"x": x, "z": z, "type": t, "who": who, "claimed": claimed})
        self.taken = taken

        # tokens, in fixed type order with per-type caps
        rows: list[tuple[int, dict]] = []
        for ttype, group in ((T_OWN, own), (T_ENEMY, enemy), (T_GHOST, ghost), (T_RADAR, radar)):
            rows += [(ttype, u) for u in group[: CAPS[ttype]]]
        rows += [(T_MASS, s) for s in spots if s["type"] == "mass"][: CAPS[T_MASS]]
        rows += [(T_HYDRO, s) for s in spots if s["type"] != "mass"][: CAPS[T_HYDRO]]
        rows += [(T_WRECK, w) for w in self.wrecks]
        rows = rows[:MAX_TOKENS]
        n = len(rows)
        feats = np.zeros((n, N_FEATURES), np.float32)
        last = np.zeros(n, np.int64)
        last_feat = np.zeros((n, N_LAST_FEATURES), np.float32)
        bp_idx = np.zeros(n, np.int64)
        orders = np.zeros(n, np.int64)
        types = np.zeros(n, np.int64)
        ids: list[str | None] = []
        seen_now: dict[str, tuple] = {}
        W, H, D = self.width, self.height, self.diag
        hx, hz = self.start
        ex, ez = self.enemy_start
        builds: dict[int, list[str]] = {}          # token -> what it can build now
        for i, (t, u) in enumerate(rows):
            x, z = float(u["x"]), float(u["z"])
            f = feats[i]
            f[t] = 1.0
            f[7], f[8] = x / W, z / H
            f[9], f[10] = (x - hx) / W, (z - hz) / H
            f[11] = math.hypot(x - hx, z - hz) / D
            f[12] = math.hypot(x - ex, z - ez) / D
            types[i] = t
            if t <= T_RADAR:
                bp = u.get("bp")
                ids.append(str(u["id"]))
                if t != T_RADAR and bp:
                    maxhp = u.get("maxhp") or 0.0
                    f[13] = (u.get("hp", 0.0) / maxhp) if maxhp else 0.0
                    f[14] = u.get("frac", 1.0)
                    f[23:49] = self.static.get(bp, self.zero_static)
                    bp_idx[i] = self.bp_index.get(bp, 0)
                    heading = u.get("heading") or 0.0
                    f[49], f[50] = math.sin(heading), math.cos(heading)
                # velocity from the last decision's position of the same id: a radar blip that moves fast is
                # probably an aircraft or a scout, and the network can see it without having to work it out
                uid = str(u["id"])
                prev = self.last_pos.get(uid)
                if prev is not None and 0 < tick - prev[2] <= 60:
                    dt = (tick - prev[2]) / 10.0
                    f[51], f[52], f[53] = (x - prev[0]) / dt / 20.0, (z - prev[1]) / dt / 20.0, 1.0
                seen_now[uid] = (x, z, tick)
                if t == T_OWN:
                    f[15] = float(bool(u.get("idle")))
                    f[16] = float(u.get("work") or 0.0)
                    f[17] = min(float(u.get("queue") or 0), 10.0) / 10.0
                    f[18] = float(u.get("focus") is not None)
                    order = u.get("order")
                    if order is not None:
                        orders[i] = min(int(order) + 1, N_ORDERS - 1)
                    dest = u.get("dest")
                    if dest:
                        f[19], f[20], f[21] = 1.0, (dest[0] - x) / W, (dest[1] - z) / H
                    since = self.commanded.get(str(u["id"]))
                    f[22] = 1.0 if since is None else min(1.0, (tick - since) / 600.0)
                    given = self.last_order.get(str(u["id"]))
                    if given is not None:
                        last[i] = given[0]
                        if given[1] is not None:
                            last_feat[i] = ((given[1][0] - x) / W, (given[1][1] - z) / H, 1.0)
                    # FAF inflates an upgrading extractor's production value while mass stalls (the engine
                    # scales it back down), so the readings are capped
                    f[55] = min(1.0, math.log1p(float(u.get("pm") or 0.0)) / 3.5)
                    f[56] = min(1.5, math.log1p(float(u.get("pe") or 0.0)) / 6.0)
                    f[57] = min(1.5, math.log1p(float(u.get("ce") or 0.0)) / 6.0)
                    f[58] = len(u.get("enh") or []) / 4.0
                    f[59] = min(float(u.get("tml") or 0), 10.0) / 5.0
                    f[60] = min(float(u.get("nukes") or 0), 4.0) / 2.0
                    f[61] = min(float(u.get("cargo") or 0), 20.0) / 10.0
                    fire = int(u.get("fire") or 0)
                    if 0 <= fire <= 2:
                        f[62 + fire] = 1.0
                    toggles = u.get("toggles") or []
                    f[65] = float(bool(toggles))
                    f[66] = len(u.get("toggles_on") or []) / max(1, len(toggles))
                    f[68] = float(bool(u.get("paused")))
                    f[69] = float(u.get("shield") or 0.0)
                    if u.get("can_build") is not None or self._builder(bp):
                        builds[i] = self.can_build_of(u)
                        f[67] = math.log1p(len(builds[i])) / 4.0
            elif t in (T_MASS, T_HYDRO):       # slots 13-15 mean free / ours / the enemy's for spots
                ids.append(None)
                who = u["who"]
                f[13] = 1.0 if who == 0 and not u["claimed"] else 0.0     # free
                f[14] = 1.0 if who == 1 else 0.0                          # ours
                f[15] = 1.0 if who == 2 else 0.0                          # enemy's
            else:   # wreck
                ids.append(str(u.get("id")))
                f[13] = math.log1p(float(u.get("mass") or 0)) / 8.0
                f[14] = math.log1p(float(u.get("energy") or 0)) / 10.0

        # how many own buildings touch each own building (engine adjacency rule, vectorised)
        s_idx = [i for i, (t, u) in enumerate(rows) if t == T_OWN and self.is_structure.get(u.get("bp"), False)
                 and (u.get("frac") or 1.0) >= 1.0]
        if len(s_idx) > 1:
            r = np.array([skirt_rect(self.bps.get(rows[i][1]["bp"], {}), rows[i][1]["x"], rows[i][1]["z"])
                          for i in s_idx], np.float32)
            x0, z0, x1, z1 = (r[:, k] for k in range(4))
            xe = (np.abs(x0[:, None] - x1[None, :]) < 1) | (np.abs(x1[:, None] - x0[None, :]) < 1)
            ze = (np.abs(z0[:, None] - z1[None, :]) < 1) | (np.abs(z1[:, None] - z0[None, :]) < 1)
            zin = ((z0[:, None] >= z0[None, :]) & (z1[None, :] >= z1[:, None])) | \
                  ((z0[None, :] >= z0[:, None]) & (z1[:, None] >= z1[None, :]))
            xin = ((x0[:, None] >= x0[None, :]) & (x1[None, :] >= x1[:, None])) | \
                  ((x0[None, :] >= x0[:, None]) & (x1[:, None] >= x1[None, :]))
            adj = np.where(xe, zin, ze & xin)
            np.fill_diagonal(adj, False)
            feats[s_idx, 54] = adj.sum(1) / 4.0
        # positions for next decision's velocities (units out of sight for a while are forgotten)
        self.last_pos = {k: v for k, v in self.last_pos.items() if tick - v[2] <= 60}
        self.last_pos.update(seen_now)
        # memory maps
        dt = max(0, tick - self.last_tick)
        self.last_tick = tick
        self.enemy_memory *= math.exp(-dt / 1800.0)
        self._mark_seen_many([u["x"] for u in own], [u["z"] for u in own],
                             [float(self.bps.get(u.get("bp"), {}).get("vision") or 20.0) for u in own], tick)
        planes = np.zeros((len(MAP_CHANNELS), GRID, GRID), np.float32)
        planes[0:5] = self.static_map
        for t, u in rows:
            if t > T_RADAR:
                continue
            gx, gz = self.kit.cell(u["x"], u["z"])
            structure = self.is_structure.get(u.get("bp"), False)
            if t == T_OWN:
                planes[6 if structure else 5, gz, gx] += 1.0
            elif t == T_ENEMY:
                planes[8 if structure else 7, gz, gx] += 1.0
                self.enemy_memory[gz, gx] = max(self.enemy_memory[gz, gx],
                                                math.log1p(float(self.bps.get(u.get("bp"), {}).get("mass") or 0)))
            elif t == T_GHOST:
                planes[9, gz, gx] += 1.0
            else:
                planes[10, gz, gx] += 1.0
        planes[5:11] = np.log1p(planes[5:11])
        planes[11] = np.clip((tick - self.last_looked) / 3000.0, 0.0, 1.0)
        planes[12] = self.enemy_memory / 8.0
        planes[13:15] = self.start_maps
        planes[15] = self.buildable
        planes[16:21] = self._danger(rows)

        # scalars
        me_army = next((a for a in obs.get("armies", []) if a.get("index") == self.me), {})
        eco = me_army.get("eco") or {}
        mass, energy = eco.get("mass") or {}, eco.get("energy") or {}
        lg = math.log1p
        n_own_mobile = sum(1 for u in own if self.is_mobile.get(u.get("bp"), False))
        acu = next((u for u in own if "COMMAND" in cats(self.bps.get(u.get("bp"), {}))), None)
        engineers = sum(1 for u in own if "ENGINEER" in cats(self.bps.get(u.get("bp"), {})))
        factories = sum(1 for u in own if self.kit.role.get(u.get("bp")) in ("land_factory", "air_factory", "naval_factory"))
        combat = sum(1 for u in own if self.is_mobile.get(u.get("bp"), False)
                     and (self.bps.get(u.get("bp"), {}).get("weapons")) and "ENGINEER" not in cats(self.bps.get(u.get("bp"), {})))
        focus = self._focus(rows, own, acu)
        scalars = np.array([
            tick / 18000.0,
            (mass.get("stored") or 0.0) / 1000.0, (mass.get("max") or 0.0) / 1000.0, (mass.get("income") or 0.0) / 50.0,
            (mass.get("requested") or 0.0) / 50.0, (mass.get("trend") or 0.0) / 20.0,
            (energy.get("stored") or 0.0) / 10000.0, (energy.get("max") or 0.0) / 10000.0,
            (energy.get("income") or 0.0) / 500.0, (energy.get("requested") or 0.0) / 500.0,
            (energy.get("trend") or 0.0) / 200.0,
            float((mass.get("stored") or 0) < 20 and (mass.get("requested") or 0) > (mass.get("income") or 0)),
            float((energy.get("stored") or 0) < 100 and (energy.get("requested") or 0) > (energy.get("income") or 0)),
            float(me_army.get("unitCount") or 0) / max(1.0, float(me_army.get("unitCap") or 1.0)),
            float(acu is not None), (acu["hp"] / acu["maxhp"]) if acu and acu.get("maxhp") else 0.0,
            lg(len(own)), lg(n_own_mobile), lg(engineers), lg(factories), lg(combat),
            lg(len(enemy)), lg(len(ghost)), lg(len(radar)), lg(len(self.wrecks)),
            lg(sum(1 for s in spots if s["who"] == 0 and not s["claimed"])),
            *self._tech(own, enemy + ghost),
            # the map's scale: positions are fractions of the map, so without these 5 and 20 km look alike
            math.log2(max(self.width, self.height) / 256.0) / 2.0,
            math.hypot(self.start[0] - self.enemy_start[0], self.start[1] - self.enemy_start[1]) / 500.0,
            *self.success,
        ], np.float32)
        assert len(scalars) == N_SCALARS, f"{len(scalars)} scalars, the network expects {N_SCALARS}"

        counts: dict[str, int] = {}
        for u in own:
            counts[u.get("bp")] = counts.get(u.get("bp"), 0) + 1
        masks = self._masks(rows, spots, builds, counts)
        priv = obs.get("privileged") or {}
        mine = self._priv(priv.get(str(self.me)) or {})
        theirs = np.zeros(len(PRIVILEGED), np.float32)
        for idx in self.enemy_indices:
            theirs += self._priv(priv.get(str(idx)) or {})
        clean = lambda a: np.clip(np.nan_to_num(a, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)  # noqa: E731
        out = {
            "tokens": clean(feats), "token_bp": bp_idx, "token_order": orders, "token_type": types,
            "token_last": last, "token_last_feat": last_feat,
            "map": clean(planes), "scalars": clean(scalars),
            "privileged": clean(np.concatenate([mine, theirs])), "belief": clean(theirs),
            # fine pixels per cell side (fa_env.tools.fine_split: 1 on a 5 km map) and the map's size
            "fine": np.array([fine_split(self.width / GRID), fine_split(self.height / GRID),
                              self.width, self.height], np.float32),
            "focus": focus,
            **masks,
        }
        out["ids"] = ids
        return out

    def _focus(self, rows: list, own: list, acu: dict | None) -> np.ndarray:
        """The close-ups' centres: our commander; our army (the mass-weighted centre of our mobile armed units,
        engineers and commanders aside); the fight (the seen enemy unit nearest to any of ours)."""
        out = np.zeros((N_FOCUS, 3), np.float32)
        W, H = self.width, self.height
        if acu is not None:
            out[0] = (acu["x"] / W, acu["z"] / H, 1.0)
        army = [u for u in own if self.is_mobile.get(u.get("bp"), False) and self.bps.get(u.get("bp"), {}).get("weapons")
                and not ({"ENGINEER", "COMMAND"} & set(cats(self.bps.get(u.get("bp"), {}))))]
        if army:
            w = np.array([max(1.0, float(self.bps.get(u.get("bp"), {}).get("mass") or 0)) for u in army], np.float32)
            xs = np.array([u["x"] for u in army], np.float32)
            zs = np.array([u["z"] for u in army], np.float32)
            out[1] = (float((xs * w).sum() / w.sum()) / W, float((zs * w).sum() / w.sum()) / H, 1.0)
        seen = [u for t, u in rows if t == T_ENEMY]
        if seen and own:
            ex = np.array([u["x"] for u in seen], np.float32)
            ez = np.array([u["z"] for u in seen], np.float32)
            ox = np.array([u["x"] for u in own], np.float32)
            oz = np.array([u["z"] for u in own], np.float32)
            d = np.full(len(seen), np.inf, np.float32)
            for i in range(0, len(own), 256):                     # in blocks: a few hundred by a few hundred
                d = np.minimum(d, ((ex[:, None] - ox[None, i:i + 256]) ** 2 +
                                   (ez[:, None] - oz[None, i:i + 256]) ** 2).min(1))
            j = int(d.argmin())
            out[2] = (ex[j] / W, ez[j] / H, 1.0)
        return out

    def _tech(self, own: list, enemy: list) -> list[float]:
        """Tech state: highest own factory tech, own T2 and T3 headquarters (support factories need one), own
        experimentals; highest enemy tech seen and enemy experimentals seen."""
        def tech(u):
            return float(self.bps.get(u.get("bp"), {}).get("tech") or 0)

        own_fac = [u for u in own if "FACTORY" in cats(self.bps.get(u.get("bp"), {})) and (u.get("frac") or 1) >= 1]
        hq = [u for u in own_fac if "RESEARCH" in cats(self.bps.get(u.get("bp"), {}))]
        exp = lambda group: sum(1 for u in group if "EXPERIMENTAL" in cats(self.bps.get(u.get("bp"), {})))  # noqa: E731
        return [max([tech(u) for u in own_fac] or [0.0]) / 3.0,
                math.log1p(sum(1 for u in hq if tech(u) == 2)), math.log1p(sum(1 for u in hq if tech(u) == 3)),
                math.log1p(exp(own)),
                max([min(tech(u), 4.0) for u in enemy] or [0.0]) / 3.0, math.log1p(exp(enemy))]

    def _builder(self, bp_id: str | None) -> bool:
        c = cats(self.bps.get(bp_id, {}))
        return "ENGINEER" in c or "FACTORY" in c

    def _priv(self, p: dict) -> np.ndarray:
        v = []
        for key, scale in zip(PRIVILEGED, PRIV_SCALE):
            x = float(p.get(key) or 0.0)
            v.append(math.log1p(max(0.0, x)) if scale is None else x / scale)
        return np.asarray(v, np.float32)

    @staticmethod
    def _flags(bp: dict) -> dict:
        c = cats(bp)
        commander = "COMMAND" in c
        mobile = "MOBILE" in c
        return {"structure": "STRUCTURE" in c, "air_transport": "TRANSPORTATION" in c and "AIR" in c,
                "transport": "TRANSPORTATION" in c, "commander": commander, "sub": "SUBCOMMANDER" in c,
                "mobile": mobile, "builder": mobile and ("ENGINEER" in c or commander or "SUBCOMMANDER" in c),
                "factory": "FACTORY" in c, "weapons": bool(bp.get("weapons")), "submersible": "SUBMERSIBLE" in c,
                "loadable": "LAND" in c and "EXPERIMENTAL" not in c}

    def _cover(self, xs: list, zs: list, radii: list, values: list, reduce: str = "sum") -> np.ndarray:
        """[GRID, GRID]: each cell's sum (or max) of `values` over the circles that contain its centre."""
        out = np.zeros((GRID, GRID), np.float32)
        if not xs:
            return out
        x, z = np.asarray(xs, np.float32), np.asarray(zs, np.float32)
        r2, v = np.asarray(radii, np.float32) ** 2, np.asarray(values, np.float32)
        dx2 = (self.cell_cx.astype(np.float32)[None, :] - x[:, None]) ** 2                   # [N,GRIDx]
        dz2 = (self.cell_cz.astype(np.float32)[None, :] - z[:, None]) ** 2                   # [N,GRIDz]
        inside = dz2[:, :, None] + dx2[:, None, :] <= r2[:, None, None]                      # [N,GRIDz,GRIDx]
        if reduce == "max":
            return (inside * v[:, None, None]).max(0)
        return (inside * v[:, None, None]).sum(0)

    def _danger(self, rows) -> np.ndarray:
        """[5, GRID, GRID]: enemy ground fire, enemy anti-air, own ground fire (log1p of the damage per second
        reaching a cell, /8), enemy shield cover, own shield cover (1 inside a bubble)."""
        out = np.zeros((5, GRID, GRID), np.float32)
        acc = {k: ([], [], [], []) for k in ("eg", "ea", "og", "es", "os")}

        def add(key, u, r, v):
            a = acc[key]
            a[0].append(float(u["x"]))
            a[1].append(float(u["z"]))
            a[2].append(r)
            a[3].append(v)

        for t, u in rows:
            if t not in (T_OWN, T_ENEMY, T_GHOST):
                continue
            prof = self.threat.get(u.get("bp"))
            if prof is None:
                continue
            ground, air, shield = prof
            own = t == T_OWN
            for dps, rng in ground:
                add("og" if own else "eg", u, rng, dps)
            if not own:
                for dps, rng in air:
                    add("ea", u, rng, dps)
            if shield > 0:
                add("os" if own else "es", u, shield, 1.0)
        for k, key in enumerate(("eg", "ea", "og")):
            out[k] = np.log1p(self._cover(*acc[key])) / 8.0
        out[3] = self._cover(*acc["es"], reduce="max")
        out[4] = self._cover(*acc["os"], reduce="max")
        return out

    def _mark_seen_many(self, xs: list, zs: list, radii: list, tick: int) -> None:
        """The vision circles of many units on the last-looked map at once: the same cells as `_mark_seen` for
        each (its box and its centre test), without a Python call and two aranges per unit."""
        if not xs:
            return
        cw, ch = self.width / GRID, self.height / GRID
        x, z, r = np.asarray(xs, np.float64), np.asarray(zs, np.float64), np.asarray(radii, np.float64)
        x0 = np.maximum(0, ((x - r) / cw).astype(np.int64))
        x1 = np.minimum(GRID - 1, ((x + r) / cw).astype(np.int64))
        z0 = np.maximum(0, ((z - r) / ch).astype(np.int64))
        z1 = np.minimum(GRID - 1, ((z + r) / ch).astype(np.int64))
        ix = np.arange(GRID)
        in_x = (ix[None, :] >= x0[:, None]) & (ix[None, :] <= x1[:, None])                 # [N,GRID]
        in_z = (ix[None, :] >= z0[:, None]) & (ix[None, :] <= z1[:, None])
        dx2 = (self.cell_cx[None, :] - x[:, None]) ** 2                                      # [N,GRID]
        dz2 = (self.cell_cz[None, :] - z[:, None]) ** 2
        inside = (dx2[:, None, :] + dz2[:, :, None] <= (r * r)[:, None, None]) \
            & in_z[:, :, None] & in_x[:, None, :]                                            # [N,GRIDz,GRIDx]
        self.last_looked[inside.any(0)] = tick

    def _mark_seen(self, x: float, z: float, radius: float, tick: int) -> None:
        cw, ch = self.width / GRID, self.height / GRID
        x0, x1 = max(0, int((x - radius) / cw)), min(GRID - 1, int((x + radius) / cw))
        z0, z1 = max(0, int((z - radius) / ch)), min(GRID - 1, int((z + radius) / ch))
        if x1 < x0 or z1 < z0:
            return
        xs = (np.arange(x0, x1 + 1) + 0.5) * cw
        zs = (np.arange(z0, z1 + 1) + 0.5) * ch
        inside = (xs[None, :] - x) ** 2 + (zs[:, None] - z) ** 2 <= radius * radius
        block = self.last_looked[z0:z1 + 1, x0:x1 + 1]
        block[inside] = tick

    # ------------------------------------------------------------------ what may be ordered
    def _cand_features(self, bp_id: str, count: int) -> np.ndarray:
        bp = self.bps.get(bp_id, {})
        f = np.zeros(N_CAND_FEATURES, np.float32)
        f[:N_STATIC] = self.static.get(bp_id, self.zero_static)
        f[N_STATIC] = math.log1p(float(bp.get("buildTime") or 0)) / 10.0
        f[N_STATIC + 1] = math.log1p(float(bp.get("energy") or 0)) / 12.0
        f[N_STATIC + 2] = math.log1p(count) / 3.0
        f[N_STATIC + 3 + self.place_class.get(bp_id, CAND_MOBILE)] = 1.0
        return f

    def _enh_features(self, e: dict, replaces: bool) -> np.ndarray:
        f = np.zeros(N_ENH_FEATURES, np.float32)
        f[0] = math.log1p(float(e.get("mass") or 0)) / 10.0
        f[1] = math.log1p(float(e.get("energy") or 0)) / 12.0
        f[2] = math.log1p(float(e.get("time") or 0)) / 10.0
        f[3] = math.log1p(float(e.get("health") or 0)) / 10.0
        f[4] = math.log1p(float(e.get("buildRate") or 0)) / 5.0
        slot = e.get("slot")
        f[5 + (ENH_SLOTS.index(slot) if slot in ENH_SLOTS else len(ENH_SLOTS))] = 1.0
        f[5 + len(ENH_SLOTS) + 1] = float(bool(e.get("requires")))
        f[5 + len(ENH_SLOTS) + 2] = float(replaces)
        return f

    def _masks(self, rows, spots, builds: dict, counts: dict) -> dict:
        n = len(rows)
        V = len(VERBS)
        vi = V_INDEX
        subject = np.zeros((V, n), bool)
        target = np.zeros((len(TARGET_ROWS), n), bool)
        mask_silo = np.zeros((n, len(SILO_KINDS)), bool)
        mask_toggle = np.zeros((n, N_TOGGLES), bool)
        anchor = np.zeros(n, bool)
        free = {"mass": np.zeros((GRID, GRID), bool), "hydro": np.zeros((GRID, GRID), bool)}
        for s in spots:
            if s["who"] == 0 and not s["claimed"]:
                gx, gz = self.kit.cell(s["x"], s["z"])
                free["mass" if s["type"] == "mass" else "hydro"][gz, gx] = True
        land = self.buildable & self.move_cells
        cell_build = np.stack([land, self.water & self.move_cells, free["mass"], free["hydro"]])
        class_ok = cell_build.reshape(len(PLACEMENT_CLASSES), -1).any(1)
        # blueprint candidates: the union of what our builders can build now
        cand_set = set()
        for i, lst in builds.items():
            cand_set.update(b for b in lst if b in self.bps)
        cand_ids = sorted(cand_set, key=lambda b: self.bp_index.get(b, 0))[:MAX_CANDIDATES]
        cand_pos = {b: j for j, b in enumerate(cand_ids)}
        cand_kind = np.array([self.place_class.get(b, CAND_MOBILE) for b in cand_ids], np.int64)
        can_build = []
        for i, lst in builds.items():
            for b in lst:
                j = cand_pos.get(b)
                if j is None:
                    continue
                k = cand_kind[j]
                if k != CAND_MOBILE and not class_ok[k]:
                    continue                    # nowhere to put it (no free deposit, no water)
                can_build.append((i, j))
        enh_keys: list[tuple[str, str]] = []
        enh_feat, can_enh = [], []
        transports = False
        for i, (t, u) in enumerate(rows):
            if t == T_ENEMY:
                target[0, i] = True
            elif t == T_WRECK:
                target[3, i] = True
            if t != T_OWN:
                continue
            target[1, i] = True
            bp_id = u.get("bp")
            bp = self.bps.get(bp_id, {})
            fl = self.flags.get(bp_id, self.zero_flags)
            complete = (1.0 if u.get("frac") is None else float(u["frac"])) >= 1.0
            if not complete or (u.get("maxhp") and u.get("hp", 0) < u["maxhp"]):
                target[2, i] = True             # damaged or unfinished: something to repair
            if not complete:
                continue
            if fl["structure"]:
                anchor[i] = True
            if fl["air_transport"]:
                target[4, i] = True
                transports = True
            commander = fl["commander"]
            mobile = fl["mobile"]
            builder = fl["builder"]
            factory = fl["factory"]
            if builder:
                subject[vi["assist"], i] = True
                subject[vi["repair"], i] = True
                if not commander:
                    subject[vi["reclaim"], i] = True
                    subject[vi["capture"], i] = True
            if bp.get("upgradesTo") in self.bps and u.get("idle"):
                subject[vi["upgrade"], i] = True
            if factory:
                subject[vi["rally"], i] = True
            if factory or (builder and not commander) or (fl["structure"] and (u.get("ce") or u.get("work"))) \
                    or u.get("paused"):
                subject[vi["pause"], i] = True
            if u.get("toggles"):
                for bit in u["toggles"]:
                    if 0 <= int(bit) < N_TOGGLES:
                        mask_toggle[i, int(bit)] = True
                subject[vi["toggle"], i] = mask_toggle[i].any()
            kinds = silo_kinds(u, bp)
            for k, name in enumerate(SILO_KINDS):
                mask_silo[i, k] = name in kinds
            subject[vi["silo_build"], i] = mask_silo[i].any()
            if u.get("tml"):
                subject[vi["tactical"], i] = True
            if u.get("nukes"):
                subject[vi["nuke"], i] = True
            if fl["transport"] and (u.get("cargo") or 0) > 0:
                subject[vi["unload"], i] = True
            if commander or fl["sub"]:
                installed = set(u.get("enh") or [])
                by_name = {e["name"]: e for e in bp.get("enhancements") or []}
                for name in available_enhancements(u, bp)[:MAX_ENHANCEMENTS]:
                    key = (bp_id, name)
                    if key not in enh_keys:
                        if len(enh_keys) >= MAX_ENHANCEMENTS:
                            continue
                        e = by_name[name]
                        replaces = any(by_name.get(n, {}).get("slot") == e.get("slot") for n in installed)
                        enh_keys.append(key)
                        enh_feat.append(self._enh_features(e, replaces))
                    can_enh.append((i, enh_keys.index(key)))
                subject[vi["enhance"], i] = any(p[0] == i for p in can_enh)
            if commander:                       # its own commands; never in a set but base work
                subject[vi["stop"], i] = True    # explicit cancellation before a replacement build
                subject[vi["commander_move"], i] = True
                subject[vi["commander_attack_move"], i] = True
                subject[vi["overcharge"], i] = True
            elif mobile:
                for v in ("move", "attack_move", "patrol", "stop"):
                    subject[vi[v], i] = True
                if fl["weapons"]:
                    subject[vi["attack"], i] = True
                if fl["submersible"]:
                    subject[vi["dive"], i] = True
                if fl["loadable"]:
                    subject[vi["load"], i] = True
            if fl["weapons"] and not commander:
                subject[vi["fire_state"], i] = True
        if not transports:
            subject[vi["load"]] = False
        # what a builder can build: units are produced; structures are built by mobile builders, and are a
        # factory's upgrades (T2 headquarters or support factory), chosen with the same pointer
        for i, j in can_build:
            u = rows[i][1]
            if cand_kind[j] == CAND_MOBILE:
                if (u.get("queue") or 0) < 5:
                    subject[vi["produce"], i] = True
            elif self.is_mobile.get(u.get("bp"), False):
                subject[vi["build"], i] = True
            elif u.get("idle"):
                subject[vi["upgrade"], i] = True
        verb = np.zeros(V, bool)
        verb[vi["end"]] = True
        for v in VERBS[1:]:
            if not subject[vi[v]].any():
                continue
            need = TARGET_VERBS.get(v)
            if need is not None and not target[TARGET_ROWS.index(need)].any():
                continue
            verb[vi[v]] = True
        pairs = lambda p: np.asarray(p, np.int64).reshape(-1, 2)  # noqa: E731
        return {
            "mask_verb": verb, "mask_subject": subject, "mask_target": target, "mask_anchor": anchor,
            "mask_silo": mask_silo, "mask_toggle": mask_toggle,
            "mask_build_cell": cell_build, "mask_move_cell": self.move_cells.copy(),
            "cand_bp": np.array([self.bp_index.get(b, 0) for b in cand_ids], np.int64),
            "cand_feat": (np.stack([self._cand_features(b, counts.get(b, 0)) for b in cand_ids])
                          if cand_ids else np.zeros((0, N_CAND_FEATURES), np.float32)),
            "cand_kind": cand_kind, "can_build": pairs(can_build), "cand_ids": cand_ids,
            "enh_idx": np.array([self.enh_vocab.get(name, 0) for _, name in enh_keys], np.int64),
            "enh_feat": np.stack(enh_feat) if enh_feat else np.zeros((0, N_ENH_FEATURES), np.float32),
            "can_enh": pairs(can_enh), "enh_names": [name for _, name in enh_keys],
        }


def _resize(grid: np.ndarray, size: int) -> np.ndarray:
    """Area-average (or nearest, when upsampling) resample of a square grid to size x size."""
    n = grid.shape[0]
    if n >= size and n % size == 0:
        k = n // size
        return grid.reshape(size, k, size, k).mean(axis=(1, 3))
    idx = (np.arange(size) * n // size).clip(0, n - 1)
    return grid[np.ix_(idx, idx)]


assert SINGLE_SUBJECT | SET_SUBJECT == set(VERBS) - {"end"}
assert N_OPTIONS >= N_TOGGLES
