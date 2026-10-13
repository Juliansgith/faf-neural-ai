"""Turn bridge observations into fixed-shape numpy arrays for a neural network.

A starting point in the spirit of AlphaStar's inputs, not a tuned design:

  spatial(obs)  [C, S, S] float32 map planes: terrain, water, deposits, and per-cell counts / health
                of own, allied, enemy and radar-only units, split into mobile units and structures
  scalars(obs)  [F] float32: economy, game time, unit counts
  units(obs)    [N, U] float32 per-unit rows (padded, with a mask) plus int64 blueprint indices for an
                embedding table; own units first, then visible enemies

    feats = Featurizer(info, size=64, max_units=256)
    x_map, x_scalar, (x_units, bp_ids, mask) = feats(obs)
"""
from __future__ import annotations

import numpy as np

SPATIAL_CHANNELS = (
    "terrain", "water", "mass_deposit", "hydro_deposit",
    "own_mobile", "own_structure", "own_health",
    "ally_mobile", "ally_structure",
    "enemy_mobile", "enemy_structure", "enemy_health", "enemy_radar",
)
UNIT_FEATURES = (
    "own", "ally", "enemy", "x", "z", "hp_frac", "maxhp_log", "build_frac", "idle", "work", "queue",
    "is_structure", "is_air", "is_naval", "heading_sin", "heading_cos",
)


class Featurizer:
    def __init__(self, info: dict, size: int = 64, max_units: int = 256):
        self.size = size
        self.max_units = max_units
        m = info["map"]
        self.width, self.height = float(m["width"]), float(m["height"])

        # blueprint vocabulary: index 0 is padding / unknown (radar contacts)
        self.bps = {bp["id"]: bp for bp in info.get("blueprints", [])}
        self.bp_index = {bp_id: i + 1 for i, bp_id in enumerate(sorted(self.bps))}
        self.vocab_size = len(self.bp_index) + 1
        self._structure = {i for i, bp in self.bps.items() if "STRUCTURE" in (bp.get("categories") or [])}
        self._air = {i for i, bp in self.bps.items() if "AIR" in (bp.get("categories") or [])}
        self._naval = {i for i, bp in self.bps.items() if "NAVAL" in (bp.get("categories") or [])}

        self.static = np.zeros((4, size, size), np.float32)
        hm = m.get("heightmap") or {}
        n = hm.get("n")
        if n:
            terrain = np.asarray(hm["terrain"], np.float32).reshape(n, n)
            surface = np.asarray(hm["surface"], np.float32).reshape(n, n)
            self.static[0] = _resize(terrain / max(1.0, float(terrain.max())), size)
            self.static[1] = _resize((surface > terrain + 0.01).astype(np.float32), size)
        for d in m.get("deposits", []):
            gx, gz = self._cell(d["x"], d["z"])
            self.static[2 if d["type"] == "mass" else 3, gz, gx] = 1.0

    def _cell(self, x: float, z: float) -> tuple[int, int]:
        gx = int(np.clip(x / self.width * self.size, 0, self.size - 1))
        gz = int(np.clip(z / self.height * self.size, 0, self.size - 1))
        return gx, gz

    def spatial(self, obs: dict) -> np.ndarray:
        planes = np.zeros((len(SPATIAL_CHANNELS), self.size, self.size), np.float32)
        planes[:4] = self.static
        relation = {a["index"]: a["relation"] for a in obs.get("armies", [])}
        for u in obs.get("units", []):
            gx, gz = self._cell(u["x"], u["z"])
            rel = relation.get(u.get("army"), "enemy")
            if u.get("radar"):
                planes[12, gz, gx] += 1.0
                continue
            structure = u.get("bp") in self._structure
            hp = u.get("hp", 0.0) / 1000.0
            if rel == "self":
                planes[5 if structure else 4, gz, gx] += 1.0
                planes[6, gz, gx] += hp
            elif rel == "ally":
                planes[8 if structure else 7, gz, gx] += 1.0
            else:
                planes[10 if structure else 9, gz, gx] += 1.0
                planes[11, gz, gx] += hp
        planes[4:] = np.log1p(planes[4:])
        return planes

    def scalars(self, obs: dict) -> np.ndarray:
        me = next((a for a in obs.get("armies", []) if a["index"] == obs.get("me")), {})
        eco = me.get("eco", {})
        mass, energy = eco.get("mass", {}), eco.get("energy", {})
        own = sum(1 for u in obs.get("units", []) if u.get("army") == obs.get("me"))
        seen = sum(1 for u in obs.get("units", []) if u.get("army") != obs.get("me"))
        return np.array([
            obs.get("time", 0.0) / 1800.0,
            mass.get("stored", 0.0) / 1000.0, (mass.get("max") or 0.0) / 1000.0,
            mass.get("income", 0.0) / 50.0, mass.get("requested", 0.0) / 50.0,
            energy.get("stored", 0.0) / 10000.0, (energy.get("max") or 0.0) / 10000.0,
            energy.get("income", 0.0) / 500.0, energy.get("requested", 0.0) / 500.0,
            np.log1p(own), np.log1p(seen),
        ], np.float32)

    def units(self, obs: dict):
        me = obs.get("me")
        relation = {a["index"]: a["relation"] for a in obs.get("armies", [])}
        rows = sorted(obs.get("units", []), key=lambda u: (u.get("army") != me, u.get("radar", False)))
        rows = rows[: self.max_units]
        feats = np.zeros((self.max_units, len(UNIT_FEATURES)), np.float32)
        bp_ids = np.zeros(self.max_units, np.int64)
        mask = np.zeros(self.max_units, bool)
        for i, u in enumerate(rows):
            rel = relation.get(u.get("army"), "enemy")
            bp = u.get("bp")
            heading = u.get("heading", 0.0) or 0.0
            maxhp = u.get("maxhp", 0.0) or 0.0
            feats[i] = (
                rel == "self", rel == "ally", rel not in ("self", "ally"),
                u["x"] / self.width, u["z"] / self.height,
                (u.get("hp", 0.0) / maxhp) if maxhp else 0.0, np.log1p(maxhp) / 12.0,
                u.get("frac", 1.0), float(bool(u.get("idle"))), u.get("work", 0.0) or 0.0,
                min(u.get("queue", 0) or 0, 20) / 20.0,
                bp in self._structure, bp in self._air, bp in self._naval,
                np.sin(heading), np.cos(heading),
            )
            bp_ids[i] = self.bp_index.get(bp, 0)
            mask[i] = True
        return feats, bp_ids, mask

    def __call__(self, obs: dict):
        return self.spatial(obs), self.scalars(obs), self.units(obs)


def _resize(grid: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resample of a square grid to size x size."""
    n = grid.shape[0]
    idx = (np.arange(size) * n // size).clip(0, n - 1)
    return grid[np.ix_(idx, idx)]
