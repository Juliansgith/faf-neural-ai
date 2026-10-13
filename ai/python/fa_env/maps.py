"""The map pool for general (multi-map, both-sides) games: built-in 1v1-able land maps of 5, 10 and 20 km.

Each entry lists the start-slot pairs a 1v1 uses (GameConfig.slots = (ours, the AI's)): both ways round on
2-player maps, the two farthest-apart pairs (both ways) on the 4-player 20 km maps, whose other slots stay empty.
Games get a time limit by map size: T3 armies need time to cross a bigger map.

    from fa_env.maps import MAP_POOL, sample_map
    map_id, slots, minutes = sample_map(random.Random(1))
"""
from __future__ import annotations

import random

MINUTES_BY_KM = {5: 60.0, 10: 75.0, 20: 90.0}

MAP_POOL: dict[str, dict] = {
    # 5 km, 2 players
    "SCMP_016": {"name": "Canis River", "km": 5, "slots": [(1, 2), (2, 1)]},
    "SCMP_012": {"name": "Theta Passage", "km": 5, "slots": [(1, 2), (2, 1)]},
    "SCMP_013": {"name": "Winter Duel", "km": 5, "slots": [(1, 2), (2, 1)]},
    "X1MP_001": {"name": "Crag Dunes", "km": 5, "slots": [(1, 2), (2, 1)]},
    "X1MP_002": {"name": "Williamson", "km": 5, "slots": [(1, 2), (2, 1)]},
    # 10 km, 2 players
    "SCMP_019": {"name": "Finn", "km": 10, "slots": [(1, 2), (2, 1)]},
    "SCMP_023": {"name": "Varga Pass", "km": 10, "slots": [(1, 2), (2, 1)]},
    # 20 km, 4 players: the farthest pairs
    "SCMP_003": {"name": "Drake", "km": 20, "slots": [(3, 4), (4, 3), (1, 2), (2, 1)]},
    "SCMP_004": {"name": "Emerald Crater", "km": 20, "slots": [(3, 4), (4, 3), (1, 2), (2, 1)]},
    "SCMP_006": {"name": "Ian", "km": 20, "slots": [(1, 2), (2, 1), (3, 4), (4, 3)]},
    "X1MP_007": {"name": "Strip Mine", "km": 20, "slots": [(1, 3), (3, 1), (2, 4), (4, 2)]},
}


def sample_map(rng: random.Random, pool: dict[str, dict] | None = None, by_size: bool = True) -> tuple[str, tuple, float]:
    """(map, slots, minutes). `by_size`: each size is equally likely (then a map of that size), so the few
    10 km maps are not drowned out by the five 5 km ones."""
    pool = pool or MAP_POOL
    if by_size:
        km = rng.choice(sorted({m["km"] for m in pool.values()}))
        map_id = rng.choice(sorted(k for k, m in pool.items() if m["km"] == km))
    else:
        map_id = rng.choice(sorted(pool))
    entry = pool[map_id]
    return map_id, tuple(rng.choice(entry["slots"])), MINUTES_BY_KM[entry["km"]]
