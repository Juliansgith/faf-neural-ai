"""A small synthetic game for observer tests: a 5 km map (or larger), our army 1, the enemy 2 and neutral
civilians 3, a tank, an engineer, a wall and a power generator."""
from fa_env.tools import ToolKit
from fa_env.tools_obs import ToolsObserver

TANK = {"id": "tank", "categories": ["MOBILE", "LAND", "DIRECTFIRE"], "mass": 50, "health": 300,
        "weapons": [{"label": "Gun", "rangeCategory": "UWRC_DirectFire", "range": 20, "damage": 10, "rof": 1}]}
BPS = {
    "tank": TANK,
    "eng": {"id": "eng", "categories": ["MOBILE", "LAND", "ENGINEER"], "mass": 50, "health": 100},
    "wall": {"id": "wall", "categories": ["STRUCTURE", "WALL"], "mass": 2, "health": 100},
    "pgen": {"id": "pgen", "categories": ["STRUCTURE", "ENERGYPRODUCTION"], "mass": 75, "health": 500},
}
ME, ENEMY, CIVILIAN = 1, 2, 3


def info(size: float = 256.0) -> dict:
    return {"map": {"width": size, "height": size, "deposits": []},
            "armies": [{"index": ME, "start": [20.0, 20.0], "relation": "self"},
                       {"index": ENEMY, "start": [size - 20.0, size - 20.0], "relation": "enemy"},
                       {"index": CIVILIAN, "start": [0.0, 0.0], "relation": "neutral", "civilian": True}]}


def observer(size: float = 256.0) -> ToolsObserver:
    i = info(size)
    return ToolsObserver(i, BPS, ME, ToolKit(i, BPS))


def unit(uid, bp, x, z, army=ME, **kw) -> dict:
    return {"id": uid, "bp": bp, "x": float(x), "z": float(z), "army": army, "hp": 100.0, "maxhp": 100.0,
            "frac": 1.0, **kw}


def obs(units, tick=0) -> dict:
    return {"tick": tick, "units": units,
            "armies": [{"index": ME, "relation": "self"}, {"index": ENEMY, "relation": "enemy"},
                       {"index": CIVILIAN, "relation": "neutral", "civilian": True}]}


def token_of(x: dict, uid: str) -> int:
    return x["ids"].index(uid)
