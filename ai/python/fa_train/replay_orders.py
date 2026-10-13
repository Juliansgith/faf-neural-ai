"""A FAF replay's player commands as the raw orders our recordings hold (the bridge's order capture format,
agentbridge.lua CaptureRecord), per army and game tick.

    orders = replay_orders(scfa_body)          # {army: [(tick, order), ...]}, ticks ascending

The replay's command stream (fafreplay) carries every player's IssueCommand / IssueFactoryCommand with the units,
the command type (EUnitCommandType), a target (a position or an entity id) and a blueprint; the ticks come from
its Advance messages. A unit's army is in its entity id (bits 20+: army index - 1). The orders go into a recorded
step's `sent`, where the encoder turns them into labels (ToolKit.commands_from_actions) as it does for any game.
A factory order for N units arrives as N identical commands (IssueBlueprintCommand issues `count` copies); they
become one order with count N. Not covered yet: pause, fire state, toggles and silo builds (they travel as Lua sim
callbacks or info pairs), changed production counts, and tactical missiles or overcharge aimed at the ground (the
encoder has no verb for them).
"""
from __future__ import annotations

from fafreplay import Parser, commands

# EUnitCommandType (moho/command/SSTICommandIssueData.h) -> (order type, argument kind)
COMMAND_TYPES = {
    1: ("stop", None), 2: ("move", "pos"), 3: ("dive", None), 4: ("move", "pos"),
    5: ("silo_build", None), 6: ("silo_build_nuke", None), 7: ("factory_build", "bp"), 8: ("build", "pos+bp"),
    9: ("guard", "target"), 10: ("attack", "target"), 11: ("attack", "target"), 12: ("nuke", "pos"),
    13: ("tactical", "target"), 15: ("guard", "target"), 16: ("patrol", "pos"), 18: ("patrol", "pos"),
    19: ("reclaim", "target"), 20: ("repair", "target"), 21: ("capture", "target"), 22: ("transport_load", "target"),
    24: ("transport_unload", "pos"), 25: ("transport_unload", "pos"), 27: ("upgrade", "bp"),
    28: ("enhance", "script"), 29: ("guard", "target"), 34: ("overcharge", "target"), 35: ("attack_move", "pos"),
    36: ("attack_move", "pos"),
}
FACTORY_TYPES = {2: "rally", 4: "rally", 35: "rally", 36: "rally", 16: "rally"}   # a factory's own order: its rally


def army_of(entity_id: int) -> int:
    return (int(entity_id) >> 20) + 1


def _text(v):
    return v.decode(errors="replace") if isinstance(v, bytes) else v


def to_order(cmd: dict, tick: int) -> dict | None:
    units = [str(u) for u in cmd.get("entity_ids") or []]
    if not units:
        return None
    ctype = int(cmd.get("type") or 0)
    target = cmd.get("target")
    if cmd["name"] == "IssueFactoryCommand" and ctype in FACTORY_TYPES:
        kind, arg = FACTORY_TYPES[ctype], "pos"
    elif ctype in COMMAND_TYPES:
        kind, arg = COMMAND_TYPES[ctype]
    else:
        return None
    order = {"type": kind, "units": units, "tick": tick}
    if kind == "silo_build_nuke":
        order.update(type="silo_build", nuke=True)
    pos = [float(target["x"]), float(target["z"])] if isinstance(target, dict) and "x" in target else None
    ent = str(target["id"]) if isinstance(target, dict) and "id" in target else None
    if arg in ("pos", "pos+bp"):
        if pos is None:
            return None
        order["pos"] = pos
    if arg in ("bp", "pos+bp"):
        bp = _text(cmd.get("blueprint"))
        if not bp:
            return None
        order["bp"] = bp.lower()
    if arg == "target":
        if ent is not None:
            order["target"] = ent
        elif pos is not None and kind == "attack":
            order.update(type="attack_ground", pos=pos)
        elif pos is not None and kind == "reclaim":
            order["pos"] = pos                       # an area reclaim
        else:
            return None
    if arg == "script":
        up = cmd.get("upgrades") or {}
        if _text(up.get("TaskName")) != "EnhanceTask" or not up.get("Enhancement"):
            return None
        order["name"] = _text(up["Enhancement"])
    if kind == "factory_build":
        order["count"] = 1
    return order


def replay_orders(scfa: bytes) -> dict[int, list[tuple[int, dict]]]:
    wanted = [commands.Advance, commands.IssueCommand, commands.IssueFactoryCommand]
    stream = Parser(commands=wanted, save_commands=True).parse(scfa)["body"]["commands"]
    tick, out = 0, {}
    for cmd in stream:
        if cmd["name"] == "Advance":
            tick += int(cmd.get("ticks") or 0)
            continue
        if cmd.get("clear_queue"):
            ids = [str(u) for u in cmd.get("entity_ids") or []]
            if ids:
                out.setdefault(army_of(ids[0]), []).append((tick, {"type": "clear", "units": ids, "tick": tick}))
        order = to_order(cmd, tick)
        if order is None:
            continue
        mine = out.setdefault(army_of(order["units"][0]), [])
        if order["type"] == "factory_build" and not cmd.get("clear_queue") and mine:
            last = mine[-1][1]
            if (last["type"] == "factory_build" and last["tick"] == tick and last["units"] == order["units"]
                    and last["bp"] == order["bp"]):
                last["count"] += 1                       # one of the copies of a queued factory order
                continue
        mine.append((tick, order))
    return out
