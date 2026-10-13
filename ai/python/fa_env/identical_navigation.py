"""Omit only redundant clear+navigation pairs against a single active order.

No cooldown or guessed intent: changed destinations, queued routes, idle units,
and units touched earlier in this decision retain replacement semantics.
"""
import math


def identical_navigation(actions, units):
    out, touched = [], set()
    index = 0
    while index < len(actions):
        action = actions[index]
        following = actions[index + 1] if index + 1 < len(actions) else {}
        ids = [str(uid) for uid in action.get('units', [])]
        if (action.get('type') == 'clear'
                and following.get('type') in ('move', 'attack_move')
                and ids == [str(uid) for uid in following.get('units', [])]):
            order = 2 if following['type'] == 'move' else 35
            goal = following.get('pos')
            keep = []
            for uid in ids:
                unit = units.get(uid) or {}
                dest = unit.get('dest')
                same = (uid not in touched and unit.get('queue') == 1
                        and unit.get('idle') is False and unit.get('order') == order
                        and isinstance(goal, (tuple, list)) and len(goal) == 2
                        and isinstance(dest, (tuple, list)) and len(dest) == 2
                        and all(isinstance(v, (int, float)) and math.isfinite(v)
                                for v in (*goal, *dest))
                        and all(abs(a-b) <= 1e-4 for a,b in zip(goal,dest)))
                if not same:
                    keep.append(uid)
            if keep:
                out.extend([dict(action, units=keep), dict(following, units=keep)])
                touched.update(keep)
            index += 2
        else:
            out.append(action)
            touched.update(ids)
            index += 1
    return out
