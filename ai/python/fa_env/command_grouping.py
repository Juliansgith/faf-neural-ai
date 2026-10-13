"""Coalesce independent, identical set orders without changing a unit's queue.

This operates on already reduced commands. It does not drop, replace, delay or
approximate an order; callers still apply observation masks and slot budgets.
"""


def group_independent(commands, set_verbs, verb_order):
    pending = [dict(c, subject=list(c.get("subject") or [])) for c in commands]
    out = []
    signature = lambda c: {k: v for k, v in c.items() if k != "subject"}
    while pending:
        ready, blocked = [], set()
        for index, command in enumerate(pending):
            subjects = set(command["subject"])
            if not subjects.intersection(blocked):
                ready.append(index)
            blocked.update(subjects)
        rank = min(verb_order.get(pending[i]["verb"], 99) for i in ready)
        ready = [i for i in ready if verb_order.get(pending[i]["verb"], 99) == rank]
        index = ready[0]
        if out and out[-1]["verb"] in set_verbs:
            matching = [i for i in ready if signature(pending[i]) == signature(out[-1])]
            if matching:
                index = matching[0]
        command = pending.pop(index)
        # Never collapse two separate commands for the same unit. That could
        # erase a queue entry even if both entries have identical arguments.
        if (out and command["verb"] in set_verbs and
                signature(command) == signature(out[-1]) and
                not set(command["subject"]).intersection(out[-1]["subject"])):
            out[-1]["subject"].extend(command["subject"])
        else:
            out.append(command)
    return out
