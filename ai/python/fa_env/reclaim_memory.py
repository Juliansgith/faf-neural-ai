"""Reclaim memory updates shared by live execution and recording replay.

Apply after the current decision's labels are encoded. Successful empty scan
batches are observations; missing or failed replies are not evidence of absence.
"""
from __future__ import annotations


def after_step(previous, queries, answers, sent=(), results=()):
    scans = [(i, q) for i, q in enumerate(queries) if q.get("type") == "reclaim"]
    successful = []
    for i, _ in scans:
        reply = answers[i] if i < len(answers) else None
        if reply and reply.get("ok"):
            successful.append(reply)
    # Full scan batches replace the old token set, even when nothing remains.
    # With incomplete coverage, retain prior knowledge until a complete scan or
    # an explicit target-missing result; merge any newly observed props.
    props = {} if scans and len(successful) == len(scans) else {
        str(p["id"]): p for p in previous}
    for reply in successful:
        props.update((str(p["id"]), p) for p in reply.get("value") or [])
    for action, result in zip(sent, results):
        if action.get("type") != "reclaim" or result.get("ok"):
            continue
        target = action.get("target")
        # Lua appends a traceback to action errors. Match only its message line,
        # still requiring the exact target (never infer absence from another error).
        lines = str(result.get("error") or "").splitlines()
        message = lines[0] if lines else ""
        if target is not None and message == f"unknown target {target}":
            props.pop(str(target), None)
    return list(props.values())
