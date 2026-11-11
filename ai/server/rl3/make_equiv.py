"""Write ~/fa/rl3/acting_equivalence.json: the acting-file revisions deployed since environment 1 (rounds
20261107T2008Z..20261109T1606Z, where rl_000 played 40 rounds) declared equal to environment 1's."""
import collections
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/faf-neural-ai/ai/python"))
from fa_train.rl_loop import _ACTING_TRAIN_SOURCES  # noqa: E402

VERIFIED = {"fa_env/bridge.py", "fa_env/env.py", "fa_env/maps.py", "fa_train/tools_model.py", "fa_train/trainer.py"}
home = os.path.expanduser("~")
code = Path(home) / "faf-neural-ai" / "ai" / "python"
rounds = sorted(glob.glob(home + "/fa/rl3/rounds/lane*_*") + glob.glob(home + "/fa/player/rounds/lane*_*"),
                key=lambda d: os.path.basename(d)[6:])


def acting(name):
    return name.startswith("fa_env/") or name in _ACTING_TRAIN_SOURCES


env1 = collections.defaultdict(collections.Counter)
later = collections.defaultdict(set)
for d in rounds:
    stamp = os.path.basename(d)[6:]
    try:
        src = json.load(open(d + "/eval/eval_manifest.json")).get("policy_source_sha256") or {}
    except (OSError, ValueError):
        continue
    for name, digest in src.items():
        if not acting(name):
            continue
        if "20261107T2008" <= stamp <= "20261109T1607":
            env1[name][digest] += 1
        elif stamp > "20261109T1607":
            later[name].add(digest)
canonical = {name: c.most_common(1)[0][0] for name, c in env1.items()}
for name in sorted(canonical):                       # the files as deployed now (future rounds)
    p = code / name
    if p.exists():
        later[name].add(hashlib.sha256(p.read_bytes()).hexdigest())
sources, bad = {}, []
for name, digests in sorted(later.items()):
    extra = {d: canonical.get(name) for d in digests if d != canonical.get(name)}
    if not extra:
        continue
    if name not in VERIFIED or canonical.get(name) is None:
        bad.append(name)
        continue
    sources[name] = extra
if bad:
    sys.exit(f"unverified acting files changed: {bad}")
out = {
    "sources": sources,
    "gate_exclude_maps": [],
    "new_maps": ["SCMP_017", "X1MP_005", "X1MP_006", "X1MP_017"],
    "notes": [
        "2026-11-10: environment 1 = rounds 20261107T2008Z..20261109T1606Z (rl_000 played 40 rounds there).",
        "Declared equal, diffs read: bridge.py shader-cache slots on Linux (31326c7), the FAF affinity pin skipped and "
        "threads widened on Linux (a680a86, 8d53e0b); env.py passes the init file name to unpin_threads (8d53e0b); "
        "maps.py four more pool entries (e818b72, the schedule comes from the player config); tools_model.py a "
        "zero-initialised aux_econ head acting never reads (3ca236e) and a channels_last switch that is off when "
        "acting (3c8713e) and the units' attention in bf16 only when amp is on (acting has it off; lane 3's bf16 test "
        "and the learner have it on; rl_005 drift -0.0189 vs -0.0190 for the bf16 encoder alone); trainer.py no new game of a round during a learning pause (36a65a0).",
        "The four maps added on 2026-11-10 were out of the gate until 2026-11-11 (rl_000 never played them); with "
        "the control lane the champion plays them too, so the gate counts them again. The collapse and regression "
        "guards of the loop still read them apart (new_maps).",
        "Lane 3 acts with the bf16 encoder (FA_ENV_ACT_BF16, decisions move -0.009 log-probability) and lane 1 with "
        "2 threads: neither is in the source key.",
    ],
}
path = Path(home) / "fa" / "rl3" / "acting_equivalence.json"
path.write_text(json.dumps(out, indent=1))
print({k: len(v) for k, v in sources.items()}, "->", path)
