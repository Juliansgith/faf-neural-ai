"""Summarize the twister draw sites of two beat_probe --watch runs: per beat, draws per call site.

    draw_diff.py official.jsonl ours.jsonl [--seq]

Official addresses are named with binary_index (the addresses faf-re's sources document); ours come
symbolized from main.pdb.
"""
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from binary_index import BinaryIndex  # noqa: E402

RNG = re.compile(r"CMersenneTwister|CRandomStream|NextUInt32|FRand|^FA$|MulTwisterWord|SampleSymmetricRange")


def clean(name: str) -> str:
    name = name.split("+")[0]
    name = re.sub(r"^\?(\w+)@(\w+)@Moho@@.*", r"\2::\1", name)
    return name.replace("moho::", "").replace("Moho::", "")


def site(names):
    rng = [n for n in names if RNG.search(n)]
    rest = [n for n in names if not RNG.search(n)]
    return (rng[0] if rng else "?") + " <- " + " <- ".join(rest[:2])


def load(path):
    return {r["beat"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def main():
    src = REPO.parent / "faf-re-dev" / "src"
    idx = BinaryIndex(src)
    off, ours = load(sys.argv[1]), load(sys.argv[2])
    for beat in sorted(set(off) & set(ours)):
        a, b = off[beat].get("draws", []), ours[beat].get("draws", [])
        if not a and not b:
            continue
        print(f"=== draws before AdvanceBeat of beat {beat}: official {len(a)}, ours {len(b)}")
        sa = [site([clean(idx.name(int(x, 16))) for x in d[1:]]) for d in a]
        sb = [site([clean(x) for x in d[1:]]) for d in b]
        ca, cb = collections.Counter(sa), collections.Counter(sb)
        for k in sorted(set(ca) | set(cb), key=lambda k: -(ca[k] + cb[k])):
            print(f"   {ca[k]:5d} {cb[k]:5d}  {k}")
        if "--seq" in sys.argv:
            for i in range(max(len(sa), len(sb))):
                x = sa[i] if i < len(sa) else ""
                y = sb[i] if i < len(sb) else ""
                print(f"    {i:4d} {'  ' if x == y else '!='} {x:80s} | {y}")


if __name__ == "__main__":
    main()
