"""Diff the Lua Random() traces (RND lines from make_lua_trace.py's tracer) of two game logs.

    rnd_diff.py official.log ours.log

Prints calls per tick, the first call whose site or value differs, and an aligned view around it.
"""
import collections
import difflib
import re
import sys

LINE = re.compile(r"RND (-?\d+) (\S+) n=(\d+) (\S+)")


def load(path):
    rows = []
    for line in open(path, encoding="latin-1"):
        m = LINE.search(line)
        if m:
            where = m.group(2).replace("\\\\", "\\")
            k = where.lower().find("lua.nx2")  # the two engines spell the archive path differently
            where = where[k + 8:] if k >= 0 else where
            rows.append((int(m.group(1)), where, m.group(3), m.group(4)))
    return rows


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print(f"official {len(a)} calls, ours {len(b)} calls")
    ta, tb = collections.Counter(r[0] for r in a), collections.Counter(r[0] for r in b)
    for t in sorted(set(ta) | set(tb)):
        print(f"{'  ' if ta[t] == tb[t] else '!!'} tick {t:5d}: official {ta[t]:5d} ours {tb[t]:5d}")
    ka, kb = [r[:3] for r in a], [r[:3] for r in b]
    first = next((i for i, (x, y) in enumerate(zip(ka, kb)) if x != y), None)
    vfirst = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print("first call-site divergence at call", first, "; first value divergence at call", vfirst)
    if vfirst is not None:
        for i in range(max(0, vfirst - 3), min(len(a), vfirst + 3)):
            print(f"  {i:5d} official {a[i]}  ours {b[i] if i < len(b) else None}")
    if first is not None:
        lo = max(0, first - 8)
        sm = difflib.SequenceMatcher(a=ka[lo:first + 200], b=kb[lo:first + 200], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                print(f"  = {i2 - i1} calls equal")
            else:
                print(f"  {tag}: official[{lo + i1}:{lo + i2}] vs ours[{lo + j1}:{lo + j2}]")
                for x in ka[lo + i1:lo + i2][:12]:
                    print(f"      official {x}")
                for y in kb[lo + j1:lo + j2][:12]:
                    print(f"      ours     {y}")


if __name__ == "__main__":
    main()
