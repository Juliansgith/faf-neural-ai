"""Compare two beat_probe .jsonl files beat by beat; print the first beat whose RNG fingerprint differs.

    probe_diff.py official.jsonl ours.jsonl [--all]
"""
import json
import sys

KEYS = ("tick", "rng_index", "has_pair", "pair_bits", "state_md5")


def load(path):
    return {r["beat"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    show_all = "--all" in sys.argv
    first = None
    for beat in sorted(set(a) & set(b)):
        ra, rb = a[beat], b[beat]
        same = all(ra[k] == rb[k] for k in KEYS)
        if not same and first is None:
            first = beat
        if show_all or (not same and beat - first < 8):
            print(f"{'  ' if same else '!!'} beat {beat:5d}  A idx={ra['rng_index']:3d} pair={ra['has_pair']} "
                  f"{ra['state_md5']}   B idx={rb['rng_index']:3d} pair={rb['has_pair']} {rb['state_md5']}")
    print(f"compared {len(set(a) & set(b))} beats; first differing beat: {first}")


if __name__ == "__main__":
    main()
