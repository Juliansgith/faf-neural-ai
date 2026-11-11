"""A run's storage under a budget, applied by fa_train.rl_loop every iteration (--storage-budget-gb). Taken from
_runs/tools/storage_cleanup.py (2026-10-16, one run's paths written in) and parameterised: the rounds folder, the
archive folder, the budget, the lineage, the networks whose games are kept and the champion.

Why: recordings grow ~10 MB a game and the cache ~8.6 MB, about 100 GB a day at ~235 games an hour; the archive disk
(D:, 165 GB free) filled in 1-3 days, and then archive_old raised on every pass and every restart (2026-10-16
review, bugs-loop F1). Nothing reads the archived cache again, and an abandoned branch's raw games are never
trained on.

The policy, in this order (nothing else is ever deleted):
  1. archived cache files (<archive>/cache/*.npz: moved there by archive_old after --archive-cache-hours; a game can
     be encoded again from its raw recording) and the set-aside copies of rounds that arrived incomplete
     (<rounds>/_incomplete);
  2. raw recordings (in rounds/<r>/raw or archived) of networks OUTSIDE the lineage: known checkpoints of this run
     that the current network does not descend from (abandoned branches);
  3. only while the archive folder is over the budget: archived raw recordings of the lineage, oldest first --
     networks older than the newest `keep` lineage networks first, then the kept ones.
Never: the champion's games, the games listed for self-imitation (sil.json), rounds of unknown networks,
checkpoints, the rounds' round.json/eval data, anything younger than `min_age_hours` since it arrived. A deleted raw
folder leaves a RAW_DELETED note in its round folder.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

_SIZES: dict = {}        # (folder, its mtime_ns) -> bytes: a raw folder never changes once it is complete


def dir_size(p: Path) -> int:
    """Bytes under `p` (a file's own size). Folders are remembered by their modification time: an archived raw
    folder never changes, and summing ~150,000 recordings every iteration cost seconds."""
    try:
        if p.is_file():
            return p.stat().st_size
        key = (str(p), p.stat().st_mtime_ns)
    except OSError:
        return 0
    hit = _SIZES.get(key)
    if hit is not None:
        return hit
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    _SIZES[key] = total
    return total


def tree_size(p: Path) -> int:
    """Bytes under `p`, each subfolder through dir_size's memory (rounds_raw: one entry per round)."""
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for child in p.iterdir():
        if child.is_dir() and child.name == "rounds_raw":
            total += sum(dir_size(r) for r in child.iterdir())
        else:
            total += dir_size(child)
    return total


def _round_sha(r: Path) -> str | None:
    try:
        return json.loads((r / "round.json").read_text()).get("checkpoint_sha256")
    except (OSError, ValueError):
        return None


def _arrived(r: Path) -> float:
    """When a round reached this machine: its ARRIVED stamp, else its folder's creation time (Windows; extraction
    and a same-disk move keep the player's modification time, not the creation time), else its mtime."""
    stamp = r / "ARRIVED"
    try:
        if stamp.exists():
            return stamp.stat().st_mtime
        st = r.stat()
        return getattr(st, "st_birthtime", None) or st.st_ctime
    except OSError:
        return time.time()


def _raw_of(r: Path) -> Path | None:
    if (r / "raw").is_dir():
        return r / "raw"
    note = r / "RAW_ARCHIVED"
    if note.exists():
        p = Path(note.read_text().strip())
        return p if p.is_dir() else None
    return None


def sil_rounds(sil_file: Path) -> set:
    """The round names of the self-imitation games ([source_root, source_name]; a root is <round>/raw)."""
    try:
        return {Path(root).parent.name for root, _ in json.loads(sil_file.read_text())}
    except (OSError, ValueError, TypeError):
        return set()


def _under(p: Path, base: Path | None) -> bool:
    return base is not None and str(p.resolve()).lower().startswith(str(base.resolve()).lower())


def plan(rounds: Path, archive: Path | None, budget_gb: float, lineage_shas: set, keep_shas: set,
         champion_sha: str | None, known_shas: set, protect_rounds: set = frozenset(),
         min_age_hours: float = 24.0, now: float | None = None, archive_bytes: int | None = None) -> list:
    """[(path, why, bytes)] to delete, by the policy above. `lineage_shas`: the networks the current one descends
    from; `keep_shas`: the newest lineage networks whose games stay while the budget allows; `known_shas`: every
    checkpoint of the run (a round of an unknown network is never touched); `protect_rounds`: round names never
    touched (self-imitation). `archive_bytes`: the archive's size now (measured when None)."""
    now = time.time() if now is None else now
    min_age = min_age_hours * 3600
    out: list = []
    # 1. archived cache and set-aside incomplete copies
    if archive is not None and (archive / "cache").is_dir():
        for f in sorted((archive / "cache").glob("*.npz")):
            out.append((f, "archived cache (a game can be encoded again from its recording)", dir_size(f)))
    inc = rounds / "_incomplete"
    if inc.is_dir():
        for f in sorted(inc.iterdir()):
            try:
                old = now - f.stat().st_mtime > min_age
            except OSError:
                old = False
            if old:
                out.append((f, "copy of a round that arrived incomplete", dir_size(f)))
    # 2. raw recordings of networks outside the lineage; the lineage's archived ones are budget candidates
    lineage_old, lineage_kept = [], []
    for r in sorted(rounds.glob("lane*_*"), key=lambda p: p.name.split("_", 1)[-1]):
        if not (r / "DONE").exists() or r.name in protect_rounds:
            continue
        raw = _raw_of(r)
        sha = _round_sha(r)
        if raw is None or sha is None or sha not in known_shas or sha == champion_sha:
            continue
        if now - _arrived(r) < min_age:
            continue
        if sha not in lineage_shas:
            out.append((raw, f"raw games of {r.name}: a network outside the lineage", dir_size(raw)))
        elif _under(raw, archive):
            (lineage_kept if sha in keep_shas else lineage_old).append(raw)
    # 3. over the budget: the lineage's archived raw games, oldest first (older networks before the kept ones)
    if archive is not None and budget_gb > 0:
        size = tree_size(archive) if archive_bytes is None else archive_bytes
        size -= sum(b for p, _, b in out if _under(p, archive))
        for raw in lineage_old + lineage_kept:
            if size <= budget_gb * 2 ** 30:
                break
            b = dir_size(raw)
            out.append((raw, f"oldest lineage games over the {budget_gb:g} GB budget", b))
            size -= b
    return out


def apply(items: list, rounds: Path) -> int:
    """Delete the plan's items; a deleted raw folder leaves RAW_DELETED in its round folder. Returns bytes freed."""
    freed = 0
    for path, why, b in items:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                if path.name == "raw":
                    name = path.parent.name
                    note = rounds / name / "RAW_DELETED"
                    if note.parent.is_dir():
                        note.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {why}")
            elif path.exists():
                path.unlink()
            else:
                continue
            freed += b
        except OSError:
            continue
    return freed


def summary(items: list) -> str:
    by: dict = {}
    for _, why, b in items:
        key = why.split(": ")[-1] if why.startswith("raw games of") else why
        n, s = by.get(key, (0, 0))
        by[key] = (n + 1, s + b)
    return "; ".join(f"{key}: {n} ({s / 2 ** 30:.1f} GB)" for key, (n, s) in by.items())
