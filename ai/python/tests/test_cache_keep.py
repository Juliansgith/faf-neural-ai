"""archive_old keeps the training window's cache files whatever their age (2026-10-20: by age alone a restart after a
long stop archived 980 freshly re-encoded rl_089 games, and the storage budget then deleted them)."""
try:                                   # first: no test may reach the server or start a real process
    import no_network  # noqa: F401
except ImportError:
    from tests import no_network  # noqa: F401
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from fa_train.rl_loop import archive_old


def cached_game(cache: Path, name: str, round_name: str, days_old: float):
    manifest = {"source_root": str(cache.parent / "rounds" / round_name / "raw"), "source_name": f"{name}.jsonl.gz"}
    p = cache / f"{name}.v3.npz"
    np.savez(p, cache_manifest=np.array(json.dumps(manifest)), bc_supervise=np.zeros(1, bool))
    t = time.time() - days_old * 86400
    os.utime(p, (t, t))


class CacheKeepTest(unittest.TestCase):
    def test_the_window_stays_and_the_rest_ages_out(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cache, rounds, archive = root / "cache", root / "rounds", root / "archive"
            cache.mkdir()
            rounds.mkdir()
            cached_game(cache, "keep_old", "lane0_20261014T000000Z", 5)      # the champion's round, 5 days old
            cached_game(cache, "drop_old", "lane1_20261012T000000Z", 5)      # an old network's round
            cached_game(cache, "new", "lane2_20261020T000000Z", 0.1)         # fresh: never archived
            keep = {"lane0_20261014T000000Z"}
            raws, files = archive_old(rounds, cache, archive, 24, 72, keep_cache=lambda name: name in keep)
            self.assertEqual(files, 1)
            self.assertEqual(sorted(p.name for p in cache.glob("*.npz")), ["keep_old.v3.npz", "new.v3.npz"])
            self.assertTrue((archive / "cache" / "drop_old.v3.npz").exists())

    def test_without_a_rule_age_alone_decides(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cache").mkdir()
            (root / "rounds").mkdir()
            cached_game(root / "cache", "old", "lane0_20261014T000000Z", 5)
            self.assertEqual(archive_old(root / "rounds", root / "cache", root / "archive", 24, 72)[1], 1)


if __name__ == "__main__":
    unittest.main()
