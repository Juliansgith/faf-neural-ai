"""The compact memory-mapped layout (tools_bc.MAPPED_LAYOUT) reads back exactly the cached arrays."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train import tools_bc  # noqa: E402


def fake_game(steps: int = 30, rows: int = 41, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = rng.random((steps, 16, 32, 32)).astype(np.float16)
    m[:, [0, 2, 5, 15]] = m[:1, [0, 2, 5, 15]]                  # constant over the game
    return {
        "map": m,
        "mask_build_cell": rng.random((steps, 4, 32, 32)) < 0.3,
        "mask_move_cell": rng.random((steps, 32, 32)) < 0.5,
        "mask_subject_t": rng.random((rows, 27)) < 0.2,
        "set_t": rng.random((rows, 24)) < 0.1,
        "mask_silo": rng.random((rows, 2)) < 0.5,
        "unit_supervise": rng.random(steps) < 0.5,                # 1-D: stays as it is
        "tokens": rng.random((rows, 70)).astype(np.float16),
        "cand_feat": rng.random((9, 34)).astype(np.float32)[rng.integers(0, 9, rows * 40)],  # rows repeat
        "offsets_t": np.linspace(0, rows, steps + 1).astype(np.int64),
        "returns": rng.random(steps).astype(np.float32),
        "cache_manifest": np.array(json.dumps({"format": 10})),
    }


class MappedLayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)   # memory maps stay open on Windows
        self.root = Path(self.tmp.name)
        self.reserve = tools_bc.MAPPED_DISK_RESERVE
        tools_bc.MAPPED_DISK_RESERVE = 0

    def tearDown(self):
        tools_bc.MAPPED_DISK_RESERVE = self.reserve
        self.tmp.cleanup()

    def write(self, game: dict, name: str = "g.v3.npz") -> Path:
        path = self.root / name
        np.savez_compressed(path, **game)
        return path

    def assert_same(self, original: dict, mapped: dict):
        self.assertEqual(set(original), set(mapped))
        for k, a in original.items():
            b = mapped[k]
            if a.ndim == 0:
                self.assertEqual(str(a.item()), str(np.asarray(b).item()))
                continue
            self.assertEqual(tuple(a.shape), tuple(b.shape), k)
            np.testing.assert_array_equal(np.asarray(b), a, err_msg=k)
            for t in (0, len(a) - 1, len(a) // 2):
                np.testing.assert_array_equal(b[t], a[t], err_msg=f"{k}[{t}]")
                self.assertEqual(b[t].dtype, a.dtype, k)
            np.testing.assert_array_equal(b[1:4], a[1:4], err_msg=f"{k}[1:4]")
            np.testing.assert_array_equal(b[2:2], a[2:2], err_msg=f"{k}[2:2]")

    def test_round_trip_is_exact_and_smaller(self):
        game = fake_game()
        folder = self.root / "mapped" / "g"
        mapped = tools_bc._mapped_arrays(self.write(game), folder)
        self.assert_same(game, mapped)
        self.assertIsInstance(mapped["map"], tools_bc.SplitMap)
        self.assertEqual(mapped["map"].static_idx.tolist(), [0, 2, 5, 15])
        self.assertIsInstance(mapped["set_t"], tools_bc.PackedBool)
        self.assertIsInstance(mapped["unit_supervise"], np.ndarray)
        self.assertIsInstance(mapped["cand_feat"], tools_bc.IndexedRows)          # 9 distinct rows of 1640
        self.assertIsInstance(mapped["tokens"], np.ndarray)                       # all distinct: stored as is
        size = lambda pattern: sum(p.stat().st_size for p in folder.glob(pattern))  # noqa: E731
        self.assertLess(size("map.*.npy"), game["map"].nbytes * 13 / 16)        # 12 of 16 channels change here

    def test_repeating_map_planes_are_stored_once(self):
        game = fake_game(seed=4)
        m = game["map"]
        m[5:, [3, 7, 8, 9, 10, 11]] = m[5, [3, 7, 8, 9, 10, 11]]   # six channels stop changing after step 5
        folder = self.root / "mapped" / "p"
        mapped = tools_bc._mapped_arrays(self.write(game), folder)
        self.assert_same(game, mapped)
        self.assertIsInstance(mapped["map"].dynamic, tools_bc.IndexedRows)
        self.assertEqual(mapped["set_t"].bits.nbytes, game["set_t"].shape[0] * 3)            # 24 bools: 3 bytes
        reopened = tools_bc.open_mapped(folder)              # what a loader process sees
        self.assert_same(game, reopened)

    def test_an_old_layout_copy_is_rewritten(self):
        game = fake_game(seed=1)
        path = self.write(game)
        folder = self.root / "mapped" / "g"
        folder.mkdir(parents=True)
        for k, a in game.items():                          # layout 1: every member as it is, old marker
            np.save(folder / f"{k}.npy", a)
        stat = path.stat()
        (folder / "complete.json").write_text(json.dumps({"source": path.name, "bytes": stat.st_size,
                                                          "mtime_ns": stat.st_mtime_ns}))
        mapped = tools_bc._mapped_arrays(path, folder)
        self.assert_same(game, mapped)
        self.assertFalse((folder / "map.npy").exists())
        self.assertFalse((folder / "set_t.npy").exists())
        self.assertEqual(json.loads((folder / "complete.json").read_text())["layout"], tools_bc.MAPPED_LAYOUT)

    def test_a_game_where_every_map_channel_changes(self):
        game = fake_game(seed=2)
        game["map"] = np.random.default_rng(3).random((5, 16, 32, 32)).astype(np.float16)
        game["offsets_t"] = np.linspace(0, 41, 6).astype(np.int64)
        game["returns"] = game["returns"][:5]
        game["mask_build_cell"], game["mask_move_cell"] = game["mask_build_cell"][:5], game["mask_move_cell"][:5]
        game["unit_supervise"] = game["unit_supervise"][:5]
        mapped = tools_bc._mapped_arrays(self.write(game), self.root / "m" / "g")
        self.assert_same(game, mapped)


if __name__ == "__main__":
    unittest.main()


class LazyAndIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.reserve = tools_bc.MAPPED_DISK_RESERVE
        tools_bc.MAPPED_DISK_RESERVE = 0

    def tearDown(self):
        tools_bc.MAPPED_DISK_RESERVE = self.reserve
        self.tmp.cleanup()

    def test_lazy_game_reads_what_open_mapped_reads(self):
        game = fake_game(seed=7)
        path = self.root / "g.v3.npz"
        np.savez_compressed(path, **game)
        folder = self.root / "m" / "g"
        eager = tools_bc._mapped_arrays(path, folder)
        lazy = tools_bc._mapped_arrays(path, folder, lazy=True)
        self.assertIsInstance(lazy, tools_bc.LazyMapped)
        self.assertEqual(set(lazy.keys()), set(eager))
        for k in game:
            if np.asarray(game[k]).ndim:
                np.testing.assert_array_equal(np.asarray(lazy[k]), np.asarray(eager[k]), err_msg=k)
                np.testing.assert_array_equal(lazy[k][1:3], eager[k][1:3], err_msg=k)
        self.assertNotIn("nope", lazy)
        self.assertIsNone(lazy.get("nope"))
        lazy["_cache_manifest"] = {"x": 1}
        self.assertEqual(lazy["_cache_manifest"], {"x": 1})
        with self.assertRaises(KeyError):
            lazy["nope"]

    def test_index_reads_each_file_once_and_follows_changes(self):
        cache = self.root / "cache"
        cache.mkdir()
        for i in range(2):
            np.savez_compressed(cache / f"g{i}.v3.npz", returns=np.zeros(3, np.float32), bc_supervise=np.ones(3, bool),
                                cache_manifest=np.array(json.dumps({"format": tools_bc.CACHE_FORMAT})))
        first = tools_bc.cache_index(cache)
        self.assertEqual(set(first), {"g0.v3.npz", "g1.v3.npz"})
        self.assertTrue((cache / tools_bc.CACHE_INDEX).exists())
        self.assertEqual(tools_bc.cache_index(cache), first)               # unchanged files: from the index
        (cache / "g1.v3.npz").unlink()
        np.savez_compressed(cache / "g2.v3.npz", returns=np.zeros(3, np.float32),
                            cache_manifest=np.array(json.dumps({"format": 0})))
        again = tools_bc.cache_index(cache)
        self.assertEqual(set(again), {"g0.v3.npz", "g2.v3.npz"})
        self.assertFalse(again["g2.v3.npz"]["bc"])
