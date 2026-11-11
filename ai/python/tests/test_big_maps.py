import json
import unittest

import numpy as np

from examples.tools_eval import schedule_rows, wants_labels
from fa_train.learn_rl import map_km


class BigMapTest(unittest.TestCase):
    def test_map_km_from_the_recording_meta(self):
        self.assertEqual(map_km({"meta": np.array(json.dumps({"map": "SCMP_004"}))}), 20)
        self.assertEqual(map_km({"meta": np.array(json.dumps({"map": "SCMP_012"}))}), 5)
        self.assertEqual(map_km({}), 5)

    def test_large_maps_get_their_own_time_limit(self):
        rows = schedule_rows(["rush"], 2, ["SCMP_012", "SCMP_004"], [1], [1], 40.0, 0, minutes_large=55.0)
        self.assertEqual({r["map"]: r["minutes"] for r in rows}, {"SCMP_012": 40.0, "SCMP_004": 55.0})
        rows = schedule_rows(["rush"], 2, ["SCMP_012", "SCMP_004"], [1], [1], 40.0, 0)
        self.assertEqual({r["minutes"] for r in rows}, {40.0})


class VariedConditionsTest(unittest.TestCase):
    def test_rounds_vary_but_opponents_share_each_row(self):
        maps = ["SCMP_012", "SCMP_016", "SCMP_004"]
        a = schedule_rows(["rush", "tech"], 12, maps, [1, 2, 3, 4], [1, 2, 3, 4], 40.0, 10000, vary=True)
        b = schedule_rows(["rush", "tech"], 12, maps, [1, 2, 3, 4], [1, 2, 3, 4], 40.0, 20000, vary=True)
        cond = lambda r: (r["map"], r["slots"], r["faction"], r["ai_faction"])       # noqa: E731
        rush, tech = [cond(r) for r in a if r["ai"] == "rush"], [cond(r) for r in a if r["ai"] == "tech"]
        self.assertEqual(rush, tech)                                   # paired within a round
        self.assertNotEqual(rush, [cond(r) for r in b if r["ai"] == "rush"])          # another round, other games
        self.assertGreater(len({c[2] for c in rush}), 1)               # not one faction


class LabelsShareTest(unittest.TestCase):
    def test_share(self):
        rows = [{"ai": "rush", "map": "SCMP_012", "seed": s} for s in range(1000)]
        self.assertTrue(all(wants_labels(r, None) for r in rows))
        n = sum(wants_labels(r, 0.5) for r in rows)
        self.assertTrue(420 < n < 580, n)
        self.assertEqual([wants_labels(r, 0.5) for r in rows[:50]], [wants_labels(r, 0.5) for r in rows[:50]])

    def test_large_maps_and_self_play_keep_them(self):
        large = next(m for m, e in __import__("fa_env.maps", fromlist=["MAP_POOL"]).MAP_POOL.items() if e["km"] > 10)
        self.assertTrue(all(wants_labels({"ai": "rush", "map": large, "seed": s}, 0.0) for s in range(50)))
        self.assertTrue(all(wants_labels({"ai": "self@rl_089", "map": "SCMP_012", "seed": s}, 0.0) for s in range(50)))


if __name__ == "__main__":
    unittest.main()
