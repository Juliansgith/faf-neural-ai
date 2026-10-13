import json
import unittest

import numpy as np

from examples.tools_eval import schedule_rows
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


if __name__ == "__main__":
    unittest.main()
