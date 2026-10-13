import json
import unittest
from types import SimpleNamespace

import numpy as np

from fa_train.learn_rl import ACU_SPEED, phi_commander, potential, rewards_of


def game(xs, present, size=1024.0, mp="SCMP_004"):
    n = len(xs)
    focus = np.zeros((n, 3, 3), np.float32)
    focus[:, 0, 0] = np.asarray(xs, np.float32) / size
    focus[:, 0, 1] = 0.5
    focus[:, 0, 2] = present
    g = {"focus": focus, "fine": np.tile(np.array([4, 4, size, size], np.float32), (n, 1)),
         "meta": np.array(json.dumps({"map": mp}))}
    for k in ("terminal", "trade", "economy", "commander"):
        g[f"rew_{k}"] = np.zeros(n, np.float32)
    return g


def args(**kw):
    base = dict(econ_source="recorded", econ_coef=0.0, intel_coef=0.0, map_coef=0.0, waste_coef=0.0, stall_coef=0.0,
                shaping=1.0, gamma=0.997, acu_coef=3.0, acu_grace=1.0, acu_span=4.0, acu_all_maps=False)
    base.update(kw)
    return SimpleNamespace(**base)


class CommanderExposureTest(unittest.TestCase):
    def test_minutes_of_walk_home(self):
        minute = ACU_SPEED * 60.0
        g = game([100, 100 + minute, 100 + 3 * minute, 100 + 9 * minute, 0], [1, 1, 1, 1, 0])
        np.testing.assert_allclose(phi_commander(g, 5, 1.0, 4.0), [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-5)   # held when dead

    def test_walking_out_costs_and_coming_back_returns_it(self):
        minute = ACU_SPEED * 60.0
        out_and_back = game([100, 100 + 5 * minute, 100], [1, 1, 1])
        r = rewards_of(out_and_back, args(gamma=1.0))
        self.assertAlmostEqual(float(r[0]), -3.0, places=4)            # out: the full exposure
        self.assertAlmostEqual(float(r.sum()), 0.0, places=4)          # back home: nothing kept
        dies_out = game([100, 100 + 5 * minute, 0], [1, 1, 0])
        dies_out["rew_terminal"][-1] = -10.0
        self.assertAlmostEqual(float(rewards_of(dies_out, args(gamma=1.0)).sum()), -13.0, places=4)
        wins_out = game([100, 100 + 5 * minute, 100 + 5 * minute], [1, 1, 1])
        wins_out["rew_terminal"][-1] = 10.0
        self.assertAlmostEqual(float(rewards_of(wins_out, args(gamma=1.0)).sum()), 10.0, places=4)   # given back

    def test_small_maps_untouched_unless_asked(self):
        minute = ACU_SPEED * 60.0
        g = game([10, 10 + 2 * minute], [1, 1], size=256.0, mp="SCMP_012")
        self.assertEqual(float(np.abs(rewards_of(g, args())).sum()), 0.0)
        self.assertNotEqual(float(np.abs(rewards_of(g, args(acu_all_maps=True))).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
