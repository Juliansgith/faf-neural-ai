import unittest

from fa_train.tools_bc import ECONOMY_HORIZON, economy_bonus, local_trade_arrays


class Kit:
    def cell(self, x, z):
        return int(x), int(z)


def step(*units):
    return {"msg": {"obs": {"units": [dict(id=i, bp=bp, x=x, z=0.0, hp=hp, maxhp=100.0, army=army)
                                      for i, bp, x, hp, army in units]}}}


BPS = {"mex1": {"mass": 36, "massProduction": 2.0, "categories": ["STRUCTURE", "MASSEXTRACTION"]},
       "mex2": {"mass": 900, "massProduction": 6.0}, "tank": {"mass": 50},
       "eng": {"mass": 52, "categories": ["ENGINEER", "MOBILE"]}, "acu": {"mass": 18000, "categories": ["COMMAND", "ENGINEER"]}}


class LocalTradeTest(unittest.TestCase):
    def values(self, steps):
        out = local_trade_arrays(steps, Kit(), BPS, me=1)
        return [float(v) for v in out["loc_values"]]

    def test_upgraded_own_mex_is_no_loss(self):
        # the T1 mex vanishes at full health and a T2 mex appears: an upgrade, not a loss
        self.assertEqual(self.values([step(("a", "mex1", 3, 100, 1)), step(("b", "mex2", 3, 30, 1))]), [])

    def test_killed_own_unit_is_a_loss(self):
        self.assertEqual(self.values([step(("t", "tank", 2, 40, 1)), step()]), [-0.4 * 50])

    def test_economy_units_are_worth_more_than_their_cost(self):
        out = local_trade_arrays([step(("m", "mex1", 2, 20, 2)), step()], Kit(), BPS, me=1)
        self.assertAlmostEqual(float(out["loc_values"][0]), 0.2 * (36 + 2.0 * ECONOMY_HORIZON), places=4)   # a raided mex
        self.assertAlmostEqual(float(out["econ_trade"][0]), 0.2 * 2.0 * ECONOMY_HORIZON / 1000.0, places=5)
        self.assertEqual(float(out["econ_trade"][1]), 0.0)
        lost = local_trade_arrays([step(("e", "eng", 2, 50, 1)), step()], Kit(), BPS, me=1)
        self.assertAlmostEqual(float(lost["econ_trade"][0]), -0.5 * 52 / 1000.0, places=5)             # an engineer: 2x
        self.assertEqual(economy_bonus(BPS["acu"]), 0.0)                                       # not the commander

    def test_enemy_that_walks_out_of_sight_is_no_kill(self):
        self.assertEqual(self.values([step(("e", "tank", 2, 90, 2)), step()]), [])
        self.assertEqual(self.values([step(("e", "tank", 2, 20, 2)), step()]), [0.2 * 50])


if __name__ == "__main__":
    unittest.main()
