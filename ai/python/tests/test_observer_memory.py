"""The observer's per-unit memory and enemy split (2026-10-17 review, bugs-env F1, L1, F5): recycled ids start
clean, units at a build are not given replacing orders, neutral civilians are no enemy tokens."""
import unittest

import numpy as np

from fa_env.tools import ToolKit
from fa_env.tools_obs import MAP_CHANNELS, V_INDEX
from observer_fixture import BPS, CIVILIAN, ENEMY, info, obs, observer, token_of, unit


class RecycledIdsTest(unittest.TestCase):
    def test_a_recycled_id_starts_without_the_dead_units_order(self):
        ob = observer()
        ob.observe(obs([unit("7", "eng", 30, 30), unit("8", "tank", 40, 40)], tick=0))
        ob.note_commands([{"verb": "attack_move", "subject": ["7", "8"], "cell": (20, 20)}], 0)
        ob.observe(obs([unit("8", "tank", 40, 40)], tick=20))           # 7 died
        x = ob.observe(obs([unit("7", "tank", 30, 30, frac=0.0), unit("8", "tank", 40, 40)], tick=40))
        i, j = token_of(x, "7"), token_of(x, "8")
        self.assertEqual(x["token_last"][i], 0)                          # a new unit: never ordered
        self.assertEqual(x["tokens"][i, 22], 1.0)
        self.assertFalse(x["token_last_feat"][i].any())
        self.assertEqual(x["token_last"][j], V_INDEX["attack_move"])      # a unit that stayed keeps its order

    def test_a_unit_ordered_in_its_first_decision_keeps_the_order(self):
        ob = observer()
        ob.observe(obs([unit("8", "tank", 40, 40)], tick=0))
        ob.observe(obs([unit("8", "tank", 40, 40), unit("9", "tank", 41, 41)], tick=20))
        ob.note_commands([{"verb": "move", "subject": ["9"], "cell": (5, 5)}], 20)
        x = ob.observe(obs([unit("8", "tank", 40, 40), unit("9", "tank", 41, 41)], tick=40))
        self.assertEqual(x["token_last"][token_of(x, "9")], V_INDEX["move"])


class DroppedCommandsTest(unittest.TestCase):
    def test_a_unit_at_a_build_is_not_given_replacing_orders(self):
        ob = observer()
        ob.observe(obs([unit("7", "eng", 30, 30, order=8), unit("8", "eng", 31, 31)], tick=0))
        ob.note_commands([{"verb": "assist", "subject": ["7", "8"], "target": "8"},
                          {"verb": "stop", "subject": ["7"]}], 0)
        self.assertNotIn("7", ob.last_order)
        self.assertNotIn("7", ob.commanded)
        self.assertEqual(ob.last_order["8"][0], V_INDEX["assist"])
        # a build queues behind the build at hand: it is the unit's order
        ob.note_commands([{"verb": "build", "subject": ["7"], "bp": "pgen", "cell": (4, 4)}], 0)
        self.assertEqual(ob.last_order["7"][0], V_INDEX["build"])

    def test_the_planner_drops_the_same_units(self):
        kit = ToolKit(info(), BPS)
        units = {"7": unit("7", "eng", 30, 30, order=8), "8": unit("8", "eng", 31, 31)}
        actions, _, _ = kit.plan([{"verb": "assist", "subject": ["7", "8"], "target": "8"}], units, set())
        self.assertEqual([a["units"] for a in actions if a["type"] == "guard"], [["8"]])


class CiviliansTest(unittest.TestCase):
    def test_neutral_ghosts_and_contacts_are_no_enemy_tokens(self):
        ob = observer()
        units = [unit("1", "tank", 30, 30)]
        units += [unit(f"c{k}", "wall", 100 + k, 100, army=CIVILIAN, ghost=True) for k in range(5)]
        units += [{"id": "r1", "x": 120.0, "z": 120.0, "army": CIVILIAN, "radar": True}]
        units += [unit("g1", "pgen", 200, 200, army=ENEMY, ghost=True)]
        x = ob.observe(obs(units))
        self.assertEqual(sorted(t for t in x["ids"] if t is not None), ["1", "g1"])
        remembered = x["map"][MAP_CHANNELS.index("remembered")]
        self.assertAlmostEqual(float(np.expm1(remembered).sum()), 1.0, places=5)   # the enemy's generator only
        self.assertEqual(float(x["map"][MAP_CHANNELS.index("radar")].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
