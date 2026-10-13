import unittest

from unittest import mock

from fa_train import replay_orders as ro
from fa_train.replay_orders import army_of, to_order
from fa_train.replay_rec import orders_per_step


def issue(ctype, ids=(1048577,), target=None, bp="", name="IssueCommand", upgrades=None):
    return {"name": name, "entity_ids": list(ids), "type": ctype, "target": target, "blueprint": bp,
            "upgrades": upgrades, "clear_queue": None}


class ReplayOrdersTest(unittest.TestCase):
    def test_army_from_entity_id(self):
        self.assertEqual(army_of(0), 1)
        self.assertEqual(army_of(1048576), 2)
        self.assertEqual(army_of(2 * 1048576 + 17), 3)

    def test_move_and_build(self):
        move = to_order(issue(2, target={"x": 10.0, "y": 3.0, "z": 20.0}), 5)
        self.assertEqual(move, {"type": "move", "units": ["1048577"], "tick": 5, "pos": [10.0, 20.0]})
        build = to_order(issue(8, target={"x": 1.5, "y": 0.0, "z": 2.5}, bp="UAB1103"), 7)
        self.assertEqual((build["type"], build["bp"], build["pos"]), ("build", "uab1103", [1.5, 2.5]))
        produce = to_order(issue(7, bp="url0105"), 9)
        self.assertEqual((produce["type"], produce["bp"], produce["count"]), ("factory_build", "url0105", 1))

    def test_targets(self):
        attack = to_order(issue(10, target={"id": 805306374}), 1)
        self.assertEqual((attack["type"], attack["target"]), ("attack", "805306374"))
        ground = to_order(issue(10, target={"x": 1.0, "y": 0.0, "z": 2.0}), 1)
        self.assertEqual(ground["type"], "attack_ground")
        assist = to_order(issue(15, target={"id": 1048580}), 1)
        self.assertEqual((assist["type"], assist["target"]), ("guard", "1048580"))
        self.assertIsNone(to_order(issue(20, target=None), 1))     # a repair without a target

    def test_factory_rally_and_enhancement(self):
        rally = to_order(issue(2, ids=(1,), target={"x": 4.0, "y": 0.0, "z": 5.0}, name="IssueFactoryCommand"), 3)
        self.assertEqual((rally["type"], rally["pos"]), ("rally", [4.0, 5.0]))
        enh = to_order(issue(28, ids=(0,), upgrades={"TaskName": b"EnhanceTask", "Enhancement": b"CrysalisBeam"}), 4)
        self.assertEqual((enh["type"], enh["name"]), ("enhance", "CrysalisBeam"))
        self.assertIsNone(to_order(issue(99), 1))                   # an unknown command type


    def test_queued_factory_copies_become_one_order(self):
        five = [issue(7, ids=(1,), bp="url0106", name="IssueFactoryCommand") for _ in range(5)]
        stream = [{"name": "Advance", "ticks": 3}] + five + [issue(7, ids=(1,), bp="url0105", name="IssueFactoryCommand"),
                                                            {"name": "Advance", "ticks": 1}, five[0]]
        with mock.patch.object(ro, "Parser") as parser:
            parser.return_value.parse.return_value = {"body": {"commands": stream}}
            orders = ro.replay_orders(b"")[1]
        self.assertEqual([(t, o["bp"], o["count"]) for t, o in orders],
                         [(3, "url0106", 5), (3, "url0105", 1), (4, "url0106", 1)])


class OrdersPerStepTest(unittest.TestCase):
    def test_window_and_reclaim_position(self):
        obs = [{"tick": 0, "data": {"units": []}},
               {"tick": 20, "data": {"units": [{"id": "7", "order": 19, "dest": [30.0, 40.0]}]}},
               {"tick": 40, "data": {"units": []}}]
        orders = [(5, {"type": "reclaim", "units": ["7"], "tick": 5, "target": "805306400"}),
                  (19, {"type": "move", "units": ["8"], "tick": 19, "pos": [1.0, 2.0]}),
                  (25, {"type": "reclaim", "units": ["9"], "tick": 25, "target": "805306401"})]
        steps = orders_per_step(obs, orders)
        self.assertEqual([len(s) for s in steps], [2, 1, 0])
        self.assertEqual(steps[0][0]["target_pos"], [30.0, 40.0])      # from the next observation's engineer
        self.assertNotIn("target_pos", steps[1][0])                     # nothing to tell where it is
        self.assertNotIn("target_pos", orders[0][1])                    # the replay's own orders stay as they were


if __name__ == "__main__":
    unittest.main()
