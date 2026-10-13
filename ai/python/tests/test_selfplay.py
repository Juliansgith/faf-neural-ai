"""Self-play's shared bridge (fa_env.selfplay): trips meet, go as one, come back split."""
import threading
import unittest

from fa_env.bridge import GameConfig
from fa_env.selfplay import SelfPlayGame, side_info


class FakeFA:
    def __init__(self):
        self.sent = []
        self.last_message = None

    def step(self, actions, queries, wait):
        self.sent.append((actions, queries, wait))
        obs = {"me": 1, "tick": 20, "armies": [{"index": 1, "relation": "self"}, {"index": 2, "relation": "enemy"}],
               "units": [{"id": "1", "army": 1}], "privileged": {"1": {}, "2": {}},
               "orders": [{"type": "move", "units": ["1"], "army": 1}, {"type": "move", "units": ["9"], "army": 2}],
               "views": {"2": {"me": 2, "tick": 20, "units": [{"id": "9", "army": 2}],
                               "armies": [{"index": 1, "relation": "enemy"}, {"index": 2, "relation": "self"}]}}}
        self.last_message = {"type": "obs", "obs": obs, "tick": 20,
                             "result": {"actions": [{"ok": True, "i": i} for i in range(len(actions))],
                                        "queries": [{"answer": i} for i in range(len(queries))]}}
        info = {"results": self.last_message["result"]["actions"], "queries": self.last_message["result"]["queries"]}
        return obs, 0.0, False, False, info

    def close(self):
        pass


class SelfPlayLinkTest(unittest.TestCase):
    def setUp(self):
        self.game = SelfPlayGame(GameConfig(map="SCMP_012"))
        self.game.fa = FakeFA()

    def trip(self, side, actions, queries, out):
        out[side] = self.game.sides[side].fa.step(actions, queries, wait=1)

    def test_one_trip_for_both_sides_split_back(self):
        out = [None, None]
        threads = [threading.Thread(target=self.trip, args=(0, [{"type": "move", "units": ["1"]}], [], out)),
                   threading.Thread(target=self.trip, args=(1, [{"type": "stop", "units": ["9"]},
                                                                {"type": "move", "units": ["9"]}],
                                                            [{"type": "can_build"}], out))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(len(self.game.fa.sent), 1)                         # one trip to the game
        actions, queries, wait = self.game.fa.sent[0]
        self.assertEqual([a["army"] for a in actions], [1, 2, 2])           # each order names its army
        self.assertEqual([q["army"] for q in queries], [2])
        obs_a, _, _, _, info_a = out[0]
        obs_b, _, _, _, info_b = out[1]
        self.assertEqual([r["i"] for r in info_a["results"]], [0])           # each side its own results ...
        self.assertEqual([r["i"] for r in info_b["results"]], [1, 2])
        self.assertEqual(info_a["queries"], [])                              # ... and query answers
        self.assertEqual(len(info_b["queries"]), 1)
        self.assertEqual(obs_b["me"], 2)                                     # ... its own view
        self.assertNotIn("views", obs_a)
        self.assertEqual([o["army"] for o in obs_a["orders"]], [1])          # ... its own shadow orders
        self.assertEqual([o["army"] for o in obs_b["orders"]], [2])
        self.assertIn("privileged", obs_b)
        self.assertIn(b'"me": 2', self.game.sides[1].fa.server.last_line)     # its recording line

    def test_side_two_sees_itself(self):
        info = {"armies": [{"index": 1, "relation": "self"}, {"index": 2, "relation": "enemy"},
                           {"index": 3, "relation": "neutral", "civilian": True}]}
        self.assertEqual([a["relation"] for a in side_info(info, 2)["armies"]], ["enemy", "self", "neutral"])


if __name__ == "__main__":
    unittest.main()
