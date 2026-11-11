"""A productive experiment (it produced a champion) runs again before untried queue entries (2026-10-22: X3 made the
first champion past the plateau and would otherwise have waited behind X4..X7)."""
try:                                   # first: no test may reach the server or start a real process
    import no_network  # noqa: F401
except ImportError:
    from tests import no_network  # noqa: F401
import unittest

from fa_train.autopilot import next_experiment

QUEUE = [{"name": "X1"}, {"name": "X3"}, {"name": "X4"}, {"name": "X6"}]


class RequeueTest(unittest.TestCase):
    def test_productive_first(self):
        results = {"X1": {"verdict": "FAIL"}, "X3": {"verdict": "FAIL", "promotions": 1, "requeue": True}}
        e = next_experiment(QUEUE, results, "best_stalled")
        self.assertEqual((e["name"], e.get("_requeue")), ("X3", True))

    def test_then_untried_entries(self):
        results = {"X1": {"verdict": "FAIL"},
                   "X3": {"verdict": "FAIL", "promotions": 1, "requeue": True, "requeued_done": True}}
        self.assertEqual(next_experiment(QUEUE, results, "best_stalled")["name"], "X4")

    def test_a_champion_again_reopens_it(self):
        results = {"X1": {"verdict": "FAIL"}, "X4": {"verdict": "FAIL"}, "X6": {"verdict": "STALL"},
                   "X3": {"verdict": "FAIL", "promotions": 2, "requeue": True, "requeued_done": False}}
        self.assertEqual(next_experiment(QUEUE, results, "best_stalled")["name"], "X3")


if __name__ == "__main__":
    unittest.main()
