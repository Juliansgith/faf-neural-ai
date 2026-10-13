import unittest
from fa_env.tools_diagnostics import action_failures, transition_record


class DiagnosticsTest(unittest.TestCase):
    def test_both_bridge_batches_and_missing_description(self):
        sent = [{"type": "move", "units": [12]}, {"type": "build", "units": [13], "bp": "ueb1103"}]
        count, details = action_failures(sent, [{"ok": False, "error": "unit gone"}, {"ok": False}])
        self.assertEqual(count, 2)
        self.assertEqual(details[0]["type"], "move")
        self.assertEqual(details[1]["bp"], "ueb1103")
        self.assertIn("no error description", details[1]["error"])

    def test_bounded_details_do_not_reduce_total(self):
        count, details = action_failures([], [{"ok": False}] * 25)
        self.assertEqual(count, 25)
        self.assertEqual(len(details), 16)

    def test_actual_terminal_transition_and_unweighted_components(self):
        reward = {"trade": 0.2, "commander": -0.5, "terminal": 12.1}
        record = transition_record(100, 120, reward, True, "victory", 18000)
        reward["terminal"] = 0
        self.assertEqual(record["reward_components"]["terminal"], 12.1)
        self.assertEqual(record["end_tick"], 120)
        self.assertTrue(record["done"])


if __name__ == "__main__":
    unittest.main()
