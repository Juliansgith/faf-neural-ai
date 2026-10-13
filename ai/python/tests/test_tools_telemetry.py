"""Command conversion must not hide empty group selections from diagnostic counts."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools import VERBS
from fa_train.tools_trainer import decoded_set_counts


class DecodedSetTelemetryTest(unittest.TestCase):
    def test_empty_and_nonempty_groups_are_counted_before_end_only(self):
        actions = {"verb": np.array([[VERBS.index(v) for v in ("move", "move", "build", "end", "move")]]),
                   "set": np.zeros((1, 5, 3), bool)}
        actions["set"][0, 1, 2] = True
        selected, empty = decoded_set_counts(actions, 0)
        self.assertEqual(selected, {"move": 2})
        self.assertEqual(empty, {"move": 1})


if __name__ == "__main__":
    unittest.main()
