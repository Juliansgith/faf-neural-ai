"""The game schedule must support paired comparisons and varied starts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.tools_eval import schedule_rows  # noqa: E402


class ToolsEvalScheduleTest(unittest.TestCase):
    def test_six_game_smoke_uses_both_starts_with_one_map_and_faction(self):
        rows = schedule_rows(["easy"], 6, ["SCMP_016"], [1], [1], None, 900000)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["slots"] for row in rows}, {(1, 2), (2, 1)})

    def test_opponents_get_paired_and_diverse_games(self):
        rows = schedule_rows(["easy", "medium"], 8, ["SCMP_016", "SCMP_019"],
                             [1, 2, 3, 4], [1, 2, 3, 4], None, 900000)
        self.assertEqual(len(rows), 16)
        easy, medium = rows[:8], rows[8:]
        self.assertEqual([{k: v for k, v in row.items() if k != "ai"} for row in easy],
                         [{k: v for k, v in row.items() if k != "ai"} for row in medium])
        self.assertEqual({row["map"] for row in easy}, {"SCMP_016", "SCMP_019"})
        self.assertEqual({row["slots"] for row in easy if row["map"] == "SCMP_016"},
                         {(1, 2), (2, 1)})
        self.assertEqual({row["faction"] for row in easy}, {1, 2, 3, 4})
        self.assertEqual({(row["map"], row["faction"]) for row in easy},
                         {(m, f) for m in ("SCMP_016", "SCMP_019") for f in (1, 2, 3, 4)})
        self.assertGreater(len({(row["faction"], row["ai_faction"]) for row in easy}), 4)
        self.assertEqual(len({row["seed"] for row in easy}), 8)

    def test_sixteen_games_cross_both_starts_for_every_map_and_faction(self):
        rows = schedule_rows(["easy"], 16, ["SCMP_016", "SCMP_019"],
                             [1, 2, 3, 4], [1, 2, 3, 4], None, 900000)
        self.assertEqual(len({(r["map"], r["faction"], r["slots"]) for r in rows}), 16)


if __name__ == "__main__":
    unittest.main()
