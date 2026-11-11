"""The commander leash on large maps: an option, off by default (2026-10-17, architecture review P2a)."""
import math
import unittest

from fa_env.tools import LEASH_MIN_UNITS, ToolKit


class CommanderLeashTest(unittest.TestCase):
    def kit(self, size, fraction=0.5):
        k = ToolKit({"map": {"width": size, "height": size, "deposits": []}}, {"acu": {"categories": ["COMMAND"]}})
        k.set_commander_leash((100.0, 100.0), (size - 100.0, size - 100.0), fraction)
        return k

    def test_off_by_default(self):
        k = ToolKit({"map": {"width": 1024.0, "height": 1024.0, "deposits": []}}, {})
        self.assertIsNone(k.leash)
        self.assertEqual(k.leashed((900.0, 900.0)), [900.0, 900.0])

    def test_large_maps_only(self):
        self.assertIsNone(self.kit(512.0).leash)                          # 10 km: not above 10 km
        self.assertIsNone(self.kit(1024.0, 0.0).leash)
        self.assertGreater(1024.0, LEASH_MIN_UNITS)
        self.assertIsNotNone(self.kit(1024.0).leash)

    def test_commander_moves_end_at_the_leash(self):
        k = self.kit(1024.0)
        r = 0.5 * math.hypot(824.0, 824.0)
        units = {"1": {"id": "1", "bp": "acu", "x": 100.0, "z": 100.0}, "2": {"id": "2", "bp": "x", "x": 0, "z": 0}}
        far = k.place(1000.0, 1000.0)
        actions, _, _ = k.plan([{"verb": "commander_attack_move", "subject": ["1"], **far},
                                {"verb": "attack_move", "subject": ["2"], **far}], units, set())
        cmd = next(a for a in actions if a["type"] == "attack_move" and a["units"] == ["1"])
        self.assertAlmostEqual(math.hypot(cmd["pos"][0] - 100.0, cmd["pos"][1] - 100.0), r, places=3)
        self.assertAlmostEqual(cmd["pos"][0], cmd["pos"][1], places=3)                  # on the line from home
        army = next(a for a in actions if a["type"] == "attack_move" and a["units"] == ["2"])
        self.assertEqual(tuple(army["pos"]), k.where(far["cell"], far["sub"]))          # the army goes anywhere
        near = k.place(150.0, 120.0)
        actions, _, _ = k.plan([{"verb": "commander_move", "subject": ["1"], **near}], units, set())
        self.assertEqual(tuple(next(a for a in actions if a["type"] == "move")["pos"]),
                         k.where(near["cell"], near["sub"]))


if __name__ == "__main__":
    unittest.main()
