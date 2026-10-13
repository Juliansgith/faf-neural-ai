import unittest

from fa_env.tools_obs import _threat_profile


def weapon(label, category, rng, damage, rof):
    return {"label": label, "rangeCategory": category, "range": rng, "damage": damage, "rof": rof}


class ThreatProfileTest(unittest.TestCase):
    def test_commander_counts_its_main_gun_only(self):
        acu = {"categories": ["COMMAND", "MOBILE"], "weapons": [
            weapon("RightZephyr", "UWRC_DirectFire", 22, 100, 1),
            weapon("TacMissile", "UWRC_IndirectFire", 256, 6000, 5)]}       # an upgrade's weapon
        ground, air, shield = _threat_profile(acu)
        self.assertEqual(ground, ((100.0, 22.0),))
        self.assertEqual((air, shield), ((), 0.0))

    def test_each_weapon_covers_its_own_range(self):
        tank = {"categories": ["MOBILE", "LAND"], "weapons": [
            weapon("Gun", "UWRC_DirectFire", 20, 30, 1), weapon("Arty", "UWRC_IndirectFire", 60, 10, 0.5),
            weapon("AA", "UWRC_AntiAir", 40, 20, 2), weapon("Nuke", "UWRC_IndirectFire", 20000, 0, 1)]}
        ground, air, _ = _threat_profile(tank)
        self.assertEqual(ground, ((30.0, 20.0), (5.0, 60.0)))                # the zero-damage nuke slot is skipped
        self.assertEqual(air, ((40.0, 40.0),))

    def test_shield_radius(self):
        self.assertEqual(_threat_profile({"categories": ["STRUCTURE"], "shieldSize": 24})[2], 12.0)


if __name__ == "__main__":
    unittest.main()
