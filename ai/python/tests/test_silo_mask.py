"""silo_build from the unit's enhancements and silo weapons (2026-10-17 review, bugs-env F3 / B14)."""
import unittest

from fa_env.tools import silo_kinds


class SiloMaskTest(unittest.TestCase):
    MISSILE = {"label": "TacMissile", "category": "Missile", "rangeCategory": "UWRC_IndirectFire"}
    NUKE = {"label": "TacNukeMissile", "category": "Missile", "rangeCategory": "UWRC_IndirectFire"}
    AA = {"label": "NMissile", "category": "Anti Air", "rangeCategory": "UWRC_AntiAir"}

    def test_commanders_need_the_enhancement(self):
        uef = {"categories": ["COMMAND", "SILO", "MOBILE"], "weapons": [self.MISSILE, self.NUKE]}
        self.assertEqual(silo_kinds({}, uef), set())                                       # minute 0
        self.assertEqual(silo_kinds({"enh": ["TacticalMissile"]}, uef), {"tactical"})
        self.assertEqual(silo_kinds({"enh": ["TacticalNukeMissile"]}, uef), {"nuke"})
        sera_sacu = {"categories": ["SUBCOMMANDER", "SILO", "MOBILE"],
                     "weapons": [{"label": "Missile", "category": "Missile", "rangeCategory": "UWRC_IndirectFire"}]}
        self.assertEqual(silo_kinds({}, sera_sacu), set())
        self.assertEqual(silo_kinds({"enh": ["Missile"]}, sera_sacu), {"tactical"})

    def test_an_anti_air_launcher_is_no_silo(self):
        cybran_sacu = {"categories": ["SUBCOMMANDER", "MOBILE"], "weapons": [self.AA]}
        self.assertEqual(silo_kinds({"enh": ["NaniteMissileSystem"]}, cybran_sacu), set())

    def test_launchers_unchanged(self):
        self.assertEqual(silo_kinds({}, {"categories": ["STRUCTURE", "TACTICALMISSILEPLATFORM"]}), {"tactical"})
        self.assertEqual(silo_kinds({}, {"categories": ["STRUCTURE", "NUKE", "SILO"]}), {"nuke"})


if __name__ == "__main__":
    unittest.main()
