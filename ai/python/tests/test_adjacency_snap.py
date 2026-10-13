"""A building with an adjacency bonus placed by spot asks for a touching site beside a partner first."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools import ToolKit


class AdjacencySnapTest(unittest.TestCase):
    def setUp(self):
        self.bps = {
            'eng': {'categories': ['ENGINEER', 'MOBILE'], 'footprint': [1, 1]},
            'pgen': {'categories': ['STRUCTURE', 'ENERGYPRODUCTION'], 'footprint': [2, 2]},
            'fac': {'categories': ['STRUCTURE', 'FACTORY'], 'footprint': [8, 8]},
            'mex': {'categories': ['STRUCTURE', 'MASSEXTRACTION'], 'footprint': [2, 2]},
            'wall': {'categories': ['STRUCTURE', 'WALL'], 'footprint': [1, 1]},
        }
        self.kit = ToolKit({'map': {'width': 256, 'height': 256, 'deposits': []}}, self.bps)
        for b in self.bps.values():
            b.setdefault('can_build', [])

    def plan(self, bp, units):
        cmd = {'verb': 'build', 'subject': ['e'], 'bp': bp, 'place': 'cell', 'cell': (3, 3), 'spot': (1, 1)}
        units = dict(units, e={'bp': 'eng', 'x': 0, 'z': 0, 'can_build': [bp]})
        return self.kit.plan([cmd], units, set())

    def test_power_generator_beside_a_nearby_factory(self):
        x, z = self.kit.spot_centre((3, 3), (1, 1))
        _, queries, pending = self.plan('pgen', {'f': {'bp': 'fac', 'x': x + 3, 'z': z}})
        self.assertEqual(queries[0]['adjacent_to'], ['f'])
        self.assertTrue(queries[0]['adjacent_only'])
        self.assertNotIn('adjacent_to', queries[1])                       # the chosen spot as the fallback
        self.assertEqual(pending[0][2], 2)

    def test_no_partner_far_away_or_without_a_bonus(self):
        x, z = self.kit.spot_centre((3, 3), (1, 1))
        _, far, _ = self.plan('pgen', {'f': {'bp': 'fac', 'x': x + 40, 'z': z}})
        self.assertNotIn('adjacent_to', far[0])
        _, wall, _ = self.plan('wall', {'f': {'bp': 'fac', 'x': x + 2, 'z': z}})
        self.assertNotIn('adjacent_to', wall[0])

    def test_partners_nearest_first(self):
        x, z = self.kit.spot_centre((3, 3), (1, 1))
        partners = self.kit.adjacency_partners('pgen', x, z, {'a': {'bp': 'fac', 'x': x + 5, 'z': z},
                                                              'b': {'bp': 'fac', 'x': x + 1, 'z': z},
                                                              'm': {'bp': 'mex', 'x': x, 'z': z}})
        self.assertEqual(partners, ['b', 'a'])                             # extractors give a generator nothing


if __name__ == "__main__":
    unittest.main()
