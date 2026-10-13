"""Observer eligibility must reach the real production planner, not just raw actions."""
import unittest

from fa_env.tools import ToolKit
from fa_env.tools_obs import ToolsObserver


class FactoryProductionPlanTest(unittest.TestCase):
    def setUp(self):
        self.bps = {
            'factory': {'categories': ['FACTORY', 'STRUCTURE'], 'tech': 1,
                        'buildable': ['engineer', 'upgrade', 't2']},
            'engineer': {'categories': ['MOBILE', 'ENGINEER'], 'tech': 1},
            'upgrade': {'categories': ['STRUCTURE', 'FACTORY'], 'tech': 1},
            't2': {'categories': ['MOBILE'], 'tech': 2},
        }
        self.kit = ToolKit({'map': {'width':256, 'height':256, 'deposits':[]}}, self.bps)
        self.observer = ToolsObserver.__new__(ToolsObserver)
        self.observer.bps = self.bps
        self.observer.place_class = {'factory':0, 'upgrade':0}

    def plan(self, unit, bp='engineer'):
        return self.kit.plan([{'verb':'produce', 'subject':['f'], 'bp':bp, 'count':1}],
                             {'f':unit}, set())[0]

    def test_native_missing_choices_reaches_factory_build(self):
        u = {'bp':'factory', 'frac':.95}
        self.assertIn('engineer', self.observer.can_build_of(u))
        self.assertEqual(self.plan(u), [{'type':'factory_build', 'units':['f'], 'bp':'engineer', 'count':1}])

    def test_explicit_engine_restriction_is_authoritative(self):
        self.assertEqual(self.plan({'bp':'factory', 'frac':.95, 'can_build':[]}), [])

    def test_early_factory_and_unfinished_engineer_blocked(self):
        for fraction in [0, .5, .899]:
            self.assertEqual(self.plan({'bp':'factory', 'frac':fraction}), [])
        self.assertEqual(self.plan({'bp':'engineer', 'frac':.95, 'can_build':['engineer']}), [])

    def test_near_complete_factory_no_structure_or_advanced_fallback(self):
        u = {'bp':'factory', 'frac':.95}
        self.assertEqual(self.plan(u, 'upgrade'), [])
        self.assertEqual(self.plan(u, 't2'), [])


if __name__ == '__main__':
    unittest.main()
