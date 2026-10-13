"""A unit at a build keeps it: replacing orders skip it; STOP still cancels."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools import ToolKit


class KeepsBuildsTest(unittest.TestCase):
    def setUp(self):
        self.kit = ToolKit({'map': {'width': 256, 'height': 256, 'deposits': []}},
                           {'acu': {'categories': ['COMMAND', 'MOBILE']}, 'eng': {'categories': ['ENGINEER', 'MOBILE']}})

    def plan(self, command, units):
        return self.kit.plan([command], units, set())[0]

    def test_building_commander_ignores_a_move(self):
        units = {'a': {'bp': 'acu', 'x': 0, 'z': 0, 'order': 8}}
        self.assertEqual(self.plan({'verb': 'commander_move', 'subject': ['a'], 'cell': (5, 5)}, units), [])

    def test_idle_commander_moves(self):
        units = {'a': {'bp': 'acu', 'x': 0, 'z': 0, 'order': 2}}
        acts = self.plan({'verb': 'commander_move', 'subject': ['a'], 'cell': (5, 5)}, units)
        self.assertEqual([a['type'] for a in acts], ['clear', 'move'])

    def test_set_leaves_out_its_builders(self):
        units = {'e1': {'bp': 'eng', 'x': 0, 'z': 0, 'order': 8}, 'e2': {'bp': 'eng', 'x': 0, 'z': 0}}
        acts = self.plan({'verb': 'move', 'subject': ['e1', 'e2'], 'cell': (5, 5)}, units)
        self.assertEqual(acts[0], {'type': 'clear', 'units': ['e2']})
        self.assertEqual(acts[1]['units'], ['e2'])

    def test_stop_leaves_a_build_alone_but_stops_others(self):
        units = {'e1': {'bp': 'eng', 'x': 0, 'z': 0, 'order': 8}, 'e2': {'bp': 'eng', 'x': 0, 'z': 0, 'order': 2}}
        acts = self.plan({'verb': 'stop', 'subject': ['e1', 'e2']}, units)
        self.assertEqual(acts, [{'type': 'clear', 'units': ['e2']}, {'type': 'stop', 'units': ['e2']}])

    def test_a_busy_builder_takes_a_few_more_builds_only(self):
        self.kit.bps['p'] = {'categories': ['STRUCTURE']}
        cmd = {'verb': 'build', 'subject': ['e1'], 'bp': 'p', 'place': 'cell', 'cell': (3, 3), 'spot': (1, 1)}
        busy = {'e1': {'bp': 'eng', 'x': 0, 'z': 0, 'order': 8, 'queue': 3, 'can_build': ['p']}}
        self.assertEqual(self.kit.plan([cmd], busy, set()), ([], [], []))
        busy['e1']['queue'] = 1
        _, queries, pending = self.kit.plan([cmd], busy, set())
        self.assertEqual(len(queries), 1)
        self.assertFalse(pending[0][3])          # queued behind the current build, no clear


if __name__ == '__main__':
    unittest.main()
