"""Reclaim is an area command: labels carry a cell, execution queues the best props found around it."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools import CELL_VERBS, RECLAIM, RECLAIM_TARGETS, TARGET_VERBS, ToolKit, reclaim_route


class ReclaimAreaTest(unittest.TestCase):
    def setUp(self):
        self.kit = ToolKit({'map': {'width': 256, 'height': 256, 'deposits': []}},
                           {'eng': {'categories': ['MOBILE', 'ENGINEER']}})
        self.units = {'e1': {'bp': 'eng', 'x': 10.0, 'z': 10.0}, 'e2': {'bp': 'eng', 'x': 14.0, 'z': 10.0}}

    def test_reclaim_takes_a_cell_not_a_target(self):
        self.assertIn('reclaim', CELL_VERBS)
        self.assertNotIn('reclaim', TARGET_VERBS)

    def test_label_cell_from_captured_prop_position(self):
        out = self.kit.commands_from_actions(
            [{'type': 'reclaim', 'units': ['e1'], 'target': '77', 'target_pos': [200.0, 100.0]}], self.units)
        self.assertEqual(out, [{'verb': 'reclaim', 'subject': ['e1'], 'cell': self.kit.cell(200.0, 100.0)}])

    def test_label_cell_falls_back_to_the_engineers(self):
        out = self.kit.commands_from_actions(
            [{'type': 'reclaim', 'units': ['e1', 'e2'], 'target': '77'}], self.units)
        self.assertEqual(out[0]['cell'], self.kit.cell(12.0, 10.0))
        self.assertEqual(self.kit.commands_from_actions(
            [{'type': 'reclaim', 'units': ['gone'], 'target': '77'}], self.units), [])

    def test_plan_clears_then_queries_and_complete_queues_the_route(self):
        actions, queries, pending = self.kit.plan(
            [{'verb': 'reclaim', 'subject': ['e1', 'e2'], 'cell': (3, 4)}], self.units, set())
        self.assertEqual(actions, [{'type': 'clear', 'units': ['e1', 'e2']}])
        self.assertEqual(queries[0]['type'], 'reclaim')
        self.assertEqual(queries[0]['pos'], list(self.kit.centre((3, 4))))
        self.assertEqual(pending, [(('e1', 'e2'), (RECLAIM, list(self.kit.centre((3, 4)))), 1, False, True)])
        props = [{'id': i, 'x': float(i), 'z': 0.0, 'mass': float(i), 'energy': 0.0} for i in range(1, 9)]
        done = self.kit.complete(pending, [{'ok': True, 'value': props}])
        self.assertEqual([a['target'] for a in done], ['8', '7', '6', '5', '4'][:RECLAIM_TARGETS])
        self.assertTrue(all(a['type'] == 'reclaim' and a['units'] == ['e1', 'e2'] for a in done))
        # the recorded actions give the command back (the teacher's games become labels this way)
        self.assertEqual(self.kit.commands_from_actions(actions + done, self.units),
                         [{'verb': 'reclaim', 'subject': ['e1', 'e2'], 'cell': (3, 4)}])

    def test_complete_keeps_answer_offsets_with_builds(self):
        pending = [(('e1',), (RECLAIM, [1.0, 1.0]), 1, False, True), ('e2', 'bp', 1, True, True)]
        done = self.kit.complete(pending, [{'ok': True, 'value': []}, {'ok': True, 'value': [5.0, 6.0]}])
        self.assertEqual(done, [{'type': 'clear', 'units': ['e2']},
                                {'type': 'build', 'units': ['e2'], 'bp': 'bp', 'pos': [5.0, 6.0]}])

    def test_route_is_nearest_first_from_the_richest(self):
        props = [{'id': 'a', 'x': 0.0, 'z': 0.0, 'mass': 50.0}, {'id': 'b', 'x': 100.0, 'z': 0.0, 'mass': 40.0},
                 {'id': 'c', 'x': 10.0, 'z': 0.0, 'mass': 1.0, 'energy': 300.0}, {'id': 'd', 'mass': 99.0}]
        self.assertEqual([p['id'] for p in reclaim_route(props, 3)], ['a', 'c', 'b'])


if __name__ == '__main__':
    unittest.main()
