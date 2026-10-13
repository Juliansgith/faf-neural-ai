"""Recorded policy intent must survive executor suppression and replay masking."""
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train import tools_bc as bc
from test_tools_bc import tiny_observation


class PolicyRecordingLabelsTest(unittest.TestCase):
    def encode(self, steps, policy=True):
        hello = {'hello': {'map': {'width': 256, 'height': 256}}, 'meta': {'me': 1}}
        if policy:
            hello['meta']['policy_command_labels'] = True
        observer = mock.Mock()
        observer.wrecks = []
        observer.observe.return_value = tiny_observation(1)
        rows = [{'step': {'tick': i * 2, **s}, 'msg': {'obs': {'tick': i * 2, 'units': []}}}
                for i, s in enumerate(steps)]
        labels = []
        def pack(xs, labs):
            labels.extend(labs)
            return {}
        with mock.patch.object(bc, 'read_game', return_value=(hello, rows, {'outcome': 'victory'})), \
             mock.patch.object(bc, 'ToolsObserver', return_value=observer), \
             mock.patch.object(bc, 'pack', side_effect=pack):
            result, _ = bc.encode_game(Path('unused'), {})
        return result, labels, observer

    def test_suppressed_intent_still_labels_and_updates_history(self):
        command = {'verb': 'fire_state', 'subject': ['u0'], 'option': 0}
        result, labels, observer = self.encode([
            {'policy_command_labels': True, 'commands': [command], 'sent': [], 'results': []},
            {'policy_command_labels': True, 'commands': [], 'sent': [], 'results': []},
        ])
        self.assertEqual(result['bc_supervise'].tolist(), [True, True])
        self.assertEqual(labels[0]['verb'][0], bc.V_INDEX['fire_state'])
        self.assertTrue(np.all(labels[1]['verb'] == bc.V_INDEX['end']))
        self.assertEqual(observer.note_commands.call_args_list,
                         [mock.call([command], 0, []), mock.call([], 2, [])])

    def test_masked_prefix_preserves_unfiltered_history_without_carry(self):
        command = {'verb': 'stop', 'subject': ['not_visible']}
        result, labels, observer = self.encode([
            {'policy_command_labels': True, 'commands': [command], 'supervised': False},
            {'policy_command_labels': True, 'commands': []},
        ])
        self.assertEqual(result['bc_supervise'].tolist(), [False, True])
        self.assertEqual(observer.note_commands.call_args_list[0], mock.call([command], 0, None))
        self.assertTrue(np.all(labels[1]['verb'] == bc.V_INDEX['end']))

    def test_explicit_false_and_interventions_always_mask(self):
        self.assertFalse(bc.supervise_recorded_step({'supervised': False}, [], []))
        self.assertFalse(bc.supervise_recorded_step({'supervised': True, 'teacher_interventions': [{}]}, [], []))
        self.assertTrue(bc.supervise_recorded_step({}, [], []))

    def test_inconsistent_policy_metadata_fails_closed(self):
        for step, header in [({'commands': []}, True),
                             ({'policy_command_labels': True, 'commands': None}, True),
                             ({'policy_command_labels': True, 'commands': []}, False)]:
            with self.subTest(step=step, header=header), self.assertRaisesRegex(ValueError, 'policy.command'):
                self.encode([copy.deepcopy(step)], policy=header)


if __name__ == '__main__':
    unittest.main()
