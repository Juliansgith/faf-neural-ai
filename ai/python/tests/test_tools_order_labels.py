"""Chronology, partial label groups, and non-destructive deferred placement regressions."""
import sys
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools import ToolKit, merge_commands, VERBS
from fa_train import tools_bc as bc
from test_tools_bc import tiny_observation
import test_tools_model
from fa_train import tools_model as tm


class OrderLabelsTest(unittest.TestCase):
    def setUp(self):
        self.k = ToolKit({'map': {'width': 256, 'height': 256, 'deposits': [
            {'x': 64, 'z': 64, 'type': 'mass'}]}}, {
            'p': {'categories': ['STRUCTURE']}, 'mex': {'categories': ['STRUCTURE', 'MASSEXTRACTION']},
            'acu': {'categories': ['COMMAND']}})
        self.units = {'u': {'bp': 'p', 'x': 0, 'z': 0, 'can_build': ['p', 'mex'], 'order': 1},
                      'v': {'bp': 'p', 'x': 0, 'z': 0}, 'a': {'bp': 'acu', 'x': 0, 'z': 0}}

    def action(self, t, subjects=('u',), **kw):
        return dict(type=t, units=list(subjects), **kw)

    def convert(self, *actions, prior=()):
        out = self.k.commands_from_actions(list(actions), self.units, prior)
        self.assertTrue(all(c['verb'] in VERBS for c in out))
        return out

    def test_original_four_repros_and_controls(self):
        for first in ('patrol', 'attack'):
            a = self.action(first, **({'target': 'enemy'} if first == 'attack' else {'pos': [200, 0]}))
            self.assertEqual([c['verb'] for c in self.convert(a, self.action('move', pos=[0, 200]))], ['move'])
        self.assertEqual([c['verb'] for c in self.convert(self.action('attack_move', ['a'], pos=[200, 0]),
            self.action('move', ['a'], pos=[0, 200]))], ['commander_move'])
        self.assertEqual([c['verb'] for c in self.convert(self.action('move', pos=[0, 200]),
            self.action('fire_state', state=1), self.action('dive'))], ['move', 'dive', 'fire_state'])

    def test_clear_across_classes_and_carry(self):
        build = self.action('build', ['a'], bp='mex', pos=[64, 64])
        move = self.action('move', ['a'], pos=[0, 200])
        clear = self.action('clear', ['a'])
        self.assertEqual([c['verb'] for c in self.convert(build, clear, move)], ['commander_move'])
        self.assertEqual([c['verb'] for c in self.convert(move, clear, build)], ['build'])
        self.assertEqual(self.convert(build, clear), [])
        prior = [{'verb': 'move', 'subject': ['u', 'v'], 'cell': (1, 1), 'carried': True},
                 {'verb': 'build', 'subject': ['a'], 'bp': 'mex', 'cell': (8, 8)}]
        out = self.convert(self.action('clear', ['u', 'a']), self.action('move', pos=[0, 200]), prior=prior)
        self.assertEqual([(c['subject'], c.get('carried', False)) for c in out], [(['v'], True), (['u'], False)])

    def test_patrol_runs_clear_and_unrelated_events(self):
        out = self.convert(self.action('patrol', pos=[200, 0]), self.action('fire_state', state=1),
            self.action('clear', ['v']), self.action('patrol', pos=[0, 50]))
        self.assertEqual(out[0]['cell'], (25, 0))
        out = self.convert(self.action('patrol', pos=[200, 0]), self.action('clear'),
            self.action('patrol', pos=[0, 50]))
        self.assertEqual(out[0]['cell'], (0, 6))
        self.assertEqual([c['verb'] for c in self.convert(self.action('patrol', pos=[200, 0]),
            self.action('build', bp='p', pos=[64, 64]), self.action('patrol', pos=[0, 50]))], ['build', 'patrol'])

    def test_queue_accumulation_and_dependency_order(self):
        out = self.convert(self.action('build', bp='p', pos=[64, 64]), self.action('move', pos=[0, 200]))
        self.assertEqual([c['verb'] for c in out], ['build', 'move'])
        out = self.convert(self.action('factory_build', bp='p', count=1), self.action('factory_build', bp='mex', count=1))
        self.assertEqual(len(out), 2)
        x = tiny_observation(1)
        x.update(cand_ids=['p', 'mex'], cand_kind=np.array([bc.CAND_MOBILE]*2),
                 can_build=np.array([[0, 0], [0, 1]]))
        out = [dict(c, subject=['u0']) for c in out]
        keep, diag = bc.filter_commands(out, x)
        self.assertEqual(keep, out[:1])
        self.assertEqual(diag['rejections'], {'unrepresentable_queue_duplicate': 1})
        self.assertEqual(bc.commands_to_indices(keep, x)['verb'][:2].tolist(), [bc.V_INDEX['produce'], bc.V_INDEX['end']])

    def test_queued_duplicates_filter_to_actual_decoder_support(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for verb in ('build', 'produce', 'upgrade', 'silo_build'):
                with self.subTest(verb=verb):
                    v = bc.V_INDEX[verb]
                    net, inputs = test_tools_model.DecoderExhaustionTest().fixture([[0, 1]])
                    b = inputs[3]
                    b['mask_verb'].zero_()
                    b['mask_verb'][0, [tm.V_END, v]] = True
                    b['mask_subject'].zero_()
                    b['mask_subject'][0, v, :] = True
                    b['mask_silo'][0, :, 0] = True
                    kind = bc.CAND_MOBILE if verb == 'produce' else 2
                    b['cand_kind'][:] = kind
                    with torch.no_grad():
                        net.verb_head.bias.zero_()
                        net.verb_head.bias[v] = 100
                    x = tiny_observation(2)
                    x.update(cand_ids=['bp'], cand_kind=np.array([kind]),
                             can_build=np.array([[0, 0], [1, 0]]),
                             mask_build_cell=np.ones((4, 32, 32), bool),
                             mask_silo=np.array([[True, False], [True, False]]))
                    x['mask_subject'][tm.V_BUILD, :] = verb == 'build'
                    args = ({'cell': (8, 8), 'place': 'cell'} if verb == 'build' else
                            {'count': 1} if verb == 'produce' else {'option': 0} if verb == 'silo_build' else {})
                    commands = [dict(verb=verb, subject=[u], bp='bp', **args)
                                for u in ('u0', 'u0', 'u1')]
                    keep, diag = bc.filter_commands(commands, x)
                    self.assertEqual(keep, [commands[0], commands[2]])
                    self.assertEqual(diag['rejections'], {'unrepresentable_queue_duplicate': 1})
                    total = bc._new_command_diagnostics('test')
                    bc._merge_command_diagnostics(total, diag)
                    self.assertEqual(total['per_verb'][verb]['rejections'], diag['rejections'])
                    def score(labels):
                        actions = {key: torch.as_tensor(value).unsqueeze(0)
                                   for key, value in tm.commands_to_indices(labels, x).items()}
                        with torch.no_grad():
                            _, logp, _, loss, _ = net.decode(*inputs, actions=actions, smoothing=0.05)
                        return logp.item(), loss.item()
                    raw_logp, raw_loss = score(commands)
                    logp, loss = score(keep)
                    self.assertLess(raw_logp, -1e8)
                    self.assertGreater(raw_loss, 1e8)
                    self.assertGreater(logp, -1000)
                    self.assertLess(loss, 1000)
        finally:
            torch.set_num_threads(previous_threads)

    def test_failed_build_does_not_clear(self):
        for args in ({'bp': 'bad', 'cell': (1, 1)}, {'bp': 'p', 'cell': (-1, 0)},
                     {'bp': 'p', 'cell': (1, 1), 'spot': (5, 0)}, {'bp': 'mex', 'cell': (1, 1)},
                     {'bp': 'p', 'place': 'next', 'anchor': 'missing'}):
            self.assertEqual(self.k.plan([dict(verb='build', subject=['u'], **args)], self.units, set()), ([], [], []))
        actions, queries, pending = self.k.plan([{'verb': 'build', 'subject': ['u'], 'bp': 'p', 'cell': (1, 1)}], self.units, set())
        self.assertEqual(actions, [])
        self.assertEqual(self.k.complete(pending, [{'ok': False}]), [])
        self.assertEqual([a['type'] for a in self.k.complete(pending, [{'ok': True, 'value': [13, 13]}])], ['clear', 'build'])
        actions, _, _ = self.k.plan([{'verb': 'build', 'subject': ['u'], 'bp': 'mex', 'cell': (8, 8)}], self.units, set())
        self.assertEqual([a['type'] for a in actions], ['clear', 'build'])
        self.units['u']['order'] = 8
        _, _, pending = self.k.plan([{'verb': 'build', 'subject': ['u'], 'bp': 'p', 'cell': (1, 1)}], self.units, set())
        self.assertEqual([a['type'] for a in self.k.complete(pending, [{'ok': True, 'value': [13, 13]}])], ['build'])

    def test_mixed_delayed_build_and_move_precedence(self):
        b = {'verb': 'build', 'subject': ['u'], 'bp': 'p', 'cell': (1, 1)}
        m = {'verb': 'move', 'subject': ['u'], 'cell': (2, 2)}
        actions, _, pending = self.k.plan([b, m], self.units, set())
        self.assertEqual([a['type'] for a in actions], ['clear', 'move'])
        self.assertEqual(self.k.complete(pending, [{'ok': True, 'value': [13, 13]}]), [])
        actions, _, pending = self.k.plan([m, b], self.units, set())
        self.assertEqual([a['type'] for a in actions + self.k.complete(pending, [{'ok': True, 'value': [13, 13]}])], ['clear', 'move', 'clear', 'build'])
        self.assertEqual([a['type'] for a in actions + self.k.complete(pending, [{'ok': False}])], ['clear', 'move'])

    def test_stop_cancels_only_its_units_pending_builds(self):
        self.units['v']['can_build'] = ['p']
        build = lambda u: dict(verb='build', subject=[u], bp='p', cell=(1, 1))
        actions, queries, pending = self.k.plan([build('u'), build('v'),
            {'verb': 'stop', 'subject': ['u']}], self.units, set())
        self.assertEqual(actions, [self.action('clear'), self.action('stop')])
        self.assertEqual(len(queries), 2)
        completed = self.k.complete(pending, [{'ok': True, 'value': [13, 13]},
                                               {'ok': True, 'value': [14, 14]}])
        self.assertEqual(completed, [self.action('clear', ['v']), self.action('build', ['v'], bp='p', pos=[14, 14])])
        actions, _, pending = self.k.plan([{'verb': 'stop', 'subject': ['u']}, build('u')], self.units, set())
        self.assertEqual([a['type'] for a in actions + self.k.complete(pending,
            [{'ok': True, 'value': [13, 13]}])], ['clear', 'stop', 'clear', 'build'])

    def test_successful_immediate_build_cancels_deferred_replacement_only(self):
        deferred = dict(verb='build', subject=['u'], bp='p', cell=(1, 1))
        deposit = dict(verb='build', subject=['u'], bp='mex', cell=(8, 8))
        answer = [{'ok': True, 'value': [13, 13]}]
        actions, _, pending = self.k.plan([deferred, deposit], self.units, set())
        self.assertEqual(actions, [self.action('clear'), self.action('build', bp='mex', pos=[64, 64])])
        self.assertEqual(self.k.complete(pending, answer), [])
        for later in (dict(deposit, cell=(1, 1)), dict(deposit, bp='bad'),
                      dict(deposit, cell=(-1, 1)), dict(deferred, place='next', anchor='missing')):
            actions, _, pending = self.k.plan([deferred, later], self.units, set())
            self.assertEqual(actions, [])
            self.assertEqual([a['type'] for a in self.k.complete(pending, answer)], ['clear', 'build'])
            self.assertEqual(self.k.complete(pending, [{'ok': False}]), [])
        self.units['u']['order'] = 8
        actions, _, pending = self.k.plan([deferred, deposit], self.units, set())
        self.assertEqual([a['type'] for a in actions + self.k.complete(pending, answer)], ['build', 'build'])
        self.units['u']['order'] = 1
        self.units['v']['can_build'] = ['mex']
        actions, _, pending = self.k.plan([deferred, dict(deposit, subject=['v'])], self.units, set())
        self.assertEqual([a['units'] for a in actions + self.k.complete(pending, answer)], [['v'], ['v'], ['u'], ['u']])


class PartialSubjectsTest(unittest.TestCase):
    def test_encode_masks_raw_intent_lost_before_labels(self):
        # Exercise encode_game's actual conversion, carry, filtering and label encoding. The
        # observation and array packing are stubs because this regression concerns supervision.
        k = OrderLabelsTest()
        k.setUp()
        turns = [[], [k.action('clear')], [k.action('unsupported')],
                 [k.action('build', bp='p', pos=[64, 64]), k.action('clear')],
                 [k.action('move', ['fresh'], pos=[0, 200])],
                 [k.action('clear', ['fresh'])], []]
        steps = [{'step': {'tick': i, 'sent': sent},
                  'msg': {'obs': {'tick': i, 'units': [dict(id='u', army=1, **k.units['u'])]}}}
                 for i, sent in enumerate(turns)]
        x = tiny_observation(1)
        x['ids'] = ['u']
        x['mask_move_cell'] = np.ones((32, 32), bool)
        observer = mock.Mock()
        observer.wrecks = []
        observer.observe.return_value = x
        packed_labels = []
        def pack(observations, labels):
            packed_labels.extend(labels)
            return {}
        hello = {'hello': {'map': {'width': 256, 'height': 256}}, 'meta': {'me': 1}}
        with mock.patch.object(bc, 'read_game', return_value=(hello, steps, {'outcome': 'victory'})), \
             mock.patch.object(bc, 'ToolsObserver', return_value=observer), \
             mock.patch.object(bc, 'pack', side_effect=pack):
            out, stats = bc.encode_game(Path('unused'), k.k.bps)
        self.assertEqual(out['bc_supervise'].tolist(), [True, False, False, False, False, False, True])
        self.assertTrue(all(np.all(lab['verb'] == tm.V_END) for lab in packed_labels))
        self.assertEqual(stats['command_diagnostics']['nonempty_unrepresentable_turns'], 5)
        self.assertEqual(stats['command_diagnostics']['clear_only_turns'], 2)
        total = bc._new_command_diagnostics('test')
        bc._merge_command_diagnostics(total, stats['command_diagnostics'])
        self.assertEqual(total['nonempty_unrepresentable_turns'], 5)
        self.assertEqual(total['clear_only_turns'], 2)

    def test_present_missing_masked_and_retry(self):
        x = tiny_observation(3)
        x['mask_subject'][bc.V_INDEX['stop'], 2] = False
        c = {'verb': 'stop', 'subject': ['u0', 'u1', 'u2', 'fresh']}
        keep, diag = bc.filter_commands([c], x)
        self.assertEqual(keep, [dict(c, subject=['u0', 'u1'])])
        self.assertEqual(diag['after_filter'], 1)
        self.assertEqual(diag['partial_subjects'], {'retained_subjects': 2, 'missing_subjects': 1,
            'masked_subjects': 1, 'partially_retained_groups': 1})
        total = bc._new_command_diagnostics('test')
        bc._merge_command_diagnostics(total, diag)
        self.assertEqual(total['partial_subjects'], diag['partial_subjects'])
        carried = bc.carry_commands([c], x, bc.lookups(x))
        self.assertEqual(carried, [dict(c, subject=['fresh'], carried=True)])
        self.assertEqual(bc.carry_commands(carried, x, bc.lookups(x)), [])
        self.assertEqual(bc.commands_to_indices(keep, x)['verb'][0], bc.V_INDEX['stop'])

    def test_bad_shared_target_rejects_all_and_never_carries(self):
        x = tiny_observation(2)
        x['mask_target'] = np.ones((len(bc.TARGET_ROWS), 2), bool)
        c = {'verb': 'attack', 'subject': ['u0', 'fresh'], 'target': 'absent'}
        keep, diag = bc.filter_commands([c], x)
        self.assertEqual(keep, [])
        self.assertEqual(diag['rejections'], {'target_missing': 1})
        self.assertEqual(bc.carry_commands([c], x, bc.lookups(x)), [])
        self.assertFalse(bc.should_supervise_commands([c], keep))
        malformed = {'verb': 'stop'}
        self.assertEqual(bc.filter_commands([malformed], x)[1]['rejections'], {'subject_missing': 1})
        x['mask_subject'][bc.V_INDEX['stop'], :] = False
        self.assertEqual(bc.filter_commands([{'verb': 'stop', 'subject': ['u0']}], x)[1]['rejections'], {'subject_mask': 1})

if __name__ == '__main__':
    unittest.main()
