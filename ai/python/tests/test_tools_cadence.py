"""CPU/mock integration checks for cadence; no models, optimizer steps or games."""
import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from fa_train import night_tools
from fa_train.tools_cadence import checkpoint_cadence
from fa_train.tools_trainer import ToolsConfig, ToolsTrainer, Trainer
from fa_train.worker import worker_main
from test_tools_distillation import Pipe, trainer


class ToolsCadenceTest(unittest.TestCase):
    def bare(self, ticks=2):
        t = trainer(ToolsConfig(run_dir='unused', decision_ticks=ticks,
                                distill_teacher='none', maps=False))
        t.mode, t.eval_queue, t.rng = 'ppo', [], random.Random(1)
        t.opponents = lambda: {'easy': 1.0}
        return t

    def test_config_and_mutation_reject_before_allocation(self):
        for value in (0, 1, -1, True, 2.0, '2', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'decision_ticks'):
                ToolsConfig(run_dir='unused', decision_ticks=value)
        t = self.bare()
        t.cfg.decision_ticks = 0
        with patch.object(Trainer, '__init__') as base, self.assertRaises(ValueError):
            ToolsTrainer(t.cfg, device='cpu')
        base.assert_not_called()
        with self.assertRaises(ValueError):
            t.schedule_eval(1, 'easy')
        self.assertEqual(t.eval_queue, [])

    def test_cli_config_preserves_default_and_explicit_ticks(self):
        for args, ticks in (([], 20), (['--decision-ticks', '2'], 2)):
            parsed = night_tools.parse_args(['--run', 'unused', '--stop-at', '07:00'] + args)
            self.assertEqual(night_tools.make_config(parsed, 'unused').decision_ticks, ticks)
        for value in ('0', '1', '-2', '1.5'):
            with patch.object(Path, 'mkdir') as mkdir, patch.object(night_tools, 'ToolsTrainer') as constructor, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises((ValueError, SystemExit)):
                night_tools.main(['--run', 'unused', '--stop-at', '07:00', '--decision-ticks', value])
            mkdir.assert_not_called()
            constructor.assert_not_called()

    def test_training_and_scheduled_evaluation_reach_real_worker_path(self):
        for ticks in (2, 20):
            t = self.bare(ticks)
            episodes = [t._episode('ppo')]
            t.schedule_eval(2, 'medium')
            self.assertEqual(len(t.eval_queue), 2)
            episodes += [t._episode('ppo'), t._episode('ppo')]
            for ep in episodes:
                with self.subTest(ticks=ticks, evaluation=ep['eval']), tempfile.TemporaryDirectory() as root:
                    self.assertEqual(ep['decision_ticks'], ticks)
                    env = Mock()
                    env.fa = None
                    env.reset.return_value = ({'tokens': np.zeros((1, 1)), 'map': np.zeros((1, 1))}, {})
                    pipe = Pipe([('reset', ep), ('stop',)])
                    with patch('fa_env.tools_env.ToolsEnv', return_value=env) as constructor:
                        worker_main(0, pipe, root)
                    self.assertEqual(constructor.call_args.kwargs['decision_ticks'], ticks)
                    self.assertEqual(pipe.sent[0][0], 'obs')
                    self.assertEqual(pipe.sent[0][4]['decision_ticks'], ticks)

    def test_explicit_queued_override_is_retained(self):
        t = self.bare(20)
        t.eval_queue.append({'eval': True, 'decision_ticks': 2})
        self.assertEqual(t._episode('ppo')['decision_ticks'], 2)

    def test_worker_invalid_ticks_never_construct_environment(self):
        for value in (0, 1, -1, True, 1.5, '2'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as root:
                pipe = Pipe([('reset', {'kind': 'tools', 'distill_teacher': 'none',
                                        'decision_ticks': value}), ('stop',)])
                with patch('fa_env.tools_env.ToolsEnv') as constructor:
                    worker_main(0, pipe, root)
                constructor.assert_not_called()
                self.assertEqual(pipe.sent[0][0], 'error')
                self.assertIn('decision_ticks', pipe.sent[0][1])

    def test_save_and_load_keep_explicit_runtime_authority_and_report_mismatch(self):
        t = self.bare(2)
        with tempfile.TemporaryDirectory() as root:
            t.run_dir = Path(root)
            with patch('fa_train.tools_trainer.torch.save') as save, patch.object(Path, 'replace'):
                t.save('mock.pt')
            self.assertEqual(save.call_args.args[0]['config']['decision_ticks'], 2)
        state = {'model': {'mock_weights': 2}, 'config': {'decision_ticks': 20}}
        output = io.StringIO()
        with patch('fa_train.tools_trainer.torch.load', return_value=state), contextlib.redirect_stdout(output):
            t.load('unread.pt', optimizer=False)
        self.assertEqual(t.cfg.decision_ticks, 2)
        event, = [call.args[0] for call in t._log_event.call_args_list
                  if call.args[0]['event'] == 'checkpoint_cadence']
        self.assertTrue(event['mismatch'])
        self.assertEqual(event['saved_decision_ticks'], 20)
        self.assertEqual(event['active_decision_ticks'], 2)
        self.assertIn('WARNING: checkpoint decision cadence differs', output.getvalue())
        self.assertEqual(checkpoint_cadence(None, 2)['saved_provenance'], 'unavailable')
        self.assertIsNone(checkpoint_cadence({}, 2)['saved_decision_ticks'])

    def test_invalid_checkpoint_cadence_fails_before_state_mutation(self):
        t = self.bare(2)
        state = {'model': {'mock_weights': 2}, 'config': {'decision_ticks': 0}}
        with patch('fa_train.tools_trainer.torch.load', return_value=state), self.assertRaises(ValueError):
            t.load('unread.pt')
        t.net.load_state_dict.assert_not_called()
        t.acting.load_state_dict.assert_not_called()
        t.optimizer.load_state_dict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
