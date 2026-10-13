"""Production-pipeline failures must preserve learning and not start an unintended policy."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fa_train import night_tools, tools_bc


class TinyNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def initial_state(self, *args):
        return None

    def evaluate(self, b, *args, **kwargs):
        n = len(b['belief'])
        value = self.weight.expand(n)
        return value, value, value, value[:, None], value.square()


class TinyGames:
    def __init__(self):
        self.games = [{'returns': np.zeros(2, np.float32)} for _ in range(2)]
        self.lengths = np.array([2, 2])

    def batch(self, picks, length, device):
        n = len(picks) * length
        return ({'belief': torch.zeros(n, 1)}, {}, torch.zeros(n, dtype=torch.bool),
                torch.zeros(n), torch.ones(n))


class PipelineSafetyTest(unittest.TestCase):
    def test_failed_clone_cannot_start_ppo(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(night_tools, 'Games', return_value=TinyGames()), \
             patch.object(night_tools, 'train_bc', side_effect=RuntimeError('clone failed')), \
             patch.object(night_tools, 'ToolsTrainer') as trainer, \
             patch.object(night_tools, 'keep_awake'), \
             patch.object(night_tools, 'raise_learner_priority'), \
             patch.object(night_tools, 'write_report'), \
             patch.object(night_tools, 'Tee', side_effect=lambda first, *_: first), \
             patch('sys.stdout', new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, 'clone failed'):
                night_tools.main(['--run', temp, '--stop-at', '23:59', '--collect-games', '0'])
            trainer.assert_not_called()

    def test_base_clone_saves_recovery_before_validation_failure(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(tools_bc, 'ToolsNet', TinyNet), \
             patch.object(tools_bc, 'chunks', return_value=[(0, 0)]), \
             patch.object(tools_bc, 'prefetch', side_effect=lambda fn, n: (fn() for _ in range(n))), \
             patch.object(tools_bc, 'evaluate_held_out', side_effect=RuntimeError('validation failed')):
            out = Path(temp) / 'model.pt'
            with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                tools_bc.train(Path(temp), out, steps=1, length=2, burn=0, batch_chunks=1,
                               eval_every=1, device='cpu', games=TinyGames(), log=lambda *_: None)
            state = torch.load(out.with_name('model_recovery_000001.pt'), weights_only=False)
            self.assertEqual(state['completed_steps'], 1)
            self.assertEqual(state['validation_status'], 'pending')
            self.assertIn('optimizer', state)
            self.assertFalse(out.exists())

    def test_fresh_collection_seed_is_explicit(self):
        args = night_tools.parse_args(['--run', 'unused', '--stop-at', '13:10', '--collect-seed', '84'])
        self.assertEqual(args.collect_seed, 84)


if __name__ == '__main__':
    unittest.main()
