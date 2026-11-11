from types import SimpleNamespace
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fa_train.learn_rl import critic_bias, discounted, reward_spec, value_warmup


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(1, 1, bias=False)
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        self.opp_value = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(1)

    def initial_state(self, streams, device):
        return None

    def core(self, b, state, length, starts, return_state):
        return None, None, self.trunk(b['x']), state

    def value_of(self, z, b):
        return (self.value_head(z) + self.opp_value(z)).flatten()


class ValueWarmupTests(unittest.TestCase):
    def test_critic_bias_does_not_cancel_opposite_errors_on_different_m28_levels(self):
        games = SimpleNamespace(games=[
            {'_cache_manifest': {'source_name': '0000001_m28aicheat@0.75_100.jsonl.gz'}},
            {'_cache_manifest': {'source_name': '0000002_m28aicheat@0.9_101.jsonl.gz'}}])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            critic_bias({0: np.array([-.5]), 1: np.array([.5])},
                        {0: np.zeros(1), 1: np.zeros(1)}, [0, 1], games, 'test')
        self.assertIn('m28aicheat@0.75 -0.50', output.getvalue())
        self.assertIn('m28aicheat@0.9 +0.50', output.getvalue())

    def test_changed_discount_invalidates_legacy_critic_signature(self):
        a = SimpleNamespace(econ_source='recorded', econ_coef=0., reclaim_coef=.5, intel_coef=0.,
                            map_coef=0., waste_coef=0., shaping=1., stall_coef=0., acu_coef=0., gamma=.997)
        legacy = reward_spec(a)
        self.assertNotIn('gamma=', legacy)
        returns = np.zeros(601); returns[-1] = 10
        old_target = discounted(returns, a.gamma)[0]
        a.gamma = .999
        self.assertNotEqual(reward_spec(a), legacy)
        self.assertGreater(discounted(returns, a.gamma)[0], 3 * old_target)
        a.gamma = .997
        self.assertEqual(reward_spec(a), legacy)

    def test_requested_rate_changes_only_critic_heads(self):
        for requested, policy_rate, expected in ((.01, .1, .01), (None, .02, .02)):
            with self.subTest(requested=requested):
                net = TinyNet()
                args = SimpleNamespace(value_lr=requested, lr=policy_rate, value_warmup=1,
                                       streams=1, length=1, stop_file=None)
                batch = dict(b={'x': torch.ones(1, 1)}, starts=torch.ones(1),
                             valid=torch.ones(1), ret=torch.zeros(1))
                with patch('fa_train.learn_rl.chunk_batch', return_value=batch):
                    value_warmup(net, object(), [0], {0: 1}, {0: np.zeros(1)}, args, 'cpu')
                self.assertEqual(float(net.trunk.weight.detach()), 1.)
                self.assertIsNone(net.trunk.weight.grad)
                self.assertAlmostEqual(float(net.value_head.weight.detach()), 1. - expected, places=6)
                self.assertAlmostEqual(float(net.opp_value.weight.detach()), 1. - expected, places=6)


    def test_epochs_override_for_the_every_iteration_warmup(self):
        net = TinyNet()
        args = SimpleNamespace(value_lr=.01, lr=.1, value_warmup=3, streams=1, length=1, stop_file=None)
        batch = dict(b={'x': torch.ones(1, 1)}, starts=torch.ones(1), valid=torch.ones(1), ret=torch.zeros(1))
        output = io.StringIO()
        with patch('fa_train.learn_rl.chunk_batch', return_value=batch), contextlib.redirect_stdout(output):
            value_warmup(net, object(), [0], {0: 1}, {0: np.zeros(1)}, args, 'cpu', epochs=1)
        self.assertEqual(output.getvalue().count('value warm-up epoch'), 1)

    def test_every_iteration_flag_is_declared_off_by_default(self):
        import inspect
        from fa_train import learn_rl
        src = inspect.getsource(learn_rl.main)
        self.assertIn('ap.add_argument("--value-warmup-every", type=int, default=0', src)
        self.assertIn('if changed or a.value_warmup_every > 0:', src)


if __name__ == '__main__':
    unittest.main()
