"""Own-information boundaries, checkpoint parity, and the actual learning input."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fa_env.tools import ToolKit
from fa_env.tools_obs import ToolsObserver
from fa_train import tools_bc as bc, tools_model as tm
from observer_fixture import BPS, info, obs, unit


def observation(enhancement='Gun', enemy=False):
    blueprints = copy.deepcopy(BPS)
    blueprints['acu'] = dict(id='acu', categories=['COMMAND', 'ENGINEER', 'MOBILE', 'LAND'],
                            enhancements=[dict(name='Gun', slot='RCH'), dict(name='Shield', slot='Back')])
    data = info()
    observer = ToolsObserver(data, blueprints, 1, ToolKit(data, blueprints))
    units = [unit('0', 'acu', 20, 20, enh=[enhancement])]
    if enemy:
        units.append(unit('enemy', 'acu', 80, 80, army=2, enh=['Gun', 'Shield']))
    return observer.observe(obs(units))


class InstalledEnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_identity_reaches_cache_and_batch_without_enemy_information(self):
        x = observation(enemy=True)
        own, enemy = x['ids'].index('0'), x['ids'].index('enemy')
        self.assertEqual(int(np.count_nonzero(x['token_enhancements'][own])), 1)
        self.assertFalse(x['token_enhancements'][enemy].any())
        packed = bc.pack([x], [tm.empty_labels(len(x['tokens']))])
        self.assertEqual(packed['token_enhancements'].dtype, np.int16)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'game.npz'
            np.savez(p, **packed)
            with np.load(p) as saved:
                loaded = object.__new__(bc.Games).step_obs(saved, 0)
            np.testing.assert_array_equal(loaded['token_enhancements'], x['token_enhancements'])
            batch = tm.collate([loaded], 'cpu')
            self.assertEqual(batch['token_enhancements'].dtype, torch.int64)
            self.assertFalse(bool(batch['token_enhancements'][0, enemy].any()))

    def test_legacy_checkpoint_and_enabled_zero_residual_have_identical_outputs(self):
        torch.manual_seed(7)
        net = tm.ToolsNet('S', decoder='units').eval()
        legacy = {k: v for k, v in net.state_dict().items() if not k.startswith('installed_enh_')}
        migrated = tm.ToolsNet('S', decoder='units').eval()
        migrated.load_state_dict(legacy)
        migrated.installed_enhancements = True
        batch = tm.collate([observation()], 'cpu')
        with torch.no_grad():
            old = net.encode(batch)
            new = migrated.encode(batch)
        for a, b in zip(old, new):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        options = tm.model_options(migrated)
        tm.apply_model_options(net, {'model_options': options})
        self.assertTrue(net.installed_enhancements)
        tm.apply_model_options(net, {})
        self.assertFalse(net.installed_enhancements)

    def test_enabled_policy_rejects_missing_input_and_zero_adapter_can_learn(self):
        torch.manual_seed(8)
        net = tm.ToolsNet('S', decoder='units').eval()
        net.installed_enhancements = True
        batch = tm.collate([observation('Gun'), observation('Shield')], 'cpu')
        with self.assertRaisesRegex(ValueError, 'current observation'):
            net.encode({k: v for k, v in batch.items() if k != 'token_enhancements'})
        for p in net.parameters():
            p.requires_grad_(False)
        net.installed_enh_weight.requires_grad_(True)
        # Check the same batch on both paths. Different rows can differ by a few
        # float32 ULPs even with identical inputs in batched attention kernels.
        torch.testing.assert_close(batch['tokens'][0], batch['tokens'][1], rtol=0, atol=0)
        net.installed_enhancements = False
        baseline = net.encode(batch)[0]
        net.installed_enhancements = True
        before = net.encode(batch)[0]
        torch.testing.assert_close(before, baseline, rtol=0, atol=0)
        loss = ((before[:, 0, 0] - torch.tensor([1., -1.])) ** 2).mean()
        loss.backward()
        self.assertGreater(float(net.installed_enh_weight.grad.abs().sum()), 0.)
        with torch.no_grad():
            net.installed_enh_weight -= .001 * net.installed_enh_weight.grad
            after = net.encode(batch)[0]
            self.assertGreater(float((after[0] - after[1]).abs().max()), 1e-6)
            net.installed_enhancements = False
            disabled = net.encode(batch)[0]
            torch.testing.assert_close(disabled, before, rtol=0, atol=0)

    def test_legacy_adam_moments_stay_on_their_original_parameters(self):
        net = tm.ToolsNet('S', decoder='units')
        old_params = [p for n, p in net.named_parameters() if n != 'installed_enh_weight']
        old = torch.optim.Adam(old_params, lr=.001)
        # Populate distinct moments on two differently shaped old parameters.
        for i, p in enumerate(old_params[:2]):
            p.grad = torch.full_like(p, (i + 1) * .01)
        old.step()
        state = old.state_dict()
        legacy_model = {k: v for k, v in net.state_dict().items() if k != 'installed_enh_weight'}
        resumed = torch.optim.Adam(net.parameters())
        migrated = tm.compatible_optimizer_state(net, state, legacy_model)
        self.assertEqual(len(state['param_groups'][0]['params']), len(old_params))
        resumed.load_state_dict(migrated)
        self.assertNotIn(net.installed_enh_weight, resumed.state)
        for p in old_params[:2]:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(resumed.state[p][key], old.state[p][key], rtol=0, atol=0)
        # The next real optimizer step succeeds, including the newly added input.
        resumed.zero_grad()
        net.installed_enh_weight.grad = torch.ones_like(net.installed_enh_weight)
        resumed.step()
        self.assertEqual(float(resumed.state[net.installed_enh_weight]['step']), 1.)
        self.assertIs(tm.compatible_optimizer_state(net, migrated, net.state_dict()), migrated)


if __name__ == '__main__':
    unittest.main()
