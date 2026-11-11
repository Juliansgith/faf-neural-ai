"""Loading pairs, information boundary, label compatibility and actual decoder scores."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch

from observer_fixture import BPS, info, obs, unit
from fa_env.tools import ToolKit
from fa_env.tools_obs import ToolsObserver
from fa_env.refueling import enabled_observation
from fa_train import tools_bc as bc, tools_model as tm
from fa_train.learn_rl import chunk_batch


def world():
    bps = copy.deepcopy(BPS)
    for name, categories in {
        'plane': ['AIR', 'MOBILE'], 'transport': ['AIR', 'MOBILE', 'TRANSPORTATION'],
        'stage': ['STRUCTURE', 'AIRSTAGINGPLATFORM'],
        'nostage': ['AIR', 'MOBILE', 'CANNOTUSEAIRSTAGING']}.items():
        bps[name] = dict(id=name, categories=categories, mass=100, health=100)
    units = [unit('air', 'plane', 30, 30, fuel=.2, fuel_use_time=300),
             unit('unknown', 'plane', 31, 30), unit('land', 'tank', 30, 31),
             unit('transport', 'transport', 32, 30), unit('stage', 'stage', 33, 30),
             unit('enemy', 'stage', 100, 100, army=2, fuel=.2, fuel_use_time=300),
             unit('unfinished', 'stage', 34, 30, frac=.9),
             unit('nostage', 'nostage', 35, 30, fuel=.2, fuel_use_time=300)]
    hello = info()
    observer = ToolsObserver(hello, bps, 1, ToolKit(hello, bps))
    return observer.observe(obs(units)), hello, bps, units


class RefuelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_known_unknown_enemy_and_unfinished_facility_boundaries(self):
        x, *_ = world()
        ix = {name: x['ids'].index(name) for name in ('air', 'unknown', 'enemy', 'unfinished')}
        self.assertEqual(x['token_refueling'][ix['air'], 0], 1)
        self.assertAlmostEqual(float(x['token_refueling'][ix['air'], 1]), .2)
        self.assertFalse(x['token_refueling'][ix['unknown']].any())
        self.assertFalse(x['token_refueling'][ix['enemy']].any())
        self.assertEqual(x['load_target_class'][ix['enemy']], 0)
        self.assertEqual(x['load_target_class'][ix['unfinished']], 0)
        self.assertEqual(x['load_subject_class'][x['ids'].index('nostage')], 0)
        self.assertFalse(x['mask_subject'][tm.V_LOAD, ix['air']])
        rx = enabled_observation(x)
        self.assertTrue(rx['mask_subject'][tm.V_LOAD, ix['air']])
        self.assertFalse(x['mask_subject'][tm.V_LOAD, ix['air']])  # no mutation of legacy view

    def test_labels_allow_only_matching_subject_target_pairs(self):
        x = enabled_observation(world()[0])
        for subject, target, allowed in [('air', 'stage', True), ('land', 'transport', True),
                                         ('air', 'transport', False), ('land', 'stage', False),
                                         ('air', 'enemy', False), ('air', 'unfinished', False)]:
            cmd = dict(verb='load', subject=[subject], target=target)
            with self.subTest(subject=subject, target=target):
                self.assertEqual(bc.command_allowed(cmd, x, bc.lookups(x), set()), allowed)
        kept, _ = bc.filter_commands([dict(verb='load', subject=['land', 'air'], target='stage')], x)
        self.assertEqual(kept, [dict(verb='load', subject=['air'], target='stage')])

    def test_both_decoders_sample_legal_pairs_and_score_the_same_distribution(self):
        x, *_ = world()
        b = tm.collate([x], 'cpu')
        for decoder in ('units', 'slots'):
            with self.subTest(decoder=decoder):
                torch.manual_seed(31)
                net = tm.ToolsNet('S', decoder=decoder).eval()
                net.air_refueling = True
                net.act_min_p = 0  # isolate typed masking from the separately tracked legacy verb-floor bug
                with torch.no_grad():
                    net.verb_head.weight.zero_()
                    net.verb_head.bias.fill_(-50)
                    net.verb_head.bias[tm.V_LOAD] = 50
                    # All eligible members try joining; mixed land/air sets must still be impossible.
                    net.member_q.weight.zero_()
                    net.member_q.bias.zero_()
                    net.set_bias.weight.zero_()
                    net.set_bias.bias.fill_(50)
                    state = net.initial_state(1, 'cpu')
                    actions, lp, _, _ = net.forward_step(b, state, mode='sample')
                    lp_again, *_ = net.evaluate(b, actions, state, seq_len=1)
                torch.testing.assert_close(lp, lp_again, rtol=1e-5, atol=1e-5)
                load_count = 0
                if decoder == 'units':
                    for i in range(len(x['ids'])):
                        if actions['uverb'][0, i] == tm.V_LOAD:
                            j = int(actions['utarget'][0, i])
                            self.assertTrue(x['load_subject_class'][i] & x['load_target_class'][j])
                            load_count += 1
                else:
                    for k in range(tm.K):
                        if actions['verb'][0, k] == tm.V_LOAD:
                            j = int(actions['target'][0, k])
                            selected = actions['set'][0, k].numpy()
                            self.assertTrue(selected.any())
                            self.assertTrue(np.all(x['load_subject_class'][selected] & x['load_target_class'][j]))
                            load_count += 1
                self.assertGreater(load_count, 0)

    def test_real_encoder_keeps_legacy_and_refueling_teacher_labels_separate(self):
        x, hello, bps, units = world()
        cmd = dict(verb='load', subject=['air'], target='stage')
        header = dict(hello=hello, meta=dict(me=1, faction=1, label_source='m28'))
        records = [dict(step=dict(tick=0, labels=[cmd], commands=[], sent=[]), msg=dict(obs=obs(units)))]
        with patch.object(bc, 'read_game', return_value=(header, records, dict(outcome='victory', minutes=1))):
            arrays, _ = bc.encode_game(None, bps)
        air = x['ids'].index('air')
        self.assertNotEqual(arrays['uverb'][air], tm.V_LOAD)
        self.assertEqual(arrays['rf_uverb'][air], tm.V_LOAD)
        games = object.__new__(bc.Games)
        games.games, games.units, games.air_refueling = [arrays], True, False
        before = games.batch([(0, 0)], 1, 'cpu')[1]
        games.air_refueling = True
        after = games.batch([(0, 0)], 1, 'cpu')[1]
        self.assertNotEqual(int(before['uverb'][0, air]), tm.V_LOAD)
        self.assertEqual(int(after['uverb'][0, air]), tm.V_LOAD)

        # PPO uses a different batching path: change its teacher labels only,
        # never the actual behavior actions or recorded behavior probabilities.
        arrays.update({'pol_' + k: arrays[k].copy() for k in tm.UNIT_KEYS})
        arrays.update(pol_has=np.ones(1, bool), pol_unit_logp=np.zeros(len(x['ids']), np.float32),
                      pol_logp=np.zeros(1, np.float32))
        games.lengths = np.array([1])
        games.air_refueling = False
        legacy = chunk_batch(games, [(0, 0)], 1, 'cpu')
        games.air_refueling = True
        enabled = chunk_batch(games, [(0, 0)], 1, 'cpu')
        self.assertNotEqual(int(legacy['lab']['uverb'][0, air]), tm.V_LOAD)
        self.assertEqual(int(enabled['lab']['uverb'][0, air]), tm.V_LOAD)
        for key in tm.UNIT_KEYS:
            torch.testing.assert_close(legacy['act'][key], enabled['act'][key], rtol=0, atol=0)
        torch.testing.assert_close(legacy['old_ulp'], enabled['old_ulp'], rtol=0, atol=0)

    def test_zero_fuel_adapter_preserves_encoder_and_has_real_gradient(self):
        x, *_ = world()
        b = tm.collate([x], 'cpu')
        torch.manual_seed(51)
        net = tm.ToolsNet('S', decoder='units').eval()
        old = net.encode(b)[0]
        net.air_refueling = True
        new = net.encode(b)[0]
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        new[0, x['ids'].index('air'), 0].backward()
        self.assertGreater(float(net.refuel_weight.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, 'current observation'):
            net.encode({k: v for k, v in b.items() if k != 'token_refueling'})

    def test_old_adam_moments_survive_both_new_root_parameters(self):
        net = tm.ToolsNet('S')
        added = ('installed_enh_weight', 'refuel_weight')
        old_params = [p for n, p in net.named_parameters() if n not in added]
        optimizer = torch.optim.Adam(old_params, lr=.01)
        old_params[0].grad = torch.ones_like(old_params[0])
        optimizer.step()
        model = {n: p for n, p in net.state_dict().items() if n not in added}
        migrated = torch.optim.Adam(net.parameters())
        migrated.load_state_dict(tm.compatible_optimizer_state(net, optimizer.state_dict(), model))
        torch.testing.assert_close(migrated.state[old_params[0]]['exp_avg'], optimizer.state[old_params[0]]['exp_avg'])
        self.assertNotIn(net.refuel_weight, migrated.state)
        self.assertNotIn(net.installed_enh_weight, migrated.state)


if __name__ == '__main__':
    unittest.main()
