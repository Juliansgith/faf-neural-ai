"""Sampling and scoring use one distribution; what a recorded log-probability means is versioned."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fa_train import probability_accounting as pa, tools_model as tm
from fa_train.learn_rl import load_net
from test_units_decoder import batch

torch.set_num_threads(1)


def policy(**changes):
    return dict(action={'units': []}, logp_semantics=pa.SMOOTH, mode='sample',
                temperature=.5, logp=-.2, unit_logp=[-.2], **changes)


class ProbabilityAccountingTest(unittest.TestCase):
    def test_legacy_default_and_exact_selection(self):
        self.assertEqual(pa.recording_semantics([{'action': {'units': []}}]), pa.LEGACY)
        corpus=SimpleNamespace(games=[{}, {'pol_logp_semantics': np.array(pa.SMOOTH),
                                          'pol_temperature': np.array(.5)}])
        self.assertEqual(pa.compatible_ppo_games(SimpleNamespace(smooth_verb_floor=False), corpus,[0,1]),([0],[1]))
        self.assertEqual(pa.compatible_ppo_games(SimpleNamespace(smooth_verb_floor=True), corpus,[0,1]),([1],[0]))

    def test_mixed_and_unknown_raw_tags_rejected(self):
        for tag in ('unsupported',None,pa.LEGACY):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                pa.recording_semantics([policy(),{**policy(),'logp_semantics':tag}])

    def test_unknown_cache_tag_rejected(self):
        with self.assertRaises(ValueError):
            pa.cached_semantics({'pol_logp_semantics':np.array('unknown')})

    def test_missing_unit_probabilities_rejected(self):
        row=policy();del row['unit_logp']
        with self.assertRaises(ValueError):pa.recording_semantics([row])

    def test_invalid_smooth_metadata_rejected(self):
        for key,value in [('mode','greedy'),('temperature',None),('temperature',True),
                          ('temperature',0),('temperature',float('nan')),
                          ('logp',float('inf')),('unit_logp',[float('nan')])]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                pa.recording_semantics([{**policy(),key:value}])

    def test_mixed_raw_temperatures_rejected(self):
        with self.assertRaises(ValueError):
            pa.recording_semantics([policy(),{**policy(),'temperature':1.}])

    def test_cached_temperature_mismatch_rejected(self):
        games=SimpleNamespace(games=[{'pol_logp_semantics':np.array(pa.SMOOTH),
                                     'pol_temperature':np.array(t)} for t in (.5,1.)])
        self.assertEqual(pa.compatible_ppo_games(SimpleNamespace(smooth_verb_floor=True),games,[0,1],temperature=.5),([0],[1]))
        games.games.append({'pol_logp_semantics':np.array(pa.SMOOTH)})
        with self.assertRaises(ValueError):
            pa.compatible_ppo_games(SimpleNamespace(smooth_verb_floor=True),games,[2],temperature=.5)

    def test_distribution_support_and_gradient(self):
        x=torch.tensor([-.1,-3.95,-3.8,-2.],dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([True,True,True,False])
        q=pa.smooth_log_probabilities(x,mask).exp()
        self.assertEqual(float(q[-1]),0.)
        self.assertTrue(bool((q[:3]>0).all()))
        self.assertAlmostEqual(float(q.sum()),1.,places=12)
        self.assertTrue(torch.autograd.gradcheck(lambda z:pa.smooth_log_probabilities(z,mask)[:3],(x,)))
        score=torch.autograd.grad((q.detach()*pa.smooth_log_probabilities(x,mask)).sum(),x)[0]
        torch.testing.assert_close(score,torch.zeros_like(score),atol=1e-12,rtol=0)

    def test_actor_configuration_rejected(self):
        net=tm.ToolsNet('S',decoder='units');net.smooth_verb_floor=True
        for key,value in [('act_min_p',0),('act_min_p_verbs',['build']),('act_top_p',.9),('commander_greedy',True)]:
            original=getattr(net,key);setattr(net,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):pa.validate_acting_settings(net,'sample',.5)
            setattr(net,key,original)
        with self.assertRaises(ValueError):pa.validate_acting_settings(net,'greedy',.5)
        with self.assertRaises(ValueError):
            tm.apply_model_options(tm.ToolsNet('S',decoder='slots'),{'model_options':{'smooth_verb_floor':True}})

    def test_actual_checkpoint_and_reference_family(self):
        net=tm.ToolsNet('S',decoder='units');net.smooth_verb_floor=True
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            torch.save({'size':'S','decoder':'units','model':net.state_dict(), 'model_options':tm.model_options(net)},path)
            restored,_=load_net(path,'cpu')
        self.assertTrue(restored.smooth_verb_floor)
        ref=tm.ToolsNet('S',decoder='units');before={k:v.clone() for k,v in ref.state_dict().items()}
        pa.align_reference_distribution(net,ref)
        self.assertTrue(ref.smooth_verb_floor)
        self.assertTrue(all(torch.equal(v,ref.state_dict()[k]) for k,v in before.items()))
        net.smooth_verb_floor=False;pa.align_reference_distribution(net,ref)
        self.assertFalse(ref.smooth_verb_floor)

    def _decoder_inputs(self, net):
        net.candidates = lambda obs: (torch.zeros(2, 1, net.d), torch.zeros(2, 1, net.d))
        return (torch.randn(2, 3, net.d), torch.randn(2, net.map_ch, tm.GRID, tm.GRID), torch.randn(2, net.core_dim), batch())

    def _biased(self, probabilities):
        net = tm.ToolsNet('S', decoder='units').eval()
        net.verb_head.weight.data.zero_()
        net.verb_head.bias.data.fill_(-50)
        for verb, p in probabilities.items():
            net.verb_head.bias.data[tm.VERBS.index(verb)] = math.log(p)
        return net

    def test_retention_formula_by_hand(self):
        """q ~ p * [.01 + .99 sigmoid(8 ln(p/.02))], END kept, then renormalised."""
        p = torch.tensor([.98, .01, .01], dtype=torch.float64)
        mask = torch.ones(3, dtype=torch.bool)
        keep = .01 + .99 / (1 + (.01 / .02) ** -8)
        expected = torch.tensor([.98, .01 * keep, .01 * keep], dtype=torch.float64)
        expected = expected / expected.sum()
        got = pa.smooth_log_probabilities(p.log(), mask).exp()
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=0)

    def test_free_verb_keeps_its_probability_ratio(self):
        p = torch.tensor([.97, .01, .01, .01], dtype=torch.float64)
        mask = torch.ones(4, dtype=torch.bool)
        q = pa.smooth_log_probabilities(p.log(), mask, (2,)).exp()
        self.assertAlmostEqual(float(q[2] / q[0]), .01 / .97, places=10)
        self.assertLess(float(q[1]), .001)

    def test_hard_floor_records_another_distribution(self):
        """The defect this module exists for: the floored decoder can only choose END, and records 0.98 for it."""
        net = self._biased({'end': .98, 'move': .01, 'fire_state': .01})
        with torch.no_grad():
            torch.manual_seed(5)
            acts, _, ent, _, _ = net.decode(*self._decoder_inputs(net), temperature=.5)
        self.assertTrue(bool((acts['uverb'] == 0).all()))
        known = ent['unit_known'] > 0
        self.assertGreater(float(ent['unit_logp'][known].exp().max()), .9)
        self.assertLess(float(ent['unit_logp'][known].exp().max()), .99)

    def test_sampling_frequency_matches_recorded_probability(self):
        for free in (False, True):
            with self.subTest(free=free):
                net = self._biased({'end': .90, 'move': .05, 'fire_state': .05})
                net.smooth_verb_floor = True
                net.upgrade_floor_free = free
                torch.manual_seed(11)
                inputs = self._decoder_inputs(net)
                counts, recorded, n = {}, {}, 0
                with torch.no_grad():
                    for _ in range(1500):
                        acts, _, ent, _, _ = net.decode(*inputs, temperature=.5)
                        _, _, scored, _, _ = net.decode(*inputs, actions=acts, temperature=.5)
                        torch.testing.assert_close(ent['unit_logp'], scored['unit_logp'], atol=1e-5, rtol=1e-5)
                        known = ent['unit_known'] > 0
                        for v, lp in zip(acts['uverb'][known].tolist(), ent['unit_logp'][known].tolist()):
                            counts[v] = counts.get(v, 0) + 1
                            recorded.setdefault(v, []).append(lp)
                            n += 1
                # a verb's recorded probability (no arguments for END) against how often it was drawn
                self.assertAlmostEqual(counts[0] / n, math.exp(recorded[0][0]), delta=.02)

    def test_upgrade_free_option_saved_and_tagged(self):
        net = tm.ToolsNet('S', decoder='units')
        net.smooth_verb_floor = net.upgrade_floor_free = True
        self.assertEqual(pa.semantics(net), pa.SMOOTH_UPGRADE_FREE)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            torch.save({'size': 'S', 'decoder': 'units', 'model': net.state_dict(), 'model_options': tm.model_options(net)}, path)
            restored, _ = load_net(path, 'cpu')
        self.assertTrue(restored.upgrade_floor_free and restored.smooth_verb_floor)
        with self.assertRaises(ValueError):
            tm.apply_model_options(tm.ToolsNet('S', decoder='units'), {'model_options': {'upgrade_floor_free': True}})
        games = SimpleNamespace(games=[{'pol_logp_semantics': np.array(pa.SMOOTH), 'pol_temperature': np.array(.5)},
                                       {'pol_logp_semantics': np.array(pa.SMOOTH_UPGRADE_FREE), 'pol_temperature': np.array(.5)}])
        self.assertEqual(pa.compatible_ppo_games(net, games, [0, 1]), ([1], [0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
