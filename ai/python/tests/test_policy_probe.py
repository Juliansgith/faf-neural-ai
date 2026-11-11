import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from fa_train import tools_model as tm
from fa_train.learn_rl import check_stop, self_imitation, policy_probe
try:
    from test_units_decoder import batch
except ImportError:
    from tests.test_units_decoder import batch


class PolicyProbeTests(unittest.TestCase):
    def test_commander_drift_is_not_diluted_by_the_army_or_unknown_actions(self):
        # One changed commander among 101 scored units looks quiet in the
        # aggregate. Two unknown actions have large changes but must not count.
        n = 103
        subject = torch.zeros(1, len(tm.VERBS), n, dtype=torch.bool)
        subject[0, tm.V_CMOVE, [0, 101]] = True
        known = torch.ones(1, n)
        known[0, 101:] = 0
        lp = torch.zeros(1, n)
        lp[0, 0] = 1
        lp[0, 101:] = 20
        mode = torch.zeros(1, n, dtype=torch.long)
        mode[0, [0, 101, 102]] = 1
        ent = dict(unit_known=known, unit_logp=lp, verb_mode=mode,
                   verb_removed_mass=torch.zeros(1, n), verb_below_floor=torch.zeros(1, n),
                   commander_count=torch.ones(1), commander_verb_probability=torch.zeros(1, len(tm.VERBS)),
                   commander_verb_removed_probability=torch.zeros(1, len(tm.VERBS)))
        c = dict(b={'mask_subject': subject}, valid=torch.ones(1), starts=torch.ones(1), act={},
                 prox_ulp=torch.zeros(1, n), ref_ulp=torch.zeros(1, n))
        games = SimpleNamespace(games=[dict(offsets_t=np.array([0, n]),
                                           _cache_manifest={'source_name': 'diagnostic'})])
        net = Mock(training=True, audit_probabilities=False)
        net.core.return_value = (None, None, None, None)
        net.score.return_value = (None, ent, None)
        before = {0: dict(logp=np.zeros(n, np.float32), mode=np.zeros(n, np.int16))}
        a = SimpleNamespace(micro=1, length=1, temperature=.5, stop_file=None)
        with patch('fa_train.learn_rl.chunk_batch', return_value=c):
            stats, _ = policy_probe(net, games, [0], {0: 1}, a, 'cpu', {0: before[0]['logp']}, before=before)
        self.assertEqual(stats['commander_scored_units'], 1)
        self.assertAlmostEqual(stats['commander_sample_k3_to_init'], np.e - 2, places=6)
        self.assertAlmostEqual(stats['sample_k3_to_init'], (np.e - 2) / 101, places=6)
        self.assertEqual(stats['commander_verb_mode_changed_by_sil'], 1.)
        self.assertAlmostEqual(stats['verb_mode_changed_by_sil'], 1 / 101, places=6)
        self.assertNotIn('commander_sample_k3_to_ref', stats)
        self.assertFalse(net.audit_probabilities)
        net.train.assert_called_with(True)

    def test_zero_sil_is_a_true_no_update_even_with_adam_momentum(self):
        net, opt = Mock(), Mock()
        a = SimpleNamespace(sil_coef=0.0, sil_value_coef=0.0, stop_file=None)
        self.assertIsNone(self_imitation(net, opt, a, 'cpu', 1, 1))
        opt.step.assert_not_called()
        net.train.assert_not_called()

    def test_stop_prevents_sil_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'STOP'
            path.write_text('operator')
            a = SimpleNamespace(stop_file=path)
            with self.assertRaises(SystemExit):
                check_stop(a)
            with self.assertRaises(SystemExit):
                self_imitation(Mock(), Mock(), a, 'cpu', 1, 1)

    def test_audit_measures_removed_mass_without_changing_actions_or_scores(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            net = tm.ToolsNet('S', decoder='units').eval()
            net.candidates = lambda b: (torch.zeros(2, 1, net.d), torch.zeros(2, 1, net.d))
            with torch.no_grad():
                for p in net.parameters():
                    p.zero_()
                net.verb_head.bias[tm.V_END] = 6
            b = batch()
            b['mask_subject'][:, tm.V_CMOVE, 0] = True
            b['mask_verb'][:, tm.V_CMOVE] = True
            args = (torch.zeros(2, 3, net.d), torch.zeros(2, net.map_ch, tm.GRID, tm.GRID),
                    torch.zeros(2, net.core_dim), b)
            with torch.no_grad():
                torch.manual_seed(91)
                acts, lp, ent, _, _ = net.decode(*args)
                net.audit_probabilities = True
                torch.manual_seed(91)
                acts2, lp2, audit, _, _ = net.decode(*args)
                _, scored, rescored, _, _ = net.decode(*args, actions=acts2)
            for k in acts:
                torch.testing.assert_close(acts[k], acts2[k], rtol=0, atol=0)
            torch.testing.assert_close(lp, lp2, rtol=0, atol=0)
            torch.testing.assert_close(scored, lp2, rtol=0, atol=0)
            self.assertNotIn('verb_removed_mass', ent)
            self.assertGreater(float(audit['verb_removed_mass'].sum()), 0)
            self.assertFalse(bool(audit['verb_below_floor'].any()))
            torch.testing.assert_close(audit['verb_removed_mass'], rescored['verb_removed_mass'])
            torch.testing.assert_close(audit['commander_count'], torch.ones(2))
            torch.testing.assert_close(audit['commander_verb_probability'].sum(-1), torch.ones(2))
            torch.testing.assert_close(audit['commander_verb_removed_probability'].sum(-1),
                                       audit['verb_removed_mass'][:, 0])
            # Every eligible non-END verb is below .02: actual END probability is
            # one, while the legacy recorded logp is negative. The audit exposes it.
            self.assertTrue(bool((acts['uverb'] == tm.V_END).all()))
            self.assertTrue(bool((lp < 0).all()))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
