"""CPU execution changes must preserve recurrent gradients and PPO updates."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fa_train import tools_model as tm
from fa_train.cpu_runtime import configure_cpu
from fa_train.learn_rl import micro_view, ppo_step, value_warmup_on_features
from test_units_decoder import batch


def training_fixture():
    torch.manual_seed(42)
    net = tm.ToolsNet('S', decoder='units')
    net.verbs_after_attention = True
    net.smooth_verb_floor = True
    # Exercise the attention gradient, including projections initially zeroed in production.
    torch.nn.init.normal_(net.unit_attn.self_attn.out_proj.weight, std=.01)
    b = batch(n=4, tokens=5)
    b.update(tokens=torch.randn(4, 5, tm.N_FEATURES) * .1,
             token_bp=torch.zeros(4, 5, dtype=torch.long),
             token_order=torch.zeros(4, 5, dtype=torch.long),
             token_last=torch.zeros(4, 5, dtype=torch.long),
             token_last_feat=torch.zeros(4, 5, tm.N_LAST_FEATURES),
             token_type=torch.zeros(4, 5, dtype=torch.long),
             map=torch.randn(4, len(tm.MAP_CHANNELS), tm.GRID, tm.GRID) * .1,
             scalars=torch.randn(4, tm.N_SCALARS) * .1,
             privileged=torch.zeros(4, 2 * len(tm.PRIVILEGED)),
             cand_bp=torch.zeros(4, 1, dtype=torch.long),
             cand_feat=torch.zeros(4, 1, tm.N_CAND_FEATURES),
             enh_idx=torch.zeros(4, 1, dtype=torch.long),
             enh_feat=torch.zeros(4, 1, tm.N_ENH_FEATURES))
    b['token_mask'][:2, 2:] = False
    b['token_mask'][2:, 4:] = False
    starts = torch.tensor([True, False, True, False])
    state = net.initial_state(2, 'cpu')
    with torch.no_grad():
        h, m, z = net.core(b, state, 2, starts)
        act, _, ent, _, _ = net.decode(h, m, z, b)
        value = net.value_of(z, b)
    lab = {k: v.clone() for k, v in act.items()}
    lab['uverb'][:, 0] = tm.VERBS.index('move')
    lab['ucell'][:, 0] = 0
    c = dict(b=b, act=act, lab=lab, starts=starts, valid=torch.tensor([1., 1., 1., 0.]),
             old_ulp=ent['unit_logp'], prox_ulp=ent['unit_logp'], ref_ulp=ent['unit_logp'],
             v_old=value, ret=value + .2, adv=torch.tensor([.3, -.4, .2, 0.]),
             adv_up=torch.ones(4) * .1, sup=torch.tensor([1., 1., 1., 0.]), supw=torch.ones(4))
    a = SimpleNamespace(length=2, micro=1, temperature=.5, distill=.1, no_undo_labels=False,
                        local_credit=0., upgo=.2, clip=.2, value_clip=.2, vf_coef=.5,
                        ent_coef=.01, ref_coef=.1, max_grad=.5, lr=.0001, warmup=1,
                        separate_decoder=False, no_trim_batches=False)
    return net, c, state, a


class CPUTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sequence_lstm_matches_steps_with_independent_midsequence_resets(self):
        torch.manual_seed(2)
        net = tm.ToolsNet('S', decoder='units')
        starts = torch.tensor([[True, False, False, True, False],
                               [False, False, True, False, False]])
        inputs = torch.randn(2, 5, net.core_dim)
        initial = tuple(torch.randn(1, 2, net.core_dim) for _ in range(2))
        results = []
        for fast in (False, True):
            net.sequence_lstm = fast
            net.zero_grad(set_to_none=True)
            x = inputs.clone().requires_grad_()
            state = tuple(v.clone().requires_grad_() for v in initial)
            z, final = net.recurrent(x, state, starts)
            (z.square().mean() + sum(v.square().mean() for v in final)).backward()
            results.append((z.detach(), *[v.detach() for v in final], x.grad,
                            *[v.grad for v in state], *[p.grad.clone() for p in net.lstm.parameters()]))
        for left, right in zip(*results):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_shared_decoder_matches_separate_gradients_with_checkpointing(self):
        for checkpoint in (False, True):
            net, c, state, a = training_fixture()
            net.grad_checkpoint = checkpoint
            results = []
            for shared in (False, True):
                net.zero_grad(set_to_none=True)
                h, m, z = net.core(c['b'], state, 2, c['starts'])
                if shared:
                    lp, ent, bc = net.score_pair(h, m, z, c['b'], c['act'], c['lab'], temperature=.5)
                else:
                    lp, ent, _ = net.score(h, m, z, c['b'], c['act'], temperature=.5)
                    _, _, bc = net.score(h, m, z, c['b'], c['lab'], smoothing=.05)
                (-lp.mean() + .2 * bc.mean() - .01 * ent['verb'].mean()).backward()
                results.append((lp.detach(), bc.detach(),
                                {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}))
            torch.testing.assert_close(results[0][0], results[1][0])
            torch.testing.assert_close(results[0][1], results[1][1])
            self.assertEqual(results[0][2].keys(), results[1][2].keys())
            for key in results[0][2]:
                torch.testing.assert_close(results[0][2][key], results[1][2][key], atol=2e-6, rtol=1e-4)

    def test_trimmed_ppo_preserves_update_and_state(self):
        base, c, state, args = training_fixture()
        results = []
        for optimized in (False, True):
            net = copy.deepcopy(base)
            net.sequence_lstm = optimized
            a = copy.copy(args)
            a.no_trim_batches = not optimized
            a.separate_decoder = not optimized
            opt = torch.optim.SGD(net.parameters(), lr=a.lr)
            stats, memory, _ = ppo_step(net, net, opt, c, state, state, a)
            results.append((net.state_dict(), stats, memory))
        for key in results[0][0]:
            torch.testing.assert_close(results[0][0][key], results[1][0][key], atol=1e-6, rtol=1e-5)
        for key in results[0][1]:
            self.assertAlmostEqual(results[0][1][key], results[1][1][key], delta=2e-5, msg=key)
        for left, right in zip(results[0][2], results[1][2]):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_parallel_groups_preserve_update_and_state(self):
        base, c, state, args = training_fixture()
        results = []
        for workers in (1, 2):
            net = copy.deepcopy(base)
            a = copy.copy(args)
            a.parallel_groups = workers
            opt = torch.optim.SGD(net.parameters(), lr=a.lr)
            stats, memory, _ = ppo_step(net, net, opt, c, state, state, a)
            results.append((net.state_dict(), stats, memory))
        for key in results[0][0]:
            torch.testing.assert_close(results[0][0][key], results[1][0][key], atol=1e-6, rtol=1e-5)
        for key in results[0][1]:
            self.assertAlmostEqual(results[0][1][key], results[1][1][key], delta=2e-5, msg=key)
        for left, right in zip(results[0][2], results[1][2]):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_sparse_mask_keeps_last_present_token(self):
        _, c, _, _ = training_fixture()
        c['b']['token_mask'][0] = torch.tensor([True, False, False, False, True])
        c['scalar'] = torch.tensor(7.)
        sub, b, act, lab = micro_view(c, torch.tensor([0]), 2, 2)
        self.assertEqual(b['tokens'].shape[1], 5)
        self.assertEqual(sub['scalar'].item(), 7.)

    def test_bf16_encoder_preserves_float32_scores_and_finite_gradients(self):
        net, c, state, _ = training_fixture()
        net.amp = True
        h, m, z = net.core(c['b'], state, 2, c['starts'])
        self.assertTrue(all(value.dtype == torch.float32 for value in (h, m, z)))
        lp, _, bc = net.score_pair(h, m, z, c['b'], c['act'], c['lab'])
        (-lp.mean() + bc.mean()).backward()
        self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_cpu_flags_reject_invalid_configuration_before_changing_threads(self):
        args = SimpleNamespace(cpu_threads=0, cpu_interop_threads=None, cpu_affinity=None,
                               cpu_bf16=False, cpu_compile_map=False)
        with self.assertRaises(ValueError), patch('torch.set_num_threads') as setter:
            configure_cpu(args, 'cpu')
        setter.assert_not_called()
        args.cpu_threads = 4
        with self.assertRaises(ValueError):
            configure_cpu(args, 'cuda')

    def test_cached_critic_uses_one_dense_call_and_leaves_trunk_frozen(self):
        net, _, _, _ = training_fixture()
        games = SimpleNamespace(games=[{'pol_has': np.array([True, True])} for _ in range(2)],
                                opponent={0: np.zeros(tm.N_OPPONENT), 1: np.ones(tm.N_OPPONENT)})
        features = {i: (np.ones((2, net.core_dim), np.float32),
                        np.zeros((2, 2 * len(tm.PRIVILEGED)), np.float32)) for i in range(2)}
        ret = {i: np.ones(2, np.float32) for i in range(2)}
        a = SimpleNamespace(value_lr=.001, lr=.001, streams=2, length=2, stop_file=None)
        before = {k: p.clone() for k, p in net.named_parameters()}
        expected = copy.deepcopy(net)
        optimizer = torch.optim.Adam(list(expected.value_head.parameters()) + list(expected.opp_value.parameters()),
                                     lr=a.value_lr)
        loss = 0.
        for i in (0, 1):
            inputs = dict(privileged=torch.from_numpy(features[i][1]),
                          opponent=torch.from_numpy(np.tile(games.opponent[i], (2, 1))).float())
            prediction = expected.value_of(torch.from_numpy(features[i][0]), inputs)
            loss = loss + ((prediction - torch.from_numpy(ret[i])) ** 2).sum()
        (loss / 8).backward()
        optimizer.step()
        with patch.object(net, 'value_of', wraps=net.value_of) as head:
            value_warmup_on_features(net, games, [0, 1], features, ret, a, 'cpu', epochs=1)
            self.assertEqual(head.call_count, 1)
            self.assertEqual(head.call_args.args[0].shape, (4, net.core_dim))
        for key, parameter in net.named_parameters():
            if not key.startswith(('value_head.', 'opp_value.')):
                torch.testing.assert_close(parameter, before[key], atol=0, rtol=0)
            else:
                torch.testing.assert_close(parameter, dict(expected.named_parameters())[key], atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
