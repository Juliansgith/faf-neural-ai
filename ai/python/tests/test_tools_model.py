"""Exercise the real decoder when single-subject verbs exhaust their eligible units."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train import tools_model as tm
from fa_env.tools import merge_commands


class DecoderExhaustionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, eligible):
        batch, tokens = len(eligible), 2
        net = tm.ToolsNet("S")
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net.verb_head.bias[tm.V_BUILD] = 100
        net.candidates = lambda b: (torch.zeros(batch, 1, net.d), torch.zeros(batch, 1, net.d))
        b = {
            "token_mask": torch.ones(batch, tokens, dtype=torch.bool),
            "cand_mask": torch.ones(batch, 1, dtype=torch.bool),
            "enh_mask": torch.ones(batch, 1, dtype=torch.bool),
            "cand_kind": torch.full((batch, 1), 2),
            "mask_subject": torch.zeros(batch, tm.V, tokens, dtype=torch.bool),
            "can_build": torch.zeros(batch, tokens, 1, dtype=torch.bool),
            "can_enh": torch.zeros(batch, tokens, 1, dtype=torch.bool),
            "mask_silo": torch.zeros(batch, tokens, len(tm.SILO_KINDS), dtype=torch.bool),
            "mask_toggle": torch.zeros(batch, tokens, tm.N_OPTIONS, dtype=torch.bool),
            "mask_build_cell": torch.ones(batch, 4, tm.GRID, tm.GRID, dtype=torch.bool),
            "mask_move_cell": torch.ones(batch, tm.GRID, tm.GRID, dtype=torch.bool),
            "mask_target": torch.zeros(batch, tm.NR, tokens, dtype=torch.bool),
            "mask_anchor": torch.zeros(batch, tokens, dtype=torch.bool),
            "mask_verb": torch.zeros(batch, tm.V, dtype=torch.bool),
        }
        b["mask_verb"][:, [tm.V_END, tm.V_BUILD]] = True
        for row, subjects in enumerate(eligible):
            b["mask_subject"][row, tm.V_BUILD, subjects] = True
            b["can_build"][row, subjects, 0] = True
        h = torch.zeros(batch, tokens, net.d)
        m = torch.zeros(batch, net.map_ch, tm.GRID, tm.GRID)
        z = torch.zeros(batch, net.core_dim)
        return net, (h, m, z, b)

    def decode(self, net, inputs, mode):
        torch.manual_seed(17)
        with torch.no_grad():
            actions, acting_lp, _, _, _ = net.decode(*inputs, mode=mode)
            _, scoring_lp, entropy, loss, _ = net.decode(*inputs, actions=actions, smoothing=0.05)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in entropy.values()))
        torch.testing.assert_close(acting_lp, scoring_lp)
        return actions

    def commands(self, actions, row=0):
        return tm.indices_to_commands({key: value.numpy() for key, value in actions.items()}, row,
                                      {"ids": ["u0", "u1"], "cand_ids": ["bp"], "cand_kind": [2]})

    def test_sole_builder_at_either_token_ends_after_one_build(self):
        for builder in (0, 1):
            for mode in ("greedy", "sample"):
                with self.subTest(builder=builder, mode=mode):
                    net, inputs = self.fixture([[builder]])
                    actions = self.decode(net, inputs, mode)
                    self.assertEqual(actions["verb"][0].tolist(), [tm.V_BUILD] + [tm.V_END] * (tm.K - 1))
                    self.assertEqual(actions["single"][0, 0].item(), builder)
                    commands = self.commands(actions)
                    self.assertEqual(len(commands), 1)
                    self.assertEqual(commands[0]["subject"], [f"u{builder}"])
                    self.assertEqual(len(merge_commands(commands)), 1)

    def test_two_builders_are_distinct_before_end(self):
        for mode in ("greedy", "sample"):
            with self.subTest(mode=mode):
                net, inputs = self.fixture([[0, 1]])
                actions = self.decode(net, inputs, mode)
                self.assertEqual(actions["verb"][0].tolist(), [tm.V_BUILD] * 2 + [tm.V_END] * (tm.K - 2))
                self.assertEqual(set(actions["single"][0, :2].tolist()), {0, 1})
                self.assertEqual(len(self.commands(actions)), 2)

    def test_mixed_batch_exhaustion_is_per_row(self):
        for mode in ("greedy", "sample"):
            with self.subTest(mode=mode):
                net, inputs = self.fixture([[1], [0, 1]])
                actions = self.decode(net, inputs, mode)
                self.assertEqual(actions["verb"][0, :3].tolist(), [tm.V_BUILD, tm.V_END, tm.V_END])
                self.assertEqual(actions["verb"][1, :3].tolist(), [tm.V_BUILD, tm.V_BUILD, tm.V_END])
                self.assertEqual([len(self.commands(actions, row)) for row in range(2)], [1, 2])

    def test_distinct_single_subject_verbs_can_share_a_subject(self):
        pause = tm.VERBS.index("pause")
        for mode in ("greedy", "sample"):
            with self.subTest(mode=mode):
                net, inputs = self.fixture([[1]])
                inputs[3]["mask_verb"][0, pause] = True
                inputs[3]["mask_subject"][0, pause, 1] = True
                with torch.no_grad():
                    net.verb_head.bias[pause] = 80
                actions = self.decode(net, inputs, mode)
                self.assertEqual(actions["verb"][0, :3].tolist(), [tm.V_BUILD, pause, tm.V_END])
                self.assertEqual(actions["single"][0, :2].tolist(), [1, 1])
                self.assertEqual(len(self.commands(actions)), 2)

    def test_true_end_is_retained(self):
        for mode in ("greedy", "sample"):
            with self.subTest(mode=mode):
                net, inputs = self.fixture([[0]])
                with torch.no_grad():
                    net.verb_head.bias[tm.V_END] = 200
                actions = self.decode(net, inputs, mode)
                self.assertEqual(actions["verb"][0].tolist(), [tm.V_END] * tm.K)
                self.assertEqual(self.commands(actions), [])

    def test_duplicate_teacher_build_receives_masked_verb_penalty(self):
        net, inputs = self.fixture([[0]])
        actions = self.decode(net, inputs, "greedy")
        duplicate = {key: value.clone() for key, value in actions.items()}
        for key in duplicate:
            duplicate[key][:, 1] = duplicate[key][:, 0]
        with torch.no_grad():
            _, logp, _, loss, _ = net.decode(*inputs, actions=duplicate, smoothing=0.05)
        self.assertTrue(torch.isfinite(logp).all() and torch.isfinite(loss).all())
        self.assertLess(logp.item(), -1e8)
        self.assertGreater(loss.item(), 1e8)

    def test_set_always_has_its_leader_and_loss_mode_keeps_likelihood(self):
        net, inputs = self.fixture([[0, 1]])
        move = tm.VERBS.index("move")
        inputs[3]["mask_verb"][:] = False
        inputs[3]["mask_verb"][:, [tm.V_END, move]] = True
        inputs[3]["mask_subject"][:, move] = True
        with torch.no_grad():
            net.verb_head.bias[move] = 200
            net.set_bias.bias.fill_(-2)
            labels = net.decode(*inputs, mode="greedy")[0]
            labels["verb"][:, 1:] = tm.V_END
            labels["set"][0, 0, 0] = True
            _, lp1, _, _, metrics = net.decode(*inputs, actions=labels)
            # include/exclude alone would choose nobody here (bias -2); the leader pointer still names one
            self.assertEqual([x.item() for x in metrics["set_nonempty"]], [1., 1.])
            self.assertEqual([x.item() for x in metrics["set_leader"]], [1., 1.])
            self.assertEqual([x.item() for x in metrics["set_recall"]], [1., 1.])
            net.bc_set_loss_mode = "balanced"
            _, lp2, _, loss, _ = net.decode(*inputs, actions=labels)
        torch.testing.assert_close(lp1, lp2)
        self.assertTrue(torch.isfinite(loss).all())


if __name__ == "__main__":
    unittest.main()
