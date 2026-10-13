"""Factored PPO for the units decoder: per-unit log-probabilities, the KL anchor's estimator, the M28 ladder."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train import tools_model as tm  # noqa: E402
from fa_train.tools_trainer import LADDER, LADDER_M28, LADDERS, PERSONALITIES, stage_index  # noqa: E402
from fa_train.worker import opponent  # noqa: E402
from test_units_decoder import batch  # noqa: E402


class PerUnitLogProbTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.net = tm.ToolsNet("S", decoder="units")
        self.net.candidates = lambda b: (torch.zeros(2, 1, self.net.d), torch.zeros(2, 1, self.net.d))
        self.net.act_min_p = 0.0
        self.b = batch()
        self.inputs = (torch.randn(2, 3, self.net.d), torch.randn(2, self.net.map_ch, tm.GRID, tm.GRID),
                       torch.randn(2, self.net.core_dim), self.b)

    def test_unit_logp_sums_to_the_decision_and_matches_when_scored(self):
        with torch.no_grad():
            acts, lp, ent_act, _, _ = self.net.decode(*self.inputs, mode="sample")
            _, lp2, ent, _, _ = self.net.decode(*self.inputs, actions=acts)
        for e in (ent_act, ent):
            self.assertEqual(tuple(e["unit_logp"].shape), (2, 3))
            torch.testing.assert_close(e["unit_logp"].sum(1), lp)
            self.assertTrue(bool((e["unit_logp"][:, 2] == 0).all()))        # the third token can never act
            self.assertEqual(e["unit_known"][:, 2].tolist(), [0.0, 0.0])
        torch.testing.assert_close(ent_act["unit_logp"], ent["unit_logp"])
        torch.testing.assert_close(lp, lp2)

    def test_arguments_count_at_their_unit(self):
        """A unit given 'move' carries its cell's log-probability; a unit given nothing only its verb's."""
        move = tm.VERBS.index("move")
        acts = {k: torch.full((2, 3), -1, dtype=torch.long) for k in tm.UNIT_KEYS}
        acts["uverb"] = torch.tensor([[move, 0, 0], [0, move, 0]])
        acts["ucell"] = torch.tensor([[5, -1, -1], [-1, 7, -1]])
        with torch.no_grad():
            _, lp, ent, _, _ = self.net.decode(*self.inputs, actions=acts)
        ulp = ent["unit_logp"]
        self.assertLess(float(ulp[0, 0]), float(ulp[0, 1]))     # verb + a cell out of GRID^2 vs. 'nothing' alone
        self.assertLess(float(ulp[1, 1]), float(ulp[1, 0]))
        torch.testing.assert_close(ulp.sum(1), lp)

    def test_gradient_flows_through_unit_logp(self):
        acts, *_ = self.net.decode(*self.inputs, mode="sample")
        _, _, ent, _, _ = self.net.decode(*self.inputs, actions=acts)
        (ent["unit_logp"] * ent["unit_known"]).sum().backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in self.net.verb_head.parameters()))

    def test_the_slots_decoder_has_no_per_unit_log_probabilities(self):
        net = tm.ToolsNet("S", decoder="slots")
        net.candidates = lambda b: (torch.zeros(2, 1, net.d), torch.zeros(2, 1, net.d))
        with torch.no_grad():
            _, _, ent, _, _ = net.decode(*self.inputs, mode="sample")
        self.assertNotIn("unit_logp", ent)
        self.assertIsNone(getattr(net, "last_unit_logp", None))


class AnchorEstimatorTest(unittest.TestCase):
    def test_k3_is_zero_at_the_reference_and_positive_elsewhere(self):
        d = torch.zeros(4)
        self.assertEqual(float((d.exp() - d - 1).sum()), 0.0)
        d = torch.tensor([-0.5, 0.3, 1.0, -2.0])
        self.assertTrue(bool(((d.exp() - d - 1) > 0).all()))


class LadderTest(unittest.TestCase):
    def test_every_opponent_key_parses_and_the_m28_rungs_rise(self):
        known = {"easy", "medium", "m28ai", "m28aicheat", *PERSONALITIES, *(f"{p}cheat" for p in PERSONALITIES)}
        for ladder in LADDERS.values():
            for rung in ladder:
                self.assertAlmostEqual(sum(rung["opponents"].values()), 1.0, places=6)
                for key in list(rung["opponents"]) + list(rung["evals"]):
                    ai, cheat = opponent(key)
                    self.assertIn(ai, known)
                    self.assertTrue(cheat is None or cheat >= 1.0)
        mults = [opponent(r["promote"][0])[1] or 1.0 for r in LADDER_M28[3:]]
        self.assertEqual(mults, sorted(mults))
        self.assertEqual(LADDER_M28[:3], LADDER[:3])

    def test_stage_index_by_name_or_number(self):
        self.assertEqual(stage_index(LADDER_M28, "M28"), 3)
        self.assertEqual(stage_index(LADDER_M28, "1"), 1)
        with self.assertRaises(ValueError):
            stage_index(LADDER_M28, "L9")


if __name__ == "__main__":
    unittest.main()


class RecordedActionTest(unittest.TestCase):
    """The network's own choice, recorded with each step: compact_action <-> expand_action."""

    def test_units_round_trip_scores_the_same(self):
        from fa_train.tools_trainer import compact_action, expand_action
        torch.manual_seed(5)
        net = tm.ToolsNet("S", decoder="units")
        net.candidates = lambda b: (torch.zeros(2, 1, net.d), torch.zeros(2, 1, net.d))
        net.act_min_p = 0.0
        inputs = (torch.randn(2, 3, net.d), torch.randn(2, net.map_ch, tm.GRID, tm.GRID),
                  torch.randn(2, net.core_dim), batch())
        with torch.no_grad():
            acts, lp, _, _, _ = net.decode(*inputs, mode="sample")
        acts_np = {k: v.numpy() for k, v in acts.items()}
        rows = [expand_action(compact_action(acts_np, i, 3, "units"), 3, "units") for i in range(2)]
        for i, row in enumerate(rows):
            for k in tm.UNIT_KEYS:
                self.assertEqual(row[k].tolist(), acts_np[k][i][:3].tolist(), k)
        stacked = {k: torch.as_tensor(v) for k, v in tm.stack_actions(rows, 3, "units").items()}
        with torch.no_grad():
            _, lp2, _, _, _ = net.decode(*inputs, actions=stacked)
        torch.testing.assert_close(lp, lp2)

    def test_slots_round_trip(self):
        from fa_train.tools_trainer import compact_action, expand_action
        acts = {k: np.full((1, tm.K), -1, np.int64) for k in tm.ACTION_KEYS if k != "set"}
        acts["verb"][0, :2] = [tm.VERBS.index("move"), tm.V_END]
        acts["set"] = np.zeros((1, tm.K, 4), bool)
        acts["set"][0, 0, [1, 3]] = True
        row = expand_action(compact_action(acts, 0, 4, "slots"), 4, "slots")
        self.assertEqual(row["set"][0].tolist(), [False, True, False, True])
        self.assertEqual(row["verb"].tolist(), acts["verb"][0].tolist())


class VtraceUpgoTest(unittest.TestCase):
    def test_vtrace_with_unit_ratios_gives_the_gae_returns(self):
        from fa_train.learn_rl import gae, vtrace
        rng = np.random.default_rng(0)
        r, v = rng.normal(size=50).astype(np.float32), rng.normal(size=50).astype(np.float32)
        adv_g, ret_g = gae(r, v, 0.99, 0.95)
        _, vs = vtrace(r, v, np.zeros(50, np.float32), 0.99, 0.95)
        np.testing.assert_allclose(vs, ret_g, rtol=1e-4, atol=1e-4)

    def test_vtrace_truncates_a_policy_that_moved_away(self):
        from fa_train.learn_rl import vtrace
        r, v = np.ones(10, np.float32), np.zeros(10, np.float32)
        a_same, _ = vtrace(r, v, np.zeros(10, np.float32), 0.99, 0.95)
        a_less, _ = vtrace(r, v, np.full(10, -1.0, np.float32), 0.99, 0.95)    # actions now less likely
        self.assertTrue((a_less < a_same).all())

    def test_upgo_follows_the_game_while_it_beats_expectations(self):
        from fa_train.learn_rl import upgo
        r = np.array([1.0, 1.0, 1.0], np.float32)
        v = np.zeros(3, np.float32)                       # every step better than expected
        g = upgo(r, v, 0.5)
        np.testing.assert_allclose(g, [1 + 0.5 + 0.25, 1 + 0.5, 1.0])
        v_high = np.array([0.0, 5.0, 0.0], np.float32)     # step 2 is worse than its state's value: bootstrap there
        g2 = upgo(r, v_high, 0.5)
        self.assertAlmostEqual(float(g2[0]), 1 + 0.5 * 5.0)
