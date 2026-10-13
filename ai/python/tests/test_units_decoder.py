"""The 'units' decoder: per-unit labels, their commands, and acting/scoring agreement."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train import tools_bc  # noqa: E402
from fa_train import tools_model as tm  # noqa: E402
from test_tools_bc import tiny_observation  # noqa: E402


def batch(n: int = 2, tokens: int = 3) -> dict:
    move, fire = tm.VERBS.index("move"), tm.VERBS.index("fire_state")
    b = {
        "token_mask": torch.ones(n, tokens, dtype=torch.bool),
        "cand_mask": torch.ones(n, 1, dtype=torch.bool), "enh_mask": torch.ones(n, 1, dtype=torch.bool),
        "cand_kind": torch.full((n, 1), 2),
        "mask_subject": torch.zeros(n, tm.V, tokens, dtype=torch.bool),
        "can_build": torch.zeros(n, tokens, 1, dtype=torch.bool), "can_enh": torch.zeros(n, tokens, 1, dtype=torch.bool),
        "mask_silo": torch.zeros(n, tokens, len(tm.SILO_KINDS), dtype=torch.bool),
        "mask_toggle": torch.zeros(n, tokens, tm.N_OPTIONS, dtype=torch.bool),
        "mask_build_cell": torch.ones(n, 4, tm.GRID, tm.GRID, dtype=torch.bool),
        "mask_move_cell": torch.ones(n, tm.GRID, tm.GRID, dtype=torch.bool),
        "mask_target": torch.zeros(n, tm.NR, tokens, dtype=torch.bool),
        "mask_anchor": torch.zeros(n, tokens, dtype=torch.bool),
        "mask_verb": torch.zeros(n, tm.V, dtype=torch.bool),
    }
    b["mask_verb"][:, [tm.V_END, move, fire]] = True
    b["mask_subject"][:, move, :2] = True          # the third token can do nothing: never an actor
    b["mask_subject"][:, fire, :2] = True
    return b


class UnitsDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = tm.ToolsNet("S", decoder="units")
        self.net.candidates = lambda b: (torch.zeros(2, 1, self.net.d), torch.zeros(2, 1, self.net.d))
        self.b = batch()
        self.inputs = (torch.randn(2, 3, self.net.d), torch.randn(2, self.net.map_ch, tm.GRID, tm.GRID),
                       torch.randn(2, self.net.core_dim), self.b)

    def test_fused_attention_scores_like_the_stock_layer(self):
        import os
        self.net.act_min_p = 0.0
        with torch.no_grad():
            acts, *_ = self.net.decode(*self.inputs, mode="greedy")
            os.environ["FA_TOOLS_MATH_ATTENTION"] = "1"
            try:
                _, lp_stock, _, loss_stock, _ = self.net.decode(*self.inputs, actions=acts)
            finally:
                os.environ["FA_TOOLS_MATH_ATTENTION"] = "0"
            _, lp_fused, _, loss_fused, _ = self.net.decode(*self.inputs, actions=acts)
        torch.testing.assert_close(lp_fused, lp_stock, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(loss_fused, loss_stock, atol=1e-5, rtol=1e-5)

    def test_held_out_games_have_supervised_decisions(self):
        class Fake:
            units = True
            games = [{"unit_supervise": np.array([False, False])}, {"unit_supervise": np.array([True, False])},
                     {"unit_supervise": np.array([False])}, {"unit_supervise": np.array([True])}] * 3
        for seed in range(5):
            held, train = tools_bc.held_out_split(Fake(), np.random.default_rng(seed))
            self.assertEqual(len(held), 1)
            self.assertTrue(all(Fake.games[i]["unit_supervise"].any() for i in held))
            self.assertEqual(sorted(held + train), list(range(12)))

    def test_acting_and_scoring_agree_and_only_actors_act(self):
        self.net.act_min_p = 0.0
        with torch.no_grad():
            for mode in ("sample", "greedy"):
                acts, lp, _, _, _ = self.net.decode(*self.inputs, mode=mode)
                _, lp2, ent, loss, metrics = self.net.decode(*self.inputs, actions=acts)
                torch.testing.assert_close(lp, lp2)
                self.assertTrue(bool((acts["uverb"][:, 2] == 0).all()))
                self.assertEqual(set(acts), set(tm.UNIT_KEYS))
                self.assertTrue(torch.isfinite(loss).all())
                given = acts["uverb"] > 0
                self.assertTrue(bool(((acts["ucell"] >= 0) == (given & (acts["uverb"] == tm.VERBS.index("move")))).all()))

    def test_rare_verbs_are_not_chosen_and_the_choice_is_a_rate(self):
        move, fire = tm.VERBS.index("move"), tm.VERBS.index("fire_state")
        with torch.no_grad():
            self.net.unit_in[-1].weight.zero_()
            self.net.unit_in[-1].bias.zero_()
            self.net.verb_head.weight.zero_()
            self.net.verb_head.bias.zero_()
            self.net.verb_head.bias[move] = -1.0          # move: 27 % of the time; fire_state: 0.07 %
            self.net.verb_head.bias[fire] = -7.0
            counts = torch.zeros(tm.V)
            for _ in range(300):
                acts, _, _, _, _ = self.net.decode(*self.inputs, mode="greedy")     # greedy: still a rate
                counts += torch.bincount(acts["uverb"][:, :2].flatten(), minlength=tm.V)
        self.assertEqual(int(counts[fire]), 0)
        self.assertTrue(0.15 < float(counts[move] / counts.sum()) < 0.40)

    def test_unknown_units_are_not_scored(self):
        with torch.no_grad():
            acts, _, _, _, _ = self.net.decode(*self.inputs, mode="greedy")
            known = {k: v.clone() for k, v in acts.items()}
            known["uverb"][:] = 0
            _, lp_none, _, loss_none, _ = self.net.decode(*self.inputs, actions=known)
            unknown = {k: v.clone() for k, v in known.items()}
            unknown["uverb"][:, 0] = -1
            _, lp_part, _, loss_part, _ = self.net.decode(*self.inputs, actions=unknown)
        self.assertTrue(bool((loss_part < loss_none).all()))      # one unit fewer is scored
        self.assertTrue(bool((lp_part > lp_none).all()))

    def test_a_move_ahead_of_another_order_is_only_the_way_there(self):
        x = tiny_observation(4)
        x["mask_move_cell"] = np.ones((tm.GRID, tm.GRID), bool)
        cmds = [{"verb": "move", "subject": ["u0", "u1"], "cell": (3, 4)},
                {"verb": "attack_move", "subject": ["u1"], "cell": (5, 5)},          # what u1 was sent there for
                {"verb": "move", "subject": ["u1"], "cell": (9, 9)}]                 # after it: not u1's command
        lab = tools_bc.unit_labels(cmds, x, tools_bc.lookups(x))
        self.assertEqual(lab["uverb"].tolist()[:2], [tm.VERBS.index("move"), tm.VERBS.index("attack_move")])
        self.assertEqual(lab["ucell"].tolist()[:2], [4 * tm.GRID + 3, 5 * tm.GRID + 5])

    def test_labels_first_command_wins_and_rejected_is_unknown(self):
        x = tiny_observation(4)
        x["mask_move_cell"] = np.ones((tm.GRID, tm.GRID), bool)
        cmds = [{"verb": "attack_move", "subject": ["u0", "u1"], "cell": (3, 4)},
                {"verb": "patrol", "subject": ["u1"], "cell": (5, 5)},               # u1 already has its command
                {"verb": "fire_state", "subject": ["u2"], "option": 99},             # cannot be expressed
                {"verb": "move", "subject": ["gone"], "cell": (1, 1)}]               # not in the observation
        lab = tools_bc.unit_labels(cmds, x, tools_bc.lookups(x))
        move = tm.VERBS.index("attack_move")
        self.assertEqual(lab["uverb"].tolist(), [move, move, -1, 0])
        self.assertEqual(lab["ucell"].tolist(), [4 * tm.GRID + 3, 4 * tm.GRID + 3, -1, -1])
        back = tm.unit_actions_to_commands({k: np.maximum(v, 0)[None] if k == "uverb" else v[None]
                                            for k, v in lab.items()}, 0, x)
        self.assertEqual(back, [{"verb": "attack_move", "subject": ["u0", "u1"], "cell": (3, 4)}])

    def test_command_arg_indices_match_the_slot_labels(self):
        x = tiny_observation(4)
        x["mask_move_cell"] = np.ones((tm.GRID, tm.GRID), bool)
        x["mask_target"] = np.ones((tm.NR, 4), bool)
        x.update(cand_ids=["fac", "tank"], cand_kind=np.array([0, tm.CAND_MOBILE]), can_build=np.array([[0, 0], [1, 1]]),
                 mask_anchor=np.ones(4, bool))
        look = tools_bc.lookups(x)
        for c in ({"verb": "move", "subject": ["u2", "u3"], "cell": (7, 9)},
                  {"verb": "build", "subject": ["u0"], "bp": "fac", "place": "cell", "cell": (2, 3), "spot": (1, 2)},
                  {"verb": "build", "subject": ["u0"], "bp": "fac", "place": "next", "anchor": "u1"},
                  {"verb": "produce", "subject": ["u1"], "bp": "tank", "count": 5},
                  {"verb": "assist", "subject": ["u2"], "target": "u0"},
                  {"verb": "fire_state", "subject": ["u3"], "option": 1}):
            slot = tm.commands_to_indices([c], x)
            one = tools_bc.command_arg_indices(c, x, look)
            self.assertEqual(one["verb"], int(slot["verb"][0]), c)
            for key in ("bp", "enh", "option", "count", "place", "anchor", "cell", "spot", "target"):
                self.assertEqual(one.get(key, -1), int(slot[key][0]), (c, key))
        self.assertIsNone(tools_bc.command_arg_indices({"verb": "assist", "subject": ["u2"], "target": "gone"}, x, look))

    def test_stack_and_row_round_trip(self):
        rows = [tm.empty_unit_labels(2), tm.empty_unit_labels(3)]
        rows[1]["uverb"][2] = 5
        stacked = tm.stack_actions(rows, 4, "units")
        self.assertEqual(stacked["uverb"].shape, (2, 4))
        self.assertEqual(tm.action_row(stacked, 1, 3, "units")["uverb"].tolist(), [0, 0, 5])
        self.assertEqual(int(stacked["ucell"][0, 3]), -1)


if __name__ == "__main__":
    unittest.main()


class VerbsAfterAttentionTest(unittest.TestCase):
    """The final verbs from the attended context (model option, 2026-10-09)."""

    def setUp(self):
        torch.manual_seed(3)
        self.net = tm.ToolsNet("S", decoder="units")
        self.net.candidates = lambda b: (torch.zeros(2, 1, self.net.d), torch.zeros(2, 1, self.net.d))
        self.net.act_min_p = 0.0
        self.inputs = (torch.randn(2, 3, self.net.d), torch.randn(2, self.net.map_ch, tm.GRID, tm.GRID),
                       torch.randn(2, self.net.core_dim), batch())

    def score(self, acts):
        with torch.no_grad():
            _, lp, _, loss, _ = self.net.decode(*self.inputs, actions=acts)
        return lp, loss

    def test_an_untrained_attention_changes_nothing(self):
        with torch.no_grad():
            acts, *_ = self.net.decode(*self.inputs, mode="greedy")
        before = self.score(acts)
        self.net.verbs_after_attention = True
        torch.testing.assert_close(self.score(acts), before)

    def test_a_trained_attention_reaches_the_verbs_only_with_the_option(self):
        with torch.no_grad():
            for p in list(self.net.unit_attn.parameters()) + list(self.net.intent_in.parameters()):
                p.normal_(0, 0.2)
            acts, *_ = self.net.decode(*self.inputs, mode="greedy")
        off = self.score(acts)
        self.net.verbs_after_attention = True
        on = self.score(acts)
        self.assertFalse(torch.allclose(off[0], on[0]))
        with torch.no_grad():                              # acting and scoring agree with the option on
            acts2, lp, _, _, _ = self.net.decode(*self.inputs, mode="sample")
            _, lp2, _, _, _ = self.net.decode(*self.inputs, actions=acts2)
        torch.testing.assert_close(lp, lp2)

    def test_the_option_travels_with_the_checkpoint(self):
        self.assertEqual(tm.model_options(self.net), {"verbs_after_attention": False, "fine_positions": False})
        self.net.verbs_after_attention = True
        state = {"model": self.net.state_dict(), "model_options": tm.model_options(self.net)}
        other = tm.ToolsNet("S", decoder="units")
        self.assertEqual(tm.apply_model_options(other, state), {"verbs_after_attention": True, "fine_positions": False})
        self.assertTrue(other.verbs_after_attention)
        self.assertEqual(tm.apply_model_options(other, {"model": {}}), {"verbs_after_attention": False, "fine_positions": False})
