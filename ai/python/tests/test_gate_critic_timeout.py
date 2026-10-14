import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fa_train import tools_model as tm
from fa_train.learn_rl import rewards_of
from fa_train.rl_loop import champion_gate, sha256


def write_round(root: Path, name: str, sha: str, games: list):
    """games: [(ai, faction, ai_faction, won)] on a small map."""
    r = root / name
    (r / "eval").mkdir(parents=True)
    (r / "round.json").write_text(json.dumps({"checkpoint_sha256": sha}))
    (r / "eval" / "episodes.jsonl").write_text("\n".join(json.dumps(
        {"ai": ai, "faction": f, "ai_faction": af, "map": "SCMP_012", "outcome": "victory" if won else "defeat"})
        for ai, f, af, won in games))
    (r / "DONE").write_text("x")


class ChampionGateTest(unittest.TestCase):
    def setup(self, root: Path, rates: dict):
        """rates: network -> win rate against rush, 200 games each (UEF vs UEF)."""
        ckpts = root / "checkpoints"
        ckpts.mkdir()
        start = root / "start.pt"
        start.write_bytes(b"start")
        for i, (net, rate) in enumerate(rates.items()):
            path = ckpts / f"{net}.pt"
            path.write_bytes(net.encode())
            write_round(root / "rounds", f"lane{i}_20261014T0{i}0000Z", sha256(path),
                        [("rush", 1, 1, j < rate * 200) for j in range(200)])
        return root / "rounds", start, ckpts

    def gate(self, rates, order, champion="rl_001"):
        with tempfile.TemporaryDirectory() as d:
            rounds, start, ckpts = self.setup(Path(d), rates)
            return champion_gate(rounds, start, ckpts, champion, order, 100, "20261013T2310Z")

    def test_level_networks_pass(self):
        self.assertIsNone(self.gate({"rl_001": 0.85, "rl_002": 0.82}, ["rl_001", "rl_002"]))

    def test_a_clearly_worse_newest_goes_back_to_the_champion(self):
        kind, what, name = self.gate({"rl_001": 0.85, "rl_002": 0.80, "rl_003": 0.65}, ["rl_001", "rl_002", "rl_003"])
        self.assertEqual((kind, name), ("worse", "rl_001"))
        self.assertIn("rl_003", what)

    def test_a_clearly_better_network_becomes_the_champion(self):
        kind, _, name = self.gate({"rl_001": 0.70, "rl_002": 0.85, "rl_003": 0.83}, ["rl_001", "rl_002", "rl_003"])
        self.assertEqual((kind, name), ("better", "rl_003"))           # the newest clearly better one
        kind, _, name = self.gate({"rl_001": 0.70, "rl_002": 0.85, "rl_003": 0.72}, ["rl_001", "rl_002", "rl_003"])
        self.assertEqual((kind, name), ("better", "rl_002"))           # rl_003 level with the champion

    def test_worse_after_a_better_one_goes_back_to_the_better_one(self):
        kind, _, name = self.gate({"rl_001": 0.70, "rl_002": 0.88, "rl_003": 0.50}, ["rl_001", "rl_002", "rl_003"])
        self.assertEqual((kind, name), ("worse", "rl_002"))

    def test_only_networks_after_the_champion_and_in_the_lineage(self):
        self.assertIsNone(self.gate({"rl_001": 0.85, "rl_002": 0.40}, ["rl_001"]))
        self.assertIsNone(self.gate({"rl_001": 0.40, "rl_002": 0.85}, ["rl_001", "rl_002"], champion="rl_002"))


class OpponentCriticTest(unittest.TestCase):
    def test_features(self):
        np.testing.assert_allclose(tm.opponent_features("m28aicheat@0.6"), [0, 0, 0, 0, 1, 1, 0.6, 0])
        np.testing.assert_array_equal(tm.opponent_features("m28ai"), [0, 0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(tm.opponent_features("self@rl_089"), [0, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(tm.opponent_features("rush"), [0, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(tm.opponent_features("medium"), [0, 0, 0, 1, 0, 0, 0, 0])

    def test_opponent_of_recording(self):
        self.assertEqual(tm.opponent_of_recording("0000122_self@rl_089_657141010.jsonl.gz"), "self@rl_089")
        self.assertEqual(tm.opponent_of_recording("0000007_m28aicheat@0.6_12.jsonl.gz"), "m28aicheat@0.6")

    def test_a_fresh_opponent_part_leaves_the_value_unchanged(self):
        net = tm.ToolsNet("S", decoder="units").eval()
        z = torch.randn(5, net.value_head[0].in_features - 2 * len(tm.PRIVILEGED))
        b = {"privileged": torch.randn(5, 2 * len(tm.PRIVILEGED))}
        plain = net.value_of(z, b)
        told = net.value_of(z, {**b, "opponent": torch.as_tensor(np.stack([tm.opponent_features("rush")] * 5))})
        torch.testing.assert_close(plain, told)

    def test_an_older_checkpoint_loads_without_it(self):
        net = tm.ToolsNet("S", decoder="units")
        old = {k: v for k, v in net.state_dict().items() if not k.startswith("opp_")}
        tm.ToolsNet("S", decoder="units").load_state_dict(old)


def game(outcome: str, terminal: float, n: int = 4) -> dict:
    g = {"meta": np.array(json.dumps({"map": "SCMP_012", "outcome": outcome}))}
    for k in ("terminal", "trade", "economy", "commander"):
        g[f"rew_{k}"] = np.zeros(n, np.float32)
    g["rew_terminal"][-1] = terminal
    return g


def args(**kw):
    base = dict(econ_source="recorded", econ_coef=0.0, intel_coef=0.0, map_coef=0.0, waste_coef=0.0, stall_coef=0.0,
                shaping=1.0, gamma=0.997, acu_coef=0.0, acu_grace=1.0, acu_span=4.0, acu_all_maps=False,
                timeout_scale=None, timeout_cost=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


class TimeoutTest(unittest.TestCase):
    def test_as_recorded_by_default(self):
        self.assertAlmostEqual(float(rewards_of(game("timeout", -2.5), args())[-1]), -2.5)

    def test_scored_by_who_was_ahead(self):
        a = args(timeout_scale=10.0, timeout_cost=2.0)
        self.assertAlmostEqual(float(rewards_of(game("timeout", -2.5), a)[-1]), -7.0)    # behind: va -0.5
        self.assertAlmostEqual(float(rewards_of(game("timeout", 5.0), a)[-1]), 8.0)      # far ahead: 10 - 2
        self.assertAlmostEqual(float(rewards_of(game("defeat", -10.0), a)[-1]), -10.0)   # only timeouts change


if __name__ == "__main__":
    unittest.main()
