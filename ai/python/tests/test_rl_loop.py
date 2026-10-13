import json
import tempfile
import unittest
from pathlib import Path

from fa_train.rl_loop import lineage, m28_level_rate, matched, opponent_mix, regression, sha256


def write_round(root: Path, name: str, sha: str, outcomes: dict, done: bool = True):
    r = root / name
    r.mkdir(parents=True)
    (r / "round.json").write_text(json.dumps({"checkpoint_sha256": sha, "outcomes": {
        ai: {"games": g, "outcomes": {"victory": w}} for ai, (w, g) in outcomes.items()}}))
    if done:
        (r / "DONE").write_text("x")


class RlLoopTest(unittest.TestCase):
    def test_matched_reweights_the_start_to_the_networks_mix(self):
        by_ai = {"start": {"medium": [90, 100], "rush": [60, 100], "m28ai": [0, 50]},
                 "rl_001": {"rush": [70, 100], "m28ai": [1, 50]}}
        w, g, base = matched(by_ai, "rl_001")
        self.assertEqual((w, g), (70, 100))            # M28 and opponents the start did not play are left out
        self.assertAlmostEqual(base, 0.6)               # the start on rush only, not its easier pooled 75 %

    def test_opponent_mix(self):
        self.assertEqual(opponent_mix(0.65, "adaptive,rush", 2.0),
                         "adaptive,rush,m28aicheat@0.65:2,m28aicheat@0.8,m28ai")
        self.assertEqual(opponent_mix(0.9, "adaptive", 2.0), "adaptive,m28aicheat@0.9:2,m28ai")   # no stretch at 1
        self.assertEqual(opponent_mix(0.6, "rush", 2.0, "rl_077,rl_089"),
                         "rush,m28aicheat@0.6:2,m28aicheat@0.75,m28ai,self@rl_077,self@rl_089")

    def test_m28_level_rate_counts_only_allowed_finished_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_round(root, "lane0_a", "net1", {"m28aicheat@0.6": (5, 12), "m28aicheat@0.7": (1, 12)})
            write_round(root, "lane1_b", "net2", {"m28aicheat@0.6": (7, 12)})
            write_round(root, "lane2_c", "old", {"m28aicheat@0.6": (12, 12)})
            write_round(root, "lane3_d", "net1", {"m28aicheat@0.6": (12, 12)}, done=False)
            self.assertEqual(m28_level_rate(root, {"net1", "net2"}, 0.6), (12, 24))


class LineageTest(unittest.TestCase):
    def test_a_fallback_cuts_the_abandoned_branch(self):
        rows = [{"checkpoint": "n1"}, {"checkpoint": "n2"}, {"checkpoint": "n3"}, {"exit": 1},
                {"checkpoint": "n1", "fallback": True}, {"checkpoint": "n4"},
                {"checkpoint": "g", "fallback": True}, {"checkpoint": "n5"}]
        self.assertEqual([p.name for p in lineage(Path("start"), rows)], ["start", "n1", "n4", "g", "n5"])


class RegressionGuardTest(unittest.TestCase):
    def setup(self, root: Path, results: dict):
        """results: network -> (wins, games) against rush; 'start' plays 200 games at 75 %."""
        ckpts = root / "checkpoints"
        ckpts.mkdir()
        start = root / "start.pt"
        start.write_bytes(b"start")
        write_round(root / "rounds", "lane0_s", sha256(start), {"rush": (150, 200)})
        for i, (net, (w, g)) in enumerate(results.items()):
            path = ckpts / f"{net}.pt"
            path.write_bytes(net.encode())
            write_round(root / "rounds", f"lane{i + 1}_{net}", sha256(path), {"rush": (w, g)})
        return root / "rounds", start, ckpts

    def test_one_bad_old_network_is_not_a_trend(self):
        with tempfile.TemporaryDirectory() as d:
            rounds, start, ckpts = self.setup(Path(d), {"rl_001": (95, 144), "rl_002": (110, 144)})
            self.assertIsNone(regression(rounds, start, ckpts, 8.0, 140))      # the newest (76 %) decides

    def test_bad_newest_falls_back_to_the_newest_good_network(self):
        with tempfile.TemporaryDirectory() as d:
            rounds, start, ckpts = self.setup(Path(d), {"rl_001": (120, 144), "rl_002": (110, 144),
                                                         "rl_003": (80, 144)})
            what, back = regression(rounds, start, ckpts, 8.0, 140)
            self.assertIn("rl_003", what)
            self.assertEqual(back.name, "rl_002.pt")                            # 76 % >= the start's 75 %

    def test_only_the_lineage_is_judged(self):
        with tempfile.TemporaryDirectory() as d:
            rounds, start, ckpts = self.setup(Path(d), {"rl_001": (120, 144), "rl_002": (40, 144)})
            self.assertIsNotNone(regression(rounds, start, ckpts, 8.0, 140))                 # rl_002 by name
            self.assertIsNone(regression(rounds, start, ckpts, 8.0, 140, order=["rl_001"]))  # abandoned: not judged

    def test_pooled_newest_networks(self):
        with tempfile.TemporaryDirectory() as d:
            rounds, start, ckpts = self.setup(Path(d), {"rl_001": (110, 144), "rl_002": (40, 72),
                                                         "rl_003": (40, 72)})
            what, back = regression(rounds, start, ckpts, 8.0, 140)          # 80/144 = 56 %
            self.assertIn("rl_002+rl_003", what)
            self.assertEqual(back.name, "rl_001.pt")


if __name__ == "__main__":
    unittest.main()
