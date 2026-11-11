"""2026-10-26 data review: the 400-game look falls back too (B1), unstable updates are not played (T1), the stretch
slot's offset (T3), and "run" names the next experiment even when it is the running one."""
try:                                   # first: no test may reach the server or start a real process
    import no_network  # noqa: F401
except ImportError:
    from tests import no_network  # noqa: F401
import unittest
from pathlib import Path
from unittest.mock import patch

from fa_train import autopilot
from fa_train.rl_loop import judge_gate, opponent_mix, unstable_update

COND = ("m28aicheat@0.75", 1, 2, ("env",))


def games(rate_first: float, n_first: int, rate_rest: float = None, n_rest: int = 0) -> list:
    out = [(COND, int(i < round(rate_first * n_first))) for i in range(n_first)]
    if n_rest:
        out += [(COND, int(i < round(rate_rest * n_rest))) for i in range(n_rest)]
    return out


class PromotionLookFallsBack(unittest.TestCase):
    def test_clearly_worse_at_400_goes_back_to_the_champion(self):
        # passes the 150-game look (level), then collapses: 28 % over 400 against the champion's 60 %
        seqs = {"rl_1": games(0.6, 2000), "rl_2": games(0.6, 150, 0.1, 250)}
        state: dict = {}
        verdict = judge_gate(seqs, "rl_1", ["rl_1", "rl_2"], state)
        self.assertEqual(verdict[0], "worse")
        self.assertEqual(verdict[2], "rl_1")
        self.assertEqual(state["verdicts"]["rl_2@rl_1"]["rollback"]["verdict"], "ok")
        self.assertEqual(state["verdicts"]["rl_2@rl_1"]["promotion"]["verdict"], "worse")

    def test_level_at_400_is_only_not_promoted(self):
        seqs = {"rl_1": games(0.6, 2000), "rl_2": games(0.6, 400)}
        state: dict = {}
        self.assertIsNone(judge_gate(seqs, "rl_1", ["rl_1", "rl_2"], state))
        self.assertEqual(state["verdicts"]["rl_2@rl_1"]["promotion"]["verdict"], "not better")

    def test_a_later_network_that_passed_does_not_hide_a_worse_400_look(self):
        seqs = {"rl_1": games(0.6, 2000), "rl_2": games(0.6, 150, 0.1, 250), "rl_3": games(0.6, 150)}
        verdict = judge_gate(seqs, "rl_1", ["rl_1", "rl_2", "rl_3"], {})
        # rl_3 passed its 150 look after rl_2: the line recovered (as for 150-game rollbacks)
        self.assertIsNone(verdict)


class UnstableUpdate(unittest.TestCase):
    def test_last_epoch_decides(self):
        hist = [{"gnorm": 1.2, "clip": 0.05}, {"gnorm": 0.7, "clip": 0.02}, {"sil_pg": 0.3}]
        self.assertIsNone(unstable_update(hist, 0.9, 0.03))
        self.assertIn("gradient norm", unstable_update([{"gnorm": 0.95, "clip": 0.02}], 0.9, 0.03))
        self.assertIn("clip fraction", unstable_update([{"gnorm": 0.8, "clip": 0.04}], 0.9, 0.03))

    def test_off_and_missing_history(self):
        self.assertIsNone(unstable_update([{"gnorm": 5.0, "clip": 0.5}], 0.0, 0.0))
        self.assertIsNone(unstable_update(None, 0.9, 0.03))
        self.assertIsNone(unstable_update([], 0.9, 0.03))

    def test_the_four_champions_pass(self):
        for gnorm, clip in ((0.769, 0.0202), (0.686, 0.0248), (0.735, 0.0241), (0.721, 0.0227)):
            self.assertIsNone(unstable_update([{"gnorm": gnorm, "clip": clip}], 0.9, 0.03))


class StretchOffset(unittest.TestCase):
    def test_offset(self):
        self.assertIn("m28aicheat@0.9", opponent_mix(0.75, "adaptive", 3, full_weight=0))
        mix = opponent_mix(0.75, "adaptive,rush,tech", 6, full_weight=0, stretch_weight=3, stretch_offset=0.1)
        self.assertEqual(mix, "adaptive,rush,tech,m28aicheat@0.75:6,m28aicheat@0.85:3")


class RunCommand(unittest.TestCase):
    def test_run_names_the_next_experiment_even_when_it_is_running(self):
        p = object.__new__(autopilot.Autopilot)
        p.dry, p.s = True, {}
        p.dir = Path(__file__).resolve().parent / "unused"
        p.state = {"mode": "running", "experiment": {"name": "X3d"}, "results": {"X3d": {"verdict": "SKIP"}}}
        p.queue = lambda: [{"name": "X3d"}, {"name": "X10"}]
        with patch.object(autopilot, "log_event"):
            p.apply_command({"cmd": "run", "experiment": "X3d"}, "rl_188")
        self.assertEqual(p.state["run_next"], "X3d")
        self.assertTrue(p.state["experiment"]["skip"])
        self.assertNotIn("X3d", p.state["results"])

    def test_champion_command_runs_the_experiment_again(self):
        p = object.__new__(autopilot.Autopilot)
        p.dry, p.s = True, {}
        p.run = Path(__file__).resolve().parent
        p.state = {"mode": "running", "experiment": {"name": "X3d"}, "results": {"X3d": {}}}
        with patch.object(Path, "exists", return_value=True), patch.object(autopilot, "log_event"):
            p.apply_command({"cmd": "champion", "name": "rl_178"}, "rl_188")
        self.assertEqual((p.state["set_champion"], p.state["run_next"]), ("rl_178", "X3d"))


class OverrideFile(unittest.TestCase):
    def test_bom_is_read_and_a_bad_file_is_kept_aside(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = object.__new__(autopilot.Autopilot)
            p.dir, p.s, p.state = Path(d), {}, {"mode": "running"}
            p.alerts = type("A", (), {"send": lambda *a, **k: None})()
            seen = []
            p.apply_command = lambda c, champion: seen.append(c["cmd"])
            (Path(d) / "override.json").write_bytes(b'\xef\xbb\xbf{"cmd": "hold"}')
            with patch.object(autopilot, "ntfy_commands", return_value=[]), patch.object(autopilot, "log_event"):
                p.commands("rl_188")
                self.assertEqual(seen, ["hold"])
                (Path(d) / "override.json").write_text("{not json", encoding="utf-8")
                p.commands("rl_188")
            self.assertEqual(len(list(Path(d).glob("override.*.bad.json"))), 1)


if __name__ == "__main__":
    unittest.main()
