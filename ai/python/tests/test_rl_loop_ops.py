"""The RL loop's unattended-operation fixes (2026-10-17, the 2026-10-16 review's B8-B11 and bugs-loop F1-F17): the
champion file, the judge-once gate, the rejection cap, retries instead of crashes, the STOP/YIELD/lock contract,
the rounds of other networks, arrival-based archiving, the storage budget, the checkpoint hash cache, the player's
seeds. CPU only, temp folders; ssh/scp, the encoder and learn_rl are replaced by recorders (nothing is started)."""
try:                                   # first: no test may reach the server or start a real process
    import no_network  # noqa: F401
except ImportError:
    from tests import no_network  # noqa: F401
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fa_train import player, rl_loop, server_sync, storage
from fa_train.rl_loop import judge_gate, sha256


def seq(rate: float, n: int, conds=(("rush", 1, 1), ("adaptive", 2, 3), ("m28aicheat@0.6", 3, 2)), seed=0):
    rng = random.Random(seed)
    return [(conds[i % len(conds)], int(rng.random() < rate)) for i in range(n)]


class JudgeGateTest(unittest.TestCase):
    ORDER = ["rl_001", "rl_002", "rl_003"]

    def test_judged_once_on_the_first_games(self):
        state = {}
        seqs = {"rl_001": seq(0.85, 600), "rl_002": seq(0.84, 149, seed=1)}
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state))
        self.assertNotIn("rollback", state["verdicts"]["rl_002@rl_001"])         # 149 games: not yet
        # later games lose them all (fewer than the 400-game look, which would now fall back on them: review B1)
        seqs["rl_002"] = seq(0.84, 150, seed=1) + seq(0.0, 200, seed=2)
        notes = []
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state, notes=notes))
        rb = state["verdicts"]["rl_002@rl_001"]["rollback"]
        self.assertEqual((rb["verdict"], rb["n"]), ("ok", 150))                  # judged on exactly the first 150
        self.assertEqual(len([n for n in notes if "kept" in n]), 1)
        # never again: the 300 lost games after them change nothing (no repeated looks)
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state))
        self.assertEqual(state["verdicts"]["rl_002@rl_001"]["rollback"], rb)

    def test_a_clearly_worse_network_goes_back_to_the_champion(self):
        state = {}
        kind, what, name = judge_gate({"rl_001": seq(0.85, 600), "rl_002": seq(0.55, 160, seed=3)}, "rl_001",
                                      self.ORDER, state)
        self.assertEqual((kind, name), ("worse", "rl_001"))
        self.assertIn("rl_002", what)
        # a restart re-reads the same verdict (state kept in gate_state.json) and does not judge again
        again = judge_gate({"rl_001": seq(0.85, 600), "rl_002": seq(0.85, 160, seed=3)}, "rl_001", self.ORDER, state)
        self.assertEqual(again[0], "worse")

    def test_a_later_network_that_passed_clears_an_earlier_worse_one(self):
        state = {}
        seqs = {"rl_001": seq(0.85, 600), "rl_002": seq(0.55, 160, seed=3), "rl_003": seq(0.86, 160, seed=4)}
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state))

    def test_promotion_needs_400_games_z_2_5_and_3_points(self):
        state = {}
        seqs = {"rl_001": seq(0.70, 900), "rl_002": seq(0.90, 399, seed=5)}
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state))       # passed, not yet 400 games
        seqs["rl_002"] = seq(0.90, 400, seed=5)
        kind, what, name = judge_gate(seqs, "rl_001", self.ORDER, state)
        self.assertEqual((kind, name), ("better", "rl_002"))
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state))       # acted on once
        # a small real gain (+2 points) is not promoted even when significant
        state = {}
        seqs = {"rl_001": seq(0.70, 900), "rl_002": seq(0.72, 400, seed=6)}
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state, se_inflation=0.01))
        self.assertEqual(state["verdicts"]["rl_002@rl_001"]["promotion"]["verdict"], "not better")

    def test_se_inflation_widens_the_interval(self):
        seqs = {"rl_001": seq(0.85, 600), "rl_002": seq(0.78, 150, seed=7)}
        s1, s2 = {}, {}
        judge_gate(seqs, "rl_001", self.ORDER, s1, se_inflation=1.0)
        judge_gate(seqs, "rl_001", self.ORDER, s2, se_inflation=1.2)
        a, b = s1["verdicts"]["rl_002@rl_001"]["rollback"], s2["verdicts"]["rl_002@rl_001"]["rollback"]
        self.assertAlmostEqual(b["se"] / a["se"], 1.2, places=2)

    def test_a_replaced_file_is_judged_anew(self):
        state = {}
        seqs = {"rl_001": seq(0.85, 600), "rl_002": seq(0.55, 160, seed=3)}
        judge_gate(seqs, "rl_001", self.ORDER, state, shas={"rl_002": "aa"})
        seqs["rl_002"] = seq(0.86, 160, seed=8)
        self.assertIsNone(judge_gate(seqs, "rl_001", self.ORDER, state, shas={"rl_002": "bb"}))

    def test_off_without_the_champion_in_the_lineage(self):
        self.assertIsNone(judge_gate({"rl_001": seq(0.85, 600), "rl_002": seq(0.2, 200)}, "rl_009",
                                     self.ORDER, {}))


class Run:
    """A run folder: checkpoints, journal, rounds with eval data (and raw recordings), mocked players."""

    def __init__(self):
        self.d = Path(tempfile.mkdtemp())
        self.run = self.d / "run"
        self.ckpts = self.run / "checkpoints"
        self.ckpts.mkdir(parents=True)
        (self.run / "rounds").mkdir()
        self.init = self.d / "init.pt"
        self.init.write_bytes(b"init")
        self.pushed, self.configs, self.encoded, self.learned = [], [], [], []
        self.push_fail = 0
        self.learn_codes = []

    def ckpt(self, name: str) -> Path:
        p = self.ckpts / f"{name}.pt"
        p.write_bytes(name.encode())
        return p

    def journal(self, *names, fallback_to=None):
        rows = [{"iteration": i + 1, "checkpoint": str(self.ckpt(n))} for i, n in enumerate(names)]
        if fallback_to:
            rows.append({"iteration": len(rows) + 1, "gate": "x", "checkpoint": str(self.ckpts / f"{fallback_to}.pt"),
                         "fallback": True})
        (self.run / "rl.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def round(self, name: str, net: str, rate: float = 0.8, n: int = 30, seed: int = 0, arrived: bool = True,
              where: Path | None = None):
        r = (where or self.run / "rounds") / name
        (r / "eval").mkdir(parents=True)
        (r / "raw").mkdir()
        sha = sha256(self.init if net == "start" else self.ckpts / f"{net}.pt")
        (r / "round.json").write_text(json.dumps({"checkpoint_sha256": sha, "games": n}))
        rng = random.Random(seed)
        (r / "eval" / "episodes.jsonl").write_text("\n".join(json.dumps(
            {"ai": "rush", "faction": 1, "ai_faction": 1, "map": "SCMP_012", "seed": i,
             "outcome": "victory" if rng.random() < rate else "defeat"}) for i in range(n)))
        for i in range(n):
            (r / "raw" / f"{i:07d}_rush_{i}.jsonl.gz").write_bytes(b"x" * 10)
        (r / "DONE").write_text("x")
        if arrived:
            (r / "ARRIVED").write_text("now")
        return r

    def rows(self):
        return rl_loop.read_journal(self.run / "rl.jsonl")

    def main(self, *extra, sleep=None):
        def push(host, home, ck, **kw):
            if self.push_fail:
                self.push_fail -= 1
                raise subprocess.CalledProcessError(255, "scp")
            self.pushed.append(Path(ck).stem)

        def fake_run(args, log):
            if "fa_train.encode_parallel" in args:
                self.encoded.append(Path(args[args.index("--raw") + 1]).parent.name)
                return 0
            self.learned.append(args)
            code = self.learn_codes.pop(0) if self.learn_codes else 0
            if code == 0:
                Path(args[args.index("--out") + 1]).write_bytes(b"new" + str(len(self.learned)).encode())
            return code

        def stop_sleep(s):
            if s == 300:                                     # the wait for new games: end the test run there
                (self.run / "STOP").write_text("test")

        argv = ["--run", str(self.run), "--init", str(self.init), "--push", "--min-free-gb", "0", "--min-new-games",
                "1", *extra]
        with mock.patch.object(server_sync, "push", push), \
                mock.patch.object(server_sync, "pull", lambda *a, **k: 0), \
                mock.patch.object(server_sync, "set_config", lambda h, home, u: self.configs.append(u) or u), \
                mock.patch.object(rl_loop, "run", fake_run), \
                mock.patch.object(rl_loop, "learn_history", lambda out: [{"epoch": 1}]), \
                mock.patch.object(server_sync, "ssh", lambda *a, **k: self.configs.append("ssh!")), \
                mock.patch.object(server_sync, "remote_rounds", lambda *a, **k: []), \
                mock.patch.object(rl_loop, "league_push", lambda *a, **k: self.configs.append("league")), \
                mock.patch.object(rl_loop.time, "sleep", sleep or stop_sleep):
            try:
                rl_loop.main(argv)
            except SystemExit as exc:
                return exc.code
        return None

    def close(self):
        shutil.rmtree(self.d, ignore_errors=True)


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.r = Run()

    def tearDown(self):
        self.r.close()

    def test_champion_file_wins_over_the_flag_and_set_champion_replaces_it(self):
        r = self.r
        r.journal("rl_001", "rl_002")
        (r.run / "champion.json").write_text(json.dumps({"name": "rl_002"}))
        (r.run / "YIELD").write_text("test")
        self.assertEqual(r.main("--champion", "rl_001"), rl_loop.EXIT_YIELD)
        self.assertEqual(json.loads((r.run / "champion.json").read_text())["name"], "rl_002")
        self.assertEqual(r.main("--champion", "rl_001", "--set-champion", "rl_001"), rl_loop.EXIT_YIELD)
        self.assertEqual(json.loads((r.run / "champion.json").read_text())["name"], "rl_001")
        (r.run / "champion.json").unlink()
        r.main("--champion", "rl_001")                       # missing: the flag seeds it
        self.assertEqual(json.loads((r.run / "champion.json").read_text())["name"], "rl_001")

    def test_wide_map_level_writes_the_opponents_and_their_maps_and_steps(self):
        r = self.r
        r.journal("rl_001", "rl_002")
        home = r.d / "player"
        home.mkdir()
        (home / "config.json").write_text(json.dumps({"lanes": 6, "ai_maps": {"m28aicheat": "SCMP_012,SCMP_019",
                                                                               "rush": "SCMP_017"}}))
        wide = ("--local-home", str(home), "--m28-curriculum", "--m28-start", "0.75", "--m28-weight", "4",
                "--stretch-weight", "2", "--full-m28-weight", "0", "--m28-min-games", "10", "--m28-wide-start", "0.55",
                "--m28-wide-min-games", "10", "--m28-maps", "SCMP_012,SCMP_013", "--m28-wide-maps", "SCMP_019,SCMP_017")
        (r.run / "YIELD").write_text("test")
        r.main(*wide)
        cfg = json.loads((home / "config.json").read_text())
        self.assertEqual(cfg["lanes"], 6)
        self.assertEqual(cfg["ai"], "adaptive,rush,tech,m28aicheat@0.75:4,m28aicheat@0.9:2,m28aicheat@0.55:4")
        self.assertEqual(cfg["ai_maps"], {"rush": "SCMP_017", "m28aicheat@0.75": "SCMP_012,SCMP_013",
                                          "m28aicheat@0.9": "SCMP_012,SCMP_013", "m28aicheat@0.55": "SCMP_019,SCMP_017"})
        self.assertEqual(json.loads((r.run / "curriculum.json").read_text()), {"m28_level": 0.75, "m28_wide_level": 0.55})
        # 12 of 12 won on the wide maps at x0.55 and 2 of 12 on the small ones at x0.75: the wide level goes up, the
        # other one down, each on its own maps' games (the wide maps' x0.75 wins and the small maps' x0.55 losses,
        # games from before the split, move nothing)
        (r.run / "YIELD").unlink()
        rd = r.round("lane0_20270101T000000Z", "rl_002")
        games = [("m28aicheat@0.55", "SCMP_019", "victory")] * 12 + [("m28aicheat@0.55", "SCMP_012", "defeat")] * 30 \
            + [("m28aicheat@0.75", "SCMP_013", "victory")] * 2 + [("m28aicheat@0.75", "SCMP_013", "defeat")] * 10 \
            + [("m28aicheat@0.75", "SCMP_019", "victory")] * 30
        (rd / "eval" / "episodes.jsonl").write_text("\n".join(json.dumps(
            {"ai": ai, "faction": 1, "ai_faction": 1, "map": m, "seed": i, "outcome": o})
            for i, (ai, m, o) in enumerate(games)))
        r.main(*wide)
        self.assertEqual(json.loads((r.run / "curriculum.json").read_text()), {"m28_level": 0.7, "m28_wide_level": 0.6})
        cfg = json.loads((home / "config.json").read_text())
        self.assertEqual(cfg["ai"], "adaptive,rush,tech,m28aicheat@0.7:4,m28aicheat@0.85:2,m28aicheat@0.6:4")
        self.assertEqual(cfg["ai_maps"], {"rush": "SCMP_017", "m28aicheat@0.7": "SCMP_012,SCMP_013",
                                          "m28aicheat@0.85": "SCMP_012,SCMP_013", "m28aicheat@0.6": "SCMP_019,SCMP_017"})

    def test_every_start_pushes_the_current_network(self):
        r = self.r
        r.journal("rl_001", "rl_002")
        (r.run / "YIELD").write_text("test")
        r.main()
        self.assertEqual(r.pushed, ["rl_002"])               # the journal is not empty: it pushed nothing before

    def test_start_from_champion_pushes_before_its_row(self):
        r = self.r
        r.journal("rl_001", "rl_002")
        (r.run / "champion.json").write_text(json.dumps({"name": "rl_001"}))
        (r.run / "YIELD").write_text("test")
        r.push_fail = 3                                      # the server is away: the row says so, no crash
        self.assertEqual(r.main("--start-from", "champion", "--tag", "X1"), rl_loop.EXIT_YIELD)
        last = r.rows()[-1]
        self.assertEqual((Path(last["checkpoint"]).stem, last["fallback"], last["pushed"], last["experiment"]),
                         ("rl_001", True, False, "X1"))
        self.assertIn("time", last)

    def test_stop_at_start_is_not_deleted(self):
        r = self.r
        r.journal("rl_001")
        (r.run / "STOP").write_text("")
        self.assertEqual(r.main(), rl_loop.EXIT_STOP_PRESENT)
        self.assertTrue((r.run / "STOP").exists())
        self.assertEqual(json.loads((r.run / rl_loop.EXIT_FILE).read_text())["code"], rl_loop.EXIT_STOP_PRESENT)
        (r.run / "YIELD").write_text("test")
        self.assertEqual(r.main("--clear-stop"), rl_loop.EXIT_YIELD)
        self.assertFalse((r.run / "STOP").exists())
        self.assertTrue(list(r.run.glob("STOP.*")))
        (r.run / "STOP").write_text("rl_loop: its own")
        self.assertEqual(r.main(), rl_loop.EXIT_YIELD)

    def test_one_loop_per_run(self):
        r = self.r
        r.journal("rl_001")
        held = rl_loop._lock(r.run)
        try:
            self.assertEqual(r.main(), rl_loop.EXIT_LOCKED)
        finally:
            rl_loop._unlock(held)

    def test_learn_rl_is_retried_once_then_exit_3(self):
        r = self.r
        r.journal("rl_001")
        r.round("lane0_20261017T010000Z", "rl_001")
        r.learn_codes = [1, 1]
        self.assertEqual(r.main("--learn-retry-wait", "0", "--gate-games", "0"), rl_loop.EXIT_LEARN_FAILED)
        self.assertEqual(len(r.learned), 2)
        self.assertEqual(r.rows()[-1]["exit"], 1)
        r.learn_codes = [1, 0]                               # a transient failure: the retry trains
        r.round("lane0_20261017T020000Z", "rl_001", seed=1)  # (the failed iteration's row consumed the first)
        code = r.main("--learn-retry-wait", "0", "--gate-games", "0", "--iterations", "3")
        self.assertEqual(code, rl_loop.EXIT_DONE)
        self.assertTrue(r.rows()[-1].get("retried"))
        self.assertTrue(r.rows()[-1]["checkpoint"].endswith("rl_003.pt"))

    def test_a_failed_push_after_training_is_retried_next_pass(self):
        r = self.r
        r.journal("rl_001")
        r.round("lane0_20261017T010000Z", "rl_001")
        r.push_fail = 4               # the start's push (3 tries) and the next pass's first try of the pending one
        code = r.main("--gate-games", "0", "--iterations", "2")
        self.assertEqual(code, rl_loop.EXIT_DONE)
        self.assertEqual(r.pushed, ["rl_001", "rl_002"])     # the pending push went through, then the new network

    def test_regression_guard_only_warns(self):
        r = self.r
        r.journal("rl_001")
        r.round("lane0_20261017T000000Z", "start", rate=0.8, n=200)
        r.round("lane1_20261017T010000Z", "rl_001", rate=0.3, n=200, seed=1)
        code = r.main("--gate-games", "0", "--collapse-games", "0", "--iterations", "3")
        self.assertEqual(code, rl_loop.EXIT_DONE)
        rows = r.rows()
        self.assertTrue(any(row.get("warning") == "regression" for row in rows))
        self.assertTrue(r.learned)                           # it trained on after the warning

    def test_collapse_goes_back_to_the_champion(self):
        r = self.r
        r.journal("rl_001", "rl_002", "rl_003")
        (r.run / "champion.json").write_text(json.dumps({"name": "rl_001"}))
        r.round("lane0_20261017T010000Z", "rl_002", rate=0.9, n=30)        # above the floor, but not the champion
        r.round("lane0_20261017T020000Z", "rl_003", rate=0.1, n=30, seed=2)
        with mock.patch.object(rl_loop, "judge_gate", lambda *a, **k: None):     # (the collapse alone)
            r.main("--iterations", "4")
        fb = [row for row in r.rows() if row.get("collapse")]
        self.assertEqual(Path(fb[0]["checkpoint"]).stem, "rl_001")          # not rl_002

    def test_gate_rejections_are_capped(self):
        r = self.r
        r.journal("rl_001", "rl_002")
        (r.run / "champion.json").write_text(json.dumps({"name": "rl_001"}))
        r.round("lane0_20261017T010000Z", "rl_001", rate=0.85, n=300)
        r.round("lane1_20261017T020000Z", "rl_002", rate=0.5, n=200, seed=3)
        code = r.main("--max-rejections", "1", "--collapse-games", "0")
        self.assertEqual(code, rl_loop.EXIT_GATE_EXHAUSTED)
        self.assertTrue((r.run / "GATE_EXHAUSTED.json").exists())
        fb = r.rows()[-1]
        self.assertEqual((Path(fb["checkpoint"]).stem, fb["fallback"]), ("rl_001", True))
        state = json.loads((r.run / "gate_state.json").read_text())
        self.assertEqual(state["verdicts"]["rl_002@rl_001#" + rl_loop.GATE_SCHEMA]["rollback"]["verdict"], "worse")
        # a restart with the same configuration stops at once; another configuration goes on
        self.assertEqual(r.main("--max-rejections", "1", "--collapse-games", "0"), rl_loop.EXIT_GATE_EXHAUSTED)
        (r.run / "YIELD").write_text("x")
        self.assertEqual(r.main("--max-rejections", "1", "--collapse-games", "0", "--tag", "X2"),
                         rl_loop.EXIT_YIELD)

    def test_champion_rounds_become_usable_after_a_fallback(self):
        r = self.r
        r.journal("rl_001", "rl_002")
        (r.run / "champion.json").write_text(json.dumps({"name": "rl_001"}))
        r.round("lane0_20261017T010000Z", "rl_001", rate=0.85, n=300)
        r.round("lane1_20261017T020000Z", "rl_002", rate=0.5, n=200, seed=3)
        r.main("--collapse-games", "0", "--recent-networks", "1", "--iterations", "5")
        # rl_002 allowed first (its round encoded), then the gate went back to rl_001: its round encoded too
        self.assertIn("lane1_20261017T020000Z", r.encoded)
        self.assertIn("lane0_20261017T010000Z", r.encoded)
        self.assertTrue(r.learned)

    def test_disk_low_pauses_the_intake(self):
        r = self.r
        r.journal("rl_001")
        r.round("lane0_20261017T010000Z", "rl_001")
        pulls = []
        with mock.patch.object(rl_loop, "free_gb", lambda p: 5.0), \
                mock.patch.object(server_sync, "pull", lambda *a, **k: pulls.append(1)):
            code = r.main("--min-free-gb", "30", "--gate-games", "0", "--iterations", "2")
        self.assertEqual(code, rl_loop.EXIT_DONE)            # waited (STOP from the sleep), no crash
        self.assertEqual((pulls, r.encoded), ([], []))

    def test_a_torn_journal_line_is_skipped(self):
        r = self.r
        r.journal("rl_001")
        with open(r.run / "rl.jsonl", "a") as f:
            f.write('{"iteration": 2, "checkp')
        (r.run / "YIELD").write_text("x")
        self.assertEqual(r.main(), rl_loop.EXIT_YIELD)
        self.assertEqual(r.pushed, ["rl_001"])


class ArchiveTest(unittest.TestCase):
    def test_age_from_arrival_and_hold(self):
        with tempfile.TemporaryDirectory() as d:
            rounds, cache, arch = Path(d) / "rounds", Path(d) / "cache", Path(d) / "arch"
            cache.mkdir()
            for name in ("lane0_A", "lane0_B", "lane0_C"):
                (rounds / name / "raw").mkdir(parents=True)
                (rounds / name / "DONE").write_text("x")
                old = time.time() - 5 * 86400                # the server's folder time: days old
                os.utime(rounds / name, (old, old))
            (rounds / "lane0_A" / "ARRIVED").write_text("x")  # arrived now: young
            for name in ("lane0_B", "lane0_C"):
                (rounds / name / "ARRIVED").write_text("x")
                t = time.time() - 30 * 3600
                os.utime(rounds / name / "ARRIVED", (t, t))
            moved = rl_loop.archive_old(rounds, cache, arch, 24, 72, hold=lambda r: r.name == "lane0_C")
            self.assertEqual(moved, (1, 0))
            self.assertTrue((rounds / "lane0_A" / "raw").is_dir())
            self.assertTrue((rounds / "lane0_B" / "RAW_ARCHIVED").exists())
            self.assertTrue((rounds / "lane0_C" / "raw").is_dir())                # held: not encoded yet
            self.assertEqual(rl_loop.archive_old(rounds, cache, arch, 24, 72, hold=lambda r: True,
                                                 hold_hours=10), (1, 0))          # but not past hold_hours

    def test_success_archive_follows_archived_recordings(self):
        with tempfile.TemporaryDirectory() as d:
            rounds = Path(d) / "rounds"
            r = rounds / "lane0_20261017T010000Z"
            (r / "eval").mkdir(parents=True)
            (r / "round.json").write_text(json.dumps({"checkpoint_sha256": "aa"}))
            (r / "DONE").write_text("x")
            (r / "eval" / "episodes.jsonl").write_text(json.dumps(
                {"ai": "m28aicheat@0.6", "outcome": "victory", "seed": 7, "map": "SCMP_012"}))
            moved = Path(d) / "arch" / r.name / "raw"
            moved.mkdir(parents=True)
            (moved / "0000001_m28aicheat@0.6_7.jsonl.gz").write_bytes(b"x")
            (r / "RAW_ARCHIVED").write_text(str(moved))
            picks = rl_loop.success_archive(rounds, {"aa"}, 0.6, 10)
            self.assertEqual(picks, [[str(moved.resolve()), "0000001_m28aicheat@0.6_7.jsonl.gz"]])


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.rounds, self.arch = self.d / "rounds", self.d / "arch"
        (self.arch / "cache").mkdir(parents=True)
        (self.arch / "cache" / "a.npz").write_bytes(b"x" * 100)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def round(self, name, sha, archived=True, size=1000, age_h=48):
        r = self.rounds / name
        r.mkdir(parents=True)
        (r / "round.json").write_text(json.dumps({"checkpoint_sha256": sha}))
        (r / "DONE").write_text("x")
        (r / "ARRIVED").write_text("x")
        t = time.time() - age_h * 3600
        os.utime(r / "ARRIVED", (t, t))
        raw = self.arch / "rounds_raw" / name / "raw" if archived else r / "raw"
        raw.mkdir(parents=True)
        (raw / "g.jsonl.gz").write_bytes(b"x" * size)
        if archived:
            (r / "RAW_ARCHIVED").write_text(str(raw))
        return raw

    def test_policy(self):
        old = self.round("lane0_20261010T000000Z", "lin_old")
        kept = self.round("lane0_20261011T000000Z", "lin_new")
        champ = self.round("lane0_20261009T000000Z", "champ")
        aband = self.round("lane0_20261012T000000Z", "abandoned", archived=False)
        unknown = self.round("lane0_20261013T000000Z", "who")
        sil = self.round("lane0_20261008T000000Z", "lin_old")
        young = self.round("lane0_20261014T000000Z", "abandoned2", archived=False, age_h=1)
        known = {"lin_old", "lin_new", "champ", "abandoned", "abandoned2"}
        lineage = {"lin_old", "lin_new", "champ"}
        # within the budget: archived cache and the abandoned network's games only
        plan = storage.plan(self.rounds, self.arch, 1.0, lineage, {"lin_new", "champ"}, "champ", known,
                            {"lane0_20261008T000000Z"})
        self.assertEqual({p for p, _, _ in plan}, {self.arch / "cache" / "a.npz", aband})
        # over a tiny budget: the lineage's archived games too, older networks first; never the champion's, the
        # self-imitation round's, an unknown network's or a round that just arrived
        plan = storage.plan(self.rounds, self.arch, 1e-9, lineage, {"lin_new", "champ"}, "champ", known,
                            {"lane0_20261008T000000Z"})
        paths = [p for p, _, _ in plan]
        self.assertEqual(paths[2:], [old, kept])
        for p in (champ, unknown, sil, young):
            self.assertNotIn(p, paths)
        freed = storage.apply(plan, self.rounds)
        self.assertGreater(freed, 0)
        self.assertFalse(old.exists())
        self.assertTrue((self.rounds / "lane0_20261010T000000Z" / "RAW_DELETED").exists())
        self.assertTrue(champ.exists() and sil.exists())


class PullTest(unittest.TestCase):
    """server_sync.pull with ssh/tar replaced: an earlier incomplete copy is set aside, never a crash."""

    def tar_of(self, src: Path, name: str) -> bytes:
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as t:
            t.add(src / name, arcname=name)
        return buf.getvalue()

    def test_incomplete_copy_set_aside_and_arrived(self):
        import io
        with tempfile.TemporaryDirectory() as d:
            src, dest = Path(d) / "src", Path(d) / "dest"
            r = src / "lane0_X"
            (r / "raw").mkdir(parents=True)
            (r / "raw" / "1.jsonl.gz").write_bytes(b"x")
            (r / "raw" / "blueprints.json.gz").write_bytes(b"x")
            (r / "round.json").write_text(json.dumps({"games": 1}))
            (r / "DONE").write_text("x")
            (dest / "lane0_X" / "raw").mkdir(parents=True)       # an earlier incomplete copy, without DONE
            data = self.tar_of(src, "lane0_X")

            class Proc:
                def __init__(self, *a, **k):
                    self.stdout = io.BytesIO(data)
                    self.returncode = 0

                def wait(self):
                    return 0

                def kill(self):
                    pass
            with mock.patch.object(server_sync, "remote_rounds", lambda h, home: ["lane0_X"]), \
                    mock.patch.object(server_sync.subprocess, "Popen", Proc), \
                    mock.patch.object(server_sync, "ssh", lambda *a, **k: None):
                self.assertEqual(no_network.REAL["pull"]("h", "~", dest, keep=False), 1)
            self.assertTrue((dest / "lane0_X" / "DONE").exists())
            self.assertTrue((dest / "lane0_X" / "ARRIVED").exists())
            self.assertTrue((dest / "_incomplete" / "lane0_X").exists())

    def test_push_raises_instead_of_exiting(self):
        with self.assertRaises(FileNotFoundError):
            no_network.REAL["push"]("h", "~", Path("no_such.pt"))

    def test_the_guard_blocks_real_ssh(self):
        with self.assertRaises(no_network.NetworkBlocked):
            server_sync.set_config("my-server", "~/fa/player", {"maps": "x"})
        with self.assertRaises(no_network.NetworkBlocked):
            no_network.REAL["ssh"]("my-server", "true")          # the real function: subprocess refuses ssh
        with self.assertRaises(no_network.NetworkBlocked):
            subprocess.run(["scp", "a", "my-server:b"])


class RetryTest(unittest.TestCase):
    def test_gives_up_without_raising(self):
        calls = []

        def bad():
            calls.append(1)
            raise subprocess.CalledProcessError(255, "ssh")
        with mock.patch.object(rl_loop.time, "sleep", lambda s: None):
            self.assertEqual(rl_loop.retry(bad, waits=(1, 1)), (False, None))
        self.assertEqual(len(calls), 3)


class ShaCacheTest(unittest.TestCase):
    def test_hashed_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rl_001.pt"
            p.write_bytes(b"abc")
            with mock.patch.object(rl_loop, "sha256", wraps=rl_loop.sha256) as h:
                rl_loop.sha256_cached(p)
                rl_loop.sha256_cached(p)
                self.assertEqual(h.call_count, 1)
                p.write_bytes(b"abcd")                    # changed: hashed again
                rl_loop.sha256_cached(p)
                self.assertEqual(h.call_count, 2)


class MixTest(unittest.TestCase):
    def test_small_map_mix_without_full_m28(self):
        self.assertEqual(rl_loop.opponent_mix(0.6, "rush", 3.0, full_weight=0, stretch_weight=2),
                         "rush,m28aicheat@0.6:3,m28aicheat@0.75:2")


class PlayerTest(unittest.TestCase):
    def test_seed_bases_never_collide(self):
        bases = {player.seed_base_for(m, lane) for m in range(1000, 1003) for lane in (0, 1, 2, 10, 11)}
        self.assertEqual(len(bases), 15)
        self.assertNotEqual(player.seed_base_for(5, 10), player.seed_base_for(6, 0))
        self.assertLess(player.seed_base_for(19_999, 99) + 999, 2 ** 31)

    def test_lane_network(self):
        self.assertEqual(player.lane_network({}, 0), "current.pt")
        self.assertEqual(player.lane_network({"network": "champion.pt"}, 10), "champion.pt")
        self.assertEqual(player.lane_network({"lane_networks": {"10": "champion.pt"}}, 10), "champion.pt")
        self.assertEqual(player.lane_network({"lane_networks": {"10": "champion.pt"}}, 0), "current.pt")
        self.assertEqual(player.lane_network({"network": "..\\x.pt"}, 0), "current.pt")


if __name__ == "__main__":
    unittest.main()
