"""The autopilot (fa_train.autopilot) and its watchdog: command building, the experiment verdicts, the queue, the
state machine on a fake run (dry: nothing is launched, killed or deleted), rl_loop's exit codes, STOP ownership,
user commands. CPU only, temp folders; the process list is replaced."""
try:                                   # first: no test may reach the server or start a real process
    import no_network  # noqa: F401
except ImportError:
    from tests import no_network  # noqa: F401
import json
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from fa_train import autopilot as ap
from fa_train import autopilot_watchdog as wd
from fa_train import server_sync

RULES = ap.DEFAULTS["rules"]


def cells(rate: float, n: int, conds=(("adaptive", 1, 2), ("rush", 2, 3), ("m28aicheat@0.6", 3, 1)), seed=0):
    rng = random.Random(seed)
    out = {}
    for c in conds:
        w = sum(rng.random() < rate for _ in range(n // len(conds)))
        out[c] = [w, n // len(conds)]
    return out


def scan_of(**nets):
    return {k: {"cells": v, "large": [0, 0], "self": [0, 0], "games": 0, "first": "20261017T000000Z",
                "last": "20261017T000000Z"} for k, v in nets.items()}


class Args(unittest.TestCase):
    def test_merge_and_remove(self):
        self.assertEqual(ap.merged_args({"--a": "1", "--b": None, "--c": "x"}, {"--a": "2", "--c": False, "--d": None}),
                         ["--a", "2", "--b", "--d"])

    def test_loop_command(self):
        s = ap.load_settings(Path("nonexistent.json"))
        exp = {"name": "X2", "loop": {"--reference": "champion", "--reference-lag": "0", "--league": "CHAMPION,rl_077"},
               "learn": {"--lr": "1e-5", "--upgo": "0", "--opponent-critic": False}}
        cmd = ap.loop_command(s, exp, "rl_140")
        joined = " ".join(cmd)
        self.assertIn("--champion rl_140", joined)
        self.assertIn("--tag X2 --start-from champion", joined)
        self.assertIn(str(Path(s["run"]) / "checkpoints" / "rl_140.pt"), cmd)
        self.assertIn("--league rl_140,rl_077", joined)
        self.assertIn("--control-lane", cmd)                   # the local lane plays the champion
        self.assertIn("--storage-budget-gb", cmd)
        learn = cmd[cmd.index("--learn-args") + 1]
        self.assertIn("--lr 1e-5", learn)
        self.assertNotIn("--lr 2e-5", learn)
        self.assertNotIn("--opponent-critic", learn)
        self.assertIn("--upgo 0", learn)
        self.assertNotIn("--start-from", ap.loop_command(s, exp, "rl_140", first=False))   # a crash restart resumes
        self.assertIn("--set-champion", ap.loop_command(s, exp, "rl_140", set_champion="rl_150"))
        no_ctl = ap.loop_command(s, {**exp, "loop": {"--control-lane": False}}, "rl_140")
        self.assertNotIn("--control-lane", no_ctl)

    def test_default_queue(self):
        names = [e["name"] for e in ap.DEFAULT_QUEUE]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([n.split("-")[0] for n in names], [f"X{i}" for i in range(1, 8)])
        x2 = ap.expand(ap.DEFAULT_QUEUE[1], ap.DEFAULT_QUEUE)    # X1's changes under its own
        self.assertEqual(x2["learn"]["--capacity-clip"], "1")
        self.assertEqual(x2["learn"]["--lam"], "0.98")
        self.assertEqual(x2["player"]["maps"], ap.SMALL_MAPS)
        self.assertNotIn("PLACEHOLDER", json.dumps(ap.expand(ap.DEFAULT_QUEUE[0], ap.DEFAULT_QUEUE)))


class Judge(unittest.TestCase):
    def test_k_fallbacks_fail(self):
        v = ap.judge(RULES, scan_of(champ=cells(0.75, 1500)), "champ", [], 2, 5, 5)
        self.assertEqual(v.kind, "FAIL")

    def test_pooled_fail(self):
        s = scan_of(champ=cells(0.75, 1500), a=cells(0.55, 300, seed=1), b=cells(0.55, 300, seed=2))
        v = ap.judge(RULES, s, "champ", ["a", "b"], 0, 6, 6)
        self.assertEqual(v.kind, "FAIL", v.why)

    def test_equal_continues_early(self):
        s = scan_of(champ=cells(0.75, 1500), a=cells(0.75, 350, seed=3))
        self.assertEqual(ap.judge(RULES, s, "champ", ["a"], 0, 10, 10).kind, "continue")

    def test_stall_after_36h(self):
        s = scan_of(champ=cells(0.75, 1500), a=cells(0.74, 700, seed=4))
        self.assertEqual(ap.judge(RULES, s, "champ", ["a"], 1, 37, 37).kind, "STALL")

    def test_promising_window_extends_to_max(self):
        s = scan_of(champ=cells(0.70, 1500, seed=5), a=cells(0.78, 700, seed=6))
        self.assertEqual(ap.judge(RULES, s, "champ", ["a"], 0, 40, 40).kind, "continue")
        self.assertEqual(ap.judge(RULES, s, "champ", ["a"], 0, 61, 25).kind, "MAX")
        self.assertEqual(ap.judge(RULES, s, "champ", ["a"], 0, 61, 10).kind, "continue")   # promoted 10 h ago

    def test_user_extension(self):
        s = scan_of(champ=cells(0.75, 1500), a=cells(0.74, 700, seed=4))
        self.assertEqual(ap.judge(RULES, s, "champ", ["a"], 1, 37, 37, extra_hours=24).kind, "continue")


class Queue(unittest.TestCase):
    Q = [{"name": "A"}, {"name": "B"}, {"name": "C"}]

    def test_order(self):
        self.assertEqual(ap.next_experiment(self.Q, {}, "hold")["name"], "A")
        self.assertEqual(ap.next_experiment(self.Q, {"A": {"verdict": "FAIL"}}, "hold")["name"], "B")
        done = {"A": {"verdict": "STALL", "score": -1, "requeue": True}, "B": {"verdict": "FAIL"},
                "C": {"verdict": "STALL", "score": 1.5}}
        nxt = ap.next_experiment(self.Q, done, "best_stalled")
        self.assertEqual((nxt["name"], nxt.get("_requeue")), ("A", True))
        done["A"]["requeued_done"] = True
        nxt = ap.next_experiment(self.Q, done, "best_stalled")
        self.assertEqual((nxt["name"], nxt.get("_rerun")), ("C", True))
        done["C"]["rerun"] = True
        self.assertIsNone(ap.next_experiment(self.Q, done, "best_stalled"))
        self.assertIsNone(ap.next_experiment(self.Q, {"A": {}, "B": {}, "C": {}}, "hold"))


class Journal(unittest.TestCase):
    ROWS = [{"iteration": 1, "start": "start of X", "checkpoint": "c/rl_089.pt", "fallback": True},
            {"iteration": 2, "checkpoint": "c/rl_002.pt", "learn_rl": [{"epoch": 1, "ent": 0.58}]},
            {"iteration": 3, "gate": "worse", "checkpoint": "c/rl_089.pt", "fallback": True},
            {"iteration": 4, "exit": 1},
            {"iteration": 5, "warning": "regression", "what": "x"},
            {"iteration": 6, "checkpoint": "c/rl_005.pt", "learn_rl": [{"epoch": 1, "ent": 0.33}]}]

    def test_networks_and_fallbacks(self):
        self.assertEqual(ap.experiment_networks(self.ROWS), ["rl_002", "rl_005"])
        self.assertEqual(ap.fallbacks(self.ROWS), 1)          # the start row and the warning are no fallbacks
        self.assertEqual(ap.fallbacks(self.ROWS, 3), 0)

    def test_entropy_alarm(self):
        self.assertIn("rl_005", ap.entropy_alarm(self.ROWS, 0.58, 0.15))
        self.assertIsNone(ap.entropy_alarm(self.ROWS[:2], 0.58, 0.15))


class Rounds(unittest.TestCase):
    def test_scan_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d) / "lane0_20261017T010000Z"
            (r / "eval").mkdir(parents=True)
            (r / "round.json").write_text(json.dumps({"checkpoint_sha256": "aa"}))
            (r / "DONE").write_text("x")
            eps = [{"ai": "rush", "faction": 1, "ai_faction": 2, "outcome": "victory", "map": "SCMP_016"},
                   {"ai": "rush", "faction": 1, "ai_faction": 2, "outcome": "defeat", "map": "SCMP_016"},
                   {"ai": "tech", "faction": 1, "ai_faction": 2, "outcome": "victory", "map": "SCMP_003"},
                   {"ai": "self@rl_077", "faction": 1, "ai_faction": 2, "outcome": "victory", "map": "SCMP_016"}]
            (r / "eval" / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps) + "\n")
            old = Path(d) / "lane0_20261001T010000Z"                     # before `since`
            shutil.copytree(r, old)
            out = ap.scan_rounds(Path(d), {"aa": "rl_200"}, "20261013T2310Z")
            self.assertEqual(out["rl_200"]["cells"],
                             {("rush", 1, 2, (None, (), None, None, None, None, None, ())): [1, 2]})
            self.assertEqual(out["rl_200"]["games"], 4)
            self.assertEqual(out["rl_200"]["self"], [1, 1])
            self.assertEqual(out["rl_200"]["large"], [1, 1])                   # SCMP_003 is 20 km


class StateMachine(unittest.TestCase):
    """Ticks of the Autopilot on a fake run, dry: start X1, two gate fallbacks -> FAIL -> X2 starts."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        run = self.d / "run"
        (run / "checkpoints").mkdir(parents=True)
        (run / "rounds").mkdir()
        for n in ("rl_089", "rl_127", "rl_128"):
            (run / "checkpoints" / f"{n}.pt").write_bytes(n.encode())
        (self.d / "init.pt").write_bytes(b"init")
        (run / "champion.json").write_text(json.dumps({"name": "rl_089"}))
        (run / "rl.jsonl").write_text(json.dumps({"iteration": 89, "checkpoint": str(run / "checkpoints/rl_089.pt"),
                                                  "m28_level": 0.6, "distill": 0.05,
                                                  "learn_rl": [{"epoch": 4, "ent": 0.58}]}) + "\n")
        cfg = {"run": str(run), "init": str(self.d / "init.pt"), "log": str(self.d / "rl2.log"),
               "local_home": None, "gpu_watchdog_ps1": None, "state_dir": str(self.d / "state"),
               "disk": {"archive": str(self.d)}, "report": {"every_hours": 1000}}
        (self.d / "cfg.json").write_text(json.dumps(cfg))
        self.run = run
        # nothing in these tests may reach the server or start a process, dry or not: every way out is replaced
        self.server_calls = []
        self.guards = [mock.patch.object(server_sync, name, self.recorder(name))
                       for name in ("ssh", "set_config", "push", "pull", "remote_rounds")]
        self.guards += [mock.patch.object(ap, "launch", lambda *a, **k: self.server_calls.append("launch") or 4242),
                        mock.patch.object(ap, "kill_tree", lambda pid: self.server_calls.append(f"kill {pid}")),
                        mock.patch.object(ap, "gpu_driver_errors", lambda h: 0),
                        mock.patch.object(ap.subprocess, "run", self.recorder("subprocess.run")),
                        mock.patch.object(ap.subprocess, "Popen", self.recorder("subprocess.Popen")),
                        mock.patch.object(ap.urllib.request, "urlopen", self.recorder("urlopen"))]
        for g in self.guards:
            g.start()

    def recorder(self, name):
        def call(*a, **k):
            self.server_calls.append(name)
            return {}
        return call

    def tearDown(self):
        for g in self.guards:
            g.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def tick(self, pilot, procs=()):
        with mock.patch.object(ap, "processes", lambda: list(procs)), mock.patch.object(ap, "commit_gb", lambda: 0.0):
            pilot.tick()

    def loop(self):
        return [{"pid": 1, "ppid": 0, "name": "python.exe", "cmd": f"python -m fa_train.rl_loop --run {self.run}"}]

    def exit_file(self, code, why=""):
        (self.run / "rl_loop_exit.json").write_text(json.dumps(
            {"code": code, "why": why, "time": datetime.now().isoformat(timespec="seconds")}))

    def test_fail_then_next(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        self.tick(pilot)
        self.assertEqual(pilot.state["experiment"]["name"], "X1-clean")
        self.assertEqual(pilot.state["champion_level"], 0.6)
        self.assertEqual(pilot.state["champion_entropy"], 0.58)
        rows = [{"iteration": 90, "start": "x", "experiment": "X1-clean",
                 "checkpoint": str(self.run / "checkpoints/rl_089.pt"), "fallback": True},
                {"iteration": 91, "checkpoint": str(self.run / "checkpoints/rl_127.pt")},
                {"iteration": 92, "gate": "worse", "checkpoint": str(self.run / "checkpoints/rl_089.pt"), "fallback": True},
                {"iteration": 93, "checkpoint": str(self.run / "checkpoints/rl_128.pt")},
                {"iteration": 94, "gate": "worse", "checkpoint": str(self.run / "checkpoints/rl_089.pt"), "fallback": True}]
        with open(self.run / "rl.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self.tick(pilot, self.loop())                   # verdict FAIL; the loop runs: YIELD (dry: not written)
        self.assertEqual(pilot.state["verdict"]["kind"], "FAIL")
        self.assertIsNotNone(pilot.state["experiment"])
        self.tick(pilot)                                # the loop is gone: the experiment is recorded
        self.assertEqual(pilot.state["results"]["X1-clean"]["verdict"], "FAIL")
        self.tick(pilot)                                # the next one starts
        self.assertEqual(pilot.state["experiment"]["name"], "X2-terminal")
        launches = [json.loads(x) for x in (self.d / "state" / "events.jsonl").read_text().splitlines()
                    if '"dry_launch"' in x]
        self.assertIn("--lam", launches[-1]["cmd"][-1])         # X2 = X1 + its own learn flags
        self.assertIn("--capacity-clip", launches[-1]["cmd"][-1])
        self.assertTrue((self.d / "state" / "report" / "report.html").exists())

    def test_placeholder_entries_are_skipped(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        (self.d / "state").mkdir(exist_ok=True)
        (self.d / "state" / "queue.json").write_text(json.dumps([{"name": "P", "learn": {"--x": "PLACEHOLDER"}},
                                                                 {"name": "Q"}]))
        self.tick(pilot)
        self.assertEqual(pilot.state["results"]["P"]["verdict"], "INVALID")
        self.tick(pilot)
        self.assertEqual(pilot.state["experiment"]["name"], "Q")

    def test_no_process_list_no_action(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        with mock.patch.object(ap, "processes", lambda: None):
            pilot.tick()
        self.assertIsNone(pilot.state.get("experiment"))   # an unknown process table never starts a second loop

    def test_foreign_stop_holds_and_stays(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=False)
        (self.run / "STOP").write_text("")
        with mock.patch.object(ap, "gpu_driver_errors", lambda h: 3):
            self.tick(pilot)
        self.assertEqual(pilot.state["mode"], "gpu_hold")
        self.assertTrue((self.run / "STOP").exists())
        self.assertIsNone(pilot.state.get("experiment"))   # nothing launched

    def test_own_stop_is_removed_before_a_relaunch(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=False)
        self.tick(pilot)                                    # X1 launched (the server config call is a recorder)
        self.assertIn("set_config", self.server_calls)
        pilot.write_stop("test")
        self.assertTrue((self.run / "STOP").exists())
        pilot.state["loop_started"] = time.time() - 600
        self.exit_file(0, "STOP")
        self.tick(pilot)                                    # our STOP: no hold; the loop is gone, counted
        self.assertNotEqual(pilot.state["mode"], "gpu_hold")
        self.assertEqual(len(pilot.state["crashes"]), 1)    # (a stall is no plan)
        self.assertTrue((self.run / "STOP").exists())       # the first backoff (1 min) still runs
        pilot.state["crashes"][-1]["t"] -= 120
        self.tick(pilot)                                    # relaunched: our STOP removed first
        self.assertFalse((self.run / "STOP").exists())
        self.assertEqual(self.server_calls.count("launch"), 2)

    def test_gate_exhausted_ends_the_experiment(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        self.tick(pilot)
        pilot.state["loop_started"] = time.time() - 600
        self.exit_file(ap.EXIT_GATE_EXHAUSTED, "3 fallbacks")
        self.tick(pilot)
        self.assertEqual(pilot.state["results"]["X1-clean"]["verdict"], "FAIL")
        self.assertIn("gate exhausted", pilot.state["results"]["X1-clean"]["why"])

    def test_crash_budget(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        self.tick(pilot)                                # X1 launched (dry)
        st = pilot.state
        for i in range(6):                              # six exits of the loop within the hour
            st["loop_started"] = time.time() - 600 - i
            self.exit_file(ap.EXIT_ERROR, "Traceback")
            with mock.patch.object(ap, "commit_gb", lambda: 0.0):
                pilot.restart([], [])
        self.assertEqual(st["results"]["X1-clean"]["verdict"], "ERROR")

    def test_orphan_learn_rl_blocks_relaunch(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        self.tick(pilot)
        pilot.state["loop_started"] = 0
        pilot.restart([], [{"pid": 7, "cmd": "python -m fa_train.learn_rl"}])
        self.assertEqual(pilot.state["mode"], "waiting_orphan_learn_rl")

    def test_independent_cpu_learner_does_not_block_next_experiment(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        cmd = subprocess.list2cmdline([
            "python", "-c", "import runpy;runpy.run_module('fa_train.learn_rl',run_name='__main__')",
            "--device", "cpu", "--out", str(self.d / "separate experiment" / "candidate.pt")])
        self.tick(pilot, [{"pid": 8, "ppid": 0, "cmd": cmd}])
        self.assertEqual(pilot.state["experiment"]["name"], "X1-clean")

    def test_gpu_or_same_run_cpu_learner_still_blocks_next_experiment(self):
        for device, output in (("cuda", self.d / "other" / "candidate.pt"),
                               ("cpu", self.run / "checkpoints" / "candidate.pt")):
            with self.subTest(device=device, output=output):
                pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
                cmd = subprocess.list2cmdline(["python", "-m", "fa_train.learn_rl",
                    "--device", device, "--out", str(output)])
                self.tick(pilot, [{"pid": 8, "ppid": 0, "cmd": cmd}])
                self.assertIsNone(pilot.state.get("experiment"))

    def test_silent_learn_rl_gets_a_stop_never_a_kill(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=False)
        log = self.d / "rl2.log"
        log.write_text("x")
        old = time.time() - 5 * 3600
        import os
        os.utime(log, (old, old))
        kills = []
        with mock.patch.object(ap, "kill_tree", lambda pid: kills.append(pid)):
            pilot.wedge(self.loop(), [{"pid": 9, "cmd": "python -m fa_train.learn_rl"}])
        self.assertTrue((self.run / "STOP").read_text().startswith("autopilot"))
        self.assertEqual(kills, [])

    def test_wedged_loop_stop_then_kill(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=False)
        log = self.d / "rl2.log"
        log.write_text("x")
        old = time.time() - 3 * 3600
        import os
        os.utime(log, (old, old))
        kills = []
        with mock.patch.object(ap, "kill_tree", lambda pid: kills.append(pid)):
            pilot.wedge(self.loop(), [])
            self.assertEqual(kills, [])                     # first our STOP
            self.assertTrue((self.run / "STOP").exists())
            pilot.state["wedge_stop_t"] -= 31 * 60
            pilot.wedge(self.loop(), [])
        self.assertEqual(kills, [1])                        # still silent, no learn_rl anywhere: CPU tree ended

    def test_champion_command_sets_it_at_the_next_launch(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        self.tick(pilot)
        (self.d / "state" / "override.json").write_text(json.dumps({"cmd": "champion", "name": "rl_127"}))
        self.tick(pilot)                                    # the experiment is skipped (no loop running) ...
        self.tick(pilot)                                    # ... and starts again with --set-champion
        launches = [json.loads(x) for x in (self.d / "state" / "events.jsonl").read_text().splitlines()
                    if '"dry_launch"' in x]
        self.assertIn("--set-champion", launches[-1]["cmd"])
        self.assertEqual(launches[-1]["cmd"][launches[-1]["cmd"].index("--set-champion") + 1], "rl_127")

    def test_bad_commands_are_ignored(self):
        pilot = ap.Autopilot(self.d / "cfg.json", dry_run=True)
        pilot.apply_command({"cmd": "champion", "name": "..\\..\\evil"}, "rl_089")
        pilot.apply_command({"cmd": "rm -rf"}, "rl_089")
        self.assertNotIn("set_champion", pilot.state)


class Watchdog(unittest.TestCase):
    def test_decide(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            now = time.time()
            (d / "heartbeat.json").write_text(json.dumps({"time": now - 60}))
            self.assertFalse(wd.decide(d, now, {"restarts": []})[0])          # fresh
            (d / "heartbeat.json").write_text(json.dumps({"time": now - 3600}))
            self.assertTrue(wd.decide(d, now, {"restarts": []})[0])           # stale
            self.assertFalse(wd.decide(d, now, {"restarts": [now - 60 * i for i in range(5)]})[0])   # budget
            (d / "WATCHDOG_OFF").write_text("")
            self.assertFalse(wd.decide(d, now, {"restarts": []})[0])


if __name__ == "__main__":
    unittest.main()
