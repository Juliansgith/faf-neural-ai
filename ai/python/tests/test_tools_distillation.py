"""Tiny CPU/mock-only tests. No networks, training, datasets or game processes are constructed."""
import contextlib
import io
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train import night_tools
from fa_train.tools_trainer import ToolsConfig, ToolsTrainer, Trainer
from fa_train.worker import worker_main


def trainer(config=None):
    t = object.__new__(ToolsTrainer)  # base constructor allocates networks; never call it in these tests
    t.cfg = config or ToolsConfig(run_dir="unused")
    t.update = 0
    t.device = torch.device("cpu")
    t.net, t.acting, t.optimizer = Mock(), Mock(), Mock()
    t.net.state_dict.return_value = {"mock_weights": 1}
    t.optimizer.state_dict.return_value = {"mock_optimizer": 1}
    t.episodes, t.stage, t.stages = 0, 0, [{"name": "test"}]
    t.rms = SimpleNamespace(mean=0, var=1, count=1)
    t.entropy_ref, t.thread = None, None
    t._log_event = Mock()
    return t


class Pipe:
    def __init__(self, messages):
        self.messages, self.sent = iter(messages), []

    def recv(self):
        return next(self.messages)

    def send(self, message):
        self.sent.append(message)


class DistillationTest(unittest.TestCase):
    def test_defaults_schedule_and_disabled_schedule(self):
        cfg = ToolsConfig(run_dir="unused", planned_updates=100)
        self.assertEqual((cfg.distill_teacher, cfg.distill_start, cfg.distill_fraction, cfg.distill_end),
                         ("v3", 1.0, 0.3, 0.1))
        t = trainer(cfg)
        self.assertEqual(t.distill_beta(), 1)
        t.update = 15
        self.assertAlmostEqual(t.distill_beta(), .55)
        t.update = 30
        self.assertAlmostEqual(t.distill_beta(), .1)
        cfg.distill_teacher = "none"
        for t.update in (0, 15, 30, 100):
            self.assertEqual(t.distill_beta(), 0)

    def test_invalid_config_and_mutation_fail_before_base_constructor(self):
        for field, value in (("distill_teacher", "checkpoint"), ("distill_start", -1),
                             ("distill_end", float("nan")), ("distill_fraction", float("inf"))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ToolsConfig(run_dir="unused", **{field: value})
            cfg = ToolsConfig(run_dir="unused")
            setattr(cfg, field, value)
            with patch.object(Trainer, "__init__") as base, self.assertRaises(ValueError):
                ToolsTrainer(cfg, device="cpu")
            base.assert_not_called()
        for fraction in (0, 2):
            ToolsConfig(run_dir="unused", distill_fraction=fraction)

    def test_episode_selector_and_no_stale_requests(self):
        for source in ("v3", "none"):
            t = trainer(ToolsConfig(run_dir="unused", distill_teacher=source))
            t.mode = "ppo"
            with patch.object(Trainer, "_episode", return_value={"teacher_labels": True}):
                ep = t._episode("ppo")
            self.assertEqual(ep["distill_teacher"], source)
            self.assertEqual(bool(ep.get("teacher_labels")), source == "v3")
            with patch.object(Trainer, "_episode", return_value={"eval": True, "teacher_labels": True}):
                self.assertNotIn("teacher_labels", t._episode("eval"))

    def test_checkpoint_teacher_api_rejects_without_path_access(self):
        path = Mock()
        with patch("fa_train.tools_trainer.torch.load") as load, self.assertRaisesRegex(
                NotImplementedError, "unsupported checkpoint teacher"):
            trainer().set_teacher(path)
        load.assert_not_called()
        self.assertEqual(path.mock_calls, [])

    def test_night_cli_config_actual_main_with_fake_trainer(self):
        args = night_tools.parse_args(["--run", "unused", "--stop-at", "07:00"])
        cfg = night_tools.make_config(args, "unused")
        self.assertEqual((cfg.distill_teacher, cfg.distill_start, cfg.distill_fraction, cfg.distill_end),
                         ("v3", 1, 1 / 3, 0))
        with tempfile.TemporaryDirectory() as root:
            fake = Mock()
            fake.restarts = 0
            with patch.object(night_tools, "ToolsTrainer", return_value=fake) as constructor, \
                 patch.object(night_tools, "parse_stop", return_value=datetime.now()), \
                 patch.object(night_tools, "keep_awake"), patch.object(night_tools, "raise_learner_priority"), \
                 patch.object(night_tools, "write_report"), \
                 patch.object(night_tools, "open", mock_open(), create=True), \
                 patch.object(night_tools, "Tee", side_effect=lambda stream, log: stream):
                original = sys.stdout, sys.stderr
                try:
                    night_tools.main(["--run", root, "--stop-at", "07:00", "--clone", "unread.pt",
                                      "--distill-teacher", "none", "--distill-start", "2",
                                      "--distill-fraction", "0", "--distill-end", "3"])
                finally:
                    sys.stdout, sys.stderr = original
            cfg = constructor.call_args.args[0]
            self.assertEqual((cfg.distill_teacher, cfg.distill_start, cfg.distill_fraction, cfg.distill_end),
                             ("none", 2, 0, 3))
            fake.run.assert_not_called()
            fake.load.assert_not_called()

    def test_cli_invalid_before_run_mutation(self):
        for option, value in (("--distill-teacher", "checkpoint"), ("--distill-start", "nan"),
                              ("--distill-start", "-1"), ("--distill-end", "inf"),
                              ("--distill-end", "-1"), ("--distill-fraction", "-1"),
                              ("--distill-fraction", "nan")):
            with self.subTest(option=option, value=value), \
                 patch.object(Path, "mkdir") as mkdir, patch.object(Path, "unlink") as unlink, \
                 patch.object(night_tools, "ToolsTrainer") as constructor, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError, SystemExit)):
                    night_tools.main(["--run", "unused", "--stop-at", "07:00", option, value])
                mkdir.assert_not_called()
                unlink.assert_not_called()
                constructor.assert_not_called()

    def test_worker_real_flow_sources_legacy_and_invalid(self):
        for source, labels, expected in (("v3", True, True), ("none", True, False),
                                         (None, True, True), ("v3", False, False),
                                         ("unknown", True, False)):
            with self.subTest(source=source, labels=labels), tempfile.TemporaryDirectory() as root:
                observation = {"tokens": np.zeros((1, 1)), "map": np.zeros((1, 1))}
                env = Mock()
                env.fa = None
                env.reset.return_value = (observation, {})
                env.step_tools.return_value = (observation, {}, False, {})
                env.teacher_commands.return_value = ["mock_label"]
                ep = {"kind": "tools", "teacher_labels": labels, "uid": 9}
                if source is not None:
                    ep["distill_teacher"] = source
                messages = [("reset", ep)]
                if source != "unknown":
                    messages.append(("act", []))
                messages.append(("stop",))
                pipe = Pipe(messages)
                with patch("fa_env.tools_env.ToolsEnv", return_value=env) as constructor, \
                     patch("fa_env.teacher_v3.TeacherV3") as teacher:
                    worker_main(0, pipe, root)
                if source == "unknown":
                    constructor.assert_not_called()
                    teacher.assert_not_called()
                    self.assertEqual(pipe.sent[0][0], "error")
                    self.assertIn("unsupported distill_teacher", pipe.sent[0][1])
                else:
                    constructor.assert_called_once()
                    self.assertEqual(teacher.call_count, int(expected))
                    self.assertEqual(env.teacher_commands.call_count, 2 if expected else 0)
                    self.assertEqual(pipe.sent[0][4]["distill_teacher"], source or "v3")
                    self.assertEqual(pipe.sent[0][4]["teacher_labels"], expected)
                    self.assertEqual("teacher" in pipe.sent[1][4], expected)

    def test_worker_preserves_requested_decision_cadence(self):
        for requested, expected in ((None, 20), (2, 2)):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as root:
                env = Mock()
                env.fa = None
                env.reset.return_value = ({"tokens": np.zeros((1, 1)), "map": np.zeros((1, 1))}, {})
                episode = {"kind": "tools", "distill_teacher": "none", "teacher_labels": False}
                if requested is not None:
                    episode["decision_ticks"] = requested
                pipe = Pipe([("reset", episode), ("stop",)])
                with patch("fa_env.tools_env.ToolsEnv", return_value=env) as constructor:
                    worker_main(0, pipe, root)
                self.assertEqual(pipe.sent[0][0], "obs")
                self.assertEqual(constructor.call_args.kwargs["decision_ticks"], expected)

    def test_checkpoint_metadata_uses_resumed_update(self):
        for end in (.1, 0):
            with self.subTest(end=end):
                t = trainer(ToolsConfig(run_dir="unused", planned_updates=100, distill_end=end))
                state = {"model": {"mock_weights": 2}, "update": 100, "config": {"distill_teacher": "none"}}
                output = io.StringIO()
                with patch("fa_train.tools_trainer.torch.load", return_value=state), \
                     contextlib.redirect_stdout(output):
                    t.load("unread.pt")
                self.assertEqual(t.update, 100)
                beta = t.distill_beta()
                self.assertAlmostEqual(beta, end)
                self.assertEqual(t._log_event.call_args.args[0]["active"]["effective_beta"], beta)
                self.assertIn(f"'effective_beta': {beta}", output.getvalue())
                self.assertIn("WARNING:", output.getvalue())

    def test_checkpoint_save_and_load_authority_provenance(self):
        t = trainer(ToolsConfig(run_dir="unused", distill_teacher="none"))
        with tempfile.TemporaryDirectory() as root:
            t.run_dir = Path(root)
            with patch("fa_train.tools_trainer.torch.save") as save, patch.object(Path, "replace"):
                t.save("mock.pt")
            state = save.call_args.args[0]
            self.assertEqual(state["config"]["distill_teacher"], "none")
            self.assertEqual(state["distillation"]["effective_beta"], 0)
            self.assertEqual(state["distillation"]["effective_distill_start"], 0)
        for config, provenance in (({"distill_teacher": "none", "distill_start": 9}, "explicit"),
                                    ({"distill_start": 1}, "legacy_v3"),
                                    ({"distill_teacher": "unknown"}, "explicit"), (None, "unavailable")):
            t = trainer()
            active = t.cfg
            state = {"model": {"mock_weights": 2}, "ret_std": 3}
            if config is not None:
                state["config"] = config
            output = io.StringIO()
            with patch("fa_train.tools_trainer.torch.load", return_value=state), \
                 contextlib.redirect_stdout(output):
                t.load("unread.pt", optimizer=False)
            self.assertIs(t.cfg, active)
            self.assertEqual(t.cfg.distill_teacher, "v3")
            event = t._log_event.call_args.args[0]
            self.assertEqual(event["saved_provenance"], provenance)
            self.assertEqual(event["saved_config"], config)
            self.assertEqual(event["active_config"]["distill_teacher"], "v3")
            self.assertEqual(t.rms.var, 9)
            t.net.load_state_dict.assert_called_once_with(state["model"])
            if config and config.get("distill_teacher") in ("none", "unknown"):
                self.assertIn("WARNING:", output.getvalue())
                self.assertIn("distill_teacher", event["mismatches"])
            if provenance == "legacy_v3":
                self.assertEqual(event["saved"]["distill_teacher"], "v3")
                self.assertIn("legacy v3", output.getvalue())
            if config is None:
                self.assertIsNone(event["saved"])
                self.assertIn("provenance unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
