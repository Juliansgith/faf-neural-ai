"""Evaluation game logs survive worker cleanup and include the final close flush."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train.worker import worker_main


class Pipe:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    def recv(self):
        return next(self.messages)

    def send(self, message):
        self.sent.append(message)


class WorkerLogsTest(unittest.TestCase):
    def run_worker(self, root, messages, fail=False):
        class GameEnv:
            def __init__(self, config, **kwargs):
                self.path = config.log_dir / "game_unique.log"
                self.path.write_bytes(b"start\n")
                self.closed = False

            def reset(self):
                return {"tokens": np.zeros((1, 1)), "map": np.zeros((1, 1))}, {"tick": 0}

            def step(self, action):
                if fail:
                    raise RuntimeError("game failed")
                return self.reset()[0], {}, True, {"episode": {"outcome": "victory"}}

            step_tools = step

            def close(self):
                if not self.closed:
                    with self.path.open("ab") as stream:
                        stream.write(b"final Lua warning\n")
                    self.closed = True

        pipe = Pipe(messages)
        with patch("fa_env.factored.FactoredEnv", GameEnv), patch("fa_env.tools_env.ToolsEnv", GameEnv):
            worker_main(3, pipe, str(root))
        return pipe

    def check_artifact(self, artifact):
        path = Path(artifact["path"])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.read_bytes(), b"start\nfinal Lua warning\n")
        self.assertEqual(artifact["bytes"], path.stat().st_size)
        self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_terminal_eval_attaches_closed_logs_and_unique_episode_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipe = self.run_worker(root, [("reset", {"eval": True, "uid": 7}), ("act", {}),
                                          ("reset", {"eval": True, "uid": 7}), ("act", {}), ("stop",)])
            terminal = [msg[4]["episode"] for msg in pipe.sent if msg[0] == "obs" and msg[3]]
            self.assertEqual(len(terminal), 2)
            paths = []
            for episode in terminal:
                self.assertEqual(episode["outcome"], "victory")
                self.assertEqual(len(episode["game_logs"]), 1)
                artifact = episode["game_logs"][0]
                self.check_artifact(artifact)
                paths.append(artifact["path"])
                manifest = json.loads((Path(artifact["path"]).parent / "manifest.json").read_text())
                self.assertEqual(manifest["uid"], 7)
            self.assertNotEqual(*paths)
            self.assertEqual(list((root / "worker03").iterdir()), [])

    def test_terminal_tools_eval_attaches_closed_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = self.run_worker(Path(directory), [("reset", {"eval": True, "uid": 12, "kind": "tools"}),
                                                      ("act", []), ("stop",)])
            terminal = next(msg[4]["episode"] for msg in pipe.sent if msg[0] == "obs" and msg[3])
            self.check_artifact(terminal["game_logs"][0])

    def test_failed_eval_keeps_full_log_and_uid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipe = self.run_worker(root, [("reset", {"eval": True, "uid": 8}), ("act", {}), ("stop",)], fail=True)
            error = next(msg for msg in pipe.sent if msg[0] == "error")
            self.assertEqual(error[2], 8)
            self.assertIn("game failed", error[1])
            manifests = list((root / "episodes").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text())
            self.assertEqual(manifest["uid"], 8)
            self.check_artifact(manifest["game_logs"][0])

    def test_interrupted_eval_is_retained_on_close_and_stop(self):
        for ending in [("close",), ("stop",)]:
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                messages = [("reset", {"eval": True, "uid": 9}), ending]
                if ending[0] == "close":
                    messages.append(("stop",))
                self.run_worker(root, messages)
                artifacts = list((root / "episodes").glob("*/*.log"))
                self.assertEqual(len(artifacts), 1)
                self.assertEqual(artifacts[0].read_bytes(), b"start\nfinal Lua warning\n")

    def test_training_success_cleanup_and_failure_cap_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_worker(root, [("reset", {"uid": 10}), ("act", {}), ("stop",)])
            self.assertFalse((root / "episodes").exists())
            self.assertFalse((root / "failed").exists())
            self.assertEqual(list((root / "worker03").iterdir()), [])
            failed = root / "failed"
            for i in range(200):
                (failed / str(i)).mkdir(parents=True)
            self.run_worker(root, [("reset", {"uid": 11}), ("act", {}), ("stop",)], fail=True)
            self.assertEqual(len(list(failed.iterdir())), 200)
            self.assertEqual(list((root / "worker03").iterdir()), [])
            self.assertFalse((root / "episodes").exists())

    def test_archive_failure_recovers_original_uid_before_worker_reuse(self):
        for ending in [("act", {}), ("close",)]:
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch("fa_train.worker.shutil.copy2", side_effect=PermissionError("archive unavailable")):
                    with self.assertRaisesRegex(PermissionError, "archive unavailable"):
                        self.run_worker(root, [("reset", {"eval": True, "uid": 91}), ending, ("stop",)])
                    pending = root / "worker03" / "pending_eval.json"
                    self.assertEqual(json.loads(pending.read_text())["uid"], 91)
                    source = root / "worker03" / "game_unique.log"
                    self.assertEqual(source.read_bytes(), b"start\nfinal Lua warning\n")
                    # A restart while archiving is still unavailable must fail before UID 92
                    # can launch a game and overwrite UID 91's source log.
                    with self.assertRaisesRegex(PermissionError, "archive unavailable"):
                        self.run_worker(root, [("reset", {"uid": 92}), ("act", {}), ("stop",)])
                    self.assertEqual(source.read_bytes(), b"start\nfinal Lua warning\n")
                    self.assertEqual(json.loads(pending.read_text())["uid"], 91)
                self.run_worker(root, [("reset", {"uid": 92}), ("act", {}), ("stop",)])
                manifests = list((root / "episodes").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text())
                self.assertEqual(manifest["uid"], 91)
                self.check_artifact(manifest["game_logs"][0])
                self.assertEqual(list((root / "worker03").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
