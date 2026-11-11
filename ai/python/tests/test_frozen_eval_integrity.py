"""Exercise the evaluator's real completion path, with game execution replaced.
These fixtures test accounting only; they provide no performance evidence.
"""
try:
    import no_network
except ImportError:
    from tests import no_network
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from examples import tools_eval
from fa_env import bridge
from fa_train import tools_trainer
from fa_train.benchmark import schedule_hash, suite


class FrozenEvalIntegrityTests(unittest.TestCase):
    def run_evaluation(self, restarts=0, change_checkpoint=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, schedule = root / 'frozen.pt', root / 'suite.json'
            torch.save({'size': 'S', 'decoder': 'units'}, checkpoint)
            (root / 'main.exe').write_bytes(b'engine fixture, never executed')
            rows = suite(101000)
            schedule.write_text(json.dumps({'rows': rows, 'schedule_sha256': schedule_hash(rows)}))

            class FakeTrainer:
                def __init__(self, *args, **kwargs):
                    self.acting = SimpleNamespace()
                    self.eval_queue = []
                    self.eval_results = [dict(row, outcome='victory') for row in rows]

                def load(self, *args, **kwargs):
                    pass

                def schedule_eval(self, n, ai):
                    self.eval_queue.extend({} for _ in range(n))

                def run(self, mode):
                    if change_checkpoint:
                        checkpoint.write_bytes(b'replaced during run')
                    (root / 'run/episodes.jsonl').write_text('')
                    return {'restarts': restarts}

                def close(self):
                    pass

            argv = ['tools_eval', '--checkpoint', str(checkpoint), '--schedule-file', str(schedule),
                    '--run', str(root / 'run'), '--device', 'cpu']
            with mock.patch('sys.argv', argv), mock.patch.object(bridge, 'DEFAULT_BIN_DIR', root), \
                    mock.patch.object(bridge, 'resolve_mods', return_value={}), \
                    mock.patch.object(tools_trainer, 'ToolsTrainer', FakeTrainer), \
                    contextlib.redirect_stdout(io.StringIO()):
                if restarts or change_checkpoint:
                    with self.assertRaisesRegex(RuntimeError, 'retried, or source-changing'):
                        tools_eval.main()
                else:
                    tools_eval.main()
            return json.loads((root / 'run/benchmark.json').read_text())

    def test_retried_games_cannot_be_hidden_by_a_perfect_final_score(self):
        result = self.run_evaluation(restarts=1)
        self.assertTrue(result['complete'])
        self.assertEqual(result['total']['rate'], 1)
        self.assertEqual(result['restarted_attempts'], 1)
        self.assertFalse(result['evaluation_valid'])
        self.assertFalse(result['statistical_threshold_met'])

    def test_replacing_checkpoint_during_evaluation_invalidates_result(self):
        result = self.run_evaluation(change_checkpoint=True)
        self.assertFalse(result['provenance_unchanged'])
        self.assertFalse(result['statistical_threshold_met'])

    def test_clean_complete_result_passes_accounting(self):
        result = self.run_evaluation()
        self.assertTrue(result['evaluation_valid'])


if __name__ == '__main__':
    unittest.main()
