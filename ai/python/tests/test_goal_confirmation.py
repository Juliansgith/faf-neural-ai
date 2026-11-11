"""Accounting/orchestration fixtures only, never evidence of playing strength."""
import copy
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from fa_train.benchmark import schedule_hash, suite, summarize
from fa_train.goal_confirmation import load_suites, require_finalized, sentinel_suites, verify_result
from fa_train import goal_confirmation


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def plan(self):
        specs = {}
        for name, rows in dict(confirmation=suite(100000), **sentinel_suites(200000)).items():
            path = self.root / (name + '.json')
            path.write_text(json.dumps(dict(rows=rows, schedule_sha256=schedule_hash(rows))))
            specs[name] = dict(path=str(path), sha256=schedule_hash(rows))
        return dict(suites=specs)

    def test_sentinels_cannot_be_laundered_into_target_or_use_target_seeds(self):
        plan = self.plan()
        path = Path(plan['suites']['scripted']['path'])
        rows = json.loads(path.read_text())['rows']
        for field, value in [('seed', 100000), ('ai', 'm28aicheat@0.9')]:
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            digest = schedule_hash(changed)
            path.write_text(json.dumps(dict(rows=changed, schedule_sha256=digest)))
            plan['suites']['scripted']['sha256'] = digest
            with self.assertRaises(ValueError):
                load_suites(plan)

    def test_plan_rejects_wrong_opponent_and_joint_stratum_even_with_new_hash(self):
        plan = self.plan()
        self.assertEqual(len(load_suites(plan)['confirmation']), 640)
        path = Path(plan['suites']['confirmation']['path'])
        original = json.loads(path.read_text())['rows']
        for change in ('opponent', 'joint_stratum'):
            rows = copy.deepcopy(original)
            if change == 'opponent':
                rows[0]['ai'] = 'm28aicheat@0.75'
            else:
                rows[0]['faction'], rows[100]['faction'] = rows[100]['faction'], rows[0]['faction']
            digest = schedule_hash(rows)
            path.write_text(json.dumps(dict(rows=rows, schedule_sha256=digest)))
            plan['suites']['confirmation']['sha256'] = digest
            with self.assertRaises(ValueError):
                load_suites(plan)

    def test_holdout_cannot_run_early_or_before_finalization(self):
        state = self.root / 'autopilot/state.json'
        state.parent.mkdir()
        state.write_text('{"mode":"running"}')
        plan = dict(training_run=str(self.root), not_before='2026-10-30T12:00:00+01:00',
                    deadline='2026-10-30T23:59:59+01:00')
        with self.assertRaisesRegex(RuntimeError, 'not due'):
            require_finalized(plan, datetime(2026, 10, 30, 10, tzinfo=timezone.utc))
        with self.assertRaisesRegex(RuntimeError, 'not finalized'):
            require_finalized(plan, datetime(2026, 10, 30, 12, tzinfo=timezone.utc))
        state.write_text('{"mode":"finalized"}')
        require_finalized(plan, datetime(2026, 10, 30, 12, tzinfo=timezone.utc))
        with self.assertRaisesRegex(RuntimeError, 'deadline has passed'):
            require_finalized(plan, datetime(2026, 10, 31, 12, tzinfo=timezone.utc))

    def test_result_recomputes_denominator_and_rejects_retries_or_different_checkpoint(self):
        folder = self.root / 'confirmation'
        (folder / 'eval').mkdir(parents=True)
        rows = suite(500000)
        results = [dict(r, outcome='victory' if i < 410 else 'defeat') for i, r in enumerate(rows)]
        manifest = dict(checkpoint_sha256='pinned', rows=rows, schedule_sha256=schedule_hash(rows))
        result = summarize(rows, results)
        self.assertTrue(result['statistical_threshold_met'])
        result.update(evaluation_valid=True, provenance_unchanged=True, restarted_attempts=0)
        (folder / 'eval/eval_manifest.json').write_text(json.dumps(manifest))
        (folder / 'eval/eval_results.json').write_text(json.dumps(dict(games=results)))
        report = folder / 'eval/benchmark.json'
        report.write_text(json.dumps(result))
        measured, _ = verify_result(folder, rows, {'checkpoint_sha256': 'pinned'})
        self.assertEqual(measured['total']['wins'], 410)
        with self.assertRaisesRegex(RuntimeError, 'checkpoint_sha256'):
            verify_result(folder, rows, {'checkpoint_sha256': 'changed'})
        result['restarted_attempts'] = 1
        report.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, 'retried'):
            verify_result(folder, rows, {'checkpoint_sha256': 'pinned'})
        result['restarted_attempts'] = 0
        result['total']['wins'] = 640
        report.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, 'accounting mismatch'):
            verify_result(folder, rows, {'checkpoint_sha256': 'pinned'})

    def test_controller_uses_one_snapshot_and_releases_own_pause_on_success_or_failure(self):
        from fa_env import bridge
        for fail in (False, True):
            plan = self.plan()
            home, training, engine = self.root / 'home', self.root / 'training', self.root / 'engine'
            for p in (home, training / 'checkpoints', training / 'autopilot', engine):
                p.mkdir(parents=True, exist_ok=True)
            (training / 'autopilot/state.json').write_text('{"mode":"finalized"}')
            (training / 'champion.json').write_text('{"name":"fixture"}')
            (training / 'checkpoints/fixture.pt').write_bytes(b'not a real model')
            (engine / 'main.exe').write_bytes(b'not an engine')
            run = self.root / ('failed' if fail else 'success')
            plan.update(training_run=str(training), home=str(home), bin_dir=str(engine), run=str(run),
                        not_before='2000-01-01T00:00:00+00:00', deadline='2100-01-01T00:00:00+00:00')
            path = self.root / 'plan.json'
            path.write_text(json.dumps(plan))
            calls = []

            def fake_evaluator(cmd, **kwargs):
                def arg(flag):
                    return cmd[cmd.index(flag) + 1]
                calls.append(cmd)
                self.assertEqual(arg('--device'), 'cpu')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(kwargs['creationflags'], 0x08000000)
                if fail:
                    return 1
                folder = Path(arg('--run'))
                folder.mkdir()
                rows = json.loads(Path(arg('--schedule-file')).read_text())['rows']
                games = [dict(r, outcome='victory' if len(rows) == 640 else 'defeat') for r in rows]
                result = summarize(rows, games)
                result.update(evaluation_valid=True, provenance_unchanged=True, restarted_attempts=0)
                manifest = dict(checkpoint_sha256=goal_confirmation.sha256(arg('--checkpoint')),
                                engine_sha256=goal_confirmation.sha256(engine / 'main.exe'),
                                rows=rows, schedule_sha256=schedule_hash(rows), policy='sample T=0.5 top_p=None',
                                build_budget=None, commander_greedy=False, policy_source_sha256={'x': 'frozen'},
                                game_sources=dict(bridge_sha256='b', hooks_sha256='h', mods={}))
                for name, data in [('benchmark', result), ('eval_results', {'games': games}),
                                   ('eval_manifest', manifest)]:
                    (folder / (name + '.json')).write_text(json.dumps(data))
                return 0

            with mock.patch('sys.argv', ['confirmation', '--plan', str(path)]), \
                    mock.patch('sys.platform', 'win32'), \
                    mock.patch.object(goal_confirmation, 'active_player_evaluations', return_value=[]), \
                    mock.patch.object(goal_confirmation, 'freeze_runtime', return_value=(self.root, self.root)), \
                    mock.patch.object(bridge, 'KNOWN_MODS', {'M28AI': self.root}), \
                    mock.patch.object(goal_confirmation.subprocess, 'run', return_value=SimpleNamespace(stdout='2')), \
                    mock.patch.object(goal_confirmation.subprocess, 'call', side_effect=fake_evaluator), \
                    contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'evaluator exited 1'):
                        goal_confirmation.main()
                else:
                    goal_confirmation.main()
            self.assertFalse((home / 'PAUSE').exists())
            self.assertEqual(len(calls), 1 if fail else 4)
            self.assertEqual(len({c[c.index('--checkpoint') + 1] for c in calls}), 1)
            if not fail:
                result = json.loads((run / 'result.json').read_text())
                self.assertTrue(result['completed_by_deadline'])
                self.assertEqual(result['target']['total']['games'], 640)
                self.assertEqual(result['target']['total']['wins'], 640)
                self.assertEqual(sum(s['total']['wins'] for s in result['sentinels'].values()), 0)


if __name__ == '__main__':
    unittest.main()
