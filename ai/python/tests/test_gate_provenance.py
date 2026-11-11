"""An engine, map, side, or opponent change must not promote a network."""
try:
    import no_network
except ImportError:
    from tests import no_network
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fa_train import autopilot, rl_loop
from tests.test_rl_loop_ops import Run


class GateProvenanceTests(unittest.TestCase):
    def test_sparse_cell_variance_keeps_its_finite_sample_correction(self):
        diff, se, games = rl_loop._stratified({'cell': [1, 1]}, {'cell': [0, 1]})
        self.assertEqual((diff, se, games), (-1., 1., 1))

    def setUp(self):
        self.r = Run()
        self.r.journal('rl_001', 'rl_002')
        self.champ = self.r.round('lane0_20261025T000000Z', 'rl_001', rate=.2, n=500)
        self.candidate = self.r.round('lane1_20261025T010000Z', 'rl_002', rate=.9, n=500)
        self.names = {rl_loop.sha256(p): p.stem for p in self.r.ckpts.glob('*.pt')}
        self.manifest = dict(engine_sha256='engine-A', policy='sample T=0.5 top_p=None',
                             build_budget=None, commander_greedy=False,
                             game_sources={'mods': {'m28': {'lua_sha256': 'mod-A'}}})
        for folder in (self.champ, self.candidate):
            self.write_manifest(folder, self.manifest)
            self.edit_episodes(folder, slots=[1, 2])

    def tearDown(self):
        self.r.close()

    def write_manifest(self, folder, value):
        (folder / 'eval/eval_manifest.json').write_text(json.dumps(value))

    def edit_episodes(self, folder, **changes):
        p = folder / 'eval/episodes.jsonl'
        rows = [dict(json.loads(line), **changes) for line in p.read_text().splitlines()]
        p.write_text('\n'.join(json.dumps(row) for row in rows))

    def sequences(self):
        return rl_loop.gate_sequences(self.r.run / 'rounds', self.names, '')

    def assert_unmatched(self):
        seqs = self.sequences()
        self.assertEqual(rl_loop.candidate_games_pending(seqs, 'rl_001', 'rl_002', 400), 400)
        self.assertIsNone(rl_loop.judge_gate(seqs, 'rl_001', ['rl_001', 'rl_002'], {}))
        scan = autopilot.scan_rounds(self.r.run / 'rounds', self.names, '')
        self.assertIsNone(autopilot.versus(scan, 'rl_001', ['rl_002']))

    def test_different_engines_never_count_as_candidate_improvement(self):
        self.write_manifest(self.candidate, dict(self.manifest, engine_sha256='engine-B'))
        self.assert_unmatched()
        self.write_manifest(self.champ, dict(self.manifest, engine_sha256='engine-B'))
        seqs = self.sequences()
        verdict = rl_loop.judge_gate(seqs, 'rl_001', ['rl_001', 'rl_002'], {})
        self.assertEqual(verdict[0], 'better')
        scan = autopilot.scan_rounds(self.r.run / 'rounds', self.names, '')
        self.assertEqual(autopilot.versus(scan, 'rl_001', ['rl_002'])[2], 500)

    def test_unknown_engine_is_not_assumed_to_match_a_known_one(self):
        (self.champ / 'eval/eval_manifest.json').unlink()
        self.assert_unmatched()

    def test_maps_and_starting_sides_pool_but_opponent_versions_stay_separate(self):
        for changes in ({'map': 'SCMP_013'}, {'slots': [2, 1]}):
            with self.subTest(changes=changes):
                self.edit_episodes(self.candidate, **changes)
                self.assertEqual(rl_loop.candidate_games_pending(self.sequences(), 'rl_001', 'rl_002', 400), 0)
                self.edit_episodes(self.candidate, map='SCMP_012', slots=[1, 2])
        self.write_manifest(self.candidate, dict(self.manifest,
                            game_sources={'mods': {'m28': {'lua_sha256': 'mod-B'}}}))
        self.assert_unmatched()

    def test_legacy_verdict_is_retained_but_cannot_decide_new_comparison(self):
        old = {'rollback': {'verdict': 'worse'}}
        state = {'verdicts': {'rl_002@rl_001': old.copy()}}
        self.write_manifest(self.candidate, dict(self.manifest, engine_sha256='engine-B'))
        self.assertIsNone(rl_loop.judge_gate(self.sequences(), 'rl_001', ['rl_001', 'rl_002'], state,
                                           comparison_schema=rl_loop.GATE_SCHEMA))
        self.assertEqual(state['verdicts']['rl_002@rl_001'], old)
        self.assertNotIn('rollback', state['verdicts']['rl_002@rl_001#' + rl_loop.GATE_SCHEMA])

    def test_bridge_and_hooks_changes_do_not_count_as_network_improvement(self):
        for field in ('bridge_sha256', 'hooks_sha256'):
            with self.subTest(field=field):
                base = dict(self.manifest, game_sources={**self.manifest['game_sources'], field: 'old'})
                changed = dict(base, game_sources={**base['game_sources'], field: 'new'})
                self.write_manifest(self.champ, base)
                self.write_manifest(self.candidate, changed)
                self.assert_unmatched()

    def test_observer_decoder_and_executor_changes_stay_separate(self):
        for source in ('fa_env/tools_obs.py', 'fa_env/tools.py', 'fa_train/tools_model.py',
                       'fa_train/tools_trainer.py', 'fa_train/worker.py'):
            with self.subTest(source=source):
                self.write_manifest(self.champ, dict(self.manifest, policy_source_sha256={source: 'old'}))
                self.write_manifest(self.candidate, dict(self.manifest, policy_source_sha256={source: 'new'}))
                self.assert_unmatched()

    def test_learner_and_reporting_edits_do_not_discard_valid_controls(self):
        self.write_manifest(self.champ, dict(self.manifest, policy_source_sha256={
            'fa_train/learn_rl.py': 'old', 'fa_train/autopilot.py': 'old'}))
        self.write_manifest(self.candidate, dict(self.manifest, policy_source_sha256={
            'fa_train/learn_rl.py': 'new', 'fa_train/autopilot.py': 'new'}))
        self.assertEqual(rl_loop.candidate_games_pending(self.sequences(), 'rl_001', 'rl_002', 400), 0)

    def test_unknown_actor_source_does_not_match_known_source(self):
        self.write_manifest(self.candidate, dict(self.manifest, policy_source_sha256={'fa_env/tools_obs.py': 'known'}))
        self.assert_unmatched()

    def test_newline_alias_needs_exact_recorded_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'fa_env/tools_obs.py'
            source.parent.mkdir()
            lf = b'def observe():\n    return 7\n'
            crlf = lf.replace(b'\n', b'\r\n')
            source.write_bytes(crlf)
            def manifest(data):
                return dict(self.manifest, policy_source_sha256={'fa_env/tools_obs.py': hashlib.sha256(data).hexdigest()})
            with patch.object(rl_loop, 'ROOT', root):
                self.write_manifest(self.champ, manifest(lf))
                self.write_manifest(self.candidate, manifest(crlf))
                self.assertEqual(rl_loop.candidate_games_pending(self.sequences(), 'rl_001', 'rl_002', 400), 0)
                self.write_manifest(self.candidate, manifest(lf.replace(b'return 7', b'return 8')))
                self.assert_unmatched()


if __name__ == '__main__':
    unittest.main()
