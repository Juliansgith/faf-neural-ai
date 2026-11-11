"""A schema change must not erase the historical part of an RL training window."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from unittest.mock import patch
from fa_train import rl_loop, tools_bc
from tests.test_rl_loop_ops import Run


class CacheVersionTrainingTests(unittest.TestCase):
    def test_failed_learner_does_not_consume_fresh_games_on_restart(self):
        r = Run()
        try:
            r.journal('rl_001')
            failed_round = r.round('lane0_20261021T000000Z', 'rl_001', n=30)
            fresh_round = r.round('lane1_20261022T000000Z', 'rl_001', n=30)
            with (r.run / 'rl.jsonl').open('a') as stream:
                stream.write(json.dumps({'iteration': 2, 'from': str(r.run / 'checkpoints/rl_001.pt'),
                                         'rounds': [failed_round.name], 'new_games': 30, 'exit': 1}) + '\n')
            self.assertEqual(r.main('--min-new-games', '50', '--iterations', '3'), rl_loop.EXIT_DONE)
            self.assertEqual(len(r.learned), 1)
            self.assertEqual(r.rows()[-1]['new_games'], 60)
            self.assertEqual(set(r.rows()[-1]['rounds']), {failed_round.name, fresh_round.name})
        finally:
            r.close()

    def test_hold_during_sil_encoding_prevents_a_new_learner(self):
        for marker, expected in (("YIELD", rl_loop.EXIT_YIELD), ("STOP", rl_loop.EXIT_DONE)):
            with self.subTest(marker=marker):
                r = Run()
                try:
                    r.journal('rl_001')
                    r.round('lane0_20261021T000000Z', 'rl_001', n=30)
                    def hold_during_sil(*args, should_stop=None):
                        self.assertFalse(should_stop())
                        (r.run / marker).write_text('deployment')
                        self.assertTrue(should_stop())
                        return 0
                    with patch.object(rl_loop, 'ensure_encoded', side_effect=hold_during_sil):
                        self.assertEqual(r.main('--sil-archive', '1'), expected)
                    self.assertEqual(r.learned, [])
                finally:
                    r.close()

    def test_cache_stop_finishes_current_game_without_starting_another(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = [root / 'one.jsonl.gz', root / 'two.jsonl.gz']
            stop = False
            def encode(*args):
                nonlocal stop
                stop = True
                return {'returns': np.zeros(1)}, {}
            with patch.object(tools_bc, 'read_blueprints', return_value=[]), \
                    patch.object(tools_bc, '_cache_manifest', return_value={'key': 'fixture'}), \
                    patch.object(tools_bc, 'encode_game', side_effect=encode) as encoder, \
                    patch.object(tools_bc, '_write_cache_atomic') as writer, \
                    patch.object(tools_bc, '_add_game_stats'):
                result = tools_bc.build_cache(root, root / 'cache', files=files, log=lambda *_: None,
                                              should_stop=lambda: stop)
            self.assertEqual(encoder.call_count, 1)
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(result['games'], 1)
            self.assertTrue(result['cancelled'])

    def test_yield_arriving_during_intake_prevents_a_new_learner(self):
        r = Run()
        try:
            r.journal('rl_001')
            r.round('lane0_20261021T000000Z', 'rl_001', n=30)
            def yield_during_collect(*args):
                (r.run / 'YIELD').write_text('deployment')
                return 0
            with patch.object(rl_loop, 'local_collect', side_effect=yield_during_collect):
                self.assertEqual(r.main('--local-home', str(r.d / 'player')), rl_loop.EXIT_YIELD)
            self.assertEqual(r.encoded, [])
            self.assertEqual(r.learned, [])
        finally:
            r.close()

    def test_main_repairs_consumed_round_before_learning_without_counting_it_as_fresh(self):
        r = Run()
        try:
            r.journal('rl_001')
            old = r.round('lane0_20261021T000000Z', 'rl_001', n=30)
            fresh = r.round('lane1_20261022T000000Z', 'rl_001', n=30)
            rows = r.rows()
            rows[-1]['rounds'] = [old.name]
            (r.run / 'rl.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(r.main('--champion', 'rl_001', '--iterations', '2'), rl_loop.EXIT_DONE)
            self.assertEqual(r.encoded, [fresh.name, old.name])
            self.assertEqual(len(r.learned), 1)
            self.assertEqual(r.rows()[-1]['new_games'], 30)
            self.assertEqual(r.rows()[-1]['rounds'], [fresh.name])
        finally:
            r.close()

    def test_stale_and_partially_reencoded_rounds_are_repaired_with_a_bounded_window(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            names = ['lane0_20261020T000000Z', 'lane1_20261021T000000Z',
                     'lane2_20261022T000000Z', 'lane3_20261023T000000Z']
            for name in names:
                p = root / name
                (p / 'raw').mkdir(parents=True)
                (p / 'round.json').write_text(json.dumps({'games': 2}))
            sem = tools_bc._encoder_semantics_hash()
            index = {}
            for rnd, count, current in ((names[0], 2, False), (names[1], 2, False),
                                        (names[2], 1, True), (names[3], 2, True)):
                for i in range(count):
                    index[f'{rnd}_{i}'] = {'bc': True, 'manifest': {
                        'format': tools_bc.CACHE_FORMAT if current else tools_bc.CACHE_FORMAT - 1,
                        'encoder_sha256': sem, 'key': str(i),
                        'observation_precision': tools_bc.PRECISION_CONTRACT,
                        'source_root': str(root / rnd / 'raw'), 'source_name': f'{i}.jsonl.gz'}}
            with patch.object(tools_bc, 'cache_index', return_value=index):
                consumed = set(names)
                result = rl_loop.stale_training_rounds(root, root / 'cache', set(names), consumed, 6)
                self.assertEqual([p.name for p in result], [names[2], names[1]])
                self.assertEqual(consumed, set(names))  # re-encoding is not new experience
                # The normal intake path handles previously unconsumed rounds.
                self.assertEqual(rl_loop.stale_training_rounds(root, root / 'cache', set(names), set(), 6), [])
                # Two entries for the same SIL source still do not complete its round.
                index['duplicate'] = index[f'{names[2]}_0']
                self.assertEqual([p.name for p in rl_loop.stale_training_rounds(
                    root, root / 'cache', set(names), consumed, 4)], [names[2]])


if __name__ == '__main__':
    unittest.main()
