try:
    import no_network
except ImportError:
    from tests import no_network
import unittest
from fa_train.rl_loop import candidate_games_pending
from fa_train import rl_loop
from tests.test_rl_loop_ops import Run


class CandidateCompletionTests(unittest.TestCase):
    def test_312_game_candidate_keeps_playing_until_fixed_400_look(self):
        condition = ('m28aicheat@0.9', 1, 1)
        seqs = {'rl_188': [(condition, 1)] * 500, 'rl_219': [(condition, 0)] * 312}
        self.assertEqual(candidate_games_pending(seqs, 'rl_188', 'rl_219', 400), 88)
        seqs['rl_219'] += [(condition, 0)] * 104
        self.assertEqual(candidate_games_pending(seqs, 'rl_188', 'rl_219', 400), 0)

    def test_unmatched_opponents_do_not_fill_the_gate(self):
        seqs = {'champ': [(('adaptive', 1, 1), 1)] * 400,
                'candidate': [(('m28aicheat@0.9', 1, 1), 1)] * 500}
        self.assertEqual(candidate_games_pending(seqs, 'champ', 'candidate', 400), 400)

    def test_restart_from_champion_and_legacy_mode_do_not_wait(self):
        self.assertEqual(candidate_games_pending({}, 'champ', 'champ', 400), 0)
        self.assertEqual(candidate_games_pending({}, 'champ', 'candidate', 0), 0)

    def test_loop_waits_without_training_then_uses_the_completed_candidate(self):
        r = Run()
        try:
            r.journal('rl_001', 'rl_002')
            r.round('lane0_20261021T000000Z', 'rl_001', rate=1, n=500)
            r.round('lane1_20261021T010000Z', 'rl_002', rate=1, n=312)
            waits = []

            def arrive(_):
                self.assertEqual(r.learned, [])
                self.assertEqual(waits, [])
                waits.append(True)
                r.round('lane2_20261021T020000Z', 'rl_002', rate=1, n=104)

            self.assertEqual(r.main('--champion', 'rl_001', '--min-candidate-games', '400',
                                    '--iterations', '3', sleep=arrive), rl_loop.EXIT_DONE)
            self.assertEqual(len(waits), 1)
            self.assertEqual(len(r.learned), 1)
        finally:
            r.close()


if __name__ == '__main__':
    unittest.main()
