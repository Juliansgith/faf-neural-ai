"""Gate waits preserve replay work until a candidate can be replaced."""
import json
import unittest
from unittest import mock

try:                                   # first: no test may reach the server or start a real process
    import no_network  # noqa: F401
    from test_rl_loop_ops import Run
except ImportError:
    from tests import no_network  # noqa: F401
    from tests.test_rl_loop_ops import Run
from fa_train import rl_loop, storage


class GateIntakeTest(unittest.TestCase):
    def setUp(self):
        self.r=Run()
        self.r.journal('rl_001','rl_002')
        self.reference=self.r.round('lane0_20261026T000000Z','rl_001',rate=.8,n=400,seed=42)
        self.flags=('--champion','rl_001','--gate-games','150','--min-candidate-games','400',
                    '--games-per-iteration','720','--guard-games','0','--collapse-games','0')

    def tearDown(self):
        self.r.close()

    def test_unready_candidate_preserves_raw_and_does_not_encode(self):
        candidate=self.r.round('lane1_20261026T000000Z','rl_002',rate=.8,n=141,seed=42)
        before=(self.r.run/'rl.jsonl').read_bytes()
        self.r.main(*self.flags)
        self.assertEqual(self.r.encoded,[])
        self.assertEqual(self.r.learned,[])
        self.assertEqual((self.r.run/'rl.jsonl').read_bytes(),before)
        self.assertEqual(len(list((candidate/'raw').glob('*.jsonl.gz'))),141)
        self.assertEqual(len(list((self.reference/'raw').glob('*.jsonl.gz'))),400)

    def test_early_rollback_runs_without_encoding_rejected_candidate(self):
        self.r.round('lane1_20261026T000000Z','rl_002',rate=0,n=200)
        code=self.r.main(*self.flags,'--max-rejections','1')
        self.assertEqual(code,rl_loop.EXIT_GATE_EXHAUSTED)
        self.assertTrue(self.r.rows()[-1]['fallback'])
        self.assertEqual(self.r.pushed[-1],'rl_001')
        self.assertEqual(self.r.encoded,[])
        self.assertEqual(self.r.learned,[])

    def test_housekeeping_runs_during_candidate_wait(self):
        self.r.round('lane1_20261026T000000Z','rl_002',n=100)
        with mock.patch.object(storage,'plan',return_value=[]) as housekeeping:
            self.r.main(*self.flags,'--storage-budget-gb','1')
        housekeeping.assert_called_once()
        self.assertEqual(self.r.encoded,[])

    def test_arriving_games_resume_encoding_and_learning(self):
        self.r.round('lane1_20261026T000000Z','rl_002',rate=.8,n=100,seed=42)
        waits=[]
        def arrive(seconds):
            if seconds!=300:return
            self.assertFalse(waits,'The candidate should become ready after one wait')
            waits.append(list(self.r.encoded))
            folder=self.r.round('lane1_20261026T000001Z','rl_002',rate=.8,n=300,seed=42)
            path=folder/'eval/episodes.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:row['seed']+=100
            path.write_text('\n'.join(json.dumps(row) for row in rows))
        self.r.main(*self.flags,'--iterations','3',sleep=arrive)
        self.assertEqual(waits,[[]])
        self.assertEqual(len(self.r.learned),1)
        self.assertEqual(set(self.r.encoded),{'lane0_20261026T000000Z','lane1_20261026T000000Z','lane1_20261026T000001Z'})

    def test_disabled_gate_keeps_normal_intake(self):
        self.r.round('lane1_20261026T000000Z','rl_002',n=20)
        self.r.main(*self.flags,'--gate-games','0','--iterations','3')
        self.assertEqual(len(self.r.learned),1)
        self.assertEqual(len(self.r.encoded),2)

    def test_current_champion_keeps_normal_intake(self):
        self.r.main(*self.flags,'--set-champion','rl_002','--iterations','3')
        self.assertEqual(len(self.r.learned),1)
        self.assertEqual(self.r.encoded,['lane0_20261026T000000Z'])


if __name__=='__main__':unittest.main()
