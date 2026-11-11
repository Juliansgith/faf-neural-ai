"""Fixed gate looks must not select games by outcome-dependent finish time."""
try:
    import no_network
except ImportError:
    from tests import no_network
import json
import unittest
from fa_train import rl_loop
from tests.test_rl_loop_ops import Run


class GateOrderingTests(unittest.TestCase):
    def setUp(self):
        self.r=Run();self.r.journal('rl_001','rl_002')
        self.base=self.r.round('lane0_20261025T000000Z','rl_001',n=600)
        self.candidate=self.r.round('lane1_20261025T010000Z','rl_002',n=600)
        self.rows=[dict(ai='m28aicheat@0.9',map='SCMP_012',faction=1,ai_faction=2,slots=[1,2],seed=i,
                        outcome='victory' if i<250 else 'defeat') for i in range(600)]
        self.write(self.base,self.rows)
        self.write(self.candidate,self.rows) # wins finish first; both networks have identical games/outcomes
        self.names={rl_loop.sha256(p):p.stem for p in self.r.ckpts.glob('*.pt')}
    def tearDown(self):self.r.close()
    def write(self,folder,rows):
        (folder/'eval/episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    def sequences(self):return rl_loop.gate_sequences(self.r.run/'rounds',self.names,'')
    def test_identical_games_with_fast_wins_cannot_create_a_promotion(self):
        state={}
        verdict=rl_loop.judge_gate(self.sequences(),'rl_001',['rl_001','rl_002'],state)
        self.assertIsNone(verdict)
        self.assertEqual(state['verdicts']['rl_002@rl_001']['promotion']['verdict'],'not better')
    def test_reversing_finish_order_preserves_fixed_gate_sequences_and_verdicts(self):
        before=self.sequences();state1={}
        verdict1=rl_loop.judge_gate(before,'rl_001',['rl_001','rl_002'],state1)
        self.write(self.candidate,list(reversed(self.rows)))
        after=self.sequences();state2={}
        verdict2=rl_loop.judge_gate(after,'rl_001',['rl_001','rl_002'],state2)
        self.assertEqual(before,after)
        self.assertEqual(verdict1,verdict2)
        for state in (state1,state2):
            for v in state['verdicts'].values():
                for k in ('rollback','promotion'):v.get(k,{}).pop('time',None)
        self.assertEqual(state1,state2)
    def test_missing_or_duplicate_identity_excludes_the_round(self):
        for rows in ([dict(self.rows[0],seed=None),*self.rows[1:]],
                     [self.rows[0],self.rows[0],*self.rows[2:]]):
            with self.subTest(invalid=rows[1]['seed']):
                self.write(self.candidate,rows)
                self.assertNotIn('rl_002',self.sequences())
    def test_order_ignores_outcomes_and_timing(self):
        first=rl_loop.ordered_gate_episodes(self.rows)
        changed=[dict(r,outcome='defeat' if r['outcome']=='victory' else 'victory',minutes=999-i)
                 for i,r in enumerate(reversed(self.rows))]
        second=rl_loop.ordered_gate_episodes(changed)
        self.assertEqual([r['seed'] for r in first],[r['seed'] for r in second])


if __name__=='__main__':unittest.main()
