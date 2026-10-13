"""Real message-handler coverage for varying simulation cadence and resets."""
import unittest
from types import SimpleNamespace
from fa_train.trainer import Trainer, WorkerState


class ThroughputTest(unittest.TestCase):
    def fixture(self, cadence=2):
        trainer=Trainer.__new__(Trainer)
        worker=WorkerState(SimpleNamespace(send=lambda message:None),None)
        worker.episode={'uid':1,'eval':True,'decision_ticks':cadence}
        trainer.workers={0:worker};trainer.mode='eval';trainer.eval_queue=[]
        trainer.decisions_total=trainer.throughput_decisions=0
        trainer.throughput_game_s=0.
        trainer.reward=lambda rewards:0.
        trainer._episode_end=lambda wid,info:None
        return trainer,worker

    def message(self, trainer, tick, first=False, done=False, uid=1):
        info={'tick':tick,'uid':uid,'first':first}
        rewards=None if first else {'terminal':0.,'trade':0.,'economy':0.,'commander':0.}
        return trainer._handle(0,('obs',{},rewards,done,info),'policy')

    def test_two_tick_steps_and_short_final_step(self):
        t,w=self.fixture();self.message(t,1,first=True)
        self.message(t,3);self.message(t,5);self.message(t,6,done=True)
        self.assertAlmostEqual(t.throughput_game_s,.5)
        self.assertEqual(t.throughput_decisions,3)

    def test_twenty_tick_cadence_and_legacy_missing_tick(self):
        t,w=self.fixture(20);self.message(t,10,first=True)
        self.message(t,30);self.message(t,None)
        self.assertAlmostEqual(t.throughput_game_s,4.)

    def test_stale_message_and_reset_do_not_leak_elapsed_time(self):
        t,w=self.fixture();self.message(t,100,first=True)
        self.message(t,10000,uid=999)
        self.assertEqual(w.last_tick,100)
        self.assertEqual(t.throughput_game_s,0.)
        t.next_uid=1;t._episode=lambda mode:{'eval':True,'decision_ticks':2}
        t._send_reset(0,'policy');self.assertIsNone(w.last_tick)
        self.message(t,1,first=True,uid=2);self.message(t,3,uid=2)
        self.assertAlmostEqual(t.throughput_game_s,.2)


if __name__=='__main__':unittest.main()
