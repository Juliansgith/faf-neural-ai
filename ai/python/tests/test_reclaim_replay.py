import unittest
from fa_env.tools_obs import ToolsObserver
from fa_env.tools_env import ToolsEnv
from fa_env.reclaim_memory import after_step
from fa_train.tools_bc import replay_reclaim_memory


def observer(props=()):
    value=ToolsObserver.__new__(ToolsObserver);value.wrecks=[];value.set_wrecks(list(props));return value


class ReclaimReplayTest(unittest.TestCase):
    def step(self,target='0',count=49):
        props=[dict(id=str(i),mass=100-i,energy=0,x=i,z=0) for i in range(count)]
        return dict(queries=[dict(type='reclaim')],answers=[dict(ok=True,value=props)],
            sent=[dict(type='reclaim',target=target)],results=[dict(ok=False,error=f'unknown target {target}\nstack traceback:')])

    def live(self,ob,step):
        env=ToolsEnv.__new__(ToolsEnv);env.observer=ob
        env._refresh_reclaim(step['queries'],step['answers'])
        env._refresh_reclaim([],[],step['sent'],step['results'])
        return ob.wrecks

    def test_cap_then_failure_does_not_admit_discarded_prop(self):
        step=self.step();old=observer();old.set_wrecks(after_step([],step['queries'],step['answers'],step['sent'],step['results']))
        self.assertEqual(len(old.wrecks),48)  # Reproduces the previous defect.
        replay=observer();replay_reclaim_memory(replay,step)
        self.assertEqual(replay.wrecks,self.live(observer(),step))
        self.assertEqual(len(replay.wrecks),47)
        self.assertNotIn('48',{p['id'] for p in replay.wrecks})

    def test_failure_of_already_discarded_target_preserves_cap(self):
        step=self.step(target='48');replay=observer();replay_reclaim_memory(replay,step)
        self.assertEqual(replay.wrecks,self.live(observer(),step));self.assertEqual(len(replay.wrecks),48)

    def test_no_scan_removes_only_failed_target(self):
        step=self.step(count=3);props=step['answers'][0]['value'];step['queries']=[];step['answers']=[]
        replay=observer(props);replay_reclaim_memory(replay,step)
        self.assertEqual(replay.wrecks,self.live(observer(props),step));self.assertEqual(len(replay.wrecks),2)

    def test_successful_empty_scan_clears_previous_memory(self):
        step=self.step(count=0);replay=observer([dict(id='prior',mass=5)])
        replay_reclaim_memory(replay,step);self.assertEqual(replay.wrecks,[])


if __name__=='__main__':unittest.main()
