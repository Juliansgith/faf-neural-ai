"""A placement/reclaim query must not erase intervening teacher commands."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from fa_env.tools_env import ToolsEnv


class CaptureSubstepsTest(unittest.TestCase):
    def run_step(self, batches, *, capture=True, terminate_first=False, query=True):
        env=object.__new__(ToolsEnv)
        env.state=SimpleNamespace(tick=2)
        env.me=1;env.obs={'tick':2,'units':[]};env.decisions=1
        env.decision_ticks=2;env.time_limit_ticks=100
        env.rallied_by_player=set();env.claimed={}
        own_action={'type':'stop','units':['1']}
        env.kit=SimpleNamespace(plan=lambda *args:([own_action], [{'type':'reclaim'}] if query else [], []),
                                complete=lambda *args:[],role={})
        env.observer=SimpleNamespace(taken=set(),note_commands=Mock(),observe=lambda obs:obs)
        env.recorder=Mock()
        env._refresh_reclaim=Mock();env._note_economy_snapshot=Mock();env._note_recent_state=Mock()
        responses=[]
        for i,orders in enumerate(batches):
            responses.append(({'tick':3+i,'orders':orders},0,terminate_first and i==0,False,
                              {'results':[{'ok':True}] if i==0 else [],'queries':[]}))
        env.fa=SimpleNamespace(capture=capture,server=SimpleNamespace(last_line=b'input'),step=Mock(side_effect=responses))
        def finish(obs,info,terminated,truncated,choice):
            env.obs=obs
            return None,{},terminated or truncated,{}
        env._finish=finish
        _,_,_,info=env.step_tools([])
        return env.recorder.step.call_args.args[1]['sent'],info,env.fa.step.call_count,own_action

    def test_first_substep_order_survives_empty_final_capture(self):
        order={'type':'factory_build','units':['7'],'bp':'uel0105','tick':3}
        sent,info,calls,_=self.run_step([[order],[]])
        self.assertEqual(sent,[order]);self.assertEqual(info['captured'],1);self.assertEqual(calls,2)

    def test_chronology_and_early_termination(self):
        a={'type':'clear','units':['7'],'tick':3};b={'type':'factory_build','units':['7'],'tick':4}
        self.assertEqual(self.run_step([[a],[b]])[0],[a,b])
        sent,_,calls,_=self.run_step([[a]],terminate_first=True)
        self.assertEqual(sent,[a]);self.assertEqual(calls,1)

    def test_human_actions_and_no_query_capture(self):
        ignored={'type':'move','units':['9']}
        sent,info,_,own=self.run_step([[ignored],[]],capture=False)
        self.assertEqual(sent,[own]);self.assertNotIn('captured',info)
        self.assertEqual(self.run_step([[ignored]],query=False)[0],[ignored])


if __name__=='__main__':unittest.main()
