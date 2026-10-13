import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from fa_env.identical_navigation import identical_navigation
from fa_env.tools_env import ToolsEnv


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.units = {'1': dict(queue=1,idle=False,order=35,dest=[100.,120.])}
        self.actions = [dict(type='clear',units=['1']),dict(type='attack_move',units=['1'],pos=[100.,120.])]

    def test_identical_only_without_mutation(self):
        before=copy.deepcopy((self.actions,self.units))
        self.assertEqual(identical_navigation(self.actions,self.units),[])
        self.assertEqual((self.actions,self.units),before)

    def test_changed_idle_queued_or_missing_are_not_suppressed(self):
        for change in [dict(dest=[101.,120.]),dict(idle=True),dict(order=2),dict(queue=2),dict(queue=None),dict(dest=None),dict(dest=[float('nan'),120.])]:
            units={'1':dict(self.units['1'],**change)}
            self.assertEqual(identical_navigation(self.actions,units),self.actions,change)
        self.assertEqual(identical_navigation(self.actions,{}),self.actions)

    def test_partial_group_filters_both_clear_and_navigation(self):
        actions=[dict(a,units=['1','2']) for a in self.actions]
        self.assertEqual(identical_navigation(actions,self.units),[dict(a,units=['2']) for a in self.actions])

    def test_prior_action_prevents_stale_observation_suppression(self):
        for action in [dict(type='clear',units=['1']),dict(type='build',units=['1'],bp='ueb1101',pos=[3,4])]:
            actions=[action]+self.actions
            self.assertEqual(identical_navigation(actions,self.units),actions)

    def test_different_order_then_original_must_execute_both(self):
        changed=[self.actions[0],dict(self.actions[1],pos=[200.,120.])]
        actions=changed+self.actions
        self.assertEqual(identical_navigation(actions,self.units),actions)

    def test_non_navigation_and_unpaired_actions_untouched(self):
        for verb in ['patrol','build','guard','attack','factory_build']:
            actions=[self.actions[0],dict(self.actions[1],type=verb)]
            self.assertEqual(identical_navigation(actions,self.units),actions)
        self.assertEqual(identical_navigation(self.actions[1:],self.units),self.actions[1:])

    def test_move_requires_move_order_and_exact_pair(self):
        actions=[self.actions[0],dict(self.actions[1],type='move')]
        self.assertEqual(identical_navigation(actions,self.units),actions)
        self.units['1']['order']=2
        self.assertEqual(identical_navigation(actions,self.units),[])
        mismatch=[dict(actions[0],units=['2']),actions[1]]
        self.assertEqual(identical_navigation(mismatch,self.units),mismatch)

    def test_env_filter_is_opt_in_and_records_actual_sent_actions(self):
        for enabled in (False, True):
            env=object.__new__(ToolsEnv)
            env.me=1;env.state=SimpleNamespace(tick=20);env.decisions=1
            env.obs={'units':[dict(self.units['1'],id='1',army=1)]}
            env.deduplicate_navigation=enabled;env.navigation_repeats_suppressed=0
            env.decision_ticks=2;env.rallied_by_player=set();env.claimed={}
            env.observer=SimpleNamespace(taken=set())
            env.kit=SimpleNamespace(plan=Mock(return_value=(self.actions,[],[])),role={})
            env.fa=SimpleNamespace(server=None,capture=False,step=Mock(return_value=({},0,False,False,{})))
            env._finish=Mock(return_value='finished');env._tools_result=Mock()
            commands=[dict(verb='attack_move',subject=['1'],cell=[2,3])]
            env.step_tools(commands)
            expected=[] if enabled else self.actions
            self.assertEqual(env.fa.step.call_args.args[0],expected)
            self.assertEqual(env._tools_result.call_args.args[1],commands)
            self.assertEqual(env._tools_result.call_args.args[2],expected)
            self.assertEqual(env.navigation_repeats_suppressed,int(enabled))


if __name__=='__main__':unittest.main()
