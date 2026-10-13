import unittest
import numpy as np
from fa_env.tools import ToolKit, GRID
from fa_env.build_cancellation import add_build_cancellations
from fa_train.tools_bc import filter_commands, K
from fa_train.tools_model import commands_to_indices, indices_to_commands
from test_tools_bc import tiny_observation


class BuildCancellationTest(unittest.TestCase):
    def setUp(self):
        self.kit = ToolKit({'map': {'width':256,'height':256,'deposits':[]}}, {
            'acu': {'categories':['COMMAND','MOBILE','ENGINEER']},
            'factory': {'categories':['STRUCTURE','FACTORY']}})
        self.units = {'u0':{'bp':'acu','order':8,'queue':1,'x':0,'z':0,'can_build':['factory']}}
        self.raw = [{'type':'clear','units':['u0']},
                    {'type':'build','units':['u0'],'bp':'factory','pos':[20,20]}]
        self.x = tiny_observation(K + 1)
        self.x.update(cand_ids=['factory'],cand_kind=np.array([0]),
                      mask_build_cell=np.ones((4,GRID,GRID),bool),
                      can_build=np.array([[0,0]]),enh_names=[],can_enh=np.empty((0,2),int))

    def test_busy_builders_clear_and_build_is_a_queued_build(self):
        # since format 7: a unit at a build finishes it, so an AI's clear + build there is a queued build
        commands = self.kit.commands_from_actions(self.raw,self.units)
        self.assertEqual([c['verb'] for c in commands],['build'])
        kept,diag=filter_commands(commands,self.x)
        self.assertEqual(diag['rejections'],{})
        labels=commands_to_indices(kept,self.x)
        decoded=indices_to_commands({k:v[None] for k,v in labels.items()},0,self.x)
        actions,queries,pending=self.kit.plan(decoded,self.units,set())
        self.assertEqual([a['type'] for a in actions],[])
        self.assertEqual(len(queries),1)
        self.assertEqual([a['type'] for a in self.kit.complete(pending,[{'ok':True,'value':[20,20]}])],['build'])

    def test_append_build_and_idle_clear_do_not_add_stop(self):
        self.assertEqual([c['verb'] for c in self.kit.commands_from_actions(self.raw[1:],self.units)],['build'])
        idle={'u0':{**self.units['u0'],'order':0,'queue':0}}
        self.assertEqual([c['verb'] for c in self.kit.commands_from_actions(self.raw,idle)],['build'])

    def test_existing_replacement_navigation_needs_no_extra_stop(self):
        cmds=[{'verb':'commander_move','subject':['u0'],'cell':(1,1)},
              {'verb':'build','subject':['u0'],'bp':'factory','cell':(2,2)}]
        self.assertEqual(add_build_cancellations(cmds,self.raw,self.units,{8},{'commander_move','build'}),cmds)

    def test_invalid_build_is_dropped_alone(self):
        commands=self.kit.commands_from_actions(self.raw,self.units)
        commands[0]['bp']='unavailable'
        kept,diag=filter_commands(commands,self.x)
        self.assertEqual(kept,[])
        self.assertEqual(diag['rejections'],{'blueprint_missing':1})

    def test_stop_labels_only_units_not_at_a_build(self):
        self.assertEqual(self.kit.commands_from_actions([{'type':'stop','units':['u0']}],self.units),[])
        idle={'u0':{**self.units['u0'],'order':0,'queue':0}}
        self.assertEqual(self.kit.commands_from_actions([{'type':'stop','units':['u0']}],idle),
                         [{'verb':'stop','subject':['u0']}])

if __name__=='__main__':unittest.main()
