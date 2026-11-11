"""Expert command-history overflow must not inherit the slots label limit."""
from pathlib import Path
import unittest
from unittest import mock
import numpy as np

from fa_train import tools_bc as bc
from test_tools_bc import tiny_observation


class ExpertHistoryTest(unittest.TestCase):
    def encode(self,captured):
        count=bc.K+3
        x=tiny_observation(count)
        x.update(tokens=np.zeros((count,70),np.float32),token_type=np.zeros(count,np.int64),
                 token_order=np.zeros(count,np.int64),load_subject_class=np.zeros(count,np.int64),
                 load_target_class=np.zeros(count,np.int64),mask_target=np.zeros((5,count),bool))
        commands=[{'verb':'fire_state','subject':[uid],'option':0} for uid in x['ids']]
        hello={'hello':{'map':{'width':256,'height':256}},'meta':{'me':1}}
        if captured:hello['meta'].update(capture=True,shadow=False,player_ai='m28ai')
        observer=mock.Mock();observer.wrecks=[];observer.observe.return_value=x
        rows=[{'step':{'tick':0,'sent':[]},'msg':{'obs':{'tick':0,'units':[]}}}]
        labels=[]
        def pack(xs,labs):labels.extend(labs);return {}
        with mock.patch.object(bc,'read_game',return_value=(hello,rows,{'outcome':'victory'})), \
             mock.patch.object(bc,'ToolsObserver',return_value=observer), \
             mock.patch.object(bc.ToolKit,'commands_from_actions',return_value=commands), \
             mock.patch.object(bc,'pack',side_effect=pack):
            arrays,stats=bc.encode_game(Path('unused'),{})
        return arrays,stats,labels,observer,commands

    def test_captured_expert_history_includes_overflow(self):
        arrays,stats,labels,observer,commands=self.encode(True)
        observer.note_commands.assert_called_once_with(commands,0)
        self.assertEqual(stats['command_diagnostics']['overflow_past_k'],3)
        self.assertEqual(len(labels[0]['verb']),bc.K)
        self.assertTrue(np.all(labels[0]['uverb']==bc.V_INDEX['fire_state']))

    def test_uncaptured_legacy_history_stays_capped(self):
        *_,observer,commands=self.encode(False)
        observer.note_commands.assert_called_once_with(commands[:bc.K],0)

    def test_labels_unchanged_by_history_branch(self):
        a=self.encode(False);b=self.encode(True)
        for key in a[2][0]:np.testing.assert_array_equal(a[2][0][key],b[2][0][key])
        self.assertEqual(a[1],b[1])

    def test_uncapped_history_keeps_legality_and_default_cap(self):
        x=tiny_observation(bc.K+2)
        commands=[{'verb':'fire_state','subject':[uid],'option':0} for uid in x['ids']]
        invalid={'verb':'fire_state','subject':['missing'],'option':0}
        kept,diag=bc.filter_commands(commands+[invalid],x,cap=False)
        self.assertEqual(kept,commands)
        self.assertEqual(diag['overflow_past_k'],0)
        self.assertEqual(sum(diag['rejections'].values()),1)
        default,diag=bc.filter_commands(commands+[invalid],x)
        self.assertEqual(default,commands[:bc.K])
        self.assertEqual(diag['overflow_past_k'],2)


if __name__=='__main__':unittest.main(verbosity=2)
