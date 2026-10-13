"""CPU-only load/save regressions; no training or game construction."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from fa_train.checkpoint_contract import runtime_set_distribution, validate_checkpoint_set_distribution
from fa_train.tools_trainer import ToolsConfig, ToolsTrainer

CONTRACT={"acting":"conditional_nonempty_v1","set_objective":"sum_bce","private_sources":{"decoder.py":"abc"}}

class Independent(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.]))

class Conditional(Independent):
    set_distribution_contract=CONTRACT

def trainer(kind=Independent):
    t=object.__new__(ToolsTrainer)
    t.cfg=ToolsConfig(run_dir="unused",distill_teacher="none")
    t.net,t.acting=kind(),kind();t.optimizer=torch.optim.Adam(t.net.parameters(),lr=1e-3)
    t.device=torch.device('cpu');t.update=0;t.episodes=0;t.stage=0;t.stages=[{}]
    t.rms=SimpleNamespace(mean=0,var=1,count=1);t.entropy_ref=None;t.thread=None;t._log_event=Mock()
    return t

class CheckpointContractTest(unittest.TestCase):
    def load(self,t,state,optimizer=True):
        with patch('fa_train.tools_trainer.torch.load',return_value=state),contextlib.redirect_stdout(io.StringIO()):
            t.load('unused.pt',optimizer=optimizer)

    def test_wrong_decoder_fails_before_weights_or_optimizer_even_without_resume(self):
        for optimizer in [False,True]:
            t=trainer();state={'model':{'weight':torch.tensor([9.])},'set_distribution':CONTRACT,'optimizer':{'bad':'state'}}
            with self.assertRaisesRegex(ValueError,'SET distribution mismatch'):self.load(t,state,optimizer)
            self.assertEqual(float(t.net.weight),1);self.assertEqual(float(t.acting.weight),1)
            self.assertEqual(t.optimizer.state,{})

    def test_legacy_independent_checkpoint_load_is_unchanged(self):
        t=trainer();self.load(t,{'model':{'weight':torch.tensor([7.])},'update':3})
        self.assertEqual(float(t.net.weight),7);self.assertEqual(float(t.acting.weight),7);self.assertEqual(t.update,3)

    def test_matching_explicit_contract_loads_and_survives_save(self):
        t=trainer(Conditional);self.load(t,{'model':{'weight':torch.tensor([4.])},'set_distribution':CONTRACT})
        with tempfile.TemporaryDirectory() as root:
            t.run_dir=Path(root);(t.run_dir/'checkpoints').mkdir()
            saved=t.save('model.pt');state=torch.load(saved,map_location='cpu',weights_only=False)
            self.assertEqual(state['set_distribution'],CONTRACT)
            restored=trainer(Conditional);self.load(restored,state)
            self.assertEqual(float(restored.net.weight),4)
            with self.assertRaisesRegex(ValueError,'SET distribution mismatch'):self.load(trainer(),state)

    def test_missing_contract_cannot_load_into_alternative_decoder(self):
        with self.assertRaisesRegex(ValueError,'SET distribution mismatch'):
            self.load(trainer(Conditional),{'model':{'weight':torch.tensor([4.])}})

    def test_actor_learner_and_source_objective_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'Actor and learner'):runtime_set_distribution(Independent(),Conditional())
        for delta in [{'set_objective':'conditional_nll'},{'private_sources':{'decoder.py':'changed'}}]:
            with self.assertRaisesRegex(ValueError,'SET distribution mismatch'):
                validate_checkpoint_set_distribution({'set_distribution':{**CONTRACT,**delta}},Conditional(),Conditional())

    def test_malformed_contract_rejected_and_result_is_detached(self):
        for value in [{},'conditional_nonempty_v1',{'acting':None}]:
            with self.assertRaisesRegex(ValueError,'Invalid SET distribution'):
                validate_checkpoint_set_distribution({'set_distribution':value},Independent(),Independent())
        result=runtime_set_distribution(Conditional(),Conditional());result['private_sources']['decoder.py']='mutated'
        self.assertEqual(CONTRACT['private_sources']['decoder.py'],'abc')

    def test_save_rejects_actor_learner_mismatch_before_writing(self):
        t=trainer();t.acting=Conditional()
        with tempfile.TemporaryDirectory() as root:
            t.run_dir=Path(root);(t.run_dir/'checkpoints').mkdir()
            with self.assertRaisesRegex(ValueError,'Actor and learner'):t.save('bad.pt')
            self.assertEqual(list((t.run_dir/'checkpoints').iterdir()),[])

if __name__=='__main__':unittest.main()
