import tempfile
import unittest
from pathlib import Path
import torch
from fa_train.validation_checkpoint import checkpoint_then_validate


class ValidationCheckpointTest(unittest.TestCase):
    def test_failed_validation_preserves_weights_and_optimizer_without_selecting(self):
        with tempfile.TemporaryDirectory() as folder:
            net=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(net.parameters())
            net(torch.ones(1,2)).sum().backward();optimizer.step()
            p=Path(folder)/'unvalidated_step_000500.pt'
            def fail():
                self.assertTrue(p.exists())
                raise RuntimeError('validation CUDA failure')
            with self.assertRaisesRegex(RuntimeError,'validation CUDA failure'):
                checkpoint_then_validate(p,net,optimizer,{'completed_steps':500},fail)
            saved=torch.load(p,map_location='cpu',weights_only=False)
            self.assertEqual(saved['validation_status'],'pending')
            self.assertEqual(saved['completed_steps'],500)
            self.assertTrue(saved['optimizer']['state'])
            for name,value in net.state_dict().items():torch.testing.assert_close(saved['model'][name],value)
            self.assertFalse((Path(folder)/'best.pt').exists())
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_existing_recovery_is_preserved_and_callback_not_run(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'saved.pt';p.write_bytes(b'prior evidence')
            net=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(net.parameters())
            with self.assertRaises(FileExistsError):
                checkpoint_then_validate(p,net,optimizer,{},lambda:self.fail('must not validate'))
            self.assertEqual(p.read_bytes(),b'prior evidence')


if __name__=='__main__':unittest.main()
