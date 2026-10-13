import unittest
import torch
from fa_train.tools_model import _reduce_set_bc


class SetCloningLossTest(unittest.TestCase):
    def test_positive_gradient_not_diluted_by_extra_exclusions(self):
        gradients = []
        for n in (2, 100):
            logits = torch.zeros(1, n, requires_grad=True)
            target = torch.zeros(1, n, dtype=torch.bool)
            target[0, 0] = True
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
            _reduce_set_bc(loss, target, torch.ones_like(target), balanced=True).sum().backward()
            gradients.append(float(logits.grad[0, 0]))
            self.assertAlmostEqual(float(logits.grad[0, 1:].sum()), .25, places=6)
        self.assertEqual(gradients, [-.25, -.25])

    def test_absent_class_and_masked_tokens(self):
        losses = torch.tensor([[2., 4., 999.], [1., 3., 999.], [999., 999., 999.]])
        targets = torch.tensor([[True, True, False], [False, False, True], [True, False, True]])
        eligible = torch.tensor([[True, True, False], [True, True, False], [False, False, False]])
        actual = _reduce_set_bc(losses, targets, eligible, balanced=True)
        torch.testing.assert_close(actual, torch.tensor([3., 2., 0.]))

    def test_default_matches_original_mean(self):
        losses=torch.tensor([[2., 6., 17.]])
        targets=torch.tensor([[True, False, True]])
        eligible=torch.tensor([[True, True, False]])
        torch.testing.assert_close(_reduce_set_bc(losses,targets,eligible),torch.tensor([4.]))

if __name__ == '__main__': unittest.main()
