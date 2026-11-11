import unittest

import torch

from fa_train.learn_rl import builder_distillation_labels


class BuilderDistillationTests(unittest.TestCase):
    def test_default_preserves_existing_labels_and_weights(self):
        labels = {'uverb': torch.tensor([[0, 1]]), 'uweight': torch.tensor([[.5, 3.]])}
        self.assertIs(builder_distillation_labels(labels, {}, 1), labels)

    def test_builder_orders_and_end_receive_same_gradient_multiplier(self):
        tokens = torch.zeros(1, 5, 68)
        tokens[0, [0, 1, 4], 67] = 1
        # Two builders with END/order labels, two army units, one unknown builder.
        labels = {'uverb': torch.tensor([[0, 1, 0, 1, -1]]),
                  'uweight': torch.tensor([[1., .5, 2., 1., 4.]])}
        logits = torch.zeros(1, 5, 2, requires_grad=True)
        raw = -logits.log_softmax(-1).gather(-1, labels['uverb'].clamp_min(0).unsqueeze(-1)).squeeze(-1)
        known = labels['uverb'] >= 0
        original = (raw * labels['uweight'] * known).sum()
        old_gradient, = torch.autograd.grad(original, logits, retain_graph=True)
        weighted = builder_distillation_labels(labels, {'tokens': tokens}, 2.)
        changed = (raw * weighted['uweight'] * known).sum()
        new_gradient, = torch.autograd.grad(changed, logits)
        torch.testing.assert_close(new_gradient[:, :2], 2 * old_gradient[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(new_gradient[:, 2:], old_gradient[:, 2:], rtol=0, atol=0)
        self.assertIs(weighted['uverb'], labels['uverb'])
        torch.testing.assert_close(labels['uweight'], torch.tensor([[1., .5, 2., 1., 4.]]))

    def test_rejects_nonpositive_or_nonfinite_weight(self):
        for weight in (0, -1, float('nan'), float('inf')):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                builder_distillation_labels({}, {}, weight)


if __name__ == '__main__':
    unittest.main()
