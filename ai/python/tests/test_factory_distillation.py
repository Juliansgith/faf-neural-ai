import unittest

import torch

from fa_env.tools import VERBS
from fa_train.learn_rl import idle_factory_distillation_labels


class FactoryDistillationTests(unittest.TestCase):
    def batch(self, n=1):
        tok = torch.zeros(1, n, 68)
        tok[..., 15] = 1
        scalars = torch.zeros(1, 40)
        scalars[:, [1, 6]] = .3
        scalars[:, [2, 7]] = 1
        masks = torch.zeros(1, len(VERBS), n, dtype=torch.bool)
        masks[:, VERBS.index('produce')] = True
        return dict(tokens=tok, scalars=scalars, mask_subject=masks,
                    token_mask=torch.ones(1,n,dtype=torch.bool), token_type=torch.zeros(1,n,dtype=torch.long))

    def test_default_is_exact_identity(self):
        labels = dict(uverb=torch.tensor([[0]]), uweight=torch.tensor([[2.]]))
        self.assertIs(idle_factory_distillation_labels(labels, {}, 1), labels)

    def test_only_eligible_end_changes_and_preserves_existing_weights(self):
        b = self.batch(8)
        labels = dict(uverb=torch.tensor([[0, 0, 0, 0, 0, 0, 2, -1]]), uweight=torch.full((1,8),2.))
        b['tokens'][0,1,15] = 0  # busy
        b['tokens'][0,2,17] = .1  # queued work
        b['token_type'][0,3] = 1  # enemy
        b['token_mask'][0,4] = False  # padding
        b['mask_subject'][0,VERBS.index('produce'),5] = False  # no legal production
        out = idle_factory_distillation_labels(labels,b,.25)
        torch.testing.assert_close(out['uweight'], torch.tensor([[.5,2,2,2,2,2,2,2]]))
        self.assertIs(out['uverb'],labels['uverb'])
        torch.testing.assert_close(labels['uweight'],torch.full((1,8),2.))

    def test_scarcity_unit_cap_and_unknown_capacity_keep_end_weight(self):
        for index,value in ((1,.24),(6,.24),(13,.95),(2,0),(7,0)):
            with self.subTest(index=index,value=value):
                b=self.batch();b['scalars'][0,index]=value
                out=idle_factory_distillation_labels(dict(uverb=torch.zeros(1,1,dtype=torch.long)),b,.25)
                self.assertEqual(float(out['uweight'][0,0]),1.)

    def test_selected_end_gradient_scales_without_forcing_a_production_label(self):
        b=self.batch(2)
        labels=dict(uverb=torch.tensor([[0,2]]))
        logits=torch.zeros(1,2,3,requires_grad=True)
        losses=-logits.log_softmax(-1).gather(-1,labels['uverb'].unsqueeze(-1)).squeeze(-1)
        original,=torch.autograd.grad(losses.sum(),logits,retain_graph=True)
        weighted=idle_factory_distillation_labels(labels,b,.25)
        changed,=torch.autograd.grad((losses*weighted['uweight']).sum(),logits)
        torch.testing.assert_close(changed[:,0],.25*original[:,0],rtol=0,atol=0)
        torch.testing.assert_close(changed[:,1],original[:,1],rtol=0,atol=0)
        self.assertEqual(labels['uverb'][0,0],0)

    def test_invalid_weights_rejected(self):
        for weight in (-1,1.01,float('nan'),float('inf')):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                idle_factory_distillation_labels({}, {}, weight)


if __name__=='__main__':unittest.main()
