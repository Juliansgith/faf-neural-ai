"""Queued factory upgrade support, label selection and actor/scorer agreement."""
import copy
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

from observer_fixture import info, obs, unit
from test_air_refueling import world as air_world
from fa_env.tools import ToolKit
from fa_env.tools_obs import ToolsObserver, CAND_MOBILE
from fa_env.factory_upgrades import enabled_observation, enabled_batch, label_prefix
from fa_train import tools_bc as bc, tools_model as tm
from fa_train.learn_rl import chunk_batch


def world():
    _, hello, bps, units = air_world()
    bps.update(fac=dict(id='fac', categories=['STRUCTURE','FACTORY','LAND'],
                       upgradesTo='hq', mass=200, health=1000),
               hq=dict(id='hq', categories=['STRUCTURE','FACTORY','LAND','TECH2'], mass=800, health=2000))
    for uid, changes in [('ready',{}),('queued',dict(queue=2)),('upgrading',dict(order=27)),
                         ('unfinished',dict(frac=.8)),('enemy_factory',dict(army=2)),
                         ('empty',dict(queue=0)),('paused_factory',dict(paused=True))]:
        kw=dict(order=7,queue=1,idle=False,can_build=['tank','hq']);kw.update(changes)
        units.append(unit(uid,'fac',40,40,**kw))
    # Avoid reusing the air-staging fixture's unfinished-unit ID.
    units=[u for u in units if not (u['id']=='unfinished' and u['bp']=='stage')]
    kit=ToolKit(hello,bps);x=ToolsObserver(hello,bps,1,kit).observe(obs(units))
    return x,hello,bps,units


class FactoryUpgrades(unittest.TestCase):
    def test_only_safe_own_factory_queue_gains_support(self):
        x,*_=world();y=enabled_observation(x);upgrade=tm.VERBS.index('upgrade')
        eligible={x['ids'][i] for i in np.flatnonzero(y['mask_subject'][upgrade] & ~x['mask_subject'][upgrade])}
        self.assertEqual(eligible,{'ready','paused_factory'})
        self.assertFalse(x['mask_subject'][upgrade].any())
        for uid in ('ready','queued','upgrading','unfinished','enemy_factory','empty'):
            command=dict(verb='upgrade',subject=[uid],bp='hq')
            self.assertEqual(bc.command_allowed(command,y,bc.lookups(y),set()),uid=='ready')
        # With no upgrade target in the actual menu, no new support is legal.
        z=copy.deepcopy(x);pairs=z['can_build'];z['can_build']=pairs[z['cand_kind'][pairs[:,1]]==CAND_MOBILE]
        self.assertFalse(enabled_observation(z)['mask_subject'][upgrade].any())

    def test_numpy_and_tensor_masks_match_after_cache_rounding(self):
        x,*_=world();x['tokens']=x['tokens'].astype(np.float16).astype(np.float32)
        y=enabled_observation(x);b=tm.collate([x],'cpu');c=enabled_batch(b)
        np.testing.assert_array_equal(y['mask_subject'],c['mask_subject'][0].numpy())
        np.testing.assert_array_equal(y['mask_verb'],c['mask_verb'][0].numpy())
        torch.testing.assert_close(b['mask_subject'],tm.collate([x],'cpu')['mask_subject'],rtol=0,atol=0)

    def test_decoders_sample_upgrades_and_rescore_same_support(self):
        x,*_=world();b=tm.collate([x],'cpu');upgrade=tm.VERBS.index('upgrade')
        for decoder in ('units','slots'):
            with self.subTest(decoder=decoder):
                torch.manual_seed(71);net=tm.ToolsNet('S',decoder=decoder).eval()
                net.queued_factory_upgrades=True;net.act_min_p=0
                with torch.no_grad():
                    net.verb_head.weight.zero_();net.verb_head.bias.fill_(-50);net.verb_head.bias[upgrade]=50
                    net.verb_head.bias[tm.V_END]=0
                    state=net.initial_state(1,'cpu')
                    act,lp,_,_=net.forward_step(b,state,mode='sample')
                    scored,*_=net.evaluate(b,act,state,seq_len=1)
                torch.testing.assert_close(lp,scored,rtol=1e-5,atol=1e-5)
                self.assertTrue(torch.isfinite(lp).all())
                cmds=tm.actions_to_commands(act,0,x,decoder)
                self.assertTrue(cmds)
                for command in cmds:
                    self.assertEqual(command['verb'],'upgrade')
                    self.assertEqual(command['bp'],'hq')
                    self.assertTrue(bc.command_allowed(command,enabled_observation(x),bc.lookups(x),set()))
                if decoder=='units':
                    ids=[i for i in range(len(x['ids'])) if act['uverb'][0,i]==upgrade]
                    self.assertEqual({x['ids'][i] for i in ids},{'ready','paused_factory'})
                    for i in ids:self.assertNotEqual(int(x['cand_kind'][int(act['ubp'][0,i])]),CAND_MOBILE)
                else:
                    self.assertTrue((act['verb']==upgrade).any())

    def test_all_label_variants_and_ppo_keep_actual_actions(self):
        x,hello,bps,units=world();i=x['ids'].index('ready');air=x['ids'].index('air')
        labels=[dict(verb='upgrade',subject=['ready'],bp='hq'),dict(verb='load',subject=['air'],target='stage')]
        header=dict(hello=hello,meta=dict(me=1,faction=1,label_source='m28'))
        records=[dict(step=dict(tick=0,labels=labels,commands=[],sent=[]),msg=dict(obs=obs(units)))]
        with patch.object(bc,'read_game',return_value=(header,records,dict(outcome='victory',minutes=1))):
            arrays,_=bc.encode_game(None,bps)
        arrays.update({'pol_'+k:arrays[k].copy() for k in tm.UNIT_KEYS})
        arrays.update(pol_has=np.ones(1,bool),pol_unit_logp=np.zeros(len(x['ids']),np.float32),pol_logp=np.zeros(1,np.float32))
        games=object.__new__(bc.Games);games.games=[arrays];games.units=True;games.lengths=np.array([1])
        first=None
        for rf,fu in ((False,False),(True,False),(False,True),(True,True)):
            games.air_refueling=rf;games.queued_factory_upgrades=fu
            with self.subTest(refueling=rf,factory=fu):
                batch=chunk_batch(games,[(0,0)],1,'cpu');lab=games.batch([(0,0)],1,'cpu')[1]
                self.assertEqual(int(lab['uverb'][0,i])==tm.VERBS.index('upgrade'),fu)
                self.assertEqual(int(lab['uverb'][0,air])==tm.V_LOAD,rf)
                torch.testing.assert_close(lab['uverb'],batch['lab']['uverb'],rtol=0,atol=0)
                if first is None:first=batch
                for k in tm.UNIT_KEYS:torch.testing.assert_close(first['act'][k],batch['act'][k],rtol=0,atol=0)
                torch.testing.assert_close(first['old_ulp'],batch['old_ulp'],rtol=0,atol=0)
        # A mapped loader reconstructs its Games in another process: both
        # capability choices must survive that reconstruction, not just batch().
        games.air_refueling=False;games.queued_factory_upgrades=False
        stream=bc._Batches(['unused'],True,[0],1,0,1,0,
                           air_refueling=True,queued_factory_upgrades=True)
        with patch.object(bc.Games,'reopen',return_value=games):
            loaded=next(iter(stream))[1]
        self.assertEqual(int(loaded['uverb'][0,i]),tm.VERBS.index('upgrade'))
        self.assertEqual(int(loaded['uverb'][0,air]),tm.V_LOAD)

    def test_checkpoint_option_and_legacy_default(self):
        net=tm.ToolsNet('S');self.assertFalse(net.queued_factory_upgrades)
        self.assertFalse(tm.model_options(net)['queued_factory_upgrades'])
        net.queued_factory_upgrades=True;self.assertTrue(tm.model_options(net)['queued_factory_upgrades'])
        self.assertEqual(label_prefix(types.SimpleNamespace()),'')


if __name__=='__main__':unittest.main()
