"""Engineer-built mobiles need spatial build labels and IssueBuildMobile."""
import unittest
import numpy as np

from fa_env.tools import ToolKit, VERBS
from fa_env.tools_obs import ToolsObserver, CAND_MOBILE
from fa_train.tools_bc import command_rejection_reason, lookups, unit_labels, policy_arrays
from fa_train.tools_model import UNIT_KEYS
from observer_fixture import info, unit, obs


class MobileConstructionTest(unittest.TestCase):
    def setUp(self):
        self.bps = {
            'eng': {'categories':['MOBILE','ENGINEER','LAND'], 'tech':3},
            'acu': {'categories':['MOBILE','ENGINEER','COMMAND','LAND'], 'tech':3},
            'factory': {'categories':['STRUCTURE','FACTORY','LAND'], 'tech':3},
            'tank': {'categories':['MOBILE','LAND']},
            'land_exp': {'categories':['MOBILE','LAND','NEEDMOBILEBUILD','EXPERIMENTAL']},
            'air_exp': {'categories':['MOBILE','AIR','NEEDMOBILEBUILD','EXPERIMENTAL']},
            'sonar': {'categories':['MOBILE','NAVAL','SONAR','NEEDMOBILEBUILD']},
            'upgrade': {'categories':['STRUCTURE','FACTORY']},
        }
        self.kit = ToolKit(info(), self.bps)

    def observe(self, builder, choices, water=False):
        ob = ToolsObserver(info(), self.bps, 1, self.kit)
        ob.water[:] = water
        ob.buildable[:] = not water
        u = unit('u', builder, 24,24, idle=True, order=0, queue=0, can_build=choices)
        return ob.observe(obs([u])), {'u':u}

    def test_placed_mobile_teacher_orders_are_scored_build_labels(self):
        for builder in ('eng','acu'):
            for bp,kind in [('land_exp',0),('air_exp',0),('sonar',1)]:
                with self.subTest(builder=builder,bp=bp):
                    x,units = self.observe(builder,[bp],water=kind==1)
                    i = x['ids'].index('u'); j = x['cand_ids'].index(bp)
                    self.assertEqual(int(x['cand_kind'][j]),kind)
                    self.assertTrue(x['mask_subject'][VERBS.index('build'),i])
                    self.assertFalse(x['mask_subject'][VERBS.index('produce'),i])
                    command, = self.kit.commands_from_actions(
                        [dict(type='build',units=['u'],bp=bp,pos=[40,40])],units)
                    self.assertIsNone(command_rejection_reason(command,x,lookups(x)))
                    labels = unit_labels([command],x,lookups(x))
                    self.assertEqual(int(labels['uverb'][i]),VERBS.index('build'))
                    self.assertEqual(int(labels['ubp'][i]),j)
                    actions,queries,pending = self.kit.plan([command],units,set())
                    self.assertEqual(actions,[])
                    self.assertTrue(queries)
                    completed = self.kit.complete(pending,[dict(ok=True,value=[40,40])]*len(queries))
                    self.assertEqual(completed[-1],dict(type='build',units=['u'],bp=bp,pos=[40,40]))

    def test_bad_produce_cannot_emit_factory_order_for_placed_mobile(self):
        for bp in ('land_exp','air_exp','sonar'):
            with self.subTest(bp=bp):
                x,units = self.observe('eng',[bp],water=bp=='sonar')
                actions,queries,pending = self.kit.plan([dict(verb='produce',subject=['u'],bp=bp,count=1)],units,set())
                self.assertEqual((actions,queries,pending),([],[],[]))

    def test_factory_output_and_upgrade_keep_existing_contract(self):
        x,units = self.observe('factory',['tank','upgrade'])
        self.assertEqual(int(x['cand_kind'][x['cand_ids'].index('tank')]),CAND_MOBILE)
        self.assertEqual(int(x['cand_kind'][x['cand_ids'].index('upgrade')]),0)
        for verb,bp in [('produce','tank'),('upgrade','upgrade')]:
            c=dict(verb=verb,subject=['u'],bp=bp,count=1)
            self.assertIsNone(command_rejection_reason(c,x,lookups(x)))
            actions,queries,pending=self.kit.plan([c],units,set())
            self.assertEqual(actions[0]['type'],'factory_build' if verb=='produce' else 'upgrade')
            self.assertEqual((queries,pending),([],[]))

    def test_naval_mobile_stays_unavailable_without_water(self):
        x,_=self.observe('eng',['sonar'])
        self.assertEqual(int(x['cand_kind'][0]),1)
        self.assertEqual(len(x['can_build']),0)
        i=x['ids'].index('u')
        self.assertFalse(x['mask_subject'][VERBS.index('build'),i])
        self.assertFalse(x['mask_subject'][VERBS.index('produce'),i])

    def test_missing_engine_build_permission_stays_unavailable(self):
        x,_=self.observe('eng',[])
        self.assertEqual(len(x['cand_ids']),0)
        self.assertEqual(len(x['can_build']),0)


class RecordedConstructionCompatibilityTest(unittest.TestCase):
    def arrays(self,verb,kind,index=0):
        entry=[0]+[-1]*len(UNIT_KEYS)
        entry[1+UNIT_KEYS.index('uverb')]=VERBS.index(verb)
        entry[1+UNIT_KEYS.index('ubp')]=index
        policy=dict(action=dict(units=[entry]),unit_logp=[-.7],logp=-.7,value=.2)
        step=dict(policy=policy,transition=dict(reward_components=dict(terminal=.3)))
        return policy_arrays([dict(step=step)],[dict(ids=['u'],cand_kind=np.array([kind]))])

    def test_old_factory_order_for_now_placed_unit_is_not_a_ppo_sample(self):
        result=self.arrays('produce',0)
        self.assertFalse(result['pol_has'][0])
        self.assertEqual(result['pol_uverb'].tolist(),[0])
        self.assertEqual(result['pol_unit_logp'].tolist(),[0])
        self.assertAlmostEqual(float(result['rew_terminal'][0]),.3)

    def test_valid_build_and_factory_samples_keep_recorded_probability(self):
        for verb,kind in [('produce',CAND_MOBILE),('build',0),('build',1),('build',2),('build',3)]:
            with self.subTest(verb=verb,kind=kind):
                result=self.arrays(verb,kind)
                self.assertTrue(result['pol_has'][0])
                self.assertAlmostEqual(float(result['pol_unit_logp'][0]),-.7)

    def test_wrong_build_kind_and_missing_candidate_are_unusable(self):
        for verb,kind,index in [('build',CAND_MOBILE,0),('produce',CAND_MOBILE,-1),('build',0,1)]:
            with self.subTest(verb=verb,kind=kind,index=index):
                self.assertFalse(self.arrays(verb,kind,index)['pol_has'][0])


if __name__=='__main__':
    unittest.main()
