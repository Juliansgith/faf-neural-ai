import unittest
from fa_env.tools_obs import ToolsObserver,CAND_MOBILE

class UnfinishedFactoryTest(unittest.TestCase):
    def observer(self):
        o=ToolsObserver.__new__(ToolsObserver)
        o.bps={'factory':{'categories':['FACTORY'],'tech':1,'buildable':['unit','upgrade']},
               'engineer':{'categories':['ENGINEER'],'tech':1,'buildable':['structure']},
               'unit':{'tech':1},'upgrade':{'tech':1,'categories':['STRUCTURE']},
               'structure':{'tech':1,'categories':['STRUCTURE']}}
        o.place_class={'unit':CAND_MOBILE,'upgrade':0,'structure':0}
        return o
    def test_near_complete_factory_can_queue_units_but_not_upgrade(self):
        o=self.observer()
        for fraction in [.9,.95,.999999]:
            self.assertEqual(o.can_build_of({'bp':'factory','frac':fraction}),['unit'])
        self.assertEqual(o.can_build_of({'bp':'factory','frac':1}),['unit','upgrade'])
    def test_unfinished_engineers_and_early_factory_stay_blocked(self):
        o=self.observer()
        for fraction in [0,.5,.8999]:self.assertEqual(o.can_build_of({'bp':'factory','frac':fraction}),[])
        self.assertEqual(o.can_build_of({'bp':'engineer','frac':.99}),[])
        self.assertEqual(o.can_build_of({'bp':'engineer','frac':1}),['structure'])
    def test_engine_reported_restrictions_are_preserved(self):
        o=self.observer()
        self.assertEqual(o.can_build_of({'bp':'factory','frac':.95,'can_build':[]}),[])
        self.assertEqual(o.can_build_of({'bp':'factory','frac':.95,'can_build':['upgrade']}),[])
        self.assertEqual(o.can_build_of({'bp':'factory','frac':.95,'can_build':['unit']}),['unit'])

if __name__=='__main__':unittest.main()
