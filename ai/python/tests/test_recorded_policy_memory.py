"""Real neural rollouts need played-command memory without a BC label-source flag."""
import unittest
from fa_train.tools_model import UNIT_KEYS, VERBS
import test_policy_recording_labels as fixtures


class RecordedPolicyMemoryTest(unittest.TestCase):
    encode = fixtures.PolicyRecordingLabelsTest.encode
    def step(self, commands):
        values = {key: -1 for key in UNIT_KEYS}
        values.update(uverb=VERBS.index('fire_state'),uoption=0)
        return {'commands':commands,'sent':[],'results':[{'ok':False}],
                'policy':{'action':{'units':[[0,*[values[k] for k in UNIT_KEYS]]]},
                          'unit_logp':[0.0],'logp':0.0}}

    def test_neural_record_without_bc_flag_replays_actual_intent_and_results(self):
        command = {'verb':'fire_state','subject':['u0'],'option':0}
        _,_,observer = self.encode([self.step([command])],policy=False)
        observer.note_commands.assert_called_once_with([command],0,[{'ok':False}])

    def test_capture_without_policy_does_not_infer_played_intent_from_commands(self):
        command = {'verb':'fire_state','subject':['u0'],'option':0}
        _,_,observer = self.encode([{'commands':[command],'sent':[],'results':[]}],policy=False)
        observer.note_commands.assert_called_once_with([],0)

    def test_neural_record_missing_played_commands_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'played commands'):
            self.encode([self.step(None)],policy=False)


if __name__=='__main__':
    unittest.main()
