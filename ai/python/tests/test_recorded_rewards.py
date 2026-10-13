import copy
import unittest
import numpy as np
from fa_train.recorded_rewards import recorded_returns


def row(tick, reward, done=False):
    return {"step": {"tick": tick, "transition": {"schema": "fa_tools_transition_v1",
        "start_tick": tick, "end_tick": tick+20, "done": done,
        "outcome": "victory" if done else None, "reward_components": reward}},
        "msg": {"obs": {"tick": tick}}}


class RecordedRewardsTest(unittest.TestCase):
    def setUp(self):
        self.steps = [row(2, {"terminal": 0., "trade": .5, "economy": .3, "commander": -.2}),
                      row(22, {"terminal": 13.5, "trade": .2, "economy": 0., "commander": -.4}, True)]

    def test_terminal_bonus_damage_and_discount_are_preserved(self):
        result = recorded_returns(self.steps, {"outcome": "victory"}, gamma=.5)
        np.testing.assert_allclose(result, [.6+.5*13.3, 13.3])
        result = recorded_returns(self.steps, {"outcome": "victory"}, gamma=.5, shaping=0)
        np.testing.assert_allclose(result, [6.75, 13.5])

    def test_legacy_is_explicit_but_partial_new_recording_is_rejected(self):
        self.assertIsNone(recorded_returns([{"step": {}}], {}, .99))
        with self.assertRaisesRegex(ValueError, "Partially"):
            recorded_returns(self.steps + [{"step": {}}], {"outcome": "victory"}, .99)

    def test_misalignment_nonfinite_and_outcome_mismatch_fail_closed(self):
        bad = copy.deepcopy(self.steps)
        bad[0]["step"]["transition"]["end_tick"] = 21
        with self.assertRaisesRegex(ValueError, "Misaligned"):
            recorded_returns(bad, {"outcome": "victory"}, .99)
        bad = copy.deepcopy(self.steps)
        bad[0]["step"]["transition"]["reward_components"]["trade"] = float('nan')
        with self.assertRaisesRegex(ValueError, "components"):
            recorded_returns(bad, {"outcome": "victory"}, .99)
        with self.assertRaisesRegex(ValueError, "outcome"):
            recorded_returns(self.steps, {"outcome": "defeat"}, .99)


if __name__ == '__main__':
    unittest.main()
