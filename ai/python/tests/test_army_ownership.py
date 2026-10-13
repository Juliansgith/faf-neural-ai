import unittest
from fa_env.bridge import GameConfig
from fa_env.env import GameError, validate_army_ownership


class ArmyOwnershipTests(unittest.TestCase):
    def info(self, brain='Human', civilian=False):
        return {'armies': [{'index': 1, 'brainType': brain, 'civilian': civilian}]}

    def test_neural_army_requires_human_even_with_ai_opponent(self):
        cfg = GameConfig(ai='easy', slots=(2, 1))
        validate_army_ownership(self.info(), cfg, [1])
        with self.assertRaises(GameError):
            validate_army_ownership(self.info('AI'), cfg, [1])

    def test_explicit_teacher_allows_ai_but_not_human(self):
        cfg = GameConfig(player_ai='m28ai')
        validate_army_ownership(self.info('AI'), cfg, [1])
        with self.assertRaises(GameError):
            validate_army_ownership(self.info(), cfg, [1])

    def test_missing_and_civilian_fail_closed(self):
        for info in [{}, self.info(civilian=True), self.info(brain=None)]:
            with self.assertRaises(GameError):
                validate_army_ownership(info, GameConfig(), [1])

    def test_explicit_all_ai_observation(self):
        validate_army_ownership(self.info('AI'), GameConfig(extra_args=['/agentallai']), [1])


if __name__ == '__main__':
    unittest.main()
