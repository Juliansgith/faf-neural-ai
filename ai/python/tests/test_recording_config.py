"""Raw recordings preserve launch settings needed for faithful diagnostics."""
import unittest
from unittest.mock import Mock

from fa_env.bridge import GameConfig
from fa_env.factored import FactoredEnv


class RecordingConfigTest(unittest.TestCase):
    def test_actual_begin_records_multiplier_slots_and_shadow_ownership(self):
        env = FactoredEnv(GameConfig(ai='m28aicheat', cheat=.9, slots=(2, 1), player_ai='m28ai'),
                          capture=True, shadow=True)
        env.recorder = Mock()
        for name in ('_init_blueprint_tables', '_decode', '_note_incomes'):
            setattr(env, name, Mock())
        for name in ('_trade', '_mass_income', '_acu_health', 'encode'):
            setattr(env, name, Mock(return_value=0))
        info = {'blueprints': [], 'armies': [
            {'index': 1, 'start': [10, 10], 'relation': 'own', 'civilian': False},
            {'index': 2, 'start': [90, 90], 'relation': 'enemy', 'civilian': False}],
            'map': {'width': 100, 'height': 100, 'deposits': []}}
        env._begin({'me': 1, 'tick': 0}, info)
        saved = env.recorder.start.call_args.args[1]
        self.assertEqual(saved['ai'], 'm28aicheat')
        self.assertEqual(saved['cheat'], .9)
        self.assertEqual(saved['slots'], [2, 1])
        self.assertEqual(saved['player_ai'], 'm28ai')
        self.assertTrue(saved['capture'])
        self.assertTrue(saved['shadow'])


if __name__ == '__main__':
    unittest.main()
