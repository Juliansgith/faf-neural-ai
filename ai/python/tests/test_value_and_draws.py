"""The timeout value advantage's income and draws (2026-10-17 review, bugs-env L8, L10)."""
import unittest
from collections import deque

from fa_env.factored import INCOME_WINDOW, FactoredEnv

ME, ENEMY, CIVILIAN = 1, 2, 3


def factored(me=ME, enemies=(ENEMY,)) -> FactoredEnv:
    env = FactoredEnv.__new__(FactoredEnv)
    env.me, env.enemy_indices = me, list(enemies)
    env.income_history = {}
    return env


def priv(income, army=1000.0) -> dict:
    return {"massIncome": income, "armyMass": army, "structureMass": 0.0, "engineerMass": 0.0}


class ValueAdvantageTest(unittest.TestCase):
    def test_a_last_decision_reclaim_burst_does_not_move_it(self):
        env = factored()
        for _ in range(INCOME_WINDOW):
            env._note_incomes({"privileged": {"1": priv(20.0), "2": priv(20.0)}})
        burst = {"privileged": {"1": priv(80.0), "2": priv(20.0)}}
        env._note_incomes(burst)
        self.assertAlmostEqual(env.value_advantage(burst), 0.0)
        self.assertEqual(len(env.income_history["1"]), INCOME_WINDOW)

    def test_without_history_the_current_income(self):
        env = factored()
        o = {"privileged": {"1": priv(30.0), "2": priv(10.0)}}
        mine, theirs = 1000 + 60 * 30, 1000 + 60 * 10
        self.assertAlmostEqual(env.value_advantage(o), (mine - theirs) / (mine + theirs))
        env.income_history = {"1": deque([10.0]), "2": deque([10.0])}
        self.assertAlmostEqual(env.value_advantage(o), 0.0)


class DrawTest(unittest.TestCase):
    def test_both_defeated_is_a_draw(self):
        env = factored()
        both = {"armies": [{"index": ME, "relation": "self", "defeated": True},
                           {"index": ENEMY, "relation": "enemy", "defeated": True},
                           {"index": CIVILIAN, "relation": "neutral", "civilian": True}]}
        self.assertTrue(env._draw(both))
        lost = {"armies": [{"index": ME, "relation": "self", "defeated": True},
                           {"index": ENEMY, "relation": "enemy", "defeated": False}]}
        self.assertFalse(env._draw(lost))
        status = {"armies": [{"index": ME, "relation": "self", "defeated": True, "status": "Draw"},
                             {"index": ENEMY, "relation": "enemy", "defeated": False}]}
        self.assertTrue(env._draw(status))


if __name__ == "__main__":
    unittest.main()
