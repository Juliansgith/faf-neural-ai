"""Token caps keep what matters and the planes count every unit (2026-10-17 review, B19 / architecture F3)."""
import unittest

import numpy as np

from fa_env.tools import ToolKit
from fa_env.tools_obs import CAPS, MAP_CHANNELS, T_ENEMY, T_GHOST, T_OWN, ToolsObserver
from observer_fixture import BPS, ENEMY, ME, info, obs, observer, unit


class TokenCapsTest(unittest.TestCase):
    def test_over_the_cap_the_nearest_enemies_get_tokens_and_the_planes_count_all(self):
        ob = observer(1024.0)
        n = CAPS[T_ENEMY] + 40
        far = [unit(f"f{k}", "tank", 900, 900, army=ENEMY) for k in range(n - 10)]       # first in list order
        near = [unit(f"n{k}", "tank", 60, 60, army=ENEMY) for k in range(10)]
        x = ob.observe(obs([unit("1", "tank", 50, 50)] + far + near))
        enemy_ids = [i for i, t in zip(x["ids"], x["token_type"]) if t == T_ENEMY]
        self.assertEqual(len(enemy_ids), CAPS[T_ENEMY])
        self.assertTrue(all(f"n{k}" in enemy_ids for k in range(10)))
        mobile = x["map"][MAP_CHANNELS.index("enemy_mobile")]
        self.assertAlmostEqual(float(np.expm1(mobile).sum()), n, places=2)
        gx, gz = ob.kit.cell(60, 60)
        self.assertGreater(x["map"][MAP_CHANNELS.index("enemy_ground_fire"), gz, gx], 0.0)

    def test_under_the_cap_the_order_is_the_bridges(self):
        ob = observer()
        enemies = [unit(f"e{k}", "tank", 200 - k, 200, army=ENEMY) for k in range(5)]
        x = ob.observe(obs([unit("1", "tank", 50, 50)] + enemies))
        self.assertEqual([i for i, t in zip(x["ids"], x["token_type"]) if t == T_ENEMY], [f"e{k}" for k in range(5)])

    def test_own_structures_that_take_no_orders_go_last(self):
        ob = observer(1024.0)
        walls = [unit(f"w{k}", "wall", 100 + k % 50, 100 + k // 50) for k in range(CAPS[T_OWN])]
        tanks = [unit(f"t{k}", "tank", 300, 300) for k in range(20)]
        x = ob.observe(obs(walls + tanks))
        own = [i for i, t in zip(x["ids"], x["token_type"]) if t == T_OWN]
        self.assertEqual(len(own), CAPS[T_OWN])
        self.assertTrue(all(f"t{k}" in own for k in range(20)))

    def test_the_enemys_remembered_structures_before_hostile_civilians(self):
        i = info(1024.0)
        i["armies"].append({"index": 4, "start": [0.0, 0.0], "relation": "enemy", "civilian": True})
        ob = ToolsObserver(i, BPS, ME, ToolKit(i, BPS))
        o = obs([unit("1", "tank", 50, 50)] +
                [unit(f"h{k}", "wall", 60, 60, army=4, ghost=True) for k in range(CAPS[T_GHOST])] +
                [unit(f"g{k}", "pgen", 900, 900, army=ENEMY, ghost=True) for k in range(5)])
        o["armies"].append({"index": 4, "relation": "enemy", "civilian": True})
        x = ob.observe(o)
        ghosts = [u for u, t in zip(x["ids"], x["token_type"]) if t == T_GHOST]
        self.assertTrue(all(f"g{k}" in ghosts for k in range(5)))


if __name__ == "__main__":
    unittest.main()
