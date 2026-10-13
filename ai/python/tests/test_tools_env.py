"""Early economy measurements must count completion and avoid defeated snapshots."""
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools_env import ToolsEnv


class EconomySnapshotTest(unittest.TestCase):
    def test_live_reclaim_cache_clears_empty_scan_and_failed_target(self):
        env = object.__new__(ToolsEnv)
        env.observer = SimpleNamespace(wrecks=[{"id": "123", "mass": 50}])
        env.observer.set_wrecks = lambda props: setattr(env.observer, "wrecks", props)
        env.reclaim_cache = list(env.observer.wrecks)
        env._refresh_reclaim([{"type": "reclaim"}], [{"ok": True, "value": []}])
        self.assertEqual(env.observer.wrecks, [])
        self.assertEqual(env.reclaim_cache, [])
        env.observer.wrecks = [{"id": "123", "mass": 50}]
        env._refresh_reclaim([], [], [{"type": "reclaim", "target": "123"}],
                             [{"ok": False, "error": "unknown target 123"}])
        self.assertEqual(env.observer.wrecks, [])
        self.assertEqual(env.reclaim_cache, [])

    def env(self):
        env = object.__new__(ToolsEnv)
        env.me = 1
        env.kit = SimpleNamespace(role={"mex": "mex", "pgen": "pgen"})
        env.state = SimpleNamespace(acu={"id": "acu"})
        env.economy_snapshots = []
        env.recent_state_snapshots = deque(maxlen=31)
        env.obs = {"tick": 1800, "armies": [{"index": 1, "eco": {"mass": {"income": 6.0}}}],
                   "units": [{"id": 1, "army": 1, "bp": "mex", "frac": 1.0},
                             {"id": 2, "army": 1, "bp": "mex", "frac": 0.5},
                             {"id": 3, "army": 1, "bp": "pgen", "frac": 0.0},
                             {"id": 4, "army": 2, "bp": "mex", "frac": 1.0}]}
        return env

    def test_completion_threshold_income_and_no_duplicate_measurements(self):
        env = self.env()
        env.obs["tick"] = 1799
        env._note_economy_snapshot(False)
        self.assertEqual(env.economy_snapshots, [])
        env.obs["tick"] = 1810
        env._note_economy_snapshot(False)
        env._note_economy_snapshot(False)
        self.assertEqual(len(env.economy_snapshots), 1)
        sample = env.economy_snapshots[0]
        self.assertEqual((sample["minute"], sample["tick"], sample["mass_income"]), (3, 1810, 6.0))
        self.assertEqual(sample["completed_roles"], {"mex": 1})
        self.assertEqual(sample["building_blueprints"], {"mex": 1, "pgen": 1})
        env.obs["tick"] = 3000
        env._note_economy_snapshot(False)
        self.assertEqual([s["minute"] for s in env.economy_snapshots], [3, 5])

    def test_defeated_army_is_not_reported_as_a_zero_income_sample(self):
        env = self.env()
        env.state.acu = None
        env.obs["armies"][0]["eco"] = {}
        env._note_economy_snapshot(True)
        self.assertEqual(env.economy_snapshots, [])

    def test_recent_history_is_bounded_and_excludes_incomplete_and_enemy_units(self):
        env = self.env()
        env.state.acu = {"id": "acu", "hp": 700, "maxhp": 1000, "x": 4, "z": 8}
        for index in range(40):
            env.obs["tick"] = index * 20
            env._note_recent_state()
        self.assertEqual(len(env.recent_state_snapshots), 31)
        self.assertEqual(env.recent_state_snapshots[0]["tick"], 180)
        sample = env.recent_state_snapshots[-1]
        self.assertEqual(sample["commander"]["hp"], 700)
        self.assertEqual(sample["completed_role_activity"], {"mex": {"units": 1, "idle": 0, "orders": {"0": 1}}})


if __name__ == "__main__":
    unittest.main()
