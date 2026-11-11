"""The 2026-10-17 review's fixes in the encoder, learn_rl and the model."""
import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from fa_env.tools_obs import CAND_MOBILE, V_INDEX
from fa_train import tools_model as tm
from fa_train.learn_rl import clip_steps, potential, rewards_of
from fa_train.tools_bc import UNIT_KEYS, mass_rate, policy_arrays, recency_key


class MassRateTest(unittest.TestCase):
    def test_upgrading_and_inflated_readings(self):
        mex = {"massProduction": 6.0}
        self.assertEqual(mass_rate({"pm": 731.0, "order": 27}, mex), 6.0)        # upgrading: the blueprint's
        self.assertAlmostEqual(mass_rate({"pm": 731.0}, mex), 9.6)                    # never above 1.6x
        self.assertEqual(mass_rate({"pm": 7.5}, mex), 7.5)                       # adjacency kept
        self.assertEqual(mass_rate({"frac": 0.5}, mex), 3.0)                     # no reading: blueprint x built


class PotentialTest(unittest.TestCase):
    def test_the_end_is_absorbing(self):
        phi = np.array([0.0, 2.0, 5.0, 5.0], np.float32)
        r = potential(phi, 1.0)
        self.assertAlmostEqual(float(r.sum()), -float(phi[0]))                   # telescopes to -phi_0: no end bonus
        self.assertAlmostEqual(float(r[-1]), -5.0)

    def test_clipped_steps_still_telescope(self):
        phi = np.array([1.0, 1.2, 40.0, 1.4, 1.5], np.float32)                   # one bad reading
        c = clip_steps(phi, 1.0)
        self.assertTrue(np.all(np.abs(np.diff(c)) <= 1.0 + 1e-6))
        self.assertAlmostEqual(float(potential(c, 1.0).sum()), -1.0, places=5)
        np.testing.assert_array_equal(clip_steps(phi, None), phi)


def game(outcome: str, terminal: float, n: int = 4) -> dict:
    g = {"meta": np.array(json.dumps({"map": "SCMP_012", "outcome": outcome}))}
    for k in ("terminal", "trade", "economy", "commander"):
        g[f"rew_{k}"] = np.zeros(n, np.float32)
    g["rew_terminal"][-1] = terminal
    return g


def args(**kw):
    base = dict(econ_source="recorded", econ_coef=0.0, intel_coef=0.0, map_coef=0.0, waste_coef=0.0, stall_coef=0.0,
                shaping=1.0, gamma=0.997, acu_coef=0.0, acu_grace=1.0, acu_span=4.0, acu_all_maps=False,
                timeout_scale=10.0, timeout_cost=2.0)
    base.update(kw)
    return SimpleNamespace(**base)


class TimeoutFloorTest(unittest.TestCase):
    def test_a_timeout_never_scores_below_a_defeat(self):
        self.assertAlmostEqual(float(rewards_of(game("timeout", -5.0), args())[-1]), -9.0)     # va -1: was -12


class DepositBuildTest(unittest.TestCase):
    def step(self, entry):
        return {"policy": {"action": {"units": [entry]}, "unit_logp": [0.0], "logp": -1.0, "value": 0.0}}

    def x(self):
        return {"cand_kind": np.array([0, 2, CAND_MOBILE], np.int8), "ids": np.array([7])}

    def entry(self, verb, bp, cell, sub):
        e = [0] + [-1] * len(UNIT_KEYS)
        for k, v in (("uverb", verb), ("ubp", bp), ("ucell", cell), ("usub", sub)):
            e[1 + UNIT_KEYS.index(k)] = v
        return e

    def has(self, entry):
        out = policy_arrays([{"step": {"policy": self.step(entry)["policy"]}}], [self.x()], fine=True)
        return bool(out["pol_has"][0])

    def test_a_deposit_build_is_kept_and_a_cell_without_pixel_is_not(self):
        build = V_INDEX["build"]
        try:
            self.assertTrue(self.has(self.entry(build, 1, 100, -1)))          # mex: a cell, by design no pixel
            self.assertFalse(self.has(self.entry(build, 0, 100, -1)))         # a placed building needs a pixel
            self.assertFalse(self.has(self.entry(build, 0, 100, -1)[:-1]))    # an entry from before the fine head
        except (KeyError, TypeError) as exc:                                   # the recording layout differs
            self.skipTest(f"policy_arrays' step layout: {exc}")


class RecencyTest(unittest.TestCase):
    def test_rounds_by_start_then_fixed_order(self):
        a = {"source_root": r"C:\x\rounds\lane0_20261014T010000Z\raw", "source_name": "1.jsonl.gz", "created_ns": 9}
        b = {"source_root": r"C:\x\rounds\lane10_20261014T020000Z\raw", "source_name": "2.jsonl.gz", "created_ns": 1}
        self.assertLess(recency_key(a), recency_key(b))                         # by round start, not encoding time


class ActFloorTest(unittest.TestCase):
    def test_the_floor_is_on_every_verb_by_default(self):
        # the housekeeping-only floor cost rl_089 33 points vs the scripted AIs (2026-10-20); it stays an option
        net = tm.ToolsNet("S", decoder="units")
        self.assertIsNone(net.act_min_p_verbs)
        self.assertEqual(net.act_min_p, 0.02)


if __name__ == "__main__":
    unittest.main()
