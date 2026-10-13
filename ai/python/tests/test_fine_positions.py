"""Fine positions: the pixel inside a chosen cell on maps above 5 km (fa_env.tools, tools_model._fine)."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools import FINE, ToolKit, fine_split  # noqa: E402
from fa_train import tools_model as tm  # noqa: E402
from test_units_decoder import batch  # noqa: E402


def kit(size: float) -> ToolKit:
    return ToolKit({"map": {"width": size, "height": size, "deposits": []}}, {})


class FineGeometryTest(unittest.TestCase):
    def test_every_map_gets_about_eight_units_a_pixel(self):
        self.assertEqual([fine_split(s / 32) for s in (256, 512, 1024, 2048, 4096)], [1, 2, 4, 8, 16])
        self.assertFalse(kit(256).has_fine)
        self.assertTrue(kit(1024).has_fine)

    def test_a_position_survives_cell_and_pixel(self):
        k = kit(1024)
        args = k.place(500.3, 260.9)
        self.assertEqual(args, {"cell": (15, 8), "sub": (2, 0)})
        x, z = k.where(args["cell"], args["sub"])
        self.assertLessEqual(abs(x - 500.3), 4.0)
        self.assertLessEqual(abs(z - 260.9), 4.0)
        self.assertEqual(k.where(args["cell"]), k.centre(args["cell"]))     # without the pixel: the cell, as before
        self.assertEqual(kit(256).place(100.0, 100.0), {"cell": (12, 12)})   # a 5 km map has no pixels

    def test_spots_are_the_pixels_on_a_map_with_them(self):
        k = kit(1024)
        x, z = 500.3, 260.9
        args = k.place(x, z)
        sx, sz = k.spot(x, z)
        bx, bz = k.where(args["cell"], args["sub"], (sx, sz))
        self.assertLessEqual(abs(bx - x), 1.0)                                # a quarter of an 8-unit pixel
        self.assertLessEqual(abs(bz - z), 1.0)

    def test_teacher_moves_keep_their_pixel_and_the_executor_goes_there(self):
        k = kit(1024)
        cmds = k._commands_from_actions([{"type": "move", "units": ["7"], "pos": [500.3, 260.9]}], {})
        self.assertEqual(cmds, [{"verb": "move", "subject": ["7"], "cell": (15, 8), "sub": (2, 0)}])
        actions, _, _ = k.plan(cmds, {"7": {"id": "7", "x": 10.0, "z": 10.0, "bp": "x"}}, set())
        pos = next(a["pos"] for a in actions if a["type"] == "move")
        self.assertEqual(tuple(pos), k.where((15, 8), (2, 0)))


class FineHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = tm.ToolsNet("S", decoder="units")
        self.net.candidates = lambda b: (torch.zeros(2, 1, self.net.d), torch.zeros(2, 1, self.net.d))
        for layer in (self.net.fine_q, self.net.fine_bias, self.net.fine_proj):
            torch.nn.init.normal_(layer.weight, std=0.5)                    # a trained-looking head
        self.net.act_min_p = 0.0
        self.b = batch()
        self.b["tokens"] = torch.rand(2, 3, tm.N_FEATURES)
        self.b["fine"] = torch.tensor([[4.0, 4.0, 1024.0, 1024.0], [1.0, 1.0, 256.0, 256.0]])
        self.inputs = (torch.randn(2, 3, self.net.d), torch.randn(2, self.net.map_ch, tm.GRID, tm.GRID),
                       torch.randn(2, self.net.core_dim), self.b)

    def test_pixels_on_the_large_map_only_and_scoring_agrees(self):
        self.net.fine_positions = True
        move = tm.VERBS.index("move")
        with torch.no_grad():
            for seed in range(6):
                torch.manual_seed(seed)
                acts, lp, _, _, _ = self.net.decode(*self.inputs, mode="sample")
                _, lp2, _, _, _ = self.net.decode(*self.inputs, actions=acts)
                torch.testing.assert_close(lp, lp2)
                moving = acts["uverb"] == move
                sub = acts["usub"]
                self.assertTrue(((sub >= 0) == moving)[0].all())               # the 20 km map: every move has one
                self.assertTrue((sub[1] == -1).all())                          # the 5 km map: none
                ok = sub[0][moving[0]]
                self.assertTrue(((ok % FINE) < 4).all() and ((ok // FINE) < 4).all())

    def test_a_network_without_the_trained_head_orders_cells(self):
        with torch.no_grad():
            acts, *_ = self.net.decode(*self.inputs, mode="sample")
        self.assertTrue((acts["usub"] == -1).all())

    def test_a_label_pixel_is_scored_where_it_is(self):
        self.net.fine_positions = True
        with torch.no_grad():
            for seed in range(50):                                           # a decision with a move on the large map
                torch.manual_seed(seed)
                acts, *_ = self.net.decode(*self.inputs, mode="sample")
                if (acts["usub"][0] >= 0).any():
                    break
            unit = int((acts["usub"][0] >= 0).nonzero()[0])
            unscored = {k: v.clone() for k, v in acts.items()}
            unscored["usub"][0, unit] = -1                                   # its pixel not given: not scored
            rest = self.net.decode(*self.inputs, actions=unscored)[1][0]
            scores = []
            for s in range(16):                                              # every pixel of the 4 x 4 cell
                a = {k: v.clone() for k, v in acts.items()}
                a["usub"][0, unit] = (s // 4) * FINE + s % 4
                scores.append(self.net.decode(*self.inputs, actions=a)[1][0])
            # the cell's 16 pixels hold all the probability: none of it leaks to the window around the cell
            torch.testing.assert_close(torch.stack(scores).sub(rest).exp().sum(), torch.tensor(1.0),
                                       atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
