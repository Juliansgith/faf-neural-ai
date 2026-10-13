"""Close-ups (tools_model._close_ups): fine windows around our commander, our army and the fight."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.tools_obs import MAP_CHANNELS, N_FEATURES, N_FOCUS, N_SCALARS  # noqa: E402
from fa_train import tools_model as tm  # noqa: E402


def observation(batch: int = 2, tokens: int = 4, size: float = 1024.0) -> dict:
    torch.manual_seed(1)
    return {
        "tokens": torch.rand(batch, tokens, N_FEATURES), "token_mask": torch.ones(batch, tokens, dtype=torch.bool),
        "token_bp": torch.zeros(batch, tokens, dtype=torch.long), "token_order": torch.zeros(batch, tokens, dtype=torch.long),
        "token_last": torch.zeros(batch, tokens, dtype=torch.long), "token_last_feat": torch.zeros(batch, tokens, 3),
        "token_type": torch.zeros(batch, tokens, dtype=torch.long),
        "map": torch.rand(batch, len(MAP_CHANNELS), tm.GRID, tm.GRID), "scalars": torch.rand(batch, N_SCALARS),
        "fine": torch.tensor([[4.0, 4.0, size, size]] * batch),
        "focus": torch.tensor([[[0.5, 0.5, 1.0], [0.2, 0.2, 1.0], [0.0, 0.0, 0.0]]] * batch),
    }


class CloseUpTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = tm.ToolsNet("S", decoder="units").eval()

    def test_a_fresh_close_up_changes_nothing(self):
        b = observation()
        with torch.no_grad():
            _, _, with_focus = self.net.encode(b)
            _, _, without = self.net.encode({k: v for k, v in b.items() if k != "focus"})
        torch.testing.assert_close(with_focus, without)

    def test_it_sees_the_tokens_in_its_window_only(self):
        b = observation()
        h = torch.randn(2, 4, self.net.d)
        m = torch.randn(2, self.net.map_ch, tm.GRID, tm.GRID)
        b["tokens"][0, 0, 7:9] = torch.tensor([0.5, 0.5])                # in the commander's window (row 0)
        with torch.no_grad():
            base = self.net._close_ups(h, m, b)
            moved = dict(b, tokens=b["tokens"].clone())
            moved["tokens"][0, 0, 7:9] = torch.tensor([0.5 + 20 / 1024, 0.5])   # 20 units away: still inside
            near = self.net._close_ups(h, m, moved)
            gone = dict(b, tokens=b["tokens"].clone(), token_mask=b["token_mask"].clone())
            gone["token_mask"][0, 0] = False
            absent = self.net._close_ups(h, m, gone)
        self.assertEqual(base.shape, (2, N_FOCUS * 64))
        self.assertFalse(torch.allclose(base[0, :64], near[0, :64]))     # a pixel and a half: a different picture
        self.assertFalse(torch.allclose(base[0, :64], absent[0, :64]))
        torch.testing.assert_close(base[0, 128:], torch.zeros(64))        # the absent fight: nothing

    def test_it_learns(self):
        b = observation()
        torch.nn.init.normal_(self.net.focus_out.weight, std=0.1)
        _, _, core_in = self.net.encode(b)
        core_in.square().sum().backward()
        self.assertGreater(float(self.net.focus_net[0].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
