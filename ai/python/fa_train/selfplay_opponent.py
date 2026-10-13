"""The opponent of a self-play game: a frozen network playing army 2 inside the game's worker (fa_env.selfplay),
on the CPU, one decision at a time, as the trainer acts for its games (tools_trainer.ToolsTrainer._act)."""
from __future__ import annotations

from pathlib import Path

import torch


class OpponentPlayer:
    def __init__(self, path: Path, mode: str = "sample", temperature: float = 0.5, top_p: float | None = None):
        from fa_train.tools_model import ToolsNet, apply_model_options
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.net = ToolsNet(state.get("size", "M"), decoder="units")
        self.net.load_state_dict(state["model"])
        apply_model_options(self.net, state)
        self.net.eval()
        self.mode, self.temperature, self.top_p = mode, float(temperature), top_p
        self.memory = self.net.initial_state(1, torch.device("cpu"))
        self.name = Path(path).stem

    def act(self, obs: dict) -> list[dict]:
        """Commands for army 2 from its ToolsObserver observation."""
        from fa_train.tools_model import actions_to_commands, collate
        from fa_train.tools_trainer import widen
        from fa_train.worker import compact
        b = collate([widen(compact(obs))], torch.device("cpu"))
        self.net.act_top_p = self.top_p
        with torch.no_grad():
            acts, _, _, self.memory = self.net.forward_step(b, self.memory, self.mode, temperature=self.temperature)
        acts_np = {k: v.numpy() for k, v in acts.items()}
        return actions_to_commands(acts_np, 0, obs, "units")
