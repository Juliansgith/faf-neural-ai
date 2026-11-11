"""Candidate networks for a paired acting test, from a base checkpoint: the same weights under another verb
distribution, and a weight average of a run of checkpoints (2026-11-07 plateau report). Nothing here trains.

    python -m fa_train.make_arms --base rl_258.pt --out DIR --average rl_254.pt rl_255.pt rl_256.pt rl_257.pt rl_258.pt

Writes DIR/arm_smooth.pt, DIR/arm_smooth_upgrade.pt and, with --average, DIR/arm_average.pt (acting as the base does).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def with_options(state: dict, **options) -> dict:
    out = dict(state)
    out["model_options"] = {**(state.get("model_options") or {}), **options}
    return out


def average(states: list[dict]) -> dict:
    """The mean of each floating tensor (integer buffers from the last checkpoint); shapes must agree."""
    keys = list(states[-1]["model"])
    model = {}
    for k in keys:
        tensors = [s["model"][k] for s in states]
        if any(t.shape != tensors[-1].shape for t in tensors):
            raise ValueError(f"{k}: the checkpoints have different shapes")
        model[k] = (torch.stack([t.double() for t in tensors]).mean(0).to(tensors[-1].dtype)
                    if tensors[-1].is_floating_point() else tensors[-1])
    return {**states[-1], "model": model}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--average", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = load(a.base)
    torch.save(with_options(base, smooth_verb_floor=True), a.out / "arm_smooth.pt")
    torch.save(with_options(base, smooth_verb_floor=True, upgrade_floor_free=True), a.out / "arm_smooth_upgrade.pt")
    if a.average:
        states = [load(p) for p in a.average]
        mean = average(states)
        distance = sum(float((mean["model"][k].float() - base["model"][k].float()).pow(2).sum())
                       for k in base["model"] if base["model"][k].is_floating_point()) ** .5
        print(f"average of {len(states)} checkpoints: {distance:.3f} from the base")
        torch.save(mean, a.out / "arm_average.pt")
    print("written:", sorted(p.name for p in a.out.glob("arm_*.pt")))


if __name__ == "__main__":
    main()
