"""Behaviour cloning: train the actor to reproduce the rule policy's choices, and warm-start the critic on
the rule policy's discounted returns (docs/training-design.md, section 9).

    python -m fa_train.bc --data <run>/bc_data --out <run>/checkpoints/bc.pt

Shards come from `Trainer.run("collect", ...)`: stacked observations, the rule policy's action per head,
and the discounted return of each step. 10 % of the rows (whole shards when there are several) are held
out; the report gives agreement per head on them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_env.factored import HEAD_NAMES, HEADS  # noqa: E402
from fa_train.model import Actor, Critic, masked_logits, to_torch  # noqa: E402

OBS_KEYS = ("scalars", "units", "unit_bp", "unit_group", "unit_mask", "privileged",
            *[f"mask_{h}" for h in HEAD_NAMES])


def load_shards(folder: Path) -> list[dict]:
    shards = []
    for path in sorted(folder.glob("shard_*.npz")):
        with np.load(path) as z:
            shards.append({k: z[k] for k in z.files})
    return shards


def concat(shards: list[dict]) -> dict:
    return {k: np.concatenate([s[k] for s in shards]) for k in shards[0]}


def smoothed_loss(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                  smoothing: float) -> torch.Tensor:
    """Class-weighted cross-entropy against (1 - smoothing) on the teacher's option plus `smoothing` spread
    over the options that were allowed. The rule policy is deterministic, so an exact clone would be too:
    PPO could then never try anything else. Smoothing only over allowed options keeps masked ones at zero."""
    # normalize over the allowed options only, exactly as the policy is sampled (masked_logits)
    logp = F.log_softmax(logits.masked_fill(~mask, -1e9), -1).masked_fill(~mask, 0.0)
    allowed = mask.to(logp.dtype)
    soft = smoothing * allowed / allowed.sum(-1, keepdim=True).clamp(min=1.0)
    soft = soft + (1.0 - smoothing) * F.one_hot(target, logits.shape[-1]).to(logp.dtype)
    per_row = -(soft * logp).sum(-1)
    w = weight[target]
    return (w * per_row).sum() / w.sum()


def train_bc(data_dir: Path, out: Path, epochs: int = 8, batch: int = 512, lr: float = 3e-4,
             device: str | None = None, smoothing: float = 0.05, min_steps: int = 6000) -> dict:
    """At least `min_steps` gradient steps whatever the data size (a few thousand rows need hundreds of
    epochs; 4.5 k rows x 8 epochs was 72 steps and cloned nothing). The saved checkpoint is the evaluation
    with the best held-out agreement on the rows where the teacher acted, not the last one."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    shards = load_shards(data_dir)
    if not shards:
        raise SystemExit(f"no shards in {data_dir}")
    data = concat([{k: v for k, v in s.items()} for s in shards]) if all("episodes" in s for s in shards) else None
    if data is not None:
        # hold out 10 % of whole episodes, chosen at random: rows of one game are strongly correlated
        ids = np.unique(data["episodes"])
        held = np.random.default_rng(0).choice(ids, max(1, len(ids) // 10), replace=False)
        test_rows = np.isin(data["episodes"], held)
        train = {k: v[~test_rows] for k, v in data.items()}
        test = {k: v[test_rows] for k, v in data.items()}
    elif len(shards) >= 4:
        held = max(1, len(shards) // 10)
        train, test = concat(shards[:-held]), concat(shards[-held:])
    else:
        data = concat(shards)
        n = len(data["actions"])
        cut = int(n * 0.9)
        train = {k: v[:cut] for k, v in data.items()}
        test = {k: v[cut:] for k, v in data.items()}
    n_train, n_test = len(train["actions"]), len(test["actions"])
    returns = train["returns"]
    ret_std = float(returns.std() + 1e-6)          # the critic learns returns / std, like PPO's scaled rewards
    print(f"{n_train} training rows, {n_test} held out; return mean {returns.mean():.2f} std {ret_std:.2f}")

    actor, critic = Actor().to(device), Critic().to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    # class weights: "none" dominates every head; weight each option by 1/sqrt(frequency) so the rare
    # decisions (the ones that matter) are learned too
    weights = {}
    for i, h in enumerate(HEAD_NAMES):
        counts = np.bincount(train["actions"][:, i], minlength=len(HEADS[h])).astype(np.float64) + 1.0
        w = 1.0 / np.sqrt(counts / counts.sum())
        weights[h] = torch.as_tensor(w / w.mean(), dtype=torch.float32, device=device)

    def batches(data, size, shuffle):
        idx = np.random.permutation(len(data["actions"])) if shuffle else np.arange(len(data["actions"]))
        for start in range(0, len(idx), size):
            j = idx[start:start + size]
            yield to_torch({k: data[k][j] for k in OBS_KEYS}, device), \
                torch.as_tensor(data["actions"][j], device=device), \
                torch.as_tensor(data["returns"][j] / ret_std, device=device)

    def evaluate(data):
        actor.eval()
        correct = {h: 0 for h in HEAD_NAMES}
        decided = {h: [0, 0] for h in HEAD_NAMES}   # agreement on rows where the teacher did not pick "none"
        entropy = {h: 0.0 for h in HEAD_NAMES}      # of the clone's distribution: what PPO will explore with
        v_err, n = 0.0, 0
        with torch.no_grad():
            for obs, act, ret in batches(data, 2048, False):
                logits = masked_logits(actor(obs), obs)
                for i, h in enumerate(HEAD_NAMES):
                    pred = logits[h].argmax(-1)
                    correct[h] += (pred == act[:, i]).sum().item()
                    active = act[:, i] != 0
                    decided[h][0] += (pred[active] == act[active, i]).sum().item()
                    decided[h][1] += active.sum().item()
                    entropy[h] += torch.distributions.Categorical(logits=logits[h]).entropy().sum().item()
                v_err += ((critic(obs) - ret) ** 2).sum().item()
                n += len(ret)
        actor.train()
        return ({h: correct[h] / n for h in HEAD_NAMES},
                {h: (d[0] / d[1] if d[1] else None) for h, d in decided.items()}, v_err / n,
                {h: entropy[h] / n for h in HEAD_NAMES})

    t0 = time.monotonic()
    history = []
    per_epoch = max(1, -(-n_train // batch))
    epochs = max(epochs, -(-min_steps // per_epoch))
    eval_every = max(1, -(-500 // per_epoch))           # evaluate about every 500 steps
    best, best_score = None, -1.0
    losses = []
    for epoch in range(epochs):
        for obs, act, ret in batches(train, batch, True):
            logits = actor(obs)
            loss = sum(smoothed_loss(logits[h], obs[f"mask_{h}"], act[:, i], weights[h], smoothing)
                       for i, h in enumerate(HEAD_NAMES))
            v_loss = F.mse_loss(critic(obs), ret)
            optim.zero_grad()
            (loss + 0.5 * v_loss).backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            optim.step()
            losses.append(loss.item())
        if (epoch + 1) % eval_every and epoch + 1 != epochs:
            continue
        acc, acc_decided, v_mse, entropy = evaluate(test)
        acting = [v for v in acc_decided.values() if v is not None]
        score = float(np.mean(acting)) if acting else float(np.mean(list(acc.values())))
        history.append({"epoch": epoch + 1, "steps": (epoch + 1) * per_epoch, "loss": float(np.mean(losses)),
                        "agreement": acc, "agreement_when_acting": acc_decided, "value_mse": v_mse,
                        "entropy": entropy, "score": score})
        losses = []
        if score > best_score:
            best_score = score
            best = {"actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
                    "critic": {k: v.detach().clone() for k, v in critic.state_dict().items()}, "at": history[-1]}
        print(f"epoch {epoch + 1} ({(epoch + 1) * per_epoch} steps): loss {history[-1]['loss']:.3f}; when acting "
              + ", ".join(f"{h} {v:.2f}" if v is not None else f"{h} -" for h, v in acc_decided.items())
              + f"; value mse {v_mse:.3f}; entropy " + ", ".join(f"{h} {entropy[h]:.2f}" for h in HEAD_NAMES))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": best["actor"], "critic": best["critic"], "update": 0, "episodes": 0,
                "rms": (0.0, ret_std ** 2, 1e4), "heads": HEADS, "bc": best["at"]}, out)
    report = {"rows": n_train, "held_out": n_test, "seconds": round(time.monotonic() - t0, 1),
              "best": best["at"], "history": history}
    (out.parent / "bc_report.json").write_text(json.dumps(report, indent=1))
    print(f"kept the evaluation at epoch {best['at']['epoch']} (held-out agreement when acting {best_score:.3f})")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    train_bc(Path(a.data), Path(a.out), epochs=a.epochs)


if __name__ == "__main__":
    main()
