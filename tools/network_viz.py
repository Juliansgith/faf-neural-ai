"""Build the network explorer page: a trained actor plus real moments from recorded games, in one HTML file.

    network_viz.py --checkpoint <run>/checkpoints/bc.pt --data <run>/bc_data --episodes <run>/episodes.jsonl
                   --games 2,7 --every 15 --out network.html

The page (template: network_viz.html next to this script) runs the actor in the browser -- the same layers,
the same weights -- on the saved observations, and checks itself against the probabilities PyTorch computed
here. Games are picked by their index among the recorded (collect) episodes, in recording order.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ai" / "python"))
from fa_env.factored import HEAD_NAMES, HEADS, UNIT_FEATURES  # noqa: E402
from fa_train.model import Actor, Critic, masked_logits, parameter_count, to_torch  # noqa: E402

SCALAR_NAMES = [
    "game time", "mass stored", "mass capacity", "mass income", "mass requested", "mass trend",
    "energy stored", "energy capacity", "energy income", "energy requested", "energy trend",
    "mass stalling", "energy stalling", "unit cap used",
    "has commander", "commander health", "commander distance from base",
    "engineers", "idle engineers", "commander idle", "land factories", "air factories",
    "idle land factories", "idle air factories", "mass extractors", "power generators", "hydrocarbon plants",
    "radars", "point defenses", "AA defenses", "tanks", "artillery", "mobile AA", "interceptors", "bombers",
    "scouts", "under construction", "army size", "army mass", "army distance to enemy base",
    "army distance to own base", "enemies seen", "remembered enemy buildings", "radar contacts",
    "enemy army mass seen", "enemies near base", "free mass spots", "free hydrocarbon spots",
    "reclaim targets",
]
SCALAR_NAMES += [f"last {h}: {o}" for h in HEAD_NAMES for o in HEADS[h]]
SCALAR_NAMES += [f"last {h} worked" for h in HEAD_NAMES]
FACTIONS = {1: "UEF", 2: "Aeon", 3: "Cybran", 4: "Seraphim"}


def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="folder with shard_*.npz from a collect run")
    ap.add_argument("--episodes", required=True, help="episodes.jsonl of the same run (labels)")
    ap.add_argument("--games", default="2,7", help="indices among the recorded episodes")
    ap.add_argument("--every", type=int, default=15, help="keep every n-th decision (n x 2 game seconds)")
    ap.add_argument("--trained-on", default="", help="one line for the page: what the checkpoint learned from")
    ap.add_argument("--template", default=str(Path(__file__).with_name("network_viz.html")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(state["actor"])
    actor.eval()
    names, chunks, offset, index = [], [], 0, {}
    for name, tensor in actor.state_dict().items():
        flat = tensor.detach().float().numpy().ravel()
        index[name] = [offset, list(tensor.shape)]
        chunks.append(flat)
        offset += flat.size
    weights = np.concatenate(chunks)

    shards = sorted(Path(a.data).glob("shard_*.npz"))
    data = {}
    for path in shards:
        with np.load(path) as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
    data = {k: np.concatenate(v) for k, v in data.items()}
    order = list(dict.fromkeys(data["episodes"].tolist()))
    records = [json.loads(line) for line in open(a.episodes, encoding="utf-8")]
    records = [r for r in records if r.get("mode") == "collect"]

    games = []
    for gi in (int(x) for x in a.games.split(",")):
        rows = np.flatnonzero(data["episodes"] == order[gi])
        picked = list(rows[:: a.every])
        if picked[-1] != rows[-1]:
            picked.append(rows[-1])
        batch = to_torch({k: data[k][picked] for k in ("scalars", "units", "unit_bp", "unit_group", "unit_mask",
                                                         *[f"mask_{h}" for h in HEAD_NAMES])}, "cpu")
        with torch.no_grad():
            probs = {h: torch.softmax(l, -1).numpy() for h, l in masked_logits(actor(batch), batch).items()}
        rec = records[gi] if gi < len(records) else {}
        moments = []
        for j, r in enumerate(picked):
            valid = data["unit_mask"][r]
            moments.append({
                "t": round(float(data["scalars"][r][0]) * 18000 / 600, 2),
                "scalars": b64(data["scalars"][r], "<f4"),
                "units": b64(data["units"][r][valid], "<f4"),
                "bp": data["unit_bp"][r][valid].astype(int).tolist(),
                "grp": "".join(str(int(g)) for g in data["unit_group"][r][valid]),
                "masks": {h: "".join("1" if m else "0" for m in data[f"mask_{h}"][r]) for h in HEAD_NAMES},
                "teacher": {h: int(data["actions"][r][i]) for i, h in enumerate(HEAD_NAMES)},
                "ref": {h: [round(float(p), 6) for p in probs[h][j]] for h in HEAD_NAMES},
            })
        games.append({"label": f"Teacher vs {rec.get('ai', '?')} ({FACTIONS.get(rec.get('ai_faction'), '?')})",
                      "ai": rec.get("ai"), "outcome": rec.get("outcome"), "minutes": rec.get("minutes"),
                      "decisions": int(len(rows)), "moments": moments})

    bc = state.get("bc") or {}
    payload = {
        "meta": {"params": parameter_count(actor), "critic_params": parameter_count(Critic()),
                 "checkpoint": Path(a.checkpoint).name, "trained_on": a.trained_on,
                 "agreement_when_acting": bc.get("agreement_when_acting"),
                 "heads": {h: list(o) for h, o in HEADS.items()}, "unit_features": list(UNIT_FEATURES),
                 "scalar_names": SCALAR_NAMES},
        "weights": {"b64": b64(weights, "<f4"), "index": index},
        "games": games,
    }
    text = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = Path(a.template).read_text(encoding="utf-8").replace("__DATA__", text)
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"{a.out}: {len(html) / 1e6:.1f} MB, {sum(len(g['moments']) for g in games)} moments, "
          f"{weights.size} weights")


if __name__ == "__main__":
    main()
