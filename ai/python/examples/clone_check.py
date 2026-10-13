"""Gate 2 on a small scale: collect rule-policy games, clone them, and let the greedy clone play.

    python clone_check.py --run C:/faf/_runs/clone_check --games 12 --eval 3

One worker (one game at a time). Prints the clone's held-out agreement per head (overall and when the rule
policy acted), its entropy, and the clone's results against easy next to the rule policy's.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train.bc import train_bc  # noqa: E402
from fa_train.night import by_ai  # noqa: E402
from fa_train.trainer import TrainConfig, Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--games", type=int, default=12, help="rule-policy games to clone (half easy, half medium)")
    ap.add_argument("--eval", type=int, default=3, help="greedy clone games against easy")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--eval-minutes", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = TrainConfig(run_dir=str(run), workers=a.workers, minutes=a.minutes, eval_minutes=a.eval_minutes,
                      opponents={"easy": 0.5, "medium": 0.5})
    trainer = Trainer(cfg)
    try:
        if not any((run / "bc_data").glob("shard_*.npz")):
            trainer.run("collect", max_episodes=a.games, collect_dir=run / "bc_data")
        bc = train_bc(run / "bc_data", run / "checkpoints" / "bc.pt", epochs=8)
        last = bc["best"]
        trainer.load(run / "checkpoints" / "bc.pt", optimizer=False)
        trainer.schedule_eval(a.eval, "easy")
        trainer.run("eval")
        result = {"rows": bc["rows"], "agreement": last["agreement"],
                  "agreement_when_acting": last["agreement_when_acting"], "entropy": last["entropy"],
                  "rule_policy_games": by_ai([r for k, v in trainer.recent.items() if k.startswith("collect_")
                                              for r in v]),
                  "clone_greedy": by_ai(trainer.eval_results)}
        (run / "clone_check.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
