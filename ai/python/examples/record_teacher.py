"""Record rule-teacher games (raw recordings, fa_env/record.py) with one game at a time.

    python record_teacher.py --run C:/faf/_runs/teacher_rec --games 25

Half against easy, half against medium, random AI factions, 20-minute limit. The recordings land in
<run>/raw; the menu version's cloning shards in <run>/bc_data come along for free.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa_train.trainer import TrainConfig, Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = TrainConfig(run_dir=a.run, workers=a.workers, minutes=a.minutes, opponents={"easy": 0.5, "medium": 0.5})
    trainer = Trainer(cfg)
    try:
        print(trainer.run("collect", max_episodes=a.games, collect_dir=Path(a.run) / "bc_data"))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
