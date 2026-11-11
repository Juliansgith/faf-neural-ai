# rl3: the whole RL pipeline on the server (since 2026-11-07)

Games, learner, retention and reports all run on my-server. See `_runs/plateau_20261107/REPORT.md` for why.

| Piece | Where |
|---|---|
| Player (games) | `fa-player.service`, home `~/fa/player`, engine `~/fa/bin_ultra_20261107` |
| Learner | `fa-learner.service`, runs `rl3_learner.sh`, run folder `~/fa/rl3` |
| Reports | `~/fa/rl3/reports/`, written daily by `fa-report.timer` |
| Upkeep | `fa-watchdog.timer` every 15 min: stale learning pause, end of the forced-upgrade test |
| Backups | the PC's task "faf rl3 backup" (08:15 daily): checkpoints, journal, reports to `_runs/server/rl3_backup` |
| Stop gently | `touch ~/fa/rl3/STOP` (learner stops at its next safe point); `touch ~/fa/player/PAUSE` (no new rounds) |

Rules kept from the PC pipeline: never force-kill the learner mid-update, use STOP; judge any acting change with a
drift check and paired games; a full disk pauses intake (`--min-free-gb`).
