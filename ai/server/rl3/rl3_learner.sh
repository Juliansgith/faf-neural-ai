#!/bin/bash
# rl3: the learner on my-server (fa-learner.service). Every change to these settings gets a dated note here.
# 2026-11-07: start. rl_000 = rl_258 with the smooth verb distribution (arena 11-07/08: +4.6 +-4.2 points vs rl_258
#   at M28 x0.75 on the ultra engine, 780 games). X6-self-play recipe (it produced rl_258), CPU learner with worker
#   processes and bf16 encoder (17 min for 120 games, one epoch), fresh games only (the smooth semantics reject old).
#   Gate on more games (400 to roll back, 1200 to promote). --unstable-gnorm 1.3: chunks-per-step reads ~0.98 where
#   rounds read 0.80. --full-m28-weight 0: the X6 mix had no full-strength M28 slot.
# 2026-11-08: the first iteration was killed by the kernel (out of memory) in its first epoch: 26 workers grew to
#   5 GB each. Now 20 workers, fresh ones every epoch, fixed glibc malloc thresholds (large blocks go back to the
#   system instead of fragmenting the heap), and the service capped at 36 GB (fa-learner.service MemoryMax).
# 2026-11-08 later: the cap was hit in the epoch (each worker ~1.4 GB of its own, the master ~9 GB; the games take
#   ~50 GB): 12 workers, cap 44 GB. With the games running the learner gets about 13-17 cores either way.
# 2026-11-08 evening (user: "dedicate 100% of cpu to training during training"): --learn-alone, the games wait while
#   learn_rl trains (rl_001 took 8.6 h next to 52 games), 22 workers when no game runs (cap 80 GB), 12 otherwise.
#   Self-imitation now runs on the workers (it took 1.5 h in one process); --probe-games 0 (diagnostics, 0.5 h).
# 2026-11-09: promotion at 700 games, and each network plays at least 700: at 1200 no network ever reached the
#   promotion look (each played 400-900 before the next one replaced it), so rl_000 stayed the champion for good.
# 2026-11-10: --aux-econ 0.02 from rl_005 (the first strength change after the smooth start): zero-initialised heads
#   on the memory predict income, army and capacity 1, 5 and 20 minutes ahead (plateau report: the memory forgets the
#   economy after minute 20; UNREAL-style auxiliary loss, reward unchanged). The games drain inside a round now (a
#   round is ~150 games, two hours: rl_004's pause never drained and learnt next to the games).
# 2026-11-11: player config "ai_maps": M28 plays the seven original maps only (the four new 10 km maps were 0-3 % wins
#   against it), the easy AIs play mostly the new ones. --m28-band 0.4,0.7 (was 0.4,0.6): without the lost new-map
#   games M28 x0.75 reads ~60 % on the original maps, and a step to x0.775 would leave the gate without champion
#   games against its main opponent (the gate's cells are per opponent name). Decide the anchor before stepping.
# 2026-11-11 later (user: "go with the control lane"): --control-lanes 5, lane 5 of 6 always plays the champion, so
#   the gate has champion games in whatever the candidates play: the same schedule, M28 level and acting code.
#   --gate-since 20261111T1600Z: from here every gate game comes from one schedule (M28 now plays 40 % of its games
#   on the two original 10 km maps, where the network wins 11-16 % against 49-77 % on 5 km; the champion's older
#   games had 20 %). The curriculum may step again: --m28-band 0.4,0.6, on 600 games at the level (was 200: +-3.5
#   points of noise moved the level). 29 learner workers alone (was 22: the load average sat at 22.0 of 32 CPUs for
#   the whole 150 minutes; ~1.4 GB a worker, cap 80 GB).
# 2026-11-11 evening: the 10 km maps get their own M28 level (--m28-wide-start 0.55, steps of 0.05 on 400 games,
#   band 40-60 %). Read from the recordings of rl_009/rl_010: on 10 km the network's economy stops at tech 1
#   (14 extractors and +42 mass at minute 15, M28 x0.75 has 29 and +94) and its commander dies 200-300 from its
#   start; it won 15 % on the two old 10 km maps and 1 % on the four new ones against x0.75, 67 % on 5 km. Games
#   that are almost never won teach little, so M28 x0.75 and its stretch slot play the five 5 km maps only
#   (weight 4 and 2) and a slot of weight 4 plays the six 10 km maps (old and new half each) at the wide level.
#   The loop writes the player's ai_maps for M28 now; the easy AIs' entries there stay as set by hand. The four
#   new maps count in the gate again (acting_equivalence.json: new_maps instead of gate_exclude_maps): the
#   control lane plays them with the champion. --gate-since 20261111T1700Z: one schedule from there on.
ulimit -n 200000
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=4194304 MALLOC_ARENA_MAX=2
R=$HOME/fa/rl3
cd $HOME/faf-neural-ai/ai/python
exec $HOME/venv/bin/python -u -m fa_train.rl_loop --run $R --init $R/checkpoints/rl_000.pt --reference $R/checkpoints/rl_000.pt \
  --no-remote --local-home $HOME/fa/player --champion rl_000 --tag rl3-smooth \
  --games-per-iteration 720 --min-new-games 700 --recent-networks 2 --train-since 20261107T1930Z --gate-since 20261111T1700Z \
  --m28-curriculum --m28-start 0.75 --m28-step 0.025 --m28-min-games 600 --m28-band 0.4,0.6 --m28-weight 4 --stretch-weight 2 --full-m28-weight 0 \
  --m28-maps SCMP_012,SCMP_013,SCMP_016,X1MP_001,X1MP_002 --m28-wide-start 0.55 --m28-wide-step 0.05 --m28-wide-min-games 400 --m28-wide-weight 4 \
  --m28-wide-maps SCMP_019,SCMP_023,SCMP_019,SCMP_023,SCMP_017,X1MP_005,X1MP_006,X1MP_017 \
  --league rl_000 --league-weight 4 --sil-archive 60 --anchor-champion --adaptive-distill 0 --distill 0.05 \
  --gate-games 400 --gate-promote-games 700 --gate-promote-z 2.5 --gate-promote-points 3 --min-candidate-games 700 \
  --unstable-gnorm 1.3 --unstable-clip 0.04 --control-lanes 5 --learn-alone --learn-processes-alone 29 --drain-minutes 45 \
  --archive-dir $R/archive --archive-raw-hours 48 --archive-cache-hours 24 --keep-cache-rounds 24 \
  --storage-budget-gb 250 --min-free-gb 60 \
  --learn-args "--opponent-critic --timeout-scale 10 --timeout-cost 2 --mapped $R/mapped --distill-large 4 --epochs 2 --lr 1e-5 --ref-coef 0.3 --streams 32 --chunks-per-step 16 --micro 1 --local-credit 0.5 --econ-coef 1.0 --aux-econ 0.02 --econ-source capacity --reclaim-coef 0.5 --intel-coef 1.0 --map-coef 2.0 --waste-coef 1.0 --stall-coef 0.25 --capacity-clip 1 --mc-warmup --value-warmup 3 --value-lr 1e-4 --value-clip 0 --target-kl 0.01 --value-warmup-every 1 --device cpu --processes 12 --cpu-threads 1 --cpu-bf16 --probe-games 0" \
  "$@"
