@echo off
rem RL phase 1, second start (2026-10-10 14:10, after the entropy fix; with the regression guard): PPO on the
rem server's recorded games (fa_train.rl_loop -> fa_train.learn_rl) from the second-pass-fix network, which is also
rem the KL anchor. Each iteration: pull the finished rounds, PPO over the newest 120 games played by the current or
rem the previous network (2 epochs, lr 1e-5, M28's labels 0.3, anchor 0.2, no entropy bonus), push the result.
rem The loop stops by itself and pushes the starting network back if a network wins 8 points less than it on the
rem server (140+ games). From 2026-10-10 20:40 (iteration 12) spatial credit is on (--local-credit 0.5).
rem From 2026-10-11 ~14:00 (iteration 37): lr 3e-5 (was 1e-5; the iterations moved the policy by KL ~0.004
rem against a 0.02 target, and the gains had flattened). Revert to 1e-5 if KL stops start firing often.
rem From 2026-10-11 ~15:00: games of the newest 4 networks (V-trace), 240 per iteration (was 120 of 2), and the M28
rem curriculum: the player's opponents are adaptive, rush, tech, M28 at a multiplier kept near a 50 % win rate
rem (double weight, starting at 0.6) and full M28 (rl_loop --m28-curriculum, level in rl2\curriculum.json).
rem From 2026-10-11 14:30 (iteration 37): decoupled PPO in learn_rl (the KL stop measures the real update: at lr 1e-5
rem one epoch moved the policy by KL ~0.0008), so lr 3e-5 again, now with a working early stop.
rem From 2026-10-11 ~15:30: --econ-coef 1.0 (mexes and engineers destroyed/lost count beyond their cost:
rem tools_bc.economy_bonus; in games encoded from then on).
rem From 2026-10-11 16:40 (iteration 43): reward v2 (learn_rl rewards_of) and danger maps (5 map channels, cache
rem format 11): economy = change of mass production capacity (the recorded income change was 99 % reclaim noise),
rem reclaim as mass gained x0.5, intel and map-control potentials (x1, x2), waste x1; the value head is warmed up
rem when the reward spec changes.
rem From 2026-10-11 ~17:10 (iteration 44): lr 2e-5. rl_038, the first at 3e-5 with decoupled PPO, played 7 points
rem under the start on adaptive/rush/tech (68 %, 144 games; rl_040 78 %, 72 games): smaller steps while reward v2
rem settles. The regression guard now pools the newest networks and falls back to the newest good one.
rem From 2026-10-11 ~20:00 (iteration 50): 4 epochs (was 2). rl_046 recovered to the start's 76 % on adaptive/rush/
rem tech (defeats 11 %, M28 x0.65 51 %); per epoch KL ~0.004-0.005 at 2e-5, the (now real) KL stop guards 4 epochs.
rem From 2026-10-12 ~00:00: 25 % of the games on 20 km maps (player config "maps"; their results count apart from the
rem small maps', rl_loop LARGE_KM), economy capacity from the units' actual output (adjacency counts), and the energy
rem stall's lost spending (--stall-coef 1.0): the commander rush that wins small maps against weak M28 must not be
rem all it knows (the user's observation, watching rl_055).
rem From 2026-10-12 ~00:30: --reference-lag 10, the KL anchor is the network 10 iterations back, not the start (M28's
rem imitation): M28 x0.65 results flat at ~46 % for eight networks while ref_kl to the start kept growing.
rem From 2026-10-12 ~11:30 (iteration 74): self-imitation (rl_loop --sil-archive 60, learn_rl --sil): after the PPO
rem epochs the policy imitates its own won games against M28 at the curriculum level or above (up to 60, the last
rem 30 networks) where they went better than the critic expected (Oh et al. 2018): M28 x0.6 flat at ~55 % for 30
rem networks. Big maps: M28's labels x4 in 20 km games (--distill-large 4; our commander pushed to the enemy
rem there and died, 99 % lost even to the scripted AIs, where M28 pulls its commander home) and 55 minutes on
rem them (player config "minutes": "40,55"). The curriculum is judged on small maps only (was x0.55, reset to 0.6).
rem From 2026-10-12 ~15:10 (iteration 79): fine positions and close-ups (code 1824e61): an order's cell splits into
rem ~160 m pixels on large maps (4 x 4 on 20 km), scored on a close-up of the cell; close-ups around our commander,
rem army and the fight feed the core. rl_079 = rl_078 with those weights pre-trained on M28's exact positions
rem (40 20 km games; held-out pixel = M28's 48 %, within one 76 %), the rest unchanged. Cache format 12.
rem 2026-10-12 16:50 (iteration 83): ROLLBACK. rl_079-081 lost (rl_079 3/21 vs scripted): its close-ups were
rem pre-trained with the full imitation loss, which pulled every decision toward M28 (log-prob of rl_078's choices
rem -25 a decision). rl_083 = rl_079 with the close-ups zeroed (= rl_078's decisions; the fine head on large
rem maps); the close-ups learn from zero by RL. --recent-networks 1 until rl_079-081's games are out of the
rem window (then back to 4). Self-imitation gentler from rl_082 (learn_rl --sil-coef 0.03, no critic part).
rem 2026-10-12 ~21:00 (iteration 86): resumed from rl_077 (the guard's fallback at 18:32; rl_077 88 % vs scripted on
rem 150 games, rl_083 = rl_078's play 77 % on 144) with the fine head fitted to it (fine_pretrain --train-only fine_;
rem close-ups zero). rl_079-081's 72 cached games moved to rl2\cache_quarantine_rl_079_rl_080_rl_081.
rem --recent-networks back to 4.
rem 2026-10-13 ~09:45 (iteration 106): from rl_089 again (rl_102: M28 x0.6 4/15 in its first round; a clean
rem comparison) with the commander's exposure on large maps as a potential (--acu-coef 3: its walk home past
rem 1 minute, full at 5; kept when the game is lost, given back when won; learn_rl phi_commander). The large-map
rem imitation floor 0.4 (rl_090-093) is off: it took the small maps toward M28. The loop's chain, regression
rem guard and imitation-weight basis follow the lineage (abandoned branches were anchors, rl_096).
rem 2026-10-13 ~23:30 (iteration 108): SELF-PLAY (code 2f2c38e): past networks of ours as opponents, about 30 %
rem of the games (--league rl_077,rl_086,rl_089: 'self@<name>' in the players' mix, <home>/league/<name>.pt).
rem Both armies are M28 brains in shadow mode (labels for our side as always), army 2 played by the frozen
rem network in the game's worker. Judge networks with _runs\tools\compare_nets.py (matched conditions).
rem 2026-10-14 ~02:30 (iteration 112): commander exposure reward OFF, back to rl_089. rl_107 (the first with it)
rem won 13/21 vs scripted on small maps (rl_089 91 %, same machine), matched -30 points (95 % -42..-17), and lost
rem head-to-head to rl_089 2-5-3; rl_108 3-6-1. A large-map-only term trains the shared network (as the 0.4
rem imitation floor did). Kept: self-play league, varied factions/maps/slots per round (vary_conditions in the
rem player configs; training had been 67 % UEF), the faf-re latest engine. Judge with compare_nets + self-play.
rem 2026-10-14 ~04:10: --archive-dir D:\archive\rl2: raw recordings older than 24 h and cache files older
rem than 72 h are moved to D: each iteration (C: filled up at 03:50: recordings grow 50-70 GB a day). The first
rem move by hand: D:\archive\rl2_cache_stale (format-11 cache, 56 GB), D:\archive\rl2_rounds_raw (86 GB).
rem 2026-10-14 ~19:30 (iteration 127): no network since rl_089 has been better (rl_070-089 flat ~85 % vs
rem scripted; every continuation from rl_089 went down). Changes, together:
rem * champion gate (rl_loop --champion rl_089, --gate-games 100): each network against the best so far on
rem   matched games; clearly worse -> back to the champion in place, clearly better -> the new champion. The
rem   regression guard compared with the imitation start (76 %), 10 points under rl_089.
rem * bigger batches: --games-per-iteration 480 --min-new-games 240 (were 240/60; rl_122-125 trained on ~100
rem   games each). Games read memory-mapped (learn_rl --mapped): ~99 MB a game in RAM since the close-ups.
rem * the critic told the opponent (learn_rl --opponent-critic; zero at the start, never seen when playing).
rem * timeouts scored by who was ahead (--timeout-scale 10 --timeout-cost 2; were 5 x value advantage: a
rem   timeout -4.5 against a defeat -18.6, stalling paid).
rem * self-play OFF (no --league) until these are judged; then back on as a measured experiment.
rem Also since 10:55: learn_rl trains on the lineage's rounds only (4b09750); --collapse-floor 0.4 (ccf6436).
rem Stop: create _runs\server\rl2\STOP.
set FAF=C:\faf
set INIT=%FAF%\_runs\server\vaa_M\checkpoints\step_01.pt
cd /d "%FAF%\faf-neural-ai\ai\python"
"%FAF%\ai-venv\Scripts\python.exe" -u -m fa_train.rl_loop --run "%FAF%\_runs\server\rl2" --init "%INIT%" --reference "%INIT%" --push --games-per-iteration 480 --min-new-games 240 --recent-networks 4 --m28-curriculum --m28-start 0.6 --reference-lag 10 --local-home "%FAF%\_runs\local_player" --sil-archive 60 --archive-dir D:\archive\rl2 --champion rl_089 --gate-games 100 --learn-args "--opponent-critic --timeout-scale 10 --timeout-cost 2 --mapped %FAF%\_runs\server\rl2\mapped --distill-large 4 --epochs 4 --lr 2e-5 --distill 0.3 --ref-coef 0.2 --streams 16 --micro 4 --grad-checkpoint --local-credit 0.5 --econ-coef 1.0 --econ-source capacity --reclaim-coef 0.5 --intel-coef 1.0 --map-coef 2.0 --waste-coef 1.0 --stall-coef 1.0" >> "%FAF%\_runs\server\rl2.log" 2>&1
