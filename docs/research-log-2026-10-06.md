# Research log, 2026-10-05/07: first networks that win

The state before this: every pure network lost every game to the easy FAF AI (the slots decoder cloning M28; the
rule-teacher clone at 0/14 greedy). This log records what changed, with the numbers, and what comes next.

## Results (clean: the network plays alone, 48 games on the 5/10 km maps, 30-minute cap, sampled)

| network | vs easy | vs medium |
|---|---|---|
| rule-teacher units clone S, before DAgger (2026-10-05 night) | 20/64 (31 %) | 0/24 |
| M28 units clone S, before DAgger | 1/48 | 0/24 |
| M28 DAgger round 1 (`m28_night/dagger/checkpoints/round_01.pt`) | 12/48 | – |
| round 3, after the label fix | – | 9/48 |
| round 5 | 30/48 (62 %) | – |
| round 9 | – | 20/48 (42 %) |
| round 13, temperature 0.5 | 38/48 (79 %) | 29/48 (60 %) |
| **generation 2: token memory + two-pass decoder, warm-started from round 13** (`night2/mem2pass/s_mem.pt`) | **44/48 (92 %)** | **36/48 (75 %)** |

The rule teacher v3 itself: 77 % vs easy, 34 % vs medium. M28 itself: ~97 % vs everything it was run against.
All numbers are 48-game samples: ±7 % one sigma.

## What made the difference, in order of effect

1. **M28 as a live teacher (shadow mode) + DAgger** (`fa_train/dagger.py`, bridge `{"config": {"shadow": true}}`).
   M28 sits in the network's slot, its orders are recorded but not given, and the network is cloned on again over
   M28's own games plus every round of its own games with M28's labels. A clone only ever saw the teacher's
   states; the rounds give it the teacher's answer in its own states. 1/48 -> 11/48 after the first round, with
   nothing else changed. The rule teacher as a DAgger teacher did nothing (25 -> 23 -> 27 %): its labels are
   weak where the clone gets to.
2. **Per-unit labels: a move ahead of another order is the way there** (`unit_labels`, cache format 9). M28 gives
   43 % of its engineers' builds as "move, then build"; the label kept the first command, so the clone learned to
   walk and never built (one extractor after five minutes). 29 -> 50 % in the round after the fix.
3. **Token memory + two-pass decoder** (cache format 10). Each own unit's token carries the last order it was
   given and where it pointed (the engine reports a command type, not what a unit is building or heading for;
   M28 keeps that in its tables). The units decoder first guesses every unit's verb alone, then one attention
   layer over the own units carrying their guesses, then the final verbs and arguments: which of ten idle
   engineers takes a build is a question between them. 79/60 -> 92/75 with 3000 warm-started steps.
4. **Shadow mode keeps M28's own pausing off our units** (M28 pauses units directly under a stall, not through
   orders). Checked before and after: the win rates with M28 in the slot match the clean ones.
5. Decoding: the none-or-verb choice is sampled at temperature 1 in every mode (a rate), arguments at 0.5-0.7; no
   label smoothing on the verb; `act_min_p` 0.02. Greedy arguments lose.

Tried and dropped: greedy commander destination (58 vs 62 %), loss weight on builders (54 vs 62 %), lower learning
rate (inconclusive), rule-teacher DAgger.

## What the S network cannot do

Its agreement with M28 per unit stopped improving after round 3 and stayed there for ten rounds (held-out: gives
the right unit a verb 29 %, the right verb 25 %, blueprint 59 %, cell 37 %), with training loss equal to held-out
loss: capacity, not data. The win rates kept rising because DAgger corrects the few behaviours that decide games in
the network's own states. Engineers are the weak spot (M28 wants a build where the network reclaims or moves);
the army's moves take 60 % of the verb loss and matter less. Part of the residual is the target: M28's timing
and its choice among equivalent units depend on its internal tables, so no network reaches 100 %.

## Infrastructure that now exists

- `fa_train/dagger.py`: rounds of play (tools_eval, recorded with labels) -> encode -> clone-on; journal;
  STOP file; resumes; two loops can take turns on games and training (`--turns DIR`).
- `tools_bc`: `--mapped` corpora (113 KB/decision; the M28 corpus lives on D:\faf-nn-cache), `--loader-workers`,
  `--recent-weight` (DAgger games count 3x), `--accumulate`, `--grad-checkpoint`, `--stop-file`,
  `--memory-fraction`. Attention through `scaled_dot_product_attention` (SDPALayer): the stock layer's
  [batch, heads, T, T] weights were what filled 24 GB; the unit layer attends over own units only.
- `_runs/tools`: `dagger_table.py` (the rounds), `dagger_status.py RUN` (economy and verbs per round),
  `defeats.py RAW`, `acu_moves.py RAW`, `verb_report.py CKPT CACHE` (agreement by verb and role, CPU),
  `clean_eval.py CKPT OUT [games] [ai] [tools_eval options]`, `with_turn.py`, `rescue_round.py RUN K`,
  `bench_mem.py [--gpu]` (attention equivalence and peak memory per size), `gpu_watchdog.ps1`.
- The observer is 3x faster than on 2026-10-05 (5.3 ms per decision), bit-identical.

## Hardware notes

Training (one loop alone): ~0.3 s per step of 16 x 40 decisions on the 4090; games: 48 in 5-10 minutes with 16
at once (CPU-bound). Two loops at once share the GPU between a trainer and the other loop's inference: ~0.55 s
per step. Full VRAM (the size-M attempt at the stock attention) produced ~100 nvlddmkm errors in half an hour --
the pattern before the 2026-10-05 blue screen; the watchdog stops games and (now) trainings on the first error.
With the fused attention, checkpointing and accumulation, M should fit at the full effective batch and L at a
small one; not yet measured (`bench_mem.py --gpu`).

## Next

1. Measure M and L memory; train M from scratch on the aggregate (`night2/size_m.cmd`), compare with S's 92/75.
2. Continue DAgger with the better network (`night2/m28v2.cmd` resumes the S generation-2 loop at round 2;
   `night2/m28v2_M.cmd` for M), easy + medium, then hard/adaptive, then M28 itself as the opponent.
3. Imitation's ceiling is M28. Past it: PPO with a KL penalty to the frozen DAgger network and a slice of
   imitation data in every update, opponents from a ladder (medium -> hard -> M28 -> M28 with cheat
   multipliers) mixed with the network's own past versions (self-play), win/loss reward with the existing
   shaping at small weights. Earlier PPO runs failed because they started from policies that never won.
4. Longer games (the 30-minute cap already cuts won positions), 20 km maps, air and navy.

## 2026-10-07 measurements

Training memory and step time on the 4090 (`_runs/tools/bench_mem.py --gpu [--full]`, fused attention, 16 CPU games
running beside the full-batch runs):

| size | batch | peak memory | step |
|---|---|---|---|
| S | 8 x 40, stock attention | 2.9 GB | 1.6 s (first run, warm-up included) |
| S | 8 x 40 | 2.8 GB | 0.15 s |
| S | 8 x 40 + checkpointing | 2.2 GB | 0.19 s |
| M | 8 x 40 | 5.2 GB | 0.26 s |
| S | 16 x 40 | 5.8 GB | (warm-up run) |
| M | 16 x 40 | 10.8 GB | 0.51 s |
| M | 16 x 40 + checkpointing | 7.8 GB | 0.63 s |
| L | 16 x 40 + checkpointing | 14.1 GB | 1.67 s |

So M trains at the full batch without any trick, and L fits with checkpointing; the 24 GB filled on 2026-10-06 by the
stock attention layer's [batch, heads, T, T] weights. Losses and gradients of the fused path equal the stock path's.

Engine (upstream 8977bb9d + our 53 commits, `_enginetest/bin_port`): bridge smoke game fine; determinism check
identical over 3000 ticks; game speed equal to the old engine (26.8 game-seconds per second, sim 3.7 ms per tick
with ~160 units, M28 vs easy); loads in 11.7 s instead of 18.4 s.

## 2026-10-07: size M, first measurement

M (9.1 M weights, units decoder, token memory + two-pass) trained from scratch for 8,000 steps on the aggregate
(M28's 846 games + all DAgger rounds, rounds counted 3x; batch 8 x 40 accumulated 2x, checkpointing), on the
ported engine. Held-out at 8,000 steps: ordered 0.25, verb 0.21, blueprint 0.55 -- S's final level (0.28 / 0.24 /
0.59) with a fifth of S's total steps, and still rising. Clean, 48 games at temperature 0.5: **36/48 (75 %) vs
easy, 19/48 (40 %) vs medium** (S gen-2: 92 / 75; S's own pre-DAgger clone: 1/48). Next: DAgger rounds with M
(`night2/m28v2_M.cmd`).

## 2026-10-07 afternoon: DAgger with M

`night2/m28_v2_M`: M from the scratch clone, M28 as the teacher, 24 easy + 24 medium games a round, temperature
0.5, the new engine, 40-minute cap from round 5. In-loop win rates (M28 in the slot, 24 games each):
easy 63, 75, 75, 75, 88, 88, 92, 83, 100 %; medium 50, 54, 54, 67, 58, 67, 67, 54, 79 %. Clean check of the
round-9 network (48 games, 30-minute cap): **38/48 (79 %) vs easy, 35/48 (73 %) vs medium** -- the in-loop
numbers run hotter than the clean ones. Timeouts: vs easy the network is ahead but does not close; vs medium
they are even games; M28's labels there ask for attacks and transports (`_runs/tools/timeouts.py`).

Like-for-like on the new engine (48 games each, temperature 0.5, 30-minute cap, same seeds):

| network | vs easy | vs medium |
|---|---|---|
| S generation 2 (`mem2pass/s_mem.pt`) | 36/48 (75 %) | 32/48 (67 %) |
| **M after 9 DAgger rounds** (`m28_v2_M/checkpoints/round_09.pt`) | **38/48 (79 %)** | **35/48 (73 %)** |

S measured 44/36 on the old engine: the engine change itself costs it ~15 points (FAF's intel callbacks now run,
upstream's sim changes), and S's rounds were all played on the old engine. M, with nine rounds on the new engine,
is ahead on both opponents. Continue with M.

## 2026-10-07 evening: M DAgger, rounds 10-16 (easy, medium, adaptive; 48 games each)

In-loop (M28 in the slot, sampled at 0.5): easy 92, 88, 98, 96, 88, 98, 92 %; medium 65, 73, 79, 85, 73, 71, 75 %;
adaptive 52, 52, 42, 65, 40, 44, 60 %. Clean check of round 14 (48 games, 30-minute cap): **easy 40/48 (83 %),
medium 36/48 (75 %), adaptive 17 won / 3 lost / 28 timeouts** -- against adaptive it almost never loses, it
stalls. The clean checks now use the loop's 40-minute cap. M's held-out imitation loss has been flat since round
12 (ordered 0.27-0.30, verb 0.23-0.26: the level S plateaued at); the win rates still climb slowly because each
round's data is drawn from the states the network itself reaches. The imitation ceiling is in sight.

## 2026-10-08: the RL phase, prepared (not started)

Past the imitation ceiling the signal has to change: wins. What was missing in `fa_train.tools_trainer` for PPO
from the DAgger network, now there (commit after this entry; tests in `tests/test_factored_ppo.py`; the whole
pipeline exercised on fake episodes on the CPU):

- **Factored PPO for the units decoder** (`ToolsConfig.unit_ratio`, default on). The decoder reports each
  unit's own log-probability (its verb plus its arguments, `ent["unit_logp"]`, `ToolsNet.last_unit_logp` when
  acting). A decision's joint ratio is a product over hundreds of per-unit ratios: it leaves the clip range on
  almost every decision and trips the KL stop after a few minibatches, so the joint objective learns from
  almost nothing. Per unit, each choice is its own action with its decision's advantage; a decision's loss is
  the mean over its scored units. `decision_kl` is still logged next to the per-unit `approx_kl`.
- **A KL anchor to the frozen start network** (`--ref-coef`, `ToolsTrainer.set_reference`): the k3 estimator
  exp(d) - d - 1 with d = log ref - log policy on the sampled actions (per unit), unbiased for KL(policy || ref).
  The online teacher (M28 in shadow mode, `--distill-teacher m28`, weight fading 0.5 -> 0.2 and staying) only
  sees the states the policy visits; the anchor holds the whole distribution where the reward says nothing.
- **The M28 ladder** (`--ladder m28`, `LADDER_M28`): L1 medium -> L2 the four personalities -> M28 (`m28ai`)
  -> AIx M28 at x1.1, x1.25, x1.5, x2 (`m28aicheat@<mult>`, `/agentcheat`); each rung keeps 30-40 % of its
  games against the rung below; promotion at 75 % of the last 100 training games on the M28 rungs. `--start-stage`
  skips the rungs the network has already passed. Promotion is judged on sampled training games with M28
  in the slot, which run ~15 points above a clean check.
- Odds and ends: `--grad-checkpoint`, `--device`, `--eval-mode sample --eval-temperature 0.5` (the units
  decoder's verb choice is a rate; greedy play is not how it is measured), `--micro-seqs`, `--minibatch-seqs`.

Launcher: `_runs/research_20261005/night2/rl_M.cmd [checkpoint] [stop HH:MM]` -- to be run alone (16 workers,
one GPU process; stop the DAgger loop first). Value head and belief head come trained from cloning (returns in
the checkpoint's `ret_std` units, which the trainer adopts). Not done, and the next structural step after the
ladder: self-play against past versions needs the bridge to drive two armies in one game (`Me()` is one army,
`SinglePlayerLaunch` gives every other slot an AI brain) -- a multi-day engine/Lua change.

## 2026-10-08 morning: rounds 17-25, M28 joins the opponents

Clean checks (48 games each, temperature 0.5, 40-minute cap):

| network | vs easy | vs medium | vs adaptive |
|---|---|---|---|
| round 14 (30-minute cap) | 40/48 (83 %) | 36/48 (75 %) | 17 W / 3 L / 28 timeouts |
| round 19 | 44/48 (92 %) | 40/48 (83 %), 1 timeout | 29 W / 11 L / 8 timeouts |
| **round 24** | **45/48 (94 %)** | **46/48 (96 %)**, no timeouts | **34 W / 5 L / 9 timeouts (71 %)** |

The held-out imitation loss did not move over these rounds (ordered 0.28-0.31, verb 0.23-0.26); the gains are
on-policy. In-loop rounds 17-25: easy 85-98 %, medium 67-92 %, adaptive 54-77 % (round 25: 37 W / 5 L / 6 T).
Promotion applied after round 25 (the loop stopped at the round boundary and relaunched from the same run with
`night2/m28v2_M2.cmd`): easy dropped (94 %, one game short of the 95 % rule; medium at 96 % covers it), **M28
(`m28ai`) added** -- opponents now medium, adaptive, M28, 48 games each. Smoke games on 2026-10-08 00:30: the
round-15 network lost to M28 and to AIx M28 x1.5 in 7-8 minutes, so M28's games will be mostly losses at first;
what they add is M28's labels in the states a game against M28 produces.
