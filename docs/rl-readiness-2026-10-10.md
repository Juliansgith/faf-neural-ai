# RL readiness and improvement points (2026-10-10 morning)

For the decision to start reinforcement learning. Everything below was built or measured on the night of
2026-10-09/10; nothing RL-related has been started.

## Where the network stands

Clean checks, 48 games per opponent, temperature 0.5, the same seeds (`_runs/research_20261005/night2/clean`):

| network | vs medium | vs adaptive | vs M28 |
|---|---|---|---|
| round 30 (last DAgger round) | 39 W / 8 L / 1 T (81 %) | 36 W / 6 L / 6 T (75 %) | 0 / 48 |
| **second-pass fix** (`_runs/server/vaa_M/checkpoints/step_01.pt`) | **45 W / 3 L (94 %)** | **41 W / 4 L / 3 T (85 %)** | **1 / 48** |

The fix: the units decoder chose its final verbs from logits computed *before* the attention over the own units
(an exact copy of the first pass), so the second pass reached only the arguments. A model option
(`verbs_after_attention`, saved with the checkpoint) decides them after it; existing networks are unchanged.
The fixed network is the one the server plays now.

## What is ready for RL

- **The server records the network's own choices** (log-probability per decision and per unit, value estimate,
  chosen actions) with every decision, and the reward components. Replaying a recorded game through the network
  reproduces the recorded log-probabilities exactly (`_runs/tools/policy_replay_check.py`: 0.0000 difference).
- **`fa_train.learn_rl`**: PPO on recorded games. Games run as parallel streams with the memory carried as live,
  values and GAE from the recorded rewards, factored per-unit PPO, a capped KL anchor to the starting network,
  M28's labels as a fading cloning term (averaged per unit, like the policy term), learning-rate warm-up, KL early
  stop. Tested on CPU (stable over three epochs) and at full scale on the GPU: 126 server games (64,642 decisions)
  from round 30, one epoch = 174 optimizer steps in 7.6 min; the cloned value head explains 96 % of the return
  variance on real games; KL to the recorded policy 0.0044, 3.7 % of the unit ratios clipped, anchor KL 0.004.
  An iteration on ~120 games takes 8-14 minutes: about what the server needs to play the next ~120.
- **`fa_train.rl_loop`**: pull the server's rounds, one PPO iteration, push, again. Uses only games played by the
  current or the previous network.
- **The online trainer** (`fa_train.night_tools`, PPO with games on this PC) stays available as the alternative.

Proposed first run: `rl_loop` with the server, rounds shortened to 1 game per opponent per slot (about 20 minutes
a round, so the data is never more than ~30 minutes old), opponents medium / adaptive / the four personalities
(M28 later, when it wins some), the cloning weight starting at 0.3, the anchor at 0.2, learning rate 1e-5.
A clean check every ~6 iterations; stop and roll back if adaptive drops more than 10 points.

## Improvement points, most important first

1. **The economy stops growing after minute 5 against M28.** Mass income ours vs M28's: minute 3 9 vs 15,
   minute 5 18 vs 22, minute 8 18 vs 40, minute 12 17 vs 58; engineers 15 vs 32 at minute 8, 13 vs 46 at minute 12
   (42 server games, round 30). By minute 12 M28 out-produces us three to one. Only 7 of 42 commanders died before
   the end: the games are lost to an economy and army gap, not to commander snipes.
2. **Imitation cannot fix it: M28's shadow advice is conservative in our games.** In those positions the network
   already does what M28's labels ask (mex builds 0.04 vs 0.07 per decision, engineers 0.09 vs 0.07, reclaim and
   army the same). M28 in shadow mode never sees its orders executed, so its own bookkeeping (which engineer goes
   where) drifts from the game, and its advice gets cautious. The held-out imitation loss has been flat since
   round 12. **RL is the way past this**: the reward has an economy term and wins depend on outgrowing the opponent.
   A larger economy shaping weight in the first RL iterations is worth trying.
   Confirmed by the first gated cycle (03:11): 3,000 more cloning steps from the fixed network on all data incl.
   ~880 server games left the held-out imitation scores unchanged (orders 0.299 vs 0.296) but played worse:
   medium 43 vs 45, **adaptive 29 W / 7 L / 12 T vs 41 / 4 / 3**, M28 0 vs 1. More imitation of the cautious
   advice made it less decisive (timeouts 3 -> 12). The gate kept the fixed network.
3. **Opponent variety.** Training has used easy, medium, adaptive and M28 only; rush, turtle and tech
   personalities exist and cost nothing to add. A robust policy should see them before M28.
4. **Self-play** (two networks in one game) needs the bridge to drive two armies: about 1-2 days of Lua/bridge work
   (`Me()` is one army in 6 places; the launch hook already has a two-slot mode). Valuable once the ladder's fixed
   AIs stop teaching.
5. **Server throughput is CPU-bound** (16 cores also make the decisions on the CPU). Measured: batching decisions,
   bf16, separate Wine prefixes, observation frequency: no gain; one PyTorch thread per lane: +5 %; 8-bit network:
   1.6x on the network but too lossy (99th-percentile per-unit log-probability difference 2.5). The server plays as
   many games as this PC, and both together double the total.
6. **Size L** only after RL shows M saturating: imitation capacity is not the limit now (item 2).
7. Later: 20 km maps, navy and air.

## Night of 2026-10-09/10, other changes

- Corpus opens in 16 s instead of 6-40 min (cache header index, lazily opened memory-mapped games).
- The training copy of the corpus on F: is 92 GB instead of 245 GB (compact layout, identical batches).
- Recovery checkpoints are deleted after each round; finished rounds' recordings go to `D:\faf-cold`.
- Gated cycles (`_runs/tools/cycle.py`): pull the server's games, clone on, 3 x 48 test games against the best,
  push only when not worse.
