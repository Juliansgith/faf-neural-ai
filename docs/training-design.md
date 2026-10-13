# Training design

The plan for the first real training runs, written before any of them start. Step 2 of the project
plan: an agent that reliably beats FAF's `easy` AI, then `medium`, in 1v1 on Canis River (SCMP_016).
Steps 3 (richer decisions, stronger opponents, self-play) and 4 (Lua inference in the real game) build
on the same design and are sketched where they constrain it.

## Summary

| Question | Decision | Why |
| --- | --- | --- |
| What does the network control? | Macro decisions every 2 s through **factored heads** (build, produce, army, commander, tech); scripted skills execute them | Our compute is 2-3 orders of magnitude below AlphaStar/OpenAI Five; macro RL is what has worked at small scale (TStarBot) |
| How does it start? | **Kickstarted**: a rule policy written in the same action space, cloned into the network, then PPO with a decaying KL pull towards the clone | Random macro play almost never wins, so the win/loss signal would take very long to appear; the rule policy also proves the action space can win |
| Reward | Win/loss ±10, plus annealed dense shaping (trade, economy) | Dense terms carry learning early; annealing keeps the final objective "win" |
| Algorithm | PPO with masked categorical heads, an **asymmetric critic** that sees privileged (omniscient) state, GAE | Stable and well understood; the privileged critic gives far better advantages under fog of war at no cost to fairness |
| Infrastructure | **Asynchronous game workers** with batched GPU inference, auto-restart, TensorBoard, episode archive | Lock-step vector envs stall on every map load and on the slowest game |
| Opponents | easy → medium, then the other FAF personalities, then self-play (step 3) | A ladder with promotion gates |
| Success | ≥ 90 % victories vs easy and ≥ 75 % vs medium over 100 evaluation games (30-minute limit, fog of war, no cheats) | Victories, not scores: the game has to be won |

## 1. Where we start (measured)

Baselines on SCMP_016, 20-minute limit, agent UEF, AI faction rotating through all four
(`ai/python/examples/evaluate.py`, 2026-09-29). **These games ran without a victory condition:** FAF's
command-line launch defaults to `Victory = 'sandbox'`, where nobody can lose, so "time-out" below says
nothing about who would have won. Found afterwards; `GameConfig` now passes `/victory demoralization`
(an army loses with its commander). The economy and trade numbers stand.

| Agent vs AI | Games | Result | Detail at the limit |
| --- | --- | --- | --- |
| scripted bot vs easy | 16 | 16 time-outs, no winner | mass income 17–29/s, ~35 units; mass destroyed 1.2–2.4 k vs lost 6.2–7.3 k |
| scripted bot vs medium | 8 | 8 time-outs | half the games crippled (income 3–7/s, 13–15 units); destroyed 1.5–2.8 k vs lost 5.7–7.9 k |
| do-nothing agent vs easy | 4 | 4 time-outs | the commander was never attacked (0 mass lost) |
| do-nothing agent vs medium | 4 | 4 time-outs | same; FAF's own engine agrees (still alive at 39 min vs easy, 27 min vs medium: `tools/fidelity/passive_check.py`) |

Throughput (the scripted games above, one process per game):

| Games at once | Speed per game | Aggregate, including map loads |
| --- | --- | --- |
| 1 (earlier) | ~100× early game, 10–15× at 12 min with ~200 units | — |
| 8 (vs medium) | 5.9–13.2× | 43 game-seconds per second |
| 16 (vs easy) | 6.3–10.7× | 75 game-seconds per second |

A map load takes ~15 s alone and 20–65 s when 8–16 games load together; each game uses ~0.5 GB.

What this means:

- **The existing scripted bot is not a baseline to beat or a teacher to copy.** It sends 8-tank waves into
  a defended base and trades 1:3.5. A real rule policy has to be written (gate 1 below). The first rule
  policy in the factored action space (`fa_env/rule_policy.py`) wiped out `easy`'s army and economy by
  minute 15-20 in both test games (value advantage 1.0, trade 1.3:1 and 3.3:1) -- in sandbox mode. With
  `/victory demoralization` it **won** its three games against `easy` (commander killed at 14.8, 11.5 and
  11.0 minutes). After the fix in section 9 it won **12 of 12** 20-minute games: 4/4 against `easy` and
  **8/8 against `medium`**, in 13 minutes on average. The gate at the start of each night run measures it on
  more games.
- **Every launch needs an explicit victory condition.** Without one no game can end, and a learner
  would never see a win or a loss.
- **FAF's easy and medium AIs fight what they find; they do not hunt down a passive player** (no unit
  of the passive player was ever attacked, in either engine). Passivity is only punished by the
  time-limit score, and winning requires attacking, so the reward makes being ahead at the limit worth
  less than winning outright (±10 vs at most ±5).
- **The simulation is the budget.** About 75 game-seconds per second today, i.e. ~5,400 twenty-minute
  games or ~3 M macro decisions per day, before the throughput work in section 12.
- **Update: most of that budget was going to a debug probe.** Upstream's temporary sound diagnostics
  (`[SNDDIAG]`, commit d8f948b6) swept the sound registry and called `VirtualQuery` for every audible unit
  on every beat: 70–80 % of each game's main thread, which the sim thread then waited on (it ran at 62 %
  of a core). Our engine now skips that pass under `/nosound` (agent patch 0003). One game, same seed:
  game minutes 5–8 in 5 s instead of 10 s, the sim thread at 98 %. A minimized window also stops all
  drawing (`CScApp::Main` skips the device refresh), taking a game from 1.7 to 1.3 cores. Training games
  now run minimized. The first night run measures the new aggregate throughput.

## 2. Constraints that shape the design

**Compute.** One machine: RTX 4090, Ryzen 9 5900X (12 cores / 24 threads), 128 GB. The simulation, not the
GPU, is the bottleneck: the network is small and GPU inference for 24 games costs milliseconds. Every
design choice is weighed against *game-seconds simulated per wall-second*.

**Deployment.** The trained AI must eventually run as sim-side Lua in FAF's official engine
([deploying-a-trained-ai.md](deploying-a-trained-ai.md)). So:
- every actor input must be computable in sim Lua (today all of them come from `agentbridge.lua`);
- the deployed network must fit a few hundred thousand multiply-adds per decision. The training network
  may be larger if it is later distilled into a small student (step 4 measures the real budget).

**Fairness.** The actor sees only what its army's intel shows (fog of war, radar blips), like a player.
Privileged information is allowed only where it cannot leak into play: the reward and the critic.

**Fidelity.** We train on the rebuilt engine but would ship on FAF's. The replay check found and fixed the
gameplay bugs it could reach ([replay-fidelity.md](replay-fidelity.md)); behavioural spot checks continue
alongside training.

## 3. Control architecture

Three layers, of which the network is the top one:

1. **Engine and unit AI** — pathing, targeting, formation, weapon fire. FA units already fight on their own.
2. **Skills** (Python, deterministic) — turn one macro choice into bridge actions: pick the idle builder
   nearest the site, find a build site, queue units at the right factory, gather the army and attack-move.
   This is `MacroEnv._plan` grown up.
3. **Policy** (the network) — every decision interval, choose what each skill family should do next.

Unit-level micro is out of scope until the macro agent is strong; it is where compute requirements
explode, and FA's engine-side unit AI already covers much of it.

## 4. Action space v1 — factored heads

Each decision the policy picks one option per head, all at once (a product distribution; log-probability
is the sum over heads). Every head has `none`. Options that cannot act right now are masked.

| Head | Options | Executed on |
| --- | --- | --- |
| build | mex (nearest free deposit), power generator, land factory, air factory, radar, point defense, anti-air defense, hydrocarbon, reclaim (nearest field), assist (factory or construction) | the idle builder nearest the target (engineers first, then the commander) |
| produce | engineer, main tank, artillery, anti-air, scout (land); interceptor, bomber, scout (air) | an idle factory that can build it |
| army | attack nearest known enemy, attack enemy base, raid nearest enemy mex, defend base, retreat, hold | all combat units not already on that order |
| commander | none, stay home, join army, retreat | the ACU (plus a scripted rule: retreat below 40 % health) |
| tech | upgrade a mex, upgrade a factory, build T2 power | the cheapest candidate |

Why factored rather than one choice per step: an economy needs builders, factories and the army busy at
the same time. With one macro per 2 s the agent spends its decisions on book-keeping; independent heads
let it keep every part of the base busy at once, with no extra steps. Masks keep every sampled action
executable, so failed actions are not a learning problem.

Locations are chosen by the skills in v1 (nearest free deposit, base area, enemy start). Letting the
policy pick map regions (a spatial head over a coarse grid or a pointer over candidate sites) and army
groups is step 3; the actor gains a map encoder together with that head (section 7).

Decision interval: 20 ticks (2 s). Faster decisions add cost without adding macro quality; event-driven
decisions (factory became idle) are a later optimisation.

## 5. Observation v1

**Actor inputs** (all fog-of-war filtered, all computable in sim Lua):

- *Map planes* (step 3, with the region head; the critic uses them from v1) at 32×32: terrain height,
  water, mass/hydro deposits (free / own / enemy-held), own mobile / structures, enemy mobile / structures
  (seen now), **enemy structures seen before** (FA keeps their blips), radar contacts, own vision
  coverage, reclaimable mass. 64×64 buys nothing on a 10 km map at this level of control and costs 4× in
  the CNN.
- *Scalars*: time, mass/energy stored, capacity, income, spend, trend; stall flags; unit-cap use; counts
  per role (engineers, factories by type, mexes, power, tanks, AA, defenses) including under construction;
  idle builders and factories; commander health; enemy counts seen by role; the last action per head and
  whether it succeeded.
- *Unit table*: up to 256 rows (own first, then seen enemies, then radar contacts), each with position,
  health, build progress, idle/working, heading, and **static blueprint features** (log mass cost, health,
  DPS, range, speed, tech level, role flags) next to a learned blueprint embedding. Static features let
  the network generalise to units it has rarely seen; a learned embedding for 743 blueprints alone would
  not train from a few thousand games.

Memory: v1 uses explicit memory features (previously seen structures, last-seen enemy army composition
with decay) and no recurrent core. A GRU is the planned upgrade once the macro agent works, when the
agent needs to reason about what the enemy is doing out of sight.

**Critic-only (privileged) inputs**: the enemy's true economy, army and structure value, unit counts by
role, and commander health. The critic estimates the value of a state; knowing the true state makes that
estimate far better under fog of war, and it is never used to choose actions.

## 6. Reward

```
r_t = terminal + c(t) * ( trade_t + economy_t )
terminal   +10 victory, -10 defeat; at the time limit 5 * tanh(value advantage) using privileged totals
trade_t    delta(enemy mass destroyed - own mass lost) / 1000       (existing default)
economy_t  delta(mass income) / 20
c(t)       1.0, annealed to 0.2 over the first 60 % of training
```

- The terminal reward is what the agent should maximise; shaping gets it there faster. Both dense terms
  are differences of a state function (potential-style shaping), so they pay for progress, not for
  repeating an action, and they barely move the optimum.
- Watch for known failure modes: an army that avoids every fight (trade hacking), economy over defense
  (eco hacking), stalling to the time limit. Monitor action histograms and game length per opponent.
- Returns are normalised by a running standard deviation for the value head.
- The time limit is part of the task and game time is in the observation, so reaching it is a true
  terminal (scored by the value advantage), not a truncation to bootstrap through. Only crashes and
  restarts are truncations.

## 7. Network

**Actor (the part that will ship).** In v1 the skills choose every location, so the actor needs to know
*what* the state is, not *where* things are on the map at pixel level:

- Unit encoder: MLP(static bp features ‖ unit features ‖ 16-d embedding) → 64, then masked mean and max
  pooling separately over own units, seen enemies, previously seen enemy structures and radar contacts.
- Scalars: MLP → 128.
- Torso: concatenation → 256 → 256; five masked categorical heads.
- About 150–300 k parameters and well under a million multiply-adds per decision: small enough for the
  Lua runtime of step 4 without distillation, and cheap to featurize.

The map encoder (3 convolutions, 32×32 → 8×8) arrives in step 3 together with the head that picks map
regions, where spatial structure starts to matter for the actor.

**Critic (training only).** The same encoders, plus the map encoder and the privileged scalars, as a
separate tower: sharing the actor's torso would let the value loss pull the policy's features towards
privileged information. It is never deployed, so it can be as large as helps (≈ 2 M parameters).

Batched inference for all games takes a few milliseconds on the GPU; the network is not the bottleneck.

## 8. Algorithm

PPO (clipped), with:

| Setting | Value | Note |
| --- | --- | --- |
| discount γ | 0.997 | half-life ≈ 230 decisions ≈ 8 game minutes |
| GAE λ | 0.95 | |
| rollout | 128 decisions per game segment | segments from async workers |
| batch | ~3,000 decisions (24 games × 128) | |
| minibatches × epochs | 4 × 3 | early stop when approximate KL > 0.02 |
| learning rate | 2.5e-4, linear decay | Adam, ε 1e-5 |
| clip | 0.2 (policy), 0.2 (value) | |
| entropy | 0.01 → 0.003 | per head, summed |
| gradient norm | 0.5 | |
| KL to the cloned policy | β 0.5 → 0 over the first 30 % | kickstarting |

Workers run slightly behind the newest weights (asynchronous collection). At our lag (one update) plain
PPO ratios handle it; V-trace is the fallback if lag grows.

## 9. Warm start

1. **Rule policy in the new action space.** Encodes a sound opening and macro loop: saturate nearby mass,
   add power ahead of spend, factories by income, keep factories busy, engineers then tanks with an
   artillery/AA mix, attack when the army is large enough, retreat when outnumbered.
   *Gate:* it must win ≥ 90 % of 30-minute games against `easy`. If a hand-written policy cannot win in
   this action space, RL will not either, and the action space must change first. This is also the new
   baseline (the existing scripted bot wins none, see section 1).
2. **Behaviour cloning.** 300–500 games of the rule policy (against easy and medium, several factions) →
   cross-entropy per head. *Gate:* ≥ 95 % agreement on held-out games and a win rate close to the rule
   policy's. The rule policy is deterministic, so an exact clone would be too, and PPO could never try
   anything else. The targets are therefore smoothed: 5 % of the probability is spread over the options
   that were allowed (never over masked ones), and the clone's entropy per head is reported.

   **The teacher may only decide from what the network sees.** The first clone check (12 games) failed.
   When the teacher acted, the clone matched only 33 % of its build choices and 60 % of its produce
   choices, and the greedy clone won none of 3 games against easy. There were two causes:
   - *Too few gradient steps.* 4.5 k rows × 8 epochs was 72 steps. Cloning now runs at least 10 k steps
     and keeps the checkpoint with the best agreement on held-out whole games.
   - *Hidden state.* The land-unit mix followed a private tally of past orders, which no observation
     contains, so even a perfectly fit network matched it on only ~55 % of held-out rows. The teacher now
     picks the type furthest below its share of the *current* army. That is learnable, and it plays
     better: it won 8/8 against medium instead of 1 win and 1 time-out.

   With both fixes, a clone of just 12 games won 2 of 3 against easy. The night run clones ~300 games.
3. **PPO from the clone**, with the KL term keeping early updates near it so the first thousand games are
   not spent unlearning a working opening.

## 10. Opponents and curriculum

| Stage | Opponents | Map, factions | Promote when |
| --- | --- | --- | --- |
| A | easy | SCMP_016; agent UEF; AI faction random | ≥ 90 % over 100 eval games |
| B | 30 % easy, 70 % medium | same | ≥ 75 % vs medium |
| C (step 3) | medium, adaptive, rush, turtle, tech; sampled towards the ones it loses to | + 3–5 more 1v1 maps; agent all factions | ≥ 60 % vs each |
| D (step 3) | + past checkpoints of itself (league), cheating AIs, M28AI / RNGAI | | |

A single map and faction first: the question in step 2 is whether the pipeline learns at all; breadth
comes after.

As implemented (`fa_train/trainer.py`, `STAGES`), promotion from A to B is judged on the last 100
*training* games of the stage (≥ 90 % won). Those are played by the sampling policy, a harder test than
greedy evaluation, and there are far more of them than evaluation games. B → C waits for step 3.

## 11. Evaluation

- A fixed suite per opponent: 4 AI factions × 8 seeds, spawn positions alternating; greedy and sampling
  policies. Run on saved checkpoints by separate eval workers, never mixed into training data.
- Per game: result, length, mass income at 5/10/15 min, army value over time, mass traded, action
  histogram, commander deaths. Per checkpoint: win rate with a 95 % interval, Elo against the AI pool.
- Every game's action log is kept; the simulation is deterministic, so any game can be re-simulated and
  watched.
- Before any release claim: the same policy through the Lua runtime in FAF's official engine (step 4).

## 12. Infrastructure to build first

| Piece | What | Why |
| --- | --- | --- |
| Async worker pool | one process per game, stepping independently; the learner batches whatever observations are ready onto the GPU | lock-step vector envs wait for the slowest game and for every map load |
| Pre-launch | each worker starts loading its next game while the current one plays its last minutes | a load takes 15 s alone and 20–65 s when many games load at once; hidden, it costs nothing |
| Auto-restart | a crashed, hung or watchdog-killed game is relaunched; the episode is cut (bootstrapped), counted and logged with its seed | long runs must survive the odd engine crash |
| Factored macro env | the action space above, masks, skills, memory features, privileged critic features (bridge change), random start position for the agent | today the agent always takes the first start position |
| Logging | TensorBoard (win rate per opponent, returns, entropy, KL, clip fraction, explained variance, decisions/s, game-seconds/s, restarts) + JSONL per episode | |
| Checkpoints | every N updates, with optimizer state, config and code version; resumable | |
| Headless mode | measure; if the render thread costs real CPU, add an engine switch that skips rendering | more games per machine |

Status: all of the above exist except pre-launch. Headless came for free: a minimized game skips all
drawing (`GameConfig.minimized`), so no engine switch was needed. The measurement also found the probe
described in section 1. The unattended runner is `fa_train/night.py` (`ai/night.cmd`). It runs gate,
collection, cloning, PPO with the curriculum, and a final evaluation, then writes a report. It keeps
Windows awake while running and recovers from a failed phase (PPO resumes from its last checkpoint).

## 13. Budget

| | Today | Target after section 12 |
| --- | --- | --- |
| Aggregate throughput | 75 game-s/s (16 games) | 120–150 game-s/s (20–24 games, loads hidden, headless if it pays) |
| 20-minute games per day | ~5,400 | ~9,000–11,000 |
| Macro decisions per day | ~3 M | ~5–6 M |

Expected needs (estimates from comparable macro-level RTS work, to be replaced by measurements):

| Phase | Decisions | Wall time at target |
| --- | --- | --- |
| Behaviour-cloning data (500 rule-policy games) | — | ~1–2 h |
| Stage A (easy), from the clone | 1–3 M | ½–1 day |
| Stage B (medium) | 3–10 M | 1–3 days |

**Stop rule:** if a stage shows no improvement over its starting point after ~2 M decisions, stop and
diagnose (reward, action space, features) rather than buying more compute. At this scale the design, not
the run length, decides the outcome.

## 14. Risks

| Risk | Mitigation |
| --- | --- |
| The rule policy cannot beat easy in this action space | fix the action space first (gate 1) |
| Too slow late game (navigation is ~40 % of a late-game tick) | time limit, unit-cap, profile-guided engine optimisation of the swept-path test |
| Reward hacking | annealed shaping, monitoring, the terminal reward dominates later |
| Overfitting to one map and AI | stage C breadth; held-out maps in evaluation |
| Opponents differ from FAF's (engine gaps) | behavioural checks; final evaluation in the official engine |
| Engine crash/hang mid-run | watchdog + auto-restart + crash log per seed |
| Forgetting when opponents change | keep easy in the mix; league in stage D |

## 15. Work plan and gates

1. Infrastructure: async workers, auto-restart, logging, checkpoints, evaluation harness.
2. Factored macro env + rule policy → **gate 1** (rule policy ≥ 90 % vs easy).
3. Behaviour cloning → **gate 2**.
4. PPO stage A → **gate 3** (≥ 90 % vs easy).
5. PPO stage B → **gate 4** (≥ 75 % vs medium). Then step 3.

## 16. Open questions

1. **Machine time.** *Answered:* full speed (20 games) only at night, when you say so; at most one game
   while you use the PC.
2. **Faction.** Default: UEF only for stages A and B, all four factions from stage C.
3. **Game length.** Default: a 20-minute limit in training (scored by value advantage), 30 minutes in
   evaluation so real victories can be counted.
4. **Later, not now:** downloading M28AI/RNGAI from the FAF vault (stage D) and, possibly, human replays
   for imitation learning. Both are downloads I will ask about when we get there.
