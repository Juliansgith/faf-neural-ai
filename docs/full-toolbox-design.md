# Full-toolbox design: every command, every blueprint, stronger opponents

The successor to [tools-design.md](tools-design.md). The network gets the whole game as a player has it: every
command a player can give with mouse and hotkeys, and the exact unit or building to make, from any tech level.
It learns strategy, tech and timing by itself. The rule teacher stays only as a short tutorial, and a ladder of
stronger opponents makes discovering tech necessary. Asked for on 2026-09-30 ("give it all the commands and
interactions ... it can eventually find out about tech levels"); the target named is beating M28 more than
90 % of the time.

Written before building it, like the earlier designs.

## Summary

| Question | Decision | Why |
| --- | --- | --- |
| What can it do? | Every command the bridge already issues (22 kinds, section 2) and **any blueprint** its builder or factory can make, chosen by a pointer | The 20 "kinds" of the tools version cover 239 of 743 blueprints (T1 and upgrades): no T2/T3 units, no experimentals, no commander upgrades |
| How does it know what a blueprint is? | Each candidate blueprint is embedded from its id and its stats (cost, build time, health, damage, range, speed, tech, categories) | Generalises across factions and to blueprints it rarely sees |
| Why not learn from scratch? | Random play never builds a working base, so it never wins and never gets a learning signal | AlphaStar needed imitation first; OpenAI Five needed ~45,000 years of play |
| Tutorial | The rule teacher, taught to tech up (T2/T3 factories and units, T2 power, shields, a commander upgrade); cloning, then distillation that fades **to zero** | It shows that tech exists; learning decides when and how much |
| Opponents | A ladder: FAF easy, normal, the four personalities, the cheating AIx at rising multipliers, M28 (with the user's go), then a self-play league | Against easy and normal, T1 spam wins, so it would never need tech |
| Self-play | Two network-controlled armies in one game, each with its own fog of war | The bridge already commands several armies; each needs its own observation |
| Network | The tools network plus a blueprint encoder and pointer, an enhancement head, and 8 command slots; size by the S/M/L study | The decoder must stay fast: CUDA graphs for acting |

## 1. Why a tutorial, not pure trial and error

Trackmania agents learn from nothing because they have about four controls and feedback every frame
(moving forward is reward). An RTS decision has thousands of options (which unit, which command, which
blueprint, where), and the reward arrives 15-30 minutes later. An economy built now pays off 10 minutes later.
Random play never stumbles on power, then extractors, then factory, then an army, often enough to learn from.
Night 1 showed how quickly even a good clone falls apart when its moves are noisy.

So the network starts from the teacher (about 1 000 games, cloned), and learning takes over from there. Unlike
the tools version, the teacher's weight then goes to zero, and nothing in the action space limits what it can
discover.

## 2. Action space

### Commands (all already in `ai/agentbridge.lua`)

| Verb | Subject | Arguments | Bridge action |
| --- | --- | --- | --- |
| build | one builder | **blueprint** (pointer over what it can build now) + placement (cell and spot, or next to a building) | `build` |
| produce | one factory | **blueprint** (pointer) + count 1 / 2 / 3 / 5 / 10 | `factory_build` |
| upgrade | one structure | **blueprint** when there is more than one upgrade | `upgrade` |
| enhance | a commander or support commander | **enhancement** (pointer over the ones available now) | `enhance` |
| move, attack-move, patrol | a set | cell | `move`, `attack_move`, `patrol` |
| attack | a set | enemy unit | `attack` |
| assist | a set of builders | own unit | `guard` |
| repair | a set of builders | own unit | `repair` |
| reclaim | a set of builders | wreck or enemy unit | `reclaim` |
| capture | a set of builders | enemy unit | `capture` |
| overcharge | the commander | enemy unit | `overcharge` |
| load | a set | a transport | `transport_load` |
| unload | a transport | cell | `transport_unload` |
| tactical missile | a launcher | enemy unit | `tactical` |
| nuke | a launcher | cell | `nuke` |
| build missile | a launcher | tactical or nuke | `silo_build` |
| rally | one factory | cell | `rally` |
| fire state | a set | return fire, hold fire, ground fire | `fire_state` |
| toggle | one unit | ability (shield, stealth, jamming, production) | `toggle` |
| pause | one unit | on or off | `pause` |
| dive | a set of submarines | none | `dive` |
| stop | a set | none | `stop` |

Self-destruct stays out. Commander-only verbs stay (the commander is never part of a set).

### The blueprint pointer

The candidates of a build, produce or upgrade are the subject's buildable blueprints, **as the engine allows
them right now**: the bridge adds `unit:CanBuild(id)` per candidate for idle builders and factories. That
covers FAF's HQ rule (support factories need an HQ of their tech), army restrictions and the unit cap. In the
recorded data a factory offers a median of 11 blueprints (at most 20), an engineer 15, a commander about 51.

Each candidate is embedded as `bp_embed(id) + MLP(stats)`, where the stats are mass and energy cost, build time,
health, damage per second, range, speed, tech level, and category flags (land, air, naval, structure, factory,
engineer, economy, defence, intel, shield, artillery, missile, experimental). A pointer scores the candidates
against the command so far. It replaces the `kind_s` / `kind_u` heads. Placement (cell and spot, or next to a
building) and count stay as they are.

### Enhancements

The bridge adds each commander's enhancements to `BlueprintInfo` (name, slot, mass, energy, build time, what it
requires, what it removes), and each commander's current enhancements to the unit description
(`unit:HasEnhancement`). Candidates are the enhancements available now; they are embedded like blueprints.

### Slots

Eight command slots instead of six: late games have more units and more to do. As now, a decision ends early at
`end`.

## 3. Observation additions

- Candidate blueprints: the union of what own builders and factories can make now (usually under 60), with their
  stats: the keys of the pointer.
- Per own unit: its enhancements, missiles in its silo, what it carries, fire state, toggles on, paused.
- Tech state: the highest tech of own and seen enemy factories, HQ factories per tech, enemy experimentals seen.
- The rest as in the tools version (tokens, 32 x 32 maps with the memory channels, economy numbers).

## 4. The teacher, version 3 (the tutorial)

The rule teacher learns to tech. Its choices stay simple; they only have to show that tech exists and roughly
when:

- upgrade a land factory to a T2 HQ once income passes about 20 mass per second, and to T3 at about 60; add
  support factories;
- T2 then T3 extractor upgrades (already there), T2 power at about 40 mass per second, and T3 power later;
- build T2 and T3 army units once the factories can: the strongest-per-cost tank or artillery of the tech;
- one early commander upgrade (UEF: the gun), shields over the base once T2 is up, T2 point and AA defence
  where the enemy attacks;
- keep the rules it has (economy, army, commander at home, storages and adjacency).

It records exact blueprint ids, so its actions are labels for the pointer as they are. Collection: about 1 000
games across all four factions and the ladder's lower rungs (section 6), in one night.

The teacher's distillation labels use the filter of the day run (only orders not already in effect), and the
invalid-label filter.

## 5. Learning

1. **Clone**: behaviour cloning on the new recordings, bf16 (about 25 minutes for 6000 steps).
2. **Fine-tune**: PPO at temperature 0.5, scored at the same temperature. Distillation (scored at temperature 1)
   fades from 1.0 **to 0** over the first third, rather than to the 0.1 anchor of the tools version, which keeps
   it near the teacher's style. The entropy brake stays; there is no entropy bonus (sampling at 0.5 explores
   already).
3. **League**, once it beats the scripted ladder, scaled down from AlphaStar:
   - the *main agent* plays everything: scripted AIs, M28, itself, and its past snapshots, with opponents chosen
     by how often it loses to them (prioritised fictitious self-play);
   - a *main exploiter* trains only against the current main agent, to find its weaknesses; its snapshots join
     the league;
   - a snapshot of the main agent joins the league every 50 updates.
4. **Rewards**: +10 for a win plus the faster-win bonus, -10 for a loss. Shaping (trades, economy, commander
   damage) fades to 0 by the league phase, which is decided on wins alone. A draw on time is worth 5 x the value
   lead while shaping lasts, and 0 in the league.

## 6. Opponents and gates

| Rung | Opponents | Promotion |
| --- | --- | --- |
| L0 | easy | 90 % of the last 100 |
| L1 | normal (medium) | 85 % |
| L2 | adaptive, rush, turtle, tech (FAF's four personalities, equally often) | 80 % against each |
| L3 | AIx (cheating) personalities at multiplier 1.0, then 1.2, then 1.5 | 75 % at each multiplier |
| L4 | M28 (downloaded only with the user's go) | 50 % |
| L5 | the league (section 5) | none: measured by rating |

Each rung keeps 20 % of its games against the rungs below, against forgetting. The AIx multiplier needs a new
launch switch (`/agentcheat <x>`, read by our hooks).

**Gates:**

| Gate | Test |
| --- | --- |
| F1 | The teacher, version 3, plays through the new executor as well as the current one: at least 90 % against normal, and it reaches T3 in most games |
| F2 | The clone agrees with the teacher on held-out games (at least 90 % of verbs and 80 % of blueprints) and wins at least 70 % against normal |
| F3 | Beats all four L2 personalities at least 80 % of the time |
| F4 | Beats AIx at 1.5 at least 75 % of the time |
| F5 | First wins against M28 |
| F6 | More than 90 % against M28 on Canis River; then the same on a pool of 1v1 maps |

Evaluations use fixed seeds per opponent, 10 or more games each, and an Elo rating over all league members and
scripted opponents.

## 7. Maps and factions

The blueprint pointer and the stats embedding make the network faction-independent. It trains on all four
factions from the tutorial on, on Canis River first. After gate F3, it adds a pool of five 1v1 maps (5 km and
10 km). The 32 x 32 cell grid fits those; 20 km maps would need a coarser grid or a zoom later.

## 8. Network and speed

- The tools network plus the blueprint encoder and pointer, the enhancement head, and 8 slots. The size (S, M,
  L) follows a study on the new recordings.
- Acting speed decides game throughput, and a bigger decoder is slower. Acting moves to CUDA graphs: the
  decoder is captured once for a few padded shapes (batch and token counts in buckets) and replayed as one
  launch instead of hundreds. Today a decision for 16 games takes about 170 ms, of which the encoder is about
  15 ms.
- PPO stays float32 for exact ratios. Cloning runs in bf16.

## 9. Throughput and time

Today this PC plays about 300-400 learning games an hour (20 at a time). Games with T3 and experimentals are
longer and heavier: expect 150-250. The engine work already noted would help most here: a training mode
without the user-side mirror, restarting a game inside the same process, and the pathfinding hotspot.

| Step | Work | Games |
| --- | --- | --- |
| 1 | Bridge (`CanBuild`, enhancements, per-unit state, the AIx switch, per-army observations) and the action space and observation | none |
| 2 | Teacher v3 and gate F1 | one-at-a-time tests |
| 3 | Collection (about 1 000 games), clone, gate F2 | one night |
| 4 | Fine-tuning through L0-L2, gate F3 | 2-4 nights |
| 5 | AIx ladder, gate F4 | 3-7 nights |
| 6 | M28, with the user's go: measure the gap first, then train against it (gate F5) | 3-10 nights |
| 7 | League to gate F6 | weeks |

These are estimates with wide margins. Learning is good at finding a fixed bot's repeated weaknesses, so beating
one M28 on one map can come earlier than general strength. Or learning can stall, which the size study and the
league are there for.

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| The bigger action space makes learning slower | Tutorial first; the stats embedding generalises across blueprints; distillation early |
| It learns to exploit fixed bots, not to play well | The league: self-play, past snapshots, an exploiter |
| Farming shaping rewards or drawing on time | Faster-win bonus; shaping fades to 0; draws are worth nothing in the league |
| Forgetting how to beat the lower rungs | 20 % of games against lower rungs; fixed-seed evaluations on every rung |
| Late games are slow to simulate | The engine throughput work (section 9) |
| Acting becomes the bottleneck | CUDA graphs for the decoder |
| Evaluations are noisy | 10+ games per opponent, fixed seeds, a rating rather than single win rates |

## 11. Build log

### Step 1: bridge and action space (2026-09-30, done)

What was built:

- `fa_env/tools.py`: 27 verbs. Build and produce take exact blueprint ids. There are also
  enhance, option (fire state, missile kind, toggle bit), and five target rows (enemy, own, damaged, wreck,
  transport). The commander may join assist and repair sets, and nothing else.
- `fa_env/tools_obs.py`: observation v3.
  - 70 token features (enhancements, missiles, cargo, fire state, toggles, pause, shield, what it can build).
  - Blueprint candidates: the union of `can_build`, with stats. `can_build` pairs say who builds what.
  - Enhancement candidates with `can_enh` pairs.
  - Build-cell masks per placement class (land, water, mass, hydrocarbon).
  - Tech-state scalars.
- `fa_train/tools_model.py`: blueprint and enhancement pointers, an option head, 8 slots.
- `tools_bc.py`: generic ragged cache packing (`*.v3.npz`).
- Offline and fake-pipeline tests pass:
  - 606/606 recorded commands round-trip;
  - acting and scoring log-probabilities agree exactly;
  - 29 fake PPO updates.

Found in the real-game check (one game, the teacher plus scripted checks, then an untrained network issuing every verb:
0 action errors):

- **Orders queued instead of replacing.** The engine's `Issue*` functions add to a unit's queue, like a
  shift-click. The tools version never cleared queues, so, for example, an attack-move given to a patrolling unit
  never ran. v3 clears first for movement and combat verbs, as a plain click does. A build queues only behind
  another build. Produce, upgrade and missile builds add to the factory's or silo's queue.
- **Toggles.**
  - In the sim, a blueprint's `ToggleCaps` is a string (`'RULEUTC_IntelToggle'`).
  - The engine's `Get/SetScriptBit` take the toggle's name, not its bit number. The bridge passed numbers, so
    toggling did nothing.
  - faf-re's `TestToggleCaps` tests the wrong blueprint field. This is an engine fidelity bug: FAF's
    `DisableDefaultToggleCaps` relies on it. The fix is a separate upstream-fixes task.
  - The bridge now reads the blueprint and passes names. Verified: the radar's intel toggle on and off, and the
    extractors' production toggle.
- Verified in the game:
  - fire state (hold fire);
  - pause and unpause;
  - enhancements: the commander starts the upgrade (Script order);
  - builds and production by blueprint.

  Tactical and nuclear missiles, transports and dive need T2 and above. They are next checked when the teacher v3
  reaches them.

Old recordings still load: builders without `can_build` fall back to the blueprint's list up to their own tech.
Old checkpoints do not load, because the network changed shape; v3 is cloned afresh in step 3.

### Step 2: teacher v3 and gate F1 (2026-09-30)

`fa_env/teacher_v3.py`. RulePolicy's economy, army and commander rules, through FactoredEnv's skills, plus a
tech layer that gives v3 commands directly:

- a T2 land headquarters at 20 mass/s, support factories after it, T3 at 50;
- production for every idle factory: engineers of each tech up to a target, then the army mix at the factory's
  best tech (the strongest unit per mass of each role; suicide units excluded);
- extractor upgrades from minute 8, always at least one;
- T2/T3 power when energy runs short, shields over the base;
- the commander's gun from minute 8;
- army decisions:
  - the army is counted by mass, not heads;
  - after minute 14 a big army goes for the base;
  - a visible enemy commander is attacked directly.

The rule army's attack targets now hold for up to 90 s, because orders replace (step 1). `examples/teacher_v3_game.py` runs
the gate, in parallel if asked; `fa_train/collect_v3.py` records teacher games for step 3.

Bugs found on the way:

- **Engine: category union** (faf-re patch 0029). `A + B` dropped B's members that sort before A's first
  member. `ENGINEER + FACTORY` lost every Aeon factory, so Aeon factories never reported what they can build,
  and the teacher never produced as Aeon.
  - This also crippled FAF's own AI wherever it adds categories: FAF's normal AI plays as intended only since
    this fix.
  - The bridge now tests categories one at a time.
- **Weapon stats counted death blasts.** They counted as damage per second: a Fire Beetle's 1 100-damage
  suicide blast made it the "best tank", and every commander's death nuke inflated its stats.
  `tools.combat_weapons` leaves out death, suicide, placeholder, overcharge and teleport weapons (observation
  and teacher).
- **Early investments starved the army.** The gun at minute 5 and extractor upgrades from minute 6 left the
  army too thin to hold normal's first attacks, and games dragged on to the time limit.

Gate F1: 10 games against normal (medium), seeds 11-20, all four factions, 30-minute limit. The FAF AI already
has the category fix.

| Teacher | Wins | Other games | Tech |
| --- | --- | --- | --- |
| v3 | 8/10 (median 22 min) | 2 timeouts, both ahead (+0.43, +0.50) | T2 at 8-14 min in every game past 10 minutes; T3 in both 30-minute games |
| v2 (the tools version's) | 8/10 (median 14 min) | 2 timeouts, ahead (+0.33, +0.47) | T1 only, apart from late factory upgrades |

v3 plays as well as the current teacher (the gate's first condition) while it techs. Neither reaches 90 %
against the now-intact normal AI; the 90 % was set against the broken one. Several wins in both rows are
normal's commander dying at our base around minute 9.

Then the T3 thresholds came down (headquarters at 35 mass/s, T3 extractors from 25): at 50, T3 arrived in only
1 of 5 long games against the personalities, because the upgrade itself takes minutes. With T3 at minutes 18-28,
30 minutes cut games off right as the T3 armies arrived (draws, well ahead). So, at the user's suggestion,
games now last up to **45 minutes**: in the gate, the collector and the training and evaluation games.

**Gate F1 passed** (45-minute games, 8-10 at once):

| Opponent | Wins | T3 | Median length |
| --- | --- | --- | --- |
| normal (medium), seeds 11-20, four factions | 10/10 | 7/7 of the games past 25 minutes (factory T3 at 18-28, T3 army at 20-30) | 30 min |
| adaptive, rush, turtle, tech (2 each) | 8/8 | 4/4 of the games past 25 minutes | 32 min |

A 45-minute game takes up to about 8.5 minutes of wall time with 8-10 at once, so 1 000 collection games with 20
at once should take about 5-6 hours.
