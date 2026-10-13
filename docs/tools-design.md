# Tools design: commands with arguments

The successor to the menu of 34 macro options in [training-design.md](training-design.md). The network
issues the game's own commands and fills in every argument itself: who, what, where, how many. Scripts
keep only what the engine also does for a human player: pathfinding, formations, and the exact legal
spot for a building inside the map cell the network chose.

Written before building it, like the first design. Asked for on 2026-09-29: "make the tools version,
investigate if a bigger network will make it smarter, do whatever works best to make it smartest, try
new techniques."

## Summary

| Question | Decision | Why |
| --- | --- | --- |
| What does the network control? | Up to 6 **commands per decision** (every 2 s), each a verb with arguments: build *kind* at *cell and spot*, or *next to one of our buildings*, with *builder*, produce *n* × *kind* at *factory*, move / attack-move / patrol a *set of units* to *cell*, attack *enemy unit*, assist *unit*, reclaim *wreck*, upgrade *structure*, rally *factory* to *cell*, stop | The menu version decides only *what*; scripts decide who, where and how many, and that caps how good it can get |
| What does it see? | Every unit as a token (own, enemy in sight, remembered buildings, radar contacts, mass and hydrocarbon spots, wrecks) with what each own unit is doing; a 32 × 32 map; two memory maps (where it last looked, where enemies were strong); economy numbers | Arguments need entities to point at and a map to point on |
| Network | Entity transformer + map ResNet with entity scatter + LSTM memory + an autoregressive command decoder with pointer, per-unit selection and map heads. Sizes S / M / L (≈0.5 M / 3 M / 12 M weights) | Structure that matches the arguments; size chosen by measurement (section 6) |
| How does it start? | Behaviour cloning of the rule teacher's *actual commands*, from raw recordings of its games | The teacher already wins 8/8 against medium; the recordings need no replaying |
| How does it improve? | Recurrent PPO; **online teacher distillation** (the teacher labels the states the student reaches, fading out); a **belief head** that learns to predict hidden enemy state | Faster, steadier learning from the same number of games |
| Data | **Raw recordings** of games (the game's own messages, the commands sent) | Any future network or action space trains on games already played |

## 1. Why the menu version cannot get much smarter

Each menu option runs a fixed script. "Build power" always takes the nearest idle builder and the first
legal spot around the base. "Attack nearest" sends every army unit to one point. The network can only
choose *when* to do these things. A network that learns a lot is still capped by the scripts. It cannot
expand to a far mass spot first, split its army, defend one flank while raiding another, pull back a
damaged group, or build defenses at a choke point. Those are the decisions that separate good play from
average play.

## 2. Commands

Each decision emits 0 to 6 commands, one after another; `end` stops early. Each command conditions on
the ones before it in the same decision.

| Verb | Subject | Arguments | Game command | What scripts still do |
| --- | --- | --- | --- | --- |
| build | one builder (engineer or commander) | structure kind; *place*: a map cell and a spot in it, or next to an own building (*anchor*) | `build` | the nearest legal spot to the chosen spot (`find_build_site`), or a legal spot touching the anchor; mass and hydro snap to the deposit in the cell |
| produce | one factory | unit kind, count 1 / 2 / 3 / 5 / 10 | `factory_build` | kind → the factory's blueprint |
| move, attack-move, patrol | a set of mobile units | map cell | `move`, `attack_move`, `patrol` | pathfinding |
| attack | a set of mobile units | an enemy (pointer) | `attack` | |
| assist | a set of builders | an own unit (pointer) | `guard` | |
| reclaim | a set of builders | a wreck (pointer) | `reclaim` | |
| upgrade | one structure | — | `upgrade` to its next tech level | |
| rally | one factory | map cell | `rally` | |
| stop | a set of units | — | `stop` | |

Later (step 3 onward, same mechanism): commander upgrades (`enhance`), capture, overcharge, transports,
T2/T3 kinds.

*Kinds* are faction-independent: 11 structure kinds (mass extractor, power generator, hydrocarbon plant,
land factory, air factory, radar, point defense, AA defense, mass storage, energy storage, wall) and 9 unit
kinds (engineer, tank, artillery, mobile AA, land scout, interceptor, bomber, air scout, air transport).
The executor resolves a kind to the blueprint the chosen builder or factory can make.

*Map cells*: a 32 × 32 grid over the playable area, 8 world units per cell on Canis River. Masks allow only
cells where the kind can be built (a free deposit for extractors and hydrocarbon plants, dry and flat enough
ground for the rest) and cells on the map for movement.

*Placement and adjacency.* A building touching another gets FAF's adjacency bonus: storages raise an
extractor's output (a T1 extractor 2.0 → 3.0 mass/s with 4 storages, measured), power generators cut a
factory's energy use (25 → 13.6-17.5 energy/s, measured). So a build goes either
- to a *spot*: one of the 4 × 4 spots of the chosen cell (2 world units on Canis River), or
- *next to* an own building (a pointer): the bridge's `find_build_site` with `adjacent_to` slides the new
  building along the anchor's four sides and returns the first legal spot that touches it under the
  engine's rule (`Unit::OverlapsWith` on the skirt rectangles: one pair of opposite edges less than 1 apart
  and one side's extent inside the other's; corners do not count).

The teacher places power against factories and mass storages against extractors (highest tech first),
so its recordings teach both. Labels come from where the building actually went: a building touching an
own structure is labelled *next to* it (the one its bonus is for, if several), anything else *cell + spot*.

*Sets of units*: an include/exclude choice per eligible unit, all made at once from each unit's own
features and the command so far. This can express anything from "one scout" to "the damaged tanks on the
left flank" in a single command.

Invalid commands (no legal spot, a dead target) fail and are reported back. The next observation
carries the success rate per verb.

## 3. Observation

**Tokens** (up to 512): own units; enemies in sight; remembered enemy buildings; radar contacts; every
mass and hydrocarbon spot with its owner (free, own, enemy, unknown); up to 32 wrecks worth reclaiming. Each
token has:
- what it is: blueprint embedding plus static stats (cost, health, damage, range, speed, tech, type flags);
- where it is: position, and offsets to its own base and to the enemy base;
- its state: health, build progress, idle, work progress, queue length;
- **its current order and destination** (from the unit's command queue, new in the bridge);
- decisions since the agent last commanded it;
- for buildings: how many own buildings touch it (engine adjacency rule) and its actual mass and energy
  output and energy use (the bridge reads them from the unit, so adjacency bonuses show). FAF raises an
  upgrading extractor's production value while mass stalls and the engine scales it back down, so these
  readings are capped.
- movement: heading and velocity (from its previous position).

**Map** (32 × 32): height, water, slope, deposits, densities of own, enemy, remembered and radar units
(mobile and structures). Two memory maps kept by the environment:
- *last looked*: time since own vision covered each cell, from each unit's vision radius;
- *enemy strength*: the strongest enemy presence seen there, decaying over a few minutes.

**Numbers**: economy (stored, income, requested, trend), game time, counts, per-verb success of the last
commands.

## 4. Network

```
tokens ─► MLP ─► transformer (L layers) ─────────────┬──► pointer / selection heads
map planes + tokens scattered into cells ─► ResNet ──┼──► map (cell) head
numbers ─► MLP                                       │
       └──► pooled tokens + map summary + numbers ─► LSTM (memory) ─► command decoder (4 slots)
                                                                     └► value (+ privileged, training only)
                                                                     └► belief (predict hidden enemy state)
```

- **Command decoder.** A small recurrent cell over the command slots. Each slot runs these heads in order:
  1. verb;
  2. subject (a pointer for one unit, per-unit selection for a set);
  3. kind;
  4. count;
  5. place (build only: in a cell, or next to a building), then the anchor (a pointer over own buildings);
  6. cell (the slot's query against the map features of every cell), then the spot in it (builds);
  7. target (a pointer).

  Every choice is embedded and fed to the next head and the next slot.
- **Sizes** (width / transformer layers / map channels / LSTM): S 64 / 2 / 32 / 256 · M 128 / 3 / 64 / 512 ·
  L 256 / 4 / 96 / 1024. Weight counts are measured when built.
- The **critic** shares the encoder and core. It adds the privileged vector (true totals of both armies and a
  coarse enemy map) only at the value head, so nothing privileged reaches the actor.

## 5. Learning

1. **Cloning from recordings.** The teacher's recorded game commands are turned into labels: verb,
   subject(s), kind, count, place, anchor, cell, spot, target. Training uses masked label smoothing and the best held-out
   checkpoint, and holds out whole games.
2. **Recurrent PPO.** Segments of 64 decisions carry the LSTM state from their start. A decision's
   log-probability is the sum over its commands and arguments. Entropy is weighted per argument type;
   selection entropy is divided by the number of eligible units.
3. **Online teacher distillation** (DAgger-style). The teacher is a program, so it can say what it would
   do in *any* state, including the ones the student gets itself into. Workers attach the teacher's commands
   to every observation. An imitation loss on them starts at weight 1 and fades to 0 over the first 30 % of
   training. It replaces the KL pull towards a fixed clone: the clone's own mistakes get corrected instead
   of preserved.
4. **Belief head.** From its memory, the network predicts what it cannot see: enemy army and structure
   mass, income, factory count, tech level, and an 8 × 8 enemy map. Targets come from privileged
   information, weight 0.2. It costs no games, and it teaches the network to infer the enemy from scouting.
5. Rewards, curriculum and evaluations as in the first design.

## 6. Does a bigger network make it smarter? How we find out

Size is not free, and bigger is not automatically better at this data scale: it needs more games to fill,
and runs slower in the real game later. Measure, then decide:

1. **Offline, on recordings.** Train S, M and L on the same recorded teacher games. Compare held-out
   argument accuracy, and the win rate of each clone in a few games. Recordings make this cheap: no new
   games.
2. **Online, twin runs.** Two learners on one night, 10 games each, same opponents and seeds, M against L.
   Compare win rate at equal numbers of games played.
3. **Rule.** Use the smallest size within noise of the best. Revisit whenever the data grows by 10×.
   Until measured, the default is M.

Running inside FAF's Lua later needs a small network. If a big one wins, it is distilled into a small
student then (step 4); size for now is chosen for strength.

## 7. Data: raw recordings

`fa_env/record.py`, on since 2026-09-29 for every teacher game of a night run: one gzip JSON-lines file
per game. It holds each decision's observation message exactly as the game sent it, the commands sent, and
the query answers, plus the game summary. Blueprints are stored once per folder. About 20 KB per game
minute.

## 8. Work plan and gates

1. Raw recordings. *Done* (`fa_env/record.py`).
2. Bridge: each own unit's current order and destination. *Done.*
3. Tools executor, plus the teacher speaking tools: the rule teacher's commands go through the new
   executor. **Gate T1:** it plays as well as through the menu executor (≥ 90 % against easy over 10 games,
   one at a time). This checks that the command vocabulary covers what a good player needs.
   *First result:* it beat medium with 0 failed orders out of ~550 commands (17.2 min; the same game
   through the menu executor: 11.9 min). A version that aimed every building at the base's centre cell
   dominated but did not win in 30 minutes. Building placement is a real decision: a cell chosen well
   (open, near where the building belongs) is what makes it work. The 10-game run is still to do.
4. Encoder, label extraction from recordings, the network, cloning, and the S / M / L study.
   **Gate T2:** on held-out games the clone agrees with the teacher on ≥ 95 % of verbs, ≥ 90 % of
   subjects and ≥ 80 % of cells (within one cell), and wins ≥ 70 % against easy.
   *First result*, M cloned from 25 games (8.8 k decisions, 17 minutes): held out verb 0.83, subject
   0.94, per-unit selection 0.96, kind 0.70, cell 0.84 (0.86 within one), target 0.73. Playing on its
   own, it won 1 of 2 against easy. It lost the other game while ahead on value: its commander died,
   i.e. it has not yet learned the teacher's retreat-when-hurt rule. The metrics had levelled off, so
   this is limited by data, not by capacity. The size study waits for the first night's 200+ games.
5. Recurrent PPO with distillation and the belief head. *Done;* fake-mode tested (background updates,
   distillation fading, evaluations, checkpoints); one-game real test run.
6. Adjacency and finer placement (asked for before the first tools night): place / anchor / spot heads,
   adjacency-aware `find_build_site`, the teacher building mass storages and placing power against
   factories, building features. *Done;* checked in a real teacher game (every storage touched an extractor;
   T1 extractors 2.25 / 2.75 / 3.0 mass/s with 1 / 3 / 4 storages) and on its recording (31 of 51
   placeable builds labelled next-to, none dropped). Recordings made before this lack the skirt and
   production fields and are not used.
7. A night run with the tools version (`ai/night_tools.cmd`). **Gate T3:** it beats its clone against
   medium.
   *Night 1 (2026-09-30), stopped at 08:45.* The teacher won 598 of 600 recorded games, and the clone (M, 540
   games) agreed with it on held-out games on 92 % of verbs, 96 % of subjects, 95 % of kinds and cells, and
   99-100 % of placements. Learning failed: from the first hour the network lost about 80 % against easy, then
   99 %. Single diagnostic games found why:
   - **Sampling at temperature 1 wrecks this action space.** An order stays in force until it is replaced, so
     a random order every few decisions interrupts everything. The untouched clone, sampled at temperature 1,
     re-ordered its commander 125 times, never finished a factory and lost in 10 minutes. The same clone won
     against easy greedily (16.0 min) and at temperature 0.3 (13.2 min, 8742 mass destroyed for 4301 lost). At
     temperature 0.3 it lost to medium while ahead: its commander attack-moved out and died.
   - **The distillation labels were wrong.** `teacher_commands(dry=True)` restores the teacher's bookkeeping
     after every call, so it re-issues "go home", every rally point and every attack order every decision:
     686 rallies in one 12-minute game where the teacher's own games send them once.
   - Entropy rose from 1.5 to about 10 once distillation had faded; PPO updates slowed from 8 s to 700 s with
     the GPU's memory full (23.8 of 24.5 GB); 21 games crashed on a defeated army's missing income; cloning
     never trained the first 8 decisions of a game (the LSTM burn-in also applied to chunks that start the game).
   Live and offline-encoded observations were checked identical on every decision of a game.
   *Fixed after night 1* (each checked on its own):
   - Training games act at temperature 0.5 and PPO scores them at the same temperature. On 128 recorded
     states, 0.5 changes the greedy choice in 11 % of decisions and issues as many commands as greedy
     (1.15 against 1.12 per decision; temperature 1: 38 % and 1.35). The untouched clone at 0.5 beat easy in
     19.8 minutes (13576 mass destroyed, 5849 lost). Acting and scoring log-probabilities agree exactly.
   - Distillation labels keep only orders that would change something: no rally for a factory the player
     already rallied, no move or attack-move for units already at, or heading to, that place (7 rallies in a
     20-minute game instead of about 1000). Verb agreement between student and teacher: 0.33, from 0.08.
   - Distillation fades to 0.1, not 0: an anchor against drift. The entropy bonus switches off while the entropy
     is above twice its value at the first update.
   - Updates run in micro-batches of 4 segments (gradients accumulated, halved on an out-of-memory error)
     under a cap of 70 % of the GPU, and the encoder runs once for both the PPO and the distillation losses:
     a full 3072-decision update of late-game observations peaks at 1.4 GB (night 1: the GPU's 24 GB, paging).
   - Learning rate 3e-5 with a 10-update warm-up: from the clone, 1e-4 took the KL from 0.08 to 1.0 within six
     minibatches (target 0.02); 1e-5 stayed under 0.015. A smaller entropy bonus (0.003 fading to 0.001):
     sampling at 0.5 explores already, and the bonus raised the entropy whenever the teacher's weight fell.
   - A shaping reward for the commander's lost health (-2 x the fraction lost per decision, fading with the
     other shaping): its death ends the game, and the clone's typical loss is a commander that walked into a
     fight. The teacher's commander lost 0.07 worth of health in an 8-minute game against medium.
   - A defeated army's empty economy no longer crashes the game; cloning trains the opening (chunks that start
     the game need no warm-up, and chunk starts are drawn so the opening is covered like the middle game);
     the night evaluates the clone before learning (`--baseline-evals`); `--collect-games 0` with `--extra-raw`
     and `--reuse-cache` clones again from existing recordings.

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Many arguments make RL slow | clone first, teacher distillation, curriculum |
| Per-unit selection gives large log-probability sums, so PPO ratios swing | entropy scaled per unit, per-command ratio monitoring, clip per command if needed |
| Observations get large (512 tokens + map) | float16 transfer, batched GPU encoding |
| The network learns to issue useless commands | per-verb success in the observation; failed commands cost nothing but are visible |
| A human-shaped vocabulary is still missing something | gate T1 before any learning |
