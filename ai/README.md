# Agent bridge: training AIs against the rebuilt engine

Everything needed to drive Forged Alliance (FAF, game version 3839 data) from an external program:
reinforcement-learning trainers, neural-network policies, scripted bots. Three parts:

| Part | Where | What it does |
| --- | --- | --- |
| Engine side | `engine/patches/agent` (faf-re `src/sdk/moho/agent/AgentBridge.{h,cpp}` + hooks) | Inert unless the game is launched with `/agentbridge`. At the end of each due sim tick it sends one JSON message to the agent and **blocks the sim thread** until the agent answers, so the agent and the sim advance in lock-step. |
| Sim-side script | `ai/agentbridge.lua` | Ordinary FA sim Lua. Builds observations and applies actions through the normal sim API (`IssueMove`, `IssueBuildMobile`, ...). Change it to change what the agent sees and can do; no engine rebuild needed. |
| Python package | `ai/python/fa_env` | Launches games, speaks the protocol, and provides a gym-style `FAEnv`, a numpy featurizer and two reference agents. |

## Quick start

```bash
# 1. build the engine with the patches and deploy main.exe next to the FAF game data (see engine/README.md)
powershell -File engine/scripts/build.ps1 -FafRe ../faf-re-dev
# 2. play one episode with the scripted reference bot against FAF's rush AI (window opens, ~1 minute)
python ai/python/examples/play_episode.py --agent scripted --minutes 12
```

```python
import sys; sys.path.insert(0, "ai/python")
from fa_env import FAEnv, GameConfig

env = FAEnv(GameConfig(map="SCMP_016", faction=1, ai_faction=2, seed=1), step_ticks=10)
obs, info = env.reset()                      # info: map, heightmap, deposits, armies, 743 blueprints
acu = env.my_units()[0]
obs, reward, terminated, truncated, step = env.step([
    {"type": "move", "units": [acu["id"]], "pos": [120.0, 60.0]},
])
print(step["results"])                       # [{'ok': True}] -- per-action success / error text
env.close()
```

Army 1 is the agent's (a human slot nobody else commands); every other start position gets an FAF
AI. One game process runs per episode, and `reset()` launches a fresh one (about 15 s, mostly map
loading).

## Command line (engine)

| Switch | Meaning |
| --- | --- |
| `/agentbridge <port>` | enable; the game connects to `127.0.0.1:<port>` (the agent listens) |
| `/agentscript <file>` | the sim-side Lua (`ai/agentbridge.lua`) |
| `/agentspeed <rate>` | pacing while the bridge is on: `50` (default) is unpaced, `10` is the fastest normal game speed, `0` is real time |
| `/agenttimeout <s>` | how long to retry the initial connection (default 60) |
| `/agentvfx` | keep particle effects while unpaced (they are skipped by default then; see Performance) |
| `/agentwatchdog <s>` | if the sim thread makes no progress for this long, log the Lua traceback, send it to the agent (`fa_env` raises `GameError`) and exit; default 60, 0 disables |

`fa_env` passes all of these itself. It also generates a per-run init file that mounts `ai/fa_overrides`, whose `/schook` hook lets
`/agentaifaction <1-4>` and `/agentai <personality>` pin the opponents (`GameConfig.ai_faction`, `GameConfig.ai`).
Without them FAF's command-line launch picks the AI factions at random.

## Protocol

Newline-delimited JSON over TCP. The game sends, the agent replies, once per step:

```text
game  -> {"type":"hello","protocol":1,"tick":1,"info":{...AgentBridge_Hello()...}}
agent -> {"config":{"armies":[1],"omniscient":false},"wait":1}
game  -> {"type":"obs","tick":12,"gameOver":false,"obs":{...},"result":{...},"timing":{...},"errors":[...]}
agent -> {"actions":[...],"queries":[...],"wait":10}
...
agent -> {"cmd":"quit"}
```

* `wait` is the number of ticks until the next observation (10 ticks = 1 s of game time).
* `result` holds the outcome of the *previous* reply's actions and queries, in order.
* Script errors never reach the sim; they come back in `errors`.
* If the agent disconnects, the game process exits, so a crashed trainer leaves no orphans.
* The engine only reads `wait` and `cmd`. Everything else in the reply is handed to `AgentBridge_Act`.

## Observations (`AgentBridge_Observe`)

```text
tick, time, me
armies[]  index, relation (self/ally/enemy/neutral), civilian, defeated, status,
          eco {mass, energy: stored, max, income, requested, usage, trend  (per second)},
          stats {Enemies_Killed, Units_MassValue_Lost, ...}, unitCap, unitCount   (own/allied only)
units[]   own:     id, army, bp, x, y, z, hp, maxhp, frac, layer, heading, idle, work, queue, focus, paused, shield
          visible: id, army, bp, x, y, z, hp, maxhp, frac, layer, heading
          radar:   id, army, x, y, z, radar=true   (blip position, possibly jammed)
```

Enemy units are filtered through the agent army's own intel: seen now means full detail, radar/sonar/omni means a blip, otherwise
nothing. Set `omniscient=True` to see everything (for debugging or privileged-information baselines).

`hello.info` has static data:

* map size, playable area, `heightmap` (terrain and water surface on an n×n grid) and mass/hydro `deposits`
* armies with start positions
* every unit `blueprint`: categories, costs, build time and rate, production, health, speed, weapons, and a resolved `buildable` list

## Actions and queries (`AgentBridge_Act`)

Actions (all take `units: [ids]`, which must belong to the acting army):

* `move`, `attack_move`, `patrol` take `pos`.
* `attack` and `guard` take `target` (an id) or `pos`.
* `repair`, `reclaim`, `capture` and `overcharge` take `target`.
* `stop`, `clear`, `dive` and `self_destruct` take nothing else.
* `build` takes `bp` and `pos` (mobile builders).
* `factory_build` takes `bp` and `count`.
* `upgrade` takes `bp`.
* `rally` and `transport_unload` take `pos`.
* `transport_load` takes `target`.
* `nuke` takes `pos`.
* `tactical` takes `pos` or `target`.
* `silo_build` takes `nuke` (bool).
* `enhance` takes `name`.
* `pause` takes `on`; `fire_state` takes `state`; `toggle` takes `bit` and `on`.

Queries (answered in the next message):

* `can_build` takes `bp` and `pos`.
* `find_build_site` takes `bp`, `pos` and `radius`, and returns `[x, z]` or null.
* `deposits` takes `pos`, `radius` and `kind`.
* `reclaim` takes `pos` and `radius`.
* `eval` takes `code`, and is off unless `config.allowEval`. It runs arbitrary sim Lua, for development and curriculum setup only.

Targets must be visible to the acting army unless `omniscient`. Nothing in the action set grants resources or spawns units.

## Python package

* `fa_env.FAEnv`: `reset() -> (obs, info)`; `step(actions, queries, wait) -> (obs, reward, terminated, truncated, info)`; `query(...)`; `close()`.
  * `reward_fn` defaults to the change in (enemy mass destroyed − own mass lost)/1000, with ±10 on a decided game. Replace it for real training.
* `fa_env.featurize.Featurizer`: turns observations into model inputs.
  * `[13, 64, 64]` map planes
  * scalar features
  * a padded per-unit table with blueprint indices for an embedding (vocabulary of all 743 unit blueprints)
* `fa_env.agents.ScriptedAgent`: a build order followed by tank waves. It is a baseline and an end-to-end test of builds, factory queues, assist and attack-move.
* `fa_env.agents.RandomAgent`: a fuzzer.
* `fa_env.gym_env.MacroEnv`: a `gymnasium.Env` with Featurizer observations and 9 macro actions.
  * The macros are `noop`, `build_mex`, `build_power`, `build_factory`, `train_engineer`, `train_tank`, `attack`, `retreat` and `assist`.
  * The policy picks one every `decision_ticks` (default 20 = 2 s); scripted micro executes it.
  * `obs["action_mask"]` marks the macros that can act right now.
  * It is the usual first step for RTS reinforcement learning. Grow the macro set, or move to unit-level actions, as the policy improves.
* `examples/`
  * `smoke_test.py`: handshake and step timing
  * `play_episode.py`: one episode with progress and throughput
  * `determinism_check.py`: runs the same episode twice and compares every observation
  * `vec_throughput.py`: N games in parallel (`gymnasium.vector.AsyncVectorEnv`), aggregate decisions/s
  * `train_ppo_minimal.py`: one-file PPO on the GPU. It uses a map CNN, a set encoder over units (blueprint embeddings) and masked macro actions. It proves the loop; it is not a tuned trainer.

Python environment: `C:\faf\ai-venv` has torch 2.11 (CUDA 12.8, RTX 4090), numpy and gymnasium.

```bash
C:\faf\ai-venv\Scripts\python.exe ai/python/examples/train_ppo_minimal.py --envs 4 --steps 64 --updates 10
```

## Training (`fa_train`, design in [docs/training-design.md](../docs/training-design.md))

* `fa_env.factored.FactoredEnv`: the macro action space. It has five masked heads (build, produce, army, commander, tech) executed by scripted skills, a unit-set observation, privileged critic inputs, and separate reward components.
* `fa_env.rule_policy.RulePolicy`: a hand-written policy in that action space (the teacher). `examples/rule_game.py` plays one game with it; `--minimized`, `--probe-minute` and `--profile-minute` measure CPU per thread.
* `fa_train.trainer.Trainer`: the learner.
  * Async worker processes, one minimized game each, with batched GPU inference.
  * PPO with GAE and a decaying KL pull to the cloned policy.
  * The curriculum (`STAGES`), periodic greedy evaluations, checkpoints, and logs (TensorBoard, `episodes.jsonl`, `updates.jsonl`, `events.jsonl`).
  * Crashed or hung games are relaunched.
* `fa_train.bc`: behaviour cloning of the rule policy (class-weighted, with masked label smoothing).
* `fa_train.night`: the unattended pipeline. It runs gate → collect → clone → PPO → final evaluation, then writes `report.md`.

A night run (full speed, 20 games; the PC is busy until the stop time):

```bash
ai\night.cmd 07:00 20
```

* It writes to `_runs\night-YYYYMMDD\`: `report.md`, `night.log`, `tb\` (`tensorboard --logdir ...`), `checkpoints\`, and the JSONL logs.
* To stop early, create a file named `STOP` in that folder. The run saves a checkpoint and closes its games within seconds.
* It asks Windows not to sleep while it runs. Locking or signing out is untested; turning the monitor off is fine.
* To continue PPO on a later night: `python -m fa_train.night --run <dir> --stop-at 07:00 --resume <dir>\checkpoints\latest.pt`.

Tests:

* `--fake` runs the whole pipeline without games in a few minutes.
* A short real run uses one game: `--workers 1 --scale 0.2 --minutes 6 --eval-minutes 8 --batch 128 --segment 64` plus small game counts.

## Measured (Ryzen 9 5900X, SCMP_016 Canis River 1v1 vs FAF rush AI, unpaced)

* **Determinism:** two runs with the same seed, opponents and actions produce byte-identical observations. This was checked for 10 game minutes with combat and ACU upgrades (6,000 ticks, omniscient observations). So episodes can be reproduced from their action logs.
* **Parallel games:** 4 games through `AsyncVectorEnv` gave 74–106 macro decisions/s (2 s of game each) during PPO. That includes GPU inference and updates. `vec_throughput.py` reports the raw numbers.
* **Speed:** the early game runs at about 1 ms per tick (~100× real time). Around 12 minutes, with ~200 units on the map, it is about 7–10 ms per tick (10–15×). A 12-minute episode takes ~36 s.
* **Per step:** building and encoding an observation costs 1–3 ms; applying actions is negligible.
* **Episode start:** about 15 s to load the map, then the first tick.

Engine changes that got it there (all inert without `/agentbridge` except the probe fix):

* The Lua GC period is stretched 10× in unpaced sessions. A forced full collection every 70 ticks was over a third of the sim's time.
* Particle emission is skipped in unpaced sessions. It is visual only, and it uses a random stream separate from the sim's.
* The sim can now run further ahead of the render loop.
* A `VirtualQuery` diagnostic probe in path-cluster updates now runs once per pointer instead of on every call. It had made a commander's warp-in cost ~15 s at training speed.
* The sound manager's per-beat pass is skipped under `/nosound` (agent patch 0003).
  * Upstream's temporary `[SNDDIAG]` probes in it took 70–80 % of the main thread, and the sim thread waited on that.
  * Result: twice the game speed. The sim thread went from 62 % to 98 % of a core.
  * A minimized window then removes all drawing: 1.3 instead of 1.7 cores per game.
  * Tools: `tools/thread_cpu.py` (per-thread CPU) and `tools/thread_profile.py` (attach-and-sample profiler).

Where late-game time goes (sampling profile at ~11 minutes, ~200 units; `_tests/sample_profile.py`):

* About 40% is land-unit path navigation. `CAiPathNavigator::TryAdvanceTargetPoint` → `SweptPathBlockedByUnit` → `COGrid::CollectEntitiesInBox` runs an oriented-box SAT test (`ComputeBoxBoxContactManifold`, 23% self) roughly 150 times per unit per tick.
* The rest is occupancy checks, steering, recon and Lua.
* The code matches the original's algorithm, so this is the place to optimise if late-game throughput matters. Cache swept-path results per path cell, or cull candidates harder before the SAT.

## Known limitations

* **Replays do not contain agent actions.** Agent actions are applied through sim Lua, not the command stream. Record the agent's messages instead; with determinism they reproduce the game.
* Opponents are FAF's Lua AIs as they run on this rebuilt engine. The engine does not yet implement every FAF binary patch.
  * Ported so far: `IssueBuildMobile`'s 5th argument, deposit queries, the navigator distance, and projectile zig-zag.
  * The rest of FA-Binary-Patches is unverified.
* Engine bugs found by driving the game through the bridge, all fixed in this tree:
  * `SecondsPerTick` and `LUnitMove` were never registered, so every ACU/SCU enhancement failed.
  * Script-task errors turned into an endless `TaskTick` loop that froze the sim. The protected `lua_call` had swallowed the error, returning 0 = `Repeat`.
  * `/init` was ignored.
  * `IssueBuildMobile` lacked FAF's 5th argument, which broke AI engineers.
  * `CollideSurface=false` projectiles hit the ground.
  * `FormPlatoon` read its multiplier from the wrong argument.
* Engine-to-Lua calls through `LuaFunction` use the protected `lua_call`. Most script errors in engine callbacks are therefore swallowed silently, where the original lets them reach an outer handler that logs them. `TaskTick` is fixed (`LuaFunction::CallRaising`); other call sites may still hide errors.
* One game per process. There is no headless switch. `GameConfig(minimized=True)` minimizes the window once the game runs, and the engine then draws nothing (training does this).
* User-side (UI) commands from `/map` launches are rejected with an invalid command id. This doesn't affect agents, which command through the sim.
* Self-play (several agent armies in one game) is supported by the script (`config.armies`, per-action `army`) but untested.
