# faf-neural-ai

A neural-network AI for **Supreme Commander: Forged Alliance** (FAF), trained by imitation and reinforcement
learning on a rebuilt, headless game engine.

It learned first by watching [M28AI](https://github.com/maudlin27/M28AI) play (M28 runs alongside as a "shadow" and
its decisions become labels), then improved by playing tens of thousands of games against scripted AIs, M28 at
different strengths, and past versions of itself.

**Where it stands** (network `models/rl_089.pt`, 1v1, 5–10 km maps):

| Opponent | Win rate |
| --- | --- |
| FAF's scripted AIs (adaptive, rush, tech) | ~89 % |
| M28 at 0.55 cheat multiplier | ~75 % |
| M28 at 0.6 | ~50 % |
| M28 at 0.75 | ~20 % |
| Full M28 | ~1 % |

On 20 km maps it is still weak. It has seen about 30,000 games in total: ~7,400 of M28 for imitation, then
~21,000 of its own over ~100 RL iterations.

## How it works

* **Engine.** [Draiget/faf-re](https://github.com/Draiget/faf-re) is a C++ reconstruction of the Forged Alliance
  engine. `engine/patches/` adds an *agent bridge* to it: an outside program can step the simulation, observe it
  and give orders in lock-step, headless, as fast as the CPU allows. A desktop runs ~10 games at 7–9× real time
  each; a 32-core server runs ~50.
* **Bridge.** `ai/agentbridge.lua` runs inside the simulation. It builds each observation (what the army can see,
  with exact positions) and carries out orders exactly as a player could give them.
* **Agent.** `ai/python/fa_env` turns that into an environment: observations as tokens (units) and maps,
  orders as "commands with arguments" (any verb a player has: build, move, attack, reclaim, upgrade, …, with a
  target unit, a map cell and a ~160 m fine position inside it).
* **Network.** `ai/python/fa_train/tools_model.py`, ~9 M weights:
  * a transformer over the unit tokens;
  * convolutions over the map;
  * an LSTM memory;
  * a per-unit decoder that gives every unit either nothing or an order with its arguments.
* **Training.** `ai/python/fa_train`:
  * imitation of M28's decisions (`tools_bc.py`);
  * then PPO with V-trace on recorded games (`learn_rl.py`), driven by `rl_loop.py`;
  * the games are played by `player.py` lanes on one or more machines.

  The reward is win/loss plus shaping:
  * trades and economy;
  * scouting, map control, wasted resources;
  * the commander's safety.

  The loop also has:
  * a curriculum that keeps M28's strength where the network wins about half its games;
  * self-imitation of its best wins;
  * self-play against past networks;
  * a regression guard that rolls back bad networks.

`docs/` has the design notes and research logs: how the action space, observations and training evolved, and why.

## Setup

Windows 10/11 x64. Paths below assume a workspace folder `C:\faf`; any folder works.

1. **The game.** Install Supreme Commander: Forged Alliance and the [FAF client](https://www.faforever.com/), and
   start one game through FAF once. The engine runs FAF's game data (`C:\ProgramData\FAForever`, game version
   3839).
2. **The engine.** Clone the `faf-neural-ai` branch of [our faf-re fork](https://github.com/Juliansgith/faf-re)
   and build it (`engine/README.md`). That gives `main.exe`, which
   goes into `C:\ProgramData\FAForever\bin` next to FAF's game data.
3. **M28.** `git clone https://github.com/maudlin27/M28AI C:\faf\_deps\M28AI`, or set `FA_ENV_M28_DIR` to where it is.
   It is needed for M28 opponents and for M28's shadow labels.
4. **Python 3.12.**

   ```powershell
   py -3.12 -m venv C:\faf\ai-venv
   C:\faf\ai-venv\Scripts\pip install -r ai\python\requirements.txt
   ```

   Install torch with CUDA from pytorch.org if you will train. Playing works on the CPU.

## Play and watch

From `ai\python`:

```powershell
# one game of the network against FAF's "adaptive" AI on Theta Passage, in a window at ~2x speed
C:\faf\ai-venv\Scripts\python examples\tools_eval.py --checkpoint ..\..\models\rl_089.pt --ai adaptive `
    --games 1 --workers 1 --maps SCMP_012 --mode sample --temperature 0.5 --device cpu --watch --speed 3

# against M28 at 0.6 cheat, headless, 4 games side by side, recorded for training
C:\faf\ai-venv\Scripts\python examples\tools_eval.py --checkpoint ..\..\models\rl_089.pt --ai m28aicheat@0.6 `
    --games 4 --workers 4 --maps SCMP_012,SCMP_016 --mode sample --temperature 0.5 --labels m28 --record recordings

# against itself (self-play): army 2 is played by the network named in --ai, from --league-dir
C:\faf\ai-venv\Scripts\python examples\tools_eval.py --checkpoint ..\..\models\rl_089.pt --ai self@rl_089 `
    --league-dir ..\..\models --games 1 --workers 1 --maps SCMP_012 --mode sample --temperature 0.5 --device cpu
```

The opponents (`--ai`) are:
* `adaptive`, `rush`, `tech` and FAF's other AIs;
* `m28ai` (full M28);
* `m28aicheat@<multiplier>`;
* `self@<network>`.

The maps (`--maps`) are the pool in `ai/python/fa_env/maps.py`: built-in 5, 10 and 20 km 1v1 maps.

**Seeing what it decides.** Run with `--watch --debug`, then start `python tools/watch_viewer.py`. It shows the
current decision in a browser:
* each unit's top choices;
* where it wanted to go;
* the critic's estimate;
* where the attention went.

`examples/recordings/` has one recorded game: the network against itself, with M28's labels. A recording is
gzipped JSON lines with every observation, the network's choices, and what M28 would have done
(`fa_env/record.py`, `fa_train/tools_bc.py` encodes them for training).

## Train your own

The full pipeline, as it ran for `rl_089`:

1. **Recordings with labels.** Play games with `--labels m28 --record <dir>`: M28 shadows our army and its
   decisions are stored next to ours.
2. **Imitation.** `python -m fa_train.tools_bc --raw <recordings> --cache <cache> --out start.pt --decoder units
   --sizes M --steps 4000`.
3. **RL.** `fa_train.rl_loop` pulls finished rounds from the players and trains each iteration with
   `fa_train.learn_rl` on the GPU. It then pushes the new network back.
   * **Players:** `fa_train.player` runs lanes of games with the current network, on this PC and/or a Linux
     server reached over SSH.
   * **Launch scripts:** `examples/configs/` has the ones that were used. They record every setting change,
     with the reason, as comments.
   * **Player configuration:** `examples/configs/player_config_example.json`.
4. **Judge networks** on matched conditions with `tools/rl/compare_nets.py`, and plot progress with
   `tools/rl/rl_graph.py`.

One iteration is ~240 games and ~40 minutes on an RTX 4090. The plateau notes and what was tried are in the
launch script's comments and in `docs/`.

## Credits

* **[faf-re](https://github.com/Draiget/faf-re)** by Draiget. The reconstructed engine this all runs on.
* **[M28AI](https://github.com/maudlin27/M28AI)** by maudlin27. The teacher: the network learned its first moves
  from M28's play. M28 is not included here; it is downloaded separately under its own licence (CC BY-NC-SA 4.0).
* **[FAForever](https://github.com/FAForever/fa)**. The game code, data and AIs the games are played with.
* Supreme Commander: Forged Alliance is © Gas Powered Games / Square Enix. No game assets are included.

## License

This repository's own code, documentation and the trained network are MIT licensed (`LICENSE`). The engine
patches in `engine/patches/` change [faf-re](https://github.com/Draiget/faf-re) and are offered on the same terms
for our changes only; faf-re itself, FAF's game code and data, and M28AI keep their own terms and are not part of
this repository.
