# From a trained policy to an AI people can play against

A policy trained with this project is **not plug-and-play in the FAF client**, and the reason shapes how
the network should be designed.

## Why not

* **Training runs outside the game.** The network lives in Python/PyTorch and talks to *our rebuilt
  engine* through the agent bridge. The FAF client launches FAF's official `ForgedAlliance.exe`. That
  exe has no bridge, and no way to call out to a Python process.
* **Multiplayer is lock-step.** Every player's machine runs the whole simulation and must compute
  identical results; only player commands travel over the network. A model running on one machine,
  outside the sim, cannot feed decisions into it except as that player's commands. And an AI slot has
  no machine of its own.

## Ways to ship it

1. **An FAF AI mod (recommended).** This is how M28AI, RNGAI and the other lobby AIs work: the AI is
   sim-side Lua, run identically by every client, and published in the FAF mod vault. Our observation
   and action code (`ai/agentbridge.lua`) is already sim Lua and carries over almost unchanged. What has
   to move into Lua is **inference**: the weights are exported to a `.lua` file and run by a small
   Lua runtime (matrix-vector products, activations, softmax). It is deterministic across clients
   because the Lua, the weights and the float rules are the same everywhere. No binary patch or special
   client is needed; players pick it in the lobby like any other AI.

   The cost is speed. FA's Lua is an interpreter with single-precision numbers, sharing a sim thread
   that is already busy in big games. A few hundred thousand multiply-adds per decision, one decision
   every one or two seconds, spread over ticks, is comfortable. A 64×64 convolution stack over the whole
   map is not. So either design the network Lua-sized from the start, or train a large model and
   **distill** it into a small student that runs in Lua.

2. **An external process through an FAF engine patch.** FA-Binary-Patches would have to add
   networking to the official exe, with the decisions routed through one machine's command stream like
   a human player's. It is a big change for FAF to accept, and it only fits some game setups.

3. **Our rebuilt engine as the game client.** This is only possible once the rebuild computes exactly
   what the official exe computes, so it can stay in sync with other players. That is a long way off.

## Consequences for training now

* **Keep the observation computable in Lua.** Everything `Featurizer` uses comes from
  `agentbridge.lua` today; keep it that way, or port new features to Lua as they are added.
* **Budget inference early.**
  * Prototype the Lua runtime and measure a candidate network inside a real game before investing
    in a large architecture.
  * Hierarchical designs help: a small policy picks macros, and scripted micro executes them. The
    macro layer is where the network earns its keep, and it runs rarely.
* **Only use information a player has.** The bridge's default observation is fog-of-war filtered; an
  AI trained on omniscient observations will not transfer to a fair mod.
* **Mind the gap between the two engines.** We train on the rebuild but would ship on FAF's
  official engine. Behaviour differences between the two (bugs not yet found) can hurt a learned
  policy. Validate in the official client before release; the upstream fix series narrows the gap.
* **Where AIs play.** AI mods are for custom games, co-op and offline play; ranked ladder games are
  human-only.
