# Engine

The game engine is [Draiget/faf-re](https://github.com/Draiget/faf-re), a C++ reconstruction of Forged
Alliance's Moho engine. This repository doesn't include it. `patches/` is the exact series our engine is built
from: 56 commits on top of upstream `BASE_COMMIT`, in order.

* **Engine fixes.** Commits found by running FAF's own Lua on the rebuilt engine (missing engine functions,
  Lua-binding issues, determinism), plus simulation performance work. Nothing AI-specific.
* **The agent bridge** (`AGENT:` / `Agent:` commits): `moho/agent/AgentBridge.*` plus a few hooks. Its
  switches, all inert without them:
  * `/agentbridge <port>`: an outside program steps, observes and commands the simulation over TCP.
  * `/agentscript`: the sim-side Lua (`ai/agentbridge.lua`).
  * `/agentspeed`: simulation speed.
  * `/agentbackground` and `/agentheadless`: no window focus, no drawing.
  * `/agentlean`: skip render-only work.

## Build

Needs Visual Studio 2022 (Build Tools, v143 toolset), and faf-re's dependencies (Boost 1.34.1, LuaPlus 1081,
wxWindows 2.4.2, Wild Magic, zlib, DirectX SDK June 2010) as faf-re's README describes. For Wild Magic, see
`deps/WildMagic3p8-standin/README.md`.

```powershell
git clone -b faf-neural-ai https://github.com/Juliansgith/faf-re C:\faf\faf-re-agent
# set up the dependencies (faf-re README), then:
<this repo>\engine\scripts\build.ps1 -FafRe C:\faf\faf-re-agent        # Release|Win32
```

[Juliansgith/faf-re](https://github.com/Juliansgith/faf-re) is a fork of Draiget/faf-re. Its `faf-neural-ai`
branch is exactly the engine we build (commit `454a4781`). Two other copies of the same commits are here:
* `faf-re-agent.bundle`, for offline use: clone Draiget/faf-re, then `git fetch <this repo>\engine\faf-re-agent.bundle port-20261007:agent`.
* `patches/`, for reading and review. Don't build from them: faf-re's sources have Windows line endings, which
  `git am` mangles.

`build.ps1` copies `main.exe` into `C:\ProgramData\FAForever\bin` (`-DeployTo` for elsewhere; the Python side
finds it there, or through `FA_ENV_BIN_DIR`). The project links with `/FORCE`, so MSBuild can "succeed" with
unresolved symbols. `build.ps1` checks the build log and refuses to deploy a binary with errors.

The patches are against a pinned upstream commit. faf-re moves on, so newer upstream versions may need the
series rebased.
