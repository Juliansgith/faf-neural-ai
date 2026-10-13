# Engine

The game engine is [Draiget/faf-re](https://github.com/Draiget/faf-re), a C++ reconstruction of Forged
Alliance's Moho engine. This repository doesn't include it. Our engine is upstream `BASE_COMMIT` (faf-re master,
2026-10-13) plus 52 commits:

* **Engine fixes.** Commits found by running FAF's own Lua on the rebuilt engine (missing engine functions,
  Lua-binding issues, determinism), plus simulation performance work that keeps results bit-identical. Nothing
  AI-specific.
* **The agent bridge** (`AGENT:` / `Agent:` commits): `moho/agent/AgentBridge.*` plus a few hooks. Its
  switches, all inert without them:
  * `/agentbridge <port>`: an outside program steps, observes and commands the simulation over TCP.
  * `/agentscript`: the sim-side Lua (`ai/agentbridge.lua`).
  * `/agentspeed`: simulation speed.
  * `/agentbackground` and `/agentheadless`: no window focus, no drawing.
  * `/agentlean`: skip render-only work.
* **A build option** to leave FFmpeg out (below).

## Build

Needs Visual Studio 2022 (Build Tools, v143 toolset), and faf-re's dependencies (Boost 1.34.1, LuaPlus 1081,
wxWindows 2.4.2, Wild Magic 3.8, zlib, DirectX SDK June 2010), set up as faf-re's README describes.

* **Wild Magic:** apply faf-re's own `dependencies/patches/wildmagic3p8_faf_required.patch`. Since upstream
  081cd02d it adds `Wm3IVector2.h` and `Wm3IVector3.h` and changes the Vector, Quaternion and AxisAlignedBox
  headers. `deps/WildMagic3p8-standin/README.md` describes the stand-in tree we built with; it needs those
  header additions too.
* **FFmpeg:** upstream's movie player needs a 32-bit FFmpeg 7.1 package. Without one, build with
  `/p:FafFfmpeg=false` (or `$env:FafFfmpeg='false'` before `build.ps1`). The engine then can't play intro
  movies, which the agent never uses (it launches with `/nomovie`).

```powershell
git clone -b faf-neural-ai https://github.com/Juliansgith/faf-re C:\faf\faf-re-agent
# set up the dependencies (faf-re README), then:
$env:FafFfmpeg = 'false'                                              # unless you have FFmpeg (above)
<this repo>\engine\scripts\build.ps1 -FafRe C:\faf\faf-re-agent        # Release|Win32
```

[Juliansgith/faf-re](https://github.com/Juliansgith/faf-re) is a fork of Draiget/faf-re. Its `faf-neural-ai`
branch is exactly the engine we build (commit `bb9647cf`). Two other copies of the same commits are here:
* `faf-re-agent.bundle`, for offline use: clone Draiget/faf-re, then `git fetch <this repo>\engine\faf-re-agent.bundle port-latest:agent`.
* `patches/`, for reading and review. Don't build from them: faf-re's sources have Windows line endings, which
  `git am` mangles.

`build.ps1` copies `main.exe` into `C:\ProgramData\FAForever\bin` (`-DeployTo` for elsewhere; the Python side
finds it there, or through `FA_ENV_BIN_DIR`). The project links with `/FORCE`, so MSBuild can "succeed" with
unresolved symbols. `build.ps1` checks the build log and refuses to deploy a binary with errors.

When faf-re moves on, the series gets rebased onto it. The previous base was `8977bb9d`. On 99 paired games the
network's results on the two engines matched within noise (82 % vs 86 %, p = 0.54), and the new one is ~8 % faster.
