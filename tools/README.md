# Tools

Development helpers used to find and verify the engine fixes. They run with any Python 3.10+ that has
`pefile` and `capstone`; the profiler also needs `ai/python`.

| Tool | Use |
| --- | --- |
| `fa_harness.py` | Run one scripted `/map` skirmish (no agent) with a private prefs file; summarises Lua errors and game flow from the log. `fa_harness.py <label> main.exe <outdir> <seconds> -- /map SCMP_039 /seed 1 /faction 1 /exitongameover` |
| `sample_profile.py` | Sampling profiler of the sim thread: self and approximate inclusive time, symbolised with `main.pdb` (and DLL exports). Starts its own bridged game with the scripted agent, or `--pid` attaches to a running (e.g. hung) one. |
| `symbolize.py`, `find_base.py` | Symbolise crash-log addresses; recover the ASLR image base from a crash log. |
| `x86dis.py`, `findcalls.py` | Disassemble a range of the original FAF exe; find every `call` to an address. This is how the rebuild's behaviour is checked against the binary. |
| `scan_unregistered_luafuncs.py` | List Lua bindings the engine defines but never registers (how `SecondsPerTick` was found). |
| `rdconst.py` | Print float/double constants of the original exe by address (`f:e4f714 d:e4f780`). |
| `replay_fidelity.py` | Record an all-AI replay with FAF's exe, play it in ours, compare the per-beat sync traces. See `docs/replay-fidelity.md`. |
| `beat_probe.py` | Minimal Win32 debugger: per-beat random-stream fingerprints in either engine, hardware watch of every twister draw (`--watch`), call chains and arguments at any address or PDB symbol (`--bp`). |
| `binary_index.py` | Names an address of FAF's exe after the recovered function that contains it (from the `Address:` comments in faf-re's sources). |
| `teacher_determinism.py` | Plays the same teacher-driven game twice (separate processes, different `PYTHONHASHSEED`) and compares the sim's per-beat checksums (`--extra /agentchecksum 1`), every observation (`--obs-log` shows what differs first), the teacher's commands and any command-stream input the agent did not send. `--extra-a /agentpoison 0 --extra-b /agentpoison 255` fills heap blocks differently in the two runs to expose reads of uninitialised memory; `/agentpoisonsizes`, `/agentpoisonbytes` narrow it down. From Git Bash set `MSYS_NO_PATHCONV=1`, or the `/agent...` arguments are turned into paths. |
| `obs_equivalence_probe.py` | The native observation unit list against the bridge script's Lua one, on spawned units a teacher rarely has (transports, silos, toggles, enhanced commanders, shields up / damaged / down). |
| `fidelity/` | Replay-comparison drivers: `probe_run.py` (a replay under `beat_probe`, `--fast`), `probe_diff.py`, `draw_diff.py`, `make_lua_trace.py` + `rnd_diff.py` (trace and align every Lua `Random()` call). |
