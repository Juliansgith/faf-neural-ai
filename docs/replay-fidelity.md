# Replay fidelity check

How closely does our engine (faf-re + the patches in `engine/`) reproduce FAF's own `ForgedAlliance.exe`
(game version 3839)? The check plays a replay recorded by the official engine in ours and looks for the
first place the two simulations disagree.

## Why bit-exactness is a tool, not a goal

Training does not need our engine to match FAF bit for bit: a neural network never sees the last bit of
a falling tree's rotation, and every training game is a fresh sample anyway. What training needs is the
same *rules* (unit stats, damage, economy, pathing, how the stock AIs behave).

Bit-exactness is still the sharpest instrument for finding rule differences: while both engines stay
identical, the first divergence points straight at the responsible code. Once they drift, the two games
diverge chaotically and the replay says nothing more. So the check was run until it stopped finding
anything but rounding differences, then stopped.

## Method

1. `tools/replay_fidelity.py record` plays an all-AI skirmish (SCMP_016, seed 7) in the official exe with
   `/synclog` (every beat's checksum inputs traced to a file) and keeps the replay.
2. `replay_fidelity.py playback` plays that replay in our `main.exe` with `/synclog`. The replay carries the
   official run's checksum every 50 beats, so our engine reports mismatches itself; `compare` finds the
   first differing trace line.
3. Between checksums, `tools/beat_probe.py` (a minimal debugger on the Win32 debug API, no install needed)
   breakpoints `Sim::AdvanceBeat` in either engine and fingerprints the random stream every beat: the
   Mersenne-Twister index, the cached Gaussian and an MD5 of the state. Every random draw depends on
   everything before it, so the first beat whose fingerprint differs is where the two engines started
   doing different work. `--watch A:B` adds a hardware write-watchpoint on the twister index and records
   the code address of every draw (the binary inlines the twister in places); `--bp` records the call
   chain and arguments at any address or symbol.
4. `tools/fidelity/make_lua_trace.py` builds a Lua override that logs every sim `Random()` call with its
   caller (and optionally `DamageRing` and `brain:GetUnitsAroundPoint`) in both engines, before any FAF
   module captures a local copy. `rnd_diff.py` aligns the two logs. The tracer only reads state; traced
   and untraced runs fingerprint identically.

```powershell
python tools\fidelity\probe_run.py official --beats 1500 --fast --tag _long
python tools\fidelity\probe_run.py ours --beats 1500 --fast
python tools\fidelity\probe_diff.py <work>\probe\official_long.jsonl <work>\probe\ours.jsonl
python tools\fidelity\probe_run.py ours --beats 95 --watch 88:92 --tag _w    # then draw_diff.py
python tools\fidelity\make_lua_trace.py; python tools\fidelity\probe_run.py ours --mount <work>\lua_trace
```

## Findings

The replay first would not load (engine version) and then stalled at beat 0; after that, each divergence
led to a bug. All fixes are in `engine/patches/upstream` (0019-0028), each checked against the binary's
disassembly.

| Found | Bug | Gameplay effect |
| --- | --- | --- |
| replay rejected | engine version hard-coded to 3764 (FAF 3839 writes its own) | replays and `GetVersion()` |
| stall at beat 0 | `CReplayClient` acked the local client's beats under its own index | no replay played at all |
| crash at beat 31 | sim driver seeded the focus army from the command-source id (255 in replays) | replay crash; wrong focus army in multiplayer |
| beat 50 checksum | projectile launch spreads skipped draws for zero ranges, rounded words early, normalized the direction; Lua `Random()` used a constant that is not 2^-32 | determinism only |
| RNG at beat 32 | every map prop (.scmap) and `CreatePropHPR` prop got the conjugate orientation | props' collision boxes turned the wrong way |
| beat 100 checksum | `EulerToQuaternion` had its vector lanes rotated | structures on slopes tilted about the wrong axis |
| RNG at beat 90 | `brain:GetUnitsAroundPoint` returned units outside the radius (no distance filter) | FAF's AIs (our training opponents) saw units that were not near |
| beat 400 checksum | falling-tree motor: rounding of fused x87 products, a different shortest-arc helper, angle wrap; tilt sums in a different order | determinism only |

## Status

- Bit-identical through beat 399 (40 s of game time); the random stream stays identical through beat 719
  (72 s). At the start of the check: 5 s.
- The remaining beat-400 difference is the last bit or two of falling trees' rotations (MotorFallDown).
  No gameplay effect; not pursued further.

Noticed along the way, not fixed:

- Lua's `tostring` of small numbers: the MSVC8 CRT prints three exponent digits (`8.3e-005`), ours two
  (`8.3e-05`). Lua-visible; matters only to code that uses such strings as keys.
- Lua source paths differ (`bin\..\gamedata\\lua.nx2\...` vs the official, normalized path), visible to
  `debug.getinfo`.
- About twenty other sites use the `ToUnitFloat` pattern for random spreads; like the projectile spreads
  they may differ from the binary in the last bit. `MultiplyQuat`'s association order may not match every
  binary call site.
- Trace-only: `Materialize` synclog lines print half of a double instead of the float's bits; the
  navigator's `nav tick` / `DriveToNextWaypoint` trace lines are missing.

## Next

Check behaviour rather than bits: scripted mechanics (unit-versus-unit fights, build times, economy rates,
move speeds, pathing) and aggregate statistics over many AI-versus-AI games in both engines, compared
within tolerances.
