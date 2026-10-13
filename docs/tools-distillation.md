# Tools online distillation

`ToolsConfig.distill_teacher` selects the online PPO label source: `v3` (the default rule teacher) or `none`.
The API schedule defaults remain `distill_start=1`, `distill_fraction=0.3`, `distill_end=0.1`.
The night CLI defaults remain start `1`, fraction `1/3`, end `0`, and now expose
`--distill-teacher {v3,none}`, `--distill-start`, and `--distill-fraction` alongside `--distill-end`.
Source and finite, nonnegative weights/fraction are checked before run-folder changes or learner allocation.
Fraction `0` and fractions above `1` remain supported; zero follows the existing one-update denominator floor.

`--distill-end=0` alone still gives an early teacher tutorial. Use `--distill-teacher none` to disable online
distillation entirely: the effective weight is zero even if configured weights are positive, and workers make
no teacher requests, including when a stale episode contains `teacher_labels=True`.
Episodes carry the selector; legacy episodes with labels but no selector use v3. Unknown selectors fail before
an environment or game is constructed. Worker reset metadata reports the selector and whether labels are active.

Configure supported selection through `ToolsConfig`; `ToolsTrainer.set_teacher(path)` raises an unsupported
checkpoint-teacher error without reading the path. `load()`, `--clone`, and `--resume` initialize model/training
state and do not restore the runtime configuration or select a checkpoint teacher. Runtime distillation options
remain authoritative, so a default v3 run resumed from a saved none run selects v3 again; the active schedule
at the resumed update determines the actual weight and whether labels are requested. Use explicit
CLI source/schedule options to continue or intentionally override the saved behavior.

PPO checkpoint `config` contains the selector and schedule; `distillation` stores the effective start/end and
current beta. Load prints saved versus active distillation settings, prominently warns on differences, and logs
the full saved/active configs plus selector provenance, mismatches, and diagnostics in a
`checkpoint_distillation` event before workers start. A saved config missing the selector is diagnosed as legacy
v3. An unknown saved selector is diagnosed and does not override the validated active config. A BC checkpoint
without `config` has unavailable PPO teacher provenance; it is not attributed to v3. Existing model/optimizer,
return-scale and legacy checkpoint initialization remain supported.

## Offline critic reward limitation

Legacy recordings do not contain enough information to reconstruct exact final-transition or commander-health
rewards. Their fallback targets approximate trade/economy at shaping 1 and a flat victory +10. Current tools
recordings can include `fa_tools_transition_v1`, with actual reward components and transition ticks.
`recorded_returns.py` validates alignment and computes discounted targets from those components at a declared
shaping weight, including commander damage and the victory speed bonus. Partial or inconsistent transition
records are rejected. This does not retroactively repair old data or reproduce a learner's changing shaping
schedule. Observation/action representability drops also need not imply teacher mistakes.

No policy-strength or promotion claim follows from these configuration and mock-flow checks.

## Decoder compatibility on checkpoint load

A compatible tensor shape does not establish compatible action probabilities. `ToolsTrainer.load`
checks the saved `set_distribution` against both actor and learner before loading weights or optimizer state,
including with `optimizer=False`. Legacy independent-Bernoulli checkpoints have no distribution metadata.
Alternative decoder classes must explicitly declare their complete `set_distribution_contract`; matching
only the acting name while changing the objective or pinned implementation hashes is rejected. Save retains
the declared contract, and actor/learner disagreement is rejected before writing a checkpoint.

Preserved weekend conditional checkpoints can be evaluated through the private
`_runs/weekend_20261003/full_set_stage_contract_v2.py` adapter in the workspace parent. It computes the
expected contract from the unchanged historical implementation files; it does not trust a checkpoint
to declare the behavior of arbitrary runtime code. The old private files remain preserved. This adapter
is evaluation-only and does not make conditional PPO, its cadence or its untrained critic ready.
Do not strip metadata to bypass a compatibility error. Legacy files that never recorded a contract
remain limited to their original independent decoder; missing provenance cannot establish equivalence.

## Cache precision and migration

Cache format 5 uses the same observation precision as the live worker transport:
`tokens` and `map` are stored as float16 and widened to float32 for the model;
`cand_feat` and `enh_feat` remain float32. The shared contract lives in
`fa_env/observation_precision.py`. Previously, candidate and enhancement features
were rounded to float16 only in offline caches, creating different training inputs.

Format 5 records its precision contract in the manifest and includes the shared
precision implementation in the encoder semantics hash. `Games` rejects older
or mismatched caches. Re-encode the original recordings into a fresh cache directory
before new training; do not edit manifests to make old caches appear current.
The `.v3.npz` filename suffix still identifies the observation schema, while the
manifest format identifies cache compatibility. Historical files are preserved.

The weekend validation re-encoded one 2,124-frame development recording and checked
48,852 observation arrays against actual worker compression and trainer widening:
zero differences. All labels, masks, returns, and other stored arrays were unchanged;
1,114 commands were retained with zero drops. The compressed cache grew by 36,574
bytes (about 0.33%). This verifies input consistency, not improved gameplay.
The final S/M reference assessment used the earlier, archived runtime and was not
rerun after this fix. Historical reproduction requires that archived runtime.

## Explicit tools decision cadence

`ToolsConfig.decision_ticks` now travels through generated training episodes,
scheduled evaluations and worker reset to `ToolsEnv`. The `night_tools` CLI exposes
`--decision-ticks`. The legacy default remains 20; use `--decision-ticks 2` explicitly
for the two-tick weekend demonstrations. This option configures PPO and evaluations,
not collection, previously recorded data or the discount factor. Confirm that their
cadences agree before training. Explicit cadence overrides on queued episodes are
retained. Private episode-generating subclasses must also propagate the setting.

The value must be an integer of at least two: query-dependent commands may need one
query tick and one action tick. Validation rejects malformed values before a game
is constructed. Reset metadata reports the configured cadence; actual transition
ticks in recordings remain the evidence of elapsed simulation time.

Saved trainer configs include cadence. Loading logs saved and active cadence,
warns on an explicit mismatch and keeps the runtime config authoritative. Missing
historical metadata is reported as unavailable, not guessed from tensor weights.
Invalid explicit saved metadata is rejected before loading weights or optimizer
state. This configuration plumbing is covered by CPU/mock integration tests; it
has not launched PPO or established that its critic and credit assignment are ready.

## Critic validation episode boundaries

Warmup validation now scores only contiguous withheld episodes with both a reset
(`first_is_start`) and a terminal transition. If a worker crashes, a later reset
discards its unfinished prefix rather than assigning replacement-game rewards to it.
Missing prefixes and unfinished tails are excluded and counted in the report;
malformed terminal boundaries are rejected. Validation predictions are restricted
to the same selected rows as the targets.

A CPU/mock reproduction previously scored four decisions across an abandoned
two-decision prefix and a completed two-decision replacement. The corrected validator
scores only the latter two. Eleven focused checks cover crash/reset boundaries,
complete continuations, missing prefixes, target alignment and recorded returns.
No new critic warmup or PPO run accompanies this fix; previous calibration results
are not rehabilitated by changing the validator.
The report now declares `worker0_reset_to_terminal_v1`; the bounded PPO entry point
rejects legacy calibration reports without this contract, even if their old
`ppo_ready` flag is true. No historical report is rewritten to satisfy that gate.

The current environment treats its game time limit as a terminal task outcome with
a terminal value-advantage reward. Its zero terminal bootstrap is therefore deliberate.
A future continuing-task interpretation would require changing that contract, not
simply adding a bootstrap value to the existing terminal reward.

The bounded `critic_warmup` and `weekend_ppo` spec entry points require an explicit
integer `decision_ticks` field. Their generated `ToolsConfig` uses that value;
old specs without it fail instead of silently running at the generic 20-tick
default. Warmup validation records its configured cadence, and the PPO gate
requires it to match the new spec before constructing a trainer. Missing or
malformed cadence metadata cannot authorize PPO. Seventeen focused CPU/mock
checks cover these adapters, the gate, generic worker plumbing and episode
boundaries. Historical specs and reports remain unchanged; no live warmup or PPO
run has been performed with these changes.
