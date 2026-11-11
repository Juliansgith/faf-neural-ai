"""Compare full CPU PPO updates on current encoded recordings, without saving a policy.

python -m fa_train.benchmark_cpu --init checkpoint.pt --cache current_cache --cpu-threads 8
Use separate invocations for thread/affinity choices. --cpu-bf16 tests encoder precision.
--profile prints CPU operator costs. Replay prefixes supply the actual recurrent history.
"""
import argparse
import copy
import json
import platform
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

from .cpu_runtime import add_cpu_arguments, configure_cpu, configure_model
from .learn_rl import chunk_batch, load_net, ppo_step
from .tools_bc import Games
from .probability_accounting import compatible_ppo_games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--streams', type=int, default=2)
    parser.add_argument('--length', type=int, default=16)
    parser.add_argument('--micro', type=int, default=1)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=.5)
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--out', type=Path, help='optional JSON timings and numerical comparison')
    add_cpu_arguments(parser)
    args = parser.parse_args()
    for key in ('streams', 'length', 'micro', 'repeats'):
        if getattr(args, key) < 1:
            parser.error(f'--{key} must be positive')
    if args.start_step < 0:
        parser.error('--start-step must be nonnegative')
    runtime = configure_cpu(args, 'cpu')
    torch.manual_seed(42)
    base, _ = load_net(args.init, 'cpu')
    games = Games(args.cache, newest=args.streams * 4)
    games.units = True
    games.air_refueling = base.air_refueling
    games.queued_factory_upgrades = base.queued_factory_upgrades
    ids = [i for i, game in enumerate(games.games)
           if 'pol_has' in game and bool(np.asarray(game['pol_has']).any())
           and games.lengths[i] >= args.start_step + args.length]
    ids, _ = compatible_ppo_games(base, games, ids, temperature=args.temperature)
    ids = ids[:args.streams]
    if len(ids) < args.streams:
        parser.error('Too few compatible current recordings at the requested start step')
    state = base.initial_state(args.streams, 'cpu')
    with torch.no_grad():
        for start in range(0, args.start_step, args.length):
            length = min(args.length, args.start_step - start)
            prefix = chunk_batch(games, [(i, start) for i in ids], length, 'cpu')
            _, _, _, state = base.core(prefix['b'], state, length, prefix['starts'], return_state=True)
        start = time.perf_counter()
        c = chunk_batch(games, [(i, args.start_step) for i in ids], args.length, 'cpu')
        collation_seconds = time.perf_counter() - start
        h, m, z = base.core(c['b'], state, args.length, c['starts'])
        _, ent, _ = base.score(h, m, z, c['b'], c['act'], temperature=args.temperature)
        c['old_ulp'] = c['prox_ulp'] = c['ref_ulp'] = ent['unit_logp'].detach()
        c['v_old'] = base.value_of(z, c['b']).detach()
    if not c['valid'].any():
        parser.error('The selected chunk has no recorded policy decisions')
    # Fixed benchmark targets exercise both policy and critic gradients. No strength inference.
    count = args.streams * args.length
    c['adv'] = torch.linspace(-1, 1, count)
    c['adv_up'] = c['adv'].clamp(min=0)
    c['ret'] = c['v_old'] + .1 * c['adv']
    config = SimpleNamespace(length=args.length, micro=args.micro, temperature=args.temperature,
                             distill=.1, no_undo_labels=False, local_credit=0., upgo=.5,
                             clip=.2, value_clip=0., vf_coef=.5, ent_coef=0., ref_coef=.1,
                             max_grad=.5, lr=1e-5, warmup=1)
    report = dict(runtime=runtime, platform=platform.platform(), checkpoint=str(args.init.resolve()),
                  sources=[games.games[i]['_cache_manifest']['source_name'] for i in ids],
                  start_step=args.start_step, streams=args.streams, length=args.length, micro=args.micro,
                  valid_decisions=int(c['valid'].sum()), teacher_decisions=int(c['sup'].sum()),
                  max_tokens=c['b']['tokens'].shape[1], collation_seconds=collation_seconds,
                  scope='Repeated PPO update on fixed recorded observations/actions and synthetic targets; excludes loading, prefix replay and game collection',
                  measurements=[])
    reference_weights = None
    for mode in ('baseline', 'optimized'):
        optimized = mode == 'optimized'
        config.separate_decoder = config.no_trim_batches = not optimized
        timings, cpu_times = [], []
        for repeat in range(args.repeats + 1):
            # Each timing starts at the same weights, memory and fresh Adam state.
            net = copy.deepcopy(base)
            net.sequence_lstm = optimized
            net.grad_checkpoint = args.grad_checkpoint
            configure_model(net, args)
            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=config.lr, eps=1e-5)
            config.step = 0
            wall, cpu = time.perf_counter(), time.process_time()
            stats, _, _ = ppo_step(net, net, opt, c, state, state, config)
            seconds, cpu_seconds = time.perf_counter() - wall, time.process_time() - cpu
            if not all(np.isfinite(value) for value in stats.values()):
                raise RuntimeError(f'Nonfinite update in {mode}')
            if repeat:
                timings.append(seconds)
                cpu_times.append(cpu_seconds)
            else:
                cold_seconds = seconds
        weights = {key: value.detach().clone() for key, value in net.state_dict().items()}
        if reference_weights is None:
            reference_weights = weights
        difference = max(float((weights[key] - reference_weights[key]).abs().max()) for key in weights)
        row = dict(mode=mode, cold_seconds=cold_seconds, wall_seconds=timings,
                   median_seconds=statistics.median(timings), median_cpu_seconds=statistics.median(cpu_times),
                   max_parameter_difference_from_baseline=difference, losses=stats)
        report['measurements'].append(row)
        print(json.dumps(row), flush=True)
        if args.profile and optimized:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        record_shapes=True, profile_memory=True) as profiler:
                ppo_step(net, net, opt, c, state, state, config)
            print(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))
    report['speedup'] = report['measurements'][0]['median_seconds'] / report['measurements'][1]['median_seconds']
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
