"""Run a predeclared goal confirmation after the autopilot has finalized.

One champion, copied runtime and owned local-player pause cover all suites.
Sentinels are separate from the target denominator. No training intake, retries
by this controller, checkpoint selection, or early stopping on a good score.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from fa_train.benchmark import schedule_hash, summarize, validate_rows, WEIGHTED_MAPS
from fa_train.local_benchmark import active_player_evaluations, freeze_runtime


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentinel_suites(seed_base):
    """Small descriptive suites, separate from the 640-game target holdout."""
    from fa_env.maps import MAP_POOL
    result = {'scripted': [], 'full_m28': [], 'large_maps': []}
    small = list(dict.fromkeys(WEIGHTED_MAPS))

    def add(name, ai, map_id, own, enemy, side, minutes):
        seed = seed_base + sum(len(v) for v in result.values())
        result[name].append(dict(ai=ai, map=map_id, faction=own, ai_faction=enemy,
                                 slots=list(MAP_POOL[map_id]['slots'][side]), seed=seed, minutes=minutes))

    for a, ai in enumerate(('adaptive', 'rush', 'tech')):
        for m, map_id in enumerate(small):
            for own in range(1, 5):
                add('scripted', ai, map_id, own, (own + m + a) % 4 + 1, (own + m) % 2, 40.)
    for m, map_id in enumerate(small):
        for own in range(1, 5):
            for side in range(2):
                add('full_m28', 'm28ai', map_id, own, (own + m + side) % 4 + 1, side, 40.)
    for map_id in (m for m, spec in MAP_POOL.items() if spec['km'] == 20):
        for own in range(1, 5):
            for side in range(4):
                add('large_maps', 'm28aicheat@0.9', map_id, own, (own - 1 + side) % 4 + 1, side, 90.)
    return result


def load_suites(plan):
    """Validate the complete predeclared suite without running or scoring games."""
    from collections import Counter
    from fa_env.maps import MAP_POOL
    suites, seeds = {}, set()
    for name, spec in plan['suites'].items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
            raise ValueError('Suite name must be a simple lowercase identifier')
        data = read_json(spec['path'])
        rows = validate_rows(data['rows'])
        if schedule_hash(rows) != spec['sha256'] or data.get('schedule_sha256') != spec['sha256']:
            raise ValueError(f'{name}: predeclared schedule hash mismatch')
        own_seeds = {r['seed'] for r in rows}
        if len(own_seeds) != len(rows) or seeds & own_seeds:
            raise ValueError('Every confirmation/sentinel game needs a distinct seed')
        seeds.update(own_seeds)
        suites[name] = rows
    target = suites.get('confirmation', [])
    if len(target) < 640 or len(target) % 320 or any(r['ai'] != 'm28aicheat@0.9' for r in target):
        raise ValueError('Confirmation needs >=640 target games in complete 320-game blocks')
    # Compare joint strata, not merely the faction and map marginals.
    expected = Counter((m, own, enemy, tuple(MAP_POOL[m]['slots'][side]))
                       for _ in range(len(target) // 320) for own in range(1, 5)
                       for enemy in range(1, 5) for side in range(2) for m in WEIGHTED_MAPS)
    actual = Counter((r['map'], r['faction'], r['ai_faction'], tuple(r['slots'])) for r in target)
    if actual != expected or any(r['minutes'] != 40 for r in target):
        raise ValueError('Confirmation must preserve the declared map/faction/side weights and 40-minute limit')
    if not {'scripted', 'full_m28', 'large_maps'} <= suites.keys():
        raise ValueError('Separate scripted, full M28 and large-map sentinels are required')
    if set(suites) != {'confirmation', 'scripted', 'full_m28', 'large_maps'}:
        raise ValueError('Unexpected suite in frozen confirmation plan')
    def strata(rows):
        return Counter((r['ai'], r['map'], r['faction'], r['ai_faction'], tuple(r['slots']), r['minutes'])
                       for r in rows)
    for name, expected_rows in sentinel_suites(0).items():
        if strata(suites[name]) != strata(expected_rows):
            raise ValueError(f'{name}: sentinel coverage differs from the declared protocol')
    return suites


def require_finalized(plan, now=None):
    now = now or datetime.now().astimezone()
    if now < datetime.fromisoformat(plan['not_before']):
        raise RuntimeError('Confirmation is not due; holdout remains unused')
    if now > datetime.fromisoformat(plan['deadline']):
        raise RuntimeError('The goal deadline has passed; do not silently extend the run')
    if read_json(Path(plan['training_run']) / 'autopilot/state.json').get('mode') != 'finalized':
        raise RuntimeError('Autopilot has not finalized; do not freeze a moving selection')


def verify_result(folder, rows, pinned, reference=None):
    """Recompute the denominator from raw result rows; fail closed on provenance."""
    benchmark = read_json(folder / 'eval/benchmark.json')
    manifest = read_json(folder / 'eval/eval_manifest.json')
    results = read_json(folder / 'eval/eval_results.json')['games']
    calculated = summarize(rows, results)
    for key in ('complete', 'evaluation_valid', 'provenance_unchanged'):
        if benchmark.get(key) is not True:
            raise RuntimeError(f'{folder.name}: {key} is not true')
    if benchmark.get('restarted_attempts') != 0 or not calculated['complete']:
        raise RuntimeError(f'{folder.name}: incomplete or retried evaluation')
    if (benchmark.get('schedule_sha256') != schedule_hash(rows)
            or manifest.get('schedule_sha256') != schedule_hash(rows)
            or manifest.get('rows') != rows
            or benchmark.get('total') != calculated['total']):
        raise RuntimeError(f'{folder.name}: schedule/result accounting mismatch')
    for key, value in pinned.items():
        if manifest.get(key) != value:
            raise RuntimeError(f'{folder.name}: frozen {key} mismatch')
    if reference is not None:
        for key in ('policy_source_sha256',):
            if manifest.get(key) != reference.get(key):
                raise RuntimeError(f'{folder.name}: source snapshot mismatch')
        for key in ('bridge_sha256', 'hooks_sha256'):
            if manifest['game_sources'][key] != reference['game_sources'][key]:
                raise RuntimeError(f'{folder.name}: game source mismatch')
        for mod, value in manifest['game_sources']['mods'].items():
            if mod in reference['game_sources']['mods'] and value != reference['game_sources']['mods'][mod]:
                raise RuntimeError(f'{folder.name}: opponent source mismatch')
    return calculated, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = read_json(args.plan)
    suites = load_suites(plan)
    if args.validate_only:
        print(json.dumps({'valid': True, 'games': {k: len(v) for k, v in suites.items()},
                          'holdout_executed': False}))
        return
    if sys.platform != 'win32':
        raise SystemExit('The local confirmation controller is Windows-only')
    require_finalized(plan)
    run, home = Path(plan['run']), Path(plan['home'])
    run.mkdir(parents=True, exist_ok=False)  # never overwrite or splice an attempt
    marker = 'goal-confirmation:' + uuid.uuid4().hex
    pause = home / 'PAUSE'
    acquired = False
    state = dict(pid=os.getpid(), plan=plan, phase='preparing', suites={})

    def status(phase, **fields):
        state.update(phase=phase, time=datetime.now().astimezone().isoformat(), **fields)
        atomic_json(run / 'controller.json', state)
        print(json.dumps({'phase': phase, **fields}), flush=True)

    try:
        status('waiting_for_local_pause')
        deadline = time.monotonic() + 2 * 3600
        while not acquired:
            try:
                with pause.open('x', encoding='utf-8') as f:
                    f.write(marker)
                acquired = True
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Foreign pause did not clear within two hours')
                time.sleep(60)
        status('draining_local_games')
        while active_player_evaluations(home):
            if pause.read_text(encoding='utf-8') != marker or time.monotonic() > deadline:
                raise RuntimeError('Pause ownership changed or local drain exceeded two hours')
            time.sleep(60)
        require_finalized(plan)
        champion = read_json(Path(plan['training_run']) / 'champion.json')
        source = Path(plan['training_run']) / 'checkpoints' / (champion['name'] + '.pt')
        checkpoint = run / 'checkpoint.pt'
        shutil.copy2(source, checkpoint)
        from fa_env.bridge import KNOWN_MODS
        pyroot, m28 = freeze_runtime(Path(__file__).resolve().parents[2], KNOWN_MODS['M28AI'], run / 'runtime')
        pinned = dict(checkpoint_sha256=sha256(checkpoint),
                      engine_sha256=sha256(Path(plan['bin_dir']) / 'main.exe'),
                      policy='sample T=0.5 top_p=None', build_budget=None,
                      commander_greedy=bool(plan.get('commander_greedy', False)))
        atomic_json(run / 'selection.json', dict(champion=champion, **pinned,
                                               time=datetime.now().astimezone().isoformat()))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', PYTHONPATH=str(pyroot), FA_ENV_M28_DIR=str(m28))
        check = subprocess.run([sys.executable, '-c',
                                'import torch; assert torch.get_num_threads() == 2; print(torch.get_num_threads())'],
                               cwd=pyroot, env=env, capture_output=True, text=True, check=True,
                               timeout=60, creationflags=0x08000000)
        atomic_json(run / 'resources.json', dict(device='cpu', workers=plan.get('workers', 10),
                                               verified_torch_threads=int(check.stdout.strip())))
        reference = None
        for name in ['confirmation'] + [n for n in suites if n != 'confirmation']:
            if pause.read_text(encoding='utf-8') != marker:
                raise RuntimeError('Pause ownership changed; no next suite launched')
            folder = run / name
            folder.mkdir()
            schedule = folder / 'schedule.json'
            atomic_json(schedule, dict(rows=suites[name], schedule_sha256=schedule_hash(suites[name])))
            cmd = [sys.executable, '-u', '-m', 'examples.tools_eval', '--checkpoint', str(checkpoint),
                   '--schedule-file', str(schedule), '--run', str(folder / 'eval'), '--record', str(folder / 'raw'),
                   '--bin-dir', plan['bin_dir'], '--workers', str(plan.get('workers', 10)),
                   '--device', 'cpu', '--mode', 'sample', '--temperature', '.5']
            if pinned['commander_greedy']:
                cmd.append('--commander-greedy')
            status('evaluating_cpu', active_suite=name)
            with (folder / 'eval.log').open('w', encoding='utf-8') as log:
                code = subprocess.call(cmd, cwd=pyroot, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, creationflags=0x08000000)
            if code:
                raise RuntimeError(f'{name}: evaluator exited {code}; preserve failed attempt')
            result, manifest = verify_result(folder, suites[name], pinned, reference)
            reference = reference or manifest
            state['suites'][name] = result
            status('suite_complete', active_suite=name)
        target = state['suites']['confirmation']
        finished = datetime.now().astimezone()
        in_time = finished <= datetime.fromisoformat(plan['deadline'])
        atomic_json(run / 'result.json', dict(target=target, sentinels={n: r for n, r in state['suites'].items()
                                                                     if n != 'confirmation'},
                                             selection=read_json(run / 'selection.json'),
                                             finished_at=finished.isoformat(), completed_by_deadline=in_time,
                                             statistical_target_met=target['statistical_threshold_met'],
                                             note='Sentinels reported separately. No claim of improvement without a comparable baseline.'))
        status('complete', statistical_target_met=target['statistical_threshold_met'],
               completed_by_deadline=in_time)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if acquired and pause.exists() and pause.read_text(encoding='utf-8') == marker:
            pause.unlink()


if __name__ == '__main__':
    main()
