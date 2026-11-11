"""Drain the local CPU player, run one frozen benchmark, then release our pause.

No running game or training process is killed. A failure leaves status.json and
releases only this controller's own PAUSE. A foreign pause is never removed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def freeze_runtime(source_ai: Path, m28: Path, destination: Path) -> tuple[Path, Path]:
    """Keep worker imports and Lua hooks fixed while development continues."""
    ai = destination / 'faf-neural-ai' / 'ai'
    mod = destination / '_deps' / 'M28AI'
    ignored = shutil.ignore_patterns('__pycache__', '.git', '.idea', 'tests')
    shutil.copytree(source_ai, ai, ignore=ignored)
    shutil.copytree(m28, mod, ignore=ignored)
    return ai / 'python', mod


def active_player_evaluations(home: Path) -> list[int]:
    # Query process command lines, not round folders that collection may move.
    code = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','pythonw.exe' } | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    p = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', code],
                       capture_output=True, text=True, timeout=60, check=True, creationflags=0x08000000)
    rows = json.loads(p.stdout or '[]')
    rows = [rows] if isinstance(rows, dict) else rows
    prefix = str(home.resolve()).replace('\\', '/').lower()
    return [r['ProcessId'] for r in rows
            if 'tools_eval' in str(r.get('CommandLine'))
            and prefix in str(r.get('CommandLine')).replace('\\', '/').lower()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--home', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--schedule', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--bin-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--adopt-pause', help='exact existing PAUSE marker to transfer from this operator')
    a = ap.parse_args()
    for key in ('home', 'checkpoint', 'schedule', 'run', 'bin_dir'):
        setattr(a, key, getattr(a, key).resolve())
    if sys.platform != 'win32':
        raise SystemExit('This local-player controller is Windows-only')
    if a.run.exists():
        raise SystemExit('Run directory exists; refusing to mix evaluations')
    a.run.mkdir(parents=True)
    status_path = a.run / 'controller.json'
    state = dict(pid=os.getpid(), checkpoint=str(a.checkpoint), schedule=str(a.schedule))

    def status(phase, **extra):
        state.update(phase=phase, time=time.time(), **extra)
        tmp = status_path.with_suffix('.new')
        tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        tmp.replace(status_path)
        print(json.dumps(state), flush=True)

    pause = a.home / 'PAUSE'
    marker = f'benchmark:{uuid.uuid4()}:{a.run}'
    acquired = False
    try:
        if a.adopt_pause is not None:
            if pause.read_text(encoding='utf-8') != a.adopt_pause:
                raise RuntimeError('Existing pause does not match the requested transfer')
            pause.write_text(marker, encoding='utf-8')
        else:
            with pause.open('x', encoding='utf-8') as f:
                f.write(marker)
        acquired = True
        status('draining_local_games')
        deadline = time.time() + 2 * 3600
        while active_player_evaluations(a.home):
            if time.time() > deadline:
                raise RuntimeError('Local rounds did not drain within two hours; no processes killed')
            if pause.read_text(encoding='utf-8') != marker:
                raise RuntimeError('Pause ownership changed; benchmark cancelled')
            time.sleep(60)
        if pause.read_text(encoding='utf-8') != marker:
            raise RuntimeError('Pause ownership changed; benchmark cancelled')
        from fa_env.bridge import KNOWN_MODS
        python_root, m28 = freeze_runtime(Path(__file__).resolve().parents[2],
                                         KNOWN_MODS['M28AI'], a.run / 'runtime')
        frozen_checkpoint = a.run / 'checkpoint.pt'
        shutil.copy2(a.checkpoint, frozen_checkpoint)
        frozen_schedule = a.run / 'schedule.json'
        shutil.copy2(a.schedule, frozen_schedule)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTHONPATH=str(python_root), FA_ENV_M28_DIR=str(m28))
        cmd = [sys.executable, '-u', '-m', 'examples.tools_eval', '--checkpoint', str(frozen_checkpoint),
               '--schedule-file', str(frozen_schedule), '--run', str(a.run / 'eval'), '--bin-dir', str(a.bin_dir),
               '--record', str(a.run / 'raw'),
               '--workers', str(a.workers), '--device', 'cpu', '--mode', 'sample', '--temperature', '.5']
        status('evaluating_cpu')
        with (a.run / 'eval.log').open('w', encoding='utf-8') as log:
            code = subprocess.call(cmd, cwd=python_root, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=0x08000000)
        if code:
            raise RuntimeError(f'Benchmark failed with exit {code}; see eval.log')
        result = json.loads((a.run / 'eval/benchmark.json').read_text(encoding='utf-8'))
        if not result['complete']:
            raise RuntimeError('Incomplete benchmark')
        status('complete', result=result['total'])
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if acquired and pause.exists() and pause.read_text(encoding='utf-8') == marker:
            pause.unlink()


if __name__ == '__main__':
    main()
