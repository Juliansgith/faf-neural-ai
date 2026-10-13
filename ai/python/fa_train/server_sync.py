"""The learner's side of a player machine (fa_train.player): pull finished rounds, push a network, show status.
Everything goes over SSH from this side (the player needs no way in).

    python -m fa_train.server_sync status
    python -m fa_train.server_sync pull [--dest D:/faf-server/rounds] [--keep]
    python -m fa_train.server_sync push CHECKPOINT
    python -m fa_train.server_sync config '{"lanes": 3, "ai": "medium,adaptive,m28ai"}'

--host is an SSH alias (default my-server, an alias in ~/.ssh/config), --home the player's folder there (default ~/fa/player).
A pulled round is checked (every recording the round lists has arrived) before it is deleted on the player.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path


# a dead connection ends the command instead of hanging it (2026-10-11: a pull hung for 25 min and stalled RL)
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4"]


def ssh(host: str, command: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, host, command],
                          check=check, **kw)


def remote_rounds(host: str, home: str) -> list[str]:
    out = ssh(host, f"cd {home}/rounds 2>/dev/null && ls -d lane*_*/DONE 2>/dev/null | sed 's|/DONE||'",
              check=False, capture_output=True, text=True).stdout
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def pull(host: str, home: str, dest: Path, keep: bool) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    names = remote_rounds(host, home)
    pulled = 0
    for name in names:
        target = dest / name
        if (target / "DONE").exists():
            print(f"{name}: already here")
        else:
            t0 = time.monotonic()
            tmp = dest / f".{name}.part"
            if tmp.exists():
                import shutil
                shutil.rmtree(tmp)
            tmp.mkdir()
            proc = subprocess.Popen(["ssh", *SSH_OPTS, host,
                                     f"tar -cf - -C {home}/rounds {shlex.quote(name)}"], stdout=subprocess.PIPE)
            try:
                with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                    tar.extractall(tmp, filter="data")
                # tar pads its stream to whole 10 KB records after the end marker, which the reader above stops
                # short of: unread, the padding fills the pipe, ssh blocks writing it and never exits (the RL
                # loop hung in proc.wait() twice on 2026-10-11)
                while proc.stdout.read(1 << 16):
                    pass
            except (tarfile.TarError, OSError) as exc:      # a cut connection: the round stays on the player
                proc.kill()
                proc.wait()
                print(f"{name}: transfer broke off ({type(exc).__name__}: {exc}); left on the player")
                continue
            if proc.wait() != 0:
                print(f"{name}: transfer failed (ssh exit {proc.returncode}); left on the player")
                continue
            (tmp / name).rename(target)
            tmp.rmdir()
            meta = json.loads((target / "round.json").read_text())
            have = len(list((target / "raw").glob("*.jsonl.gz")))
            if have != meta.get("games") or not (target / "raw" / "blueprints.json.gz").exists():
                print(f"{name}: incomplete ({have} of {meta.get('games')} recordings); left on the player")
                (target / "DONE").unlink(missing_ok=True)
                continue
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            print(f"{name}: {have} games, {size / 2**20:.0f} MB in {time.monotonic() - t0:.0f} s; "
                  f"{json.dumps({k: v.get('win_rate') for k, v in (meta.get('outcomes') or {}).items()})}")
            pulled += 1
        if not keep:
            ssh(host, f"rm -rf {home}/rounds/{shlex.quote(name)}", check=False)
    print(f"{pulled} new rounds in {dest}")
    return pulled


def push(host: str, home: str, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        sys.exit(f"no such checkpoint: {checkpoint}")
    subprocess.run(["scp", "-q", *SSH_OPTS, str(checkpoint), f"{host}:{home}/incoming.pt"], check=True)
    ssh(host, f"mv {home}/incoming.pt {home}/current.pt && sha256sum {home}/current.pt")
    with open(Path(__file__).resolve().parents[4] / "_runs" / "server_pushes.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {host} {checkpoint}\n")
    print(f"pushed {checkpoint.name}: the lanes play it from their next round")


def set_config(host: str, home: str, updates: dict) -> dict:
    """Merges `updates` into the player's config.json (the lanes read it before every round)."""
    cur = ssh(host, f"cat {home}/config.json 2>/dev/null || echo {{}}", capture_output=True, text=True).stdout
    cfg = json.loads(cur or "{}")
    cfg.update(updates)
    ssh(host, f"cat > {home}/config.json.new && mv {home}/config.json.new {home}/config.json",
        input=json.dumps(cfg, indent=1), text=True)
    return cfg


def status(host: str, home: str) -> None:
    ssh(host, "uptime; free -g | sed -n 2p; df -h / | tail -n 1; "
              f"echo \"games running: $(pgrep -c main.exe)\"; systemctl is-active fa-player 2>/dev/null; "
              f"ls {home}/STOP {home}/PAUSE 2>/dev/null; sha256sum {home}/current.pt 2>/dev/null | cut -c1-16; "
              f"echo \"finished rounds waiting: $(ls -d {home}/rounds/lane*_*/DONE 2>/dev/null | wc -l)\"; "
              f"journalctl -u fa-player --no-pager -n 12 -o cat 2>/dev/null | cut -c1-200", check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("status", "pull", "push", "config"))
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--host", default="my-server")
    ap.add_argument("--home", default="~/fa/player")
    ap.add_argument("--dest", type=Path, default=Path("D:/faf-server/rounds"))
    ap.add_argument("--keep", action="store_true", help="pull: leave the rounds on the player")
    a = ap.parse_args()
    if a.action == "status":
        status(a.host, a.home)
    elif a.action == "pull":
        pull(a.host, a.home, a.dest, a.keep)
    elif a.action == "push":
        push(a.host, a.home, Path(a.arg or ""))
    elif a.action == "config":
        new = json.loads(a.arg or "{}")
        cur = ssh(a.host, f"cat {a.home}/config.json 2>/dev/null || echo {{}}", capture_output=True, text=True).stdout
        cfg = {**json.loads(cur or "{}"), **new}
        ssh(a.host, f"cat > {a.home}/config.json", input=json.dumps(cfg, indent=1), text=True)
        print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
