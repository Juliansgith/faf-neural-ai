"""The learner's side of a player machine (fa_train.player): pull finished rounds, push a network, show status.
Everything goes over SSH from this side (the player needs no way in).

    python -m fa_train.server_sync status
    python -m fa_train.server_sync pull [--dest D:/faf-server/rounds] [--keep]
    python -m fa_train.server_sync push CHECKPOINT
    python -m fa_train.server_sync config '{"lanes": 3, "ai": "medium,adaptive,m28ai"}'

--host is an SSH alias (default my-server, an alias in ~/.ssh/config), --home the player's folder there (default ~/fa/player).
A pulled round is checked (every recording the round lists has arrived) before it is deleted on the player.
Each pulled round gets an ARRIVED stamp: its arrival on this side. tar keeps the player's folder times, and the RL
loop's archiving counted a round pulled a day late as a day old before it was encoded (2026-10-16 review, F6).
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path


# a dead connection ends the command instead of hanging it (2026-10-11: a pull hung for 25 min and stalled RL)
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4"]
INCOMPLETE_TRIES = 3     # pulls of a round that arrives short before it is kept as it is


def ssh(host: str, command: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, host, command],
                          check=check, **kw)


def remote_rounds(host: str, home: str) -> list[str]:
    out = ssh(host, f"cd {home}/rounds 2>/dev/null && ls -d lane*_*/DONE 2>/dev/null | sed 's|/DONE||'",
              check=False, capture_output=True, text=True).stdout
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def mark_arrived(folder: Path) -> None:
    """The ARRIVED stamp: when a round reached this machine (rl_loop.archive_old counts its age from here)."""
    (folder / "ARRIVED").write_text(datetime.now().isoformat(timespec="seconds"))


def _free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return float("inf")


def pull(host: str, home: str, dest: Path, keep: bool, min_free_gb: float = 0.0) -> int:
    """Every finished round of the player into `dest` (and off the player unless `keep`). Never raises for one
    round: a broken transfer, a full disk or an earlier incomplete copy leaves that round on the player and the
    pull goes on with the next one. 2026-10-16 review (F9): a rename onto an earlier incomplete copy raised
    FileExistsError outside the try, on the same round in every pass and after every restart, and ended the RL loop
    for good. `min_free_gb`: stop pulling while the destination's disk has less free (the rounds stay on the
    player)."""
    dest.mkdir(parents=True, exist_ok=True)
    names = remote_rounds(host, home)
    pulled = 0
    for name in names:
        if min_free_gb and _free_gb(dest) < min_free_gb:
            print(f"disk low ({_free_gb(dest):.0f} GB free on {dest.anchor}): pulling stops, {name} and later "
                  f"rounds stay on the player")
            break
        try:
            if (dest / name / "DONE").exists():
                print(f"{name}: already here")
            elif not _pull_one(host, home, dest, name):
                continue                                # left on the player (the reason is printed)
            else:
                pulled += 1
        except OSError as exc:                          # a full disk, a locked file: the round stays on the player
            print(f"{name}: could not be stored here ({type(exc).__name__}: {exc}); left on the player")
            continue
        if not keep:
            ssh(host, f"rm -rf {home}/rounds/{shlex.quote(name)}", check=False)
    print(f"{pulled} new rounds in {dest}")
    return pulled


def _pull_one(host: str, home: str, dest: Path, name: str) -> bool:
    """One round over ssh+tar into dest/name. True when it arrived complete (DONE kept, ARRIVED written)."""
    target = dest / name
    t0 = time.monotonic()
    tmp = dest / f".{name}.part"
    if tmp.exists():
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
    except (tarfile.TarError, OSError) as exc:      # a cut connection or a full disk: the round stays on the player
        proc.kill()
        proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"{name}: transfer broke off ({type(exc).__name__}: {exc}); left on the player")
        return False
    if proc.wait() != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"{name}: transfer failed (ssh exit {proc.returncode}); left on the player")
        return False
    aside_dir = dest / "_incomplete"
    if target.exists():
        # an earlier incomplete copy (its DONE was removed below): set aside, not merged -- the new copy is the
        # whole round as the player has it now. One copy per round is kept for a look (a round that arrives
        # incomplete on every pass must not fill the disk with copies)
        aside = aside_dir / name
        aside_dir.mkdir(parents=True, exist_ok=True)
        if aside.exists():
            shutil.rmtree(aside, ignore_errors=True)
        target.rename(aside)
        print(f"{name}: an earlier incomplete copy moved to {aside}")
    (tmp / name).rename(target)
    tmp.rmdir()
    mark_arrived(target)
    meta = json.loads((target / "round.json").read_text())
    have = len(list((target / "raw").glob("*.jsonl.gz")))
    if have != meta.get("games") or not (target / "raw" / "blueprints.json.gz").exists():
        attempts_file = aside_dir / f"{name}.attempts"
        try:
            attempts = int(attempts_file.read_text()) + 1
        except (OSError, ValueError):
            attempts = 1
        if attempts < INCOMPLETE_TRIES:
            aside_dir.mkdir(parents=True, exist_ok=True)
            attempts_file.write_text(str(attempts))
            print(f"{name}: incomplete ({have} of {meta.get('games')} recordings, attempt {attempts}); left on the "
                  f"player")
            (target / "DONE").unlink(missing_ok=True)
            return False
        # the same shortfall every time: it is the round as the player has it (a recording lost there). Kept as it
        # is (the encoder takes what is there), else it would be pulled again on every pass for nine days
        (target / "INCOMPLETE").write_text(f"{have} of {meta.get('games')} recordings after {attempts} pulls")
        attempts_file.unlink(missing_ok=True)
        print(f"{name}: still incomplete after {attempts} pulls ({have} of {meta.get('games')} recordings); kept "
              f"as it is")
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"{name}: {have} games, {size / 2**20:.0f} MB in {time.monotonic() - t0:.0f} s; "
          f"{json.dumps({k: v.get('win_rate') for k, v in (meta.get('outcomes') or {}).items()})}")
    return True


def push(host: str, home: str, checkpoint: Path, timeout: float = 1800) -> None:
    """The network to the player (uploaded beside, then renamed: a lane never reads half a file). Raises
    FileNotFoundError for a missing checkpoint and CalledProcessError/TimeoutExpired for a failed transfer; the RL
    loop retries those (a sys.exit here ended the loop, 2026-10-16 review F2)."""
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"no such checkpoint: {checkpoint}")
    subprocess.run(["scp", "-q", *SSH_OPTS, str(checkpoint), f"{host}:{home}/incoming.pt"], check=True,
                   timeout=timeout)
    ssh(host, f"mv {home}/incoming.pt {home}/current.pt && sha256sum {home}/current.pt", timeout=300)
    try:                                                # a note for the operator, never a reason to fail a push
        with open(Path(__file__).resolve().parents[4] / "_runs" / "server_pushes.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {host} {checkpoint}\n")
    except OSError:
        pass
    print(f"pushed {Path(checkpoint).name}: the lanes play it from their next round")


def set_config(host: str, home: str, updates: dict) -> dict:
    """Merges `updates` into the player's config.json (the lanes read it before every round)."""
    cur = ssh(host, f"cat {home}/config.json 2>/dev/null || echo {{}}", capture_output=True, text=True,
              timeout=300).stdout
    cfg = json.loads(cur or "{}")
    cfg.update(updates)
    ssh(host, f"cat > {home}/config.json.new && mv {home}/config.json.new {home}/config.json",
        input=json.dumps(cfg, indent=1), text=True, timeout=300)
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
        try:
            push(a.host, a.home, Path(a.arg or ""))
        except FileNotFoundError as exc:
            sys.exit(str(exc))
    elif a.action == "config":
        new = json.loads(a.arg or "{}")
        cur = ssh(a.host, f"cat {a.home}/config.json 2>/dev/null || echo {{}}", capture_output=True, text=True).stdout
        cfg = {**json.loads(cur or "{}"), **new}
        ssh(a.host, f"cat > {a.home}/config.json", input=json.dumps(cfg, indent=1), text=True)
        print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
