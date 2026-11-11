"""A guard for the orchestration tests: no test may reach the player server or start a real training/game process.

2026-10-17: an autopilot test that ran one non-dry tick without mocking server_sync.set_config wrote "maps" and
"minutes" into the live server's player config (restored by hand). Imported by every rl_loop/autopilot test module,
install() replaces -- at import, for the whole test process -- server_sync's ssh/push/pull/set_config/remote_rounds
with functions that raise, and makes subprocess refuse ssh/scp/taskkill and launches of rl_loop, learn_rl, the
player, the autopilot or its watchdog. A test that needs one of them patches it with a recorder on top
(mock.patch restores the guard afterwards); REAL keeps the originals for tests of server_sync itself, which then
patch the ssh/Popen underneath. NetworkBlocked derives from BaseException: no `except Exception` in the code under
test can swallow it, and BLOCKED lists every attempt.
"""
from __future__ import annotations

import os
import subprocess
import urllib.request

BLOCKED: list = []
REAL: dict = {}
_FORBIDDEN_EXE = {"ssh", "scp", "taskkill", "sftp", "rsync"}
_FORBIDDEN_MODULES = ("fa_train.rl_loop", "fa_train.learn_rl", "fa_train.player", "fa_train.autopilot",
                      "fa_train.encode_parallel", "tools_eval")


class NetworkBlocked(BaseException):
    """A test tried to reach the server or start a real process."""


def _argv(args) -> list:
    if isinstance(args, (list, tuple)):
        return [str(a) for a in args]
    return str(args).split()


def _check(args) -> None:
    argv = _argv(args)
    if not argv:
        return
    exe = os.path.basename(argv[0]).lower()
    exe = exe[:-4] if exe.endswith(".exe") else exe
    if exe in _FORBIDDEN_EXE or any(m in a for a in argv for m in _FORBIDDEN_MODULES):
        BLOCKED.append(argv)
        raise NetworkBlocked(f"blocked in tests: {' '.join(argv)[:200]}")


def _wrap(fn):
    def guarded(*args, **kw):
        _check(args[0] if args else kw.get("args", []))
        return fn(*args, **kw)
    guarded._faf_guard = True
    return guarded


def _raiser(name):
    def blocked(*args, **kw):
        BLOCKED.append([name, *map(str, args)])
        raise NetworkBlocked(f"blocked in tests: server_sync.{name}{args!r}"[:300])
    blocked._faf_guard = True
    return blocked


def install() -> None:
    from fa_train import server_sync
    if getattr(subprocess.Popen, "_faf_guard", False):
        REAL.update(getattr(server_sync, "_faf_real", {}))   # (imported again under another module name)
        return
    for name in ("run", "call", "check_call", "check_output"):
        setattr(subprocess, name, _wrap(getattr(subprocess, name)))
    real_popen = subprocess.Popen

    class GuardedPopen(real_popen):                     # (a subclass: code that type-checks Popen keeps working)
        _faf_guard = True

        def __init__(self, args, *a, **kw):
            _check(args)
            super().__init__(args, *a, **kw)
    subprocess.Popen = GuardedPopen

    def no_urlopen(*a, **kw):
        BLOCKED.append(["urlopen", *map(str, a)])
        raise NetworkBlocked("blocked in tests: urlopen")
    urllib.request.urlopen = no_urlopen
    for name in ("ssh", "push", "pull", "set_config", "remote_rounds", "status"):
        REAL[name] = getattr(server_sync, name)
        setattr(server_sync, name, _raiser(name))
    server_sync._faf_real = dict(REAL)


install()
