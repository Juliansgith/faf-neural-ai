"""Transport and process plumbing for the agent bridge (engine side: src/sdk/moho/agent/AgentBridge.cpp).

The agent listens on 127.0.0.1; the game is launched with `/agentbridge <port>` and connects to it.
Messages are single-line JSON objects in both directions. The game sends one message per agent step
and blocks its sim thread until it gets one reply.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# FA_ENV_BIN_DIR: run games from another bin folder (a test build laid out like FAF's, next to the same
# gamedata), e.g. a night run on a build that is not deployed to FAF's own folder
DEFAULT_BIN_DIR = Path(os.environ.get("FA_ENV_BIN_DIR") or r"C:\ProgramData\FAForever\bin")
AI_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = AI_DIR / "agentbridge.lua"
OVERRIDES_DIR = AI_DIR / "fa_overrides"   # its schook folder is mounted at /schook for agent games (hooks)
# Sim mods our games can enable (mounted at /mods/<name> by our init file only, so the user's own FAF install is
# untouched). M28AI: maudlin27's AI, https://github.com/maudlin27/M28AI (CC BY-NC-SA 4.0), cloned to _deps.
KNOWN_MODS = {"M28AI": Path(os.environ.get("FA_ENV_M28_DIR") or AI_DIR.parents[1] / "_deps" / "M28AI")}

# Linux (a player machine, 2026-10-09): the engine is a 32-bit Windows program and runs under Wine on a virtual
# display (FA_ENV_DISPLAY, default :99, an Xvfb service). Every path the game reads -- the command line, the init
# file, its mounts -- is given as Wine's Z: drive path; the game's per-user folders are in the Wine prefix.
WINE = os.name != "nt"
# Microsoft's own DLLs, copied next to main.exe on a Linux machine (Wine's built-in ones fail): the VC++ 2015-22
# runtime (Wine's msvcp140 lacks _Throw_Cpp_error, which our build calls), D3DX 9.43 and the legacy compiler in
# D3DX 9.31 (the engine compiles its effects with D3DXSHADER_USE_LEGACY_D3DX9_31_DLL), D3DCompiler 43.
WINE_DLL_OVERRIDES = ("msvcp140,msvcp140_1,msvcp140_2,msvcp140_atomic_wait,vcruntime140,concrt140,"
                      "d3dx9_43,d3dx9_31,d3dcompiler_43=n,b")
WINE_PREFIX = Path(os.environ.get("WINEPREFIX") or Path.home() / ".wine")
_LOCAL_APPDATA = (os.environ.get("LOCALAPPDATA") if not WINE else
                  str(WINE_PREFIX / "drive_c" / "users" / os.environ.get("USER", "agent") / "AppData" / "Local"))
PREFS_DIR = Path(_LOCAL_APPDATA or "") / "Gas Powered Games" / "Supreme Commander Forged Alliance"
CACHE_ROOT = Path(_LOCAL_APPDATA or ".") / "fa_agent_cache"   # compiled effects, a folder per running game


def game_path(p) -> str:
    """A path as the game must see it: unchanged on Windows; under Wine, the Z: drive (Z: is /)."""
    p = Path(p)
    if not WINE:
        return str(p)
    return "Z:" + str(p.resolve()).replace("/", "\\")


def game_posix(p) -> str:
    """game_path with forward slashes, for Lua source (the init file)."""
    return Path(p).as_posix() if not WINE else game_path(p).replace("\\", "/")


class BridgeClosed(ConnectionError):
    """The game closed the connection (it exited, crashed, or quit)."""


class BridgeServer:
    """Listening socket plus the one connection a game instance makes to it."""

    def __init__(self, port: int = 0):
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(1)
        self.port: int = self._listener.getsockname()[1]
        self._conn: socket.socket | None = None
        self._reader = None
        self.bytes_received = 0
        self.last_line: bytes | None = None   # the last message as received (for raw recordings)

    def accept(self, timeout: float, alive=lambda: True) -> None:
        """Wait for the game to connect. `alive()` is polled so a game that dies early is noticed."""
        deadline = time.monotonic() + timeout
        self._listener.settimeout(1.0)
        while True:
            try:
                conn, _ = self._listener.accept()
                break
            except socket.timeout:
                if not alive():
                    raise BridgeClosed("the game exited before connecting")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"the game did not connect within {timeout:.0f} s")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(None)
        self._conn = conn
        self._reader = conn.makefile("rb")

    def recv(self, timeout: float | None = None) -> dict:
        if self._conn is None:
            raise BridgeClosed("not connected")
        self._conn.settimeout(timeout)
        try:
            line = self._reader.readline()
        except socket.timeout as exc:
            raise TimeoutError(f"no message from the game within {timeout} s") from exc
        except OSError as exc:
            raise BridgeClosed(str(exc)) from exc
        if not line:
            raise BridgeClosed("the game closed the connection")
        self.bytes_received += len(line)
        self.last_line = line
        return json.loads(line.decode("utf-8", errors="replace"))

    def send(self, message: dict) -> None:
        if self._conn is None:
            raise BridgeClosed("not connected")
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self._conn.sendall(data)
        except OSError as exc:
            raise BridgeClosed(str(exc)) from exc

    def close(self) -> None:
        for closable in (self._reader, self._conn, self._listener):
            try:
                if closable is not None:
                    closable.close()
            except OSError:
                pass
        self._reader = self._conn = None


@dataclass
class GameConfig:
    """How to launch one game instance.

    `map` is a map folder name (SCMP_016 is Canis River, a small 1v1). Army 1 is the agent's
    (a human slot nobody else commands); every other start position gets FAF's "rush" AI unless
    `no_ai` is set. `speed` is the bridge's pacing rate: 50 is unpaced, 10 the fastest normal game
    speed, 0 real time.
    """

    exe: Path = DEFAULT_BIN_DIR / "main.exe"
    bin_dir: Path = DEFAULT_BIN_DIR
    script: Path = DEFAULT_SCRIPT
    map: str = "SCMP_016"
    faction: int = 1                 # 1 UEF, 2 Aeon, 3 Cybran, 4 Seraphim
    ai_faction: int | None = None    # pin every AI's faction (FAF picks at random otherwise)
    ai: str | None = None            # AI personality key from /lua/ui/lobby/aitypes.lua (FAF default: 'rush')
    cheat: float | None = None       # AIx cheat and build multiplier (the cheating personalities; lobby default 1.5)
    # FAF's command-line launch defaults to 'sandbox' -- nobody can ever lose. 'demoralization' is the
    # standard rule: an army is defeated when its commander dies.
    victory: str | None = "demoralization"
    seed: int | None = None
    speed: int = 50
    window: tuple[int, int] = (800, 600)
    no_fog: bool = False
    no_ai: bool = False
    log_dir: Path | None = None      # game logs go here (default: a temp folder per run)
    extra_args: list[str] = field(default_factory=list)
    mods: list[str] = field(default_factory=list)   # sim mods by name (KNOWN_MODS) or folder; M28AI is added for m28 AIs
    slots: tuple[int, int] | None = None   # (our army slot, the AI's): a 1v1 on any map, from either side
    player_ai: str | None = None     # our own slot played by this AI (e.g. m28ai), observed: its play as teacher data
    connect_timeout: float = 180.0   # loading a map can take a while on a cold cache
    # Minimize the window once the game is running: a minimized game skips drawing entirely
    # (CScApp::Main does no device Refresh) and sleeps until the sim has new data -- headless in effect.
    # Until then (loading) the engine keeps the window off every screen and never activates it
    # (/agentbackground): training games must not cover, or take clicks and focus from, other programs.
    # Such a game is never drawn, so it also skips the loading work only the renderer uses (/agentlean:
    # strategic-icon textures, mesh texture searches; about 2.5 s of a 10.8 s load).
    minimized: bool = False
    lean: bool = True                # with minimized: /agentlean (False keeps the full loading, for comparisons)


def default_log_dir() -> Path:
    return Path(os.environ.get("TEMP", ".")) / "fa_env_logs"


def resolve_mods(config: "GameConfig") -> dict[str, Path]:
    """name -> folder of the sim mods a game needs: the configured ones, and M28AI for an M28 opponent."""
    names = list(config.mods)
    if any(k and k.lower().startswith("m28") for k in (config.ai, config.player_ai)) and "M28AI" not in names:
        names.append("M28AI")
    out = {}
    for name in names:
        folder = KNOWN_MODS.get(name) or Path(name)
        if not (folder / "mod_info.lua").exists():
            raise FileNotFoundError(f"sim mod {name!r}: no mod_info.lua in {folder}")
        out[folder.name] = folder
    return out


def write_init_file(path: Path, bin_dir: Path = DEFAULT_BIN_DIR, background: bool = False,
                    mods: dict[str, Path] | None = None) -> Path:
    """An init file that runs FAF's init.lua and mounts the overrides folder (its /schook hooks read
    the /agent* switches). init.lua resolves its paths from InitFileDir, so point that at FAF's bin
    folder first. Works with both our main.exe and FAF's ForgedAlliance.exe.

    FAF's init raises every game to high priority. `background` (training games) sets it back to below
    normal: a game then takes the CPU nobody else wants, never the user's own programs'."""
    bin_posix = game_posix(bin_dir)
    lower = ("local SetProcessPriority = rawget(_G, 'SetProcessPriority')\n"
             "if SetProcessPriority then SetProcessPriority(0x4000) end   -- BELOW_NORMAL_PRIORITY_CLASS\n"
             if background else "")
    path.write_text(
        "-- generated by fa_env: FAF's init.lua plus the agent overrides folder\n"
        f"InitFileDir = [[{bin_posix}]]\n"
        f"dofile([[{bin_posix}/init.lua]])\n"
        + lower +
        # mounted at /schook, not at /: a directory mounted at / is searched on disk for every file the game
        # opens (ZwQueryFullAttributesFile: ~17 % of blueprint loading, ~2 % of a late-game sim thread)
        f"table.insert(path, 1, {{ dir = [[{game_posix(OVERRIDES_DIR / 'schook')}]], mountpoint = '/schook' }})\n"
        # none of the user's own mods (FAF mounts both vault folders): a game sees only the mods it names, so
        # an old copy cannot shadow ours (M28AI v292 did) and every game is the same on any machine
        "for i = table.getn(path), 1, -1 do\n"
        "    local mount = type(path[i]) == 'table' and path[i].mountpoint\n"
        "    if type(mount) == 'string' and string.find(string.lower(mount), '^/mods/') then\n"
        "        table.remove(path, i)\n"
        "    end\n"
        "end\n"
        + "".join(f"table.insert(path, {{ dir = [[{game_posix(d)}]], mountpoint = '/mods/{name}' }})\n"
                  for name, d in (mods or {}).items()),
        encoding="utf-8",
    )
    return path


def minimize_windows(pid: int) -> int:
    """Minimize a process's visible top-level windows without activating anything (no focus change).
    Returns how many were minimized. Windows only; a no-op elsewhere."""
    if os.name != "nt":
        return 0
    import ctypes
    import ctypes.wintypes as wt
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def visit(hwnd, _):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd) and not user32.GetWindow(hwnd, 4):  # GW_OWNER
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    for hwnd in found:
        user32.ShowWindow(hwnd, 7)   # SW_SHOWMINNOACTIVE
    return len(found)


def unpin_threads(pid: int) -> int:
    """Let every thread of a game run on any CPU of its process mask. The engine's startup code has a call
    that pins the main thread to the lowest CPU of the mask (THREAD_SetAffinity in WIN_AppExecute, as in
    the original exe); parallel games would then all put their main threads on one CPU, and a main thread
    that falls behind draining the sim's sync data stalls its sim. Measured on 2026-09-29, no thread was
    pinned by the time the game says hello, so this is a safety net. Returns how many threads it widened."""
    if os.name != "nt":
        return 0
    import ctypes
    import ctypes.wintypes as wt

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ThreadID", wt.DWORD),
                    ("th32OwnerProcessID", wt.DWORD), ("tpBasePri", wt.LONG), ("tpDeltaPri", wt.LONG),
                    ("dwFlags", wt.DWORD)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = k32.OpenThread.restype = k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    k32.GetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k32.SetThreadAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    k32.SetThreadAffinityMask.restype = ctypes.c_size_t
    k32.Thread32First.argtypes = k32.Thread32Next.argtypes = [wt.HANDLE, ctypes.POINTER(THREADENTRY32)]
    k32.CloseHandle.argtypes = [wt.HANDLE]
    proc = k32.OpenProcess(0x0400, False, pid)                     # PROCESS_QUERY_INFORMATION
    if not proc:
        return 0
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    ok = k32.GetProcessAffinityMask(proc, ctypes.byref(mask), ctypes.byref(system))
    k32.CloseHandle(proc)
    if not ok or not mask.value:
        return 0
    snap = k32.CreateToolhelp32Snapshot(0x4, 0)                    # TH32CS_SNAPTHREAD
    if not snap or snap == ctypes.c_void_p(-1).value:
        return 0
    entry = THREADENTRY32()
    entry.dwSize = ctypes.sizeof(THREADENTRY32)
    widened = 0
    more = k32.Thread32First(snap, ctypes.byref(entry))
    while more:
        if entry.th32OwnerProcessID == pid:
            thread = k32.OpenThread(0x0060, False, entry.th32ThreadID)   # SET_ and QUERY_INFORMATION
            if thread:
                before = k32.SetThreadAffinityMask(thread, mask.value)
                widened += bool(before) and before != mask.value
                k32.CloseHandle(thread)
        more = k32.Thread32Next(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return widened


def claim_cache_slot():
    """A private folder for a game's compiled effects (`/cachedir`), kept for the next game in it. FAF's
    init deletes the shared default cache on every launch, so every game recompiled every effect (seconds),
    and games loading at the same time could read each other's half-written files. A locked file marks a
    folder in use; the lock goes with its handle, even if this process dies. Returns (folder, lock handle)."""
    if os.name != "nt":
        return None, None
    import msvcrt
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None, None
    for slot in range(256):
        try:
            handle = open(CACHE_ROOT / f"slot{slot}.lock", "a+b")
        except OSError:
            continue
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            continue
        return CACHE_ROOT / f"slot{slot}", handle
    return None, None


def make_private_prefs(run_id: str, prefix: str = "agent") -> tuple[str, Path]:
    """Copy of the user's Game.prefs under a run-specific name, so parallel runs never collide.
    Returns (name for /prefs, path to delete afterwards)."""
    name = f"{prefix}_{run_id}.prefs"
    base = PREFS_DIR / "Game.prefs"
    if base.exists():
        shutil.copyfile(base, PREFS_DIR / name)
    return name, PREFS_DIR / name


class GameProcess:
    """One running game, launched with a private prefs file so parallel runs never collide."""

    def __init__(self, config: GameConfig, port: int):
        self.config = config
        self.run_id = uuid.uuid4().hex[:10]
        log_dir = Path(config.log_dir) if config.log_dir else default_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"game_{self.run_id}.log"
        self.prefs_name, self._prefs_path = make_private_prefs(self.run_id)
        self.mods = resolve_mods(config)
        self.init_path = write_init_file(log_dir / f"init_{self.run_id}.lua", Path(config.bin_dir),
                                         background=config.minimized, mods=self.mods)
        self.cache_dir, self._cache_lock = claim_cache_slot()
        self.loaded = False             # set by the environment once the game said hello

        args = [
            game_path(config.exe),
            "/init", game_path(self.init_path),
            "/nobugreport", "/nosound", "/nomovie",
            "/log", game_path(self.log_path),
            "/windowed", str(config.window[0]), str(config.window[1]),
            "/prefs", self.prefs_name,
            "/map", config.map,
            "/faction", str(config.faction),
            "/exitongameover",
            "/agentbridge", str(port),
            "/agentscript", game_path(config.script),
            "/agentspeed", str(config.speed),
        ]
        if config.seed is not None:
            args += ["/seed", str(config.seed)]
        if config.victory:
            args += ["/victory", config.victory]
        if config.ai_faction is not None:
            args += ["/agentaifaction", str(config.ai_faction)]
        if config.ai:
            args += ["/agentai", config.ai]
        if config.cheat is not None:
            args += ["/agentcheat", str(config.cheat)]
        if self.mods:
            args += ["/agentmods", ",".join(self.mods)]
        if config.slots:
            args += ["/agentslots", f"{config.slots[0]},{config.slots[1]}"]
        if config.player_ai:
            args += ["/agentplayerai", config.player_ai]
        if config.no_fog:
            args.append("/nofog")
        if config.no_ai:
            args.append("/noAi")
        if self.cache_dir is not None:
            args += ["/cachedir", game_path(self.cache_dir)]
        if config.minimized:
            args += ["/agentbackground"] + (["/agentlean"] if config.lean else [])
            if WINE:                          # nothing minimizes the window under Wine: never draw (engine switch)
                args.append("/agentheadless")
        args += list(config.extra_args)
        self.args = args
        # training games run below normal priority: they take the idle CPU, never the user's own programs'
        if WINE:
            env = dict(os.environ, WINEDEBUG=os.environ.get("WINEDEBUG", "-all"),
                       DISPLAY=os.environ.get("FA_ENV_DISPLAY", os.environ.get("DISPLAY", ":99")),
                       WINEDLLOVERRIDES=os.environ.get("WINEDLLOVERRIDES", WINE_DLL_OVERRIDES))
            self.process = subprocess.Popen(["wine"] + args, cwd=str(config.bin_dir), env=env,
                                            preexec_fn=(lambda: os.nice(10)) if config.minimized else None,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            priority = subprocess.BELOW_NORMAL_PRIORITY_CLASS if config.minimized else 0
            self.process = subprocess.Popen(args, cwd=str(config.bin_dir), creationflags=priority)

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self, grace: float = 5.0) -> None:
        if self.alive():
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for leftover in (self._prefs_path, self.init_path):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass
        if self._cache_lock is not None:
            if not self.loaded:         # it died loading: it may have left a half-written effect behind
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            self._cache_lock.close()
            self._cache_lock = None
