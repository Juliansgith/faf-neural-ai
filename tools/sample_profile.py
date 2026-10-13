"""Sampling profiler for the rebuilt engine's sim thread, driven through the agent bridge.

usage: sample_profile.py [--map SCMP_016] [--noai] [--seconds 15] [--warmup-steps 20] [--exe main.exe]

Starts a game through fa_env, keeps stepping it (no actions) on a background thread, finds the busiest
thread in the process (the sim thread while it runs flat out), and samples it: EIP for the self profile,
plus a scan of its stack for return addresses (validated as following a call instruction) for an
approximate inclusive profile. Symbols come from main.pdb via dbghelp.
"""
import argparse
import collections
import ctypes
import ctypes.wintypes as wt
import os
import struct
import sys
import threading
import time
from pathlib import Path

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai" / "python"))
from fa_env import FAEnv, GameConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--map", default="SCMP_016")
ap.add_argument("--noai", action="store_true")
ap.add_argument("--seconds", type=float, default=15.0)
ap.add_argument("--warmup-steps", type=int, default=20)
ap.add_argument("--exe", default=r"C:\ProgramData\FAForever\bin\main.exe")
ap.add_argument("--top", type=int, default=45)
ap.add_argument("--pid", type=int, default=0, help="attach to a running game instead of starting one")
args = ap.parse_args()

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
winmm = ctypes.WinDLL("winmm")
dbg = ctypes.WinDLL("dbghelp.dll", use_last_error=True)

TH32CS_SNAPTHREAD = 0x4
THREAD_ALL = 0x0002 | 0x0008 | 0x0040 | 0x0010  # suspend/resume, get context, query info, set context-ish
PROCESS_ALL = 0x0010 | 0x0400  # VM_READ | QUERY_INFORMATION


class THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ThreadID", wt.DWORD),
                ("th32OwnerProcessID", wt.DWORD), ("tpBasePri", wt.LONG), ("tpDeltaPri", wt.LONG), ("dwFlags", wt.DWORD)]


class WOW64_CONTEXT(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_ubyte * 716)]


k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.OpenThread.restype = wt.HANDLE
k32.OpenProcess.restype = wt.HANDLE
k32.Wow64GetThreadContext.argtypes = [wt.HANDLE, ctypes.POINTER(WOW64_CONTEXT)]
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.GetThreadTimes.argtypes = [wt.HANDLE] + [ctypes.POINTER(wt.FILETIME)] * 4


def threads_of(pid):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    entry = THREADENTRY32()
    entry.dwSize = ctypes.sizeof(entry)
    tids = []
    ok = k32.Thread32First(snap, ctypes.byref(entry))
    while ok:
        if entry.th32OwnerProcessID == pid:
            tids.append(entry.th32ThreadID)
        ok = k32.Thread32Next(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return tids


def cpu_time(handle):
    c, e, kt, ut = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
    k32.GetThreadTimes(handle, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut))
    return ((kt.dwHighDateTime << 32 | kt.dwLowDateTime) + (ut.dwHighDateTime << 32 | ut.dwLowDateTime)) / 1e7


def module_base(hproc):
    mods = (wt.HMODULE * 1024)()
    needed = wt.DWORD()
    psapi.EnumProcessModulesEx(hproc, mods, ctypes.sizeof(mods), ctypes.byref(needed), 0x01)
    return mods[0]


# ---------------------------------------------------------------- game + stepping thread
stop = threading.Event()
steps = [0]
if args.pid:
    env = None
    obs = {"tick": 0}
    pid = args.pid
else:
    from fa_env.agents import ScriptedAgent
    env = FAEnv(GameConfig(map=args.map, seed=7, no_ai=args.noai, exe=args.exe, ai_faction=2), step_ticks=10)
    obs, info = env.reset()
    agent = ScriptedAgent()
    agent.reset(obs, info)
    state = {"obs": obs, "info": None}

    def one_step():
        acts, qs = agent.act(state["obs"], state["info"])
        state["obs"], _, _, _, state["info"] = env.step(acts, qs)
        return state["obs"]

    for _ in range(args.warmup_steps):
        one_step()
    obs = state["obs"]

    def stepper():
        while not stop.is_set():
            o = one_step()
            steps[0] += 1
            steps.append(o.get("tick"))

    t = threading.Thread(target=stepper, daemon=True)
    t.start()
    pid = env.game.process.pid
hproc = k32.OpenProcess(PROCESS_ALL, False, pid)
base = module_base(hproc)
pe = pefile.PE(args.exe, fast_load=True)
image_base = pe.OPTIONAL_HEADER.ImageBase
text = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".text")
text_lo, text_hi = base + text.VirtualAddress, base + text.VirtualAddress + text.Misc_VirtualSize
code = pe.get_memory_mapped_image()


def follows_call(ret):
    rva = ret - base
    if rva < 7 or rva >= len(code):
        return False
    b = code[rva - 7:rva]
    return b[2] == 0xE8 or (b[5] == 0xFF and (b[6] & 0x38) == 0x10) or (b[4] == 0xFF and (b[5] & 0x38) == 0x10) \
        or (b[1] == 0xFF and (b[2] & 0x38) == 0x10) or (b[0] == 0xFF and (b[1] & 0x38) == 0x10)


# busiest thread = sim thread
handles = {tid: k32.OpenThread(THREAD_ALL, False, tid) for tid in threads_of(pid)}
before = {tid: cpu_time(h) for tid, h in handles.items()}
time.sleep(2.0)
after = {tid: cpu_time(h) for tid, h in handles.items()}
busy = sorted(handles, key=lambda tid: after[tid] - before[tid], reverse=True)
print("busiest threads (cpu s over 2 s):", [(tid, round(after[tid] - before[tid], 2)) for tid in busy[:5]])
target = handles[busy[0]]

# ---------------------------------------------------------------- sample
winmm.timeBeginPeriod(1)
leaf = collections.Counter()
inclusive = collections.Counter()
samples = 0
ctx = WOW64_CONTEXT()
stack = ctypes.create_string_buffer(32768)
t_end = time.monotonic() + args.seconds
tick0 = steps[-1] if len(steps) > 1 else obs.get("tick")
while time.monotonic() < t_end:
    if k32.Wow64SuspendThread(target) == 0xFFFFFFFF:
        break
    struct.pack_into("<I", ctx.raw, 0, 0x00010001 | 0x00010002)
    if k32.Wow64GetThreadContext(target, ctypes.byref(ctx)):
        eip, esp = struct.unpack_from("<I", ctx.raw, 184)[0], struct.unpack_from("<I", ctx.raw, 196)[0]
        got = ctypes.c_size_t()
        k32.ReadProcessMemory(hproc, esp, stack, len(stack), ctypes.byref(got))
        k32.ResumeThread(target)
        samples += 1
        leaf[eip] += 1
        seen = set()
        n = got.value // 4
        words = struct.unpack_from(f"<{n}I", stack.raw, 0)
        chain = [eip]
        for w in words:
            if text_lo <= w < text_hi and follows_call(w):
                chain.append(w)
                if len(chain) > 60:
                    break
        inclusive_key = tuple(chain)
        inclusive[inclusive_key] += 1
    else:
        k32.ResumeThread(target)
    time.sleep(0.001)
winmm.timeEndPeriod(1)
stop.set()
tick1 = steps[-1]

# ---------------------------------------------------------------- symbolize
SYMOPT_UNDNAME = 0x2
dbg.SymSetOptions(SYMOPT_UNDNAME | 0x40)
sproc = wt.HANDLE(0x4321)
dbg.SymInitializeW(sproc, ctypes.c_wchar_p(os.path.dirname(args.exe)), False)
dbg.SymLoadModuleExW.restype = ctypes.c_uint64
dbg.SymLoadModuleExW.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint64, wt.DWORD,
                                 ctypes.c_void_p, wt.DWORD]
dbg.SymLoadModuleExW(sproc, None, args.exe, None, base, 0, None, 0)


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.ULONG), ("TypeIndex", wt.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                ("Index", wt.ULONG), ("Size", wt.ULONG), ("ModBase", ctypes.c_uint64), ("Flags", wt.ULONG),
                ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64), ("Register", wt.ULONG),
                ("Scope", wt.ULONG), ("Tag", wt.ULONG), ("NameLen", wt.ULONG), ("MaxNameLen", wt.ULONG),
                ("Name", ctypes.c_wchar * 512)]


dbg.SymFromAddrW.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFOW)]
cache = {}


class MODULEINFO(ctypes.Structure):
    _fields_ = [("lpBaseOfDll", ctypes.c_void_p), ("SizeOfImage", wt.DWORD), ("EntryPoint", ctypes.c_void_p)]


def other_modules():
    """(base, size, path, sorted exports) for every 32-bit module except main.exe."""
    mods = (ctypes.c_void_p * 1024)()
    needed = wt.DWORD()
    psapi.EnumProcessModulesEx(hproc, mods, ctypes.sizeof(mods), ctypes.byref(needed), 0x01)
    result = []
    name = ctypes.create_unicode_buffer(1024)
    for i in range(needed.value // ctypes.sizeof(ctypes.c_void_p)):
        if not mods[i] or mods[i] == base:
            continue
        info_ = MODULEINFO()
        psapi.GetModuleInformation(hproc, ctypes.c_void_p(mods[i]), ctypes.byref(info_), ctypes.sizeof(info_))
        psapi.GetModuleFileNameExW(hproc, ctypes.c_void_p(mods[i]), name, 1024)
        result.append([mods[i], info_.SizeOfImage, name.value, None])
    return result


try:
    modules = other_modules()
except Exception:  # noqa: BLE001
    modules = []


def export_name(addr):
    for m in modules:
        mbase, size, path, exports = m
        if mbase <= addr < mbase + size:
            if exports is None:
                path32 = path.replace("\\System32\\", "\\SysWOW64\\") if "\\System32\\" in path else path
                try:
                    p = pefile.PE(path32, fast_load=True)
                    p.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
                    exports = sorted((e.address, (e.name or b"?").decode()) for e in p.DIRECTORY_ENTRY_EXPORT.symbols)
                except Exception:  # noqa: BLE001
                    exports = []
                m[3] = exports
            rva = addr - mbase
            best = None
            for e_rva, e_name in exports:
                if e_rva <= rva:
                    best = (e_rva, e_name)
                else:
                    break
            mod_name = os.path.basename(path)
            return f"{mod_name}!{best[1]}+0x{rva - best[0]:X}" if best else f"{mod_name}+0x{rva:X}"
    return None


def name_of(addr):
    if addr in cache:
        return cache[addr]
    if not (text_lo <= addr < text_hi):
        other = export_name(addr)
        if other:
            cache[addr] = other
            return other
    sym = SYMBOL_INFOW()
    sym.SizeOfStruct = ctypes.sizeof(SYMBOL_INFOW) - 512 * 2
    sym.MaxNameLen = 512
    disp = ctypes.c_uint64()
    if text_lo <= addr < text_hi and dbg.SymFromAddrW(sproc, addr, ctypes.byref(disp), ctypes.byref(sym)):
        name = sym.Name
    else:
        name = f"<0x{addr:08X} outside main.exe>" if not (text_lo <= addr < text_hi) else f"<0x{addr:08X}>"
    cache[addr] = name
    return name


self_by_fn = collections.Counter()
for eip, c in leaf.items():
    self_by_fn[name_of(eip)] += c
incl_by_fn = collections.Counter()
for chain, c in inclusive.items():
    for fn in {name_of(a) for a in chain}:
        incl_by_fn[fn] += c

ticks = (tick1 or 0) - (tick0 or 0)
print(f"\nsamples {samples} over {args.seconds:.0f} s; sim advanced {ticks} ticks "
      f"({ticks / args.seconds:.0f} ticks/s) with {'no AI' if args.noai else 'AI'} on {args.map}")
print(f"\n=== self (top {args.top}) ===")
for fn, c in self_by_fn.most_common(args.top):
    print(f"{100.0 * c / samples:6.2f}%  {fn[:150]}")
by_module = collections.Counter()
for fn, c in self_by_fn.items():
    by_module[fn.split("!")[0] if "!" in fn else ("main.exe" if not fn.startswith("<") else "?")] += c
print("\n=== self by module ===")
for mod, c in by_module.most_common(8):
    print(f"{100.0 * c / samples:6.2f}%  {mod}")
print(f"\n=== inclusive, approximate (top {args.top}) ===")
for fn, c in incl_by_fn.most_common(args.top):
    print(f"{100.0 * c / samples:6.2f}%  {fn[:150]}")
if env is not None:
    env.close()
