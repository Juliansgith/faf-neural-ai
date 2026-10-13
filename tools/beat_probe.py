"""Run a game under a minimal debugger and fingerprint the sim's random-number state at every beat.

    beat_probe.py --exe ForgedAlliance.exe --out probe_official.jsonl --beats 80 -- <game args...>
    beat_probe.py --exe main.exe --out probe_ours.jsonl --beats 80 --watch 40:52 -- <game args...>

Works on FAF's ForgedAlliance.exe (breakpoint at the binary's Sim::AdvanceBeat, 0x00749F40) and on our
main.exe (Sim::AdvanceBeat looked up in main.pdb), so two runs of the same replay can be compared beat
by beat. At each AdvanceBeat entry it reads Sim::mCurBeat (+0x8F8), mCurTick (+0x900) and the
CRandomStream at Sim::mRngState (+0x904): the Mersenne-Twister index (+0x9C0), the cached Gaussian
(+0x9C4/+0x9C8) and an MD5 of the 624 state words. The sequence of draws is fully determined by those,
so the first beat whose fingerprint differs is the first beat in which the two engines drew a different
number of random numbers.

--watch A:B also puts a hardware write-watchpoint on the twister index during beats A..B and records the
code address of every draw (inlined draws included: the binary inlines the twister in several places).
With --pdb the addresses are symbolized (our main.exe); for the official binary they stay addresses.

Needs no debugger install: the Win32 debug API through ctypes. Launch with the sandbox disabled.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

k32 = ctypes.WinDLL("kernel32", use_last_error=True)

DEBUG_ONLY_THIS_PROCESS = 0x2
INFINITE = 0xFFFFFFFF
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001
EXCEPTION_BREAKPOINT = 0x80000003
EXCEPTION_SINGLE_STEP = 0x80000004
STATUS_WX86_BREAKPOINT = 0x4000001F
STATUS_WX86_SINGLE_STEP = 0x4000001E
WOW64_CONTEXT_i386 = 0x00010000
WOW64_CONTEXT_CONTROL = WOW64_CONTEXT_i386 | 0x1
WOW64_CONTEXT_INTEGER = WOW64_CONTEXT_i386 | 0x2
WOW64_CONTEXT_DEBUG_REGISTERS = WOW64_CONTEXT_i386 | 0x10

ON_STARTED = None  # optional callback(pid) once the game process exists (e.g. to raise the game speed)

OFFICIAL_ADVANCE_BEAT = 0x00749F40
SIM_CUR_BEAT, SIM_CUR_TICK, SIM_RNG = 0x8F8, 0x900, 0x904
RNG_STATE_BYTES, RNG_INDEX, RNG_PAIR, RNG_HAS_PAIR = 0x9C0, 0x9C0, 0x9C4, 0x9C8


class EXCEPTION_RECORD(ctypes.Structure):
    _fields_ = [("ExceptionCode", wt.DWORD), ("ExceptionFlags", wt.DWORD), ("ExceptionRecord", ctypes.c_void_p),
                ("ExceptionAddress", ctypes.c_void_p), ("NumberParameters", wt.DWORD),
                ("ExceptionInformation", ctypes.c_uint64 * 15)]


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("ExceptionRecord", EXCEPTION_RECORD), ("dwFirstChance", wt.DWORD)]


class CREATE_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("hThread", wt.HANDLE), ("lpThreadLocalBase", ctypes.c_void_p), ("lpStartAddress", ctypes.c_void_p)]


class CREATE_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("hFile", wt.HANDLE), ("hProcess", wt.HANDLE), ("hThread", wt.HANDLE),
                ("lpBaseOfImage", ctypes.c_void_p), ("dwDebugInfoFileOffset", wt.DWORD),
                ("nDebugInfoSize", wt.DWORD), ("lpThreadLocalBase", ctypes.c_void_p),
                ("lpStartAddress", ctypes.c_void_p), ("lpImageName", ctypes.c_void_p), ("fUnicode", wt.WORD)]


class LOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("hFile", wt.HANDLE), ("lpBaseOfDll", ctypes.c_void_p), ("dwDebugInfoFileOffset", wt.DWORD),
                ("nDebugInfoSize", wt.DWORD), ("lpImageName", ctypes.c_void_p), ("fUnicode", wt.WORD)]


class _U(ctypes.Union):
    _fields_ = [("Exception", EXCEPTION_DEBUG_INFO), ("CreateThread", CREATE_THREAD_DEBUG_INFO),
                ("CreateProcessInfo", CREATE_PROCESS_DEBUG_INFO), ("LoadDll", LOAD_DLL_DEBUG_INFO),
                ("ExitCode", wt.DWORD)]


class DEBUG_EVENT(ctypes.Structure):
    _fields_ = [("dwDebugEventCode", wt.DWORD), ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD), ("u", _U)]


class WOW64_FLOATING_SAVE_AREA(ctypes.Structure):
    _fields_ = [("ControlWord", wt.DWORD), ("StatusWord", wt.DWORD), ("TagWord", wt.DWORD),
                ("ErrorOffset", wt.DWORD), ("ErrorSelector", wt.DWORD), ("DataOffset", wt.DWORD),
                ("DataSelector", wt.DWORD), ("RegisterArea", ctypes.c_byte * 80), ("Cr0NpxState", wt.DWORD)]


class WOW64_CONTEXT(ctypes.Structure):
    _fields_ = [("ContextFlags", wt.DWORD), ("Dr0", wt.DWORD), ("Dr1", wt.DWORD), ("Dr2", wt.DWORD),
                ("Dr3", wt.DWORD), ("Dr6", wt.DWORD), ("Dr7", wt.DWORD), ("FloatSave", WOW64_FLOATING_SAVE_AREA),
                ("SegGs", wt.DWORD), ("SegFs", wt.DWORD), ("SegEs", wt.DWORD), ("SegDs", wt.DWORD),
                ("Edi", wt.DWORD), ("Esi", wt.DWORD), ("Ebx", wt.DWORD), ("Edx", wt.DWORD), ("Ecx", wt.DWORD),
                ("Eax", wt.DWORD), ("Ebp", wt.DWORD), ("Eip", wt.DWORD), ("SegCs", wt.DWORD),
                ("EFlags", wt.DWORD), ("Esp", wt.DWORD), ("SegSs", wt.DWORD),
                ("ExtendedRegisters", ctypes.c_byte * 512)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
                ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
                ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
                ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", wt.HANDLE), ("hStdOutput", wt.HANDLE),
                ("hStdError", wt.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE), ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD)]


k32.WaitForDebugEvent.argtypes = [ctypes.POINTER(DEBUG_EVENT), wt.DWORD]
k32.ContinueDebugEvent.argtypes = [wt.DWORD, wt.DWORD, wt.DWORD]
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_size_t)]
k32.WriteProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_size_t)]
k32.FlushInstructionCache.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t]
k32.Wow64GetThreadContext.argtypes = [wt.HANDLE, ctypes.POINTER(WOW64_CONTEXT)]
k32.Wow64SetThreadContext.argtypes = [wt.HANDLE, ctypes.POINTER(WOW64_CONTEXT)]
k32.CloseHandle.argtypes = [wt.HANDLE]
k32.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
k32.CreateProcessW.argtypes = [wt.LPCWSTR, wt.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wt.BOOL, wt.DWORD,
                               ctypes.c_void_p, wt.LPCWSTR, ctypes.POINTER(STARTUPINFOW),
                               ctypes.POINTER(PROCESS_INFORMATION)]


def pdb_symbol_rva(exe: Path, name: str) -> int:
    """RVA of `name` in exe's PDB (dbghelp, offline)."""
    dbg = ctypes.WinDLL("dbghelp.dll", use_last_error=True)
    dbg.SymSetOptions(0x2 | 0x40)
    proc = wt.HANDLE(0x4242)
    dbg.SymInitializeW(proc, ctypes.c_wchar_p(str(exe.parent)), False)
    dbg.SymLoadModuleExW.restype = ctypes.c_uint64
    dbg.SymLoadModuleExW.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint64,
                                     wt.DWORD, ctypes.c_void_p, wt.DWORD]
    base = 0x10000000
    if not dbg.SymLoadModuleExW(proc, None, str(exe), None, base, 0, None, 0):
        raise SystemExit(f"SymLoadModuleEx failed: {ctypes.get_last_error()}")

    class SYMBOL_INFOW(ctypes.Structure):
        _fields_ = [("SizeOfStruct", wt.ULONG), ("TypeIndex", wt.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                    ("Index", wt.ULONG), ("Size", wt.ULONG), ("ModBase", ctypes.c_uint64), ("Flags", wt.ULONG),
                    ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64), ("Register", wt.ULONG),
                    ("Scope", wt.ULONG), ("Tag", wt.ULONG), ("NameLen", wt.ULONG), ("MaxNameLen", wt.ULONG),
                    ("Name", ctypes.c_wchar * 1024)]

    info = SYMBOL_INFOW()
    info.SizeOfStruct = ctypes.sizeof(SYMBOL_INFOW) - ctypes.sizeof(ctypes.c_wchar) * 1024  # C sizeof: 88
    info.MaxNameLen = 1024
    if not dbg.SymFromNameW(proc, ctypes.c_wchar_p(name), ctypes.byref(info)):
        raise SystemExit(f"symbol {name} not found ({ctypes.get_last_error()})")
    rva = info.Address - base
    dbg.SymCleanup(proc)
    return rva


class Symbolizer:
    """Address -> 'function+off' for our main.exe via its PDB, at the runtime image base."""

    def __init__(self, exe: Path, base: int):
        self.dbg = ctypes.WinDLL("dbghelp.dll", use_last_error=True)
        self.dbg.SymSetOptions(0x2 | 0x40)
        self.proc = wt.HANDLE(0x4343)
        self.dbg.SymInitializeW(self.proc, ctypes.c_wchar_p(str(exe.parent)), False)
        self.dbg.SymLoadModuleExW.restype = ctypes.c_uint64
        self.dbg.SymLoadModuleExW.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.c_wchar_p, ctypes.c_wchar_p,
                                              ctypes.c_uint64, wt.DWORD, ctypes.c_void_p, wt.DWORD]
        self.dbg.SymLoadModuleExW(self.proc, None, str(exe), None, base, 0, None, 0)

        class SYMBOL_INFOW(ctypes.Structure):
            _fields_ = [("SizeOfStruct", wt.ULONG), ("TypeIndex", wt.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                        ("Index", wt.ULONG), ("Size", wt.ULONG), ("ModBase", ctypes.c_uint64), ("Flags", wt.ULONG),
                        ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64), ("Register", wt.ULONG),
                        ("Scope", wt.ULONG), ("Tag", wt.ULONG), ("NameLen", wt.ULONG), ("MaxNameLen", wt.ULONG),
                        ("Name", ctypes.c_wchar * 1024)]

        self.SYMBOL_INFOW = SYMBOL_INFOW
        self.cache: dict[int, str] = {}

    def __call__(self, addr: int) -> str:
        if addr in self.cache:
            return self.cache[addr]
        info = self.SYMBOL_INFOW()
        info.SizeOfStruct = ctypes.sizeof(self.SYMBOL_INFOW) - ctypes.sizeof(ctypes.c_wchar) * 1024
        info.MaxNameLen = 1024
        disp = ctypes.c_uint64()
        if self.dbg.SymFromAddrW(self.proc, ctypes.c_uint64(addr), ctypes.byref(disp), ctypes.byref(info)):
            s = f"{info.Name}+{disp.value:#x}"
        else:
            s = f"{addr:#010x}"
        self.cache[addr] = s
        return s


class Probe:
    def __init__(self, args):
        self.args = args
        self.exe = Path(args.exe).resolve()
        self.official = self.exe.name.lower() == "forgedalliance.exe"
        self.hproc = None
        self.threads: dict[int, int] = {}
        self.bp_addr = None
        self.bp_orig = None
        self.pending_rearm: dict[int, int] = {}
        self.probes: dict[int, dict] = {}  # addr -> {"name", "orig"}
        self.bp_lo, self.bp_hi = (args.bp_window or (0, 1 << 30))
        self.hits = []
        self.current_beat = -1
        self.records = []
        self.watch_lo, self.watch_hi = (args.watch or (None, None))
        self.watching_tid = None
        self.watch_addr = None
        self.draws = []
        self.symbolize = None
        self.out = open(args.out, "w", encoding="utf-8")

    # -- memory
    def read(self, addr: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        n = ctypes.c_size_t()
        if not k32.ReadProcessMemory(self.hproc, ctypes.c_void_p(addr), buf, size, ctypes.byref(n)) or n.value != size:
            raise OSError(f"ReadProcessMemory({addr:#x},{size}) failed: {ctypes.get_last_error()}")
        return buf.raw

    def write(self, addr: int, data: bytes) -> None:
        n = ctypes.c_size_t()
        if not k32.WriteProcessMemory(self.hproc, ctypes.c_void_p(addr), data, len(data), ctypes.byref(n)):
            raise OSError(f"WriteProcessMemory({addr:#x}) failed: {ctypes.get_last_error()}")
        k32.FlushInstructionCache(self.hproc, ctypes.c_void_p(addr), len(data))

    def u32(self, addr: int) -> int:
        return struct.unpack("<I", self.read(addr, 4))[0]

    def code_before(self, addr: int, n: int = 7) -> bytes:
        out = bytearray()
        for a in range(addr - n, addr):
            page = a & ~0xFFF
            if page not in self.code_cache:
                try:
                    self.code_cache[page] = self.read(page, 0x1000)
                except OSError:
                    self.code_cache[page] = b""
            blob = self.code_cache[page]
            out.append(blob[a & 0xFFF] if blob else 0)
        return bytes(out)

    def looks_like_return(self, value: int) -> bool:
        """A stack word that points into the image just after a call instruction."""
        if not (self.image_lo <= value < self.image_hi):
            return False
        b = self.code_before(value, 7)
        return (b[2] == 0xE8                                  # call rel32
                or (b[1] == 0xFF and (b[2] & 0x38) == 0x10)   # call [disp32] / [reg+disp32]
                or (b[4] == 0xFF and (b[5] & 0x38) == 0x10)   # call [reg+disp8]
                or (b[5] == 0xFF and (b[6] & 0x38) == 0x10))  # call reg / call [reg]

    def return_chain(self, esp: int, depth: int = 4, scan: int = 256) -> list[int]:
        try:
            words = struct.unpack(f"<{scan}I", self.read(esp, scan * 4))
        except OSError:
            return []
        chain = []
        for w in words:
            if self.looks_like_return(w):
                chain.append(w)
                if len(chain) >= depth:
                    break
        return chain

    # -- thread context
    def get_ctx(self, tid: int, flags=WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER) -> WOW64_CONTEXT:
        ctx = WOW64_CONTEXT()
        ctx.ContextFlags = flags
        if not k32.Wow64GetThreadContext(self.threads[tid], ctypes.byref(ctx)):
            raise OSError(f"Wow64GetThreadContext failed: {ctypes.get_last_error()}")
        return ctx

    def set_ctx(self, tid: int, ctx: WOW64_CONTEXT) -> None:
        if not k32.Wow64SetThreadContext(self.threads[tid], ctypes.byref(ctx)):
            raise OSError(f"Wow64SetThreadContext failed: {ctypes.get_last_error()}")

    # -- events
    def on_create_process(self, info: CREATE_PROCESS_DEBUG_INFO, tid: int) -> None:
        self.hproc = info.hProcess
        self.threads[tid] = info.hThread
        base = info.lpBaseOfImage or 0
        if info.hFile:
            k32.CloseHandle(info.hFile)
        if self.official:
            self.bp_addr = OFFICIAL_ADVANCE_BEAT
        else:
            self.bp_addr = base + pdb_symbol_rva(self.exe, "moho::Sim::AdvanceBeat")
            self.symbolize = Symbolizer(self.exe, base)
        self.image_lo = base
        pe = base + self.u32(base + 0x3C)
        self.image_hi = base + self.u32(pe + 0x18 + 0x38)  # OptionalHeader.SizeOfImage
        self.code_cache: dict[int, bytes] = {}
        self.bp_orig = self.read(self.bp_addr, 1)
        self.write(self.bp_addr, b"\xCC")
        for spec in self.args.bp or []:
            addr = int(spec, 16) if spec.lower().startswith("0x") else base + pdb_symbol_rva(self.exe, spec)
            self.probes[addr] = {"name": spec, "orig": self.read(addr, 1)}
            self.write(addr, b"\xCC")
            print(f"probe breakpoint {spec} at {addr:#x}", flush=True)
        print(f"image base {base:#x}, AdvanceBeat breakpoint at {self.bp_addr:#x}", flush=True)

    def fingerprint(self, sim: int) -> dict:
        beat = self.u32(sim + SIM_CUR_BEAT)
        tick = self.u32(sim + SIM_CUR_TICK)
        rng = self.u32(sim + SIM_RNG)
        blob = self.read(rng, 0x9CC)
        idx, = struct.unpack_from("<I", blob, RNG_INDEX)
        pair, = struct.unpack_from("<I", blob, RNG_PAIR)
        return {"beat": beat, "tick": tick, "rng_index": idx, "has_pair": blob[RNG_HAS_PAIR],
                "pair_bits": f"{pair:08x}", "state_md5": hashlib.md5(blob[:RNG_STATE_BYTES]).hexdigest()[:16],
                "_rng": rng}

    def on_advance_beat(self, tid: int, ctx: WOW64_CONTEXT) -> None:
        sim = ctx.Ecx
        rec = self.fingerprint(sim)
        rng = rec.pop("_rng")
        if self.draws:
            rec["draws"] = self.draws
            self.draws = []
        if self.hits:
            rec["hits"] = self.hits
            self.hits = []
        self.records.append(rec)
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()
        beat = rec["beat"]
        self.current_beat = beat
        # hardware write-watch on the twister index while beat in [lo, hi]
        if self.watch_lo is not None:
            want = self.watch_lo <= beat <= self.watch_hi
            dctx = self.get_ctx(tid, WOW64_CONTEXT_DEBUG_REGISTERS)
            if want and not (dctx.Dr7 & 1):
                self.watch_addr = rng + RNG_INDEX
                dctx.Dr0 = self.watch_addr
                dctx.Dr7 = (dctx.Dr7 & ~0x000F0003) | 0x1 | (0b01 << 16) | (0b11 << 18)  # L0, write, 4 bytes
                self.set_ctx(tid, dctx)
                self.watching_tid = tid
            elif not want and (dctx.Dr7 & 1):
                dctx.Dr7 &= ~0x000F0003
                self.set_ctx(tid, dctx)
                self.watching_tid = None

    def on_probe(self, addr: int, ctx: WOW64_CONTEXT) -> None:
        if not (self.bp_lo <= self.current_beat <= self.bp_hi):
            return
        name = self.symbolize or (lambda a: f"{a:#010x}")
        try:
            words = struct.unpack("<16I", self.read(ctx.Esp, 64))
        except OSError:
            words = ()
        floats = [struct.unpack("<f", struct.pack("<I", w))[0] for w in words]
        deref = {}
        for i, w in enumerate(words[:6]):  # pointer-looking arguments: dump what they point at
            if w > 0x10000:
                try:
                    blob = self.read(w, 128)
                except OSError:
                    continue
                vals = struct.unpack("<32I", blob)
                deref[f"[esp+{4 * i:#x}]"] = [
                    round(struct.unpack("<f", struct.pack("<I", v))[0], 7) if 1e-30 < abs(struct.unpack("<f", struct.pack("<I", v))[0]) < 1e7 or v == 0
                    else f"{v:08x}" for v in vals]
        self.hits.append({
            "bp": self.probes[addr]["name"], "beat": self.current_beat,
            "chain": [name(r) for r in self.return_chain(ctx.Esp, depth=6)],
            "regs": {r: f"{getattr(ctx, r.capitalize()):#010x}" for r in ("eax", "ebx", "ecx", "edx", "esi", "edi", "ebp")},
            "stack": [f"{w:08x}" for w in words],
            "stack_f": [round(f, 7) if abs(f) < 1e7 else None for f in floats],
            "deref": deref,
        })

    def handle_exception(self, ev: DEBUG_EVENT) -> int:
        rec = ev.u.Exception.ExceptionRecord
        code = rec.ExceptionCode
        addr = rec.ExceptionAddress or 0
        tid = ev.dwThreadId
        if code in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT):
            if addr == self.bp_addr or addr in self.probes:
                ctx = self.get_ctx(tid)
                if addr == self.bp_addr:
                    self.on_advance_beat(tid, ctx)
                    orig = self.bp_orig
                else:
                    self.on_probe(addr, ctx)
                    orig = self.probes[addr]["orig"]
                # step over: restore the byte, rewind EIP, single-step, re-arm on the step event
                self.write(addr, orig)
                ctx = self.get_ctx(tid)
                ctx.Eip = addr
                ctx.EFlags |= 0x100
                self.set_ctx(tid, ctx)
                self.pending_rearm[tid] = addr
                return DBG_CONTINUE
            return DBG_CONTINUE  # loader breakpoints
        if code in (EXCEPTION_SINGLE_STEP, STATUS_WX86_SINGLE_STEP):
            handled = False
            if tid in self.pending_rearm:
                self.write(self.pending_rearm.pop(tid), b"\xCC")
                handled = True
            if self.watch_lo is not None and tid == self.watching_tid:
                dctx = self.get_ctx(tid, WOW64_CONTEXT_DEBUG_REGISTERS | WOW64_CONTEXT_CONTROL)
                if dctx.Dr6 & 1:
                    eip = dctx.Eip
                    idx = self.u32(self.watch_addr)
                    name = self.symbolize or (lambda a: f"{a:#010x}")
                    self.draws.append([idx, name(eip), *[name(r) for r in self.return_chain(dctx.Esp)]])
                    dctx.Dr6 = 0
                    self.set_ctx(tid, dctx)
                    handled = True
            return DBG_CONTINUE if handled else DBG_EXCEPTION_NOT_HANDLED
        return DBG_EXCEPTION_NOT_HANDLED

    def run(self) -> None:
        cmd = subprocess.list2cmdline([str(self.exe), *self.args.game_args])
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(si)
        pi = PROCESS_INFORMATION()
        if not k32.CreateProcessW(None, ctypes.create_unicode_buffer(cmd), None, None, False,
                                  DEBUG_ONLY_THIS_PROCESS, None, str(self.exe.parent), ctypes.byref(si),
                                  ctypes.byref(pi)):
            raise SystemExit(f"CreateProcess failed: {ctypes.get_last_error()}")
        if ON_STARTED is not None:
            ON_STARTED(pi.dwProcessId)
        t0 = time.time()
        ev = DEBUG_EVENT()
        done = False
        killed = False
        while not done:
            if not k32.WaitForDebugEvent(ctypes.byref(ev), 1000):
                if time.time() - t0 > self.args.timeout and not killed:
                    print("timeout, terminating", flush=True)
                    k32.TerminateProcess(pi.hProcess, 1)
                    killed = True
                continue
            status = DBG_CONTINUE
            code = ev.dwDebugEventCode
            if code == 1:
                status = self.handle_exception(ev)
            elif code == 2:
                self.threads[ev.dwThreadId] = ev.u.CreateThread.hThread
            elif code == 3:
                self.on_create_process(ev.u.CreateProcessInfo, ev.dwThreadId)
            elif code == 4:
                self.threads.pop(ev.dwThreadId, None)
            elif code == 5:
                done = True
            elif code == 6:
                if ev.u.LoadDll.hFile:
                    k32.CloseHandle(ev.u.LoadDll.hFile)
            k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status)
            if not killed and self.records and self.records[-1]["beat"] >= self.args.beats:
                print(f"reached beat {self.records[-1]['beat']}, terminating", flush=True)
                k32.TerminateProcess(pi.hProcess, 0)
                killed = True
        self.out.close()
        print(f"{len(self.records)} beats recorded -> {self.args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beats", type=int, default=80)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--watch", type=lambda s: tuple(int(x) for x in s.split(":")))
    ap.add_argument("--bp", action="append", help="extra breakpoint: hex address (official) or PDB symbol (ours)")
    ap.add_argument("--bp-window", type=lambda s: tuple(int(x) for x in s.split(":")),
                    help="record --bp hits only during beats A..B")
    ap.add_argument("game_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.game_args and args.game_args[0] == "--":
        args.game_args = args.game_args[1:]
    Probe(args).run()


if __name__ == "__main__":
    main()
