"""Sampling profiler that attaches to a running game thread: where does it spend its time?
(tools/sample_profile.py starts its own game and profiles the sim thread; this one attaches to any game.)

    thread_profile.py --pid N [--tid T] [--seconds 6] [--interval-ms 2] [--top 30]

Without --tid the busiest thread is profiled. Each sample suspends the thread for a moment, reads EIP and
the top of its stack, and resumes it; stack words that point into main.exe just after a call instruction
are taken as return addresses (a heuristic: stale frames can appear). Addresses are symbolized with
main.pdb. Prints functions by self time (EIP) and by inclusive time (anywhere in the scanned chain).
"""
import argparse
import os
import collections
import ctypes
import ctypes.wintypes as wt
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beat_probe import WOW64_CONTEXT, WOW64_CONTEXT_CONTROL, WOW64_CONTEXT_INTEGER, Symbolizer  # noqa: E402
from thread_cpu import modules, thread_ids, thread_info  # noqa: E402

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.OpenThread.restype = wt.HANDLE
k32.OpenProcess.restype = wt.HANDLE
k32.Wow64GetThreadContext.argtypes = [wt.HANDLE, ctypes.POINTER(WOW64_CONTEXT)]
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_size_t)]
k32.SuspendThread.argtypes = [wt.HANDLE]
k32.ResumeThread.argtypes = [wt.HANDLE]


class Target:
    def __init__(self, pid: int):
        self.hproc = k32.OpenProcess(0x0010 | 0x0400, False, pid)      # VM_READ | QUERY_INFORMATION
        if not self.hproc:
            raise SystemExit(f"OpenProcess failed: {ctypes.get_last_error()}")
        self.mods = modules(pid)
        main = [m for m in self.mods if m[2].lower() == "main.exe"]
        if not main:
            raise SystemExit("main.exe module not found")
        self.base, size, _ = main[0]
        self.lo, self.hi = self.base, self.base + size
        self.pages: dict[int, bytes] = {}
        self.exports: dict[str, list] = {}

    def read(self, addr: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        n = ctypes.c_size_t()
        k32.ReadProcessMemory(self.hproc, ctypes.c_void_p(addr), buf, size, ctypes.byref(n))
        return buf.raw[: n.value]

    def code_before(self, addr: int, n: int = 7) -> bytes:
        out = bytearray()
        for a in range(addr - n, addr):
            page = a & ~0xFFF
            if page not in self.pages:
                self.pages[page] = self.read(page, 0x1000)
            blob = self.pages[page]
            out.append(blob[a & 0xFFF] if len(blob) > (a & 0xFFF) else 0)
        return bytes(out)

    def looks_like_return(self, value: int) -> bool:
        if not (self.lo <= value < self.hi):
            return False
        b = self.code_before(value, 7)
        return (b[2] == 0xE8 or (b[1] == 0xFF and (b[2] & 0x38) == 0x10)
                or (b[4] == 0xFF and (b[5] & 0x38) == 0x10) or (b[5] == 0xFF and (b[6] & 0x38) == 0x10))

    def where(self, addr: int) -> str | None:
        for base, size, name in self.mods:
            if base <= addr < base + size and name.lower() != "main.exe":
                export = self.nearest_export(name, addr - base)
                return f"[{name}!{export}]" if export else f"[{name}]"
        return None

    def nearest_export(self, module: str, rva: int) -> str | None:
        """The exported function at or below `rva` in a system DLL (32-bit copy in SysWOW64): enough to tell a
        wait from a heap call from file I/O. Internal functions show as the nearest export before them."""
        table = self.exports.get(module.lower())
        if table is None:
            table = []
            path = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "SysWOW64" / module
            try:
                import pefile
                pe = pefile.PE(str(path), fast_load=True)
                pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
                for sym in getattr(pe, "DIRECTORY_ENTRY_EXPORT", None).symbols if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") else []:
                    if sym.name:
                        table.append((sym.address, sym.name.decode(errors="replace")))
                table.sort()
            except Exception:
                table = []
            self.exports[module.lower()] = table
        if not table:
            return None
        lo, hi = 0, len(table)
        while lo < hi:
            mid = (lo + hi) // 2
            if table[mid][0] <= rva:
                lo = mid + 1
            else:
                hi = mid
        return table[lo - 1][1] if lo else None


def busiest_thread(pid: int, rank: int = 1, seconds: float = 1.0) -> int:
    """The rank-th busiest thread over `seconds` (1 = busiest)."""
    before = {tid: thread_info(tid) for tid in thread_ids(pid)}
    time.sleep(seconds)
    used = []
    for tid in thread_ids(pid):
        now, old = thread_info(tid), before.get(tid)
        if now and old:
            used.append((now[0] - old[0], tid))
    used.sort(reverse=True)
    return used[min(rank, len(used)) - 1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--tid", type=int)
    ap.add_argument("--rank", type=int, default=1, help="without --tid: profile the rank-th busiest thread")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--interval-ms", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=24, help="return addresses kept per sample")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--exe", default=r"C:\ProgramData\FAForever\bin\main.exe")
    a = ap.parse_args()

    target = Target(a.pid)
    tid = a.tid or busiest_thread(a.pid, a.rank)
    h = k32.OpenThread(0x0002 | 0x0008 | 0x0040, False, tid)     # SUSPEND_RESUME | GET_CONTEXT | QUERY_INFO
    if not h:
        raise SystemExit(f"OpenThread failed: {ctypes.get_last_error()}")
    raw = []
    t_end = time.perf_counter() + a.seconds
    while time.perf_counter() < t_end:
        ctx = WOW64_CONTEXT()
        ctx.ContextFlags = WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER
        if k32.SuspendThread(h) == 0xFFFFFFFF:
            break
        try:
            ok = k32.Wow64GetThreadContext(h, ctypes.byref(ctx))
            stack = target.read(ctx.Esp, 16384) if ok else b""
        finally:
            k32.ResumeThread(h)
        if ok:
            raw.append((ctx.Eip, stack))
        time.sleep(a.interval_ms / 1000.0)

    sym = Symbolizer(Path(a.exe), target.base)

    def name(addr: int) -> str:
        label = target.where(addr)
        if label:
            return label
        s = sym(addr)
        return s.rsplit("+0x", 1)[0]

    self_counts, incl_counts = collections.Counter(), collections.Counter()
    for eip, stack in raw:
        words = struct.unpack(f"<{len(stack) // 4}I", stack[: len(stack) // 4 * 4])
        chain = [w for w in words if target.looks_like_return(w)][: a.depth]
        self_counts[name(eip)] += 1
        seen = {name(eip)} | {name(r) for r in chain}
        for s in seen:
            incl_counts[s] += 1
    n = max(1, len(raw))
    print(f"thread {tid}: {len(raw)} samples over {a.seconds:.0f} s")
    print("\nself (EIP):")
    for s, c in self_counts.most_common(a.top):
        print(f"  {100 * c / n:5.1f} %  {s}")
    print("\ninclusive (EIP or on the stack):")
    for s, c in incl_counts.most_common(a.top):
        print(f"  {100 * c / n:5.1f} %  {s}")


if __name__ == "__main__":
    main()
