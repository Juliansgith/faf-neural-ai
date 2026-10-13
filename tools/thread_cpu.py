"""Per-thread CPU use of running game processes (which thread costs what: sim, render, loader...).

    thread_cpu.py [--name main.exe] [--seconds 10] [--pid N]

Samples every thread's kernel+user time twice and prints the busiest threads as a percentage of one core,
with their start address (module+offset) and the thread's description if the engine set one.
Windows only; plain ctypes, no dependencies.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import time

TH32CS_SNAPTHREAD, TH32CS_SNAPPROCESS, TH32CS_SNAPMODULE = 0x4, 0x2, 0x8
THREAD_QUERY_INFORMATION = 0x0040
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")


class THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ThreadID", wt.DWORD),
                ("th32OwnerProcessID", wt.DWORD), ("tpBasePri", wt.LONG), ("tpDeltaPri", wt.LONG), ("dwFlags", wt.DWORD)]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", wt.LONG), ("dwFlags", wt.DWORD),
                ("szExeFile", wt.WCHAR * 260)]


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("th32ModuleID", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("GlblcntUsage", wt.DWORD), ("ProccntUsage", wt.DWORD), ("modBaseAddr", ctypes.c_void_p),
                ("modBaseSize", wt.DWORD), ("hModule", wt.HMODULE), ("szModule", wt.WCHAR * 256),
                ("szExePath", wt.WCHAR * 260)]


k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.OpenThread.restype = wt.HANDLE
for _f in ("Process32FirstW", "Process32NextW", "Module32FirstW", "Module32NextW", "Thread32First", "Thread32Next"):
    getattr(k32, _f).argtypes = [wt.HANDLE, ctypes.c_void_p]
INVALID_HANDLE = ctypes.c_void_p(-1).value


def processes(name: str) -> list[int]:
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    out = []
    ok = k32.Process32FirstW(snap, ctypes.byref(entry))
    while ok:
        if entry.szExeFile.lower() == name.lower():
            out.append(entry.th32ProcessID)
        ok = k32.Process32NextW(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return out


def modules(pid: int) -> list[tuple[int, int, str]]:
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | 0x10, pid)   # | TH32CS_SNAPMODULE32
    if not snap or snap == INVALID_HANDLE:                                 # the process has exited
        return []
    entry = MODULEENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    out = []
    ok = k32.Module32FirstW(snap, ctypes.byref(entry))
    while ok:
        out.append((entry.modBaseAddr or 0, entry.modBaseSize, entry.szModule))
        ok = k32.Module32NextW(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return out


def thread_ids(pid: int) -> list[int]:
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    entry = THREADENTRY32()
    entry.dwSize = ctypes.sizeof(entry)
    out = []
    ok = k32.Thread32First(snap, ctypes.byref(entry))
    while ok:
        if entry.th32OwnerProcessID == pid:
            out.append(entry.th32ThreadID)
        ok = k32.Thread32Next(snap, ctypes.byref(entry))
    k32.CloseHandle(snap)
    return out


def thread_info(tid: int):
    h = k32.OpenThread(THREAD_QUERY_INFORMATION, False, tid)
    if not h:
        return None
    try:
        c, e, kt, ut = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
        if not k32.GetThreadTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
            return None
        busy = ((kt.dwHighDateTime << 32) | kt.dwLowDateTime) + ((ut.dwHighDateTime << 32) | ut.dwLowDateTime)
        start = ctypes.c_void_p()
        ntdll.NtQueryInformationThread(h, 9, ctypes.byref(start), ctypes.sizeof(start), None)
        desc = ctypes.c_wchar_p()
        name = ""
        try:
            if k32.GetThreadDescription(h, ctypes.byref(desc)) >= 0 and desc.value:
                name = desc.value
        except AttributeError:
            pass
        return busy / 1e7, start.value or 0, name
    finally:
        k32.CloseHandle(h)


def where(address: int, mods) -> str:
    for base, size, name in mods:
        if base <= address < base + size:
            return f"{name}+0x{address - base:X}"
    return f"0x{address:X}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main.exe")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    pids = [a.pid] if a.pid else processes(a.name)
    if not pids:
        raise SystemExit(f"no {a.name} running")
    before = {pid: {tid: thread_info(tid) for tid in thread_ids(pid)} for pid in pids}
    t0 = time.perf_counter()
    time.sleep(a.seconds)
    wall = time.perf_counter() - t0
    for pid in pids:
        mods = modules(pid)
        if not mods:
            print(f"pid {pid}: exited during the sample")
            continue
        rows = []
        for tid in thread_ids(pid):
            now, old = thread_info(tid), before[pid].get(tid)
            if now and old:
                rows.append(((now[0] - old[0]) / wall * 100, tid, where(now[1], mods), now[2]))
        rows.sort(reverse=True)
        total = sum(r[0] for r in rows)
        print(f"pid {pid}: {total:.0f} % of one core over {wall:.1f} s, {len(rows)} threads")
        for pct, tid, start, name in rows[: a.top]:
            print(f"  {pct:6.1f} %  tid {tid:6d}  {start}  {name}")


if __name__ == "__main__":
    main()
