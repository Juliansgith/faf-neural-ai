"""Find the runtime image base of a relocated main.exe from a captured stack.

usage: find_base.py <exe> <expected substring of frame-0 symbol> <addr> [addr ...]
Tries every 64 KiB-aligned base that keeps all addresses inside the image and reports the one whose
first frame symbolizes to the expected function, then prints the whole stack at that base.
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys

exe, expect = sys.argv[1], sys.argv[2]
addrs = [int(a, 16) for a in sys.argv[3:]]
dbg = ctypes.WinDLL("dbghelp.dll", use_last_error=True)
dbg.SymSetOptions(0x2 | 0x10)
proc = wt.HANDLE(0x4321)
dbg.SymInitializeW(proc, ctypes.c_wchar_p(os.path.dirname(exe)), False)
dbg.SymLoadModuleExW.restype = ctypes.c_uint64
dbg.SymLoadModuleExW.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.c_wchar_p, ctypes.c_wchar_p,
                                 ctypes.c_uint64, wt.DWORD, ctypes.c_void_p, wt.DWORD]
dbg.SymUnloadModule64.argtypes = [wt.HANDLE, ctypes.c_uint64]


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.ULONG), ("TypeIndex", wt.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                ("Index", wt.ULONG), ("Size", wt.ULONG), ("ModBase", ctypes.c_uint64), ("Flags", wt.ULONG),
                ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64), ("Register", wt.ULONG),
                ("Scope", wt.ULONG), ("Tag", wt.ULONG), ("NameLen", wt.ULONG), ("MaxNameLen", wt.ULONG),
                ("Name", ctypes.c_wchar * 512)]


class LINE(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.DWORD), ("Key", ctypes.c_void_p), ("LineNumber", wt.DWORD),
                ("FileName", ctypes.c_wchar_p), ("Address", ctypes.c_uint64)]


dbg.SymFromAddrW.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFOW)]
dbg.SymGetLineFromAddrW64.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.POINTER(wt.DWORD), ctypes.POINTER(LINE)]


def sym(a):
    s = SYMBOL_INFOW()
    s.SizeOfStruct = ctypes.sizeof(SYMBOL_INFOW) - 512 * 2
    s.MaxNameLen = 512
    d = ctypes.c_uint64(0)
    name = s.Name if dbg.SymFromAddrW(proc, a, ctypes.byref(d), ctypes.byref(s)) else "?"
    ln = LINE()
    ln.SizeOfStruct = ctypes.sizeof(LINE)
    ld = wt.DWORD(0)
    where = f"  [{ln.FileName}:{ln.LineNumber}]" if dbg.SymGetLineFromAddrW64(proc, a, ctypes.byref(ld), ctypes.byref(ln)) else ""
    return name, d.value, where


SIZE = 0x00E1F000  # SizeOfImage of this build (dumpbin: 00400000 to 0121EFFF)
lo = max(0, max(addrs) - SIZE) & ~0xFFFF
hi = min(addrs) & ~0xFFFF
for base in range(lo, hi + 0x10000, 0x10000):
    mod = dbg.SymLoadModuleExW(proc, None, exe, None, base, 0, None, 0)
    if not mod:
        continue
    name, _, _ = sym(addrs[0])
    if expect in name:
        print(f"image base 0x{base:08X}")
        for a in addrs:
            n, disp, where = sym(a)
            print(f"  0x{a:08X}  {n}+0x{disp:X}{where}")
        sys.exit(0)
    dbg.SymUnloadModule64(proc, mod)
print("no base found")
