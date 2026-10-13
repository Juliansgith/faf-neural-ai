"""Symbolize crash-log addresses of main.exe with its PDB via dbghelp (offline, no running process).

usage: symbolize.py <exe> <image base hex> <addr> [addr ...]
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys

exe, base = sys.argv[1], int(sys.argv[2], 16)
addrs = [int(a, 16) for a in sys.argv[3:]]

dbg = ctypes.WinDLL("dbghelp.dll", use_last_error=True)
SYMOPT_UNDNAME, SYMOPT_DEFERRED_LOADS, SYMOPT_LOAD_LINES = 0x2, 0x4, 0x10
dbg.SymSetOptions(SYMOPT_UNDNAME | SYMOPT_LOAD_LINES | 0x40)  # 0x40 = SYMOPT_LOAD_ANYTHING
proc = wt.HANDLE(0x1234)  # any unique value works for offline symbol loading
if not dbg.SymInitializeW(proc, ctypes.c_wchar_p(os.path.dirname(exe)), False):
    sys.exit("SymInitialize failed")
dbg.SymLoadModuleExW.restype = ctypes.c_uint64
dbg.SymLoadModuleExW.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.c_wchar_p, ctypes.c_wchar_p,
                                 ctypes.c_uint64, wt.DWORD, ctypes.c_void_p, wt.DWORD]
mod = dbg.SymLoadModuleExW(proc, None, exe, None, base, 0, None, 0)
if not mod:
    sys.exit(f"SymLoadModuleEx failed: {ctypes.get_last_error()}")

class IMAGEHLP_MODULEW64(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.DWORD), ("BaseOfImage", ctypes.c_uint64), ("ImageSize", wt.DWORD),
                ("TimeDateStamp", wt.DWORD), ("CheckSum", wt.DWORD), ("NumSyms", wt.DWORD), ("SymType", wt.DWORD),
                ("ModuleName", ctypes.c_wchar * 32), ("ImageName", ctypes.c_wchar * 256),
                ("LoadedImageName", ctypes.c_wchar * 256), ("LoadedPdbName", ctypes.c_wchar * 256),
                ("CVSig", wt.DWORD), ("CVData", ctypes.c_wchar * 780), ("PdbSig", wt.DWORD), ("PdbSig70", ctypes.c_byte * 16),
                ("PdbAge", wt.DWORD), ("PdbUnmatched", wt.BOOL), ("DbgUnmatched", wt.BOOL), ("LineNumbers", wt.BOOL),
                ("GlobalSymbols", wt.BOOL), ("TypeInfo", wt.BOOL), ("SourceIndexed", wt.BOOL), ("Publics", wt.BOOL),
                ("MachineType", wt.DWORD), ("Reserved", wt.DWORD)]


mi = IMAGEHLP_MODULEW64()
mi.SizeOfStruct = ctypes.sizeof(IMAGEHLP_MODULEW64)
dbg.SymGetModuleInfoW64.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.POINTER(IMAGEHLP_MODULEW64)]
ok = dbg.SymGetModuleInfoW64(proc, mod, ctypes.byref(mi))
print(f"module info ok={ok} err={ctypes.get_last_error()} symtype={mi.SymType} pdb='{mi.LoadedPdbName}' cv='{mi.CVData}' unmatched={mi.PdbUnmatched}")

MAX_NAME = 1024


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.ULONG), ("TypeIndex", wt.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                ("Index", wt.ULONG), ("Size", wt.ULONG), ("ModBase", ctypes.c_uint64), ("Flags", wt.ULONG),
                ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64), ("Register", wt.ULONG),
                ("Scope", wt.ULONG), ("Tag", wt.ULONG), ("NameLen", wt.ULONG), ("MaxNameLen", wt.ULONG),
                ("Name", ctypes.c_wchar * MAX_NAME)]


class IMAGEHLP_LINEW64(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.DWORD), ("Key", ctypes.c_void_p), ("LineNumber", wt.DWORD),
                ("FileName", ctypes.c_wchar_p), ("Address", ctypes.c_uint64)]


dbg.SymFromAddrW.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFOW)]
dbg.SymGetLineFromAddrW64.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.POINTER(wt.DWORD), ctypes.POINTER(IMAGEHLP_LINEW64)]
for a in addrs:
    sym = SYMBOL_INFOW()
    sym.SizeOfStruct = ctypes.sizeof(SYMBOL_INFOW) - MAX_NAME * ctypes.sizeof(ctypes.c_wchar)
    sym.MaxNameLen = MAX_NAME
    disp = ctypes.c_uint64(0)
    name = "?"
    if dbg.SymFromAddrW(proc, a, ctypes.byref(disp), ctypes.byref(sym)):
        name = f"{sym.Name}+0x{disp.value:X}"
    line = IMAGEHLP_LINEW64()
    line.SizeOfStruct = ctypes.sizeof(IMAGEHLP_LINEW64)
    ldisp = wt.DWORD(0)
    where = ""
    if dbg.SymGetLineFromAddrW64(proc, a, ctypes.byref(ldisp), ctypes.byref(line)):
        where = f"  [{line.FileName}:{line.LineNumber}]"
    print(f"0x{a:08X}  {name}{where}")


