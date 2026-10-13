"""Index the binary addresses faf-re's sources document ("Address: 0x0040EEC0 (FUN_0040EEC0, Name)") so an
address in FAF's ForgedAlliance.exe can be named after the recovered function that contains it.

    from binary_index import BinaryIndex
    idx = BinaryIndex(Path("faf-re-dev/src"))
    idx.name(0x0040EF36)   # -> 'moho::CRandomStream::FRandGaussian+0x76' (nearest documented start below)
"""
from __future__ import annotations

import bisect
import re
from pathlib import Path

ADDR_RE = re.compile(r"Address:\s*0x([0-9A-Fa-f]{6,8})\b([^\n]*)")
SIG_RE = re.compile(r"^\s*(?:[\w:<>,*&\s\[\]]+?\s)?([\w:~]+(?:::[\w~]+)*)\s*\(")


class BinaryIndex:
    def __init__(self, src: Path):
        entries: dict[int, str] = {}
        for path in list(src.rglob("*.cpp")) + list(src.rglob("*.h")):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                m = ADDR_RE.search(line)
                if not m:
                    continue
                addr = int(m.group(1), 16)
                label = None
                paren = re.search(r"\(([^)]*)\)", m.group(2))
                if paren:
                    parts = [p.strip() for p in paren.group(1).split(",")]
                    named = [p for p in parts if p and not p.startswith(("FUN_", "sub_", "func_")) and " " not in p]
                    label = named[0] if named else None
                if label is None:
                    # the first code line after the comment block that looks like a definition
                    for j in range(i + 1, min(i + 40, len(lines))):
                        s = lines[j].strip()
                        if s.startswith(("*", "/", "//")) or not s:
                            continue
                        sm = SIG_RE.match(lines[j])
                        if sm:
                            label = sm.group(1)
                        break
                if label and (addr not in entries or len(label) > len(entries[addr])):
                    entries[addr] = label
                elif addr not in entries:
                    entries[addr] = f"FUN_{addr:08X}"
        self.addrs = sorted(entries)
        self.labels = [entries[a] for a in self.addrs]

    def name(self, addr: int, max_distance: int = 0x3000) -> str:
        i = bisect.bisect_right(self.addrs, addr) - 1
        if i < 0 or addr - self.addrs[i] > max_distance:
            return f"{addr:#010x}"
        return f"{self.labels[i]}+{addr - self.addrs[i]:#x}"
