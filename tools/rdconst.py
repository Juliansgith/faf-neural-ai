"""Print float/double constants from a PE image by virtual address.

    rdconst.py ForgedAlliance.exe f:e4f714 d:e4f780 ...
"""
import struct
import sys

import pefile


def main():
    pe = pefile.PE(sys.argv[1], fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    img = pe.get_memory_mapped_image()
    for spec in sys.argv[2:]:
        kind, addr = spec.split(":")
        off = int(addr, 16) - base
        if kind == "f":
            raw = img[off:off + 4]
            print(f"{addr} float  {struct.unpack('<f', raw)[0]!r:>24}  0x{struct.unpack('<I', raw)[0]:08x}")
        else:
            raw = img[off:off + 8]
            print(f"{addr} double {struct.unpack('<d', raw)[0]!r:>24}  0x{struct.unpack('<Q', raw)[0]:016x}")


if __name__ == "__main__":
    main()
