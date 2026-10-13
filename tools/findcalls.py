import sys, struct, pefile
exe, target = sys.argv[1], int(sys.argv[2], 16)
pe = pefile.PE(exe, fast_load=True)
base = pe.OPTIONAL_HEADER.ImageBase
img = pe.get_memory_mapped_image()
text = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".text")
lo, hi = text.VirtualAddress, text.VirtualAddress + text.Misc_VirtualSize
for rva in range(lo, hi - 5):
    if img[rva] == 0xE8:
        rel = struct.unpack_from("<i", img, rva + 1)[0]
        if base + rva + 5 + rel == target:
            print(hex(base + rva))
