import sys, pefile, capstone
exe, start, end = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3], 16)
pe = pefile.PE(exe, fast_load=True)
base = pe.OPTIONAL_HEADER.ImageBase
data = pe.get_memory_mapped_image()
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
code = data[start - base:end - base]
for ins in md.disasm(code, start):
    print(f"{ins.address:08X}  {ins.mnemonic:6} {ins.op_str}")
