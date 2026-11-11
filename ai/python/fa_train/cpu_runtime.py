"""Explicit CPU settings for the learner and its reproducible benchmark."""
import ctypes
import os
import shutil

import torch


def add_cpu_arguments(parser):
    parser.add_argument('--cpu-threads', type=int, help='intra-operation CPU threads')
    parser.add_argument('--cpu-interop-threads', type=int, help='inter-operation CPU threads')
    parser.add_argument('--cpu-affinity', help='allowed logical CPU IDs, comma separated; inspect CCD topology first')
    parser.add_argument('--cpu-bf16', action='store_true',
                        help='bfloat16 encoder; memory, action scores and losses remain float32')
    parser.add_argument('--cpu-compile-map', action='store_true',
                        help='compile the map residual blocks with Inductor; requires a supported C++ toolchain')


def configure_cpu(args, device):
    requested = (args.cpu_threads is not None or args.cpu_interop_threads is not None
                 or args.cpu_affinity is not None or args.cpu_bf16 or args.cpu_compile_map)
    if requested and torch.device(device).type != 'cpu':
        raise ValueError('CPU execution options require --device cpu')
    for name in ('cpu_threads', 'cpu_interop_threads'):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.cpu_compile_map and os.name == 'nt' and shutil.which(os.environ.get('CXX', 'cl')) is None:
        raise ValueError('Map compilation requires a C++ compiler in PATH. Use the x64 Native Tools Command Prompt.')
    if args.cpu_affinity is not None:
        cpus = {int(value.strip()) for value in args.cpu_affinity.split(',')}
        if not cpus or min(cpus) < 0 or max(cpus) >= (os.cpu_count() or 1):
            raise ValueError('CPU affinity IDs must be within the host logical CPU count')
        if os.name == 'nt':
            # One processor group covers the 32 logical CPUs of a 7950X3D.
            if max(cpus) >= ctypes.sizeof(ctypes.c_size_t) * 8:
                raise ValueError('This Windows affinity option supports one processor group')
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel.SetProcessAffinityMask.restype = ctypes.c_int
            if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), sum(1 << cpu for cpu in cpus)):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.sched_setaffinity(0, cpus)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    if args.cpu_interop_threads is not None:
        torch.set_num_interop_threads(args.cpu_interop_threads)
    return dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                affinity=args.cpu_affinity, bf16_encoder=args.cpu_bf16, compiled_map=args.cpu_compile_map,
                torch_version=torch.__version__, capability=torch.backends.cpu.get_cpu_capability())


def configure_model(net, args):
    net.amp = bool(args.cpu_bf16)
    # 2026-11-10, my-server, 1 thread, 128 decisions: the map convolutions forward+backward 1433 -> 1093 ms with a
    # channels_last input (weights untouched, so shared-memory parameters stay shared); the outputs move 1.6e-3, the
    # gradients 1.7e-2 relative, below bf16's own 5e-3 / 5e-2 against float32
    net.map_channels_last = bool(args.cpu_bf16)
    if args.cpu_compile_map:
        # Module.compile preserves parameter names and checkpoint compatibility.
        net.map_blocks.compile(backend='inductor')
