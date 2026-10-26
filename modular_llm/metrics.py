"""Process and system memory/IO measurements, with Windows support and POSIX fallback."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]


def _windows_metrics() -> dict:
    import ctypes.wintypes as wt

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD),
            ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wt.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wt.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    kernel32.GetProcessIoCounters.argtypes = [wt.HANDLE, ctypes.POINTER(IO_COUNTERS)]
    kernel32.GetProcessIoCounters.restype = wt.BOOL
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wt.BOOL
    process = kernel32.GetCurrentProcess()

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), ctypes.sizeof(counters)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    io = IO_COUNTERS()
    if not kernel32.GetProcessIoCounters(process, ctypes.byref(io)):
        raise ctypes.WinError(ctypes.get_last_error())

    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(mem)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        raise ctypes.WinError(ctypes.get_last_error())

    return {
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "current_working_set_bytes": int(counters.WorkingSetSize),
        "pagefile_peak_bytes": int(counters.PeakPagefileUsage),
        "read_transfer_bytes": int(io.ReadTransferCount),
        "write_transfer_bytes": int(io.WriteTransferCount),
        "read_operations": int(io.ReadOperationCount),
        "write_operations": int(io.WriteOperationCount),
        "system_total_bytes": int(mem.ullTotalPhys),
        "system_available_bytes": int(mem.ullAvailPhys),
        "system_memory_load_percent": int(mem.dwMemoryLoad),
        "source": "windows_psapi",
    }


def _posix_metrics() -> dict:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    peak = rss * 1024 if sys.platform.startswith("linux") else rss
    read_bytes = None
    write_bytes = None
    try:
        io = Path("/proc/self/io").read_text().splitlines()
        vals = {line.split(":")[0]: int(line.split(":")[1]) for line in io if ":" in line}
        read_bytes = vals.get("read_bytes")
        write_bytes = vals.get("wbytes")
    except OSError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        vals = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in meminfo if ":" in line}
        total = vals.get("MemTotal")
        avail = vals.get("MemAvailable")
    except OSError:
        total = None
        avail = None
    return {
        "peak_working_set_bytes": peak,
        "current_working_set_bytes": None,
        "pagefile_peak_bytes": None,
        "read_transfer_bytes": read_bytes,
        "write_transfer_bytes": write_bytes,
        "read_operations": None,
        "write_operations": None,
        "system_total_bytes": total,
        "system_available_bytes": avail,
        "system_memory_load_percent": None,
        "source": "posix_resource_proc",
    }


def collect_metrics() -> dict:
    if platform.system() == "Windows":
        return _windows_metrics()
    return _posix_metrics()


def gpu_metrics() -> dict:
    if torch.cuda.is_available():
        return {
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(0)),
        }
    return {
        "cuda_available": False,
        "gpu_name": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "current_allocated_bytes": None,
    }


def format_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


@contextmanager
def timer() -> "Timer":
    t0 = time.perf_counter()
    t = Timer()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - t0


class Timer:
    def __init__(self) -> None:
        self.seconds: float | None = None


def set_threads(n: int | None = None) -> int:
    if n is None:
        return torch.get_num_threads()
    if n > 0:
        torch.set_num_threads(n)
    return torch.get_num_threads()


def process_id() -> int:
    return os.getpid()
