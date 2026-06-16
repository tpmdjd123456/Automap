"""Background CPU/RAM heartbeat for long-running pipeline stages.

stdlib-only: reads /proc/self/status, /proc/stat, /proc/meminfo. No-op on
systems without /proc (macOS), so it's safe to wrap everywhere.

Usage::

    from heartbeat import heartbeat
    with heartbeat("stage6-overlap-set"):
        do_long_thing()

Emits one line every ``interval`` seconds (default 300 = 5 min) like::

    [heartbeat stage6-overlap-set t=10.0m CPU=92% RSS=78.4GB MemFree=170GB/251GB]

System-wide CPU% (from /proc/stat) and free memory are reported because the
WP3 scoring stage spawns worker children whose RSS the main process can't see.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Optional, Tuple


def _read_self_rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError):
        return None
    return None


def _read_meminfo_gb() -> Optional[Tuple[float, float]]:
    try:
        total = avail = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / (1024.0 * 1024.0)
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / (1024.0 * 1024.0)
                if total is not None and avail is not None:
                    return total, avail
    except (FileNotFoundError, PermissionError):
        return None
    return None


def _read_cpu_jiffies() -> Optional[Tuple[int, int]]:
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        vals = [int(p) for p in line.split()[1:]]
        return sum(vals), vals[3]
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


@contextmanager
def heartbeat(label: str, interval: float = 300.0, prefix: str = "  "):
    if not os.path.exists("/proc/self/status"):
        yield
        return

    stop = threading.Event()
    t0 = time.time()
    last = _read_cpu_jiffies()

    def _loop():
        nonlocal last
        while not stop.wait(interval):
            now = _read_cpu_jiffies()
            cpu_pct = -1.0
            if last and now:
                dt = now[0] - last[0]
                di = now[1] - last[1]
                if dt > 0:
                    cpu_pct = 100.0 * (dt - di) / dt
            last = now

            rss = _read_self_rss_mb()
            mem = _read_meminfo_gb()
            elapsed_min = (time.time() - t0) / 60.0
            parts = [f"{prefix}[heartbeat {label} t={elapsed_min:.1f}m"]
            if cpu_pct >= 0:
                parts.append(f"CPU={cpu_pct:.0f}%")
            if rss is not None:
                parts.append(f"RSS={rss / 1024.0:.1f}GB")
            if mem is not None:
                parts.append(f"MemFree={mem[1]:.0f}GB/{mem[0]:.0f}GB")
            print(" ".join(parts) + "]", flush=True)

    th = threading.Thread(target=_loop, daemon=True, name=f"heartbeat-{label}")
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=2.0)
