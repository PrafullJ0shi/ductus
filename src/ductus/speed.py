"""CPU threads and device choice for the models (TrOCR read-back, the RNN sampler).

PyTorch starts one thread per core. On a machine that is already busy, the threads of a
small model wait on each other and on other programs: on an 8-core box with a load of 20,
one RNN step took 85 ms with 4 threads and 2 ms with one. So the thread count follows the
cores that are actually free when ductus starts, and a GPU is used when one is present.

  DUCTUS_THREADS=n   fix the thread count
  DUCTUS_DEVICE=cpu|cuda|mps   force a device
"""
from __future__ import annotations

import contextlib
import os

_done = False


def free_cores() -> int:
    n = os.cpu_count() or 1
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = 0.0
    return max(1, min(n, int(round(n - load))))


def n_threads() -> int:
    v = os.environ.get("DUCTUS_THREADS", "").strip()
    if v.isdigit() and int(v) > 0:
        return int(v)
    return max(1, min(8, free_cores()))


def setup() -> int:
    """Set the torch thread count once per process; returns it."""
    global _done
    import torch
    if not _done:
        torch.set_num_threads(n_threads())
        _done = True
    return torch.get_num_threads()


def device():
    import torch
    want = os.environ.get("DUCTUS_DEVICE", "").strip().lower()
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def threads(n: int):
    """Run a block with at most n torch threads (small per-step models scale badly)."""
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, min(n, old)))
    try:
        yield
    finally:
        torch.set_num_threads(old)
