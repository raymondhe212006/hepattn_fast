"""Opt-in CUDA memory logging. Does nothing unless HEPATTN_MEM=1.

mem(tag) prints the peak allocated memory since the previous mem() call, then resets the peak,
so each line is the peak of the code between two calls. The first call of a training step also
covers the previous step's backward pass and optimizer step.
"""

import os

import torch

ENABLED = os.environ.get("HEPATTN_MEM", "0") == "1"


def mem(tag: str) -> None:
    if not ENABLED or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    gb = 2**30
    print(f"[mem] {tag:<40} now {torch.cuda.memory_allocated() / gb:6.2f} GB   peak since last {torch.cuda.max_memory_allocated() / gb:6.2f} GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
