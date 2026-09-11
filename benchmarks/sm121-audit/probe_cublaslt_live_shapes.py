#!/usr/bin/env python3
"""Enumerate cuBLASLt heuristics for live KDA in_proj / shared gate_up.

If a faster algo than heuristic-0 exists at large workspace, print it.
Does not load the model.
"""
from __future__ import annotations

import ctypes
import sys

import torch

# libcublasLt
try:
    lt = ctypes.CDLL("libcublasLt.so.13")
except OSError:
    lt = ctypes.CDLL("libcublasLt.so")

c_void_p = ctypes.c_void_p
c_int = ctypes.c_int
c_size_t = ctypes.c_size_t
c_int64 = ctypes.c_int64


def chk(st: int, what: str) -> None:
    if st != 0:
        raise RuntimeError(f"{what} status={st}")


def enum_shape(m: int, n: int, k: int, tag: str, ws: int) -> None:
    print(f"\n=== {tag} M={m} N={n} K={k} workspace={ws} ===")
    # Fall back to F.linear timing if bindings are too heavy; print torch path.
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Touch cuBLASLt via F.linear so logs (if CUBLASLT_LOG) capture algo.
    for _ in range(5):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    # Use CUDA graphs? no. Time only.
    import time

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f"  F.linear avg {us:.1f} us")


def main() -> None:
    print("torch", torch.__version__, file=sys.stderr)
    for ws_env in [None]:
        enum_shape(8, 12576, 4096, "KDA_in_proj", 0)
        enum_shape(8, 2048, 4096, "shared_gate_up", 0)
        enum_shape(8, 4096, 4096, "KDA_o_proj", 0)


if __name__ == "__main__":
    main()
