#!/usr/bin/env python3
"""Algo/workspace sweep for live-confirmed BF16 F.linear shapes.

Does not load the model. Does not change serving. GPU0 microbench only.
Shapes from live nsys Python-BT (phase7 128x1/128x2).
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

# Confirmed live ops (N,K) = weight (out, in); M from DFlash decode.
SHAPES = [
    ("KDA_in_proj", 8, 12576, 4096),  # 10.75% CUDA 128x1
    ("KDA_o_proj", 8, 4096, 4096),  # 4.77% CUDA 128x1
    ("shared_gate_up", 8, 2048, 4096),  # 6.03% CUDA 128x2
    ("MLA_o_proj", 8, 4096, 8192),  # 2.32% CUDA 128x1 (RowParallel local)
    ("lm_head_tp", 8, 77440, 4096),  # 3.61% CUDA 128x2
]


def bench(m: int, n: int, k: int, iters: int = 40, warmup: int = 10) -> float:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        y = F.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = F.linear(x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main() -> None:
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0))
    print("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"))
    print("preferred_blas", torch.backends.cuda.preferred_blas_library())
    w_mb = lambda n, k: n * k * 2 / 1e6
    peak = 273.0
    for tag, m, n, k in SHAPES:
        us = bench(m, n, k)
        mb = w_mb(n, k)
        gbs = mb / (us * 1e-3) if us else 0  # GB/s  (MB / ms)
        print(
            f"{tag:16s} M={m:3d} N={n:6d} K={k:4d}  {us:7.1f} us  "
            f"W={mb:7.2f} MB  ~{gbs:6.1f} GB/s  ({100*gbs/peak:.0f}% of 273)"
        )


if __name__ == "__main__":
    main()
