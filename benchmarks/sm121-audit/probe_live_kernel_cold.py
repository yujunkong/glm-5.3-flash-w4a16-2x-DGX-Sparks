#!/usr/bin/env python3
"""Hot vs cold weight + kernel name for live-confirmed GEMM shapes.

Isolated same-weight reuse overstates L2 hit. Live shared_experts uses a
distinct 16.8 MB weight per layer (42 layers) against 24 MB L2.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

SHAPES = [
    ("KDA_in_proj", 8, 12576, 4096),
    ("KDA_o_proj", 8, 4096, 4096),
    ("shared_gate_up", 8, 2048, 4096),
    ("lm_head_tp", 8, 77440, 4096),
]


def hot_us(m: int, n: int, k: int, iters: int = 40) -> float:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(8):
        F.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        F.linear(x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def cold_us(m: int, n: int, k: int, n_w: int = 42, rounds: int = 2) -> float:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(n_w)]
    # one pass to allocate
    for w in ws:
        F.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(rounds):
        for w in ws:
            F.linear(x, w)
            n += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def kernel_info(m: int, n: int, k: int) -> None:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(4):
        F.linear(x, w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(8):
            F.linear(x, w)
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        us = float(getattr(e, "device_time_total", 0.0) or 0.0)
        if us <= 0:
            continue
        rows.append((us, e.count, e.key[:160]))
    rows.sort(reverse=True)
    for us, cnt, key in rows[:6]:
        print(f"    {us:10.1f}us n={cnt:3d} {key}")


def main() -> None:
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0))
    print("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"))
    peak = 273.0
    for tag, m, n, k in SHAPES:
        w_mb = n * k * 2 / 1e6
        h = hot_us(m, n, k)
        c = cold_us(m, n, k, n_w=8 if n >= 30000 else 42)
        print(f"\n=== {tag} M={m} N={n} K={k} W={w_mb:.2f} MB ===")
        print(f"  hot  {h:7.1f} us  ~{w_mb/(h*1e-3):6.1f} GB/s ({100*w_mb/(h*1e-3)/peak:.0f}% of 273)")
        print(f"  cold {c:7.1f} us  ~{w_mb/(c*1e-3):6.1f} GB/s ({100*w_mb/(c*1e-3)/peak:.0f}% of 273)")
        print(f"  DRAM roof {w_mb/peak*1e3:7.1f} us")
        kernel_info(m, n, k)


if __name__ == "__main__":
    main()
