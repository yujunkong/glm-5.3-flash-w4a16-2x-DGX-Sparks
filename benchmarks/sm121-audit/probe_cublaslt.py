#!/usr/bin/env python3
"""Microbench: which CUDA kernel PyTorch eager bf16 GEMM uses on GB10.

Does not load the 180G model. Shapes match GLM-5.3 Flash + DFlash2 decode.
"""
from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def cap() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def kernel_hist(prof: profile) -> list[tuple[str, int, float]]:
    """Return [(name, count, cuda_us), ...] sorted by CUDA time."""
    rows: list[tuple[str, int, float]] = []
    for e in prof.key_averages():
        name = e.key
        us = float(getattr(e, "device_time_total", 0.0) or getattr(e, "cuda_time_total", 0.0) or 0.0)
        ncall = int(e.count)
        if us <= 0:
            continue
        rows.append((name, ncall, us))
    rows.sort(key=lambda x: -x[2])
    return rows[:12]


def run_mm(m: int, k: int, n: int, tag: str) -> None:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Warmup so heuristic caches.
    for _ in range(8):
        torch.nn.functional.linear(a, b)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function(tag):
            for _ in range(32):
                torch.nn.functional.linear(a, b)
        torch.cuda.synchronize()
    print(f"\n=== {tag}  F.linear ({m},{k})x({n},{k})T  cap={cap()} ===")
    for name, ncall, us in kernel_hist(prof):
        short = name[:140]
        print(f"  {us:10.1f}us n={ncall:4d} {short}")


def main() -> None:
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0), "cap", cap())
    print("preferred_blas", getattr(torch.backends.cuda, "preferred_blas_library", lambda: "?")())
    # Decode-shaped: spec K=7 → ~8 tokens; also M=1.
    shapes = [
        (8, 4096, 4096, "drafter_attn_in"),
        (8, 4096, 12288, "drafter_mlp_up"),
        (8, 12288, 4096, "drafter_mlp_down"),
        (8, 4096, 288, "moe_gate"),
        (1, 4096, 4096, "m1_attn"),
        (1, 4096, 12288, "m1_mlp_up"),
        (8, 4096, 1536, "q_lora"),  # q_lora_rank=1536
        (8, 4096, 512, "kv_lora"),  # kv_lora_rank=512
    ]
    for m, k, n, tag in shapes:
        run_mm(m, k, n, tag)


if __name__ == "__main__":
    main()
