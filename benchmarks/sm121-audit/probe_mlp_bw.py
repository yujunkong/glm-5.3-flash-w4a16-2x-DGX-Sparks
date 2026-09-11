#!/usr/bin/env python3
"""P0 MLP-up bandwidth / roofline (measurement only).

Shape: F.linear  (M, 4096) @ (12288, 4096)  bf16  — DFlash decode MLP-up.
Does not load the 180G model. Does not change production knobs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

# DFlash MLP-up: in=4096, intermediate=12288, bf16 operands.
K = 4096
N = 12288
ELEM = 2  # bf16
ITERS = 80
WARMUP = 16
MS = (2, 4, 8, 16, 32, 64)
# NVIDIA public: DGX Spark LPDDR5x ~273 GB/s.
PEAK_BW_GBS = 273.0
# Register-resident mma_bf16bf16f32 on GB10: 212.9 TFLOPS (NVIDIA forum, 2025-11).
PEAK_BF16_TFLOPS = 212.9

OUT = Path("/work/sm121-audit")


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def evict_l2(nbytes: int = 64 << 20) -> None:
    """Touch a buffer larger than GB10 L2 (24 MiB) so the next GEMM is cold."""
    junk = torch.empty(nbytes // 2, dtype=torch.int16, device="cuda")
    junk.fill_(1)
    torch.cuda.synchronize()
    del junk


def main() -> None:
    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    print("device", props.name, "cap", torch.cuda.get_device_capability())
    print("l2_bytes", props.L2_cache_size, "smem_optin", getattr(props, "shared_memory_per_block_optin", None))

    w_bytes = N * K * ELEM
    rows: list[dict] = []
    for m in MS:
        a = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        # Steady: reuse same A,W (matches decode: weights stationary).
        ms_hot = time_ms(lambda: F.linear(a, w), WARMUP, ITERS)
        # Cold: evict L2 between timed iters (upper bound on DRAM).
        def cold() -> None:
            evict_l2()
            F.linear(a, w)

        ms_cold = time_ms(cold, 2, 8)

        a_bytes = m * K * ELEM
        c_bytes = m * N * ELEM
        # Compulsory traffic if the full weight is read once + A + C.
        traffic = w_bytes + a_bytes + c_bytes
        flops = 2 * m * K * N
        us_hot = ms_hot * 1000.0
        us_cold = ms_cold * 1000.0
        gbs_hot = traffic / (us_hot * 1e-6) / 1e9
        gbs_cold = traffic / (us_cold * 1e-6) / 1e9
        tflops_hot = flops / (us_hot * 1e-6) / 1e12
        ai = flops / traffic  # FLOP/byte
        ridge = (PEAK_BF16_TFLOPS * 1e12) / (PEAK_BW_GBS * 1e9)
        bound = "bandwidth" if ai < ridge else "compute_or_mixed"
        rows.append(
            {
                "M": m,
                "K": K,
                "N": N,
                "weight_bytes": w_bytes,
                "act_bytes": a_bytes,
                "out_bytes": c_bytes,
                "traffic_bytes": traffic,
                "flops": flops,
                "us_hot": round(us_hot, 2),
                "us_cold": round(us_cold, 2),
                "gbs_hot": round(gbs_hot, 2),
                "gbs_cold": round(gbs_cold, 2),
                "tflops_hot": round(tflops_hot, 3),
                "ai_flop_per_byte": round(ai, 3),
                "ridge_flop_per_byte": round(ridge, 1),
                "bound": bound,
            }
        )
        print(
            f"M={m:3d}  hot={us_hot:7.1f}us  cold={us_cold:7.1f}us  "
            f"GB/s_hot={gbs_hot:6.1f}  TFLOPS={tflops_hot:5.2f}  AI={ai:5.2f}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "phase3-mlp-shapes.csv"
    with csv_path.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    print("wrote", csv_path)

    # Isolated single launch for ncu attach (print pointers/sizes).
    m = 8
    a = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    for _ in range(8):
        F.linear(a, w)
    torch.cuda.synchronize()
    F.linear(a, w)
    torch.cuda.synchronize()
    print("ncu_target_done", "W_bytes", w_bytes, "numel_W", w.numel())


if __name__ == "__main__":
    main()
