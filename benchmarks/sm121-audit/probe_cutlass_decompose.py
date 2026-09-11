#!/usr/bin/env python3
"""Decompose cutlass_80 45% into per-module F.linear shapes.

Does not load the 180G model. Shapes are TP=2 runtime shards from
checkpoint headers + vLLM fused Linear layout. Decode M=8.
Does not change production knobs / rebuild anything.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

M = 8  # 1 bonus + 7 DFlash mask tokens
ELEM = 2  # bf16
ITERS = 40
WARMUP = 12
PEAK_BW_GBS = 273.0
PEAK_BF16_TFLOPS = 212.9
# kernel-sum.txt window (phase 5)
KERNEL_US = 15_963_921.168
STEPS = 134
OUT = Path("/work/sm121-audit")

# Runtime F.linear weight is (N, K). n_per_step = launches per decode step
# (1 target verify + 1 DFlash propose). Visual / MTP not in text decode.
OPS: list[dict] = [
    # KDA: 34 linear-attention layers. in_proj fuses q,k,v,b,f_a,g_a.
    # N = 3*(8192/2) + (64/2) + 128 + 128 = 12480, K=4096.
    {"module": "KDA in_proj_qkvbfg_a", "N": 12480, "K": 4096, "n": 34},
    {"module": "KDA o_proj", "N": 4096, "K": 4096, "n": 34},
    {"module": "KDA f_b_proj", "N": 4096, "K": 128, "n": 34},
    {"module": "KDA g_b_proj", "N": 4096, "K": 128, "n": 34},
    # MLA: 11 deepseek_sparse_attention layers. fused_qkv_a is replicated.
    {"module": "MLA fused_qkv_a", "N": 2048, "K": 4096, "n": 11},
    {"module": "MLA q_b_proj", "N": 8192, "K": 1536, "n": 11},
    {"module": "MLA kv_b_proj", "N": 16384, "K": 512, "n": 11},
    {"module": "MLA o_proj", "N": 4096, "K": 8192, "n": 11},
    {"module": "MLA indexer wq_b", "N": 4096, "K": 1536, "n": 11},
    {"module": "MLA indexer wk", "N": 128, "K": 4096, "n": 11},
    # Shared experts: 42 sparse layers, fused gate_up, TP column on N.
    {"module": "shared_experts gate_up", "N": 2048, "K": 4096, "n": 42},
    {"module": "shared_experts down", "N": 4096, "K": 1024, "n": 42},
    {"module": "MoE router gate", "N": 288, "K": 4096, "n": 42},
    # Target dense MLP layers 0-2.
    {"module": "target dense gate_up", "N": 12288, "K": 4096, "n": 3},
    {"module": "target dense down", "N": 4096, "K": 6144, "n": 3},
    # DFlash2 5-layer Qwen3. gate_up fused; qkv fused; conv ReplicatedLinear.
    {"module": "DFlash MLP gate_up", "N": 12288, "K": 4096, "n": 5},
    {"module": "DFlash MLP down", "N": 4096, "K": 6144, "n": 5},
    {"module": "DFlash attn qkv", "N": 3072, "K": 4096, "n": 5},
    {"module": "DFlash attn o_proj", "N": 4096, "K": 2048, "n": 5},
    {"module": "DFlash conv kernel_proj", "N": 1024, "K": 4096, "n": 10},
    {"module": "DFlash fc aux-combine", "N": 4096, "K": 20480, "n": 1},
    {"module": "DFlash selector h_proj", "N": 256, "K": 4096, "n": 1},
]


def time_us(fn, warmup: int, iters: int) -> float:
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
    return st.elapsed_time(en) / iters * 1000.0


def kernel_name(m: int, n: int, k: int) -> str:
    """One-shape profiler: which CUDA kernel F.linear actually launches."""
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(4):
        F.linear(a, w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        F.linear(a, w)
        torch.cuda.synchronize()
    best = ("?", 0.0)
    for e in prof.key_averages():
        us = float(
            getattr(e, "device_time_total", 0.0)
            or getattr(e, "cuda_time_total", 0.0)
            or 0.0
        )
        if us > best[1]:
            best = (e.key, us)
    name = best[0]
    if "cutlass_80" in name:
        if "128x1" in name:
            return "cutlass_80_128x1"
        if "128x2" in name:
            return "cutlass_80_128x2"
        return "cutlass_80_other"
    if "gemmSN" in name or "gemv" in name.lower():
        return "cublas_gemv"
    if "nvjet" in name or "sm90" in name or "sm120" in name:
        return name[:48]
    return name[:60]


def main() -> None:
    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    print("device", props.name, "cap", torch.cuda.get_device_capability())
    print("l2_bytes", props.L2_cache_size)

    timed: dict[tuple[int, int], dict] = {}
    unique = sorted({(op["N"], op["K"]) for op in OPS})
    for n, k in unique:
        a = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        us = time_us(lambda a=a, w=w: F.linear(a, w), WARMUP, ITERS)
        kn = kernel_name(M, n, k)
        w_bytes = n * k * ELEM
        traffic = w_bytes + M * k * ELEM + M * n * ELEM
        flops = 2 * M * k * n
        timed[(n, k)] = {
            "us": us,
            "kernel": kn,
            "w_mb": w_bytes / 1e6,
            "traffic": traffic,
            "flops": flops,
            "gbs": traffic / (us * 1e-6) / 1e9,
            "tflops": flops / (us * 1e-6) / 1e12,
            "ai": flops / traffic,
        }
        print(
            f"bench N={n:5d} K={k:5d}  {us:7.1f}us  {timed[(n, k)]['w_mb']:7.2f}MB  "
            f"{kn}  {timed[(n, k)]['gbs']:.0f}GB/s"
        )
        del a, w
        torch.cuda.empty_cache()

    ridge = (PEAK_BF16_TFLOPS * 1e12) / (PEAK_BW_GBS * 1e9)
    rows = []
    for op in OPS:
        t = timed[(op["N"], op["K"])]
        us_step = t["us"] * op["n"]
        us_win = us_step * STEPS
        pct_kernel = 100.0 * us_win / KERNEL_US
        rows.append(
            {
                "module": op["module"],
                "M": M,
                "N": op["N"],
                "K": op["K"],
                "dtype": "bf16",
                "weight_mb": round(t["w_mb"], 3),
                "n_per_step": op["n"],
                "us_each": round(t["us"], 2),
                "us_per_step": round(us_step, 1),
                "calls_window": op["n"] * STEPS,
                "us_window": round(us_win, 1),
                "pct_kernel": round(pct_kernel, 2),
                "dram_mb_each": round(t["traffic"] / 1e6, 3),
                "gbs": round(t["gbs"], 1),
                "tflops": round(t["tflops"], 3),
                "ai": round(t["ai"], 2),
                "ridge": round(ridge, 1),
                "bound": "bandwidth" if t["ai"] < ridge else "compute",
                "kernel": t["kernel"],
            }
        )
    rows.sort(key=lambda r: -r["pct_kernel"])
    total_pct = sum(r["pct_kernel"] for r in rows)
    cutlass_pct = sum(
        r["pct_kernel"] for r in rows if r["kernel"].startswith("cutlass_80")
    )
    print("\n=== rank by reconstructed % of kernel-sum window ===")
    print(f"{'rk':>2} {'pct':>6} {'us/s':>8} {'n':>3} {'N':>6} {'K':>6} {'MB':>7} {'kernel':<18} module")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:2d} {r['pct_kernel']:6.2f} {r['us_each']:8.1f} {r['n_per_step']:3d} "
            f"{r['N']:6d} {r['K']:6d} {r['weight_mb']:7.2f} {r['kernel']:<18} {r['module']}"
        )
    print(f"sum_all_pct={total_pct:.2f}  sum_cutlass80_pct={cutlass_pct:.2f}")
    print(f"measured_cutlass80_family≈44.95  residual={44.95 - cutlass_pct:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "phase6-cutlass-shapes.csv"
    with csv_path.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    print("wrote", csv_path)

    txt = OUT / "phase6-cutlass-decompose.txt"
    lines = [
        f"sum_all_pct={total_pct:.2f}",
        f"sum_cutlass80_pct={cutlass_pct:.2f}",
        f"measured_cutlass80≈44.95 residual={44.95 - cutlass_pct:.2f}",
        f"steps={STEPS} kernel_us={KERNEL_US} M={M}",
        "",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i:2d}  {r['pct_kernel']:6.2f}%  {r['us_each']:7.1f}us  n={r['n_per_step']:3d}  "
            f"({r['M']},{r['K']})x({r['N']},{r['K']})  {r['weight_mb']:7.2f}MB  "
            f"{r['kernel']}  {r['gbs']:.0f}GB/s  AI={r['ai']}  {r['module']}"
        )
    txt.write_text("\n".join(lines) + "\n")
    print("wrote", txt)


if __name__ == "__main__":
    main()
