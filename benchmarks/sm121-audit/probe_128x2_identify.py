#!/usr/bin/env python3
"""Find F.linear (M,N,K) that launch cutlass_80 128x2 near ~288 us.

Does not load the 180G model. Does not change production.
Also records torch.profiler kernel name per shape.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

ELEM = 2
ITERS = 24
WARMUP = 8
PEAK_BW = 273.0
PEAK_TC = 212.9
OUT = Path("/work/sm121-audit")

# Inventory already measured in phase 6 (skip exact dupes of those).
# Add unsharded checkpoint shapes, MoE-expanded M, lm_head shard, mm-fp32.
CANDIDATES: list[tuple[str, int, int, int]] = []

def add(tag: str, m: int, n: int, k: int) -> None:
    CANDIDATES.append((tag, m, n, k))


for m in (1, 8, 16, 64):
    add("unfused_KDA_qkv_full", m, 8192, 4096)
    add("unfused_KDA_o_full", m, 4096, 8192)
    add("MLA_o_full", m, 4096, 16384)
    add("MLA_qb_full", m, 16384, 1536)
    add("MLA_kvb_full", m, 32768, 512)
    add("shared_gate_unfused", m, 2048, 4096)
    add("shared_down_full", m, 4096, 2048)
    add("expert_w1_bf16", m, 2048, 4096)
    add("expert_w2_bf16", m, 4096, 2048)
    add("dense_gate_unfused", m, 12288, 4096)
    add("KDA_in_fused", m, 12480, 4096)
    add("KDA_o_tp", m, 4096, 4096)
    add("DFlash_fc", m, 4096, 20480)

add("lm_head_tp2", 8, 77440, 4096)
add("embed_like", 8, 4096, 4096)
add("router_288", 8, 288, 4096)
add("hc_fn_T", 8, 24, 16384)
add("hc_fn", 8, 16384, 24)


def kernel_tag(m: int, n: int, k: int) -> str:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
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
    if "128x2" in name:
        return "cutlass_80_128x2"
    if "128x1" in name:
        return "cutlass_80_128x1"
    if "32x32" in name:
        return "cutlass_80_32x32"
    if "gemmSN" in name or "Gemv" in name:
        return "cublas_gemv"
    return name[:70]


def time_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(ITERS):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / ITERS * 1000.0


def main() -> None:
    print("device", torch.cuda.get_device_properties(0).name)
    seen: set[tuple[int, int, int]] = set()
    rows = []
    for tag, m, n, k in CANDIDATES:
        key = (m, n, k)
        if key in seen:
            continue
        seen.add(key)
        w_mb = n * k * ELEM / 1e6
        if w_mb > 900:
            print("skip huge", tag, w_mb)
            continue
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        us = time_us(lambda a=a, w=w: F.linear(a, w))
        kn = kernel_tag(m, n, k)
        traffic = (n * k + m * k + m * n) * ELEM
        gbs = traffic / (us * 1e-6) / 1e9
        flops = 2 * m * k * n
        ai = flops / traffic
        hit = kn == "cutlass_80_128x2" and 180 <= us <= 420
        rec = {
            "tag": tag,
            "M": m,
            "N": n,
            "K": k,
            "weight_mb": round(w_mb, 3),
            "us": round(us, 1),
            "kernel": kn,
            "gbs": round(gbs, 1),
            "ai": round(ai, 2),
            "near_288us_128x2": int(hit),
        }
        rows.append(rec)
        mark = " <<<" if hit else ""
        print(
            f"{kn:20s} {us:7.1f}us  M={m:3d} N={n:5d} K={k:5d} {w_mb:7.2f}MB  "
            f"{gbs:6.0f}GB/s  {tag}{mark}"
        )
        del a, w
        torch.cuda.empty_cache()

    # GateLinear-style mm bf16->fp32
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    wt = torch.randn(288, 4096, device="cuda", dtype=torch.bfloat16)
    us = time_us(lambda: torch.mm(x, wt.T, out_dtype=torch.float32))
    print(f"torch.mm_fp32_out {us:7.1f}us  router-like")

    rows.sort(key=lambda r: -r["us"])
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "phase7-128x2-sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)
    hits = [r for r in rows if r["near_288us_128x2"]]
    print("hits_288us_128x2", len(hits))
    for r in hits:
        print(" HIT", r)


if __name__ == "__main__":
    main()
