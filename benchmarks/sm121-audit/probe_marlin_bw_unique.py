#!/usr/bin/env python3
"""Measure Marlin w1 DRAM bytes vs unique experts (live decode shape).

Prints unique experts, packed W4 bytes touched, kernel us, implied GB/s.
"""
from __future__ import annotations

import time

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

M, TOPK, E, N, K, GROUP = 8, 8, 144, 1024, 4096, 128


def stack_experts(w: torch.Tensor):
    q_l, s_l = [], []
    perm = torch.arange(w.shape[-1], device=w.device)
    for i in range(w.shape[0]):
        _, qweight, scales, *_ = marlin_quantize(
            w[i].transpose(1, 0).contiguous(),
            scalar_types.uint4b8,
            GROUP,
            False,
            perm,
        )
        q_l.append(qweight)
        s_l.append(scales)
    return torch.stack(q_l).contiguous(), torch.stack(s_l).contiguous()


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    a = torch.randn((M, K), device=device, dtype=dtype) / 10
    n_unique = 16
    w1 = torch.randn((n_unique, 2 * N, K), device=device, dtype=dtype) / 10
    w1_q, w1_s = stack_experts(w1)
    del w1
    w2 = torch.randn((n_unique, K, N), device=device, dtype=dtype) / 10
    w2_q, w2_s = stack_experts(w2)
    del w2
    reps = E // n_unique
    w1_q, w1_s = w1_q.repeat(reps, 1, 1), w1_s.repeat(reps, 1, 1)
    w2_q, w2_s = w2_q.repeat(reps, 1, 1), w2_s.repeat(reps, 1, 1)

    # several routing draws
    bytes_w1 = 2 * N * K * 0.5  # 4-bit
    print(f"W4 bytes/expert w1={bytes_w1/1e6:.2f} MB  w2={K*N*0.5/1e6:.2f} MB")
    print(f"{'draw':4s} {'uniq':>5} {'w1_MB':>8} {'roof_us':>8} {'moe_us':>8} {'if_w1_GB/s':>10}")
    qid = scalar_types.uint4b8.id
    for d in range(5):
        score = torch.softmax(torch.randn((M, E), device=device, dtype=torch.float32), dim=-1)
        topk_w, topk_ids = torch.topk(score, TOPK)
        topk_ids = topk_ids.to(torch.int32)
        uniq = int(topk_ids.unique().numel())
        w1_mb = uniq * bytes_w1 / 1e6
        roof = w1_mb / 273.0 * 1e3
        def moe():
            return fused_marlin_moe(
                a, w1_q, w2_q, None, None, w1_s, w2_s, topk_w, topk_ids, qid
            )
        for _ in range(3):
            moe()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(8):
            moe()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 8 * 1e6
        # fused = w1+w2+silu; w1 was 819/(819+468)=64% of marlin kernel time
        w1_us = us * 0.64
        gbs = w1_mb / (w1_us * 1e-3) if w1_us else 0
        print(f"{d:4d} {uniq:5d} {w1_mb:8.1f} {roof:8.1f} {us:8.1f} {gbs:10.1f}")

        sorted_ids, expert_ids, npost = moe_align_block_size(
            topk_ids, 8, E, None, False, True
        )
        n_blocks = int(npost.item()) // 8
        n_neg = int((expert_ids[:n_blocks] < 0).sum().item()) if n_blocks else 0
        print(f"     align blocks={n_blocks} expert_ids_neg={n_neg} npost={int(npost.item())}")

    # profiler: marlin kernel us only
    score = torch.softmax(torch.randn((M, E), device=device, dtype=torch.float32), dim=-1)
    topk_w, topk_ids = torch.topk(score, TOPK)
    topk_ids = topk_ids.to(torch.int32)
    for _ in range(4):
        fused_marlin_moe(a, w1_q, w2_q, None, None, w1_s, w2_s, topk_w, topk_ids, qid)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fused_marlin_moe(a, w1_q, w2_q, None, None, w1_s, w2_s, topk_w, topk_ids, qid)
        torch.cuda.synchronize()
    print("\nprofiler kernels")
    for e in sorted(prof.key_averages(), key=lambda x: -(getattr(x, "device_time_total", 0) or 0))[:8]:
        us = float(getattr(e, "device_time_total", 0) or 0)
        if us <= 0:
            continue
        print(f"  {us:10.1f}us n={e.count} {e.key[:120]}")


if __name__ == "__main__":
    main()
