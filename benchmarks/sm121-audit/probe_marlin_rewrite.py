#!/usr/bin/env python3
"""Sweep Marlin MoE launch knobs on the live decode shape.

Live nsys (TP=2): M=8 topk=8 E_local=144
  w1 gate_up  moe_wna16_marlin_gemm  ~819 us  26.5% CUDA
  w2 down     moe_wna16_marlin_gemm  ~468 us  15.2% CUDA
Default tile from kernel name: threads=128 thread_n=128 thread_k=64 block_m=8.

Does not load the 180G checkpoint. Does not change production.
"""
from __future__ import annotations

import time

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

# Live TP=2 shards.
M, TOPK, E, N, K, GROUP = 8, 8, 144, 1024, 4096, 128
ITERS, WARM = 12, 4


def stack_experts(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [E, out, in] bf16 to Marlin GPTQ-U4B8 packed weights+scales."""
    q_l, s_l = [], []
    perm = torch.arange(w.shape[-1], device=w.device)
    for i in range(w.shape[0]):
        # marlin_quantize wants [in, out]
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


def make_routing(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    score = torch.randn((M, E), device=device, dtype=torch.float32)
    score = torch.softmax(score, dim=-1)
    w, ids = torch.topk(score, TOPK)
    return w, ids.to(torch.int32)


def bench(fn, iters: int = ITERS, warm: int = WARM) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability())
    print(f"shape M={M} topk={TOPK} E={E} N={N} K={K} group={GROUP}")

    a = torch.randn((M, K), device=device, dtype=dtype) / 10
    # 16 unique experts, tiled to E=144. Ranking of launch knobs is
    # what we need; full 144 unique would only change L2, not tile choice.
    n_unique = 16
    print("quantize w1...")
    w1 = torch.randn((n_unique, 2 * N, K), device=device, dtype=dtype) / 10
    w1_q, w1_s = stack_experts(w1)
    del w1
    print("quantize w2...")
    w2 = torch.randn((n_unique, K, N), device=device, dtype=dtype) / 10
    w2_q, w2_s = stack_experts(w2)
    del w2
    reps = E // n_unique
    w1_q = w1_q.repeat(reps, 1, 1)
    w1_s = w1_s.repeat(reps, 1, 1)
    w2_q = w2_q.repeat(reps, 1, 1)
    w2_s = w2_s.repeat(reps, 1, 1)
    torch.cuda.empty_cache()
    topk_w, topk_ids = make_routing(device)
    qid = scalar_types.uint4b8.id

    def run_moe():
        return fused_marlin_moe(
            a, w1_q, w2_q, None, None, w1_s, w2_s, topk_w, topk_ids, qid
        )

    base = bench(run_moe)
    print(f"\nBASE fused_marlin_moe  {base:8.1f} us")

    orig = ops.moe_wna16_marlin_gemm
    cfgs = [
        ("atomic=1", dict(use_atomic_add=True)),
        ("fp32_reduce=0", dict(use_fp32_reduce=False)),
        ("atomic+fp16red", dict(use_atomic_add=True, use_fp32_reduce=False)),
        ("tk=128,tn=128", dict(thread_k=128, thread_n=128)),
        ("tk=64,tn=128", dict(thread_k=64, thread_n=128)),
        ("tk=128,tn=64", dict(thread_k=128, thread_n=64)),
        ("tk=128,tn=128,atomic", dict(thread_k=128, thread_n=128, use_atomic_add=True)),
        ("bpsm=2", dict(blocks_per_sm=2)),
        ("bpsm=4", dict(blocks_per_sm=4)),
        ("tk=128,tn=128,bpsm=4", dict(thread_k=128, thread_n=128, blocks_per_sm=4)),
        ("tk=128,tn=128,atomic,fp16", dict(thread_k=128, thread_n=128, use_atomic_add=True, use_fp32_reduce=False)),
    ]

    print(f"{'cfg':32s} {'us':>10} {'vs base':>8}")
    print(f"{'BASE default':32s} {base:10.1f} {'0.0%':>8}")
    best = ("BASE", base)
    for name, extra in cfgs:
        def wrapped(*args, _extra=extra, **kwargs):
            kwargs.update(_extra)
            return orig(*args, **kwargs)

        ops.moe_wna16_marlin_gemm = wrapped
        try:
            us = bench(run_moe)
            delta = 100.0 * (us - base) / base
            print(f"{name:32s} {us:10.1f} {delta:+7.1f}%")
            if us < best[1]:
                best = (name, us)
        except Exception as exc:  # noqa: BLE001 — sweep must continue
            print(f"{name:32s} FAIL {type(exc).__name__}: {exc}")
        finally:
            ops.moe_wna16_marlin_gemm = orig

    # block_size_m: force 16 by intercepting moe_align via wrapping fused loop is
    # harder; skip if no tile win. Report best.
    print(f"\nBEST {best[0]}  {best[1]:.1f} us  ({100*(best[1]-base)/base:+.1f}%)")
    if best[1] < base * 0.85:
        print("KEEP_CANDIDATE: >=15% fused-moe speedup")
    else:
        print("NO_KEEP: below 15% fused-moe speedup (would be <6% e2e)")


if __name__ == "__main__":
    main()
