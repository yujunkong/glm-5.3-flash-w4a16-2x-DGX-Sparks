#!/usr/bin/env python3
"""W4A8-FP8 Marlin vs W4A16 on the live decode MoE shape.

SM12x is the documented W4A8-FP8 target. If this is >=15% faster, it is a
keep-candidate (env VLLM_MARLIN_INPUT_DTYPE=fp8) — not a cubin rebuild.
"""
from __future__ import annotations

import time

import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

M, TOPK, E, N, K, GROUP = 8, 8, 144, 1024, 4096, 128
N_UNIQUE = 16


def stack_experts(w: torch.Tensor, input_dtype) -> tuple:
    q_l, s_l = [], []
    perm = torch.arange(w.shape[-1], device=w.device)
    for i in range(w.shape[0]):
        _, qweight, scales, *_ = marlin_quantize(
            w[i].transpose(1, 0).contiguous(),
            scalar_types.uint4b8,
            GROUP,
            False,
            perm,
            input_dtype=input_dtype,
        )
        q_l.append(qweight)
        s_l.append(scales)
    return torch.stack(q_l).contiguous(), torch.stack(s_l).contiguous()


def bench(fn) -> float:
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(12):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 12 * 1e6


def run(tag: str, input_dtype) -> float:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    a = torch.randn((M, K), device=device, dtype=dtype) / 10
    w1 = torch.randn((N_UNIQUE, 2 * N, K), device=device, dtype=dtype) / 10
    w1_q, w1_s = stack_experts(w1, input_dtype)
    del w1
    w2 = torch.randn((N_UNIQUE, K, N), device=device, dtype=dtype) / 10
    w2_q, w2_s = stack_experts(w2, input_dtype)
    del w2
    reps = E // N_UNIQUE
    w1_q, w1_s = w1_q.repeat(reps, 1, 1), w1_s.repeat(reps, 1, 1)
    w2_q, w2_s = w2_q.repeat(reps, 1, 1), w2_s.repeat(reps, 1, 1)
    score = torch.softmax(torch.randn((M, E), device=device, dtype=torch.float32), dim=-1)
    topk_w, topk_ids = torch.topk(score, TOPK)
    topk_ids = topk_ids.to(torch.int32)
    qid = scalar_types.uint4b8.id

    def moe():
        return fused_marlin_moe(
            a, w1_q, w2_q, None, None, w1_s, w2_s, topk_w, topk_ids, qid,
            input_dtype=input_dtype,
        )

    us = bench(moe)
    print(f"{tag:12s} {us:8.1f} us")
    return us


def main() -> None:
    print("device", torch.cuda.get_device_name(0))
    a16 = run("W4A16", None)
    a8 = run("W4A8-FP8", torch.float8_e4m3fn)
    print(f"W4A8 vs W4A16: {100*(a8-a16)/a16:+.1f}%")


if __name__ == "__main__":
    main()
