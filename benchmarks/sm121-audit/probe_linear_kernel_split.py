#!/usr/bin/env python3
"""Count CUDA kernels per single F.linear launch (cutlass split?)."""
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile


def dump(m, n, k, tag):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(8):
        F.linear(a, w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        F.linear(a, w)
        torch.cuda.synchronize()
    print(f"\n==== {tag} F.linear M={m} N={n} K={k} ====")
    rows = []
    for e in prof.key_averages():
        us = float(
            getattr(e, "device_time_total", 0.0)
            or getattr(e, "cuda_time_total", 0.0)
            or 0.0
        )
        if us <= 0:
            continue
        rows.append((us, e.count, e.key[:100]))
    rows.sort(reverse=True)
    for us, ncall, key in rows[:12]:
        short = "128x2" if "128x2" in key else "128x1" if "128x1" in key else key[:80]
        print(f"  n={ncall} {us:8.1f}us  {short}")


def main():
    dump(8, 12480, 4096, "KDA_fused_in")
    dump(8, 8192, 4096, "KDA_q_unfused_full")
    dump(8, 4096, 4096, "KDA_q_tp_shard")
    dump(8, 12288, 4096, "MLP_gate_up")
    dump(8, 2048, 4096, "shared_gate_up")
    dump(8, 16384, 1536, "MLA_qb_full")
    dump(8, 8192, 1536, "MLA_qb_tp")


if __name__ == "__main__":
    main()
