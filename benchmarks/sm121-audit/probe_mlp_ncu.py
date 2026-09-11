#!/usr/bin/env python3
"""Single MLP-up launch for ncu DRAM counters. M=8, K=4096, N=12288, bf16."""
import torch
import torch.nn.functional as F

m, k, n = 8, 4096, 12288
a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
for _ in range(5):
    F.linear(a, w)
torch.cuda.synchronize()
F.linear(a, w)
torch.cuda.synchronize()
