# P7 live 128x2 — conclusion

Window: one TP=2 decode request (60 completion tokens). nsys Python CUDA backtrace on rank 0. Not a soak.

1. **Live 128x2 CUDA time:** 12.62% of CUDA kernels (1582 launches, 153.1 / 1213.3 ms). Median 14.4 µs ≠ mean 96.8 µs ≠ max 2.64 ms.

2. **Module/operation breakdown** (all 1582 launches stack-matched):
   - shared_experts.gate_up_proj 6.03%
   - lm_head 3.61% (DFlash 1.91% + target 1.71%)
   - MLA q_b_proj 0.87%
   - MLA fused_qkv_a_proj 0.59%
   - DFlash qkv_proj 0.46%
   - MoE router 0.41%
   - DFlash kernel_projection 0.37%
   - other MLA/DFlash/MoE tails ≤0.14% each
   - **KDA in_proj / q / k / v: 0% of 128x2**

3. **M / N / K:** M and K not in nsys kernel records. N estimated as `gridY×128` from this tile (lm_head 605×128=77440 checks vocab/TP=2). dtype bf16.

4. **calls/step** (≈9 decode-sized forwards in this request): shared gate_up ~39, MLA ops ~10, DFlash qkv ~5, lm_head ~2.

5. **Largest actual 128x2 operation:** `shared_experts.gate_up_proj` at **6.03% CUDA**, ~209 µs, ~39 calls/step.

6. **Bandwidth-bound?** **Not measured** (no ncu DRAM/L2). Do not reuse isolated 8192×4096 GB/s.

7. **Optimize this kernel?** **No.** 128x2 is a mixed F.linear bucket (12.6%), not a single op and not the P6 20.5% leftover story. Largest piece (6.03%) is still smaller than identified KDA fused in_proj (128x1). No fusion/tile/algorithm/FI/Marlin/NCCL/`_C` patch.

8. **Next to measure:** live **16x16_128x1 (24.91% CUDA this window)** with the same Python-BT mapping. FI / Marlin / NCCL / `_C` stay parked.
