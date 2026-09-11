# Live CUTLASS 16x16_128x2 attribution (TP=2 serve)

Date: 2026-09-10  
Image: `radixark/vllm-glm53-flash:sm121-v11-dflash2`  
Profiler: Nsight Systems 2025.3.2, `--python-backtrace=cuda` on rank 0  
Request: 28 prompt + 60 completion tokens, `enable_thinking=false`  
This run is **not a soak**; do not cite its tok/s.

## CUDA share

Live `cutlass_80_wmma` **16x16_128x2** is **12.62% of CUDA kernel time** in this window (1582 launches, 153.1 ms of 1213.3 ms).

It is **not** one GEMM: median 14.4 µs vs mean 96.8 µs vs max 2636 µs.

## Module breakdown (Python backtrace → `F.linear` / Linear.apply)

| %CUDA | calls | ~calls/step | avg µs | operation | N from grid×128 |
|------:|------:|------------:|-------:|-----------|----------------:|
| 6.03 | 351 | ~39 | 209 | shared_experts **gate_up_proj** (`model.py:147` ← `shared_experts.py`) | 2048 |
| 3.61 | 17 | ~2 | 2580 | **lm_head** (`vocab_parallel_embedding.py:75`) | 77440 |
| 0.87 | 92 | ~10 | 114 | MLA **q_b_proj** (`mla.py:213`) | 8192 |
| 0.59 | 92 | ~10 | 78 | MLA **fused_qkv_a_proj** (`mla.py:175`) | 2048 |
| 0.46 | 45 | ~5 | 124 | DFlash **qkv_proj** (`qwen3_dflash.py:256`) | 3072 |
| 0.41 | 351 | ~39 | 14 | MoE **router gate** (`model.py:270`) | — |
| 0.37 | 90 | ~10 | 50 | DFlash **kernel_projection** | 1024 |
| 0.14 | 351 | ~39 | 5 | MoE gate (runner shared path) | — |
| 0.07 | 92 | ~10 | 9 | MLA indexer **wk_weights_proj** | 256 |
| 0.06 | 92 | ~10 | 7 | MLA **kpool gate** `F.linear` | 128 |
| 0.01 | 9 | ~1 | 13 | DFlash **hidden_projection** | 256 |

lm_head split: 9 DFlash candidate GEMMs (23.1 ms) + 8 target `compute_logits` (20.7 ms).

M and K were **not** present in CUPTI kernel records. dtype = bf16 (kernel name). Weight shape = `(N_est, K_unknown)` for Column/Merged linears.

## Not in 128x2

KDA fused `in_proj_qkvbfg_a` and KDA q/k/v Linears. Those are not this tile (128x1 family is 24.91% here).

## DRAM / L2 / AI

Not collected (nsys BT pass only). Isolated microbench bandwidth is not applied.

## Optimization

No patch this phase. Do not optimize “the 128x2 kernel”. Largest confirmed 128x2 op is **shared_experts.gate_up_proj at 6.03% CUDA**.
