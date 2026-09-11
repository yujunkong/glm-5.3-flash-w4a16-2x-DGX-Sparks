# SM121 architecture baseline (stock image)

Image: `radixark/vllm-glm53-flash:sm121-v11-dflash2`  
Date: 2026-09-09. Raw dump: `baseline.txt`.

| component | SM in binary |
|---|---|
| vLLM `_C` / `_moe` | sm_80, 87, 89, 90, 90a, 100, 100f, 110, 110f, 120, 120f — **no sm_121** |
| FA2 / FA3 | sm_80 / sm_90a |
| FlashInfer `.so` | none (no jit-cache; JIT on first use) |
| flashinfer_cubin (7.1 GB, 44876 files) | sm100 22208, sm107 17171, sm103 4832 — **no sm121** |

Cause: CUDA 13.0 `CMakeLists.txt` `CUDA_SUPPORTED_ARCHS` ends at `12.0`, so `TORCH_CUDA_ARCH_LIST=12.1a` is dropped (vLLM #43003). Marlin under CUDA 13 compiles `12.0f` family only.

Rebuild: `./docker/build-sm121.sh` → `IMAGE=glm53-flash:sm121-native` is **MEASURED_DISCARD** (soak 32.288). Do not set production IMAGE to native.

2026-09-10 P0: `phase1-flashinfer.txt`, `phase2-marlin.txt`, `cutlass-owner.txt`.
Hot 45% GEMM is `libcublasLt.so.13`, not FlashInfer/Marlin.
CUBLASLt probe: `cublaslt-probe.txt` / `probe_cublaslt.py` — decode F.linear is `algoId=21 tile=16x16` and LPDDR-bound. `gate_linear` is not the 45% fix.

2026-09-10 P4: `phase4-weight-path.txt`, `phase4-weight-path.md`, `phase4-conclusion.md`.
The 100.7 MB GEMM is native BF16 (drafter + dense layers 0–2), not W4 dequant. Marlin is experts only.

2026-09-10 P5: `phase5-dflash-mlp-calls.txt`.
M=8 = 1 bonus + 7 mask tokens, one DFlash forward. 100.7 MB gate_up runs 8×/step (DFlash 5 + target dense 0–2). That subset is ~3–4% CUDA, not the 45%. The 45% is the whole BF16 F.linear family. DFlash MLP cannot be skipped without dropping speculation.

2026-09-10 P6: `phase6-cutlass-decompose.txt`, `phase6-cutlass-shapes.csv`, `probe_cutlass_decompose.py`.
Largest identified GEMM is KDA `in_proj_qkvbfg_a` (34×, 102 MB, 11.4% CUDA). DFlash+target MLP-up 2.7%. 24.4% of CUDA attributed; 20.5% of the 45% bucket still unnamed.

2026-09-10 P7: `phase7-128x2-identify.txt`, `phase7-128x2-sweep.csv`.
~288 us 128x2 is F.linear M=8 N=8192 K=4096 (67 MB, LPDDR-bound). nsys Kernel2 311.7 us.
Live decode call site not captured (no TP=2 boot).

2026-09-10 P8: `phase8-linear-caller.txt`, `phase8-linear-hook-r0.json`.
Live TP=2 serve: **no** F.linear `(8192,4096)`. KDA q/k/v are fused `in_proj` N=12576 only.
Live N=8192 is MLA `q_b_proj` (8192,1536). Not duplicate KDA projection. Do not patch fusion.

2026-09-10 P7 live nsys: `phase7-live-cutlass-128x2.txt`, `phase7-conclusion.md`.
16x16_128x2 = **12.62% CUDA**, many F.linear. Largest: shared_experts.gate_up 6.03%. KDA in_proj is 128x1, not 128x2.

2026-09-10 P7 128x1 + apply/skip: `phase7-live-cutlass-128x1.txt`, `phase9-opt-decision.txt`,
`final-optimization-report.md`. 128x1 A-tier is only KDA in_proj 10.75% (LPDDR-bound).
No production patch; soak unchanged 34.7–35.5; 60 tok/s not reached.

2026-09-10 P10 Marlin rewrite: `phase10-marlin-rewrite.txt`, `probe_marlin_rewrite.py`.
Live w1/w2 attributed. In-image Marlin launch knobs ≤1% (noise). No soak.
