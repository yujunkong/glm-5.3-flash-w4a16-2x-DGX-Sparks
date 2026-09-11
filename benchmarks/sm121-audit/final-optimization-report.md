# Final optimization report — GLM-5.3-Flash 2× DGX Spark

Date: 2026-09-10  
Image: `radixark/vllm-glm53-flash:sm121-v11-dflash2` (production, unchanged)  
Serve: TP=2, DFlash2, `DFLASH_TOKENS=7`  
Harness: soak cites `benchmarks/VERIFIED-MEASUREMENTS.md` only. This nsys window is not a soak.

## Baseline tok/s

| source | decode tok/s | accept | label |
|---|---:|---:|---|
| soak band (stock + production knobs) | **34.7–35.5** | 0.433–0.439 | `MEASURED_KEEP` |
| selector-topk-32 soak median | **34.734** | 0.439 | canonical |
| tps-probe-off soak | **35.538** | — | band top |
| keep bar | **35.57** | — | +2.5% vs 34.7 |

nsys capture TPS is discarded.

## 최적화별 변경 내용

| 작업 | 대상 | live CUDA | 조사 | 변경 | soak |
|---|---|---:|---|---|---|
| 1 | 128x1 24.91% 분해 | — | 완료 (`phase7-live-cutlass-128x1.txt`) | 없음 | 없음 |
| 2 | KDA `in_proj_qkvbfg_a` | 10.75% | LPDDR 83–86%, workspace 무이득 | **SKIP** | — |
| 3 | shared_experts `gate_up_proj` | 6.03% | 동일 128x2 커널, live 209µs vs cold 78µs는 contention | **SKIP** | — |
| 4 | `lm_head` | 3.61% | 634 MB, 85% DRAM | **SKIP** | — |
| 5 | soak A/B | — | 적용 변경 없음 | — | 생략 |
| 6 | config sweep | — | MEASURED_KEEP 재스윕 금지 | 없음 | — |
| 10 | Marlin tile/atomic | 41.7% | 기존 cubin 스윕 최대 −1.1% | **SKIP** | — |

## 각 변경 전/후 CUDA % / tok/s

적용한 변경이 없으므로 전/후가 같다.

| | CUDA % | decode tok/s | accept |
|---|---:|---:|---:|
| baseline | Marlin 41.7 / 128x1 24.91 / 128x2 12.62 / NCCL 9.2 (nsys window) | 34.7–35.5 soak | 0.433–0.439 |
| after | 동일 | 동일 | 동일 |
| improvement | 0 | 0 | 0 |

## 최종 configuration

생산 `.env` 유지 (모두 기존 `MEASURED_KEEP` 또는 고정):

```text
IMAGE=radixark/vllm-glm53-flash:sm121-v11-dflash2
DFLASH_SELECTOR_TOP_K=32
DFLASH_WALK_MODE=edge
DFLASH2_ACC_PROBE=0
DFLASH_TOKENS=7
MOE_BACKEND=marlin
ENFORCE_EAGER=1
ASYNC_SCHEDULING=1
DISABLE_FLASHINFER_AUTOTUNE=1
APPLY_APC_PATCH=1
GLM53_SM121_MLA=0
APPLY_GATE_LINEAR=0
CUBLAS_WORKSPACE_CONFIG unset
```

## 유지한 변경

없음. (측정 후 적용할 최소 변경이 없었음)

## revert한 변경

없음. (`CUBLAS_WORKSPACE_CONFIG` 마이크로벤치만 했고 serve에 넣지 않음)

## 최종 병목

1. **Marlin W4 routed experts ~42% CUDA** — live w1 26.5% + w2 15.2%. 기존 cubin 타일/atomic은 ±2%. 새 SASS만 여지가 있고 sm_121a `_moe`는 이미 32.288로 짐.
2. **Eager BF16 `F.linear` / cuBLASLt `cutlass_80_wmma` ~45% CUDA** — 대부분 LPDDR weight read.
   - 최대 단일 op: KDA fused `in_proj` 10.75%, 103 MB, 456 µs, 이미 대역폭의 83–86%.
   - 다음 A-tier: shared_experts `gate_up` 6.03%, 16.8 MB, live 209 µs (isolated-cold 78 µs; Marlin과 L2 경쟁).
   - `lm_head` 3.61%, 634 MB, 85% DRAM.
3. **NCCL AllReduce ~9%** this window (이전 soak profile ~4%). 핵심 병목 증거 부족, 금지.
4. FlashInfer MLA ~0.4%. 금지.

허용된 수단(heuristic / workspace / layout / 기존 linear path)으로 in_proj+shared+lm_head를 DRAM roof까지 끌어올려도 **대략 +2–4% CUDA ≈ +1–1.5 tok/s**. keep bar 35.57을 넘기기 어렵고 60과는 무관.

CUTLASS 45%를 전부 0으로 만들어도 Marlin 42%가 남으면 이론 상한은 약 **1.8× ≈ 64 tok/s**. 그 상한은 금지된 Marlin/_C 재빌드 없이는 열리지 않는다.

## 60 tok/s 달성 여부

**아니오.** soak는 34.7–35.5 tok/s에 머문다. 이번 루프에서 e2e tok/s를 올린 패치는 없다.

## 산출물

- `phase7-live-cutlass-128x1.txt` — 128x1 모듈 분해
- `phase7-live-cutlass-128x2.txt` — 128x2 (기존)
- `phase9-opt-decision.txt` — 적용/스킵 근거
- `probe_live_shape_bw.py`, `probe_live_kernel_cold.py`, `parse_live_128x1.py`
- `phase10-marlin-rewrite.txt`, `probe_marlin_rewrite.py`
