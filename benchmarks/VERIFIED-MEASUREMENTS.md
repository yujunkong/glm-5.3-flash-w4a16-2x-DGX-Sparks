# Verified measurements — agent handoff

**이 문서가 협의 기준이다.** `RESULTS.md` 요약, 채팅의 “이득 없다”, 빌드 로그를 섞지 말 것.  
상태 레이블만 사용한다. 레이블이 없는 숫자는 인용하지 말 것.

작성: 2026-09-09. 클러스터: head `192.168.100.10` + worker `192.168.100.20`, 2× GB10 (SM121), TP=2.  
**GOLDEN FINAL (2026-09-10):** `canada-quant/glm-5.3-w4a16-mtp` @ `4eeb77a3499fc2503290197118102eae2ed44553`. Drafter: `incoai/GLM-5.3-Flash-DFlash2`. `DFLASH_TOKENS=7` (다른 K는 부팅 wedge). Intel AutoRound는 아래 B6 — accept 0.50 미달.

---

## 상태 레이블 (필수)

| 레이블 | 의미 | 허용되는 결론 |
|---|---|---|
| `MEASURED_KEEP` | 동일 하네스로 soak/C1/C6/accept/P1이 파일로 남음. 채택됨 | 생산 노브로 유지 |
| `MEASURED_DISCARD` | 동일 하네스로 졌거나 부팅 실패가 재현됨 | 그 노브는 스톡 이미지 기준으로 버림 |
| `BUILD_ONLY` | 컴파일/cuobjdump만 됨. serve TPS 없음 | **이득/손해 결론 금지** |
| `NOT_MEASURED` | A/B가 한 번도 끝나지 않음 | 결론 금지 |
| `PARTIAL` | 관련 A/B는 있으나 가설과 조건이 다름 | 문서에 적힌 조건 안에서만 말함 |

**금지:** `BUILD_ONLY`/`NOT_MEASURED`를 “이득 없다”로 말하지 말 것.  
**금지:** C6 한 웨이브로 keep/discard. C6는 부팅마다 ~79–89. soak C1 3-run median만 keep 게이트.

---

## 피어 평가에 대한 동의/수정

다른 에이전트 평가에 대해:

1. **동의.** CUDA graphs / autotune ON / unary / TOP_K=48 / ACC_PROBE ON 은 `MEASURED_DISCARD`. 스톡 이미지 (`radixark/vllm-glm53-flash:sm121-v11-dflash2`) 위에서 재었다.
2. **동의 + 조건.** SM121 MLA overlay (`GLM53_SM121_MLA=1`) 는 decode 이득 없음으로 버렸다. 단 A/B는 **스펙 없음(no DFlash)** 이었다. DFlash + SM120 MLA 부팅은 glm5n KV 과금 버그로 실패 (`PARTIAL`).
3. **동의.** DFlash `TOP_K=32` / `WALK_MODE=edge` / `ACC_PROBE=0` 은 `MEASURED_KEEP`. 현재까지 가장 확실한 싱글스트림 이득.
4. **수정 (2026-09-09 재측정).** SM121 네이티브 커널은 이제 serve TPS가 있다. soak **32.288** vs 생산 밴드 **34.734–35.538** → `MEASURED_DISCARD`. 이전 ENTRYPOINT=sleep 실패와 혼동하지 말 것.
5. **추가 정정.** Graphs discard는 **스톡 이미지**에서만 측정됨. 네이티브 커널 위에서의 `ENFORCE_EAGER=0` 은 여전히 `NOT_MEASURED`. `APPLY_GATE_LINEAR=1` 도 `NOT_MEASURED`.

---

## 하네스 (모든 숫자 공통)

스크립트: `bench/bench_config.sh <label>` → `benchmarks/<label>/`.

1. warmup 1회 **버린다** (부팅 직후 1st request).
2. soak: `bench_decode.py` 3× C1, `max_tokens=512`, streaming. 인용값은 `soak.json`의 `decode_tok_s_median`.
3. C1/C2/C6: `bench_c.py`, 512 tok, `aggregate_tok_s`.
4. acceptance: `/metrics` `accepted/drafted` (`acceptance.txt`).
5. P1: temp=0, thinking off. 내용에 `391` 과 `Tokyo` 필수. keep 게이트.
6. 프롬프트: 코드 워크로드, `temperature=0`.
7. 노이즈: 싱글 웨이브 ±~15% 부팅 편차. **soak +5% 이상만 실이득으로 취급** (RESULTS.md). 이후 무인 A/B keep 바는 soak **+2.5% vs 당시 best** (`docker/unattended-onward.sh`, `BASELINE_SOAK=34.7` → keep 임계 ≈ **35.57**).

Soak 공식: `decode_tok_s` = completion 512 / (total_s − 대략 TTFT가 포함된 total; 파일의 `decode_tok_s` 필드를 그대로 쓴다). 재계산하지 말고 JSON을 인용.

---

## 현재 생산 설정 (`.env`, 스톡 이미지)

이미지: `IMAGE=radixark/vllm-glm53-flash:sm121-v11-dflash2`  
(2026-09-09 무인 스크립트가 잠깐 `glm53-flash:sm121-native` 로 바꿨다가, serve 실패 후 스톡으로 되돌림. 네이티브 이미지는 생산이 아니다.)

| knob | 값 | 상태 |
|---|---|---|
| `DFLASH_TOKENS` | 7 | 고정 (부팅 제약, A/B 금지) |
| `DFLASH_SELECTOR_TOP_K` | 32 | `MEASURED_KEEP` |
| `DFLASH_WALK_MODE` | edge | `MEASURED_KEEP` |
| `DFLASH2_ACC_PROBE` | 0 | `MEASURED_KEEP` |
| `MOE_BACKEND` | marlin | `MEASURED_KEEP` (다른 백엔드 부팅 실패 또는 ~2× 손해) |
| `ENFORCE_EAGER` | 1 | `MEASURED_DISCARD` of graphs **on stock image** |
| `DISABLE_FLASHINFER_AUTOTUNE` | 1 | `MEASURED_DISCARD` of autotune ON |
| `ASYNC_SCHEDULING` | 1 | 생산 유지 (별도 재측정 파일 없음, 기존 레시피) |
| `MAX_NUM_SEQS` | 6 | `MEASURED_KEEP` vs 12 |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | `MEASURED_KEEP` vs 16384 |
| `BLOCK_SIZE` | 2304 | 4608은 prefix 0 hits로 되돌림 |
| `KV_CACHE_MEMORY` | 9663676416 (9 GiB) | unpin은 1M 부팅 실패 |
| `APPLY_APC_PATCH` | 1 | prefix hits 측정됨 |
| `GLM53_SM121_MLA` | 0 | `PARTIAL` / 생산 유지 SM90 |
| `APPLY_GATE_LINEAR` | 0 | `NOT_MEASURED` |
| `CLOCK_MHZ` | 2400 | 생산 (이 문서의 A/B 테이블 밖) |

**생산 성능 인용 (스톡 + 위 overlay):**

- soak 밴드: **34.7–35.5 tok/s** (두 부팅, 아래 표).
- C6: **89.3** 은 `selector-topk-32` 한 웨이브. 같은 overlay의 다른 부팅 C6는 **78.9**. C6를 89.3으로 고정 인용하지 말 것.
- `/metrics` accept: **0.433–0.439**.

60 tok/s 싱글스트림은 **달성하지 못했다.** 미달 원인이 “대역폭 ceiling”인지는 이 문서로 증명하지 않는다. 가설일 뿐.

---

## A. MEASURED_KEEP — 파일과 숫자

공통 고정(challenger만 다름): 스톡 이미지, marlin, eager, autotune OFF, seqs=6, batched=8192, K=7, APC ON, MLA=0.

### A1. `DFLASH_SELECTOR_TOP_K=32` vs checkpoint 16

| | TOP_K=16 `final-recipe` | TOP_K=32 `selector-topk-32` |
|---|---|---|
| soak median | **31.816** (29.061 / 31.816 / 34.772) | **34.734** (36.500 / 34.019 / 34.734) |
| C1 agg | ~30.4 (RESULTS; 첫 웨이브 36.53 outlier) | **35.39** (`c1.json`) |
| C6 agg | **81.13** | **89.28** |
| `/metrics` accept | **0.418** (4965/11886) | **0.439** (5024/11438) |
| P1 | 391 / Tokyo | 391 / Tokyo |

파일:

- `benchmarks/final-recipe/soak.json`, `c6.json`, `acceptance.txt`
- `benchmarks/selector-topk-32/soak.json`, `c1.json`, `c6.json`, `acceptance.txt`, `p1.txt`

Soak +9.1% (31.816 → 34.734). 이게 keep의 근거.  
C6 81.13 → 89.28 은 **같은 부팅 한 웨이브**라 단독 keep 근거로 쓰지 말 것. 이후 부팅에서 C6가 78.9까지 떨어짐.

### A2. `DFLASH_WALK_MODE=edge` vs `unary`

이건 soak가 아니라 **reject_split 10-run mean accept** (temp=0, 256 tok).

| | unary `unary-walk-k32` | edge `topk32-walk-debug` |
|---|---|---|
| mean accept | **0.4597** (median 0.4602) | **0.482** (median 0.4599) |
| first-reject A/B/C | A 8.9% / B 91.1% / C 0% | A 11.9% / B 88.1% / C 0% |

파일: `benchmarks/unary-walk-k32/reject_split.json`, `benchmarks/topk32-walk-debug/reject_split.json`.

mean 0.460 vs 0.482. median은 둘 다 ~0.46 — mean 차로 버렸다. soak TPS A/B는 이 쌍에 없음. 생산은 edge.

### A3. `DFLASH2_ACC_PROBE=0` vs 1 (서빙 중 프로브)

동일 overlay (TOP_K=32, edge). 프로브는 GPU→CPU `.cpu()` 카피.

| | probe ON `tps-probe-on` | probe OFF `tps-probe-off` |
|---|---|---|
| soak median | **32.148** (32.148 / 35.901 / 31.575) | **35.538** (35.538 / 32.085 / 37.652) |
| C1 agg | 32.2 (RESULTS) | 32.27 (`c1.json`) |
| C6 agg | **84.67** | **78.93** |
| accept | 0.435 | 0.433 |

파일: `benchmarks/tps-probe-on/{soak,c6,acceptance}.json|txt`, `benchmarks/tps-probe-off/{soak,c1,c6,acceptance}.json|txt`.

Soak −10% (32.1 vs 35.5) 로 ON discard. C6는 ON이 더 높음(84.7 vs 78.9) — **C6로 뒤집지 말 것.** 게이트는 soak.

`34.7` 과 `35.5` 는 둘 다 probe-off + TOP_K=32 의 **다른 부팅**.  knobs를 더 켠 결과가 아니다.

### A4. `MOE_BACKEND=marlin`

- `flashinfer_cutlass`: 부팅 `ValueError` (WNA16 미지원).
- `humming`: 부팅 `TypeError` (checkpoint 스키마).
- `flashinfer_trtllm`: 부팅 `ValueError` (device/kernel).
- `triton`: 부팅·P1 OK, C1 10.80 vs marlin 19.97 (−46%), C6 22.0 vs 78.9.

출처: `benchmarks/RESULTS.md` § MoE. 원본 웨이브 파일은 RESULTS 서술. 재측정 없이 marlin 유지.

### A5. `MAX_NUM_BATCHED_TOKENS=8192` vs 16384

decode-under-load (1024 decode + ~125k cold prefill, 3 waves): mean **11.91 vs 11.01 tok/s**. 정상 prefill 거의 tie. `RESULTS.md` § batched tokens.

### A6. `MAX_NUM_SEQS=6` vs 12

C12가 C6 ~80 tok/s를 넘지 못함. 에이전트 롱컨텍스트 KV 보존 위해 6 유지. `RESULTS.md`.

### A7. APC (`APPLY_APC_PATCH=1`)

스톡 prefix: 0 hits. 패치 후 identical resend: 7.2k에서 63.7% hits, 21.7k에서 84.9%. 블록 4608은 0 hits → 2304. `RESULTS.md` § Prefix cache.

---

## B. MEASURED_DISCARD — 파일과 숫자 (스톡 이미지)

모두 TOP_K=32 overlay 이후, 스톡 `sm121-v11-dflash2`.

### B1. CUDA graphs `ENFORCE_EAGER=0`

디렉터리: `benchmarks/enforce-eager-0/`.

| | eager `selector-topk-32` | graphs `enforce-eager-0` |
|---|---|---|
| soak median | **34.734** | **32.970** (31.962 / 32.970 / 34.292) |
| C6 | 89.28 | **79.62** |
| accept | 0.439 | 0.433 |
| P1 | 391 / Tokyo | 391 / Tokyo (`p1.txt`) |

Soak −5.1% vs 34.734. **스톡에서 discard.**  
네이티브 `_C` 위 graphs는 `NOT_MEASURED`.

### B2. FlashInfer autotune ON

디렉터리: `benchmarks/flashinfer-autotune-on/`.

| soak median | **33.188** (36.276 / 33.188 / 31.592) |
| C6 | **79.01** |
| accept | **0.400** |

부팅 ~50s, 로그 **Saved 0 configs**. soak 33.2 vs 34.7. discard. `DISABLE_FLASHINFER_AUTOTUNE=1` 유지.

### B3. `DFLASH_SELECTOR_TOP_K=48`

`benchmarks/edge-walk-k48/reject_split.json`: mean accept **0.4785** vs TOP_K=32의 **0.482**. A% 12→9 수준. soak 파일 없음. discard.

### B4. live-temp rejection warmup

`benchmarks/tps-warmup-live-temp/soak.json`: median **32.832** vs probe-off **35.538**. 첫 요청 Triton JIT 여전. 되돌림.

### B5. mHC warmup overlay (TPS)

`benchmarks/tps-mhc-warmup/`: RESULTS 표에 soak **30.1**, C6 76.8, accept 0.390. TPS는 손해.  
“keep”은 TileLang mHC JIT를 부팅 워밍으로 옮긴 것(추론 중 0 JIT)이지 soak 이득이 아니다. soak 게이트로는 discard.

### B6. Intel AutoRound vs canada-quant (동일 하네스, 2026-09-10)

목표: `/metrics` accept ≥ **0.50** 이면 Intel을 최종으로 교체.  
이미지/노브/DFlash2 K=7 동일. 타깃만 `Intel/GLM-5.3-Flash-W4A16-AutoRound` (`5eee1846…`), `quant_method=auto-round` / `auto_round:auto_gptq`.  
디렉터리: `benchmarks/intel-autoround/`. P1 391 / Tokyo.

| | canada-quant 생산 밴드 | Intel AutoRound |
|---|---|---|
| soak median | **34.7–35.5** | **35.434** (37.092 / 35.434 / 31.014) |
| `/metrics` accept | **0.433–0.439** | **0.432** (5013/11606) |
| C6 agg | 한 웨이브 78.9–89.3 | **81.47** |

accept 0.432 < 0.50, soak도 생산 밴드 안. **`MEASURED_DISCARD`.** 생산 타깃은 canada-quant 유지.

---

## C. PARTIAL — SM121 MLA overlay (`GLM53_SM121_MLA`)

문서: `docs/sm121-mla-investigation.md`.  
하는 일: SM90 `FLASHINFER_MLA_SPARSE_SM90` → SM120 packed `FLASHINFER_MLA_SPARSE_SM120` + NoPE 패드. **W4A16/Marlin 안 건드림.**

**재어진 것 (2026-09-03, `SPEC_METHOD=none`, 동일 1M+9GiB pin):**

| | SM90 no-spec `sm90-nospec` | SM120 no-spec `sm120-mla-nospec` |
|---|---|---|
| soak C1 | **14.205** | **14.303** (1.007×) |
| C6 | 54.68 | 54.81 |
| P1 | 391 / Tokyo | byte-identical |
| KV pool | 1,544,105 | 1,221,126 (**−20.9%**) |

Decode 이득 없음, 용량 −21%. 생산 `GLM53_SM121_MLA=0`.

**재지 못한 것:** SM120 + DFlash. drafter 그룹이 glm5n `per_block`에 과금되어 ~40GB KV 요구로 부팅 실패. Step 4 미완.

DFlash 있는 생산 경로에서 MLA overlay TPS는 **없다.** no-spec 커널 패리티만 있다.

---

## D. BUILD_ONLY / NOT_MEASURED — SM121 네이티브 커널 (Phase 2/3)

**이게 아직 가장 큰 미지수다. TPS 결론을 내지 말 것.**

목표: 스톡 이미지는 `_C`/`_moe`에 **sm_121이 없다** (CUDA 13.0 `CUDA_SUPPORTED_ARCHS`가 12.0에서 끝, `TORCH_CUDA_ARCH_LIST=12.1a` drop). 재컴파일 후 같은 생산 노브로 soak A/B.

### D1. 스톡 arch (`BUILD_ONLY` 감사)

파일: `benchmarks/sm121-audit/baseline.txt` (2026-09-09).

- `_C` / `_moe`: sm_80,87,89,90,90a,100,100f,110,110f,120,120f. **sm_121 없음.**
- FA2 / FA3: sm_80 / sm_90a.
- flashinfer cubin 44876: sm100/107/103. **sm121 없음.**

### D2. FlashInfer AOT (`BUILD_ONLY`, 컴파일 성공)

로그: `/var/tmp/sm121-build/logs/flashinfer-cuobjdump.txt`.

- `.so` 261개, **sm_121a 260, non-sm121 0.**
- 산출물: `/var/tmp/sm121-build/fi-aot`. 캐시 `/var/tmp/sm121-build/fi-build` 지우지 말 것.

**Serve에서 이 커널이 쓰이는지, TPS 차이인지는 미측정.**

### D3. vLLM CUDA 재컴파일 (`BUILD_ONLY`)

- worktree: `/var/tmp/sm121-build/vllm-src` @ `487ecf187d3dfe74d2cf6119a92881dba403c219`. 유저 clone `docker/vllm` main 을 checkout 하지 말 것.
- `VLLM_BUILD_OK 2026-09-09T12:17:33+00:00`.
- 이미지 태그: `glm53-flash:sm121-native`.
- 사후 감사: `/var/tmp/sm121-build/logs/verify-after.txt`.

네이티브 `.so` arch (STRICT **실패** — sm_121-only가 아님):

| .so | arch |
|---|---|
| `_C_stable_libtorch.abi3.so` | **sm_121a**, sm_80, sm_90 |
| `_moe_C_stable_libtorch.abi3.so` | **sm_121a**, sm_80 |
| `_flashkda_C`, `_qutlass_C` | sm_121a |
| `_flashmla_*` | sm_100f, sm_90a (12.1a에서 FlashMLA skip, 스톡 잔존) |
| FA2 / FA3 | sm_80 / sm_75 |

**sm_121a 코드가 바이너리에 들어간 것은 확인됨.** sm_80/90 잔존. STRICT 실패가 TPS 실패는 아니다. TPS는 아래 D4.

### D4. Serve / soak — **MEASURED_DISCARD** (2026-09-09)

서빙 태그: `glm53-flash:sm121-native-serve` (`ENTRYPOINT ["vllm","serve"]`).  
sleep 태그 `glm53-flash:sm121-native` 는 쓰지 말 것.  
A/B: `ENV_FILE=.env.sm121-native SKIP_PULL=1 SKIP_SYNC=1 ./start.sh restart` — 생산 `.env` `IMAGE`는 스톡 유지.

파일: `benchmarks/sm121-native/`.

| 메트릭 | native-serve | 생산 밴드 (cite JSON) |
|---|---|---|
| soak median | **32.288** (33.682 / 29.432 / 32.288) | 34.734 (`selector-topk-32`) · 35.538 (`tps-probe-off`) |
| soak mean | 31.801 | 35.084 · 35.092 |
| C6 aggregate | 80.71 | 노이즈 78.93–89.28. keep 게이트 아님 |
| `/metrics` accept | 4908/12264 = **0.400** | ~0.43–0.44 |
| P1 | 391 / Tokyo | pass |

vs 35.538: **−9.1%**. vs 34.734: **−7.0%**. keep 바(+1%) 미달. 생산 `IMAGE` 스톡.

엔진 시그니처 동일: `Using 'MARLIN' WNA16`, `MarlinExperts`, KV 1,335,594. `_moe` cubin sm_121a×29 + sm_80×2. FI AOT `.so` 261.

남은 `NOT_MEASURED` (네이티브가 져서 후속 A/B 안 함): 네이티브 위 autotune / gate_linear / graphs.

### D5. 같은 캠페인의 다른 `NOT_MEASURED`

| 항목 | 이유 |
|---|---|
| `APPLY_GATE_LINEAR=1` (`patches/gate_linear.py`, GB10 cuBLAS out_dtype) | 네이티브 부팅 전 순번, 미실행 |
| 네이티브 위 `DISABLE_FLASHINFER_AUTOTUNE=0` | 미실행 |
| 네이티브 위 `ENFORCE_EAGER=0` | 미실행. 스톡 graphs discard와 혼동 금지 |
| 언어별 accept 매트릭스 (한/영/코드) | `acceptance-report.md`: 코드 하네스만. 네이티브 이후 재실행 예정으로 미완 |

---

## E. 한 장 요약 테이블

| 항목 | 상태 | 숫자 (파일) | 생산 |
|---|---|---|---|
| TOP_K 16→32 | `MEASURED_KEEP` | soak 31.816→34.734; accept 0.418→0.439 | 32 |
| walk unary vs edge | `MEASURED_KEEP` | reject_split mean 0.460 vs 0.482 | edge |
| ACC_PROBE 1 vs 0 | `MEASURED_KEEP` | soak 32.148 vs 35.538 | 0 |
| CUDA graphs (stock) | `MEASURED_DISCARD` | soak 32.97 vs 34.73; C6 79.6 vs 89.3 | eager |
| autotune ON (stock) | `MEASURED_DISCARD` | soak 33.19; 0 configs saved | OFF |
| TOP_K=48 | `MEASURED_DISCARD` | mean accept 0.479 vs 0.482 | 32 |
| SM121 MLA + DFlash | `PARTIAL` | no-spec soak 14.21 vs 14.30, pool −21%; DFlash 미부팅 | MLA=0 |
| **SM121 native serve TPS** | **`MEASURED_DISCARD`** | soak 32.288 vs 34.734–35.538 (`benchmarks/sm121-native/soak.json`) | 스톡 이미지 |
| Intel AutoRound W4A16 | `MEASURED_DISCARD` | soak 35.434; accept **0.432** < 0.50 (`benchmarks/intel-autoround/`) | canada-quant |
| FlashInfer AOT sm_121a | `BUILD_ONLY` | 261 .so, native-serve에 포함. TPS는 위 discard | 미적용 |
| vLLM `_C` sm_121a | `BUILD_ONLY` | sm_121a+sm_80+sm_90. TPS는 위 discard | 미적용 |
| gate_linear | `NOT_MEASURED` | — | 0 |

---

## F. 다음 에이전트가 할 일 (순서)

0. **이 스택은 최종 동결.** 생산 = canada-quant + 스톡 이미지 + 위 노브. Intel AutoRound는 B6 discard. 커널/노브 재튜닝으로 60 tok/s는 안 난다.
1. SM121 native-serve는 `MEASURED_DISCARD`. `.env` `IMAGE`를 네이티브로 바꾸지 말 것.
2. FlashInfer/Marlin 재빌드 반복 금지. FI MLA ~0.4%; Marlin은 이미 sm_120 cubin. 기록: `benchmarks/sm121-audit/phase1-flashinfer.txt`, `phase2-marlin.txt`.
3. 45% `cutlass_80_wmma` 소유자는 `libcublasLt.so.13.1.1.3` (Case B, eager ATen→cuBLASLt). 파일: `benchmarks/sm121-audit/cutlass-owner.txt`. 다음 최소 A/B는 cublasLt heuristic/로그 또는 GEMM 교체. vLLM `_C` 재빌드로는 안 잡힘.
4. NCCL은 이번 명세에서 보류. git commit하지 말 것. `/var/tmp/sm121-build` 지우지 말 것. 유저 `vllm` clone을 487ecf187로 옮기지 말 것.

---

## G. 인용 규칙

- soak는 항상 `*.json`의 `decode_tok_s_median` + 3 raw runs.
- C6는 `aggregate_tok_s` + 디렉터리명. “C6 89.3이 안정 생산치”라고 쓰지 말 것.
- accept는 `/metrics` 와 reject_split mean을 섞지 말 것. `/metrics` ~0.43–0.44, reject_split mean ~0.48.
- “이득 없다”는 `MEASURED_DISCARD`에만. SM121 native serve TPS는 `MEASURED_DISCARD` (32.288). 예전 sleep 부팅 실패와 섞지 말 것.
