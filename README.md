# GLM-5.3-Flash W4A16 + DFlash2 (2× DGX Spark)

2× NVIDIA DGX Spark (GB10 / SM121), TP=2에서 GLM-5.3-Flash를 W4A16 + DFlash2로 서빙하는 레시피.  
**이미지·가중치는 수정하지 않고**, 컨테이너에 overlay만 bind-mount 한다.

GOLDEN FINAL: `canada-quant/glm-5.3-w4a16-mtp` + 아래 KEEP overlay.  
협의 숫자·상태 레이블은 [`benchmarks/VERIFIED-MEASUREMENTS.md`](benchmarks/VERIFIED-MEASUREMENTS.md)가 기준이다.

---

## 스택

| 항목 | 값 |
|---|---|
| 베이스 모델 | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| 양자화 | [`canada-quant/glm-5.3-w4a16-mtp`](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp) (~178 GiB, 라우팅 MoE만 INT4, revision `4eeb77a…`) |
| 드래프터 | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), **`K=7` 고정** (다른 K는 부팅 wedge) |
| 이미지 | `radixark/vllm-glm53-flash:sm121-v11-dflash2` (alias `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`) |
| MoE | Marlin (`flashinfer_cutlass` 등은 W4A16에서 부팅 실패 또는 ~2× 손해) |
| KV | `fp8_e4m3`, **9 GiB pin** (`KV_CACHE_MEMORY=9663676416`), `max_model_len=1M` |
| 실행 | `--enforce-eager`, FlashInfer autotune OFF |

체크포인트·`config.json` 요약은 [`PARAMS.md`](PARAMS.md).

### 왜 overlay가 필요한가

- 스톡 vLLM은 이 W4A16 체크포인트를 그대로 못 올린다 (`gate_up_proj.weight` KeyError).
- DFlash2 checkpoint `selector_top_k=16`은 후보 풀이 좁다 → 생산은 `TOP_K=32`.
- 계측 프로브(`DFLASH2_ACC_PROBE`)는 GPU→CPU sync로 soak를 깎는다.
- 스톡 mHC TileLang warmup은 DeepSeek-V4만 워밍 → 첫 C2에서 `mhc_pre_big_fuse_with_norm_tilelang` JIT.
- 하이브리드 prefix-cache는 드래프터 SWA 때문에 hit가 0이 된다 → APC 패치.

---

## 패치 (이 레포 overlay)

`./launch-glm53-w4a16-tp2-dflash2.sh`가 아래를 컨테이너에 마운트한다. 이미지·가중치는 건드리지 않는다.

### 항상(또는 기본 ON)

| 파일 | 역할 |
|---|---|
| [`patches/glm5next_model.py`](patches/glm5next_model.py) | W4A16 dense MLP 로드. `quant_config=None` + `gate_proj`/`up_proj` → `gate_up_proj` |
| [`patches/qwen3_dflash2.py`](patches/qwen3_dflash2.py) | `DFLASH_SELECTOR_TOP_K`로 lm_head 후보 풀만 변경. `selector_rank=256`은 학습 차원 → **고정** |
| [`patches/dflash2_speculator.py`](patches/dflash2_speculator.py) | walk (`DFLASH_WALK_MODE=edge`). 승인율 프로브는 `DFLASH2_ACC_PROBE=0`이면 꺼짐 |
| [`patches/spec_decode_rejection_warmup.py`](patches/spec_decode_rejection_warmup.py) | DFlash `prepare_dflash_inputs` / `_copy_page_indices` 부팅 시 compile |
| [`patches/deepseek_v4_mhc_warmup.py`](patches/deepseek_v4_mhc_warmup.py) | GLM mHC TileLang warmup (스톡은 DSv4만 워밍) |
| [`docs/patch_hybrid_prefix_hit.py`](docs/patch_hybrid_prefix_hit.py) | 하이브리드 prefix-cache (APC). `APPLY_APC_PATCH=1` (기본). fail-closed |
| [`patches/sparse_attn_indexer_kpool.py`](patches/sparse_attn_indexer_kpool.py) | SM121 kpool indexer (Tony). `GLM53_SM121_MLA=0`일 때. `PATCH_KPOOL_HOST` 비우면 이 파일 |
| [`patches/chat_template_mm.jinja`](patches/chat_template_mm.jinja) | 비전 chat template (HF 체크포인트에 없음 → 없으면 vision 500) |

### 게이트 OFF (생산 기본값)

| 파일 / 스위치 | 상태 | 설명 |
|---|---|---|
| [`patches/gate_linear.py`](patches/gate_linear.py) / `APPLY_GATE_LINEAR=0` | `NOT_MEASURED` | GB10 MoE router cuBLAS overlay. 켜지 말 것 |
| [`docs/patch_sm121_mla.py`](docs/patch_sm121_mla.py) / `GLM53_SM121_MLA=0` | `PARTIAL` / 버림 | SM120 sparse MLA. decode 이득 없음, KV −21%. DFlash+MLA 부팅 실패 이력 |
| [`patches/hook_linear_8192.py`](patches/hook_linear_8192.py) | 조사용 | F.linear tracer. 생산 서빙에 쓰지 않음 |

부팅 로그에 `glm5next_model.py mounted`가 있어야 한다. 없으면 shard 0에서 KeyError.

---

## 유지 설정 (KEEP)

`.env.example` → `.env`에 고정. 자세한 레이블은 VERIFIED-MEASUREMENTS.

| knob | 값 |
|---|---|
| `DFLASH_TOKENS` | `7` (고정, A/B 금지) |
| `DFLASH_SELECTOR_TOP_K` | `32` |
| `DFLASH_WALK_MODE` | `edge` |
| `DFLASH2_ACC_PROBE` | `0` |
| `MOE_BACKEND` | `marlin` |
| `ENFORCE_EAGER` | `1` |
| `DISABLE_FLASHINFER_AUTOTUNE` | `1` |
| `APPLY_APC_PATCH` | `1` |
| `APPLY_GATE_LINEAR` | `0` |
| `GLM53_SM121_MLA` | `0` |
| `MAX_NUM_SEQS` | `6` |
| `MAX_NUM_BATCHED_TOKENS` | `8192` |
| `BLOCK_SIZE` | `2304` |
| `KV_CACHE_MEMORY` | `9663676416` (9 GiB pin) |
| `MAX_MODEL_LEN` | `1048576` |
| `ASYNC_SCHEDULING` | `1` |
| `CLOCK_MHZ` | `2400` (`clocks.sh`, start 후) |

**벤치 후 버린 것 (`MEASURED_DISCARD` 등):** FlashInfer autotune ON, CUDA graphs (`ENFORCE_EAGER=0`, 스톡 이미지), `WALK_MODE=unary`, `TOP_K=48`, 서빙 중 승인율 프로브 ON, live-temp rejection warmup, SM121 MLA overlay, **SM121 native 이미지** (`glm53-flash:sm121-native`, soak 32.3 vs 34.7–35.5), **Intel AutoRound 타깃** (accept 0.50 미달).  
`MODEL`을 AutoRound로 바꾸지 말 것.

---

## 벤치 결과

키트: 2× DGX Spark, TP=2, `temperature=0`, 첫 요청(부팅 warmup)은 soak에 넣지 않음. 부트 간 분산 약 ±15% — soak **+5% 이상**만 실이득으로 취급.  
원자료: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md), 레인 비교: [`benchmarks/COMPARISON.md`](benchmarks/COMPARISON.md) (W4A16 vs EXL3 vs NVFP4).

| 구성 | soak C1 | C1 | C6 | accept (`/metrics`) |
|---|---|---|---|---|
| top_k=16 (`final-recipe`) | 31.8 | ~30.4 | 81.1 | 0.418 |
| **top_k=32 + probe off (채택)** | **34.7–35.5** | **35.4** | **89.3** | **0.433–0.439** |
| probe ON | 32.1 | 32.2 | 84.7 | 0.435 |
| autotune ON / CUDA graphs | 33.2 / 33.0 | — | 79.0 / 79.6 | — |

추가 메모:

- reject_split mean (temp=0, edge+32): 약 **0.48**. 0.535는 미도달 추정치 → 게이트로 쓰지 않음.
- mHC warmup 후: `mhc_pre_big_fuse_with_norm_tilelang` **runtime JIT 0** (TP0·TP1). 첫 요청 `_rejection_kernel` Triton JIT는 남음.
- 콜드 프리필: ~1.3–1.6k tok/s, 300K까지 단일 스트림 OOM 없음.
- 콜드 부트: ~8분 (`init engine` ~87s, TileLang recompile 0, FlashInfer autotune skip).
- APC: shared prefix ≥ 2 full blocks (≥ 4,609 tok)부터 hit. block 4608은 0 hits → 2304 유지.
- C6는 부팅마다 ~79–89 밴드. keep 게이트는 soak C1 3-run median.

---

## 적용

**필요:** 2× DGX Spark, Docker+GPU, head→worker SSH(BatchMode), `/var/tmp` 약 **190 GiB**, CX7/RoCE GID는 `.env`의 `HEAD_*` / `WORKER_*`에 맞춤.

```bash
cp .env.example .env          # HEAD_IP / WORKER_IP (및 fabric)만 수정
./validate.sh                 # 가중치·오버레이·클러스터 게이트
./download.sh                 # 가중치 ~178 GiB + drafter ~2.3 GiB (없을 때)
./start.sh                    # pull → (필요 시 download) → rsync → TP=2 → /health → clocks
# 이미지가 로컬이면:
SKIP_PULL=1 ./start.sh restart
```

`start.sh`는 `.env`가 없으면 `.env.example`을 복사하고, 가중치가 없으면 `download.sh`를 호출한다. 그래도 최초에는 `validate` / `download`를 명시적으로 돌리는 것을 권장.

워커 먼저 수동 기동:

```bash
./launch-glm53-w4a16-tp2-dflash2.sh 1
sleep 25
./launch-glm53-w4a16-tp2-dflash2.sh 0
```

엔드포인트: `http://<HEAD_IP>:8000` (`/v1`, 모델명 `glm-5.3-flash`).

```bash
./status.sh                   # 또는 ./start.sh status
./stop.sh                     # 또는 ./start.sh stop
./start.sh logs               # head
./start.sh logs worker
```

스모크:

```bash
curl http://<HEAD_IP>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"2+2=?"}],"max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}'
```

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| [`benchmarks/VERIFIED-MEASUREMENTS.md`](benchmarks/VERIFIED-MEASUREMENTS.md) | KEEP/DISCARD 레이블, 생산 인용 숫자 (에이전트 핸드오프 기준) |
| [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) | A/B 원표, 콜드 프리필·부트·버린 결정 |
| [`benchmarks/COMPARISON.md`](benchmarks/COMPARISON.md) | W4A16 vs EXL3 vs NVFP4 |
| [`PARAMS.md`](PARAMS.md) | canada-quant config / recipe 스냅샷 |
| [`docs/sm121-mla-investigation.md`](docs/sm121-mla-investigation.md) | SM120 MLA overlay 조사 (생산 OFF) |
| [`docs/sm121-native-phase1.md`](docs/sm121-native-phase1.md) / [`docker/README.md`](docker/README.md) | SM121 native 빌드 — **생산 IMAGE로 쓰지 말 것** |
| [`benchmarks/sm121-audit/README.md`](benchmarks/sm121-audit/README.md) | SM121 프로파일·CUTLASS/Marlin 병목 조사 |

---

## 라이선스

이 레시피 코드는 MIT ([LICENSE](LICENSE)). 가중치·드래프터·이미지는 각 Hub/이미지 라이선스를 따른다. Zhipu AI, NVIDIA, upstream과 무관.

## 출처

- [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) — 베이스 모델
- [`canada-quant/glm-5.3-w4a16-mtp`](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp) — W4A16
- [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — DFlash2 드래프터
- [tonyd2wild](https://github.com/tonyd2wild) — SM121 이미지, kpool, chat template, fabric
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — APC / SM121 MLA 조사
- vLLM, FlashInfer
