# GLM-5.3-Flash W4A16 + DFlash2 (2× DGX Spark)

2× NVIDIA DGX Spark (GB10 / SM121), TP=2에서 GLM-5.3-Flash를 W4A16 + DFlash2로 서빙하는 레시피. 이미지·가중치는 그대로 두고, 컨테이너에 overlay만 bind-mount 한다.

## 배경

- 모델: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
- 양자화: [`canada-quant/glm-5.3-w4a16-mtp`](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp) (약 178 GiB, 라우팅 MoE만 INT4)
- 드래프터: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), `K=7` 고정
- 이미지: `radixark/vllm-glm53-flash:sm121-v11-dflash2` (alias `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`)
- KV: `fp8_e4m3` 9 GiB pin, `max_model_len=1M`, Marlin, `--enforce-eager`

스톡 vLLM은 이 W4A16 체크포인트를 그대로 못 올린다 (`gate_up_proj.weight` KeyError). DFlash2 checkpoint `selector_top_k=16`은 후보 풀이 좁고, 계측 프로브와 mHC TileLang은 첫 C2에서 JIT가 난다.

## 패치 (이 레포 overlay)

`./launch-glm53-w4a16-tp2-dflash2.sh`가 아래 파일을 컨테이너에 마운트한다. 이미지·가중치는 수정하지 않는다.

| 파일 | 역할 |
|---|---|
| [`patches/glm5next_model.py`](patches/glm5next_model.py) | W4A16 dense MLP 로드. `quant_config=None` + `gate_proj`/`up_proj` → `gate_up_proj` |
| [`patches/qwen3_dflash2.py`](patches/qwen3_dflash2.py) | `DFLASH_SELECTOR_TOP_K`로 lm_head 후보 풀만 변경. `selector_rank=256`은 학습 차원 → 고정 |
| [`patches/dflash2_speculator.py`](patches/dflash2_speculator.py) | walk (`DFLASH_WALK_MODE=edge`). 승인율 프로브는 `DFLASH2_ACC_PROBE=0`이면 꺼짐 |
| [`patches/spec_decode_rejection_warmup.py`](patches/spec_decode_rejection_warmup.py) | DFlash `prepare_dflash_inputs` / `_copy_page_indices` 부팅 시 compile |
| [`patches/deepseek_v4_mhc_warmup.py`](patches/deepseek_v4_mhc_warmup.py) | GLM mHC TileLang warmup. 스톡은 DSv4만 워밍해서 C2에서 `mhc_pre_big_fuse_with_norm_tilelang` JIT |
| [`docs/patch_hybrid_prefix_hit.py`](docs/patch_hybrid_prefix_hit.py) | 하이브리드 prefix-cache (APC). 드래프터 SWA가 hit를 0으로 만드는 것 수정. `APPLY_APC_PATCH=1` |
| [`patches/sparse_attn_indexer_kpool.py`](patches/sparse_attn_indexer_kpool.py) | SM121 kpool indexer (Tony). `GLM53_SM121_MLA=0`일 때 |
| [`patches/chat_template_mm.jinja`](patches/chat_template_mm.jinja) | 비전 chat template |

**유지 설정:** `DFLASH_SELECTOR_TOP_K=32`, `DFLASH_WALK_MODE=edge`, `DFLASH2_ACC_PROBE=0`, `DFLASH_TOKENS=7`, `ENFORCE_EAGER=1`, FlashInfer autotune OFF, APC ON.

**벤치 후 버린 것:** FlashInfer autotune ON, CUDA graphs (`ENFORCE_EAGER=0`), `WALK_MODE=unary`, `TOP_K=48`, 서빙 중 승인율 프로브 ON, live-temp rejection warmup, SM121 MLA overlay.

## 벤치 결과

키트: 2× DGX Spark, TP=2, `temperature=0`, 첫 요청(부팅 warmup)은 soak에 넣지 않음. 부트 간 분산 약 ±15%. 원자료 [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| 구성 | soak C1 | C1 | C6 | accept (`/metrics`) |
|---|---|---|---|---|
| top_k=16 (`final-recipe`) | 31.8 | ~30.4 | 81.1 | 0.418 |
| **top_k=32 + probe off (채택)** | **34.7–35.5** | **35.4** | **89.3** | **0.433–0.439** |
| probe ON | 32.1 | 32.2 | 84.7 | 0.435 |
| autotune ON / CUDA graphs | 33.2 / 33.0 | — | 79.0 / 79.6 | — |

- reject_split mean (temp=0, edge+32): 약 **0.48**. 0.535는 미도달 추정치라 게이트로 쓰지 않음.
- mHC warmup 후: `mhc_pre_big_fuse_with_norm_tilelang` **runtime JIT 0** (TP0·TP1). 첫 요청 `_rejection_kernel` Triton JIT는 남음.
- 콜드 프리필: ~1.3–1.6k tok/s, 300K까지 단일 스트림 OOM 없음. 콜드 부트 ~8분.

## 적용

필요: 2× DGX Spark, Docker+GPU, head→worker SSH, `/var/tmp` 약 190 GiB.

```bash
cp .env.example .env   # HEAD_IP / WORKER_IP
./validate.sh
./download.sh          # 가중치 ~178 GiB + drafter ~2.3 GiB
./start.sh             # pull, rsync, TP=2 launch, /health
# SKIP_PULL=1 ./start.sh restart
```

워커 먼저 수동 기동:

```bash
./launch-glm53-w4a16-tp2-dflash2.sh 1
sleep 25
./launch-glm53-w4a16-tp2-dflash2.sh 0
```

엔드포인트 `http://<HEAD_IP>:8000`. 로그에 `glm5next_model.py mounted`가 있어야 한다. 없으면 shard 0에서 KeyError.

## 라이선스

이 레시피 코드는 MIT ([LICENSE](LICENSE)). 가중치·드래프터·이미지는 각 Hub/이미지 라이선스를 따른다. Zhipu AI, NVIDIA, upstream과 무관.

## 출처

- [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) — 베이스 모델
- [`canada-quant/glm-5.3-w4a16-mtp`](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp) — W4A16
- [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — DFlash2 드래프터
- [tonyd2wild](https://github.com/tonyd2wild) — SM121 이미지, kpool, chat template, fabric
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — APC / SM121 MLA 조사
- vLLM, FlashInfer
