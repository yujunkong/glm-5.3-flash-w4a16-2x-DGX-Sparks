# SM121 Native — Phase 1 investigation (no code changes)

작성: 2026-09-09. STEP 1–6 only. `.env` / Dockerfile / compose / `start.sh` / launch / build script **미수정**.  
기준 문서: `benchmarks/VERIFIED-MEASUREMENTS.md`. 숫자는 그 파일에서 그대로 인용. 재계산 금지.

이 문서의 목적: **SM121 native 빌드가 어디에 필요한지 먼저 증명**하기 위한 조사. 빌드·serve A/B는 이 보고서를 확인한 뒤에만.

---

## [SM121 Investigation — Phase 1]

### Production image

```text
radixark/vllm-glm53-flash:sm121-v11-dflash2
  docker Id:     sha256:35c6f70ffcba...
  digest:        ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
  also tagged:   ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2  (same Id)
  size:          31.23 GB
  created:       2026-08-28
```

`.env` `IMAGE=` 은 위 스톡. 2026-09-09 조사 시점 production 컨테이너 **없음** (`docker ps` empty).

로컬에 실험 이미지 `glm53-flash:sm121-native` (`sha256:8c7c6f16…`, 34.9 GB) 가 남아 있음. **생산 아님.** ENTRYPOINT=`["sleep"]` CMD=`["infinity"]`. serve 금지.

### vLLM

`0.1.dev20051+g487ecf187` (이미지 내부). commit `487ecf187`.

### CUDA

이미지 `CUDA_VERSION=13.0.1` / `NV_CUDA_CUDART_VERSION=13.0.88-1`.  
호스트 GPU: `NVIDIA GB10`, compute **12.1**.  
이미지 빌드 시 `TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0` — **12.1 없음**.  
`VLLM_ENABLE_CUDA_COMPATIBILITY=0`.

### Torch

`2.13.0+cu130`, `torch.version.cuda=13.0`.

### FlashInfer

`0.6.18.dev20260819`. 사이트에 `libflashinfer*.so` 없음. JIT + `flashinfer_cubin` (44876 cubin: sm100/sm107/sm103, **sm121 없음**).  
이미 만든 AOT (`/var/tmp/sm121-build/fi-aot`, 261 `.so` sm_121a) 는 **serve에 미적용**. TPS `NOT_MEASURED`.

### DFlash2

별도 CUDA `.so` **없음**. Python 패키지:

- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` (`_prepare_dflash_inputs_kernel` — Triton JIT)
- `vllm/model_executor/models/qwen3_dflash2.py`
- 레포 bind-mount: `patches/qwen3_dflash2.py`, `patches/dflash2_speculator.py`, `patches/spec_decode_rejection_warmup.py`

Drafter 가중치: `incoai/GLM-5.3-Flash-DFlash2`, `DFLASH_TOKENS=7`.  
“DFlash SM121 native extension rebuild”는 대상이 없음. nsys에서 Triton/`prepare_dflash` 비중이 크면 JIT 쪽이지 `_C` rebuild가 아님.

### Marlin

생산 `MOE_BACKEND=marlin`. 커널은 `vllm/_moe_C_stable_libtorch.abi3.so` 안 cubin.  
`cuobjdump --list-elf` 히스토그램:

```text
_moe: sm_120×29  sm_80×22  sm_89×12  sm_100×10  sm_110×10  sm_87×7  sm_90×7  sm_90a×3
      PTX: 없음. sm_121: 없음.
_C:   sm_120×81  sm_80×69  sm_100×67  sm_110×65  sm_89×55  sm_90×55  sm_87×50  sm_90a×25
      PTX: 없음. sm_121: 없음.
FA2:  sm_80 only
FA3:  sm_90a only
```

GB10(12.1)에서 가장 가까운 Marlin cubin은 **sm_120**. sm_121 부재 ≠ 즉시 실패. sm_120 SASS가 12.1에서 실행될 수 있음. **실제 선택 cubin은 nsys/CUPTI로만 확인.**

NVFP4 cute_dsl (`flashinfer/.../blackwell_sm12x/moe_w4a16_*`) 는 이미지에 있으나 생산 Marlin 경로가 아님. 이번 병목 가설에서 NVFP4를 1순위로 두지 않음.

### Current benchmark baseline

`VERIFIED-MEASUREMENTS.md` 인용. 스톡 이미지 + 생산 overlay.

```text
Soak C1 median:
  selector-topk-32  = 34.734   (36.500 / 34.019 / 34.734)
  tps-probe-off     = 35.538   (35.538 / 32.085 / 37.652)
  밴드              = 34.7–35.5 tok/s
C6 aggregate:
  selector-topk-32  = 89.28    (한 웨이브)
  tps-probe-off     = 78.93    (같은 overlay, 다른 부팅)
  → C6를 89.3으로 고정 인용 금지
acceptance /metrics = 0.433–0.439
P1                  = 391 / Tokyo
keep 바 (무인 스크립트) = soak ≥ 34.7 × 1.025 ≈ 35.57
```

모델: `canada-quant/glm-5.3-w4a16-mtp` @ `4eeb77a3499fc2503290197118102eae2ed44553`.

하네스: `./bench/bench_config.sh <label>` (warmup 1회 버림, soak 3×512, C1/C2/C6, accept, P1).

### Current image entrypoint

```text
stock:  Entrypoint=["vllm","serve"]  Cmd=null
native: Entrypoint=["sleep"]         Cmd=["infinity"]   ← 이전 실패 원인
```

생산 기동은 **compose가 아님**. `start.sh` → `launch-glm53-w4a16-tp2-dflash2.sh` 가 `docker run ... "$IMAGE" "$MODEL_PATH" --served-model-name ...`  
이미지 ENTRYPOINT `vllm serve` 뒤에 모델 경로가 붙는 구조.

이전 핸드오프의 `ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server` 는 스톡과 불일치. 고칠 때는 **`["vllm","serve"]`** 로 맞출 것. builder `sleep infinity` 를 serving 이미지에 커밋하지 말 것.

`docker-compose.yml` 은 head/worker 프로파일이 있으나 launch 스크립트보다 빈약함 (APC mount, `MOE_BACKEND`, autotune, DFlash env, gate_linear 등 누락). **compose-only A/B는 생산과 다른 스택이 됨. 사용하지 말 것.**

`start.sh` 는 `source $ENV_FILE` 하므로 셸의 `IMAGE=foo ./start.sh` 는 `.env`가 덮어씀. 테스트 시:

```text
ENV_FILE=.env.sm121-test ./start.sh restart
```

처럼 **별도 env 파일**. `.env` 의 `IMAGE=` 을 sed 하지 말 것.

### Likely decode bottleneck

**nsys 커널 비중은 아직 없음.** 아래는 기존 실측에서 온 후보 순위 (가설).

근거:

1. Graphs `ENFORCE_EAGER=0` soak 32.97 vs 34.73 — launch overhead가 병목 아님 (`MEASURED_DISCARD`).
2. Exp B (`benchmarks/perf-exp/EXP-LOG.md`): C1·C6 모두 SM **~94–96%**, power **~28W** → kernel/UMA-bandwidth, idle/launch 아님.
3. Triton MoE는 C1 −46% — Marlin이 MoE 임계 경로에 있음.
4. SM90 vs SM120 MLA (no-spec) soak 14.205 vs 14.303 — attention 포맷 교체는 decode ±1%. 생산은 여전히 SM90 sparse MLA (`GLM53_SM121_MLA=0`).
5. DFlash 있음 31.8 vs no-spec 14.2 는 **spec accept** 이득이지 커널 SM 번호 이득이 아님.
6. `DFLASH2_ACC_PROBE=1` soak −10% — GPU→CPU sync에 민감. 커널 시간 외 동기화도 보임.

**가설 순위 (증명 전):**

```text
1) Target Marlin GEMM in _moe_C (small-M, K=7 spec)     — Case A 후보
2) Sparse MLA FlashInfer SM90 (JIT, not stock cubin sm121)
3) Drafter forward (DFlash2, 별도 .so 없음) + rejection Triton
4) NCCL/memcpy — soak C1에서는 2순위 이하로 추정
```

NVFP4 tensor-core 를 현재 35 tok/s 의 핵심 병목으로 가정하지 않음.

### Relevant CUDA extensions

| 파일 | 생산 decode 역할 | stock arch | SM121 native 의미 |
|---|---|---|---|
| `_moe_C_stable_libtorch.abi3.so` | Marlin W4A16 MoE | sm_120 최근접, sm_121 없음, PTX 없음 | **1순위 후보** — profiling에서 Marlin 비중 클 때만 |
| `_C_stable_libtorch.abi3.so` | 기타 vLLM CUDA (레이어 잡동사니) | 동일 | Marlin과 한 묶음 rebuild 되기 쉬움. 단독 주입은 ABI 리스크 |
| FlashInfer JIT `.so` | sparse MLA SM90 | 첫 사용 JIT; cubin pkg에 sm121 없음 | attention이 병목일 때만 AOT. 이미 fi-aot 존재, serve 미적용 |
| `_vllm_fa2/fa3` | FlashAttention | sm_80 / sm_90a | 생산 MLA가 FA2/3가 아니면 후순위 |
| `_flashmla_*` | FlashMLA | sm_90a/sm_100f | 생산 SM90 sparse 경로와 별개 |
| DFlash `.so` | — | **없음** | rebuild 대상 아님 |
| NVFP4 cute_dsl | 미사용 (marlin) | n/a | 강제 컴파일 금지 |

### SM121 native target candidates

profiling 전 빌드 금지. 결과에 따른 분기:

```text
Case A Marlin 우세  →  _moe_C 만 12.1a  (가능하면). 전체 vLLM+FlashInfer 재빌드 하지 않음.
Case B Attention 우세 →  이미 있는 fi-aot 만 stock 위에 COPY. MLA backend는 바꾸지 않음
                      (GLM53_SM121_MLA=0 유지). native AOT ≠ SM120 overlay.
Case C DFlash 우세   →  Triton/prepare_dflash / drafter 쪽. _C rebuild로 안 풀림.
Case D 분산          →  그때만 최소 native 를 하나씩 (moe → fi-aot → 전체 _C).
```

이미 만든 `glm53-flash:sm121-native` 는 **전체 _C+_moe+fi-aot** 라 Case D  bundling. ENTRYPOINT도 깨짐. 병목 증명 전에 이 이미지로 A/B하지 말 것. (재측정 가치가 생긴 뒤에 ENTRYPOINT만 `vllm serve` 로 고치는 것은 별 결정.)

Arch 표기: 이 툴체인은 **`12.1a`** (이미지/nvcc). bare `12.1` 또는 `12.1f` 를 추측으로 넣지 말 것. sm_120 cubin 존재를 실패로 보지 말 것.

### Recommended minimal modification

**지금은 파일 수정 없음.** 다음 승인 후 최소 작업:

1. 스톡 이미지를 **현행 `.env` 그대로** 기동 (`ENV_FILE=.env`, `IMAGE` 변경 금지).
2. decode 수십 step만 프로파일 (아래 방법). production 이미지 레이어 수정 금지.
3. 커널 시간 표를 이 문서에 추가한 뒤, Case A/B/C를 확정.
4. 그때만 별도 태그 테스트 이미지. 예: `glm53-flash:sm121-moe-<date>`. `:latest` 덮어쓰기 금지. 스톡 digest 보존.
5. A/B 독립변수는 이미지뿐. DFlash 노브 동시 변경 금지. soak keep ≥ 35.57 + P1.

프로파일링 방법 (이미지 미패치):

```text
A. (우선) vLLM built-in profiler
   --profiler-config.profiler=torch
   --profiler-config.torch_profiler_dir=/cache/profile-sm121
   warmup 후 /start_profile → 짧은 decode → /stop_profile
   오버헤드 있음. soak 숫자로 쓰지 말 것. 커널 분류용.

B. 호스트 nsys (이미지 안에 nsys 없음, 호스트 /usr/local/cuda/bin/nsys 2025.3.2 있음)
   nsys profile -t cuda,nvtx,osrt --gpu-metrics-device=all \
     --duration=20 --delay=...  -p <vllm-pid>
   또는 docker --pid=container 로 GPU context attach.
   생산 컨테이너를 nsys로 재기동하지 말고, 이미 떠 있는 PID에 짧은 capture.

C. nvidia-smi dmon (보조)
   SM/mem/power. Exp B와 교차확인. 커널 이름은 안 나옴.
```

장시간 `bench_config.sh` 를 프로파일 목적으로 먼저 돌리지 말 것.

### Files that would be modified (승인 후, 지금은 아님)

```text
docs/sm121-native-phase1.md          (이 문서 — 프로파일 결과 추기)
(나중) .env.sm121-test               신규. .env 복사 + IMAGE만 테스트 태그
(나중) docker/Dockerfile 또는 최소 COPY 스테이지
(나중) native 이미지 ENTRYPOINT 를 ["vllm","serve"] 로 재커밋 — .env 비변경
```

### Files that must remain untouched

```text
.env
.env IMAGE=
모델/HF cache
benchmarks/final-recipe, selector-topk-32, tps-probe-off  (baseline)
canada-quant/glm-5.3-w4a16-mtp
DFLASH_TOKENS / TOP_K / WALK_MODE / ACC_PROBE / MOE_BACKEND
start.sh / launch-glm53-w4a16-tp2-dflash2.sh / docker-compose.yml
유저 clone /home/yujunkong/workspace/docker/vllm (main)
```

### Risks

- sm_120 cubin이 GB10에서 이미 잘 돌면 sm_121a rebuild soak 이득 ≈ 0.
- 전체 native 이미지는 torch/transformers/vLLM Python은 같아도 `_C` ABI·잔존 sm_80/90·ENTRYPOINT 로 serve가 다시 죽을 수 있음.
- `IMAGE=` 를 `.env`에 쓰면 start.sh가 생산을 바꿈. 과거 재발.
- compose override는 launch 스크립트와  Drift.
- nsys/torch profiler는 TPS를 깎음. 프로파일 런과 soak 런을 섞지 말 것.
- FlashInfer AOT + Marlin rebuild + MLA overlay 를 한 번에 넣으면 원인 불명.

### Next step

```text
승인 대기: STEP 4 실행 — 스톡 이미지 그대로 짧은 decode profiling.
산출물: Marlin / MLA / Triton-DFlash / memcpy 시간 비중 표.
그 전까지 Dockerfile/compose/.env/build 수정 없음.
glm53-flash:sm121-native 로 A/B하지 않음.
```

---

## Docker 구조 (조사 메모)

| 경로 | 역할 | 생산 기동? |
|---|---|---|
| `start.sh` | pull/sync/worker+head launch/health/clocks | **예** |
| `launch-glm53-w4a16-tp2-dflash2.sh` | 실제 `docker run` | **예** |
| `docker-compose.yml` | 문서/보조. APC·moe·autotune 불완전 | 아니오 (이번 A/B에 쓰지 말 것) |
| `docker/Dockerfile` | SM121 builder+runtime 레시피 | 미사용 중 (LIVE 빌드는 `build-sm121.sh`) |
| `docker/build-sm121.sh` | LIVE=1 시 `--entrypoint sleep` 빌더 | 빌더 전용. serving에 sleep 넣지 말 것 |
| `docker/unattended-onward.sh` | `.env IMAGE=` 을 바꿈 | **재실행 금지** (생산 IMAGE 오염) |

---

## 이미 검증된 생산 노브 (A/B 때 고정)

`VERIFIED-MEASUREMENTS.md` 최신값:

```text
DFLASH_SELECTOR_TOP_K=32
DFLASH_WALK_MODE=edge
DFLASH2_ACC_PROBE=0
DFLASH_TOKENS=7
MOE_BACKEND=marlin
ENFORCE_EAGER=1
DISABLE_FLASHINFER_AUTOTUNE=1
ASYNC_SCHEDULING=1
APPLY_APC_PATCH=1
MAX_NUM_SEQS=6
BLOCK_SIZE=2304
MAX_NUM_BATCHED_TOKENS=8192
GLM53_SM121_MLA=0
APPLY_GATE_LINEAR=0
```

독립변수는 나중에 **이미지만**. 위 노브와 동시 변경 금지.

---

## 과거 SM121 실패 (재발 방지)

1. `docker commit` 이 빌더 ENTRYPOINT `sleep` 유지 → worker `sleep: unrecognized option '--served-model-name'`. soak 파일 없음 (`NOT_MEASURED`).
2. `unattended-onward.sh` 가 벤치 전에 `.env IMAGE=` 변경. 실패 후 스톡으로 되돌림.
3. STRICT sm_121-only 실패를 “커널이 없다/이득 없다”로 해석하지 말 것. native `_C`에는 sm_121a **있음** + sm_80/90 잔존.
4. SM121 MLA overlay는 no-spec에서 decode 패리티, DFlash 결합은 KV 과금으로 미부팅 (`PARTIAL`). native AOT와 혼동 금지.
