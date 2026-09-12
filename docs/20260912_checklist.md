좋아. 이번에는 네가 준 9/10 자료를 기준으로 **실제로 적용할 수 있는 작업만** 추려서 정리하겠다. 특히 기존에 이미 검증한 항목은 반복하지 않고, **W4A16 + DFlash2 현재 스택에서 바로 A/B 할 수 있는 것**에 집중하겠다.

현재 레포의 공개 실험 결과도 `2400 MHz`와 `--async-scheduling`을 각각 A/B해서 decode 개선을 확인했고, expert parallel·CUDA Graph·BF16 KV 등은 오히려 불리하다고 명시한다. 

# 1. 최우선 — GB10 2400 MHz 고정

이건 **가장 먼저 해볼 가치가 있다.**

공개 레시피에서:

- GB10 clock 2400 MHz
- decode **+5~7%**
- prefill +2~3%
- 과도한 발열 없음
- stock으로 `./clocks.sh reset` 가능

이라고 A/B 측정했다. 

현재 네 canonical 34.734 tok/s에 단순 적용하면 이론적인 기대 범위는:

```text
34.734 × 1.05 = 36.47
34.734 × 1.07 = 37.16 tok/s
```

즉 **40 tok/s는 아니지만 상당히 의미 있는 개선**이다.

### 적용 방법

현재 프로젝트에:

```text
scripts/clocks.sh
```

를 추가하거나 해당 레포의 clock 설정 방식을 그대로 가져온다.

중요:

- 두 Spark 모두 적용
- head/worker 동일 clock
- Docker 내부가 아니라 **호스트에서 실행**
- 부팅 후 serve 전에 적용
- reset 명령도 반드시 제공

### 검증

```text
stock clock
    ↓
10분 soak
    ↓
2400 MHz
    ↓
10분 soak
```

비교:

```text
decode tok/s
acceptance
GPU temperature
EMC/memory clock
GPU clock
```

**40°C 이상 온도 상승이나 clock throttling이 생기면 채택하지 않는다.**

---

# 2. 최우선 — `--async-scheduling`

이건 이미 네 `.env`에:

```text
ASYNC_SCHEDULING=1
```

이 들어가 있으므로 **새로 적용할 것은 없다.**

공개 레시피에서 약 **+2~3% decode**가 측정됐고, 네 현재 설정이 이미 이 값을 사용하고 있다. 

따라서:

> **이 항목은 완료 상태. 다시 실험하지 않는다.**

---

# 3. 2400 MHz + async 조합

이게 가장 중요한 A/B다.

현재:

```text
stock + async
```

에서:

```text
2400 MHz + async
```

로 변경.

공개 레시피에서는 합산:

> **C1 decode +8~11%**

가 보고됐다. 

네 canonical 기준으로 단순 계산하면:

```text
34.734
 ↓
약 37.5–38.5 tok/s
```

정도가 기대 범위다.

**여기서 38 tok/s 근처가 나오면 상당히 성공적인 결과다.**

---

# 4. TPS가 시간이 지나며 떨어지는 문제 — 이 레포의 runtime 패치 적용

이건 **clock보다 중요할 수도 있다.**

mmastrac 레포는 GB10에서 `busy_loop_s=0.002`를 사용하고, 이 변경으로 decode 상승 및 SoC 온도 약 20°C 감소를 보고했다. 

또한 worker memory cap과 memory tracing 패치를 제공한다. 

따라서 이건 단순 성능 튜닝이 아니라 **네가 보고한 "시간이 지나면서 TPS가 내려가는 현상"을 잡기 위한 별도 실험**으로 진행하는 게 좋다.

### 적용 순서

```text
현재 W4A16
   ↓
spin-wait patch
   ↓
20~30분 soak
   ↓
TPS drift 확인
```

그 다음:

```text
worker memory cap
   ↓
20~30분 soak
   ↓
memory / TPS correlation
```

둘을 처음부터 동시에 넣지 않는다.

**그래야 어느 패치가 효과가 있었는지 알 수 있다.**

---

# 5. Worker memory cap

이건 **장시간 안정성용**으로 적용 후보.

mmastrac 레시피는 GB10 unified memory 환경에서 worker allocation을 제한하기 위해 별도 memory cap을 사용한다. 

네 현재 증상이:

```text
시작       35 tok/s
시간 경과   ↓
30 tok/s
더 경과     ↓
```

라면 반드시 조사할 가치가 있다.

다만 **처음부터 memory cap 값을 그대로 복사하면 안 된다.**

4× GX10 TP4와 2× DGX Spark TP2는 메모리 구조와 workload가 다르기 때문에:

1. 현재 worker RSS 측정
2. KV allocation 확인
3. DFlash allocation 확인
4. 실제 runaway 여부 확인
5. cap 설정

순으로 해야 한다.

---

# 6. `gb10_topk_fallback`

이것은 **중복 여부부터 확인**.

현재 네 이미지/패치에 이미 GB10 top-k workaround가 있다면 **건드리지 않는다.**

없다면 mmastrac/rodman 계열의 GB10 fallback을 검토한다.

하지만 이건 현재 `TOP_K=32` 자체를 바꾸는 이야기가 아니다.

```text
DFLASH_SELECTOR_TOP_K=32
```

는 그대로 유지.

---

# 7. CUDA Graph — 하지 않는다

여기서는 명확하다.

공개 W4A16 레시피에서:

> CUDA Graphs = neutral/worse under load

라고 실측됐다. 

따라서:

```text
ENFORCE_EAGER=1
```

유지.

**CUDA Graph 실험은 제외.**

---

# 8. Expert Parallel — 하지 않는다

공개 W4A16 레시피에서:

- prefill −6%
- C6 −3%

가 측정됐다. 

따라서 현재 TP=2 구조에서:

```text
EP 추가
```

하지 않는다.

---

# 9. BF16 KV — 하지 않는다

이것도 매우 중요하다.

공개 레시피에서 draft BF16 KV를 사용하면 acceptance가:

```text
0.42 → 0.33
```

으로 떨어졌다. 

현재 네:

```text
kv-cache-dtype=fp8_e4m3
```

를 유지한다.

이건 acceptance 70% 목표에도 오히려 중요하다.

---

# 10. Block size — 절대 건드리지 않는다

현재:

```text
BLOCK_SIZE=2304
```

공개 레시피에서도 2304가 KDA block size와 맞고 prefix cache를 보존하는 중요한 값이다. 1152/4608은 prefix hit를 깨는 것으로 기록돼 있다. 

따라서:

```text
2304 유지
```

---

# 11. Chunk size — 8192 유지

공개 레시피에서:

> chunks ≠ 8192

는 제외했다. 

따라서 이것도 기존값 유지.

---

# 12. MTP3 AutoRound는 별도 후보

`Intel/GLM-5.3-Flash-W4A16-AutoRound` + Native MTP3 레시피도 실제 2× GB10에서 1M context/FP8 KV로 동작한다. 

하지만 현재 공개 자료에서 **40 tok/s single-stream을 입증하는 숫자가 없기 때문에**, 지금 바로 DFlash2를 버릴 이유는 없다.

따라서:

```text
DFlash2
   ↓
현재 목표 달성 실패 시
   ↓
AutoRound + MTP3
```

순서로 두는 게 좋다.

---

# 13. NVFP4는 별도 트랙

NVFP4 TP2는 실제로 동작하는 게 확인됐다. 다만 한 공개 레시피의 single-stream decode는 **21.45 tok/s**, concurrent 16 aggregate가 61.3 tok/s였다. 

그러므로 **"NVFP4니까 무조건 40+"라고 생각하면 안 된다.**

Tony 계열의 46.9 수치는 별도 조건/레시피이므로 반드시 동일 benchmark로 재검증해야 한다.

---

# 최종 적용 순서

내가 네 서버에서 실제로 진행한다면 이렇게 한다.

### Phase A — 현재 W4A16 안정화

```text
현재 baseline
34.7–35.5
     ↓
spin-wait patch
     ↓
20~30분 soak
```

TPS drift가 줄어드는지 확인.

### Phase B — 2400 MHz

```text
stock clock + async
       ↓
2400 MHz + async
       ↓
10~20분 soak
```

목표:

**≥36.5 tok/s**

이면 채택 후보.

**≥37 tok/s**면 상당히 좋은 결과.

### Phase C — memory stability

```text
worker memory cap
+
memory trace
```

TPS drift가 계속 있으면 적용.

### Phase D — 최종 W4A16

목표:

```text
~37–38 tok/s
accept ~0.44
```

정도.

---

# 그리고 그 다음이 진짜 중요한 Phase E

W4A16에서:

```text
~37–38 tok/s
accept ~0.44
```

까지 갔는데도 네 목표인:

> **40 tok/s + accept ~70%**

에 못 미친다면 더 이상 W4A16을 튜닝하지 않는다.

그때:

```text
W4A16
   ↓
NVFP4 + DFlash2
   ↓
동일 2× Spark
   ↓
동일 512-token benchmark
```

로 넘어간다.

그리고 별도로:

```text
W4A16 + DFlash2
       vs
W4A16 + MTP3
       vs
NVFP4 + DFlash2
```

를 비교한다.

---

# Cursor에 줄 최종 작업 순서

# GLM-5.3-Flash 2× DGX Spark — GB10 Runtime Optimization

## 목표

현재 production W4A16 + DFlash2 스택을 유지하면서 다음을 검증한다.

Primary target:
- decode >= 40 tok/s
- acceptance >= 0.70

Secondary target:
- 시간이 지나도 decode TPS가 지속적으로 하락하지 않을 것

현재 canonical baseline:
- decode: 34.7–35.5 tok/s
- canonical: 34.734 tok/s
- acceptance: 0.439
- TP=2
- DFlash2 k=7
- FP8 KV
- block_size=2304

이미 검증 완료된 항목은 다시 실험하지 않는다.

---

## PHASE 1 — GB10 2400 MHz

rodman80 2× DGX Spark recipe의 clocks.sh 방식을 참고하여 host-side clock control을 추가한다.

### 요구사항

- head와 worker 모두 동일하게 적용
- Docker 내부가 아닌 host에서 실행
- serve 시작 전에 clock 설정
- stock clock으로 되돌리는 reset 명령 제공
- 현재 production clock을 먼저 기록

추가 파일:

scripts/clocks.sh

지원:

scripts/clocks.sh set
scripts/clocks.sh reset
scripts/clocks.sh status

### A/B

A:
stock clocks + current production configuration

B:
2400 MHz + current production configuration

측정:

- decode tok/s
- acceptance
- GPU temperature
- SoC temperature
- GPU clock
- memory/EMC clock
- throttling 여부

10~20분 soak.

### 채택 조건

- decode 개선 >= 3%
- thermal throttling 없음
- stability regression 없음

조건을 만족하면 production에 유지한다.

---

## PHASE 2 — spin-wait runtime patch

mmastrac/glm-5.3-flash-4x-gx10의 patch-spin-wait.sh를 확인하고,
GB10/SM121 TP=2에 필요한 부분만 이식한다.

핵심 변경:

busy_loop_s = 1.0
→
busy_loop_s = 0.002

단, 실제 현재 vLLM source/image의 동일 코드 위치를 확인하고
무조건 문자열 치환하지 않는다.

### 중요

- vLLM/CUDA kernel 수정 금지
- FlashInfer rebuild 금지
- Marlin rebuild 금지
- NCCL 변경 금지

### Benchmark

현재 W4A16 baseline으로 20~30분 soak.

30초 간격으로:

- decode TPS
- acceptance
- CPU utilization
- GPU utilization
- GPU temperature
- SoC temperature

수집.

목표는 peak TPS보다 TPS drift 감소다.

---

## PHASE 3 — worker memory cap / memory tracing

mmastrac 레포의:

worker_memory_cap.py
spark_mem_trace.py

를 조사한다.

4× GX10 TP4 값을 그대로 복사하지 않는다.

현재 TP=2 production 환경에 맞는 실제 worker memory usage를 먼저 측정한다.

### 측정

- worker RSS
- GPU/Unified memory usage
- KV allocation
- prefix cache usage
- DFlash allocation
- TPS over time

TPS 감소와 memory 증가가 상관되는지 확인한다.

상관관계가 확인될 때만 memory cap을 적용한다.

---

## PHASE 4 — GB10 top-k fallback

현재 production image에 GB10 persistent_topk workaround가 이미 존재하는지 먼저 확인한다.

이미 동일 기능이 있으면 변경하지 않는다.

없을 경우에만 mmastrac/관련 GB10 patch를 검토한다.

중요:

DFLASH_SELECTOR_TOP_K=32는 변경하지 않는다.

---

## PHASE 5 — KEEP 설정 고정

다음 값은 변경하지 않는다.

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
KV cache dtype=fp8_e4m3
block_size=2304

---

## 명시적 금지

다음은 이번 작업에서 하지 않는다.

- CUDA Graph 강제
- expert parallel 추가
- BF16 KV 전환
- block_size 1152/4608 테스트
- chunk size 변경
- TOP_K 재탐색
- DFLASH_TOKENS 재탐색
- FlashInfer rebuild
- Marlin rebuild
- vLLM _C rebuild
- NCCL rebuild
- KDA fusion 수정
- cuBLAS micro-optimization
- W4 routed expert kernel 수정

---

## PHASE 6 — 최종 W4A16 benchmark

Phase 1~4 중 채택된 변경만 적용한다.

동일 benchmark로 측정:

- 512 generated tokens
- 동일 prompt
- TP=2
- temperature=0
- 동일 DFlash2
- 동일 FP8 KV
- 동일 max context

기록:

- decode tok/s
- acceptance
- accepted tokens
- drafted tokens
- TTFT
- peak memory
- GPU/SoC temperature
- 20~30분 TPS drift

---

## PHASE 7 — NVFP4 전환 판단

최종 W4A16이 다음을 만족하지 못하면 W4A16 추가 튜닝을 중단한다.

TARGET:
decode >= 40 tok/s
acceptance >= 0.70

그 경우 별도 branch에서 NVFP4 + DFlash2를 검증한다.

NVFP4 benchmark는 반드시 동일 조건으로 수행한다.

W4A16 결과와 NVFP4 결과를 다음 표로 비교한다.

| Stack | decode | acceptance | TTFT | peak memory | 20m stability |
|---|---:|---:|---:|---:|---|
| W4A16 | | | | | |
| NVFP4 | | | | | |

최종 선택은 peak TPS가 아니라
decode + acceptance + long-run stability를 함께 기준으로 한다.

---

## 결과 파일

각 단계별 raw log를 보존한다.

evidence/
  clocks/
  spin-wait/
  memory/
  final/

최종:

evidence/final/optimization-report.md

각 변경마다 반드시:

BEFORE
AFTER
DELTA
THERMAL
STABILITY
KEEP/REVERT

를 기록한다.

**내가 보기엔 이 순서가 지금 가장 좋다.** 특히 `2400 MHz`는 이미 **같은 W4A16 + 2× DGX Spark 환경에서 실제 A/B 수치가 공개된 변경**이라 먼저 해볼 가치가 높고, `spin-wait/memory cap`은 네가 말한 **시간 경과에 따른 TPS 하락**을 겨냥한다. 

그리고 **NVFP4는 이 작업을 끝낸 뒤** 넘어가는 게 맞다. NVFP4 TP2 자체는 현재도 실제 동작이 확인됐지만, 공개된 21.45 tok/s single-stream 사례도 있기 때문에 "NVFP4 = 자동으로 40 tok/s"라고 가정하면 안 된다. 