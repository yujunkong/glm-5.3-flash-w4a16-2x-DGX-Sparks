# PHASE 4 — W4A16 모델이 BF16 cuBLASLt MLP-up를 타는 이유

작성: 2026-09-10. 코드/이미지/`.env` 미수정. soak 없음.

## 경로 (실측)

```
checkpoint
  DFlash2  전부 BF16 (81 tensors)
  Target   layers 0–2 MLP BF16  (quant ignore)
  Target   routed experts  W4 packed I32 + scale   ← Marlin only
        │
        ▼
load  MergedColumnParallelLinear.weight   (persistent, BF16)
  TP=2: fused gate_up 24576 → shard 12288×4096 = 100.663 MB / GPU
        │
        ▼  (decode, no dequant)
F.linear / ATen mm
        │
        ▼
cuBLASLt algoId=21  cutlass_80_wmma 16×16_128x1
        │
        ▼
DRAM 100.7 MB / call   (L2 24 MiB에 안 들어감)
```

**W4 → BF16 materialize 단계는 이 GEMM에 없다.** operand는 처음부터 BF16이다.

## 누가 이 GEMM을 돌리나

| 모듈 | 체크포인트 | 런타임 | 이 100.7 MB 형상 |
|---|---|---|---|
| DFlash2 5-layer Qwen3 MLP | BF16 gate/up (12288,4096) | `quant_config=None` | 예 (TP 샤드 후) |
| Target dense MLP layers 0–2 | BF16, `ignore` 리스트 | `Glm5NextMLP(quant_config=None)` | 예 |
| Target shared_experts | BF16 (2048,4096) | `quant_config=None` | 아니오 (더 작음) |
| Target routed experts | W4 packed | Marlin WNA16 | 아니오 (2048, Marlin 43%) |

`patches/glm5next_model.py` 주석: ignore 키는 unfused `gate_proj`/`up_proj`인데 vLLM은 `gate_up_proj`로 합친다. `quant_config`를 넘기면 fused 모듈이 양자화되며 `.weight` KeyError. 그래서 **의도적으로** BF16 Linear다.

## Q1–Q10 요약

1. 저장 dtype = **BF16** (이 MLP). W4는 expert만.
2. W4→BF16 변환 = **없음**.
3. BF16 weight = **persistent Parameter**, per-step materialize 아님.
4. step마다 재생성 = **아니오**.
5. step 사이 유지 = **예**. DRAM 재읽기는 L2(24 MiB) ≪ 100.7 MB.
6. 원본 W4로 GEMM = **이 모듈에 W4가 없음**.
7. Marlin Linear는 있으나 이 텐서에 연결되어 있지 않음.
8. 이 MLP-up에 Marlin = **아니오**.
9. MoE 양자화 경로를 우회하는 이유 = drafter BF16 + dense/shared ignore + `quant_config=None`.
10. 추가 dequant 트래픽 = **0**. 비용은 GEMM의 persistent BF16 100.7 MB read.

코드 변경 제안은 하지 않음 (지시). 사실만: 이 45%는 “W4를 BF16으로 풀어 쓰는 버그”가 아니라 **체크포인트가 이 레이어를 BF16으로 둔 설계**다.
