# PHASE 3 결론 — MLP-up는 LPDDR bandwidth-bound

작성: 2026-09-10. 측정 전용. FlashInfer/Marlin/vLLM/cuBLASLt/NCCL 미변경. soak 없음.
숫자 파일: `phase3-mlp-shapes.csv`, `phase3-mlp-bandwidth.txt`.

## 질문

DFlash `M=8` eager `F.linear` `(8,4096)×(12288,4096)` bf16이 **매 호출마다 100.7 MB weight를 DRAM에서 읽는가**, 그리고 그게 GB10 273 GB/s 천장인가.

## 답

**예. 전체 12288×4096 bf16 weight를 호출마다 DRAM에서 한 번 읽는다. 이 GEMM은 계산 한계가 아니라 메모리 대역폭 한계다.**

근거:

1. **Weight 바이트 (정확).** `12288 × 4096 × 2 = 100,663,296` (100.663 MB). A+C는 M=8에서 0.26 MB뿐.
2. **ncu Kernel2** (`cutlass_80_wmma` 16x16_128x1, 440.7 µs): L1 global load 151.0 MB (weight의 1.50×). L1 hit 13.47%, L2 hit 22.66% → DRAM ≈ **101.1 MB**. compact weight와 일치. 초과 로드는 캐시가 흡수.
3. **유효 대역폭.** hot 419.5 µs ÷ 100.925 MB → **240.6 GB/s** (spec 273의 **88%**). ncu 시간 기준 229 GB/s (84%).
4. **FLOPS.** M=8에서 **1.92 TFLOPS**. GB10 `mma_bf16bf16f32` 피크 212.9 TFLOPS의 **0.90%**.
5. **Arithmetic intensity.** M=8에서 7.98 FLOP/byte. ridge `212.9e12/273e9 = 779.9`. 약 **100배 아래**.
6. **M을 늘리면** AI와 TFLOPS는 오르지만 (M=64에서 AI 62.7, 15.4 TFLOPS) **latency는 390–436 µs로 평평**. weight 100.7 MB가 그대로다. compute-bound가 되려면 이 식에서 **M ≳ 390**.

## 따라서

- algoId=21 / SM121 커널 교체는 이 100.7 MB DRAM read를 없애지 못한다. 이론 상한은 대략 `100.7e6/273e9 ≈ 369 µs` (지금 420 µs 대비 ~12%).
- `gate_linear.py`는 N=288이라 이 MLP-up(N=12288)과 무관.
- 다음 질문은 커널이 아니라 **weight traffic을 줄이는가** (경로에서 bf16 W를 안 읽게, 양자화 유지, fusion, 호출 횟수 감소).

프로덕션 `.env` / soak / 재빌드 없음.
