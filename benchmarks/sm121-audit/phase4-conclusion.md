# PHASE 4 결론

**100.663 MB BF16 GEMM은 W4 dequant가 아니다.** DFlash2 전체와 타깃 dense MLP(layers 0–2)는 체크포인트부터 BF16이며, 런타임 Parameter로 유지된다. 매 decode마다 다시 만드는 버퍼가 없다. DRAM을 다시 읽는 이유는 **24 MiB L2에 100.7 MB가 안 들어가서**다.

Routed MoE expert만 W4 packed → Marlin (별도 43% 버킷, intermediate 2048). 이 F.linear는 그 경로를 우회한 게 아니라 **애초에 양자화 대상이 아니다** (`quantization_config.ignore` + `quant_config=None`).

따라서 “W4 그대로 GEMM에 넣기”는 **현재 가중치가 W4가 아니라서 불가**하다. 가능한 다음 선택은 조사 범위 밖(모델/퀀트 변경)이거나, 이 45%를 LPDDR 한계로 두고 나머지 55%로 넘어가는 것이다.

파일: `phase4-weight-path.txt`, `phase4-weight-path.md`. 프로덕션 미변경.
