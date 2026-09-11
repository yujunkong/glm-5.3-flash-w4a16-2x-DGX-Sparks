# DFlash2 / MTP acceptance (Phase 7)

Model weights and architecture unchanged (`canada-quant/glm-5.3-w4a16-mtp`,
drafter `incoai/GLM-5.3-Flash-DFlash2`, `K=7`, `selector_rank=256`).

## Production knobs (A/B winners)

| knob | value | why |
|---|---|---|
| `DFLASH_SELECTOR_TOP_K` | 32 | vs ckpt 16: accept 0.418→0.439, soak 31.8→34.7, C6 81.1→89.3 |
| `DFLASH_WALK_MODE` | edge | unary mean 0.460 vs edge 0.482 |
| `DFLASH2_ACC_PROBE` | 0 | probe ON soak 32.1 vs OFF 35.5 (GPU→CPU sync) |
| `DFLASH_TOKENS` | 7 | tied to conv `block_size=8`; other K wedges boot |

Discarded: `TOP_K=48` (0.479 vs 0.482), live-temp rejection warmup.

## Rejection split (temp=0, edge+32)

`lm_head_topk_hit` = `compute_candidates()` pool, not a second edge top-k.

Typical first-reject: **A (not in pool) ~10–12%**, **B (in pool, walk miss) ~88–90%**, **C (walk hit, verify reject) 0%**.

Soak `/metrics` accept ~0.43–0.44. reject_split mean ~**0.48**. 0.535 was an unreachable oracle-in-pool estimate — do not gate on it.

Decode ceiling with decode-side hacks only: ~0.50–0.52. 60 tok/s cannot come from acceptance alone at ~0.44.

## Language

Existing harness is code prompts (`bench_c.py` / `bench_decode.py`). Korean / English / mixed splits were not a separate matrix; code is the production workload. Re-run language matrix only after SM121-native kernels land (Phase 2/3), with `DFLASH2_ACC_PROBE=1` for one measurement boot then back to 0.

## Speculative depth

`K=7` fixed. Per-position accept (probe-off reject_split mean, tps-probe-off): pos0 0.87 → pos6 0.21. Rollback = reject at first mismatch; no extra rollback counter in `/metrics`.
