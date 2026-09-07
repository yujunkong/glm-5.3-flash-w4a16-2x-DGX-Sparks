# GLM-5.3-Flash W4A16 + DFlash2 on 2× DGX Spark (GB10 / SM121)

> ⚠️ **Work in progress.** Validated on a single 2× DGX Spark kit. Not
> production-hardened. SM121 image, top-k fix, and fabric notes come from
> [**tonyd2wild**](https://github.com/tonyd2wild); sparse-MLA / prefix-cache
> groundwork from
> [**MiaAI-Lab**](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
> Full credits at the bottom.

OpenAI-compatible serving of **GLM-5.3-Flash** as
**W4A16 INT4 + BF16 MTP**
([`canada-quant/glm-5.3-w4a16-mtp`](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp),
base [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash))
on **two NVIDIA DGX Spark (GB10, SM121)** at TP=2, with
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
**fp8 KV cache**, and up to **1M context**.

On our kit this is the fastest GLM-5.3-Flash Spark recipe we measured — see
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Why this quant

| | BF16 | This recipe (W4A16) |
|---|---|---|
| Weights | ~599 GiB | **177.7 GiB (−70%)** |
| Serves 1M context on | 8× 80 GB | **2× DGX Spark** (also 4× H100/H200, 4× RTX PRO 6000) |
| MoE GEMMs | BF16 | INT4 group-128 GPTQ (36,288 routed-expert GEMMs only) |
| Quality-sensitive parts | — | stay BF16: attention, router/gate, shared experts, dense layers 0–2, embeddings, `lm_head`, norms, vision tower, MTP |

Single-stream is a tie (~30 tok/s, acceptance ~0.41). W4A16 wins on concurrency:
**C4 69.2 / C6 81.1 tok/s** vs ~42/51 (NVFP4) and ~44/48 (EXL3). Details:
[benchmarks/COMPARISON.md](benchmarks/COMPARISON.md).

Quality (checkpoint card): AIME 2026 **85.0%** (102/120), GSM8K **0.97**,
GPQA **0.8586**.

## Final recipe performance

Validated config: `MOE_BACKEND=marlin`, `MAX_NUM_SEQS=6`, `BLOCK_SIZE=2304`,
`MAX_NUM_BATCHED_TOKENS=8192`, DFlash2 `K=7`, KV `fp8_e4m3` pinned at 9 GiB,
`--enforce-eager` ([`benchmarks/final-recipe/`](benchmarks/final-recipe/)):

| Metric | Value |
|---|---|
| Soak C1 streaming median (3× 512 tok) | **31.8 tok/s**, TTFT 0.35 s |
| `bench_c` C1 / C2 / C4 / C6 aggregate | ~30.4 / ~44.4 / 69.2 / **81.1 tok/s** |
| DFlash2 acceptance (`/metrics`) | 0.418 |
| Cold boot | ~8 min (`init engine` 87 s, 0 TileLang recompiles) |
| Cold prefill | ~1.3–1.6k tok/s flat up to 300K, no OOM single-stream |

## Patches (this repo)

Stock vLLM in the Spark image does not load this W4A16 checkpoint as-is.
`./launch-glm53-w4a16-tp2-dflash2.sh` bind-mounts the files below into the
container. Do not edit the Docker image or the weights.

| Patch | Mounted as | What it does |
|---|---|---|
| [`patches/glm5next_model.py`](patches/glm5next_model.py) | `vllm/models/glm5next/nvidia/model.py` | **W4A16 weight-load fix.** Checkpoint keep-BF16 MLP is stored as `gate_proj` / `up_proj`. vLLM fuses them to `gate_up_proj` and then applies compressed-tensors, so `params_dict` has no `.weight` and boot dies with `KeyError: 'layers.0.mlp.gate_up_proj.weight'`. Fix: `quant_config=None` on dense layers 0–2 and shared experts; `packed_modules_mapping` for `gate_up_proj → [gate_proj, up_proj]`; stacked load `gate_proj`/`up_proj` → `layers.N.mlp.gate_up_proj.weight` (confirmed on DGX: that key exists after the quant skip). |
| [`patches/sparse_attn_indexer_kpool.py`](patches/sparse_attn_indexer_kpool.py) | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | SM121 persistent top-k / kpool indexer (Tony). Skipped when `GLM53_SM121_MLA=1` because the SM120 overlay generates its own indexer. |
| [`patches/chat_template_mm.jinja`](patches/chat_template_mm.jinja) | copied next to the weights | Vision chat template (not on the HF repo). |
| [`docs/patch_hybrid_prefix_hit.py`](docs/patch_hybrid_prefix_hit.py) | generated `kv_cache_coordinator.py` | Hybrid prefix-cache: stock vLLM lets the drafter SWA group zero APC hits. Default on (`APPLY_APC_PATCH=1`). Fail-closed — if anchors drift, boot continues stock. |
| [`docs/patch_sm121_mla.py`](docs/patch_sm121_mla.py) | 6-file overlay when `GLM53_SM121_MLA=1` | NoPE sparse MLA on SM120 kernels (512 → 576 zero-pad). Also reapplies the W4A16 dense-MLP `quant_config=None` / `packed_modules_mapping` fix on the generated `model.py`. Default **off**. |

DGX check (Worker_TP0) after the W4A16 model patch:

```
params_dict layer 0 MLP:
  layers.0.mlp.gate_up_proj.weight
  layers.0.mlp.down_proj.weight

checkpoint:
  model.language_model.layers.0.mlp.gate_proj.weight
  model.language_model.layers.0.mlp.up_proj.weight
```

`GLM53_SM121_MLA=0` (default) mounts `patches/glm5next_model.py`.
`=1` generates `model.py` via `docs/patch_sm121_mla.py` and applies the same
quant skip there so the KeyError does not return.

## What's in this repo

| File | Role |
|---|---|
| `launch-glm53-w4a16-tp2-dflash2.sh` | TP=2 launcher, rank `0\|1` (mounts patches, DFlash2) |
| `start.sh` | 2-node orchestrator: pull → download → rsync → launch → health |
| `stop.sh` / `status.sh` | thin wrappers over `start.sh` |
| `download.sh` | weights (~178 GiB) + drafter (~2.3 GiB) from HF |
| `validate.sh` | pre-boot gates: shards, drafter, image, GIDs, swappiness, disk |
| `docker-compose.yml` | optional head/worker compose path |
| `.env.example` | knobs (copy to `.env`, set IPs/paths) |
| `PARAMS.md` | verbatim HF checkpoint parameters |
| `clocks.sh` | lock GB10 clocks at 2400 MHz (hooked from `start.sh`) |
| `patches/` | runtime bind-mounts (model loader, kpool, chat template) |
| `docs/` | APC / SM121 overlay generators + investigation notes |
| `bench/` | decode / concurrency / prefill / prefix / acceptance probes |
| `benchmarks/` | `RESULTS.md`, `COMPARISON.md`, `final-recipe/` |

## Performance tuning

A/B on our pair (`benchmarks/perf-exp/EXP-LOG.md`):

- **GB10 clocks 2400 MHz** (`./clocks.sh`): +5–7% decode, +2–3% prefill. Reset: `./clocks.sh reset`.
- **`--async-scheduling`**: +2–3% decode. Default off.
- **Micro opts** (`VLLM_MARLIN_USE_ATOMIC_ADD=1`, `VLLM_USE_FUSED_MOE_GROUPED_TOPK=1`): ~0, harmless.
- Combined: **~+8–11% decode C1, +3% C6, +2% prefill**.
- **Do not use**: `--enable-expert-parallel`, cudagraphs under load, draft with bf16 KV, block sizes 1152/4608, chunks ≠ 8192.

## Requirements

- 2× DGX Spark (GB10, 128 GiB UMA) with CX7 RoCE
- Docker + GPU on both nodes, passwordless SSH head → worker
- ~190 GiB free in `/var/tmp` on both nodes
- `huggingface-cli` (or `hf`) for download
- `vm.swappiness=10` (or 0) — `validate.sh` / `start.sh` check this

## Quickstart

```bash
git clone <this-repo> && cd glm-5.3-flash-w4a16-2x-DGX-Sparks
cp .env.example .env   # set HEAD_IP / WORKER_IP / SSH

./validate.sh
./download.sh          # ~178 GiB weights + ~2.3 GiB drafter
./start.sh             # pull, rsync, launch TP=2, poll /health
# ./start.sh restart | stop | status | logs [worker]
```

Manual path — **worker first**:

```bash
./launch-glm53-w4a16-tp2-dflash2.sh 1
sleep 25
./launch-glm53-w4a16-tp2-dflash2.sh 0

until curl -sf http://<HEAD_IP>:8000/health >/dev/null; do sleep 20; done
```

Smoke test:

```bash
curl http://<HEAD_IP>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "Prove there are infinitely many primes."}],
  "max_tokens": 128,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Look for `[patch] glm5next_model.py mounted (W4A16 gate_up_proj load fix)` in
the launch log. Without that mount, Worker_TP0 dies on shard 0 with
`KeyError: 'layers.0.mlp.gate_up_proj.weight'`.

## Configuration reference

Image: `radixark/vllm-glm53-flash:sm121-v11-dflash2`
(alias `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`).

| Env | 1M (full) | 262K (staging) |
|---|---|---|
| `MAX_MODEL_LEN` | 1048576 | 262144 |
| `KV_CACHE_MEMORY` | 9663676416 (9 GiB) | 3221225472 (3 GiB) |
| `--max-num-seqs` | 6 | 6 |
| `--block-size` | 2304 | 2304 |
| `--kv-cache-dtype` | fp8_e4m3 | fp8_e4m3 |
| `--speculative-config` | DFlash2, `num_speculative_tokens=7` | same |
| `--enforce-eager` | 1 | 1 |
| `--gpu-memory-utilization` | 0.85 | 0.85 |

Boot notes:

- DFlash2 needs **exactly 7** speculative tokens.
- MoE backend must be **`marlin`** (or auto). `flashinfer_cutlass` is NVFP4-only.
- **fp8 KV is required** on Spark.
- Keep the **9 GiB KV pin** for 1M context.
- Poll **`/health`**, not `/v1/models`.
- GB10 ritual each boot: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- Single prompts ≲ ~310K tokens in our tests.

## Experimental: SM120 sparse-MLA overlay

`GLM53_SM121_MLA=1` enables `FLASHINFER_MLA_SPARSE_SM120` (NoPE 512 → 576-wide
kernel via zero-pad). `=0` is the SM90 baseline. A/B (2026-09-03): decode
parity, **−21% KV pool** — keep `=0`. Runbook:
[`docs/sm121-mla-investigation.md`](docs/sm121-mla-investigation.md).

## Credits

- [**zai-org/GLM-5.3-Flash**](https://huggingface.co/zai-org/GLM-5.3-Flash) — base model.
- [**canada-quant/glm-5.3-w4a16-mtp**](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp) — W4A16 quant.
- [**tonyd2wild**](https://github.com/tonyd2wild) — SM121 image, kpool top-k, `chat_template_mm.jinja`, fabric notes.
- [**incoai/GLM-5.3-Flash-DFlash2**](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — drafter.
- [**MiaAI-Lab**](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — SM120 MLA + hybrid APC ports.
- **vLLM** and **FlashInfer**.

## License

Recipe: MIT — see [LICENSE](LICENSE). Weights, drafter, and images keep their
own licenses. Not affiliated with Zhipu AI, NVIDIA, or upstream projects.
