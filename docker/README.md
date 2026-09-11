# SM121-native rebuild (Phase 2 / 3)

Derives from `radixark/vllm-glm53-flash:sm121-v11-dflash2`. Does not change
`canada-quant/glm-5.3-w4a16-mtp`, `.env` keys, or `start.sh`.
Native serve A/B is `MEASURED_DISCARD` — do not promote that tag to `.env`.

## Why

Stock `_C` / `_moe` `.so` contain sm_80…sm_120, **no sm_121**. FlashInfer ships
7.1 GB of sm100/sm107 cubins and **no jit-cache**. CUDA 13.0 vLLM CMake
(`487ecf187`) drops `12.1a` unless `12.1` is added to `CUDA_SUPPORTED_ARCHS`.

Use `12.1a` (not bare `12.1`) — nvcc’s arch-specific GB10 target.

Image wheel `487ecf187` is not on GitHub. Phase 3 uses a local checkout
(`VLLM_SRC`, default `/home/yujunkong/workspace/docker/vllm`). That tree is
current `main`, not the image commit — `.so` ABI may not match the stock Python.

## Build

```bash
./docker/build-sm121.sh          # Phase 2 AOT then Phase 3. Cache: /var/tmp/sm121-build
LIVE=0 ./docker/build-sm121.sh   # docker build instead of live container
```

Progress while AOT runs:

```bash
tail -1 /var/tmp/sm121-build/logs/flashinfer-aot.log
```

```bash
./docker/verify-sm121.sh                         # baseline (expect multi-arch WARN)
IMAGE=glm53-flash:sm121-native STRICT=1 ./docker/verify-sm121.sh
```

## Size

Phase 1 “≤5 GB runtime” is not reachable: `flashinfer_cubin` is 7.1 GB and
torch+CUDA runtime already exceed 5 GB. This Dockerfile keeps the working
base and only replaces CUDA objects.

## Serve

Do **not** set production `IMAGE=glm53-flash:sm121-native`. Serve A/B soak 32.288 vs 34.7–35.5 (`MEASURED_DISCARD`). Use `glm53-flash:sm121-native-serve` only for forensics (`ENTRYPOINT vllm serve`).

HF cache stays `$CACHE_HOST_PATH/huggingface` (`HF_HOME` in `.env`).
