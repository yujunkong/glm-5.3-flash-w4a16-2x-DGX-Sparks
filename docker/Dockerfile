# syntax=docker/dockerfile:1.7
# SM121-only FlashInfer AOT + vLLM CUDA extension rebuild.
# Base is the production image so GLM overlays / start.sh / .env stay compatible.
# TORCH_CUDA_ARCH_LIST=12.1a (nvcc needs the 'a' suffix; spec "12.1").
#
# Phase 1 5 GB Ubuntu-runtime squash is not done here: flashinfer_cubin alone
# is 7.1 GB and torch+cuda-runtime already exceed 5 GB. Runtime stage only
# copies rebuilt artifacts onto the same base.

ARG BASE_IMAGE=radixark/vllm-glm53-flash:sm121-v11-dflash2
ARG VLLM_COMMIT=487ecf187

# ---------------------------------------------------------------------------
# builder: gcc/cmake/git exist only here
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ARG VLLM_COMMIT
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=8 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    # Keep nvcc temp files off overlay for incremental rebuilds.
    CCACHE_DIR=/var/cache/ccache \
    FLASHINFER_CACHE_DIR=/var/cache/flashinfer

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ninja-build ccache python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build
COPY docker/patches/patch_vllm_sm121.py /opt/build/patch_vllm_sm121.py
COPY docker/aot_sm121.py /opt/build/aot_sm121.py

# --- Phase 2: FlashInfer AOT, SM121 only ----------------------------------
# aot_sm121.py points nvcc at flashinfer/data (wheel layout). Skip Gemma/OSS/XQA.
RUN --mount=type=cache,target=/var/cache/ccache \
    --mount=type=cache,target=/var/cache/flashinfer \
    --mount=type=cache,target=/opt/build/fi-build \
    mkdir -p /opt/build/fi-aot /opt/build/fi-build /opt/flashinfer-cache \
    && FI_SRC=/opt/build/fi-src FI_AOT_OUT=/opt/build/fi-aot FI_AOT_BUILD=/opt/build/fi-build \
       python3 /opt/build/aot_sm121.py \
    && cp -a /var/cache/flashinfer/. /opt/flashinfer-cache/ || true

# --- Phase 3: vLLM CUDA extensions at the image's git commit ---------------
# setup.py / CMake must see 12.1 in CUDA_SUPPORTED_ARCHS or 12.1a is dropped.
RUN --mount=type=cache,target=/var/cache/ccache \
    git clone --filter=blob:none https://github.com/vllm-project/vllm.git /opt/build/vllm \
    && cd /opt/build/vllm \
    && git checkout "${VLLM_COMMIT}" \
    && python3 /opt/build/patch_vllm_sm121.py CMakeLists.txt \
    && python3 setup.py build_ext --inplace -j"${MAX_JOBS}" \
    && python3 -c "import vllm; print('vllm', vllm.__version__)"

# ---------------------------------------------------------------------------
# runtime: no compilers. Overwrite CUDA .so + AOT modules on the same image.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1

# AOT / JIT artifacts only — do not overlay the FlashInfer Python package.
COPY --from=builder /opt/build/fi-aot /opt/flashinfer-aot
COPY --from=builder /opt/flashinfer-cache /root/.cache/flashinfer

# vLLM rebuilt extensions — copy every .so the inplace build refreshed.
COPY --from=builder /opt/build/vllm/vllm /tmp/vllm-built

# Place rebuilt objects over the dist-packages copies. Keep image Python.
RUN set -eux; \
    dest=/usr/local/lib/python3.12/dist-packages/vllm; \
    find /opt/flashinfer-aot -name "*.so" -exec cp -a {} /usr/local/lib/python3.12/dist-packages/flashinfer/ \; ; \
    find /tmp/vllm-built -name "*.so" | while read -r so; do \
      rel=${so#/tmp/vllm-built/}; \
      mkdir -p "$dest/$(dirname "$rel")"; \
      cp -a "$so" "$dest/$rel"; \
    done; \
    rm -rf /tmp/vllm-built
