#!/usr/bin/env bash
# build-sm121.sh — SM121-only FlashInfer AOT + vLLM CUDA rebuild.
# Incremental: host cache at /var/tmp/sm121-build + docker buildx cache.
# Does not change .env. After a passing STRICT verify, set IMAGE= to the tag.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-radixark/vllm-glm53-flash:sm121-v11-dflash2}"
TAG="${TAG:-glm53-flash:sm121-native}"
VLLM_COMMIT="${VLLM_COMMIT:-487ecf187d3dfe74d2cf6119a92881dba403c219}"
# Host checkout used for Phase 3 (image hash is not on GitHub). Default: sibling clone.
VLLM_SRC="${VLLM_SRC:-/home/yujunkong/workspace/docker/vllm}"
CACHE_DIR="${CACHE_DIR:-/var/tmp/sm121-build}"
# 0 = docker build (reproducible). 1 = live container compile then docker commit
# (survives AOT failures; preferred for the first multi-hour run).
LIVE="${LIVE:-1}"

mkdir -p "$CACHE_DIR"/{ccache,flashinfer,fi-aot,fi-build,vllm,logs}
chmod +x "$ROOT/docker/verify-sm121.sh" "$ROOT/docker/patches/patch_vllm_sm121.py"

echo "[build-sm121] base=$BASE_IMAGE tag=$TAG live=$LIVE cache=$CACHE_DIR"

if [ "$LIVE" != "1" ]; then
  docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg VLLM_COMMIT="$VLLM_COMMIT" \
    -t "$TAG" \
    -f "$ROOT/docker/Dockerfile" \
    "$ROOT"
  IMAGE="$TAG" STRICT=1 OUT="$CACHE_DIR/logs/verify-after.txt" \
    "$ROOT/docker/verify-sm121.sh"
  exit 0
fi

# --- live container: Phase 2 AOT first, then Phase 3 vLLM -----------------
NAME="${NAME:-sm121-builder}"
docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" --gpus all \
  --network host \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e CUDA_HOME=/usr/local/cuda \
  -e MAX_JOBS="${MAX_JOBS:-8}" \
  -e CCACHE_DIR=/work/ccache \
  -e FLASHINFER_CACHE_DIR=/work/flashinfer \
  -v "$CACHE_DIR:/work" \
  -v "$ROOT/docker/patches/patch_vllm_sm121.py:/opt/patch_vllm_sm121.py:ro" \
  -v "$ROOT/docker/aot_sm121.py:/opt/aot_sm121.py:ro" \
  --entrypoint sleep \
  "$BASE_IMAGE" infinity

# Build tools only inside the builder container (not the runtime image).
docker exec "$NAME" bash -lc 'export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git cmake build-essential ninja-build ccache python3-dev
rm -rf /var/lib/apt/lists/*'

echo "[build-sm121] Phase 2 FlashInfer AOT (12.1a only) — long compile"
docker exec "$NAME" bash -lc '
set -euo pipefail
mkdir -p /work/fi-aot /work/fi-build
python3 /opt/aot_sm121.py
echo AOT_OK
' 2>&1 | tee "$CACHE_DIR/logs/flashinfer-aot.log"

echo "[build-sm121] cuobjdump AOT .so"
docker exec "$NAME" bash -lc '
CU=/usr/local/cuda/bin/cuobjdump
find /work/fi-aot /work/flashinfer /work/fi-build -name "*.so" 2>/dev/null | head
find /work/fi-aot /work/flashinfer /work/fi-build -name "*.so" -print0 2>/dev/null \
  | xargs -0 -n1 -I{} sh -c "echo ==== {} ====; $CU {} 2>/dev/null | grep -E \"sm_|arch =\" | sort -u"
' | tee "$CACHE_DIR/logs/flashinfer-cuobjdump.txt"

echo "[build-sm121] Phase 3 vLLM CUDA from /work/vllm-src (host $VLLM_SRC)"
# FetchContent GIT_TAG is a commit SHA; cmake's default shallow clone cannot
# check it out. Prefetch each pin, then point SRC_DIR env vars at the trees.
docker exec "$NAME" bash -lc '
set -euo pipefail
clone_at() {
  local repo="$1" sha="$2" dest="$3"
  if [ -d "$dest/.git" ]; then
    git -C "$dest" fetch --depth=1 origin "$sha"
    git -C "$dest" checkout --detach "$sha"
    return
  fi
  git clone --filter=blob:none "$repo" "$dest"
  git -C "$dest" fetch --depth=1 origin "$sha"
  git -C "$dest" checkout --detach "$sha"
}
d=/work/vllm-src/.deps
clone_at https://github.com/JaredforReal/FlashMLA.git 8447acbcb558db892bf7c1197d225be1c95b168c "$d/flashmla-manual"
clone_at https://github.com/JaredforReal/flash-attention.git 2b84100f50e1d2a8726a86c86d14c2f9c9e5a67c "$d/vllm-flash-attn-src"
git -C "$d/vllm-flash-attn-src" submodule update --init --depth=1 csrc/cutlass || git -C "$d/vllm-flash-attn-src" submodule update --init csrc/cutlass
clone_at https://github.com/vllm-project/FlashKDA.git b5d11010ff01c1d4a683c0dde42e76cbeaa8107f "$d/flashkda-src"
git -C "$d/flashkda-src" submodule update --init --depth=1 cutlass || git -C "$d/flashkda-src" submodule update --init cutlass
clone_at https://github.com/vllm-project/MSA.git 087c161814d4d9c735b46c21212a09e5f8eb92fa "$d/fmha_sm100-src"
clone_at https://github.com/vllm-project/tml-fa4.git b206834606ed5b5f21f8eed6b0683f528ea9cf7d "$d/tml_fa4-src"
clone_at https://github.com/IST-DASLab/qutlass.git e74319e3405ce6d71965732880f5dc1f52371f64 "$d/qutlass-src"
'
docker exec -e MAX_JOBS="${MAX_JOBS:-8}" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev20051+g487ecf187 \
  -e VLLM_USE_PRECOMPILED_RUST=1 \
  -e FLASH_MLA_SRC_DIR=/work/vllm-src/.deps/flashmla-manual \
  -e VLLM_FLASH_ATTN_SRC_DIR=/work/vllm-src/.deps/vllm-flash-attn-src \
  -e FLASH_KDA_SRC_DIR=/work/vllm-src/.deps/flashkda-src \
  -e FMHA_SM100_SRC_DIR=/work/vllm-src/.deps/fmha_sm100-src \
  -e TML_FA4_SRC_DIR=/work/vllm-src/.deps/tml_fa4-src \
  -e QUTLASS_SRC_DIR=/work/vllm-src/.deps/qutlass-src \
  "$NAME" bash -lc '
set -euo pipefail
cd /work/vllm-src
python3 /opt/patch_vllm_sm121.py CMakeLists.txt
pip install -q setuptools-rust
cp -n /usr/local/lib/python3.12/dist-packages/vllm/_rust_*.so /work/vllm-src/vllm/ || true
python3 setup.py build_ext --inplace -j"${MAX_JOBS:-8}"
python3 -c "import os; print(\"built\", os.getcwd())"
' 2>&1 | tee "$CACHE_DIR/logs/vllm-build.log"

echo "[build-sm121] install artifacts into container + commit $TAG"
docker exec "$NAME" bash -lc '
set -euo pipefail
# AOT modules: copy .so next to flashinfer package (loader searches FLASHINFER_CACHE_DIR too).
if [ -d /work/fi-aot ]; then
  cp -a /work/fi-aot/. /usr/local/lib/python3.12/dist-packages/flashinfer/ 2>/dev/null || true
fi
dest=/usr/local/lib/python3.12/dist-packages/vllm
src=/work/vllm-src/vllm
if [ ! -d "$src" ]; then src=/work/vllm/vllm; fi
if [ -d "$src" ]; then
  find "$src" -maxdepth 1 -name "*.so" -exec cp -a {} "$dest/" \;
  if [ -d "$src/vllm_flash_attn" ]; then
    find "$src/vllm_flash_attn" -maxdepth 1 -name "*.so" \
      -exec cp -a {} "$dest/vllm_flash_attn/" \;
  fi
fi
'

docker commit "$NAME" "$TAG"
echo "[build-sm121] committed $TAG"
IMAGE="$TAG" STRICT=1 OUT="$CACHE_DIR/logs/verify-after.txt" \
  "$ROOT/docker/verify-sm121.sh" || true
echo "[build-sm121] next: IMAGE=$TAG SKIP_PULL=1 ./start.sh restart"
echo "[build-sm121] builder container kept as $NAME for incremental rebuilds"
