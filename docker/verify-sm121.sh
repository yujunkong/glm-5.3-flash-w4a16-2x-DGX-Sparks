#!/usr/bin/env bash
# verify-sm121.sh — cuobjdump SM inventory for FlashInfer + vLLM CUDA objects.
#
# Spec command:  cuobjdump libflashinfer*.so | grep sm_
# Stock image has no libflashinfer*.so (JIT cubins + flashinfer_cubin package).
# This script dumps every CUDA object we can find. STRICT=1 fails if any
# non-sm_121 SASS is present (use after the SM121-only rebuild).
set -euo pipefail

IMAGE="${IMAGE:-radixark/vllm-glm53-flash:sm121-v11-dflash2}"
OUT="${OUT:-}"
STRICT="${STRICT:-0}"

run() {
  docker run --rm --entrypoint bash "$IMAGE" -lc "$1"
}

audit() {
  echo "=== SM121 audit  image=$IMAGE  $(date -Is) ==="
  echo

  echo "=== versions ==="
  run 'python3 - <<PY
import torch, vllm, flashinfer, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("flashinfer", getattr(flashinfer, "__version__", "?"))
PY
echo "nvcc:"; nvcc --version | tail -2'

  echo
  echo "=== flashinfer_cubin filename SM histogram ==="
  run 'python3 - <<PY
from pathlib import Path
from collections import Counter
root = Path("/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins")
c = Counter()
n = 0
if not root.exists():
    print("no flashinfer_cubin")
else:
    for p in root.rglob("*.cubin"):
        n += 1
        name = p.name.lower()
        hit = False
        for tok in ("sm_121", "sm121", "sm_120", "sm120", "sm_110", "sm110",
                    "sm_100", "sm100", "sm107", "sm103", "sm90", "sm80"):
            if tok in name:
                c[tok] += 1
                hit = True
        if not hit:
            c["other"] += 1
    print("total_cubins", n)
    for k, v in c.most_common():
        print(f"  {k}: {v}")
PY'

  echo
  echo "=== FlashInfer JIT / site .so (spec: libflashinfer*.so) ==="
  run 'CU=/usr/local/cuda/bin/cuobjdump
mapfile -t SOS < <(find /root/.cache/flashinfer /usr/local/lib/python3.12/dist-packages/flashinfer \
  -name "libflashinfer*.so" -o -name "*.so" 2>/dev/null | head -200)
if [ ${#SOS[@]} -eq 0 ]; then
  echo "NO_FLASHINFER_SO (kernels JIT on first use; flashinfer-jit-cache not installed)"
else
  for so in "${SOS[@]}"; do
    echo "---- $so ----"
    $CU "$so" 2>/dev/null | grep -E "sm_|arch =" | sort -u || echo "cuobjdump_empty"
  done
fi'

  echo
  echo "=== vLLM CUDA .so unique archs ==="
  run 'python3 - <<PY
import glob, subprocess, os
cu = "/usr/local/cuda/bin/cuobjdump"
root = "/usr/local/lib/python3.12/dist-packages/vllm"
for so in sorted(glob.glob(root + "/**/*.so", recursive=True)):
    base = os.path.basename(so)
    if not any(x in base for x in ("_C", "flash", "moe", "qutlass", "fa2", "fa3")):
        continue
    out = subprocess.run([cu, so], capture_output=True, text=True)
    uniq = sorted({ln.split("=", 1)[-1].strip() for ln in out.stdout.splitlines() if "arch =" in ln})
    print(os.path.relpath(so, root), "mb", round(os.path.getsize(so) / 1e6, 1), "archs", uniq or ["(none)"])
PY'

  echo
  echo "=== STRICT sm_121-only (vLLM _C + _moe) ==="
  ARCHS=$(run 'CU=/usr/local/cuda/bin/cuobjdump
for so in /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so \
          /usr/local/lib/python3.12/dist-packages/vllm/_moe_C_stable_libtorch.abi3.so; do
  [ -f "$so" ] || continue
  $CU "$so" 2>/dev/null | grep -oE "sm_[0-9]+[a-z]*" | sort -u
done' | sort -u)
  echo "$ARCHS"
  BAD=$(echo "$ARCHS" | grep -vE "^sm_121a?$" | grep -v "^$" || true)
  if [ -n "$BAD" ]; then
    echo "NON_SM121: $(echo "$BAD" | tr "\n" " ")"
    if [ "$STRICT" = "1" ]; then
      echo "FAIL: expected sm_121 only" >&2
      return 1
    fi
    echo "WARN: multi-arch baseline (expected until rebuild)"
  else
    echo "OK: sm_121 only"
  fi
  echo
  echo "=== done ==="
}

if [ -n "$OUT" ]; then
  mkdir -p "$(dirname "$OUT")"
  audit 2>&1 | tee "$OUT"
else
  audit
fi
