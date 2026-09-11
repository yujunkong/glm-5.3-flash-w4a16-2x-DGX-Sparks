#!/usr/bin/env bash
# Unattended Phase 3 finish + A/B. Keep only measured wins (>=2.5% soak C1
# vs current best, P1 must pass). Does not git-commit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${CACHE_DIR:-/var/tmp/sm121-build}"
LOG="$CACHE/logs/unattended.log"
TAG="${TAG:-glm53-flash:sm121-native}"
NAME="${NAME:-sm121-builder}"
BASELINE_SOAK="${BASELINE_SOAK:-34.7}"
BASELINE_C6="${BASELINE_C6:-89.3}"
KEEP_PCT="${KEEP_PCT:-2.5}"
BEST_SOAK="$BASELINE_SOAK"
mkdir -p "$CACHE/logs" "$ROOT/benchmarks"
exec >>"$LOG" 2>&1

log() { echo "[$(date -Is)] $*"; }
cd "$ROOT"
set -a; source "$ROOT/.env"; set +a
STOCK_IMAGE="${STOCK_IMAGE:-$IMAGE}"
echo "$STOCK_IMAGE" > "$CACHE/stock-image.txt"
WORKER_SSH="${WORKER_SSH:-${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}}"
grep -q '^APPLY_GATE_LINEAR=' "$ROOT/.env" || echo 'APPLY_GATE_LINEAR=0' >> "$ROOT/.env"

soak_median() {
  python3 -c "import json,sys
m=None
for ln in open(sys.argv[1]):
    o=json.loads(ln)
    if o.get('summary'): m=o.get('decode_tok_s_median')
print(m if m is not None else '')" "$1" 2>/dev/null || true
}
c6_agg() {
  python3 -c "import json,sys
print(json.load(open(sys.argv[1])).get('aggregate_tok_s',''))" "$1" 2>/dev/null || true
}
p1_ok() { grep -qiE '391|tokyo' "$1" 2>/dev/null; }
keep_win() {
  python3 -c "
soak=float('${1:-0}' or 0)
best=float('$BEST_SOAK')
pct=float('$KEEP_PCT')
print('yes' if soak >= best * (1 + pct/100.0) and soak > 0 else 'no')
"
}

wait_compile() {
  log "wait vLLM compile (VLLM_BUILD_OK)"
  local i=0 last
  while [ "$i" -lt 480 ]; do
    if grep -q '^VLLM_BUILD_OK ' "$CACHE/logs/vllm-build.log" 2>/dev/null; then
      log "compile OK"
      return 0
    fi
    if docker exec "$NAME" bash -lc 'pgrep -f "setup.py build_ext" >/dev/null' 2>/dev/null; then
      :
    else
      last=$(grep -n 'VLLM_BUILD_START' "$CACHE/logs/vllm-build.log" | tail -1 | cut -d: -f1)
      if [ -n "$last" ] && tail -n +"$last" "$CACHE/logs/vllm-build.log" | grep -q 'CalledProcessError\|ninja: build stopped'; then
        log "compile FAILED after last start"
        return 1
      fi
    fi
    sleep 60
    i=$((i+1))
    if [ $((i % 10)) -eq 0 ]; then
      log "still compiling $(grep -oE '\[[0-9]+/[0-9]+\]' "$CACHE/logs/vllm-build.log" | tail -1)"
    fi
  done
  log "compile timeout"
  return 1
}

install_commit() {
  log "install AOT + vLLM .so and docker commit $TAG"
  docker exec "$NAME" bash -lc '
set -euo pipefail
if [ -d /work/fi-aot ]; then
  mkdir -p /usr/local/lib/python3.12/dist-packages/flashinfer /root/.cache/flashinfer
  cp -a /work/fi-aot/. /usr/local/lib/python3.12/dist-packages/flashinfer/ || true
  cp -a /work/fi-aot/. /root/.cache/flashinfer/ || true
fi
dest=/usr/local/lib/python3.12/dist-packages/vllm
while IFS= read -r so; do
  base=$(basename "$so")
  case "$base" in
    _C*.so|_moe*.so|_vllm_fa*.so|_flash*.so|_qutlass*.so|_deep_gemm*.so) ;;
    *) continue ;;
  esac
  if [[ "$so" == */vllm_flash_attn/* ]]; then
    mkdir -p "$dest/vllm_flash_attn"
    cp -a "$so" "$dest/vllm_flash_attn/"
  else
    cp -a "$so" "$dest/"
  fi
  echo "copied $so"
done < <(find /work/vllm-src/vllm /work/vllm-src/build -name "*.so" 2>/dev/null)
ls -l "$dest"/*.so 2>/dev/null | head
'
  mkdir -p "${CACHE_HOST_PATH:-/var/tmp/glm53-w4a16-cache}/root-cache/flashinfer"
  cp -a "$CACHE/fi-aot/." "${CACHE_HOST_PATH:-/var/tmp/glm53-w4a16-cache}/root-cache/flashinfer/" 2>/dev/null || true
  docker commit "$NAME" "$TAG"
  log "committed $TAG"
  IMAGE="$TAG" STRICT=1 OUT="$CACHE/logs/verify-after.txt" \
    "$ROOT/docker/verify-sm121.sh" || log "STRICT verify non-zero (logged)"
  docker stop "$NAME" || true
}

ship_image() {
  log "docker save $TAG -> worker $WORKER_SSH"
  docker save "$TAG" | ssh -o ConnectTimeout=30 "$WORKER_SSH" "docker load"
}

set_env() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ROOT/.env"; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ROOT/.env"
  else
    echo "${key}=${val}" >> "$ROOT/.env"
  fi
}

restore_stock_image() { set_env IMAGE "$STOCK_IMAGE"; log "restored IMAGE=$STOCK_IMAGE"; }

boot_and_bench() {
  local label="$1"
  log "boot+bench $label"
  rsync -av --exclude '.git' --exclude '__pycache__' --exclude 'benchmarks' \
    "$ROOT/" "$WORKER_SSH:$ROOT/" >/dev/null || log "repo rsync warn"
  SKIP_PULL=1 SKIP_SYNC=1 "$ROOT/start.sh" restart
  "$ROOT/bench/bench_config.sh" "$label" || log "bench_config $label had failures"
  local soak c6
  soak=$(soak_median "$ROOT/benchmarks/$label/soak.json")
  c6=$(c6_agg "$ROOT/benchmarks/$label/c6.json")
  log "RESULT $label soak=$soak c6=$c6 p1=$(head -c 200 "$ROOT/benchmarks/$label/p1.txt" 2>/dev/null || true)"
  echo "$soak $c6" > "$CACHE/logs/last-metrics.txt"
}

log "==== unattended start stock=$STOCK_IMAGE tag=$TAG best=$BEST_SOAK ===="

if ! wait_compile; then
  log "vLLM compile failed — still commit AOT-only image"
fi
install_commit
ship_image

set_env IMAGE "$TAG"
boot_and_bench sm121-native
read -r SOAK C6 < "$CACHE/logs/last-metrics.txt"
WIN_SM121=no
if p1_ok "$ROOT/benchmarks/sm121-native/p1.txt" && [ "$(keep_win "$SOAK")" = "yes" ]; then
  WIN_SM121=yes
  BEST_SOAK="$SOAK"
  log "KEEP SM121-native soak=$SOAK best=$BEST_SOAK"
else
  log "DROP SM121-native soak=$SOAK best=$BEST_SOAK"
  restore_stock_image
fi

set_env DISABLE_FLASHINFER_AUTOTUNE 0
boot_and_bench sm121-autotune-on
read -r SOAK C6 < "$CACHE/logs/last-metrics.txt"
if p1_ok "$ROOT/benchmarks/sm121-autotune-on/p1.txt" && [ "$(keep_win "$SOAK")" = "yes" ]; then
  BEST_SOAK="$SOAK"
  log "KEEP autotune ON soak=$SOAK best=$BEST_SOAK"
else
  log "DROP autotune ON soak=$SOAK best=$BEST_SOAK"
  set_env DISABLE_FLASHINFER_AUTOTUNE 1
fi

set_env APPLY_GATE_LINEAR 1
boot_and_bench sm121-gate-linear
read -r SOAK C6 < "$CACHE/logs/last-metrics.txt"
if p1_ok "$ROOT/benchmarks/sm121-gate-linear/p1.txt" && [ "$(keep_win "$SOAK")" = "yes" ]; then
  BEST_SOAK="$SOAK"
  log "KEEP gate_linear soak=$SOAK best=$BEST_SOAK"
else
  log "DROP gate_linear soak=$SOAK best=$BEST_SOAK"
  set_env APPLY_GATE_LINEAR 0
fi

set_env ENFORCE_EAGER 0
boot_and_bench sm121-graphs
read -r SOAK C6 < "$CACHE/logs/last-metrics.txt"
if p1_ok "$ROOT/benchmarks/sm121-graphs/p1.txt" && [ "$(keep_win "$SOAK")" = "yes" ]; then
  BEST_SOAK="$SOAK"
  log "KEEP CUDA graphs soak=$SOAK best=$BEST_SOAK"
else
  log "DROP CUDA graphs soak=$SOAK best=$BEST_SOAK"
  set_env ENFORCE_EAGER 1
fi

SKIP_PULL=1 SKIP_SYNC=1 "$ROOT/start.sh" restart || true
{
  echo "# Unattended A/B $(date -Is)"
  echo "stock_image=$STOCK_IMAGE"
  echo "final_image=$(grep ^IMAGE= "$ROOT/.env")"
  echo "ENFORCE_EAGER=$(grep ^ENFORCE_EAGER= "$ROOT/.env")"
  echo "DISABLE_FLASHINFER_AUTOTUNE=$(grep ^DISABLE_FLASHINFER_AUTOTUNE= "$ROOT/.env")"
  echo "APPLY_GATE_LINEAR=$(grep ^APPLY_GATE_LINEAR= "$ROOT/.env" || true)"
  echo "best_soak=$BEST_SOAK baseline_soak=$BASELINE_SOAK keep_pct=$KEEP_PCT"
  echo "sm121_native_win=$WIN_SM121"
  for lab in sm121-native sm121-autotune-on sm121-gate-linear sm121-graphs; do
    echo -n "$lab soak=$(soak_median "$ROOT/benchmarks/$lab/soak.json") c6=$(c6_agg "$ROOT/benchmarks/$lab/c6.json") p1="
    head -c 120 "$ROOT/benchmarks/$lab/p1.txt" 2>/dev/null | tr '\n' ' '
    echo
  done
} | tee "$CACHE/logs/unattended-summary.txt" "$ROOT/benchmarks/unattended-summary.txt"
log "==== unattended done ===="
