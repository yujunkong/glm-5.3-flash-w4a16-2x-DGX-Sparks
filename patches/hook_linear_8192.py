# Observation only. Mount as sitecustomize.py when LINEAR_HOOK=1.
# Traces F.linear / unquantized GEMM on the live TP=2 decode path.
# Does not change scores, kernels, or model weights.
# Ubuntu images already ship /usr/lib/python3.12/sitecustomize.py (apport),
# which shadows dist-packages; launch mounts this file over that path.

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
import traceback
from collections import Counter

# Keep Ubuntu apport if present (we replace that sitecustomize file).
try:
    import apport_python_hook

    apport_python_hook.install()
except Exception:
    pass

if not getattr(sys, "_glm53_linear_hook", False):
    sys._glm53_linear_hook = True  # type: ignore[attr-defined]

    _OUT_DIR = os.environ.get("LINEAR_HOOK_OUT", "/cache")
    # Capture stacks for full and TP=2-sharded shapes of the unidentified GEMM
    # plus fused KDA in_proj / MLA q_b so we can compare counts.
    _STACK_NK = {
        (8192, 4096),
        (4096, 4096),
        (8192, 2048),
        (12480, 4096),
        (6240, 4096),
        (16384, 1536),
        (8192, 1536),
        (4096, 8192),
    }
    _started = time.time()
    _shape_counts: Counter[tuple[int, int, int]] = Counter()
    _hits: dict[str, dict] = {}
    _call_n = 0
    _armed = False
    _orig_linear = None
    _orig_mm = None
    _gemm_patched = False
    _tls = threading.local()
    _ARM = os.path.join(_OUT_DIR, "phase8-hook-arm")

    def _rank() -> int:
        r = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or ""
        if r.isdigit():
            return int(r)
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
        return 0

    def _prefix_from_frame(frame) -> str | None:
        loc = frame.f_locals
        for key in ("self", "layer", "module"):
            obj = loc.get(key)
            if obj is None:
                continue
            p = getattr(obj, "prefix", None)
            if isinstance(p, str) and p:
                return p
        return None

    def _classify(prefix: str, stack: list[str]) -> str:
        blob = (prefix + " " + " ".join(stack)).lower()
        if "dflash" in blob or "qwen3_dflash" in blob or "spec_decode" in blob:
            return "dflash"
        if "kda" in blob or "linearattention" in blob or "gateddelta" in blob:
            return "kda"
        if "mla" in blob or "indexer" in blob:
            return "mla"
        if "shared_expert" in blob or "fusedmoe" in blob or "marlin" in blob:
            return "moe"
        if "q_proj" in blob or "k_proj" in blob or "v_proj" in blob:
            return "attn_proj"
        if "glm5" in blob:
            return "target"
        return "other"

    def _stack_and_prefix(given_prefix: str = "") -> tuple[str, list[str], str]:
        stack: list[str] = []
        for fr in traceback.extract_stack()[:-2]:
            fn = fr.filename or ""
            if "sitecustomize" in fn or "hook_linear" in fn:
                continue
            if not any(
                s in fn
                for s in (
                    "/vllm/",
                    "qwen3_dflash",
                    "glm5next",
                    "fused_moe",
                    "linear.py",
                    "kda.py",
                    "attention.py",
                    "utils.py",
                )
            ):
                continue
            stack.append(f"{fn}:{fr.lineno}:{fr.name}")
        prefix = given_prefix
        if not prefix:
            try:
                f = sys._getframe(2)
                while f is not None:
                    p = _prefix_from_frame(f)
                    if p:
                        prefix = p
                        break
                    f = f.f_back
            except Exception:
                pass
        return prefix or "?", stack[-10:], _classify(prefix or "", stack)

    def _maybe_arm() -> None:
        """Drop warmup/load traffic once /cache/phase8-hook-arm appears."""
        global _armed, _call_n
        if _armed:
            return
        if os.path.isfile(_ARM):
            _shape_counts.clear()
            _hits.clear()
            _call_n = 0
            _armed = True

    def _record(m: int, n: int, k: int, prefix: str = "", src: str = "linear") -> None:
        global _call_n
        _maybe_arm()
        if not _armed:
            return
        _shape_counts[(m, n, k)] += 1
        _call_n += 1
        if (n, k) in _STACK_NK:
            pfx, stack, kind = _stack_and_prefix(prefix)
            key = f"{kind}|{pfx}|{m}|{n}|{k}|{src}|{stack[-1] if stack else '?'}"
            rec = _hits.get(key)
            if rec is None:
                rec = {
                    "kind": kind,
                    "prefix": pfx,
                    "src": src,
                    "M": m,
                    "N": n,
                    "K": k,
                    "count": 0,
                    "stack": stack,
                }
                _hits[key] = rec
            rec["count"] += 1
        if _call_n == 8 or _call_n % 64 == 0:
            _flush()

    def _flush() -> None:
        try:
            os.makedirs(_OUT_DIR, exist_ok=True)
            payload = {
                "schema": "phase8-linear-hook-v2",
                "rank": _rank(),
                "armed": _armed,
                "elapsed_s": round(time.time() - _started, 1),
                "linear_calls": _call_n,
                "hits": list(_hits.values()),
                "top_shapes": [
                    {"M": a, "N": b, "K": c, "n": n}
                    for (a, b, c), n in _shape_counts.most_common(60)
                ],
            }
            path = os.path.join(_OUT_DIR, f"phase8-linear-hook-r{_rank()}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def _patch_vllm_gemm() -> None:
        """Record layer.prefix from the actual unquantized GEMM dispatcher."""
        global _gemm_patched
        if _gemm_patched:
            return
        mod = sys.modules.get("vllm.model_executor.layers.utils")
        if mod is None:
            return
        orig = getattr(mod, "default_unquantized_gemm", None)
        if orig is None:
            return

        def wrapped(layer, x, weight, bias=None):
            _tls.in_gemm = True
            try:
                if hasattr(x, "shape") and hasattr(weight, "shape") and weight.ndim == 2:
                    k_in = int(x.shape[-1]) if x.ndim >= 1 else 0
                    m = int(x.numel() // k_in) if k_in else int(x.shape[0])
                    _record(
                        m,
                        int(weight.shape[0]),
                        int(weight.shape[1]),
                        prefix=str(getattr(layer, "prefix", "") or ""),
                        src="gemm",
                    )
                return orig(layer, x, weight, bias)
            finally:
                _tls.in_gemm = False

        mod.default_unquantized_gemm = wrapped  # type: ignore[assignment]
        _gemm_patched = True

    def _install() -> None:
        global _orig_linear, _orig_mm
        import torch
        import torch.nn.functional as F

        _orig_linear = F.linear
        _orig_mm = torch.mm

        def linear(inp, weight, bias=None):
            _patch_vllm_gemm()
            # Skip if already counted in default_unquantized_gemm.
            if not getattr(_tls, "in_gemm", False):
                if torch.is_tensor(inp) and torch.is_tensor(weight) and weight.ndim == 2:
                    try:
                        k_in = int(inp.shape[-1]) if inp.ndim >= 1 else 0
                        m = int(inp.numel() // k_in) if k_in else int(inp.shape[0])
                        _record(m, int(weight.shape[0]), int(weight.shape[1]), src="linear")
                    except Exception:
                        pass
            return _orig_linear(inp, weight, bias)

        def mm(a, b, *args, **kwargs):
            if torch.is_tensor(a) and torch.is_tensor(b) and a.ndim == 2 and b.ndim == 2:
                try:
                    _record(int(a.shape[0]), int(b.shape[1]), int(a.shape[1]), src="mm")
                except Exception:
                    pass
            return _orig_mm(a, b, *args, **kwargs)

        F.linear = linear  # type: ignore[assignment]
        torch.nn.functional.linear = linear  # type: ignore[assignment]
        torch.mm = mm  # type: ignore[assignment]
        atexit.register(_flush)
        try:
            os.makedirs(_OUT_DIR, exist_ok=True)
            with open(os.path.join(_OUT_DIR, "phase8-hook-installed.txt"), "w") as fh:
                fh.write(f"pid={os.getpid()} py={sys.executable}\n")
        except OSError:
            pass

    try:
        _install()
    except Exception as exc:
        sys.stderr.write(f"[linear-hook] install failed: {exc}\n")
