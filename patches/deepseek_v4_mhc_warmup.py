# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up mHC TileLang kernels before serving requests.

Stock file is DSv4-only (`model_type == deepseek_v4` returns immediately for
GLM). GLM-5.3-Flash uses the same `mhc_pre_big_fuse_with_norm_tilelang` kernel
via `Glm5NextDecoderLayer.hc_pre` / `hc_fused_post_pre`. TileLang specializes
on `n_splits = f(num_tokens)` (and the fused path's T<=16 vs T>16 branch),
so a dummy_run of 16 tokens does not cover C2/C6 prefill.

This overlay keeps the DSv4 path and adds a GLM path that calls the live
layer ops with the runtime dtypes/weights. Token sizes are the real compile
axis, not dummy hidden/dtype.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    48,  # C6 decode: max_num_seqs=6 * DFlash query (1+K=8)
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)

_GLM_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})
_DSV4_MODEL_TYPES = frozenset({"deepseek_v4"})


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


def _find_first_dsv4_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_post",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None


def _find_first_glm_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """First Glm5NextDecoderLayer that actually runs mHC (skips MTP)."""
    for module in model.modules():
        if module.__class__.__name__ != "Glm5NextDecoderLayer":
            continue
        if not getattr(module, "mhc", False) or getattr(module, "is_mtp_layer", False):
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_fused_post_pre",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
                "input_layernorm",
                "post_attention_layernorm",
                "n",
                "hidden_size",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_glm_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    """Compile GLM standalone hc_pre and fused post+pre at runtime token sizes.

    Layer 0 calls ``hc_pre`` with fused RMSNorm (this JIT name). Later layers
    call ``hc_fused_post_pre``, which uses the same with_norm kernel only when
    ``num_tokens > 16``; T<=16 takes ``mhc_fused_tilelang`` instead.
    """
    max_tokens = max(token_sizes)
    n = int(layer.n)
    hidden_size = int(layer.hidden_size)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens, n, hidden_size, dtype=torch.bfloat16, device=device
    )
    x_attn = torch.zeros(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    in_w = layer.input_layernorm.weight.data
    post_w = layer.post_attention_layernorm.weight.data
    in_eps = float(layer.input_layernorm.variance_epsilon)
    post_eps = float(layer.post_attention_layernorm.variance_epsilon)

    for size in token_sizes:
        residual_slice = residual[:size]
        x_slice = x_attn[:size]
        # Standalone pre = layer 0 path (always with_norm when norm_weight set).
        post, comb, _layer_in = layer.hc_pre(
            residual_slice,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            norm_weight=in_w,
            norm_eps=in_eps,
        )
        # Fused post+pre = later-layer / post-attn path. T>16 hits with_norm.
        layer.hc_fused_post_pre(
            x_slice,
            residual_slice,
            post,
            comb,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            norm_weight=post_w,
            norm_eps=post_eps,
        )


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    # Upstream a8887c208 ("[DSV4] aiter mhc support (ROCm)") refactored
    # ``hc_head`` from a free function into the ``HCHeadOp`` CustomOp
    # instance attached to the model as ``hc_head_op``. We call through
    # that instance so the warmup exercises the same dispatched
    # implementation as the inference path.
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


def _config_model_type(model: torch.nn.Module) -> str | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    mt = getattr(config, "model_type", None)
    if mt is not None:
        return str(mt)
    text = getattr(config, "text_config", None)
    return getattr(text, "model_type", None) if text is not None else None


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    model_type = _config_model_type(model)
    is_glm = model_type in _GLM_MODEL_TYPES
    is_dsv4 = model_type in _DSV4_MODEL_TYPES or model_type is None
    if model_type is not None and not is_glm and not is_dsv4:
        return

    glm_layer = _find_first_glm_mhc_layer(model) if is_glm else None
    dsv4_layer = _find_first_dsv4_mhc_layer(model) if not is_glm else None
    layer = glm_layer or dsv4_layer
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up mHC TileLang kernels (model_type=%s, token sizes: %s).",
        model_type,
        token_sizes,
    )
    with torch.inference_mode():
        if glm_layer is not None:
            _warmup_glm_mhc(glm_layer, token_sizes)
        else:
            _warmup_layer_mhc(layer, token_sizes)
            deepseek_model = _find_deepseek_v4_model(model)
            if deepseek_model is not None:
                _warmup_hc_head(deepseek_model, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
