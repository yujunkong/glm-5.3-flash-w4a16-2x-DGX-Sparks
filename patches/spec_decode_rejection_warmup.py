# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up spec-decode rejection-sampler Triton kernels.

The rejection sampler kernels (``_compute_local_logits_stats_kernel``,
``_rejection_kernel``, ``_resample_kernel``) are JIT-compiled by Triton on
first use. Without warmup, the first spec-decode request pays a multi-second
compilation cost. This pre-compiles them with dummy data matching the
server's vocab size and speculative config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@torch.inference_mode()
def spec_decode_rejection_warmup(worker: Worker) -> None:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    model_config = worker.vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    num_spec = spec_config.num_speculative_tokens
    if num_spec <= 0 or vocab_size <= 0:
        _warmup_prepare_dflash_inputs_kernel(worker)
        return

    # Mirror the constexpr-relevant flags the runtime uses.
    rejection_method = getattr(spec_config, "rejection_sample_method", None)
    use_block_verification = rejection_method == "block"
    use_synthetic = rejection_method == "synthetic"

    device = torch.device("cuda")
    num_reqs = 1
    tokens_per_req = num_spec + 1
    num_logits = num_reqs * tokens_per_req

    # Triton JIT-specializes on tensor dtypes. The target logits may be fp32
    # (apply_sampling_params copies to fp32 when processing is needed) or the
    # model dtype (pass-through otherwise), while draft logits are always the
    # model dtype. Warm every (target, draft) combination the runtime can hit.
    model_dtype = model_config.dtype
    warmup_dtype_pairs = {
        (model_dtype, model_dtype),
        (torch.float32, torch.float32),
        (torch.float32, model_dtype),
        (model_dtype, torch.float32),
    }

    logger.info(
        "Warming up spec-decode rejection sampler kernels "
        "(vocab=%d, num_spec=%d, dtype_pairs=%s, block_verify=%s).",
        vocab_size,
        num_spec,
        [(str(t), str(d)) for t, d in warmup_dtype_pairs],
        use_block_verification,
    )
    for tgt_dtype, draft_dtype in warmup_dtype_pairs:
        target_logits = torch.zeros(
            (num_logits, vocab_size), dtype=tgt_dtype, device=device
        )
        draft_logits = torch.zeros(
            (num_reqs, num_spec, vocab_size), dtype=draft_dtype, device=device
        )
        synthetic_rates = (
            torch.full((num_spec,), 0.5, dtype=torch.float32, device=device)
            if use_synthetic
            else None
        )
        try:
            rejection_sample(
                target_logits=target_logits,
                draft_logits=draft_logits,
                draft_sampled=torch.zeros(num_logits, dtype=torch.int64, device=device),
                cu_num_logits=torch.tensor(
                    [0, num_logits], dtype=torch.int32, device=device
                ),
                pos=torch.zeros(num_logits, dtype=torch.int64, device=device),
                idx_mapping=torch.zeros(num_reqs, dtype=torch.int32, device=device),
                expanded_idx_mapping=torch.zeros(
                    num_logits, dtype=torch.int32, device=device
                ),
                expanded_local_pos=torch.arange(
                    num_logits, dtype=torch.int32, device=device
                ),
                temperature=torch.zeros(num_reqs, dtype=torch.float32, device=device),
                seed=torch.full((num_reqs,), 42, dtype=torch.int64, device=device),
                num_speculative_steps=num_spec,
                synthetic_conditional_rates=synthetic_rates,
                use_fp64=False,
                use_block_verification=use_block_verification,
            )
        except Exception:
            logger.warning(
                "Skipping spec-decode rejection sampler warmup.", exc_info=True
            )
            break

    # Runtime uses int64 idx_mapping, DFlash2 sparse fp32 draft_logits of
    # shape [max_num_reqs, K, V], and the sampler's UVA temperature/seeds.
    _warmup_rejection_runtime_shapes(worker)

    # DFlash prepare kernel is skipped by dummy/profile runs, so the first
    # real request otherwise pays Triton JIT. Compile the BLOCK_SIZE ladder
    # that decode + OpenCode prefills actually hit.
    _warmup_prepare_dflash_inputs_kernel(worker)
    _warmup_copy_page_indices_kernel(worker)


def _dflash_prepare_block_sizes(num_query_per_req: int) -> list[tuple[int, int, int]]:
    """Unique (query_len, BLOCK_SIZE, num_blocks) for the prepare kernel.

    Runtime: BLOCK_SIZE = min(256, next_power_of_2(scheduled + num_query_per_req)).
    Decode scheduled=1 -> 16; 16 -> 32; 32 -> 64; 64 -> 128; 128+ -> 256.
    """
    from vllm.triton_utils import triton

    seen: set[int] = set()
    ladder: list[tuple[int, int, int]] = []
    for query_len in (1, 16, 32, 64, 128):
        max_tokens_per_req = query_len + num_query_per_req
        block_size = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
        if block_size in seen:
            continue
        seen.add(block_size)
        ladder.append(
            (query_len, block_size, triton.cdiv(max_tokens_per_req, block_size))
        )
    return ladder


@torch.inference_mode()
def _warmup_prepare_dflash_inputs_kernel(worker: Worker) -> None:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None or getattr(spec_config, "method", None) != "dflash":
        return

    runner = getattr(worker, "model_runner", None)
    speculator = getattr(runner, "speculator", None)
    if runner is None or speculator is None:
        return
    group_ids = getattr(speculator, "draft_kv_cache_group_ids", None)
    if not group_ids:
        return
    block_tables = getattr(speculator, "block_tables", None)
    req_states = getattr(runner, "req_states", None)
    sampler = getattr(runner, "sampler", None)
    target_buffers = getattr(runner, "input_buffers", None)
    if block_tables is None or req_states is None or sampler is None:
        return
    if target_buffers is None:
        return

    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

    gid = group_ids[0]
    device = speculator.device
    ladder = _dflash_prepare_block_sizes(speculator.num_query_per_req)

    logger.info(
        "Warming up DFlash _prepare_dflash_inputs_kernel "
        "(query_lens=%s, BLOCK_SIZE=%s).",
        [q for q, _, _ in ladder],
        [b for _, b, _ in ladder],
    )

    # Use the target runner tensors so Triton specialization matches decode.
    last_sampled = req_states.last_sampled_tokens
    next_prefill = req_states.next_prefill_tokens
    temperature = sampler.sampling_states.temperature.gpu
    seeds = sampler.sampling_states.seeds.gpu
    num_sampled = torch.ones(1, dtype=torch.int32, device=device)
    num_rejected = torch.zeros(1, dtype=torch.int32, device=device)

    try:
        for query_len, _, _ in ladder:
            input_batch = InputBatch.make_dummy(1, query_len, target_buffers)
            prepare_dflash_inputs(
                speculator.input_buffers,
                block_tables.slot_mappings[gid],
                speculator.context_positions,
                speculator._context_slot_mappings[0],
                speculator.sample_indices,
                speculator.sample_pos,
                speculator.sample_idx_mapping,
                speculator.temperature,
                speculator.seeds,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill,
                temperature,
                seeds,
                block_tables.input_block_tables[gid],
                block_tables.kernel_block_sizes[gid],
                speculator.parallel_drafting_token_id,
                speculator.num_query_per_req,
                speculator.num_speculative_steps,
                speculator.max_num_reqs,
                speculator.max_num_tokens,
                speculator.max_model_len,
                speculator.sample_from_anchor,
            )
        torch.cuda.synchronize()
    except Exception:
        logger.warning(
            "Skipping DFlash _prepare_dflash_inputs_kernel warmup.",
            exc_info=True,
        )


@torch.inference_mode()
def _warmup_rejection_runtime_shapes(worker: Worker) -> None:
    """Compile rejection kernels against DFlash2 runtime dtypes and batch sizes.

    Do not cast live runtime tensors. Dummy buffers are created with the same
    dtypes the decode path actually passes: temp=0 keeps target logits in the
    model dtype (bf16 here); processed requests use fp32. idx_mapping is int64.
    """
    runner = getattr(worker, "model_runner", None)
    speculator = getattr(runner, "speculator", None)
    sampler = getattr(runner, "sampler", None)
    if runner is None or speculator is None or sampler is None:
        return
    draft_logits = getattr(speculator, "draft_logits", None)
    device = speculator.device
    num_spec = speculator.num_speculative_steps
    tokens_per = num_spec + 1
    vocab_size = worker.vllm_config.model_config.get_vocab_size()
    if draft_logits is not None:
        vocab_size = min(vocab_size, int(draft_logits.size(-1)))
    model_dtype = worker.vllm_config.model_config.dtype
    max_seqs = int(
        getattr(worker.vllm_config.scheduler_config, "max_num_seqs", 1) or 1
    )
    max_seqs = max(1, min(max_seqs, int(speculator.max_num_reqs)))

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    dtypes = []
    for dt in (model_dtype, torch.float32):
        if dt not in dtypes:
            dtypes.append(dt)
    req_counts = []
    for n in (1, max_seqs):
        if n not in req_counts:
            req_counts.append(n)

    logger.info(
        "Warming up rejection sampler with runtime tensors "
        "(draft_logits=%s, idx_mapping=int64, dtypes=%s, num_reqs=%s, temp=[0,1]).",
        None if draft_logits is None else tuple(draft_logits.shape),
        [str(d) for d in dtypes],
        req_counts,
    )
    temp_src = sampler.sampling_states.temperature.gpu
    seed_src = sampler.sampling_states.seeds.gpu
    try:
        for n_req in req_counts:
            num_logits = n_req * tokens_per
            idx_mapping = torch.arange(n_req, dtype=torch.int64, device=device)
            cu_num_logits = torch.arange(
                0, num_logits + 1, tokens_per, dtype=torch.int32, device=device
            )
            expanded_idx_mapping = idx_mapping.repeat_interleave(tokens_per)
            expanded_local_pos = torch.arange(
                tokens_per, dtype=torch.int32, device=device
            ).repeat(n_req)
            for tgt_dtype in dtypes:
                target = torch.zeros(
                    (num_logits, vocab_size), dtype=tgt_dtype, device=device
                )
                # Compile greedy (temp=0) and Leviathan (temp=1) paths separately.
                # Do not mutate the live sampler temperature buffer.
                for temp_val in (0.0, 1.0):
                    temperature = torch.full_like(temp_src, temp_val)
                    rejection_sample(
                        target_logits=target,
                        draft_logits=draft_logits,
                        draft_sampled=torch.zeros(
                            num_logits, dtype=torch.int64, device=device
                        ),
                        cu_num_logits=cu_num_logits,
                        pos=torch.zeros(num_logits, dtype=torch.int64, device=device),
                        idx_mapping=idx_mapping,
                        expanded_idx_mapping=expanded_idx_mapping,
                        expanded_local_pos=expanded_local_pos,
                        temperature=temperature,
                        seed=seed_src,
                        num_speculative_steps=num_spec,
                        synthetic_conditional_rates=None,
                        use_fp64=bool(getattr(sampler, "use_fp64_gumbel", False)),
                        use_block_verification=False,
                    )
        torch.cuda.synchronize()
    except Exception:
        logger.warning(
            "Skipping runtime-shape rejection sampler warmup.",
            exc_info=True,
        )


@torch.inference_mode()
def _warmup_copy_page_indices_kernel(worker: Worker) -> None:
    """Hybrid MLA/FlashInfer skips the all-FLASHINFER dummy_run path."""
    runner = getattr(worker, "model_runner", None)
    block_tables = getattr(runner, "block_tables", None)
    tables = getattr(block_tables, "input_block_tables", None) if block_tables else None
    if not tables:
        return

    from vllm.v1.attention.backends.flashinfer import _copy_page_indices_kernel

    device = runner.device
    warmed = 0
    try:
        for table in tables:
            if not isinstance(table, torch.Tensor) or table.ndim != 2:
                continue
            num_pages = min(8, int(table.shape[1]) or 1)
            indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
            indices = torch.zeros(max(num_pages, 1), dtype=table.dtype, device=device)
            _copy_page_indices_kernel[(1,)](
                indices,
                table,
                table.stride(0),
                indptr,
                BLOCK_SIZE=1024,
            )
            warmed += 1
        if warmed:
            torch.cuda.synchronize()
            logger.info(
                "Warmed _copy_page_indices_kernel for %s FlashInfer block tables.",
                warmed,
            )
    except Exception:
        logger.warning(
            "Skipping _copy_page_indices_kernel warmup.",
            exc_info=True,
        )
