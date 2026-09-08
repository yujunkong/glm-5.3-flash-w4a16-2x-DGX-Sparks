#!/usr/bin/env python3
"""Per-request DFlash2 acceptance + first-reject A/B/C split.

Runs on the host against a live server (temperature=0). Snapshots /metrics and
the probe file written by patches/dflash2_speculator.py:

    A  lm_head_topk_miss          target not in compute_candidates() pool
    B  lm_head_topk_hit_walk_miss target in pool, walk picked someone else
    C  walk_hit_verify_reject     walk token == target but verifier rejected

Does not change TOP_K, scores, walk, or rejection.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Write a compact Python function that parses a log line, validates fields, and returns a typed dictionary. Explain edge cases after the code.",
    "Write a compact Python function that normalizes a CSV row, coerces types, and returns a dataclass. Explain edge cases after the code.",
    "Write a compact Python function that parses an INI-style config, validates keys, and returns a dict. Explain edge cases after the code.",
    "Write a compact Python function that parses a JSON payload, validates required fields, and returns a typed object. Explain edge cases after the code.",
    "Write a compact Python function that parses a key=value env string, validates values, and returns a dict. Explain edge cases after the code.",
    "Write a compact Python function that parses a syslog line, extracts the fields, and returns a struct. Explain edge cases after the code.",
    "Write a compact Python function that parses a semicolon-separated record, validates fields, and returns a tuple. Explain edge cases after the code.",
    "Write a compact Python function that parses a tab-separated line, validates columns, and returns a list. Explain edge cases after the code.",
    "Write a compact Python function that tokenizes a Makefile rule, validates targets, and returns a graph node. Explain edge cases after the code.",
    "Write a compact Python function that parses a YAML-like indent list, validates keys, and returns nested dicts. Explain edge cases after the code.",
]

PROBE_CANDIDATES = [
    Path("/var/tmp/glm53-w4a16-cache/dflash2-acc-state.json"),
    Path("/tmp/dflash2-acc-state.json"),
]


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p10": round(percentile(vals, 0.10), 4),
        "p25": round(percentile(vals, 0.25), 4),
        "p75": round(percentile(vals, 0.75), 4),
        "p90": round(percentile(vals, 0.90), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def get_metrics(api: str) -> dict:
    with urllib.request.urlopen(api.rstrip("/") + "/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(
            r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*?position="(\d+)"[^}]*\} ([0-9.eE+]+)$',
            line,
        )
        if m:
            out[f"per_pos{m.group(1)}"] = float(m.group(2))
            continue
        m = re.match(r"^(vllm:[a-z_]+)(?:\{[^}]*\})? ([0-9.eE+]+)$", line)
        if m and m.group(1) not in out:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def read_probe() -> dict | None:
    for path in PROBE_CANDIDATES:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return None


def probe_cum(blob: dict | None) -> dict:
    if not blob:
        return {}
    return dict(blob.get("cum") or {})


def complete(api: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        api.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read().decode())
    wall = time.perf_counter() - t0
    usage = out.get("usage") or {}
    return {
        "wall_s": round(wall, 3),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def delta_num(a: dict, b: dict, key: str, default: float = 0.0) -> float:
    return float(b.get(key, default)) - float(a.get(key, default))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("warmup (discarded)...", flush=True)
    complete(a.base, a.model, PROMPTS[0] + "\n# warmup", min(a.max_tokens, 64))

    rows = []
    for i in range(a.runs):
        prompt = PROMPTS[i % len(PROMPTS)] + f"\n# salt={i}"
        m0 = get_metrics(a.base)
        p0 = probe_cum(read_probe())
        done = complete(a.base, a.model, prompt, a.max_tokens)
        m1 = get_metrics(a.base)
        p1 = probe_cum(read_probe())

        drafts = delta_num(m0, m1, "vllm:spec_decode_num_drafts_total")
        accepted = delta_num(m0, m1, "vllm:spec_decode_num_accepted_tokens_total")
        drafted = delta_num(m0, m1, "vllm:spec_decode_num_draft_tokens_total")
        per_pos = [
            delta_num(m0, m1, f"per_pos{j}") for j in range(7)
        ]
        skip_lists = {"pos_accept", "reject_at_pos"}
        probe = {
            k: delta_num(p0, p1, k)
            for k in set(p0) | set(p1)
            if k not in skip_lists and not isinstance((p1 or p0).get(k), list)
        }
        pos_a = p0.get("pos_accept") or [0] * 7
        pos_b = p1.get("pos_accept") or [0] * 7
        if isinstance(pos_a, list) and isinstance(pos_b, list):
            probe["pos_accept"] = [
                int(pos_b[j]) - int(pos_a[j]) if j < len(pos_a) and j < len(pos_b) else 0
                for j in range(7)
            ]
        rej_a = p0.get("reject_at_pos") or [0] * 7
        rej_b = p1.get("reject_at_pos") or [0] * 7
        reject_at_pos = [0] * 7
        if isinstance(rej_a, list) and isinstance(rej_b, list):
            reject_at_pos = [
                int(rej_b[j]) - int(rej_a[j]) if j < len(rej_a) and j < len(rej_b) else 0
                for j in range(7)
            ]
        rejected = probe.get("rejected_rounds") or probe.get("rejected") or 0
        row = {
            "request_id": i,
            "acceptance_ratio": round(accepted / drafted, 4) if drafted else None,
            "accepted_count": accepted,
            "drafted_count": drafted,
            "drafts": drafts,
            "pos0_accept": per_pos[0],
            "pos1_accept": per_pos[1],
            "pos2_accept": per_pos[2],
            "pos3_accept": per_pos[3],
            "pos4_accept": per_pos[4],
            "pos5_accept": per_pos[5],
            "pos6_accept": per_pos[6],
            "lm_head_topk_miss": probe.get("lm_head_topk_miss", 0),
            "lm_head_topk_hit_walk_miss": probe.get("lm_head_topk_hit_walk_miss", 0),
            "walk_hit_verify_reject": probe.get("walk_hit_verify_reject", 0),
            "b_unary_would_hit": probe.get("b_unary_would_hit", 0),
            "b_unary_also_miss": probe.get("b_unary_also_miss", 0),
            "b_tgt_unary_rank0": probe.get("b_tgt_unary_rank0", 0),
            "b_tgt_unary_rank_le2": probe.get("b_tgt_unary_rank_le2", 0),
            "b_tgt_unary_rank_le7": probe.get("b_tgt_unary_rank_le7", 0),
            "b_tgt_edge_rank_sum": probe.get("b_tgt_edge_rank_sum", 0),
            "b_tgt_edge_rank_n": probe.get("b_tgt_edge_rank_n", 0),
            "reject_at_pos": reject_at_pos,
            "rejected_rounds": rejected,
            "in_topk": probe.get("in_topk", 0),
            "not_in_topk": probe.get("not_in_topk", 0),
            **done,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    ratios = [r["acceptance_ratio"] for r in rows if r["acceptance_ratio"] is not None]
    a_sum = sum(r["lm_head_topk_miss"] for r in rows)
    b_sum = sum(r["lm_head_topk_hit_walk_miss"] for r in rows)
    c_sum = sum(r["walk_hit_verify_reject"] for r in rows)
    b1 = sum(r["b_unary_would_hit"] for r in rows)
    b2 = sum(r["b_unary_also_miss"] for r in rows)
    rank0 = sum(r["b_tgt_unary_rank0"] for r in rows)
    rank_le2 = sum(r["b_tgt_unary_rank_le2"] for r in rows)
    rank_le7 = sum(r["b_tgt_unary_rank_le7"] for r in rows)
    edge_n = sum(r["b_tgt_edge_rank_n"] for r in rows)
    edge_sum = sum(r["b_tgt_edge_rank_sum"] for r in rows)
    rej = sum(r["rejected_rounds"] for r in rows)
    classified = a_sum + b_sum + c_sum
    pos_means = []
    draft_sum = sum(r["drafts"] for r in rows)
    for j in range(7):
        hits = sum(r[f"pos{j}_accept"] for r in rows)
        pos_means.append(round(hits / draft_sum, 4) if draft_sum else None)

    summary = {
        "topk_definition": (
            "lm_head_topk_hit uses compute_candidates() pool "
            "(DFLASH_SELECTOR_TOP_K), not a second edge-score top-k"
        ),
        "runs": len(rows),
        "acceptance_ratio": summarize(ratios),
        "mean_acceptance_by_position": pos_means,
        "first_reject_split": {
            "rejected_rounds": rej,
            "classified": classified,
            "A_lm_head_topk_miss": a_sum,
            "B_lm_head_topk_hit_walk_miss": b_sum,
            "C_walk_hit_verify_reject": c_sum,
            "A_pct": round(100.0 * a_sum / classified, 2) if classified else None,
            "B_pct": round(100.0 * b_sum / classified, 2) if classified else None,
            "C_pct": round(100.0 * c_sum / classified, 2) if classified else None,
        },
        "B_walk_debug": {
            "B1_unary_would_hit": b1,
            "B2_unary_also_miss": b2,
            "B1_pct_of_B": round(100.0 * b1 / b_sum, 2) if b_sum else None,
            "B2_pct_of_B": round(100.0 * b2 / b_sum, 2) if b_sum else None,
            "tgt_unary_rank0_pct": round(100.0 * rank0 / b_sum, 2) if b_sum else None,
            "tgt_unary_rank_le2_pct": round(100.0 * rank_le2 / b_sum, 2) if b_sum else None,
            "tgt_unary_rank_le7_pct": round(100.0 * rank_le7 / b_sum, 2) if b_sum else None,
            "tgt_edge_rank_mean": round(edge_sum / edge_n, 3) if edge_n else None,
            "reject_at_pos": [
                sum(r.get("reject_at_pos", [0] * 7)[j] for r in rows) for j in range(7)
            ],
        },
        "verdict_hint": (
            "A high → try TOP_K 40/48; "
            "B1 high → unary argmax walk A/B (edge override); "
            "B2 high → target is not unary#1, raising TOP_K or dropping edges won't recover most rejects; "
            "C high → inspect verifier/sampling"
        ),
    }
    print(json.dumps({"summary": True, **summary}, indent=2), flush=True)
    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
