#!/usr/bin/env python3
"""Attribute live cutlass_80 16x16_128x1 launches via nsys Python backtrace.

Read-only. Does not start serve. Source:
  /var/tmp/glm53-w4a16-cache/sm121-audit-p7/phase7-live-blobs.sqlite
"""
from __future__ import annotations

import sqlite3
import struct
from collections import defaultdict

DB = "/var/tmp/glm53-w4a16-cache/sm121-audit-p7/phase7-live-blobs.sqlite"
NEED = "s161616gemm_bf16_16x16_128x1"
TOTAL_CUDA_NS = 1_213_300_000  # phase7-live-kern-sum window


def parse_py(strings: dict, blob: bytes) -> list[tuple[str, str, int]]:
    n = struct.unpack_from("<I", blob, 16 + 24 + 4)[0]
    off = 16 + 24 + 8
    frames = []
    for _ in range(n):
        if off + 24 > len(blob):
            break
        fn, fl, line, _ = struct.unpack_from("<QQII", blob, off)
        frames.append((strings.get(fn, "?"), strings.get(fl, "?"), line))
        off += 24
    return frames


def classify(frames: list[tuple[str, str, int]]) -> str:
    files = " ".join(f[1] for f in frames)
    lines = {(f[1].split("/")[-1], f[2]) for f in frames}
    if "kda.py" in files:
        # Line numbers from this image's kda.py (Python BT).
        if ("kda.py", 327) in lines:
            return "KDA in_proj_qkvbfg_a"
        if ("kda.py", 371) in lines:
            return "KDA o_proj"
        if ("kda.py", 344) in lines:
            return "KDA f_b/g_b"
        return "KDA other"
    if "qwen3_dflash2.py" in files or "qwen3_dflash.py" in files:
        if ("qwen3_dflash2.py", 177) in lines:
            return "DFlash MLP gate_up"
        if ("qwen3_dflash.py", 271) in lines:
            return "DFlash MLP down"
        if ("qwen3_dflash.py", 782) in lines:
            return "DFlash combine"
        if ("qwen3_dflash.py", 533) in lines:
            return "DFlash project_context_kv"
        return "DFlash other"
    if "mla.py" in files or "attention.py" in files:
        if ("mla.py", 245) in lines:
            return "MLA o_proj"
        return "MLA other"
    if "model.py" in files:
        if ("model.py", 147) in lines:
            return "target dense gate_up"
        if ("model.py", 149) in lines:
            return "target MLP down_proj"
        return "target MLP other"
    top = frames[0] if frames else ("?", "?", 0)
    return f"other {top[1].split('/')[-1]}:{top[2]}"


def main() -> None:
    c = sqlite3.connect(DB)
    strings = {i: v for i, v in c.execute("SELECT id,value FROM StringIds")}

    bts = []
    for start, blob in c.execute(
        "SELECT start, binaryData FROM NVTX_EVENTS WHERE domainId=2 AND binaryData IS NOT NULL"
    ):
        bts.append((start, parse_py(strings, blob)))
    bts.sort()
    starts = [x[0] for x in bts]

    rt = {
        cid: st
        for st, cid in c.execute(
            "SELECT start,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }

    import bisect

    buckets: dict[str, list] = defaultdict(list)
    n_miss = 0
    # shortName is just "Kernel2"; the tile lives in demangledName.
    for row in c.execute(
        "SELECT start, end, correlationId, gridX, gridY, gridZ, demangledName "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        start, end, cid, gx, gy, gz, sid = row
        name = strings.get(sid, "")
        if NEED not in name:
            continue
        rst = rt.get(cid)
        if rst is None:
            n_miss += 1
            continue
        i = bisect.bisect_right(starts, rst) - 1
        frames = bts[i][1] if i >= 0 else []
        tag = classify(frames)
        dur = end - start
        buckets[tag].append((dur, gx, gy, gz, frames))

    tot_ns = sum(d for rows in buckets.values() for d, *_ in rows)
    print(f"128x1 launches classified={sum(len(v) for v in buckets.values())} miss_rt={n_miss}")
    print(f"128x1 total ms={tot_ns/1e6:.1f}  CUDA%={100*tot_ns/TOTAL_CUDA_NS:.2f}")
    print()
    print(
        f"{'pctCUDA':>7} {'n':>5} {'avg_us':>8} {'med_us':>8} {'grid':>10} {'N_est':>6}  operation"
    )
    for tag, rows in sorted(buckets.items(), key=lambda kv: -sum(r[0] for r in kv[1])):
        ns = [r[0] for r in rows]
        ns.sort()
        avg = sum(ns) / len(ns)
        med = ns[len(ns) // 2]
        gx, gy, gz = rows[0][1], rows[0][2], rows[0][3]
        n_est = gy * 128
        pct = 100 * sum(ns) / TOTAL_CUDA_NS
        print(
            f"{pct:7.2f} {len(rows):5d} {avg/1e3:8.1f} {med/1e3:8.1f} "
            f"{gx}x{gy}x{gz:>1} {n_est:6d}  {tag}"
        )
        # sample python caller
        fr = rows[0][4]
        for fn, fl, ln in fr[:8]:
            if any(
                x in fl
                for x in (
                    "kda.py",
                    "mla.py",
                    "model.py",
                    "linear.py",
                    "dflash",
                    "qwen3",
                )
            ):
                print(f"         caller {fl.split('/')[-1]}:{ln} {fn}")
                break


if __name__ == "__main__":
    main()
