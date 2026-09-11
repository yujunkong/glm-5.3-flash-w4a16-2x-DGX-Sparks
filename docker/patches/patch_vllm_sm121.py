#!/usr/bin/env python3
"""Patch vLLM CMakeLists.txt so TORCH_CUDA_ARCH_LIST=12.1a survives CUDA 13.0.

On CUDA >= 13.0 this tree's CUDA_SUPPORTED_ARCHS is "...;12.0" with no 12.1,
so cmake/utils.cmake drops 12.1a (vLLM #43003). Marlin/MoE then compile only
12.0f family cubins — which is what the stock sm121-v11 image ships.

SM121-only rebuild needs 12.1 in CUDA_SUPPORTED_ARCHS and 12.1a next to
every CUDA-13 12.0f intersection so Marlin W4A16 actually emits sm_121a.
"""
from __future__ import annotations

import sys
from pathlib import Path

# CUDA 13.0 branch (not the 13.4 Rubin list which has 10.7 and no 12.1).
OLD_SUPPORTED = 'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")'
NEW_SUPPORTED = 'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")'

# Family 12.0f intersections used under CUDA >= 13.0. Append 12.1a so a
# TORCH_CUDA_ARCH_LIST=12.1a-only build still produces Marlin/MoE SASS.
REPLACEMENTS = [
    ('"8.0+PTX;12.0f"', '"8.0+PTX;12.0f;12.1a"'),
    ('"8.0+PTX;9.0+PTX;12.0f"', '"8.0+PTX;9.0+PTX;12.0f;12.1a"'),
    ('"8.9;12.0f"', '"8.9;12.0f;12.1a"'),
    ('"12.0f"', '"12.0f;12.1a"'),
    ('"9.0a;10.0f;10.7f;11.0f;12.0f"', '"9.0a;10.0f;10.7f;11.0f;12.0f;12.1a"'),
    ('"9.0a;10.0f;12.0f"', '"9.0a;10.0f;12.0f;12.1a"'),
]


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "CMakeLists.txt")
    text = path.read_text()
    cuda13 = text.split("VERSION_GREATER_EQUAL 13.0", 1)[-1].split("VERSION_GREATER_EQUAL 12.8", 1)[0]
    if NEW_SUPPORTED in cuda13:
        print(f"{path}: CUDA 13.0 list already has 12.1 — skip supported-archs")
    elif OLD_SUPPORTED not in text:
        print(f"{path}: expected CUDA 13.0 CUDA_SUPPORTED_ARCHS not found", file=sys.stderr)
        return 1
    else:
        # Replace only the first CUDA 13.0 occurrence (13.4 list also matches
        # a similar string but includes 10.7 — OLD_SUPPORTED is unique).
        text = text.replace(OLD_SUPPORTED, NEW_SUPPORTED, 1)

    for old, new in REPLACEMENTS:
        if old not in text:
            print(f"{path}: skip missing {old}")
            continue
        text = text.replace(old, new)
        print(f"{path}: {old} -> {new}")

    # 8.9+PTX matches 12.1a via cuda_archs_loose_intersection PTX-across-majors
    # and would fatbin sm_89 into _C. 12.1a is already covered by c3x_sm120.
    c2x = """  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  # subtract out the archs that are already built for 3x"""
    c2x_fix = """  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  if("${CUDA_ARCHS}" STREQUAL "12.1a")
    set(SCALED_MM_2X_ARCHS)
  endif()
  # subtract out the archs that are already built for 3x"""
    if c2x_fix in text:
        print(f"{path}: C2X 12.1a skip already applied")
    elif c2x in text:
        text = text.replace(c2x, c2x_fix, 1)
        print(f"{path}: skip scaled_mm_c2x sm_89 leak on 12.1a")
    else:
        print(f"{path}: skip missing C2X intersection block")

    path.write_text(text)
    print(f"{path}: patched for SM121")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
