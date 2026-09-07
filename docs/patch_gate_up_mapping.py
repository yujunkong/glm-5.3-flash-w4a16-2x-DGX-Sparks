#!/usr/bin/env python3
"""Patch vLLM's GLM5Next loader with the fused gate/up mapping.

The GLM-5.3 W4A16 checkpoint stores expert gate/up projections separately,
while vLLM builds a single ``gate_up_proj`` module.  Quantization code needs
the model's packed_modules_mapping to resolve those two checkpoint names to
the fused destination.  Without it, the loader can report missing
``gate_up_proj`` weight tensors even though the checkpoint contains the
corresponding gate/up weights.

This script is intentionally fail-closed: it refuses to patch an unexpected
vLLM source and never silently replaces an already-present mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


MAPPING = '    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}\n'
CLASSES = (
    "Glm5NextForConditionalGeneration",
    "Glm5NextForCausalLM",
)


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")

    if "packed_modules_mapping" in text and '"gate_up_proj": ["gate_proj", "up_proj"]' in text:
        print(f"[gate-up] already patched: {path}")
        return False

    if "packed_modules_mapping" in text:
        raise SystemExit(
            f"[gate-up] REFUSE: packed_modules_mapping already exists but does not "
            f"contain the expected gate/up mapping: {path}"
        )

    changed = 0
    for cls in CLASSES:
        pattern = rf"(?m)^(class {re.escape(cls)}\([^\n]+\):\n)"
        match = re.search(pattern, text)
        if not match:
            continue
        insert_at = match.end()
        text = text[:insert_at] + MAPPING + text[insert_at:]
        changed += 1

    if changed == 0:
        raise SystemExit(
            f"[gate-up] REFUSE: none of the expected GLM5Next classes were found in {path}"
        )

    path.write_text(text, encoding="utf-8")
    print(f"[gate-up] patched {changed} class(es): {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    patch(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
