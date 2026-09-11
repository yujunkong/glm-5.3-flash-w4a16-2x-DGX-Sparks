#!/usr/bin/env python3
"""Read-only safetensors dtype/shape dump. Does not load full tensors."""
from collections import Counter
import json
from safetensors import safe_open


def inspect_file(path: str, title: str, filt) -> None:
    print(f"\n==== {title} ====")
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print("n_tensors", len(keys))
        dtypes: Counter[str] = Counter()
        shown = []
        for k in keys:
            sl = f.get_slice(k)
            shape = tuple(sl.get_shape())
            dt = str(sl.get_dtype())
            dtypes[dt] += 1
            if filt(k, shape, dt):
                shown.append((dt, shape, k))
        print("dtype_hist", dict(dtypes.most_common()))
        print("shown", len(shown))
        for dt, shape, k in shown[:80]:
            print(f"  {dt:12s} {str(shape):32s} {k}")


inspect_file(
    "/dflash/model.safetensors",
    "DFlash2 drafter",
    lambda k, s, d: any(
        x in k
        for x in (
            "mlp.",
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "embed",
            "lm_head",
            "weight_scale",
        )
    ),
)

idx = json.load(open("/tgt/model.safetensors.index.json"))
wm = idx["weight_map"]
need: dict[str, list[str]] = {}
for k, shard in wm.items():
    if any(
        x in k
        for x in (
            "layers.0.mlp.",
            "layers.3.mlp.shared_experts",
            "layers.3.mlp.experts.0.",
            "layers.3.mlp.gate.weight",
            "layers.5.mlp.experts.0.w",
        )
    ):
        need.setdefault(shard, []).append(k)

for sh, ks in need.items():
    print(f"\n==== target {sh} ({len(ks)} keys) ====")
    with safe_open(f"/tgt/{sh}", framework="pt", device="cpu") as f:
        for k in sorted(ks):
            sl = f.get_slice(k)
            print(f"  {str(sl.get_dtype()):12s} {str(tuple(sl.get_shape())):32s} {k}")
