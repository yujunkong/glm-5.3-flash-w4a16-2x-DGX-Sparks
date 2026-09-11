#!/usr/bin/env python3
"""Run FlashInfer AOT with an explicit C++ project root.

`python -m flashinfer.aot` sets project_root = Path(__file__).parents[1], which
for the installed wheel is site-packages. That collides with other packages'
`include/` and misses `data/csrc`. This driver points nvcc at the wheel's
`flashinfer/data` layout instead.
"""
from __future__ import annotations

import os
from pathlib import Path

from flashinfer.aot import compile_and_package_modules, get_default_config


def main() -> None:
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.1a")
    # CUDA 13 toolkit image ships NVRTC runtime only; headers live in the pip
    # nvidia-cu13 package. fused_moe deepgemm_jit_setup.cu includes <nvrtc.h>.
    nvrtc_inc = Path("/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include")
    if (nvrtc_inc / "nvrtc.h").is_file():
        for key in ("CPATH", "CPLUS_INCLUDE_PATH"):
            prev = os.environ.get(key, "")
            os.environ[key] = f"{nvrtc_inc}:{prev}" if prev else str(nvrtc_inc)
        cuda_inc = Path("/usr/local/cuda/include")
        dest = cuda_inc / "nvrtc.h"
        if not dest.exists():
            dest.symlink_to(nvrtc_inc / "nvrtc.h")
    # CUDA 13 ships versioned libnvrtc.so.13 only; fused_moe links -lnvrtc.
    lib64 = Path("/usr/local/cuda/lib64")
    nvrtc_so = lib64 / "libnvrtc.so.13"
    if nvrtc_so.exists() and not (lib64 / "libnvrtc.so").exists():
        (lib64 / "libnvrtc.so").symlink_to(nvrtc_so.name)
    builtins = lib64 / "libnvrtc-builtins.so.13.0"
    if builtins.exists() and not (lib64 / "libnvrtc-builtins.so").exists():
        (lib64 / "libnvrtc-builtins.so").symlink_to(builtins.name)
    data = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/data")
    root = Path(os.environ.get("FI_SRC", "/work/fi-src"))
    (root / "3rdparty").mkdir(parents=True, exist_ok=True)
    for name in ("csrc", "include"):
        dest = root / name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(data / name)
    for name in ("cutlass", "spdlog", "cccl"):
        dest = root / "3rdparty" / name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(data / name)

    cfg = get_default_config()
    cfg.update(
        add_gemma=False,
        add_oai_oss=False,
        add_xqa=False,
        add_moe=True,
        add_act=True,
        add_misc=True,
        add_comm=True,
    )
    compile_and_package_modules(
        out_dir=Path(os.environ.get("FI_AOT_OUT", "/work/fi-aot")),
        build_dir=Path(os.environ.get("FI_AOT_BUILD", "/work/fi-build")),
        project_root=root,
        config=cfg,
        verbose=True,
        skip_prebuilt=False,
    )


if __name__ == "__main__":
    main()
