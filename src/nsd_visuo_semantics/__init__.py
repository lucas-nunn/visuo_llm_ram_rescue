"""nsd_visuo_semantics package initialisation.

On import we preload the pip-installed NVIDIA CUDA 12 shared libraries so that
TensorFlow can use the GPU.

Why this is needed: TensorFlow's ``tensorflow[and-cuda]`` manylinux wheels do
not embed an RPATH pointing at the ``nvidia-*-cu12`` packages in site-packages,
so on a plain ``import tensorflow`` it fails to ``dlopen`` libcudart / libcublas
/ libcudnn and silently falls back to CPU (the "Cannot dlopen some GPU
libraries" warning). PyTorch's wheels *do* embed that RPATH, which is why
importing ``torch`` first happens to make TF find the GPU -- but the TF-only
scripts (e.g. the searchlight) never import torch.

Preloading the libraries here with ``RTLD_GLOBAL`` puts their symbols in the
global namespace, so TensorFlow's later ``dlopen``-by-soname succeeds without
having to set ``LD_LIBRARY_PATH``. The import chain guarantees this runs before
TensorFlow is imported: importing any ``nsd_visuo_semantics.*`` submodule (which
is how every TF script/notebook in this repo starts) executes this file first.
"""

import ctypes
import glob
import os


def _preload_cuda_libraries():
    """Load every NVIDIA CUDA .so from site-packages with RTLD_GLOBAL.

    No-op on CPU-only installs (no ``nvidia`` package). Failures are tolerated:
    some libraries depend on siblings in other ``nvidia/*/lib`` dirs that are
    not on each other's RUNPATH, so we retry in a few passes until no further
    progress is made.
    """
    try:
        import nvidia
    except ImportError:
        return

    lib_dirs = [
        d for base in nvidia.__path__ for d in glob.glob(os.path.join(base, "*", "lib"))
    ]
    pending = sorted(
        f for d in lib_dirs for f in glob.glob(os.path.join(d, "*.so*"))
    )
    for _ in range(3):
        if not pending:
            break
        still_pending = []
        for so in pending:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                still_pending.append(so)
        if len(still_pending) == len(pending):
            break  # no progress this pass; remaining libs are unloadable
        pending = still_pending


_preload_cuda_libraries()

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
