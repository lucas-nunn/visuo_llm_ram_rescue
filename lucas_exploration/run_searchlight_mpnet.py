#!/usr/bin/env python
"""Reproducible runner for the searchlight RDM comparison (adapted from searchlight.ipynb).

This is a scriptable version of the relevant cells of
``lucas_exploration/searchlight.ipynb``. It runs the memory-safe streaming
searchlight correlation, projects completed maps to fsaverage, and can render
the notebook's pycortex brain plot. It also emits basic telemetry (per-stage
timing, RAM + GPU snapshots, and searchlight-sphere diagnostics).

Pipeline stages (mirrors the notebook):
  1. (optional) prepare model RDMs            -> nsd_prepare_modelrdms
  2. diagnose the precomputed searchlight      -> sphere-size telemetry (cheap)
  3. run the searchlight + RDM correlation     -> nsd_searchlight_main_tf
  4. (optional) project and plot fsaverage map -> --plot / --plot-only

Run from the ``lucas_exploration`` directory so the default ``./results``
relative paths line up with the notebook, e.g.:

    ../.venv/bin/python run_searchlight_mpnet.py --subject 2 --n-sessions 20

Quick, no-GPU inspection of the "too many voxel centers" hypothesis:

    ../.venv/bin/python run_searchlight_mpnet.py --subject 2 --diagnose-only
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Telemetry helpers
# --------------------------------------------------------------------------- #
log = logging.getLogger("searchlight")
_TF_GPU_INITIALIZED = False


def bootstrap_cuda_library_path():
    """Restart once with pip-installed NVIDIA runtime libraries discoverable.

    ``tensorflow[and-cuda]`` installs CUDA libraries below
    ``site-packages/nvidia/*/lib``. The dynamic loader does not discover those
    directories in a plain ``.venv/bin/python`` invocation on this system,
    although notebook launchers may inject them. Set the loader path before
    TensorFlow is imported so notebook and script execution are consistent.
    """
    marker = "_SEARCHLIGHT_CUDA_PATH_BOOTSTRAPPED"
    if os.environ.get(marker) == "1":
        return

    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_root = (
        Path(sys.prefix) / "lib" / py_version / "site-packages" / "nvidia"
    )
    library_dirs = sorted(
        str(path)
        for path in nvidia_root.glob("*/lib")
        if path.is_dir()
    )
    if not library_dirs:
        return

    current = [
        path
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    ]
    missing = [path for path in library_dirs if path not in current]
    if not missing:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(missing + current)
    env[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def initialize_tensorflow(allow_cpu=False):
    """Initialise TensorFlow and verify the device used by the searchlight."""
    global _TF_GPU_INITIALIZED
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        message = (
            "TensorFlow did not register a GPU. "
            "Run with --allow-cpu only if CPU execution is intentional."
        )
        if not allow_cpu:
            raise RuntimeError(message)
        log.warning(message)
        return tf

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    _TF_GPU_INITIALIZED = True
    log.info(
        "TensorFlow %s registered GPU device(s): %s",
        tf.__version__,
        ", ".join(gpu.name for gpu in gpus),
    )
    return tf


def setup_logging(logfile=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def ram_snapshot():
    """Return (rss_gb, available_gb) for the current process / machine."""
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1e9
        avail = psutil.virtual_memory().available / 1e9
        return rss, avail
    except Exception:
        return float("nan"), float("nan")


def gpu_snapshot():
    """Return a short string describing GPU memory use, or '' if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
        parts = []
        for i, line in enumerate(out.strip().splitlines()):
            used, free, total = (int(x) for x in line.split(","))
            parts.append(f"GPU{i}: {used}/{total} MiB used, {free} MiB free")
        return "; ".join(parts)
    except Exception:
        return ""


def tf_gpu_peak():
    """Return TF's own peak GPU allocation (MiB) for GPU:0, or None."""
    if not _TF_GPU_INITIALIZED:
        return None
    try:
        import tensorflow as tf

        info = tf.config.experimental.get_memory_info("GPU:0")
        return info.get("peak", 0) / 2**20
    except Exception:
        return None


def report_resources(tag):
    rss, avail = ram_snapshot()
    gpu = gpu_snapshot()
    log.info(
        "[resources @ %s] RAM rss=%.1f GB, avail=%.1f GB%s",
        tag,
        rss,
        avail,
        f" | {gpu}" if gpu else "",
    )


@contextmanager
def stage(name):
    log.info("=" * 70)
    log.info(">>> STAGE START: %s", name)
    report_resources(f"{name}:start")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        report_resources(f"{name}:end")
        peak = tf_gpu_peak()
        log.info(
            ">>> STAGE END:   %s  (%s%s)",
            name,
            time.strftime("%H:%M:%S", time.gmtime(dt)),
            f", tf gpu peak={peak:.0f} MiB" if peak else "",
        )


# --------------------------------------------------------------------------- #
# Searchlight sphere diagnostics (the heart of the "too many voxels" question)
# --------------------------------------------------------------------------- #
def diagnose_searchlight(precompsl_dir, subj, targetspace, radius, batch_size, n_conditions):
    """Load the precomputed searchlight and report sphere-size telemetry.

    This is cheap (no GPU, no betas) and directly characterises the size-group
    structure that feeds the RDM correlation: how many searchlight centers
    there are, how they cluster by voxel count, and how large the resulting
    brain-RDM matrix will be.
    """
    sl_indices = os.path.join(
        precompsl_dir, subj, f"{subj}-{targetspace}-{radius}rad-searchlight_indices.npy"
    )
    if not os.path.exists(sl_indices):
        log.warning("No precomputed searchlight found at %s -- it will be built on first run.", sl_indices)
        return

    with open(sl_indices, "rb") as fp:
        all_indices = pickle.load(fp)

    log.info("searchlight pickle: %s", sl_indices)
    log.info("  type(all_indices)     = %s  (len=%d)", type(all_indices).__name__, len(all_indices))
    # NOTE: all_indices is a Python *list*. tf_searchlight indexes it per chunk;
    # `indices[chunk]` (numpy fancy-indexing) only works if this is an ndarray,
    # whereas `[indices[i] for i in chunk]` works for a list. Flag the mismatch.
    if isinstance(all_indices, list):
        log.info("  -> list: tf_searchlight must use the list-comprehension gather, NOT indices[chunk].")

    sizes = np.asarray([len(ix) for ix in all_indices])
    uniq, counts = np.unique(sizes, return_counts=True)
    n_centers = len(sizes)
    dom = counts.argmax()

    # number of upper-triangle pairs in each sampled 100x100 brain RDM
    n_pairs = n_conditions * (n_conditions - 1) // 2
    legacy_brain_rdm_gb = n_centers * n_pairs * 4 / 1e9
    batch_rdm_mb = min(n_centers, batch_size) * n_pairs * 4 / 1e6
    correlation_mb = n_centers * 4 / 1e6

    log.info("  n_centers (spheres)   = %d", n_centers)
    log.info("  sphere size min/max   = %d / %d voxels", sizes.min(), sizes.max())
    log.info("  unique sizes          = %d", len(uniq))
    log.info(
        "  dominant size group   = size %d -> %d centers (%.1f%% of all)",
        uniq[dom], counts[dom], 100 * counts[dom] / n_centers,
    )
    log.info("  top-5 size groups (size -> n_centers):")
    for k in np.argsort(counts)[::-1][:5]:
        nchunks = int(np.ceil(counts[k] / batch_size))
        log.info("      size %4d -> %7d centers  (%d batches of %d)", uniq[k], counts[k], nchunks, batch_size)
    log.info(
        "  => legacy full brain-RDM matrix would be %d x %d float32 = %.2f GB",
        n_centers,
        n_pairs,
        legacy_brain_rdm_gb,
    )
    log.info(
        "  => streaming path retains one RDM batch (~%.1f MB) and a final one-model map (~%.1f MB)",
        batch_rdm_mb,
        correlation_mb,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all-mpnet-base-v2", help="model name (default: all-mpnet-base-v2)")
    p.add_argument("--subject", type=int, default=2, help="subject number, 1-based (default: 2)")
    p.add_argument("--n-sessions", type=int, default=20, help="number of NSD sessions (default: 20)")
    p.add_argument("--rdm-distance", default="correlation", help="model RDM distance (default: correlation)")
    p.add_argument("--overwrite", action="store_true", help="recompute corr vols even if present")
    p.add_argument("--n-samples", type=int, default=None, help="cap number of 100x100 samples to run (quick test)")

    # paths (defaults mirror searchlight.ipynb cell 2, run from lucas_exploration/)
    p.add_argument("--base-save-dir", default="./results")
    p.add_argument("--nsd-dir", default="/media/chuddy/120876114737F70A/data/NSD")
    p.add_argument(
        "--nsd-derivatives-dir",
        default="./results/searchlight/",
        help="precompsl_dir / parent of betas (notebook passes this as precompsl_dir)",
    )

    # model RDM preparation (off by default: the RDMs already exist on disk)
    p.add_argument("--prepare-rdms", action="store_true", help="(re)build model RDMs before the searchlight")
    p.add_argument(
        "--n-subjects-prepare",
        type=int,
        default=None,
        help="n_subjects for nsd_prepare_modelrdms (defaults to --subject so the target is covered)",
    )

    p.add_argument("--diagnose-only", action="store_true", help="only print sphere telemetry, do not run the searchlight")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="continue when TensorFlow cannot register a GPU",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="project the selected subject to fsaverage and create a brain plot after the searchlight",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="skip searchlight computation and only project/plot existing results",
    )
    p.add_argument(
        "--figures-dir",
        default=None,
        help="plot output directory (default: <base-save-dir>/searchlight_respectedsampling_correlation/figures)",
    )
    p.add_argument(
        "--roi-overlay",
        default="streams",
        help="named fsaverage ROI overlay, or 'none' to disable (default: streams)",
    )
    p.add_argument(
        "--max-cmap-val",
        type=float,
        default=None,
        help="optional symmetric color-map limit for the brain plot",
    )
    p.add_argument("--logfile", default=None, help="also write logs to this file")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.plot_only:
        bootstrap_cuda_library_path()
    setup_logging(args.logfile)

    # constants fixed inside nsd_searchlight_main_tf -- duplicated here only for telemetry
    radius = 6
    targetspace = "func1pt8mm"
    batch_size = 250

    subj = f"subj0{args.subject}"
    betas_dir = os.path.join(args.nsd_derivatives_dir, "betas")

    log.info("#" * 70)
    log.info("# searchlight reproducible run")
    log.info("#   model        = %s", args.model)
    log.info("#   subject      = %d (%s)", args.subject, subj)
    log.info("#   n_sessions   = %d", args.n_sessions)
    log.info("#   rdm_distance = %s", args.rdm_distance)
    log.info("#   base_save    = %s", os.path.abspath(args.base_save_dir))
    log.info("#   precompsl    = %s", os.path.abspath(args.nsd_derivatives_dir))
    log.info("#   cwd          = %s", os.getcwd())
    log.info("#" * 70)
    report_resources("startup")
    if not args.plot_only:
        initialize_tensorflow(allow_cpu=args.allow_cpu)

    # number of "conditions" per sampled brain RDM is fixed at 100 in the pipeline
    n_sampled_conditions = 100

    if not args.plot_only:
        # ---- stage 2 first: cheap sphere telemetry (also useful standalone) ----
        with stage("diagnose searchlight spheres"):
            diagnose_searchlight(
                args.nsd_derivatives_dir,
                subj,
                targetspace,
                radius,
                batch_size,
                n_sampled_conditions,
            )

        if args.diagnose_only:
            log.info("--diagnose-only set: stopping before the searchlight run.")
            return

        # ---- stage 1: optionally (re)build model RDMs ----
        if args.prepare_rdms:
            from nsd_visuo_semantics.utils.nsd_prepare_modelrdms import nsd_prepare_modelrdms

            n_subjects_prepare = args.n_subjects_prepare or args.subject
            saved_embeddings_dir = f"{args.base_save_dir}/saved_embeddings"
            rdms_dir = f"{args.base_save_dir}/serialised_models_{args.rdm_distance}"
            with stage("prepare model RDMs"):
                nsd_prepare_modelrdms(
                    [args.model],
                    args.rdm_distance,
                    saved_embeddings_dir,
                    rdms_dir,
                    args.nsd_dir,
                    "",
                    "",
                    args.overwrite,
                    n_sessions=args.n_sessions,
                    n_subjects=n_subjects_prepare,
                )

        from nsd_visuo_semantics.searchlight_analyses.nsd_searchlight_main_tf import nsd_searchlight_main_tf

        with stage("searchlight main (RDM compute + correlation)"):
            nsd_searchlight_main_tf(
                args.model,
                args.rdm_distance,
                args.nsd_dir,
                args.nsd_derivatives_dir,
                betas_dir,
                args.base_save_dir,
                args.overwrite,
                subject=args.subject,
                n_sessions=args.n_sessions,
                max_samples=args.n_samples,
            )

    if args.plot or args.plot_only:
        from nsd_visuo_semantics.searchlight_analyses.nsd_project_fsaverage import nsd_project_fsaverage
        from nsd_visuo_semantics.utils.py_plot_brain_utils import pyplot_brains_from_models_list

        figures_dir = args.figures_dir or os.path.join(
            args.base_save_dir,
            f"searchlight_respectedsampling_{args.rdm_distance}",
            "figures",
        )
        roi_overlay = None if args.roi_overlay.lower() == "none" else args.roi_overlay

        with stage("project searchlight map to fsaverage"):
            nsd_project_fsaverage(
                [args.model],
                args.rdm_distance,
                args.nsd_dir,
                args.base_save_dir,
                subjects=[args.subject],
            )

        with stage("plot fsaverage brain map"):
            pyplot_brains_from_models_list(
                [args.model],
                [args.model],
                os.path.join(
                    args.base_save_dir,
                    f"searchlight_respectedsampling_{args.rdm_distance}",
                ),
                layer="last",
                contrast_layer="same",
                contrast_same_model=False,
                max_cmap_val=args.max_cmap_val,
                save_type="png",
                figpath=figures_dir,
                plot_indiv_sub=True,
                plot_subj_avg=False,
                roi_overlay=roi_overlay,
                nsd_dir=args.nsd_dir,
                roi_linecolor="black",
                roi_linewidth=0.8,
                subjects=[args.subject],
            )
        log.info(
            "Brain plot saved to %s",
            os.path.join(figures_dir, f"{args.model}_{subj}.png"),
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
