"""Extract raw-pixel feature vectors for the NSD 73K stimuli.

This is the pixel-space analogue of ``betavae_nsd_embeddings.py`` /
``stable_diffusion_vae_embeddings.ipynb``: it reuses the same NSD stimulus
loading and streams the 73K images to disk, but instead of encoding each image
through a model it simply resizes it and flattens the raw pixels into a feature
vector. The result is a ``(73000, n_pixels)`` float32 matrix in NSD image order
written to ``results/saved_embeddings``.

The point of this is a *pixel-level RDM baseline*: the searchlight then asks how
much of the brain's representational geometry is explained by low-level image
statistics alone, before any model is involved.

Preprocessing mirrors the beta-VAE extractor exactly (RGB, bicubic resize,
inputs in **[0, 1]**) so that, at the default ``--image-size 64``, the pixel RDM
lives in the very same input space the beta-VAE sees. That makes the pixel map a
directly comparable low-level reference for the VAE / MPNet maps. Pass
``--grayscale`` for a luminance-only pixel space.

Because the searchlight RDMs use correlation distance by default, the absolute
pixel scale ([0, 1] vs [0, 255]) is irrelevant -- correlation z-scores each row.

After running this, the file is auto-registered by
``nsd_visuo_semantics.utils.get_name2file_dict`` (see
``_add_pixel_embedding_files``) so it can be passed straight to
``nsd_prepare_modelrdms`` and the searchlight, exactly like the VAE models:

    python lucas_exploration/pixel_nsd_embeddings.py
    # then, mirroring run_searchlight_mpnet.py:
    #   nsd_prepare_modelrdms(["pixels_rgb_64px"], "correlation", ...)
    #   nsd_searchlight_main_tf(["pixels_rgb_64px"], "correlation", ...)
    # or simply:
    #   python run_searchlight_mpnet.py --model pixels_rgb_64px --prepare-rdms \
    #       --subject 2 --n-sessions 20

Smoke test (first 8 images, rows-suffixed file that is intentionally NOT
registered for RDM construction):

    python lucas_exploration/pixel_nsd_embeddings.py --stop-index 8 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
DEFAULT_NSD_DIR = Path("/media/chuddy/120876114737F70A/data/NSD")
DEFAULT_STIM_HDF5 = (
    DEFAULT_NSD_DIR / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
)
DEFAULT_EMBEDDINGS_DIR = HERE / "results" / "saved_embeddings"

# Rec. 601 luminance weights, the standard RGB -> grayscale conversion.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def open_nsd_stimulus_dataset(stim_hdf5: Path):
    """Open the NSD image HDF5 and return (file_handle, image_dataset).

    Mirrors the loader in betavae_nsd_embeddings.py /
    stable_diffusion_vae_embeddings.ipynb.
    """
    if not stim_hdf5.exists():
        raise FileNotFoundError(
            f"Could not find {stim_hdf5}. Point --stim-hdf5 at nsd_stimuli.hdf5."
        )
    h5_file = h5py.File(stim_hdf5, "r")
    for key in ("imgBrick", "stimuli", "images"):
        if key in h5_file and isinstance(h5_file[key], h5py.Dataset):
            return h5_file, h5_file[key]

    datasets: list[str] = []

    def collect_dataset(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            datasets.append(name)

    h5_file.visititems(collect_dataset)
    if len(datasets) == 1:
        return h5_file, h5_file[datasets[0]]
    h5_file.close()
    raise KeyError(
        "Could not identify the image dataset inside the HDF5. "
        f"Candidate datasets were: {datasets}"
    )


def prepare_image_batch(
    batch: np.ndarray,
    image_size: int,
    device: torch.device,
    grayscale: bool,
) -> torch.Tensor:
    """Convert uint8 NSD images to pixel features: resized, [0, 1], (B, C, H, W).

    Identical to the beta-VAE preprocessing (RGB, bicubic resize, [0, 1], no
    ``* 2 - 1`` shift), with an optional Rec. 601 grayscale collapse to a single
    channel.
    """
    x = torch.from_numpy(np.asarray(batch))
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D image batch, got shape {tuple(x.shape)}")

    if x.shape[-1] in (1, 3, 4):
        x = x[..., :3].permute(0, 3, 1, 2)
    elif x.shape[1] in (1, 3, 4):
        x = x[:, :3]
    else:
        raise ValueError(f"Could not infer channel axis for shape {tuple(x.shape)}")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    x = x.float()
    if x.max() > 2.0:
        x = x / 255.0
    x = F.interpolate(
        x, size=(image_size, image_size), mode="bicubic", align_corners=False
    )
    x = x.clamp(0.0, 1.0).to(device=device, non_blocking=True)

    if grayscale:
        weights = torch.tensor(_LUMA_WEIGHTS, device=x.device).view(1, 3, 1, 1)
        x = (x * weights).sum(dim=1, keepdim=True)
    return x


def model_name(colorspace: str, image_size: int) -> str:
    """e.g. pixels_rgb_64px / pixels_gray_128px."""
    return f"pixels_{colorspace}_{image_size}px"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw-pixel features for NSD.")
    p.add_argument("--stim-hdf5", type=Path, default=DEFAULT_STIM_HDF5)
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    p.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="square resize before flattening (default: 64, matching the beta-VAE input)",
    )
    p.add_argument(
        "--grayscale",
        action="store_true",
        help="collapse RGB to a single luminance channel before flattening",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--stop-index", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = int(args.image_size)
    if image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {image_size}")
    colorspace = "gray" if args.grayscale else "rgb"
    n_channels = 1 if args.grayscale else 3
    n_features = image_size * image_size * n_channels
    name = model_name(colorspace, image_size)
    print(f"Device: {device}")
    print(f"Model: {name}  image_size={image_size}  n_features={n_features}")

    stim_file, images = open_nsd_stimulus_dataset(args.stim_hdf5)
    n_total = int(images.shape[0])
    start = int(args.start_index)
    stop = n_total if args.stop_index is None else min(int(args.stop_index), n_total)
    if start < 0 or stop <= start or stop > n_total:
        stim_file.close()
        raise ValueError(f"Invalid range start={start}, stop={stop}, n_total={n_total}")

    range_suffix = "" if (start == 0 and stop == n_total) else f"_rows{start}-{stop}"
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    # name already ends in "...px"; e.g. nsd_pixels_rgb_64px.npy
    out_path = args.embeddings_dir / f"nsd_{name}{range_suffix}.npy"
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    partial_path = out_path.with_name(out_path.stem + ".partial.npy")

    if out_path.exists() and not args.overwrite:
        stim_file.close()
        raise FileExistsError(f"{out_path} exists. Pass --overwrite to regenerate.")
    if partial_path.exists():
        partial_path.unlink()

    n_rows = stop - start
    est_gb = n_rows * n_features * 4 / 1e9
    features = open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=(n_rows, n_features)
    )
    print(f"Images: {n_rows:,} of {n_total:,}")
    print(f"Writing: {out_path}  (~{est_gb:.2f} GB float32)")

    t0 = time.time()
    with torch.inference_mode():
        for batch_start in range(start, stop, args.batch_size):
            batch_stop = min(batch_start + args.batch_size, stop)
            x = prepare_image_batch(
                images[batch_start:batch_stop], image_size, device, args.grayscale
            )
            flat = x.reshape(x.shape[0], -1).float().cpu().numpy()
            features[batch_start - start : batch_stop - start] = flat
            features.flush()
            done = batch_stop - start
            if (batch_start - start) % (args.batch_size * 20) == 0 or batch_stop == stop:
                print(f"  flattened {done:,}/{n_rows:,}")

    del features
    os.replace(partial_path, out_path)
    stim_file.close()

    meta_path.write_text(
        json.dumps(
            {
                "model_name": name,
                "source_hdf5": str(args.stim_hdf5),
                "image_size": image_size,
                "colorspace": colorspace,
                "n_channels": n_channels,
                "n_features": n_features,
                "start_index": start,
                "stop_index": stop,
                "num_images": n_rows,
                "feature": (
                    "Flattened raw pixels (bicubic resize, inputs in [0, 1]); "
                    f"{colorspace} channel order, channels-first flatten"
                ),
                "dtype": "float32",
                "elapsed_seconds": time.time() - t0,
            },
            indent=2,
        )
    )
    print(f"Saved features: {out_path}")
    print(f"Saved metadata: {meta_path}")
    if range_suffix:
        print("NOTE: rows-suffixed smoke-test file is NOT registered for RDMs.")


if __name__ == "__main__":
    main()
