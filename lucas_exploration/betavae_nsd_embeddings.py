"""Extract simple beta-VAE latents for the NSD 73K stimuli.

This is the beta-VAE analogue of ``stable_diffusion_vae_embeddings.ipynb``:
it reuses the same NSD stimulus loading, streams the 73K images through a
trained :class:`VAE.BetaVAE`, and writes a ``(73000, latent_dim)`` float32
feature matrix in NSD image order to ``results/saved_embeddings``.

The only substantive difference from the SD-VAE extractor is preprocessing:
our beta-VAE has a sigmoid/BCE decoder and was trained on images in **[0, 1]**,
so we do *not* apply the SD-VAE's ``* 2 - 1`` shift to ``[-1, 1]``.

The exported feature is the deterministic encoder mean (``mu``), matching the
"posterior mode" feature the SD-VAE extractor saves.

After running this, the file is auto-registered by
``nsd_visuo_semantics.utils.get_name2file_dict`` (see
``_add_simple_betavae_embedding_files``) so it can be passed straight to
``nsd_prepare_modelrdms`` and the searchlight, exactly like the SD-VAE model:

    python lucas_exploration/betavae_nsd_embeddings.py
    # then in searchlight.ipynb:
    #   nsd_prepare_modelrdms(["simplebetavae_beta4_z32_seed0"], "correlation", ...)
    #   nsd_searchlight_main_tf(["simplebetavae_beta4_z32_seed0"], "correlation", ...)

Smoke test (first 8 images, rows-suffixed file that is intentionally NOT
registered for RDM construction):

    python lucas_exploration/betavae_nsd_embeddings.py --stop-index 8 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

# Import BetaVAE / helpers from the sibling VAE.py regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from VAE import BetaVAE, format_float_for_name  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_NSD_DIR = Path("/media/chuddy/120876114737F70A/data/NSD")
DEFAULT_STIM_HDF5 = (
    DEFAULT_NSD_DIR / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
)
DEFAULT_CHECKPOINT = (
    HERE / "results" / "simple_beta_vae" / "betavae_beta4_z32_seed0" / "best.pt"
)
DEFAULT_EMBEDDINGS_DIR = HERE / "results" / "saved_embeddings"


def open_nsd_stimulus_dataset(stim_hdf5: Path):
    """Open the NSD image HDF5 and return (file_handle, image_dataset).

    Mirrors the loader in stable_diffusion_vae_embeddings.ipynb.
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
    batch: np.ndarray, image_size: int, device: torch.device
) -> torch.Tensor:
    """Convert uint8 NSD images to beta-VAE input: RGB, image_size, [0, 1].

    Same as the SD-VAE preprocessing but WITHOUT the final ``* 2 - 1`` step,
    because the beta-VAE's sigmoid/BCE decoder expects pixels in [0, 1].
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
    return x.clamp(0.0, 1.0).to(device=device, non_blocking=True)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[BetaVAE, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = BetaVAE(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def model_name_from_config(config: dict, seed: int) -> str:
    """e.g. simplebetavae_beta4_z32_seed0 (distinct from the 128px family)."""
    beta = format_float_for_name(float(config["beta"]))
    return f"simplebetavae_beta{beta}_z{int(config['latent_dim'])}_seed{seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export beta-VAE latents for NSD.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--stim-hdf5", type=Path, default=DEFAULT_STIM_HDF5)
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="seed used in the model name")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--stop-index", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    model, checkpoint = load_model(args.checkpoint, device)
    image_size = int(model.image_size)
    latent_dim = int(model.latent_dim)
    name = model_name_from_config(checkpoint["config"], args.seed)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")
    print(f"Model: {name}  image_size={image_size}  latent_dim={latent_dim}")

    stim_file, images = open_nsd_stimulus_dataset(args.stim_hdf5)
    n_total = int(images.shape[0])
    start = int(args.start_index)
    stop = n_total if args.stop_index is None else min(int(args.stop_index), n_total)
    if start < 0 or stop <= start or stop > n_total:
        stim_file.close()
        raise ValueError(f"Invalid range start={start}, stop={stop}, n_total={n_total}")

    range_suffix = "" if (start == 0 and stop == n_total) else f"_rows{start}-{stop}"
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.embeddings_dir / f"nsd_{name}_{image_size}px{range_suffix}.npy"
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    partial_path = out_path.with_name(out_path.stem + ".partial.npy")

    if out_path.exists() and not args.overwrite:
        stim_file.close()
        raise FileExistsError(f"{out_path} exists. Pass --overwrite to regenerate.")
    if partial_path.exists():
        partial_path.unlink()

    features = open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=(stop - start, latent_dim)
    )
    print(f"Images: {stop - start:,} of {n_total:,}")
    print(f"Writing: {out_path}")

    t0 = time.time()
    with torch.inference_mode():
        for batch_start in range(start, stop, args.batch_size):
            batch_stop = min(batch_start + args.batch_size, stop)
            x = prepare_image_batch(
                images[batch_start:batch_stop], image_size, device
            )
            mu, _ = model.encode(x)
            features[batch_start - start : batch_stop - start] = (
                mu.float().cpu().numpy()
            )
            features.flush()
            done = batch_stop - start
            if (batch_start - start) % (args.batch_size * 20) == 0 or batch_stop == stop:
                print(f"  encoded {done:,}/{stop - start:,}")

    del features
    os.replace(partial_path, out_path)
    stim_file.close()

    meta_path.write_text(
        json.dumps(
            {
                "model_name": name,
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "config": checkpoint["config"],
                "source_hdf5": str(args.stim_hdf5),
                "image_size": image_size,
                "latent_dim": latent_dim,
                "start_index": start,
                "stop_index": stop,
                "num_images": stop - start,
                "feature": "BetaVAE encoder posterior mean (mu), inputs in [0, 1]",
                "dtype": "float32",
                "elapsed_seconds": time.time() - t0,
            },
            indent=2,
        )
    )
    print(f"Saved embeddings: {out_path}")
    print(f"Saved metadata:   {meta_path}")
    if range_suffix:
        print("NOTE: rows-suffixed smoke-test file is NOT registered for RDMs.")


if __name__ == "__main__":
    main()
