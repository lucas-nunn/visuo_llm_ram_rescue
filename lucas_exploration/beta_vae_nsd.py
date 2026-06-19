"""Train and export small beta-VAE representations for NSD images.

Typical smoke test:

    python lucas_exploration/beta_vae_nsd.py --stop-index 512 --epochs 1

Typical first full run:

    python lucas_exploration/beta_vae_nsd.py --betas 1 4 10 --seeds 0

The exported feature matrix is written in NSD image order and can be passed
directly to nsd_prepare_modelrdms after registration in get_name2file_dict.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


DEFAULT_NSD_DIR = Path("/media/chuddy/120876114737F70A/data/NSD")
DEFAULT_STIM_RELATIVE = Path(
    "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
)


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "lucas_exploration").is_dir() and (
            candidate / "src"
        ).is_dir():
            return candidate
    return cwd


def default_stim_hdf5() -> Path:
    if os.environ.get("NSD_STIM_HDF5"):
        return Path(os.environ["NSD_STIM_HDF5"]).expanduser()
    if os.environ.get("NSD_DIR"):
        return Path(os.environ["NSD_DIR"]).expanduser() / DEFAULT_STIM_RELATIVE
    return DEFAULT_NSD_DIR / DEFAULT_STIM_RELATIVE


def discover_image_dataset(h5_file: h5py.File) -> str:
    for key in ("imgBrick", "stimuli", "images"):
        if key in h5_file and isinstance(h5_file[key], h5py.Dataset):
            return key

    datasets: list[str] = []

    def collect_dataset(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            datasets.append(name)

    h5_file.visititems(collect_dataset)
    if len(datasets) == 1:
        return datasets[0]
    raise KeyError(
        "Could not identify image dataset in HDF5. "
        f"Candidate datasets were: {datasets}"
    )


def format_float_for_name(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def model_name(beta: float, seed: int) -> str:
    return f"betavae_beta{format_float_for_name(beta)}_seed{seed}_zmean"


def range_suffix(start_index: int, stop_index: int | None, n_total: int) -> str:
    stop = n_total if stop_index is None else stop_index
    if start_index == 0 and stop == n_total:
        return ""
    return f"_rows{start_index}-{stop}"


class NSDImageDataset(Dataset):
    """Lazy HDF5-backed NSD image dataset.

    Each worker opens its own h5py file handle on first access. Samples are
    returned as uint8 CHW tensors; resizing and normalization happen on GPU.
    """

    def __init__(
        self,
        hdf5_path: Path,
        dataset_key: str,
        indices: np.ndarray,
    ):
        self.hdf5_path = str(hdf5_path)
        self.dataset_key = dataset_key
        self.indices = np.asarray(indices, dtype=np.int64)
        self._h5_file = None
        self._images = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_images"] = None
        return state

    def _ensure_open(self):
        if self._images is None:
            self._h5_file = h5py.File(self.hdf5_path, "r")
            self._images = self._h5_file[self.dataset_key]

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_open()
        image = np.asarray(self._images[int(self.indices[idx])])
        return torch.from_numpy(to_chw_uint8(image))

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._images = None


def to_chw_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got {arr.shape}")

    if arr.shape[-1] in (1, 3, 4):
        arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] in (1, 3, 4):
        arr = arr[:3]
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
    else:
        raise ValueError(f"Could not infer channel axis for image {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float32)
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(arr)


def preprocess_batch(
    batch: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    is_uint8 = batch.dtype == torch.uint8
    x = batch.to(device=device, dtype=torch.float32, non_blocking=True)
    if is_uint8:
        x = x / 255.0
    elif torch.max(x) > 2.0:
        x = x / 255.0
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(
            x,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )
    x = x.clamp(0.0, 1.0)
    return x * 2.0 - 1.0


class ConvBetaVAE(nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 50):
        super().__init__()
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32")
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.feature_hw = self.image_size // 32
        self.feature_dim = 512 * self.feature_hw * self.feature_hw

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(self.feature_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, self.feature_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, self.feature_hw, self.feature_hw)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


@dataclass
class LossParts:
    loss: float
    recon: float
    kl: float
    active_units: int | None = None


def beta_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    kl_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x.shape[0]
    recon_loss = F.mse_loss(recon, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / batch_size
    loss = recon_loss + beta * kl_scale * kl_loss
    return loss, recon_loss, kl_loss


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


def split_indices(
    start: int,
    stop: int,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(start, stop, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if len(indices) < 2:
        return indices, indices
    n_val = max(1, int(round(len(indices) * val_fraction)))
    n_val = min(n_val, len(indices) - 1)
    return indices[n_val:], indices[:n_val]


def run_epoch(
    model: ConvBetaVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    image_size: int,
    beta: float,
    kl_warmup_steps: int,
    global_step: int,
    active_threshold: float,
    use_amp: bool,
) -> tuple[LossParts, int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_n = 0
    mu_sum = None
    mu_sumsq = None

    for raw_batch in loader:
        x = preprocess_batch(raw_batch, image_size, device)
        batch_n = x.shape[0]
        kl_scale = 1.0
        if kl_warmup_steps > 0:
            kl_scale = min(1.0, global_step / float(kl_warmup_steps))

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                recon, mu, logvar = model(x)
                loss, recon_loss, kl_loss = beta_vae_loss(
                    recon, x, mu, logvar, beta, kl_scale
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

        total_loss += float(loss.detach().cpu()) * batch_n
        total_recon += float(recon_loss.detach().cpu()) * batch_n
        total_kl += float(kl_loss.detach().cpu()) * batch_n
        total_n += batch_n

        detached_mu = mu.detach().float().cpu()
        if mu_sum is None:
            mu_sum = torch.zeros(detached_mu.shape[1], dtype=torch.float64)
            mu_sumsq = torch.zeros(detached_mu.shape[1], dtype=torch.float64)
        mu_sum += detached_mu.sum(dim=0, dtype=torch.float64)
        mu_sumsq += (detached_mu.double() ** 2).sum(dim=0)

    active_units = None
    if total_n > 1 and mu_sum is not None and mu_sumsq is not None:
        mean = mu_sum / total_n
        variance = mu_sumsq / total_n - mean**2
        active_units = int(torch.sum(variance > active_threshold).item())

    parts = LossParts(
        loss=total_loss / max(1, total_n),
        recon=total_recon / max(1, total_n),
        kl=total_kl / max(1, total_n),
        active_units=active_units,
    )
    return parts, global_step


def save_checkpoint(
    path: Path,
    model: ConvBetaVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    beta: float,
    seed: int,
    args: argparse.Namespace,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "beta": beta,
            "seed": seed,
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ConvBetaVAE,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_training_visuals(
    model: ConvBetaVAE,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    latent_dim: int,
    output_dir: Path,
    prefix: str,
    use_amp: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    raw_batch = next(iter(loader))
    x = preprocess_batch(raw_batch, image_size, device)
    x = x[:8]

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            recon, mu, _ = model(x)

    paired = torch.empty(
        (x.shape[0] * 2, *x.shape[1:]),
        dtype=torch.float32,
        device=x.device,
    )
    paired[0::2] = x
    paired[1::2] = recon
    save_image((paired + 1.0) / 2.0, output_dir / f"{prefix}_recon.png", nrow=4)

    base = mu[:1].repeat(7, 1)
    dims = min(8, latent_dim)
    traversal_rows = []
    values = torch.linspace(-3.0, 3.0, 7, device=device)
    with torch.inference_mode():
        for dim in range(dims):
            z = base.clone()
            z[:, dim] = values
            traversal_rows.append(model.decode(z).float())
    traversals = torch.cat(traversal_rows, dim=0)
    save_image(
        (traversals + 1.0) / 2.0,
        output_dir / f"{prefix}_traversals.png",
        nrow=7,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def train_one_model(
    beta: float,
    seed: int,
    args: argparse.Namespace,
    hdf5_path: Path,
    dataset_key: str,
    n_total: int,
    device: torch.device,
) -> Path:
    set_seed(seed)
    start, stop = validated_range(args.start_index, args.stop_index, n_total)
    name = model_name(beta, seed)
    run_dir = args.output_dir / name
    viz_dir = run_dir / "visuals"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"

    if best_path.exists() and not args.overwrite and not args.resume:
        print(f"{name}: found {best_path}, skipping training.")
        return best_path

    train_ids, val_ids = split_indices(start, stop, args.val_fraction, seed)
    train_dataset = NSDImageDataset(hdf5_path, dataset_key, train_ids)
    val_dataset = NSDImageDataset(hdf5_path, dataset_key, val_ids)
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        drop_last=True,
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        drop_last=False,
    )

    model = ConvBetaVAE(args.image_size, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    history: list[dict] = []
    best_val = math.inf
    if args.resume and latest_path.exists() and not args.overwrite:
        checkpoint = load_checkpoint(latest_path, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        history = list(checkpoint.get("history", []))
        if history:
            best_val = min(row["val"]["loss"] for row in history)
        print(f"{name}: resumed from epoch {start_epoch}")

    write_json(
        run_dir / "metadata.json",
        {
            "model_name": name,
            "beta": beta,
            "seed": seed,
            "stim_hdf5": str(hdf5_path),
            "dataset_key": dataset_key,
            "n_total": n_total,
            "start_index": start,
            "stop_index": stop,
            "train_images": int(len(train_ids)),
            "val_images": int(len(val_ids)),
            "args": serializable_args(args),
        },
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_parts, global_step = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.image_size,
            beta,
            args.kl_warmup_steps,
            global_step,
            args.active_threshold,
            use_amp,
        )
        val_parts, global_step = run_epoch(
            model,
            val_loader,
            None,
            scaler,
            device,
            args.image_size,
            beta,
            args.kl_warmup_steps,
            global_step,
            args.active_threshold,
            use_amp,
        )
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": elapsed,
            "train": asdict(train_parts),
            "val": asdict(val_parts),
        }
        history.append(row)
        write_json(history_path, {"history": history})
        print(
            f"{name} epoch {epoch:03d}/{args.epochs}: "
            f"train={train_parts.loss:.2f} "
            f"val={val_parts.loss:.2f} "
            f"val_recon={val_parts.recon:.2f} "
            f"val_kl={val_parts.kl:.2f} "
            f"active={val_parts.active_units} "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            global_step,
            beta,
            seed,
            args,
            history,
        )
        if val_parts.loss < best_val:
            best_val = val_parts.loss
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                global_step,
                beta,
                seed,
                args,
                history,
            )
        if epoch == 1 or epoch % args.viz_every == 0 or epoch == args.epochs:
            save_training_visuals(
                model,
                val_loader,
                device,
                args.image_size,
                args.latent_dim,
                viz_dir,
                f"epoch{epoch:03d}",
                use_amp,
            )

    train_dataset.close()
    val_dataset.close()
    return best_path


def export_one_model(
    beta: float,
    seed: int,
    args: argparse.Namespace,
    hdf5_path: Path,
    dataset_key: str,
    n_total: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> Path:
    start, stop = validated_range(args.start_index, args.stop_index, n_total)
    name = model_name(beta, seed)
    if checkpoint_path is None:
        checkpoint_path = args.output_dir / name / args.export_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    suffix = range_suffix(start, args.stop_index, n_total)
    features_path = (
        args.embeddings_dir
        / f"nsd_{name}_{args.image_size}px{suffix}.npy"
    )
    partial_path = features_path.with_name(features_path.stem + ".partial.npy")
    meta_path = features_path.with_suffix(".json")

    if features_path.exists() and not args.overwrite:
        print(f"{name}: found {features_path}, skipping export.")
        return features_path
    if partial_path.exists():
        if args.overwrite:
            partial_path.unlink()
        else:
            raise FileExistsError(
                f"Partial export exists: {partial_path}. "
                "Delete it or pass --overwrite."
            )

    model = ConvBetaVAE(args.image_size, args.latent_dim).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    indices = np.arange(start, stop, dtype=np.int64)
    dataset = NSDImageDataset(hdf5_path, dataset_key, indices)
    loader = build_loader(
        dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        drop_last=False,
    )
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    features = open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(stop - start, args.latent_dim),
    )

    offset = 0
    use_amp = args.amp and device.type == "cuda"
    t0 = time.time()
    with torch.inference_mode():
        for batch_i, raw_batch in enumerate(loader):
            x = preprocess_batch(raw_batch, args.image_size, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                mu, _ = model.encode(x)
            batch_mu = mu.float().cpu().numpy()
            features[offset : offset + batch_mu.shape[0]] = batch_mu
            features.flush()
            offset += batch_mu.shape[0]
            if batch_i % args.log_every == 0:
                print(f"{name}: exported {offset:,}/{stop - start:,}")

    del features
    os.replace(partial_path, features_path)
    dataset.close()

    write_json(
        meta_path,
        {
            "model_name": name,
            "beta": beta,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "stim_hdf5": str(hdf5_path),
            "dataset_key": dataset_key,
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "start_index": start,
            "stop_index": stop,
            "num_images": stop - start,
            "feature": "deterministic encoder mean, mu",
            "dtype": "float32",
            "elapsed_seconds": time.time() - t0,
        },
    )
    print(f"{name}: saved {features_path}")
    return features_path


def validated_range(
    start_index: int,
    stop_index: int | None,
    n_total: int,
) -> tuple[int, int]:
    start = int(start_index)
    stop = n_total if stop_index is None else min(int(stop_index), n_total)
    if start < 0 or stop <= start or stop > n_total:
        raise ValueError(
            f"Invalid range start={start}, stop={stop}, n_total={n_total}"
        )
    return start, stop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_args(args: argparse.Namespace) -> dict:
    payload = vars(args).copy()
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(
        description="Train/export practical beta-VAE latents for NSD RSA."
    )
    parser.add_argument(
        "--mode",
        choices=("train-export", "train", "export"),
        default="train-export",
    )
    parser.add_argument("--stim-hdf5", type=Path, default=default_stim_hdf5())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "lucas_exploration/results/beta_vae_nsd",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=project_root / "lucas_exploration/results/saved_embeddings",
    )
    parser.add_argument("--betas", type=float, nargs="+", default=[1.0, 4.0, 10.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--kl-warmup-steps", type=int, default=5000)
    parser.add_argument("--active-threshold", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--viz-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument("--export-checkpoint", default="best.pt")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(amp=True, resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = args.stim_hdf5.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.embeddings_dir = args.embeddings_dir.expanduser().resolve()

    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"Could not find {hdf5_path}. Pass --stim-hdf5 or set NSD_DIR."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Device: {device}")
    print(f"Stimulus HDF5: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5_file:
        dataset_key = discover_image_dataset(h5_file)
        n_total = int(h5_file[dataset_key].shape[0])
        print(f"Image dataset: {dataset_key} shape={h5_file[dataset_key].shape}")

    checkpoint_paths: dict[tuple[float, int], Path] = {}
    for beta in args.betas:
        for seed in args.seeds:
            if args.mode in ("train", "train-export"):
                checkpoint_paths[(beta, seed)] = train_one_model(
                    beta, seed, args, hdf5_path, dataset_key, n_total, device
                )

            if args.mode in ("export", "train-export"):
                export_one_model(
                    beta,
                    seed,
                    args,
                    hdf5_path,
                    dataset_key,
                    n_total,
                    device,
                    checkpoint_paths.get((beta, seed)),
                )


if __name__ == "__main__":
    main()
