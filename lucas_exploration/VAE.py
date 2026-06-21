"""A very simple beta-VAE, faithful to the architecture used in

    Higgins et al. (2021), "Unsupervised deep learning identifies semantic
    disentanglement in single inferotemporal face patch neurons",
    Nature Communications.

That paper trains a beta-VAE (Higgins et al., 2017) with the canonical
DeepMind disentanglement architecture (Burgess et al., 2018):

    * 64 x 64 input images,
    * four stride-2 4x4 convolutions with 32, 32, 64, 64 channels,
    * a single 256-unit fully connected bottleneck,
    * a diagonal-Gaussian latent with a standard-normal prior,
    * a Bernoulli decoder (sigmoid output, summed binary cross-entropy
      reconstruction) and the closed-form KL term scaled by beta.

This module deliberately keeps that architecture small and self-contained.
The richer 128px model lives in ``beta_vae_nsd.py``; this file is the simple,
paper-faithful version intended to be:

    1. trained on the 2014 MS COCO images,
    2. run over the NSD stimuli to extract latent means (``encode_mu``) as the
       per-image "neural" activity,
    3. turned into RDMs and searchlight-correlated against the NSD fMRI.

For RSA the latent *mean* (mu) is used as the deterministic representation,
matching the "deterministic encoder mean" feature exported by the sibling
pipeline.

Usage sketch
------------
    model = BetaVAE(image_size=64, in_channels=3, latent_dim=32, beta=4.0).cuda()
    recon, mu, logvar = model(x)                       # x in [0, 1], NCHW
    loss, recon_loss, kl = model.loss_function(recon, x, mu, logvar)
    ...
    feats = model.encode_mu(x)                          # (N, latent_dim) for RSA
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


class BetaVAE(nn.Module):
    """Simple convolutional beta-VAE for 64x64 (or 16*k) RGB images.

    Parameters
    ----------
    image_size:
        Side length of the (square) input. Must be divisible by 16 so that the
        four stride-2 convolutions land on an integer feature map. The paper
        uses 64.
    in_channels:
        Number of image channels (3 for RGB COCO/NSD images).
    latent_dim:
        Size of the latent code z. The key disentanglement knob alongside beta.
    beta:
        Weight on the KL term. beta = 1 recovers a standard VAE; beta > 1
        pressures the latents toward a factorised, disentangled code.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        latent_dim: int = 32,
        beta: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)

        # Four stride-2 convs each halve the spatial resolution: 16 = 2 ** 4.
        self.feature_hw = self.image_size // 16
        self.feature_dim = 64 * self.feature_hw * self.feature_hw

        # Encoder: 32, 32, 64, 64 channel 4x4 stride-2 convolutions.
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
        )
        # Separate heads for the Gaussian latent parameters.
        self.fc_mu = nn.Linear(256, self.latent_dim)
        self.fc_logvar = nn.Linear(256, self.latent_dim)

        # Decoder mirrors the encoder.
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, self.feature_hw, self.feature_hw)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, self.in_channels, 4, 2, 1),
            nn.Sigmoid(),  # Bernoulli decoder: outputs in [0, 1].
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map images to the (mu, logvar) of the approximate posterior."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) with the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a latent code back to an image in [0, 1]."""
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Beta-VAE objective: reconstruction + beta * KL, per-image.

        Reconstruction is summed binary cross-entropy over all pixels
        (Bernoulli decoder), and the KL has the standard closed form against a
        standard-normal prior. Both are averaged over the batch so the loss
        scale is independent of batch size.
        """
        beta = self.beta if beta is None else beta
        batch_size = x.shape[0]
        recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / batch_size
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return recon_loss + beta * kl, recon_loss, kl

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent (posterior mean) used as the RSA feature."""
        self.eval()
        mu, _ = self.encode(x)
        return mu


# ---------------------------------------------------------------------------
# Training program
#
# Trains the simple beta-VAE on one COCO 2014 split and evaluates on another
# (e.g. train on val2014 while the full train2014 download finishes, evaluate
# on test2014). Use --dev for a fast subset run during development.
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DEFAULT_COCO_ROOT = HERE.parent / "COCO"
DEFAULT_OUTPUT_DIR = HERE / "results" / "simple_beta_vae"


class CocoFolderDataset(Dataset):
    """A folder of COCO JPEGs returned as [0, 1] CHW float tensors.

    Images are converted to RGB (a few COCO images are grayscale) and resized
    to a square ``image_size``. ToTensor scales to [0, 1], matching the
    sigmoid/BCE decoder of :class:`BetaVAE`.
    """

    def __init__(
        self,
        root: Path,
        image_size: int,
        limit: int | None = None,
        seed: int = 0,
    ) -> None:
        self.paths = list_images(root, limit, seed)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.paths[idx]).convert("RGB")
        return cast(torch.Tensor, self.transform(image))


def list_images(root: Path, limit: int | None, seed: int) -> list[str]:
    root = Path(root)
    paths = sorted(str(p) for p in root.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in {root}")
    if limit is not None and limit < len(paths):
        # Deterministic subset so eval / dev runs are comparable across epochs.
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, limit))
    return paths


def format_float_for_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_name(beta: float, latent_dim: int, seed: int) -> str:
    return f"betavae_beta{format_float_for_name(beta)}_z{latent_dim}_seed{seed}"


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def run_epoch(
    model: BetaVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one pass; returns mean (loss, recon, kl). Train if optimizer given."""
    training = optimizer is not None
    model.train(training)
    totals = [0.0, 0.0, 0.0]
    n = 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            recon, mu, logvar = model(x)
            loss, recon_loss, kl = model.loss_function(recon, x, mu, logvar)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bs = x.shape[0]
        totals[0] += loss.item() * bs
        totals[1] += recon_loss.item() * bs
        totals[2] += kl.item() * bs
        n += bs
    return totals[0] / n, totals[1] / n, totals[2] / n


@torch.no_grad()
def save_samples(
    model: BetaVAE, batch: torch.Tensor, path: Path, device: torch.device
) -> None:
    """Save a grid of input images (top) above their reconstructions."""
    model.eval()
    x = batch[:8].to(device)
    recon, _, _ = model(x)
    grid = torch.cat([x, recon.clamp(0.0, 1.0)], dim=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path, nrow=8)


def save_checkpoint(path: Path, model: BetaVAE, epoch: int, eval_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "image_size": model.image_size,
                "in_channels": model.in_channels,
                "latent_dim": model.latent_dim,
                "beta": model.beta,
            },
            "epoch": epoch,
            "eval_loss": eval_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the simple beta-VAE on COCO 2014.")
    p.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    p.add_argument("--train-split", default="val2014")
    p.add_argument("--eval-split", default="test2014")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--eval-limit", type=int, default=4096, help="eval subset size")
    p.add_argument("--train-limit", type=int, default=None, help="cap train images")
    p.add_argument("--sample-every", type=int, default=5, help="epochs between grids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dev",
        action="store_true",
        help="quick subset run: 2000 train / 512 eval images, 3 epochs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dev:
        args.train_limit = 2000
        args.eval_limit = 512
        args.epochs = 3
        args.num_workers = min(args.num_workers, 4)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dir = args.coco_root / args.train_split
    eval_dir = args.coco_root / args.eval_split
    train_dataset = CocoFolderDataset(
        train_dir, args.image_size, args.train_limit, args.seed
    )
    eval_dataset = CocoFolderDataset(
        eval_dir, args.image_size, args.eval_limit, args.seed
    )
    train_loader = build_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    eval_loader = build_loader(
        eval_dataset, args.batch_size, False, args.num_workers, device
    )

    model = BetaVAE(args.image_size, 3, args.latent_dim, args.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    name = run_name(args.beta, args.latent_dim, args.seed)
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_batch = next(iter(eval_loader))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | params: {n_params:,}")
    print(f"Train: {train_dir} ({len(train_dataset):,} images)")
    print(f"Eval:  {eval_dir} ({len(eval_dataset):,} images)")
    print(f"Run dir: {run_dir}")

    history: list[dict] = []
    best_eval = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_recon, tr_kl = run_epoch(model, train_loader, optimizer, device)
        ev_loss, ev_recon, ev_kl = run_epoch(model, eval_loader, None, device)
        dt = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "seconds": dt,
                "train": {"loss": tr_loss, "recon": tr_recon, "kl": tr_kl},
                "eval": {"loss": ev_loss, "recon": ev_recon, "kl": ev_kl},
            }
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={tr_loss:.1f} (recon={tr_recon:.1f} kl={tr_kl:.2f}) "
            f"eval={ev_loss:.1f} (recon={ev_recon:.1f} kl={ev_kl:.2f}) "
            f"{dt:.1f}s"
        )

        save_checkpoint(run_dir / "last.pt", model, epoch, ev_loss)
        if ev_loss < best_eval:
            best_eval = ev_loss
            save_checkpoint(run_dir / "best.pt", model, epoch, ev_loss)
        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            save_samples(
                model, sample_batch, run_dir / "samples" / f"epoch{epoch:03d}.png", device
            )

    print(f"Done. Best eval loss {best_eval:.1f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
