"""Download BLT MPNet weights from the public NSD S3 bucket."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


BUCKET = "natural-scenes-dataset"
PREFIX = "other/blt_mpnet_weights/"
REGION = "us-east-2"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "blt_mpnet_weights"


def download_blt_mpnet_weights(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    anonymous: bool = True,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> None:
    """Download all files under s3://natural-scenes-dataset/other/blt_mpnet_weights."""

    output_dir = Path(output_dir)
    bucket, client = _get_s3_handles(anonymous=anonymous)

    total_files = 0
    downloaded_files = 0
    skipped_files = 0
    total_bytes = 0

    for obj in bucket.objects.filter(Prefix=PREFIX):
        if obj.key.endswith("/"):
            continue

        relative_key = obj.key[len(PREFIX) :]
        target_file = output_dir / relative_key
        total_files += 1
        total_bytes += obj.size

        if (
            skip_existing
            and target_file.exists()
            and target_file.stat().st_size == obj.size
        ):
            print(f"Skipping existing {target_file}")
            skipped_files += 1
            continue

        print(f"Downloading s3://{BUCKET}/{obj.key} -> {target_file}")
        downloaded_files += 1
        if dry_run:
            continue

        target_file.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(BUCKET, obj.key, str(target_file))

    print(
        f"Done: {total_files} files matched, {downloaded_files} downloaded, "
        f"{skipped_files} skipped, {total_bytes / (1000**3):.2f} GB total."
    )


def _get_s3_handles(*, anonymous: bool):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    config = Config(signature_version=UNSIGNED) if anonymous else None
    session = boto3.Session()
    resource = session.resource("s3", region_name=REGION, config=config)
    client = session.client("s3", region_name=REGION, config=config)
    return resource.Bucket(BUCKET), client


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory where weights should be downloaded.",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Use configured AWS credentials instead of anonymous S3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if a matching local file exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be downloaded without writing them.",
    )
    args = parser.parse_args(argv)

    download_blt_mpnet_weights(
        output_dir=args.output_dir,
        anonymous=not args.signed,
        skip_existing=not args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
