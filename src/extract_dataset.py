"""
extract_dataset.py
==================
One-time script: Extract all NIR.zip and PulseOx.zip files in the dataset.
Run this ONCE before training. Training will then be ~10x faster.

Run from src\:
    python extract_dataset.py

Or with a custom data root:
    python extract_dataset.py --data_root F:\my\custom\path\MR-NIRP-Car
"""

import sys
import zipfile
import argparse
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def extract_zip(zip_path: Path, dest_dir: Path, name: str) -> bool:
    """
    Extract zip_path into dest_dir.
    Skips if dest_dir already exists and has files.
    Returns True if extracted, False if skipped.
    """
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info(f"  SKIP (already extracted): {name}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Extracting {name} ({zip_path.stat().st_size / 1e9:.2f} GB)...")
    t0 = time.time()

    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(dest_dir))

    elapsed = time.time() - t0
    n_files = sum(1 for _ in dest_dir.rglob('*') if _.is_file())
    logger.info(f"  Done: {n_files} files in {elapsed:.1f}s")
    return True


def process_dataset(data_root: Path, dry_run: bool = False):
    if not data_root.exists():
        logger.error(f"Dataset root not found: {data_root}")
        sys.exit(1)

    # Find all sessions with zip files
    sessions_to_extract = []
    for subj_dir in sorted(data_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        for cond_dir in sorted(subj_dir.iterdir()):
            if not cond_dir.is_dir():
                continue

            nir_zip = cond_dir / 'NIR.zip'
            pox_zip = cond_dir / 'PulseOx.zip'

            if nir_zip.exists() or pox_zip.exists():
                sessions_to_extract.append(cond_dir)

    if not sessions_to_extract:
        logger.info("No zip files found. Either already extracted or no data downloaded.")
        return

    logger.info(f"Found {len(sessions_to_extract)} sessions to extract:")
    for s in sessions_to_extract:
        logger.info(f"  {s.relative_to(data_root)}")

    if dry_run:
        logger.info("\nDry run — not extracting. Remove --dry_run to actually extract.")
        return

    # Calculate total size
    total_gb = 0
    for cond_dir in sessions_to_extract:
        for z in [cond_dir / 'NIR.zip', cond_dir / 'PulseOx.zip']:
            if z.exists():
                total_gb += z.stat().st_size / 1e9
    logger.info(f"\nTotal zip size to extract: {total_gb:.1f} GB")
    logger.info("Starting extraction...\n")

    total_extracted = 0
    total_skipped   = 0

    for cond_dir in sessions_to_extract:
        logger.info(f"Session: {cond_dir.name}")

        # Extract NIR.zip -> NIR\
        nir_zip = cond_dir / 'NIR.zip'
        nir_dir = cond_dir / 'NIR'
        if nir_zip.exists():
            extracted = extract_zip(nir_zip, nir_dir, 'NIR.zip')
            if extracted:
                total_extracted += 1
                # Optional: delete zip after successful extraction to save disk space
                # nir_zip.unlink()
            else:
                total_skipped += 1

        # Extract PulseOx.zip -> PulseOx\
        pox_zip = cond_dir / 'PulseOx.zip'
        pox_dir = cond_dir / 'PulseOx'
        if pox_zip.exists():
            extracted = extract_zip(pox_zip, pox_dir, 'PulseOx.zip')
            if extracted:
                total_extracted += 1
            else:
                total_skipped += 1

        logger.info("")

    logger.info("=" * 50)
    logger.info(f"Extraction complete.")
    logger.info(f"  Extracted : {total_extracted}")
    logger.info(f"  Skipped   : {total_skipped} (already done)")
    logger.info("")
    logger.info("Now run training:")
    logger.info("  python train.py --data_root ..\\data\\MR-NIRP-Car --epochs 50")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract MR-NIRP dataset zips')
    p.add_argument('--data_root', default=r'..\data\MR-NIRP-Car',
                   help='Path to MR-NIRP-Car folder')
    p.add_argument('--dry_run', action='store_true',
                   help='List what would be extracted without doing it')
    args = p.parse_args()

    root = Path(args.data_root)
    if not root.is_absolute():
        root = (Path(__file__).parent / root).resolve()

    logger.info(f"Dataset root: {root}")
    process_dataset(root, dry_run=args.dry_run)