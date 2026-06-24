"""
dataset.py
===========
PyTorch Dataset for MR-NIRP Car Dataset (Windows, zip-aware)

Handles ALL 19 subjects x 10 conditions automatically.
Works with zipped (NIR.zip, PulseOx.zip) or extracted folders.

Subject split (from README — Subject2 == Subject16, never split them):
  Train : Subject1,3,4,5,6,7,8,9,10,11,12,13
  Val   : Subject14,15,17,18
  Test  : Subject19, Subject2  (Subject16 is excluded -- same person as Subject2)

Valid conditions (large_motion excluded per original paper):
  *still_940, *still_975, *small_motion_940, *small_motion_975
"""

import os
import sys
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

# Make sure src/ imports work whether running from src/ or root
sys.path.insert(0, str(Path(__file__).parent))

from face_roi import FaceROIExtractor, load_clip_frames, load_clip_from_arrays
from gt_preprocessing import SessionData, HR_MIN_BPM, HR_MAX_BPM

logger = logging.getLogger(__name__)

# ─── Subject splits ───────────────────────────────────────────────────────────
TRAIN_SUBJECTS = [f'Subject{i}' for i in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
VAL_SUBJECTS   = [f'Subject{i}' for i in [14, 15, 17, 18]]
TEST_SUBJECTS  = [f'Subject{i}' for i in [19, 2]]
# Subject16 intentionally excluded from TEST — it is the same person as Subject2

VALID_CONDITIONS = ['still_940', 'still_975', 'small_motion_940', 'small_motion_975']


def find_sessions(dataset_root: str, subjects: list) -> list:
    """
    Find all valid, loadable condition folders for the given subjects.
    Skips large_motion and any folder missing NIR/PulseOx data.
    Returns sorted list of Path objects.
    """
    root     = Path(dataset_root)
    sessions = []

    for subj_name in subjects:
        subj_dir = root / subj_name
        if not subj_dir.exists():
            logger.debug(f"Subject not found (not downloaded yet): {subj_dir}")
            continue

        for cond_dir in sorted(subj_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            name = cond_dir.name.lower()

            # Skip large_motion (too noisy, excluded in original paper)
            if 'large_motion' in name:
                continue

            # Must match at least one valid condition suffix
            if not any(c in name for c in VALID_CONDITIONS):
                continue

            # Must have NIR data
            has_nir = (cond_dir / 'NIR').exists() or (cond_dir / 'NIR.zip').exists()
            # Must have PulseOx data
            has_pox = ((cond_dir / 'PulseOx').exists() or
                       (cond_dir / 'PulseOx.zip').exists() or
                       (cond_dir / 'pulseOx.mat').exists())

            if has_nir and has_pox:
                sessions.append(cond_dir)
            else:
                logger.debug(f"Skipping {cond_dir.name} (missing NIR={has_nir} PulseOx={has_pox})")

    logger.info(f"Found {len(sessions)} valid sessions for subjects {subjects}")
    return sessions


class MRNIRPDataset(Dataset):
    """
    Sliding-window Dataset over all sessions for the given subjects.

    Each __getitem__ returns:
        clip  : torch.float32 tensor (1, T, H, W)   [C=1 = grayscale NIR]
        bpm   : torch.float32 scalar

    Supports both:
        - NIR\ folder   (if you extracted the zips manually)
        - NIR.zip       (direct from Google Drive, no extraction needed)
    """

    def __init__(self, dataset_root: str, subjects: list,
                 clip_len: int = 128, stride: int = 64, img_size: int = 64):
        """
        Parameters
        ----------
        dataset_root : path to MR-NIRP-Car root  e.g. r'F:\\data\\MR-NIRP-Car'
        subjects     : list of subject folder names e.g. ['Subject1', 'Subject3']
        clip_len     : frames per clip (128 @ ~57fps ≈ 2.2 seconds)
        stride       : sliding window step
        img_size     : face ROI resize target (square)
        """
        self.clip_len  = clip_len
        self.img_size  = img_size
        self.extractor = FaceROIExtractor(target_size=img_size)

        # Index: list of (SessionData, start_frame, end_frame, bpm)
        self.samples = []
        sessions = find_sessions(dataset_root, subjects)

        for sess_dir in sessions:
            try:
                sess  = SessionData(str(sess_dir))
                clips = sess.get_clips(clip_len, stride)
                for start, end, bpm in clips:
                    self.samples.append((sess, start, end, bpm))
                if clips:
                    logger.info(f"  {sess_dir.name}: {len(clips)} clips | "
                                f"BPM~{sess.get_session_bpm():.1f}")
            except Exception as ex:
                logger.warning(f"Skipping {sess_dir.name}: {ex}")

        logger.info(f"Dataset ready: {len(self.samples)} total clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sess, start, end, bpm = self.samples[idx]

        # Load frames — zip-aware via SessionData.load_frames()
        raw_frames = sess.load_frames(start, end)   # list of uint8 (640,640)
        clip_arr   = load_clip_from_arrays(raw_frames, self.extractor)  # (T, H, W) float32

        clip_tensor = torch.from_numpy(clip_arr).unsqueeze(0)          # (1, T, H, W)
        bpm_tensor  = torch.tensor(bpm, dtype=torch.float32)

        return clip_tensor, bpm_tensor


def build_dataloaders(dataset_root: str,
                      batch_size:   int = 4,
                      clip_len:     int = 128,
                      stride:       int = 64,
                      img_size:     int = 64,
                      num_workers:  int = 0):
    """
    Build train / val / test DataLoaders.

    num_workers=0 is the safe default on Windows
    (multiprocessing with workers > 0 can cause issues on Windows).
    """
    train_ds = MRNIRPDataset(dataset_root, TRAIN_SUBJECTS, clip_len, stride,       img_size)
    val_ds   = MRNIRPDataset(dataset_root, VAL_SUBJECTS,   clip_len, stride // 2,  img_size)
    test_ds  = MRNIRPDataset(dataset_root, TEST_SUBJECTS,  clip_len, stride // 2,  img_size)

    kw = dict(batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)
    test_dl  = DataLoader(test_ds,  shuffle=False, drop_last=False, **kw)

    return train_dl, val_dl, test_dl


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    # Try to find the dataset
    candidates = [
        Path(r'F:\UST\project_2\rppg_project\data\MR-NIRP-Car'),
        Path(__file__).parent.parent / 'data' / 'MR-NIRP-Car',
    ]
    root = next((c for c in candidates if c.exists()), None)

    if root is None:
        print("Dataset not found — skipping dataset test")
        print("Download subjects first, then run this again.")
    else:
        print(f"Dataset root: {root}")
        sessions = find_sessions(str(root), TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS)
        print(f"Total valid sessions found: {len(sessions)}")
        for s in sessions[:5]:
            print(f"  {s.relative_to(root)}")
        if len(sessions) > 5:
            print(f"  ... and {len(sessions)-5} more")