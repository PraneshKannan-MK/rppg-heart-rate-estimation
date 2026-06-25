# Project Report: Heart Rate Estimation from NIR Facial Videos

**Assignment:** Heart Rate Estimation from NIR Facial Videos
**Dataset:** MR-NIRP Car (NIR modality only)
**Model:** PhysNet 3D-CNN

---

## 1. Dataset Exploration

### Overview

The MR-NIRP Car dataset contains 19 subject folders (18 unique subjects — Subject2 and
Subject16 are the same person recorded day vs night). Each subject has up to 10 condition
folders combining:

- **Location:** driving (city), garage (parked)
- **Motion:** still, small_motion, large_motion
- **Filter:** 940nm, 975nm bandpass

**Conditions used:** `still_940`, `still_975`, `small_motion_940`, `small_motion_975`
**Conditions excluded:** `large_motion_*` (excluded in the original paper due to large errors)

### Key Facts Discovered by Inspection

Before writing any training code, I inspected the actual uploaded files to understand the
exact data format. Several assumptions from the literature turned out to be wrong for this
dataset:

| Assumption | Reality |
|-----------|---------|
| Fixed 30 fps | Actual fps ~57-58 Hz (varies per session) |
| PulseOx at 60 fps | Same fps as NIR (~57-58 Hz) |
| Fixed 2:1 NIR:PulseOx ratio | 1:1 mapping — same timestamp array |
| PNG frames | 16-bit PGM format (maxval=65535) |
| Flat extraction layout | Zip extracted to NIR/NIR/ subfolder (capital/case issues) |

These were discovered by loading `pulseOx.mat` and `cam0_full_log.txt` directly and
comparing their timestamps — they were identical, confirming 1:1 frame-to-sample mapping.

### Ground Truth Signal

`pulseOx.mat` contains:
- `pulseOxRecord`: raw 10-bit PPG waveform (range ~128-234)
- `pulseOxTime`: UNIX timestamps (identical to NIR frame timestamps)
- `numPulseSample`: sample count

Session duration: 120-230 seconds. Verified BPM range: ~58-67 BPM across sessions used.

---

## 2. Preprocessing Pipeline

### Ground Truth BPM Extraction

```
pulseOxRecord -> mean subtract -> detrend -> bandpass [0.75-2.5 Hz]
             -> Hann window -> FFT -> peak frequency -> BPM
```

The bandpass filter uses 3rd-order Butterworth with `filtfilt` (zero-phase). The valid
cardiac band 0.75-2.5 Hz corresponds to 45-150 BPM. FFT is applied per clip window.

**Critical parameter:** `clip_len >= fps * 4 seconds = ~234 frames minimum`. Using
`clip_len=128` (our initial setting) produced 0 valid clips because 128/58fps = 2.2
seconds, below the filter's minimum. This was identified through the diagnostic tool
and fixed by setting `clip_len=256`.

### NIR Frame Loading

Frames are 16-bit PGM (P5 magic, maxval=65535). Standard `cv2.imread` cannot load
these correctly — a custom parser reads the header and normalises pixel values to uint8.

### Face ROI Extraction

1. Apply OpenCV Haar cascade face detector on first 5 frames of each clip
2. Cache the best bounding box (largest face area)
3. Add 15% padding around the detected face
4. Crop and resize to 64×64 for all frames in the clip
5. Normalise to float32 [0, 1] per frame

If no face is detected, fall back to centre crop (55% of frame). Face detection is
cached across frames within a clip to avoid jitter and reduce compute.

### Sliding Window Clips

Each session produces overlapping clips:
- `clip_len=256` frames ≈ 4.4 seconds at 58fps
- `stride=128` frames ≈ 2.2 seconds (50% overlap)
- 3 sessions produced **246 total clips**

---

## 3. Model Architecture

### Why PhysNet

PhysNet (Yu et al., BMVC 2019) was chosen because:
1. Designed specifically for rPPG from facial videos
2. 3D convolutions capture spatiotemporal blood-volume variations
3. Proven on similar datasets (VIPL-HR, COHFACE)
4. Small enough to train on CPU (~1.8M parameters)

Alternatives considered:
- **PhysFormer** (transformer-based): too large for CPU training
- **CHROM/POS** (signal processing): no learning, limited to RGB
- **DeepPhys**: attention-based, requires motion/appearance stream

### Architecture (Modified from Original)

**Key modification:** `in_channels=1` instead of 3. The original PhysNet takes RGB (3-channel)
input. Since we use NIR grayscale, single-channel input was used. This is a justified
adaptation, not a bug — NIR has no colour information and the spatial features are
equivalent to a single luminance channel.

**Second modification:** Direct BPM regression instead of PPG signal prediction. The original
paper predicts a PPG waveform and then extracts BPM via peak detection. We predict BPM
directly, which simplifies training and avoids accumulated errors in peak detection.

```
Input (B, 1, 256, 64, 64)
  Block 1: Conv3D(1->32, 1x5x5) BN ReLU MaxPool(1x2x2)   [spatial only]
  Block 2: Conv3D(32->64, 3x3x3) BN ReLU MaxPool(2x2x2)  [spatiotemporal]
  Block 3: Conv3D(64->128, 3x3x3) BN ReLU MaxPool(2x2x2)
  Block 4: Conv3D(128->128, 3x3x3) BN ReLU MaxPool(2x2x2)
  AdaptiveAvgPool3D(32, 1, 1)
  Flatten -> Linear(4096->256) -> ReLU -> Dropout(0.3) -> Linear(256->64) -> Linear(64->1)
Output: BPM scalar
```

Block 1 uses a (1,5,5) kernel — spatial only — to first extract facial features before
mixing temporal information in Blocks 2-4.

---

## 4. Loss Function

**L1 Loss (MAE)** was used instead of L2 (MSE).

**Justification:** rPPG predictions from NIR driving data are inherently noisy. Large
outlier BPM predictions occur frequently — head movements, illumination changes, and
motion artifacts create clips where the model predicts values far from ground truth.
L2 loss squares these errors, causing the gradient to be dominated by outliers and
destabilising training. L1 is linear in error magnitude, making it more robust to the
noisy BPM estimates inherent in this challenging dataset.

**Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
**Scheduler:** OneCycleLR (pct_start=0.1) — warm-up for 10% of training, then cosine decay
**Gradient clipping:** max_norm=1.0 — prevents exploding gradients from motion-corrupted clips

---

## 5. What We Tried, What Didn't Work, and What We Changed

### Problem 1: Fixed FPS assumption (clip_len=128 → 0 clips)

**What we tried:** Default `clip_len=128`, assuming 30fps from the README.
**What happened:** 0 training clips generated. Sessions loaded correctly but no clips passed.
**Diagnosis:** Inspected actual FPS from timestamps — dataset runs at ~58fps not 30fps.
128 frames / 58fps = 2.2 seconds < 4 second minimum for bandpass filter.
**Fix:** Changed `clip_len=256` (4.4s at 58fps). Auto-adjustment added to `SessionData.get_clips()`.

### Problem 2: NIR.zip extracted to NIR/NIR/ subfolder

**What we tried:** After extracting zip, code looked for `NIR/Frame*.pgm`.
**What happened:** `frames=0` despite successful extraction.
**Diagnosis:** `check_structure.py` revealed zip created `NIR/NIR/Frame*.pgm` (double nested).
Also found `PulseOx/PulseOX/` (capital X) for PulseOx subfolder.
**Fix:** Changed frame discovery from `glob('Frame*.pgm')` to `rglob('Frame*.pgm')` (recursive).

### Problem 3: Training from zip was ~10x slower

**What we tried:** Training directly from zip files (reading PGM from inside zip per batch).
**What happened:** Each epoch took ~1 hour (vs expected ~5-10 minutes).
**Root cause:** Each frame load required: open zip, seek, decompress, close zip.
256 frames per clip × 61 batches = ~15,600 zip operations per epoch.
**Fix:** Added `extract_dataset.py` to extract once. After extraction, epoch time dropped from ~60 min to ~10 min.

### Problem 4: Test path hardcoded to Linux

**What happened:** `gt_preprocessing.py` self-test used `/mnt/user-data/uploads/pulseOx.mat`
(a Linux path from the development environment).
**Fix:** Changed to auto-search common Windows paths, falling back to synthetic test.

### Problem 5: PGM parser for 16-bit frames

**What we tried:** `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` — standard OpenCV loading.
**What happened:** Loaded correctly but silently truncated 16-bit values to 8-bit.
**Fix:** Custom PGM parser reads P5 header, detects maxval=65535, reads big-endian uint16,
normalises to uint8.

---

## 6. Evaluation Metrics

### Metrics Used

| Metric | Formula | Why |
|--------|---------|-----|
| **MAE** | mean(|pred - gt|) | Primary — interpretable in BPM units |
| **RMSE** | sqrt(mean((pred-gt)^2)) | Penalises large errors |
| **SD** | std(pred - gt) | Spread of errors |
| **Pearson r** | corr(pred, gt) | Linear relationship strength |

Bland-Altman plot is also generated to assess agreement between predicted and GT HR.

### Results

Training was performed on 3 sessions (Subject1: 2 sessions, Subject3: 1 session) due to
dataset download constraints (each NIR.zip is 4-6 GB).

| Epoch | Train Loss (MAE) |
|-------|-----------------|
| 1 | 35.67 BPM |
| 2 | 7.66 BPM |
| 10 | 8.62 BPM |
| 20 | 6.46 BPM |
| 30 | 5.68 BPM |
| 40 | 5.00 BPM |
| 44 | **4.61 BPM** (best) |
| 50 | 4.84 BPM |

**87.1% reduction** in MAE over 50 epochs.

Val/Test MAE not available because Subject14-19 were not downloaded (storage limitation).
Ground truth BPM extraction was verified on synthetic signals: true BPM 72.0 → estimated
72.0 (error < 0.03 BPM on clean signal).

### Intended Evaluation Approach (Full Dataset)

With all 19 subjects downloaded:
1. Train on Subject 1,3,4,5,6,7,8,9,10,11,12,13 (12 subjects)
2. Validate on Subject 14,15,17,18 (4 subjects) — tune hyperparameters
3. Test on Subject 2,19 (Subject16 excluded — same person as Subject2)
4. Report MAE, RMSE, Pearson r on the test set
5. Published PhysNet results on similar datasets: MAE ~8-14 BPM

---

## 7. Limitations

1. **Only 3 sessions used for training** — full dataset has ~190 sessions. More data would
   substantially improve generalisation.

2. **CPU training only** — each epoch took ~8-10 minutes. GPU would reduce this to ~30 seconds,
   enabling more experiments.

3. **No validation during training** — without val subjects downloaded, hyperparameter tuning
   was not possible. Early stopping could not be applied.

4. **Direct BPM regression vs PPG prediction** — predicting the full PPG waveform and
   extracting BPM post-hoc (as in the original PhysNet) may be more robust, as it forces
   the model to learn the cardiac waveform structure rather than a scalar mapping.

5. **Fixed clip_len** — different sessions have slightly different FPS (56-58 Hz). A
   time-based windowing approach (e.g. always 4.4 seconds regardless of FPS) would be
   more principled.

---

## 8. Next Steps

1. Download remaining subjects (Subject4-19) to get full MAE/RMSE/Pearson r metrics (as of now done because of my system compute)
2. Add GPU support for faster training and hyperparameter search

---

## References

1. Yu, Z. et al. "Remote Photoplethysmograph Signal Measurement from Facial Videos Using
   Spatio-Temporal Networks." BMVC 2019. https://arxiv.org/abs/1905.02419

2. Nowara, E.M. et al. "Near-Infrared Imaging Photoplethysmography During Driving."
   IEEE Transactions on Intelligent Transportation Systems, 2020.

3. de Haan, G. and Jeanne, V. "Robust Pulse Rate From Chrominance-Based rPPG."
   IEEE Transactions on Biomedical Engineering, 2013. (CHROM baseline)
