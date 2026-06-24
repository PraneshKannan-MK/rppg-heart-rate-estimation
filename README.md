# MR-NIRP Heart Rate Estimation from NIR Facial Videos

PhysNet 3D-CNN for remote photoplethysmography (rPPG) using the MR-NIRP Car dataset.

## Dataset

**MR-NIRP Car** — [Rice Computational Imaging Lab](https://computationalimaging.rice.edu/mr-nirp-dataset/)

Key facts verified by data inspection:
- 19 subject folders (18 unique subjects; Subject2 = Subject16)
- 10 conditions per subject (driving + garage × still + small_motion + large_motion × 940nm + 975nm)
- NIR frames: PGM P5 format, 640×640, **16-bit** (maxval=65535)
- Ground truth: `pulseOx.mat` — raw PPG at ~57.76 fps (variable, not fixed 30fps)
- **Synchronization**: `cam0_full_log.txt` contains timestamps identical to `pulseOxTime`. NIR frame N ↔ PulseOx sample N (1:1 mapping, no fixed ratio)

### Download
```bash
# From Google Drive: https://drive.google.com/drive/folders/1U3fzIOESmaBAyikGF0cKI2wW3YK8JqCK
# Unzip into ./data/MR-NIRP-Car/
# Structure: data/MR-NIRP-Car/Subject1/subject1_driving_still_940/{NIR/, PulseOx/}
```

## Setup

```bash
pip install torch torchvision opencv-python scipy numpy matplotlib
```

Optional (better face detection):
```bash
pip install mediapipe  # version ≥ 0.10 required
```

## Project Structure

```
rppg_project/
├── src/
│   ├── gt_preprocessing.py   # Load pulseOx.mat, extract BPM via FFT
│   ├── face_roi.py           # Load 16-bit PGM, detect/cache face ROI
│   ├── dataset.py            # PyTorch Dataset with sliding window clips
│   ├── model.py              # PhysNet 3D-CNN (single-channel NIR)
│   ├── train.py              # Training loop, checkpointing
│   ├── evaluate.py           # Metrics (MAE, RMSE, Pearson r), plots
│   └── demo_train.py         # Demo with minimal data
├── configs/
│   └── config.yaml
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── plots/
```

## Training

```bash
# Full training
python src/train.py \
  --data_root ./data/MR-NIRP-Car \
  --epochs 50 \
  --batch_size 4 \
  --clip_len 128 \
  --stride 64

# Demo (verifies pipeline without full dataset)
python src/demo_train.py
```

## Architecture

**PhysNet 3D-CNN** (adapted from Yu et al., BMVC 2019)

```
Input: (B, 1, 128, 64, 64) — single-channel NIR clip
  ↓
Block 1: Conv3D(1→32, 1×5×5) → BN → ReLU → MaxPool(1×2×2)
  ↓
Block 2: Conv3D(32→64, 3×3×3) → BN → ReLU → MaxPool(2×2×2)
  ↓
Block 3: Conv3D(64→128, 3×3×3) → BN → ReLU → MaxPool(2×2×2)
  ↓
Block 4: Conv3D(128→128, 3×3×3) → BN → ReLU → MaxPool(2×2×2)
  ↓
AdaptiveAvgPool3D(16, 1, 1)
  ↓
Flatten → Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→64) → Linear(64→1)
  ↓
Output: BPM prediction (scalar)
```

**Modification from original**: Single-channel input (NIR grayscale) instead of 3-channel RGB.  
**Parameters**: ~1.26M

## Ground Truth Preprocessing

`pulseOx.mat` contains raw 10-bit PPG (range ~128–234). Processing pipeline:
1. Load via `scipy.io.loadmat` (or `pickle.load(encoding='latin1')` for .pkl)
2. Mean-subtract (remove DC)
3. Per-window: detrend → bandpass filter [0.75–2.5 Hz] → Hann window → FFT → peak freq → BPM
4. Timestamp-based FPS computation (actual fps varies, do not assume fixed 30fps)

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (primary metric) |
| RMSE | Root Mean Square Error |
| SD | Standard Deviation of errors |
| Pearson r | Correlation between predicted and ground truth HR |

Expected performance on MR-NIRP (still/small_motion conditions):
- Published PhysNet: MAE ~8–14 BPM (this is a challenging dataset)
- Accuracy is secondary — methodology is assessed

## Subject Split

```
Train : Subject 1,3,4,5,6,7,8,9,10,11,12,13  (12 subjects)
Val   : Subject 14,15,17,18                    (4 subjects)
Test  : Subject 19, Subject 2                  (Subject16 = Subject2; excluded)
```

**Important**: Subject2 and Subject16 are the same person. Never put them in different splits.

## References

- Yu, Z. et al. "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks." BMVC 2019. [arxiv.org/abs/1905.02419](https://arxiv.org/abs/1905.02419)
- Nowara, E.M. et al. "Near-Infrared Imaging Photoplethysmography During Driving." IEEE TITS, 2020. DOI: 10.1109/TITS.2020.3038317
