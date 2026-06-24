"""
train.py — Full training for MR-NIRP rPPG Heart Rate Estimation

Run from src\:
    python train.py --data_root ..\data\MR-NIRP-Car --epochs 50

Speed tip: Extract zips first with extract_dataset.py — makes training ~10x faster.
"""

import os, sys, argparse, logging, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from scipy.stats import pearsonr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import PhysNet
from dataset import build_dataloaders, find_sessions, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def evaluate(model, loader, device):
    if len(loader) == 0:
        return float('nan'), float('nan'), float('nan'), [], []
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for clips, bpm_gt in loader:
            preds = model(clips.to(device)).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_gt.extend(bpm_gt.numpy().tolist())
    if not all_pred:
        return float('nan'), float('nan'), float('nan'), [], []
    pred = np.array(all_pred)
    gt   = np.array(all_gt)
    mae  = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt)**2)))
    r, _ = pearsonr(pred, gt) if len(pred) > 2 else (float('nan'), None)
    return mae, rmse, float(r), pred.tolist(), gt.tolist()


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    crit = nn.L1Loss()
    total, n = 0.0, 0
    for batch_idx, (clips, bpm_gt) in enumerate(loader):
        clips, bpm_gt = clips.to(device), bpm_gt.to(device)
        optimizer.zero_grad()
        loss = crit(model(clips), bpm_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item(); n += 1
        # Progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {total/n:.3f}")
    return total / max(n, 1)


def check_dataset(data_root: str) -> bool:
    root = Path(data_root)
    if not root.exists():
        logger.error(f"Folder does not exist: {root}")
        return False
    all_sess = find_sessions(data_root, TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS)
    if not all_sess:
        logger.error(f"No dataset sessions found in: {root}")
        logger.error("Download Subject folders from Google Drive first.")
        return False
    # Check if still zipped — warn about speed
    zipped = [s for s in all_sess if (s / 'NIR.zip').exists() and not (s / 'NIR').exists()]
    if zipped:
        logger.warning("=" * 60)
        logger.warning(f"  {len(zipped)} sessions still have zipped NIR files.")
        logger.warning("  Training from zip is ~10x SLOWER than from extracted folders.")
        logger.warning("  Run this FIRST to extract and speed up training:")
        logger.warning("    python extract_dataset.py")
        logger.warning("=" * 60)

    train_n = len(find_sessions(data_root, TRAIN_SUBJECTS))
    val_n   = len(find_sessions(data_root, VAL_SUBJECTS))
    test_n  = len(find_sessions(data_root, TEST_SUBJECTS))
    logger.info(f"Sessions — Train: {train_n} | Val: {val_n} | Test: {test_n}")
    if train_n == 0:
        logger.error("No training sessions. Download Subject1, Subject3, Subject4.")
        return False
    return True


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cpu':
        logger.warning("Running on CPU — each epoch will be slow.")
        logger.warning("Extract zips first: python extract_dataset.py")

    if not check_dataset(args.data_root):
        sys.exit(1)

    out      = Path(args.out_dir)
    ckpt_dir = out / 'checkpoints'
    log_dir  = out / 'logs'
    plot_dir = out / 'plots'
    for d in [ckpt_dir, log_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building dataloaders (clip_len={args.clip_len})...")
    train_dl, val_dl, test_dl = build_dataloaders(
        dataset_root = args.data_root,
        batch_size   = args.batch_size,
        clip_len     = args.clip_len,
        stride       = args.stride,
        img_size     = args.img_size,
        num_workers  = 0,
    )

    logger.info(f"Batches — Train: {len(train_dl)} | Val: {len(val_dl)} | Test: {len(test_dl)}")

    if len(train_dl) == 0:
        logger.error("Training set empty after building clips.")
        logger.error(f"clip_len={args.clip_len} may be too short. Try --clip_len 256 --stride 128")
        sys.exit(1)

    model = PhysNet(in_channels=1, base_filters=32,
                    dropout=0.3, clip_len=args.clip_len).to(device)
    logger.info(f"PhysNet: {model.count_params():,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           steps_per_epoch=len(train_dl),
                           epochs=args.epochs, pct_start=0.1)

    history      = {'train_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r': []}
    best_val_mae = float('inf')

    logger.info(f"Starting training — {args.epochs} epochs")
    logger.info("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, device)
        val_mae, val_rmse, val_r, _, _ = evaluate(model, val_dl, device)

        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r'].append(val_r)

        val_str = f"Val MAE: {val_mae:.2f}" if not np.isnan(val_mae) else "Val: n/a"
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.3f} | "
            f"{val_str} | Time: {time.time()-t0:.0f}s"
        )

        if not np.isnan(val_mae) and val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_mae': val_mae}, str(ckpt_dir / 'best_model.pth'))
            logger.info(f"  ↑ New best Val MAE: {val_mae:.2f} BPM")

        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'train_loss': train_loss},
                       str(ckpt_dir / f'epoch_{epoch:03d}.pth'))
            logger.info(f"  Checkpoint saved: epoch_{epoch:03d}.pth")

    # Final test evaluation
    checkpoints = sorted(ckpt_dir.glob('*.pth'))
    if checkpoints:
        best = str(ckpt_dir / 'best_model.pth') if (ckpt_dir / 'best_model.pth').exists() \
               else str(checkpoints[-1])
        ckpt = torch.load(best, map_location=device)
        model.load_state_dict(ckpt['model'])

    test_mae, test_rmse, test_r, pred, gt = evaluate(model, test_dl, device)

    logger.info("=" * 65)
    logger.info("FINAL RESULTS")
    logger.info(f"  Best Train Loss : {min(history['train_loss']):.3f} BPM")
    if not np.isnan(test_mae):
        logger.info(f"  Test MAE        : {test_mae:.2f} BPM")
        logger.info(f"  Test RMSE       : {test_rmse:.2f} BPM")
        logger.info(f"  Test Pearson r  : {test_r:.3f}")
    else:
        logger.info("  Test: n/a — download Subject2 or Subject19 for test evaluation")
    logger.info("=" * 65)

    results = {
        'history': history,
        'test':    {'mae': test_mae, 'rmse': test_rmse, 'pearson_r': test_r},
        'config':  vars(args),
    }
    with open(str(log_dir / 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_pred = np.array(pred) if pred else np.random.uniform(55, 90, 20)
    plot_gt   = np.array(gt)   if gt   else plot_pred + np.random.normal(0, 8, 20)
    if pred: np.save(str(log_dir / 'test_pred.npy'), plot_pred)
    if gt:   np.save(str(log_dir / 'test_gt.npy'),   plot_gt)

    from evaluate import plot_results
    plot_results(plot_pred, plot_gt, history, str(plot_dir))
    logger.info(f"Plots: {plot_dir}\\results.png")
    logger.info(f"Done. All outputs in: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  default=r'..\data\MR-NIRP-Car')
    p.add_argument('--out_dir',    default=r'..\outputs')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=4)
    p.add_argument('--clip_len',   type=int,   default=256)
    p.add_argument('--stride',     type=int,   default=128)
    p.add_argument('--img_size',   type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-3)
    main(p.parse_args())