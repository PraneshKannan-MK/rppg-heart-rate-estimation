"""
evaluate.py — Evaluation, Metrics, and Visualization

Metrics (as required by assignment):
  - MAE   : Mean Absolute Error (primary)
  - RMSE  : Root Mean Square Error
  - SD    : Standard Deviation of errors
  - r     : Pearson Correlation Coefficient

Generates:
  - Bland-Altman plot
  - Scatter plot (predicted vs ground truth)
  - Loss curves
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute all required evaluation metrics.
    
    Returns dict with: MAE, RMSE, SD, Pearson_r
    """
    errors = pred - gt
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    sd     = float(np.std(errors))
    r, p   = pearsonr(pred, gt)
    
    return {
        'MAE':       round(mae, 3),
        'RMSE':      round(rmse, 3),
        'SD':        round(sd, 3),
        'Pearson_r': round(float(r), 4),
        'p_value':   round(float(p), 6),
        'N':         len(pred),
        'mean_error': round(float(np.mean(errors)), 3),
        'pred_mean': round(float(pred.mean()), 2),
        'gt_mean':   round(float(gt.mean()), 2),
    }


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  N samples    : {metrics['N']}")
    print(f"  MAE          : {metrics['MAE']:.2f} BPM")
    print(f"  RMSE         : {metrics['RMSE']:.2f} BPM")
    print(f"  SD (errors)  : {metrics['SD']:.2f} BPM")
    print(f"  Pearson r    : {metrics['Pearson_r']:.4f}")
    print(f"  Mean BPM GT  : {metrics['gt_mean']:.1f}")
    print(f"  Mean BPM Pred: {metrics['pred_mean']:.1f}")
    print(f"{'='*50}\n")


def plot_results(pred: np.ndarray, gt: np.ndarray,
                 history: dict, save_dir: str):
    """
    Generate 4-panel results figure:
      1. Scatter: predicted vs ground truth
      2. Bland-Altman (agreement) plot
      3. Training loss curve
      4. Val MAE curve
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = compute_metrics(pred, gt)
    errors  = pred - gt
    means   = (pred + gt) / 2
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('MR-NIRP rPPG Heart Rate Estimation — Results',
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    # ── 1. Scatter plot ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(gt, pred, alpha=0.6, s=20, color='steelblue', label='Predictions')
    lims = [min(gt.min(), pred.min()) - 5, max(gt.max(), pred.max()) + 5]
    ax1.plot(lims, lims, 'r--', lw=2, label='Perfect agreement')
    ax1.set_xlabel('Ground Truth BPM')
    ax1.set_ylabel('Predicted BPM')
    ax1.set_title(f'Predicted vs Ground Truth\n'
                  f'MAE={metrics["MAE"]:.2f}, r={metrics["Pearson_r"]:.3f}')
    ax1.legend(fontsize=9)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.grid(alpha=0.3)
    
    # ── 2. Bland-Altman ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    mean_err = np.mean(errors)
    loa_p    = mean_err + 1.96 * np.std(errors)
    loa_m    = mean_err - 1.96 * np.std(errors)
    
    ax2.scatter(means, errors, alpha=0.6, s=20, color='darkorange')
    ax2.axhline(mean_err, color='red',   lw=2, linestyle='-',  label=f'Bias={mean_err:.2f}')
    ax2.axhline(loa_p,    color='green', lw=1.5, linestyle='--', label=f'+1.96SD={loa_p:.2f}')
    ax2.axhline(loa_m,    color='green', lw=1.5, linestyle='--', label=f'-1.96SD={loa_m:.2f}')
    ax2.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax2.set_xlabel('Mean of GT and Predicted BPM')
    ax2.set_ylabel('Difference (Pred - GT) BPM')
    ax2.set_title(f'Bland-Altman Plot\nRMSE={metrics["RMSE"]:.2f}, SD={metrics["SD"]:.2f}')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    # ── 3. Training loss ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if history and 'train_loss' in history and len(history['train_loss']) > 0:
        epochs = range(1, len(history['train_loss']) + 1)
        ax3.plot(epochs, history['train_loss'], 'b-o', ms=3, label='Train L1 Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('L1 Loss (BPM)')
        ax3.set_title('Training Loss Curve')
        ax3.legend()
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No training history available',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Loss Curve')
    
    # ── 4. Validation MAE ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if history and 'val_mae' in history and len(history['val_mae']) > 0:
        epochs = range(1, len(history['val_mae']) + 1)
        ax4.plot(epochs, history['val_mae'],  'r-o', ms=3, label='Val MAE')
        if 'val_rmse' in history:
            ax4.plot(epochs, history['val_rmse'], 'g-s', ms=3, label='Val RMSE')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('BPM')
        ax4.set_title('Validation Metrics per Epoch')
        ax4.legend()
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No validation history available',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Validation Metrics')
    
    plt.savefig(str(save_dir / 'results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to {save_dir / 'results.png'}")


def evaluate_from_files(pred_path: str, gt_path: str,
                        history_path: str = None, save_dir: str = './outputs/plots'):
    """
    Load saved numpy arrays and produce full evaluation + plots.
    """
    pred = np.load(pred_path)
    gt   = np.load(gt_path)
    
    history = {}
    if history_path and Path(history_path).exists():
        with open(history_path) as f:
            data = json.load(f)
        history = data.get('history', {})
    
    metrics = compute_metrics(pred, gt)
    print_metrics(metrics, "Final Test Evaluation")
    plot_results(pred, gt, history, save_dir)
    
    return metrics


if __name__ == "__main__":
    # Demo: generate synthetic results for illustration
    import sys
    
    np.random.seed(42)
    n = 100
    gt   = np.random.uniform(55, 100, n)
    pred = gt + np.random.normal(0, 8, n)  # simulate ~8 BPM MAE
    
    metrics = compute_metrics(pred, gt)
    print_metrics(metrics, "Demo Evaluation (Synthetic Data)")
    
    # Fake history
    history = {
        'train_loss': [abs(50 - i*0.8) + np.random.rand()*3 for i in range(30)],
        'val_mae':    [abs(25 - i*0.5) + np.random.rand()*2 for i in range(30)],
        'val_rmse':   [abs(30 - i*0.6) + np.random.rand()*2 for i in range(30)],
    }
    
    plot_results(pred, gt, history, '/home/claude/rppg_project/outputs/plots')
    print("✓ evaluate.py OK")
