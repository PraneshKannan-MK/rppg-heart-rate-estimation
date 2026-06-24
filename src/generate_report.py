"""
generate_report.py
==================
Generates all submission materials from the completed training run:
  1. Loss curve plot (professional quality)
  2. Metrics summary
  3. Final report text

Run from src\:
    python generate_report.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

OUT_DIR  = src_dir.parent / 'outputs'
LOG_DIR  = OUT_DIR / 'logs'
PLOT_DIR = OUT_DIR / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load training history ─────────────────────────────────────────────────────
results_path = LOG_DIR / 'training_results.json'
if not results_path.exists():
    print(f"ERROR: {results_path} not found.")
    print("Run training first: python train.py --data_root ..\\data\\MR-NIRP-Car --epochs 50")
    sys.exit(1)

with open(str(results_path)) as f:
    results = json.load(f)

history = results['history']
config  = results.get('config', {})
train_loss = history['train_loss']
epochs     = list(range(1, len(train_loss) + 1))

print(f"Loaded training history: {len(epochs)} epochs")
print(f"Best loss: {min(train_loss):.3f} BPM at epoch {train_loss.index(min(train_loss))+1}")


# ── Plot 1: Loss curve (main deliverable) ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PhysNet 3D-CNN — NIR rPPG Heart Rate Estimation\nTraining Results',
             fontsize=13, fontweight='bold')

# Left: full loss curve
ax1 = axes[0]
ax1.plot(epochs, train_loss, 'b-', lw=2, label='Train L1 Loss (MAE)')
ax1.fill_between(epochs, train_loss,
                 [min(train_loss)] * len(epochs),
                 alpha=0.15, color='blue')

# Annotate key points
ax1.annotate(f'Epoch 1\n{train_loss[0]:.1f} BPM',
             xy=(1, train_loss[0]), xytext=(5, train_loss[0] - 3),
             arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9)

best_ep  = train_loss.index(min(train_loss)) + 1
best_val = min(train_loss)
ax1.annotate(f'Best: Epoch {best_ep}\n{best_val:.2f} BPM',
             xy=(best_ep, best_val), xytext=(best_ep + 3, best_val + 3),
             arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=9)

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('L1 Loss (BPM)', fontsize=11)
ax1.set_title('Training Loss Curve (50 Epochs)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim(1, len(epochs))
ax1.set_ylim(0, max(train_loss) * 1.1)

# Right: zoomed view (epoch 2 onwards — after initial drop)
ax2 = axes[1]
zoom_ep   = epochs[1:]
zoom_loss = train_loss[1:]
ax2.plot(zoom_ep, zoom_loss, 'b-o', lw=1.5, ms=3, label='Train Loss')

# Rolling average
window = 5
if len(zoom_loss) >= window:
    roll = np.convolve(zoom_loss, np.ones(window)/window, mode='valid')
    roll_ep = zoom_ep[window-1:]
    ax2.plot(roll_ep, roll, 'r-', lw=2.5, label=f'{window}-epoch rolling avg')

ax2.axhline(min(zoom_loss), color='green', lw=1, linestyle='--',
            label=f'Best: {min(zoom_loss):.2f} BPM')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('L1 Loss (BPM)', fontsize=11)
ax2.set_title('Convergence Detail (Epochs 2-50)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xlim(2, len(epochs))

plt.tight_layout()
loss_plot_path = str(PLOT_DIR / 'loss_curves.png')
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Loss curve saved: {loss_plot_path}")


# ── Plot 2: Training summary dashboard ────────────────────────────────────────
fig2 = plt.figure(figsize=(14, 9))
fig2.suptitle('MR-NIRP rPPG Training Summary — Submission Report',
              fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

# Panel 1: Loss curve
ax_loss = fig2.add_subplot(gs[0, :2])
ax_loss.plot(epochs, train_loss, 'steelblue', lw=2)
ax_loss.fill_between(epochs, train_loss, alpha=0.15, color='steelblue')
ax_loss.set_title('Training L1 Loss (MAE) over 50 Epochs', fontsize=11)
ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss (BPM)')
ax_loss.grid(alpha=0.3)
ax_loss.annotate(f'Start: {train_loss[0]:.1f}', xy=(1, train_loss[0]),
                 xytext=(8, train_loss[0]-2), color='red', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='red'))
ax_loss.annotate(f'Best: {min(train_loss):.2f}', xy=(best_ep, best_val),
                 xytext=(best_ep+4, best_val+4), color='green', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='green'))

# Panel 2: Stats box
ax_stats = fig2.add_subplot(gs[0, 2])
ax_stats.axis('off')
stats_text = (
    "TRAINING STATISTICS\n"
    "━━━━━━━━━━━━━━━━━━\n\n"
    f"Dataset        : MR-NIRP Car\n"
    f"Subjects used  : 2 (train only)\n"
    f"Sessions       : 3\n"
    f"Total clips    : 246\n"
    f"Clip length    : 256 frames\n"
    f"Actual FPS     : ~58 fps\n"
    f"Clip duration  : ~4.4 seconds\n\n"
    f"Model          : PhysNet 3D-CNN\n"
    f"Parameters     : 1,785,697\n"
    f"Input          : (1, 256, 64, 64)\n"
    f"Loss fn        : L1 (MAE)\n"
    f"Optimizer      : Adam lr=1e-3\n"
    f"Scheduler      : OneCycleLR\n\n"
    f"Epochs         : {len(epochs)}\n"
    f"Batch size     : {config.get('batch_size', 4)}\n\n"
    f"Initial Loss   : {train_loss[0]:.2f} BPM\n"
    f"Final Loss     : {train_loss[-1]:.2f} BPM\n"
    f"Best Loss      : {min(train_loss):.2f} BPM\n"
    f"Improvement    : {((train_loss[0]-min(train_loss))/train_loss[0]*100):.1f}%"
)
ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
              fontsize=8.5, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))

# Panel 3: Epoch-by-epoch bar (first 20 epochs)
ax_bar = fig2.add_subplot(gs[1, :2])
colors = ['#2ecc71' if l <= min(train_loss) * 1.1 else
          '#3498db' if l <= 8 else '#e74c3c'
          for l in train_loss]
bars = ax_bar.bar(epochs, train_loss, color=colors, alpha=0.8, width=0.7)
ax_bar.axhline(min(train_loss), color='green', lw=1.5, linestyle='--',
               label=f'Best: {min(train_loss):.2f} BPM')
ax_bar.axhline(np.mean(train_loss[5:]), color='orange', lw=1.5, linestyle=':',
               label=f'Mean (ep 6-50): {np.mean(train_loss[5:]):.2f} BPM')
ax_bar.set_title('Per-Epoch Loss (green = near-best, blue = good, red = initial)', fontsize=10)
ax_bar.set_xlabel('Epoch'); ax_bar.set_ylabel('L1 Loss (BPM)')
ax_bar.legend(fontsize=9); ax_bar.grid(alpha=0.2, axis='y')
ax_bar.set_xlim(0.5, len(epochs) + 0.5)

# Panel 4: Pipeline overview text
ax_pipe = fig2.add_subplot(gs[1, 2])
ax_pipe.axis('off')
pipe_text = (
    "PIPELINE SUMMARY\n"
    "━━━━━━━━━━━━━━━━\n\n"
    "INPUT\n"
    "  NIR PGM frames\n"
    "  640x640, 16-bit\n\n"
    "PREPROCESSING\n"
    "  Face detect (Haar)\n"
    "  ROI: 64x64 crop\n"
    "  Normalize [0,1]\n\n"
    "GROUND TRUTH\n"
    "  pulseOx.mat -> PPG\n"
    "  Bandpass [0.75-2.5Hz]\n"
    "  FFT -> BPM label\n\n"
    "MODEL: PhysNet\n"
    "  4x Conv3D blocks\n"
    "  AdaptiveAvgPool\n"
    "  Linear -> BPM\n\n"
    "METRICS\n"
    "  MAE  (primary)\n"
    "  RMSE\n"
    "  Pearson r"
)
ax_pipe.text(0.05, 0.95, pipe_text, transform=ax_pipe.transAxes,
             fontsize=8.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f8f0', alpha=0.8))

plt.savefig(str(PLOT_DIR / 'training_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Summary dashboard saved: {PLOT_DIR / 'training_summary.png'}")


# ── Print full metrics report ─────────────────────────────────────────────────
print()
print("=" * 60)
print("  FINAL SUBMISSION REPORT")
print("=" * 60)
print(f"  Dataset       : MR-NIRP Car (NIR only)")
print(f"  Subjects      : Subject1 (2 sessions), Subject3 (1 session)")
print(f"  Total clips   : 246  |  clip_len=256  |  stride=128")
print(f"  FPS           : ~57-58 fps (verified from timestamps)")
print()
print(f"  Model         : PhysNet 3D-CNN (Yu et al., BMVC 2019)")
print(f"  Modification  : Single-channel input (NIR grayscale)")
print(f"  Parameters    : 1,785,697")
print()
print(f"  Loss function : L1 Loss (MAE) — robust to BPM outliers")
print(f"  Optimizer     : Adam (lr=1e-3, weight_decay=1e-4)")
print(f"  Scheduler     : OneCycleLR (pct_start=0.1)")
print(f"  Grad clip     : max_norm=1.0")
print()
print(f"  Epochs        : {len(epochs)}")
print(f"  Initial Loss  : {train_loss[0]:.2f} BPM")
print(f"  Final Loss    : {train_loss[-1]:.2f} BPM")
print(f"  Best Loss     : {min(train_loss):.2f} BPM (epoch {best_ep})")
print(f"  Improvement   : {((train_loss[0]-min(train_loss))/train_loss[0]*100):.1f}%")
print()
print(f"  Val MAE       : n/a (Subject14/15/17/18 not downloaded)")
print(f"  Test MAE      : n/a (Subject2/19 not downloaded)")
print()
print(f"  NOTE: The assignment states accuracy is LOW priority.")
print(f"  The training curve demonstrates correct methodology.")
print()
print(f"  Outputs saved to: {OUT_DIR}")
print(f"    plots/loss_curves.png")
print(f"    plots/training_summary.png")
print(f"    logs/training_results.json")
print("=" * 60)