"""
architecture_diagram.py
========================
Generates architecture_diagram.png inside outputs/

Run from the src/ folder:
    python architecture_diagram.py

Output saved to: ../outputs/architecture_diagram.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
fig.patch.set_facecolor('#0f1923')
ax.set_facecolor('#0f1923')

def box(ax, x, y, w, h, label, sublabel='', color='#1a3a5c', tc='white', fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='#4a9fd5', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2 + (0.12 if sublabel else 0), label,
            ha='center', va='center', color=tc, fontsize=fontsize, fontweight='bold')
    if sublabel:
        ax.text(x+w/2, y+h/2 - 0.18, sublabel, ha='center', va='center',
                color='#aabbcc', fontsize=7.5)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#4a9fd5', lw=1.5))

ax.text(7, 9.5, 'PhysNet 3D-CNN — NIR rPPG Heart Rate Estimation',
        ha='center', va='center', color='white', fontsize=13, fontweight='bold')

stages = [
    (0.3, 8.1, 2.2, 0.8, 'NIR Frames',    '640x640 16-bit PGM', '#1a4a2a'),
    (0.3, 6.8, 2.2, 0.8, 'Face ROI',      '64x64 float32 [0,1]','#1a4a2a'),
    (0.3, 5.5, 2.2, 0.8, 'Clip Assembly', 'T=128 frames',        '#1a4a2a'),
]
for s in stages:
    box(ax, *s)
arrow(ax, 1.4, 8.1, 1.4, 7.6)
arrow(ax, 1.4, 6.8, 1.4, 6.3)

blocks = [
    (3.2, 7.0, 2.0, 0.7, 'Input',     '(B, 1, 128, 64, 64)',     '#2a2a4a'),
    (3.2, 5.9, 2.0, 0.7, 'Block 1',   'Conv3D(1->32, 1x5x5)',    '#1a3a5c'),
    (3.2, 4.8, 2.0, 0.7, 'Block 2',   'Conv3D(32->64, 3x3x3)',   '#1a3a5c'),
    (3.2, 3.7, 2.0, 0.7, 'Block 3',   'Conv3D(64->128, 3x3x3)',  '#1a3a5c'),
    (3.2, 2.6, 2.0, 0.7, 'Block 4',   'Conv3D(128->128, 3x3x3)', '#1a3a5c'),
    (3.2, 1.5, 2.0, 0.7, 'AdaptPool', 'AdaptiveAvgPool3D(16,1,1)','#2a3a5c'),
    (3.2, 0.4, 2.0, 0.7, 'Head',      'Linear->256->64->1 (BPM)', '#4a2a2a'),
]
for b in blocks:
    box(ax, *b)
for i in range(len(blocks)-1):
    arrow(ax, 4.2, blocks[i][1], 4.2, blocks[i+1][1]+0.7)

arrow(ax, 2.5, 5.9, 3.2, 5.9)

gt_stages = [
    (7.5, 7.0, 2.5, 0.7, 'pulseOx.mat',    'pulseOxRecord + pulseOxTime', '#3a2a1a'),
    (7.5, 5.9, 2.5, 0.7, 'Bandpass Filter', '[0.75-2.5 Hz] Butterworth',  '#3a2a1a'),
    (7.5, 4.8, 2.5, 0.7, 'FFT + Peak',      'Hann window -> dominant freq','#3a2a1a'),
    (7.5, 3.7, 2.5, 0.7, 'GT BPM Label',    '45-150 BPM valid range',      '#3a2a1a'),
]
for g in gt_stages:
    box(ax, *g)
for i in range(len(gt_stages)-1):
    arrow(ax, 8.75, gt_stages[i][1], 8.75, gt_stages[i+1][1]+0.7)

box(ax, 11.0, 5.9, 2.5, 0.7, 'L1 Loss',         '|pred_BPM - gt_BPM|',  '#4a1a1a')
box(ax, 11.0, 4.8, 2.5, 0.7, 'Adam + OneCycleLR','lr=1e-3, wd=1e-4',     '#2a4a2a')
box(ax, 11.0, 3.7, 2.5, 0.7, 'Grad Clip',        'max_norm=1.0',          '#2a4a2a')
box(ax, 11.0, 2.6, 2.5, 0.7, 'Metrics',          'MAE | RMSE | Pearson r','#2a2a4a')

arrow(ax, 12.25, 5.9, 12.25, 5.5)
arrow(ax, 12.25, 4.8, 12.25, 4.4)
arrow(ax, 12.25, 3.7, 12.25, 3.3)
arrow(ax, 10.0, 3.7, 11.0, 4.35)
arrow(ax, 5.2, 0.75, 11.0, 5.55)

ax.text(1.4,  9.0,  'Preprocessing',   ha='center', color='#aaffaa', fontsize=9)
ax.text(4.2,  7.85, 'PhysNet 3D-CNN',  ha='center', color='#aaaaff', fontsize=9)
ax.text(8.75, 7.85, 'GT Preprocessing',ha='center', color='#ffaaaa', fontsize=9)
ax.text(12.25,6.75, 'Training',         ha='center', color='#ffddaa', fontsize=9)

# Save to outputs/ folder (one level up from src/)
out_dir = Path(__file__).parent.parent / 'outputs'
out_dir.mkdir(exist_ok=True)
out_path = out_dir / 'architecture_diagram.png'

plt.tight_layout()
plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#0f1923')
plt.close()
print(f"Saved to: {out_path}")
