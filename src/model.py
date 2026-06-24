"""
model.py — PhysNet 3D-CNN for NIR rPPG Heart Rate Estimation

Architecture: PhysNet (Yu et al., 2019) adapted for:
  - Single-channel input (NIR grayscale vs original 3-channel RGB)
  - Direct BPM regression (vs PPG signal prediction in original)

Reference:
  Yu, Z. et al. "Remote Photoplethysmograph Signal Measurement from Facial
  Videos Using Spatio-Temporal Networks." BMVC 2019.
  (https://arxiv.org/abs/1905.02419)

Modifications from original:
  1. in_channels=1 (NIR grayscale)
  2. Final head regresses to single BPM value
  3. AdaptiveAvgPool3d for variable-length temporal input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu3D(nn.Module):
    """Basic building block: Conv3D → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel, padding, stride=(1,1,1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class PhysNet(nn.Module):
    """
    PhysNet 3D-CNN for heart rate estimation.
    
    Input:  (B, 1, T, H, W) — single-channel NIR clips
    Output: (B,)            — BPM predictions
    
    Args:
        in_channels : 1 for NIR (grayscale), 3 for RGB
        base_filters: number of filters in first conv layer
        dropout     : dropout rate before final regression
        clip_len    : expected temporal length (for AdaptivePool sizing)
    """
    
    def __init__(self, in_channels: int = 1, base_filters: int = 32,
                 dropout: float = 0.3, clip_len: int = 128):
        super().__init__()
        
        f = base_filters
        
        # Block 1: Spatial feature extraction (no temporal pooling yet)
        self.block1 = nn.Sequential(
            ConvBnRelu3D(in_channels, f,   kernel=(1,5,5), padding=(0,2,2)),
            nn.MaxPool3d(kernel_size=(1,2,2))  # (B, f, T, H/2, W/2)
        )
        
        # Block 2: Begin spatiotemporal mixing
        self.block2 = nn.Sequential(
            ConvBnRelu3D(f, f*2, kernel=(3,3,3), padding=1),
            nn.MaxPool3d(kernel_size=(2,2,2))  # (B, 2f, T/2, H/4, W/4)
        )
        
        # Block 3: Deeper spatiotemporal
        self.block3 = nn.Sequential(
            ConvBnRelu3D(f*2, f*4, kernel=(3,3,3), padding=1),
            nn.MaxPool3d(kernel_size=(2,2,2))  # (B, 4f, T/4, H/8, W/8)
        )
        
        # Block 4: More depth
        self.block4 = nn.Sequential(
            ConvBnRelu3D(f*4, f*4, kernel=(3,3,3), padding=1),
            nn.MaxPool3d(kernel_size=(2,2,2))  # (B, 4f, T/8, H/16, W/16)
        )
        
        # Temporal + spatial pooling → compact feature vector
        self.pool = nn.AdaptiveAvgPool3d((clip_len // 8, 1, 1))
        
        # Regression head
        temporal_feats = f * 4 * (clip_len // 8)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(temporal_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (B, 1, T, H, W)  — single-channel NIR clip
        returns: (B,) BPM predictions
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.head(x)
        return x.squeeze(-1)  # (B,)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PhysNet(in_channels=1, base_filters=32, clip_len=128)
    print(f"Parameters: {model.count_params():,}")
    
    # Test forward pass
    dummy = torch.randn(2, 1, 128, 64, 64)
    out   = model(dummy)
    print(f"Input: {dummy.shape} → Output: {out.shape}")
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    print(f"Output BPM range: {out.min().item():.1f} – {out.max().item():.1f}")
    print("✓ model.py OK")
