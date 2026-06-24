"""
diagnose.py — Run this FIRST to find the exact problem
Run from src\: python diagnose.py
"""
import sys, zipfile, io, numpy as np, scipy.io, scipy.signal
from pathlib import Path

# ── Find dataset root ─────────────────────────────────────────────────────────
candidates = [
    Path(r'F:\UST\project_2\rppg_project\data\MR-NIRP-Car'),
    Path(__file__).parent.parent / 'data' / 'MR-NIRP-Car',
]
root = next((c for c in candidates if c.exists()), None)
if not root:
    print("ERROR: Dataset root not found"); sys.exit(1)

print(f"Dataset root: {root}")

# ── Find first session with NIR.zip ──────────────────────────────────────────
session_dir = None
for subj in sorted(root.iterdir()):
    for cond in sorted(subj.iterdir()):
        if 'large_motion' in cond.name: continue
        if (cond / 'NIR.zip').exists() and (cond / 'PulseOx.zip').exists():
            session_dir = cond
            break
    if session_dir: break

if not session_dir:
    print("ERROR: No session with NIR.zip + PulseOx.zip found"); sys.exit(1)

print(f"Testing session: {session_dir.name}")
nir_zip = session_dir / 'NIR.zip'
pox_zip = session_dir / 'PulseOx.zip'

# ── Inspect NIR.zip contents ──────────────────────────────────────────────────
print("\n--- NIR.zip contents (first 10 entries) ---")
with zipfile.ZipFile(str(nir_zip), 'r') as zf:
    all_names = zf.namelist()
    pgm_names = sorted([n for n in all_names if n.lower().endswith('.pgm')])
    print(f"Total entries in zip : {len(all_names)}")
    print(f"PGM files found      : {len(pgm_names)}")
    print(f"First 5 PGM names    : {pgm_names[:5]}")
    print(f"Last  5 PGM names    : {pgm_names[-5:]}")
    
    # Try to read frame 0
    if pgm_names:
        print(f"\nReading first frame: {pgm_names[0]}")
        raw = zf.read(pgm_names[0])
        print(f"Raw bytes length: {len(raw)}")
        header_lines = raw[:50].decode('ascii', errors='replace')
        print(f"Header preview: {repr(header_lines)}")

# ── Load PulseOx ─────────────────────────────────────────────────────────────
print("\n--- PulseOx.zip ---")
with zipfile.ZipFile(str(pox_zip), 'r') as zf:
    mat_names = [n for n in zf.namelist() if n.lower().endswith('.mat')]
    print(f"MAT files in zip: {mat_names}")
    if mat_names:
        mat_bytes = zf.read(mat_names[0])
        data = scipy.io.loadmat(io.BytesIO(mat_bytes))
        ppg = data['pulseOxRecord'].flatten().astype(np.float64)
        ts  = data['pulseOxTime'].flatten().astype(np.float64)
        n   = len(ppg)
        fps = n / (ts[-1] - ts[0])
        print(f"PPG samples: {n}")
        print(f"FPS        : {fps:.2f}")
        ppg -= ppg.mean()
        
        # Test BPM extraction on a 128-frame window
        clip_len = 128
        if n > clip_len:
            seg = ppg[0:clip_len]
            seg_ts = ts[0:clip_len]
            seg_fps = len(seg_ts) / (seg_ts[-1] - seg_ts[0])
            
            # Bandpass
            nyq = seg_fps / 2.0
            lo  = 0.75 / nyq
            hi  = min(2.5 / nyq, 0.99)
            print(f"\nWindow BPM test:")
            print(f"  Window frames : {clip_len}")
            print(f"  Window FPS    : {seg_fps:.2f}")
            print(f"  Window duration: {clip_len/seg_fps:.1f}s")
            print(f"  Bandpass lo={lo:.3f} hi={hi:.3f}")
            
            if lo >= hi:
                print("  ERROR: Bandpass range invalid! lo >= hi")
            elif clip_len < seg_fps * 3:
                print(f"  ERROR: Window too short! {clip_len} frames < {seg_fps*3:.0f} minimum (3 seconds)")
                print(f"  FIX: Use --clip_len {int(seg_fps * 4)} or larger")
            else:
                b, a = scipy.signal.butter(3, [lo, hi], btype='bandpass')
                filtered = scipy.signal.filtfilt(b, a, scipy.signal.detrend(seg))
                windowed = filtered * np.hanning(len(filtered))
                fft_mag  = np.abs(np.fft.rfft(windowed))
                freqs    = np.fft.rfftfreq(len(windowed), d=1.0/seg_fps)
                valid    = (freqs >= 0.75) & (freqs <= 2.5)
                if np.any(valid):
                    bpm = freqs[valid][np.argmax(fft_mag[valid])] * 60
                    print(f"  BPM extracted : {bpm:.1f}")
                    print(f"  BPM valid     : {45 <= bpm <= 150}")
                else:
                    print("  ERROR: No valid HR frequencies in FFT")

print("\n--- DIAGNOSIS COMPLETE ---")
print("Share the output above to identify the exact fix needed.")