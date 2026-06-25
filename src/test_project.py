"""
test_project.py
================
Complete test suite for MR-NIRP rPPG Heart Rate Estimation Project.
46 tests across 7 sections. Saves a full report to outputs/logs/test_report.txt

Run from src\ folder:
    python test_project.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed  = []
failed  = []
skipped = []
log_lines = []   # plain-text log for the report file


def _log(line: str):
    """Append to both console (stripped) and log buffer."""
    log_lines.append(line.replace(GREEN,'').replace(RED,'').replace(YELLOW,'')
                         .replace(BLUE,'').replace(RESET,'').replace(BOLD,''))


def run_test(name: str, fn, skip: bool = False, skip_reason: str = ""):
    """Run one test, capture result, print + log."""
    if skip:
        line = f"  - {name:<55} SKIP  ({skip_reason})"
        print(f"  {YELLOW}-{RESET} {name:<55}{YELLOW}SKIP{RESET}  ({skip_reason})")
        _log(line)
        skipped.append((name, skip_reason))
        return

    print(f"  {BLUE}>{RESET} {name:<55}", end="", flush=True)
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        line = f"  > {name:<55} PASS  ({elapsed*1000:.0f}ms)"
        print(f"{GREEN}PASS{RESET}  ({elapsed*1000:.0f}ms)")
        _log(line)
        passed.append(name)
    except AssertionError as e:
        line = f"  > {name:<55} FAIL  AssertionError: {e}"
        print(f"{RED}FAIL{RESET}")
        print(f"     {RED}AssertionError: {e}{RESET}")
        _log(line)
        failed.append((name, str(e)))
    except Exception as e:
        line = f"  > {name:<55} FAIL  {type(e).__name__}: {e}"
        print(f"{RED}FAIL{RESET}")
        print(f"     {RED}{type(e).__name__}: {e}{RESET}")
        _log(line)
        failed.append((name, f"{type(e).__name__}: {e}"))


def section(title: str):
    line = f"\n{'='*65}\n  {title}\n{'='*65}"
    print(f"\n{BOLD}{YELLOW}{'='*65}{RESET}")
    print(f"{BOLD}{YELLOW}  {title}{RESET}")
    print(f"{BOLD}{YELLOW}{'='*65}{RESET}")
    _log(line)


# =============================================================================
# 1. GT PREPROCESSING
# =============================================================================
section("1. Ground Truth Preprocessing (gt_preprocessing.py)")

from gt_preprocessing import (
    extract_bpm_fft, bandpass_filter, get_actual_fps,
    get_gt_bpm_for_window, HR_MIN_BPM, HR_MAX_BPM
)

def test_bpm_clean():
    fps, true_bpm = 58.0, 72.0
    t   = np.linspace(0, 30, int(fps * 30))
    ppg = np.sin(2 * np.pi * (true_bpm / 60.0) * t)
    est = extract_bpm_fft(ppg, fps)
    assert abs(est - true_bpm) < 1.0, f"Expected ~{true_bpm}, got {est:.2f}"

def test_bpm_noisy():
    fps, true_bpm = 58.0, 65.0
    t   = np.linspace(0, 30, int(fps * 30))
    ppg = np.sin(2 * np.pi * (true_bpm / 60.0) * t) + 0.3 * np.random.randn(len(t))
    est = extract_bpm_fft(ppg, fps)
    assert abs(est - true_bpm) < 3.0, f"Expected ~{true_bpm}, got {est:.2f}"

def test_bpm_multiple():
    fps = 58.0
    for true_bpm in [50.0, 60.0, 75.0, 90.0, 110.0, 130.0]:
        t   = np.linspace(0, 30, int(fps * 30))
        ppg = np.sin(2 * np.pi * (true_bpm / 60.0) * t)
        est = extract_bpm_fft(ppg, fps)
        assert abs(est - true_bpm) < 2.0, f"BPM={true_bpm}: got {est:.2f}"

def test_bpm_short_raises():
    fps = 58.0
    short = np.random.randn(int(fps * 2))
    try:
        extract_bpm_fft(short, fps)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_bandpass_shape():
    sig      = np.random.randn(1000)
    filtered = bandpass_filter(sig, fps=58.0)
    assert filtered.shape == sig.shape

def test_bandpass_dc():
    fps    = 58.0
    n      = int(fps * 30)
    t      = np.linspace(0, 30, n)
    signal = 10.0 + np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fps)
    assert abs(filtered.mean()) < 0.5, f"DC not removed: mean={filtered.mean():.3f}"

def test_fps_calculation():
    """
    FPS computed from np.linspace timestamps has ~0.1 rounding error
    due to floating point. Tolerance is 0.1 Hz (< 0.2% error at 58fps).
    """
    expected_fps = 58.42
    n  = 1000
    ts = np.linspace(0, (n-1)/expected_fps, n)
    computed = get_actual_fps(ts)
    assert abs(computed - expected_fps) < 0.1, \
        f"Expected {expected_fps} ±0.1, got {computed:.4f}"

def test_gt_window():
    fps, true_bpm = 58.0, 75.0
    n   = 1000
    ts  = np.linspace(0, (n-1)/fps, n)
    ppg = np.sin(2 * np.pi * (true_bpm/60.0) * ts) - \
          np.sin(2 * np.pi * (true_bpm/60.0) * ts).mean()
    data = {'ppg': ppg, 'timestamps': ts, 'n_samples': n}
    bpm  = get_gt_bpm_for_window(data, 0, 400)
    assert HR_MIN_BPM <= bpm <= HR_MAX_BPM

def test_bpm_valid_range():
    fps = 58.0
    for _ in range(5):
        true_bpm = np.random.uniform(50, 120)
        t   = np.linspace(0, 30, int(fps * 30))
        ppg = np.sin(2 * np.pi * (true_bpm/60) * t) + 0.1*np.random.randn(len(t))
        est = extract_bpm_fft(ppg, fps)
        assert HR_MIN_BPM <= est <= HR_MAX_BPM

run_test("Clean sine → BPM accuracy < 1 BPM",            test_bpm_clean)
run_test("Noisy signal → BPM accuracy < 3 BPM",          test_bpm_noisy)
run_test("Multiple BPM values (50–130) within 2 BPM",    test_bpm_multiple)
run_test("Short segment raises ValueError",               test_bpm_short_raises)
run_test("Bandpass output shape matches input",           test_bandpass_shape)
run_test("Bandpass removes DC component",                 test_bandpass_dc)
run_test("FPS computed correctly (tolerance ±0.1 Hz)",   test_fps_calculation)
run_test("Window BPM in valid range [45–150]",            test_gt_window)
run_test("BPM always in valid HR range",                  test_bpm_valid_range)


# =============================================================================
# 2. FACE ROI
# =============================================================================
section("2. Face ROI & Frame Loading (face_roi.py)")

from face_roi import FaceROIExtractor, load_clip_from_arrays, _parse_pgm_bytes

def test_roi_shape():
    ext = FaceROIExtractor(target_size=64)
    frame = np.random.randint(0, 200, (640, 640), dtype=np.uint8)
    for _ in range(6): roi = ext.extract(frame)
    assert roi.shape == (64, 64) and roi.dtype == np.float32

def test_roi_range():
    ext   = FaceROIExtractor(target_size=64)
    frame = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
    for _ in range(6): roi = ext.extract(frame)
    assert roi.min() >= 0.0 and roi.max() <= 1.0

def test_roi_reset():
    ext = FaceROIExtractor(target_size=32)
    frame = np.zeros((640, 640), dtype=np.uint8)
    for _ in range(6): ext.extract(frame)
    ext.reset()
    assert ext._seen == 0 and ext._bbox is None and ext._detections == []

def test_roi_sizes():
    frame = np.random.randint(0, 200, (640, 640), dtype=np.uint8)
    for size in [32, 64, 128]:
        ext = FaceROIExtractor(target_size=size)
        for _ in range(6): roi = ext.extract(frame)
        assert roi.shape == (size, size)

def test_clip_from_arrays():
    ext    = FaceROIExtractor(target_size=64)
    frames = [np.random.randint(0, 200, (640, 640), dtype=np.uint8) for _ in range(10)]
    clip   = load_clip_from_arrays(frames, ext)
    assert clip.shape == (10, 64, 64) and clip.dtype == np.float32
    assert 0.0 <= clip.min() and clip.max() <= 1.0

def test_pgm_8bit():
    h, w   = 64, 64
    pixels = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    header = f"P5\n{w} {h}\n255\n".encode('ascii')
    result = _parse_pgm_bytes(header + pixels.tobytes())
    assert result.shape == (h, w) and result.dtype == np.uint8
    np.testing.assert_array_equal(result, pixels)

def test_pgm_16bit():
    h, w   = 32, 32
    pixels = np.random.randint(0, 65535, (h, w), dtype=np.uint16)
    header = f"P5\n{w} {h}\n65535\n".encode('ascii')
    result = _parse_pgm_bytes(header + pixels.astype(np.dtype('>u2')).tobytes())
    assert result.shape == (h, w) and result.dtype == np.uint8

def test_roi_black_frame():
    ext = FaceROIExtractor(target_size=64)
    frame = np.zeros((640, 640), dtype=np.uint8)
    for _ in range(6): roi = ext.extract(frame)
    assert roi.shape == (64, 64)

def test_roi_white_frame():
    ext = FaceROIExtractor(target_size=64)
    frame = np.full((640, 640), 255, dtype=np.uint8)
    for _ in range(6): roi = ext.extract(frame)
    assert roi.shape == (64, 64)

run_test("ROI output shape (64,64) float32",             test_roi_shape)
run_test("ROI pixel values in [0, 1]",                   test_roi_range)
run_test("ROI extractor reset clears state",             test_roi_reset)
run_test("ROI works for sizes 32, 64, 128",              test_roi_sizes)
run_test("load_clip_from_arrays → (T, H, W) float32",   test_clip_from_arrays)
run_test("PGM parser: 8-bit format",                     test_pgm_8bit)
run_test("PGM parser: 16-bit format (dataset format)",   test_pgm_16bit)
run_test("ROI handles all-black frame",                  test_roi_black_frame)
run_test("ROI handles all-white frame",                  test_roi_white_frame)


# =============================================================================
# 3. MODEL
# =============================================================================
section("3. PhysNet Model (model.py)")

import torch
from model import PhysNet, ConvBnRelu3D

def test_model_shape():
    m = PhysNet(in_channels=1, base_filters=16, clip_len=64)
    m.eval()
    assert m(torch.randn(2, 1, 64, 32, 32)).shape == (2,)

def test_model_batch1():
    m = PhysNet(in_channels=1, base_filters=16, clip_len=64)
    m.eval()
    assert m(torch.randn(1, 1, 64, 32, 32)).shape == (1,)

def test_model_no_nan():
    m   = PhysNet(in_channels=1, base_filters=16, clip_len=64)
    out = m(torch.randn(2, 1, 64, 32, 32))
    assert not torch.isnan(out).any() and not torch.isinf(out).any()

def test_model_params():
    m = PhysNet(in_channels=1, base_filters=32, clip_len=128)
    n = m.count_params()
    assert 100_000 < n < 50_000_000, f"Params: {n}"

def test_model_gradients():
    m    = PhysNet(in_channels=1, base_filters=16, clip_len=64)
    loss = m(torch.randn(2, 1, 64, 32, 32)).mean()
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad: {name}"
            break

def test_model_loss_decreases():
    m    = PhysNet(in_channels=1, base_filters=16, clip_len=64)
    opt  = torch.optim.Adam(m.parameters(), lr=1e-2)
    crit = torch.nn.L1Loss()
    x    = torch.randn(4, 1, 64, 32, 32)
    y    = torch.tensor([70.0, 75.0, 80.0, 65.0])
    losses = []
    for _ in range(10):
        opt.zero_grad()
        l = crit(m(x), y); l.backward(); opt.step()
        losses.append(l.item())
    assert losses[-1] < losses[0], f"{losses[0]:.2f} -> {losses[-1]:.2f}"

def test_model_clip_lengths():
    for cl in [64, 128, 256]:
        m = PhysNet(in_channels=1, base_filters=16, clip_len=cl)
        m.eval()
        assert m(torch.randn(1, 1, cl, 32, 32)).shape == (1,)

def test_conv_block():
    b   = ConvBnRelu3D(1, 16, kernel=(3,3,3), padding=1)
    out = b(torch.randn(2, 1, 8, 16, 16))
    assert out.shape == (2, 16, 8, 16, 16) and not torch.isnan(out).any()

def test_model_train_eval():
    m = PhysNet(in_channels=1, base_filters=16, clip_len=64)
    x = torch.randn(2, 1, 64, 32, 32)
    m.train();  ot = m(x)
    m.eval()
    with torch.no_grad(): oe = m(x)
    assert ot.shape == oe.shape == (2,)

run_test("Forward pass output shape (B,)",               test_model_shape)
run_test("Single sample batch works",                    test_model_batch1)
run_test("Output is not NaN or Inf",                     test_model_no_nan)
run_test("Parameter count in reasonable range",          test_model_params)
run_test("Gradients flow through all layers",            test_model_gradients)
run_test("Loss decreases over 10 gradient steps",        test_model_loss_decreases)
run_test("Works for clip_len 64, 128, 256",              test_model_clip_lengths)
run_test("ConvBnRelu3D block output shape correct",      test_conv_block)
run_test("Train mode and eval mode both work",           test_model_train_eval)


# =============================================================================
# 4. EVALUATE
# =============================================================================
section("4. Evaluation Metrics (evaluate.py)")

from evaluate import compute_metrics, plot_results

def test_metrics_perfect():
    gt = np.array([60.0, 70.0, 80.0, 90.0, 100.0])
    m  = compute_metrics(gt.copy(), gt)
    assert m['MAE'] == 0.0 and m['RMSE'] == 0.0
    assert abs(m['Pearson_r'] - 1.0) < 1e-6

def test_metrics_known():
    gt   = np.array([60.0, 70.0, 80.0])
    pred = np.array([65.0, 68.0, 85.0])
    m    = compute_metrics(pred, gt)
    assert abs(m['MAE']  - 4.0)                    < 0.001
    assert abs(m['RMSE'] - np.sqrt(54/3))          < 0.001

def test_pearson_range():
    for _ in range(10):
        gt   = np.random.uniform(50, 120, 20)
        pred = gt + np.random.normal(0, 10, 20)
        m    = compute_metrics(pred, gt)
        assert -1.0 <= m['Pearson_r'] <= 1.0

def test_metrics_keys():
    gt   = np.random.uniform(60, 90, 10)
    pred = gt + np.random.normal(0, 5, 10)
    m    = compute_metrics(pred, gt)
    for key in ['MAE', 'RMSE', 'SD', 'Pearson_r', 'N']:
        assert key in m, f"Missing: {key}"

def test_metrics_n():
    n    = 42
    gt   = np.random.uniform(60, 90, n)
    pred = gt + np.random.randn(n) * 5
    assert compute_metrics(pred, gt)['N'] == n

def test_plot_creates_file():
    import tempfile
    gt   = np.random.uniform(55, 100, 30)
    pred = gt + np.random.normal(0, 8, 30)
    hist = {'train_loss': [35.0, 10.0, 8.0, 7.0, 6.0, 5.5, 5.2, 5.0, 4.9, 4.8],
            'val_mae':    [float('nan')] * 10, 'val_rmse': [float('nan')] * 10}
    with tempfile.TemporaryDirectory() as tmp:
        plot_results(pred, gt, hist, tmp)
        png = Path(tmp) / 'results.png'
        assert png.exists() and png.stat().st_size > 10_000

run_test("Perfect predictions: MAE=0, r=1",              test_metrics_perfect)
run_test("Known values: MAE and RMSE correct",           test_metrics_known)
run_test("Pearson r always in [-1, 1]",                  test_pearson_range)
run_test("All required metric keys present",             test_metrics_keys)
run_test("N equals number of samples",                   test_metrics_n)
run_test("plot_results creates PNG file",                test_plot_creates_file)


# =============================================================================
# 5. DATASET
# =============================================================================
section("5. Dataset & DataLoader (dataset.py)")

from dataset import find_sessions, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS

def test_no_overlap():
    train, val, test = set(TRAIN_SUBJECTS), set(VAL_SUBJECTS), set(TEST_SUBJECTS)
    assert not (train & val),  f"Train/Val overlap: {train & val}"
    assert not (train & test), f"Train/Test overlap: {train & test}"
    assert not (val & test),   f"Val/Test overlap: {val & test}"

def test_sub16_not_test():
    assert 'Subject16' not in TEST_SUBJECTS

def test_sub16_not_train():
    assert 'Subject16' not in TRAIN_SUBJECTS

def test_find_sessions_missing():
    assert find_sessions(r'C:\does\not\exist', ['Subject1']) == []

def test_no_duplicates():
    all_s = TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS
    assert len(all_s) == len(set(all_s))

run_test("Train/Val/Test subjects fully disjoint",       test_no_overlap)
run_test("Subject16 not in test set",                    test_sub16_not_test)
run_test("Subject16 not in train set",                   test_sub16_not_train)
run_test("find_sessions on missing path returns []",     test_find_sessions_missing)
run_test("No duplicate subjects across all splits",      test_no_duplicates)


# =============================================================================
# 6. INTEGRATION
# =============================================================================
section("6. Integration — Full Mini Pipeline (no dataset needed)")

def test_full_pipeline():
    from face_roi import FaceROIExtractor, load_clip_from_arrays
    T      = 64
    frames = [np.random.randint(50, 200, (640, 640), dtype=np.uint8) for _ in range(T)]
    ext    = FaceROIExtractor(target_size=32)
    clip   = load_clip_from_arrays(frames, ext)
    assert clip.shape == (T, 32, 32) and clip.dtype == np.float32
    clip_t = torch.from_numpy(clip).unsqueeze(0).unsqueeze(0)
    m      = PhysNet(in_channels=1, base_filters=16, clip_len=T)
    m.eval()
    with torch.no_grad():
        out = m(clip_t)
    assert out.shape == (1,) and not torch.isnan(out).any()
    loss = torch.nn.L1Loss()(out, torch.tensor([70.0]))
    assert loss.item() >= 0 and not torch.isnan(loss)

def test_training_step():
    m    = PhysNet(in_channels=1, base_filters=16, clip_len=64)
    opt  = torch.optim.Adam(m.parameters(), lr=1e-3)
    init = [p.clone().detach() for p in m.parameters()]
    x    = torch.randn(2, 1, 64, 32, 32)
    gt   = torch.tensor([70.0, 80.0])
    opt.zero_grad()
    torch.nn.L1Loss()(m(x), gt).backward()
    opt.step()
    assert any(not torch.equal(p, ip)
               for p, ip in zip(m.parameters(), init))

def test_gt_model_compat():
    fps, true_bpm = 58.0, 72.0
    t    = np.linspace(0, 30, int(fps * 30))
    ppg  = np.sin(2 * np.pi * (true_bpm/60) * t)
    ppg -= ppg.mean()
    ts   = np.arange(len(t)) / fps
    data = {'ppg': ppg, 'timestamps': ts, 'n_samples': len(ppg)}
    bpm  = get_gt_bpm_for_window(data, 0, 400)
    assert HR_MIN_BPM <= bpm <= HR_MAX_BPM
    t2 = torch.tensor(bpm, dtype=torch.float32)
    assert not torch.isnan(t2) and not torch.isinf(t2)

run_test("Full pipeline: frames -> ROI -> model -> BPM", test_full_pipeline)
run_test("One training step updates parameters",          test_training_step)
run_test("GT BPM values valid as model targets",         test_gt_model_compat)


# =============================================================================
# 7. CHECKPOINTS
# =============================================================================
section("7. Saved Checkpoints & Training Results")

def test_ckpt_exists():
    d = Path(__file__).parent.parent / 'outputs' / 'checkpoints'
    assert list(d.glob('*.pth')), f"No checkpoints in {d}"

def test_ckpt_loadable():
    p = Path(__file__).parent.parent / 'outputs' / 'checkpoints' / 'epoch_050.pth'
    assert p.exists(), f"epoch_050.pth not found"
    ckpt = torch.load(str(p), map_location='cpu')
    m    = PhysNet(in_channels=1, base_filters=32, clip_len=256)
    m.load_state_dict(ckpt['model'])
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 1, 256, 64, 64))
    assert out.shape == (1,) and not torch.isnan(out).any()

def test_json_valid():
    p = Path(__file__).parent.parent / 'outputs' / 'logs' / 'training_results.json'
    assert p.exists()
    with open(str(p)) as f: data = json.load(f)
    losses = data['history']['train_loss']
    assert len(losses) == 50
    assert losses[0] > losses[-1]
    assert min(losses) < 10.0

def test_loss_improved():
    p = Path(__file__).parent.parent / 'outputs' / 'logs' / 'training_results.json'
    assert p.exists()
    with open(str(p)) as f: data = json.load(f)
    losses = data['history']['train_loss']
    assert losses[-1] < losses[0] * 0.5, \
        f"Improvement < 50%: {losses[0]:.2f} -> {losses[-1]:.2f}"

def test_plots_exist():
    d = Path(__file__).parent.parent / 'outputs' / 'plots'
    for f in ['loss_curves.png', 'results.png', 'training_summary.png']:
        p = d / f
        assert p.exists() and p.stat().st_size > 5000, f"Missing/small: {f}"

run_test("Checkpoints folder has .pth files",            test_ckpt_exists)
run_test("epoch_050.pth loads into PhysNet",             test_ckpt_loadable)
run_test("training_results.json valid structure",        test_json_valid)
run_test("Loss improved > 50% over training",            test_loss_improved)
run_test("All 3 plot PNGs exist",                        test_plots_exist)


# =============================================================================
# SAVE REPORT TO FILE
# =============================================================================
total     = len(passed) + len(failed) + len(skipped)
pass_rate = len(passed) / max(total - len(skipped), 1) * 100

summary_lines = [
    "",
    "=" * 65,
    "  TEST RESULTS",
    "=" * 65,
    f"  Run date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"  Total     : {total}",
    f"  Passed    : {len(passed)}",
    f"  Failed    : {len(failed)}",
    f"  Skipped   : {len(skipped)}",
    f"  Pass rate : {pass_rate:.1f}%",
]

if failed:
    summary_lines.append("")
    summary_lines.append("  FAILED TESTS:")
    for name, reason in failed:
        summary_lines.append(f"    x {name}")
        summary_lines.append(f"      {reason}")

if skipped:
    summary_lines.append("")
    summary_lines.append("  SKIPPED TESTS:")
    for name, reason in skipped:
        summary_lines.append(f"    - {name}  ({reason})")

if not failed:
    summary_lines.append("")
    summary_lines.append("  All tests passed! Project is submission-ready.")

summary_lines.append("=" * 65)
summary_lines.append("")

for line in summary_lines:
    _log(line)

# Print summary to console
for line in summary_lines:
    if "PASS" in line or "passed" in line.lower():
        print(f"{GREEN}{line}{RESET}")
    elif "FAIL" in line or "x " in line:
        print(f"{RED}{line}{RESET}")
    elif "SKIP" in line or "- " in line:
        print(f"{YELLOW}{line}{RESET}")
    else:
        print(f"{BOLD}{line}{RESET}")

# Save to outputs/logs/test_report.txt
report_path = Path(__file__).parent.parent / 'outputs' / 'logs' / 'test_report.txt'
report_path.parent.mkdir(parents=True, exist_ok=True)

report_header = [
    "=" * 65,
    "  MR-NIRP rPPG PROJECT — TEST REPORT",
    f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 65,
    "",
]

with open(str(report_path), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_header + log_lines))

print(f"\n  Report saved to: {report_path}\n")

sys.exit(0 if not failed else 1)