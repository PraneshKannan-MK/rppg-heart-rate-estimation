"""
gt_preprocessing.py -- Ground Truth Preprocessing for MR-NIRP Car Dataset

VERIFIED FACTS from real data inspection:
  - Dataset FPS: ~57-58 fps (NOT 30fps) -- always compute from timestamps
  - clip_len must be >= fps*4 = ~234 frames minimum (4 seconds)
  - NIR frame i <-> PulseOx sample i  (1:1 timestamp mapping)
  - NIR frames: PGM P5, 640x640, 16-bit (maxval=65535)
  - Frames start at Frame00001.pgm (not Frame00000)

Extraction layout found on this machine:
  NIR.zip extracted to --> NIR/NIR/Frame00001.pgm  (double nested)
  PulseOx.zip extracted to --> PulseOx/PulseOX/    (capital X subfolder)

The code handles ALL these layouts via recursive rglob search.
"""

import numpy as np
import scipy.io
import scipy.signal
import zipfile
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HR_MIN_BPM   = 45.0
HR_MAX_BPM   = 150.0
HR_MIN_HZ    = HR_MIN_BPM / 60.0
HR_MAX_HZ    = HR_MAX_BPM / 60.0
FILTER_ORDER = 3
MIN_SECONDS  = 4.0


# --- PGM parsing --------------------------------------------------------------

def _parse_pgm_bytes(data: bytes) -> np.ndarray:
    """Parse PGM P5 bytes (16-bit or 8-bit) -> uint8 array."""
    lines, i = [], 0
    while len(lines) < 3:
        end  = data.index(b'\n', i)
        line = data[i:end].decode('ascii').strip()
        i    = end + 1
        if not line.startswith('#'):
            lines.append(line)
    magic, dims, maxval_str = lines
    w, h   = map(int, dims.split())
    maxval = int(maxval_str)
    pixel  = data[i:]
    if maxval > 255:
        arr = np.frombuffer(pixel, dtype=np.dtype('>u2'))
        img = (arr.reshape(h, w).astype(np.float32) / maxval * 255).astype(np.uint8)
    else:
        img = np.frombuffer(pixel, dtype=np.uint8).reshape(h, w)
    return img


def load_pgm_frame(path) -> np.ndarray:
    """Load 16-bit PGM from disk -> uint8."""
    with open(str(path), 'rb') as f:
        return _parse_pgm_bytes(f.read())


# --- NIR frame discovery (handles ALL extraction layouts) ---------------------

def find_pgm_files(nir_dir: Path) -> list:
    """
    Find all .pgm files under nir_dir using recursive search.

    Handles:
      nir_dir/Frame00001.pgm          (flat)
      nir_dir/NIR/Frame00001.pgm      (zip created NIR subfolder)
      nir_dir/PulseOX/Frame*.pgm      (any subfolder name)

    Returns sorted list of Path objects.
    """
    if not nir_dir.exists():
        return []

    # Recursive search -- works for any nesting depth
    all_pgm = sorted(nir_dir.rglob('Frame*.pgm'))
    if all_pgm:
        parent = all_pgm[0].parent
        rel    = parent.relative_to(nir_dir)
        if str(rel) != '.':
            logger.info(f"Frames found in subfolder: {nir_dir.name}/{rel}/  ({len(all_pgm)} files)")
        else:
            logger.info(f"Frames found flat in: {nir_dir.name}/  ({len(all_pgm)} files)")

    return all_pgm


# --- ZIP helpers --------------------------------------------------------------

def _load_mat_from_zip(zip_path: Path) -> dict:
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        mat_names = [n for n in zf.namelist() if n.lower().endswith('.mat')]
        if not mat_names:
            raise FileNotFoundError(f"No .mat inside {zip_path}")
        return scipy.io.loadmat(io.BytesIO(zf.read(mat_names[0])))

def _list_pgm_from_zip(zip_path: Path) -> list:
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        return sorted([n for n in zf.namelist() if n.lower().endswith('.pgm')])

def _load_pgm_from_zip(zip_path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        return _parse_pgm_bytes(zf.read(name))


# --- PulseOx loading ----------------------------------------------------------

def _parse_pulseox_dict(raw: dict) -> dict:
    ppg = raw['pulseOxRecord'].flatten().astype(np.float64)
    ts  = raw['pulseOxTime'].flatten().astype(np.float64)
    n   = int(raw['numPulseSample'].flatten()[0])
    ppg -= ppg.mean()
    fps  = n / (ts[-1] - ts[0])
    logger.info(f"PulseOx: {n} samples | fps={fps:.2f} | {ts[-1]-ts[0]:.1f}s")
    return {'ppg': ppg, 'timestamps': ts, 'n_samples': n}

def load_pulseox_mat(mat_path: str) -> dict:
    return _parse_pulseox_dict(scipy.io.loadmat(str(mat_path)))

def load_pulseox_from_zip(zip_path: str) -> dict:
    return _parse_pulseox_dict(_load_mat_from_zip(Path(zip_path)))

def load_pulseox_auto(condition_dir: str) -> dict:
    """
    Auto-detect PulseOx data. Handles:
      PulseOx/pulseOx.mat      (ideal)
      PulseOx/PulseOX/...mat   (capital X subfolder -- seen on this machine)
      PulseOx.zip
      pulseOx.mat              (flat)
    """
    d = Path(condition_dir)

    # Recursive search inside PulseOx folder (handles any subfolder/case)
    pox_folder = d / 'PulseOx'
    if pox_folder.exists():
        mat_files = list(pox_folder.rglob('pulseOx.mat'))
        if not mat_files:
            # Try case-insensitive
            mat_files = [f for f in pox_folder.rglob('*.mat')
                         if 'pulseox' in f.name.lower()]
        if mat_files:
            logger.debug(f"Found mat: {mat_files[0]}")
            return load_pulseox_mat(str(mat_files[0]))

    # PulseOx.zip (note: may also be PulseOX.zip with capital X)
    for zip_name in ['PulseOx.zip', 'PulseOX.zip']:
        z = d / zip_name
        if z.exists():
            return load_pulseox_from_zip(str(z))

    # Flat pulseOx.mat
    mat2 = d / 'pulseOx.mat'
    if mat2.exists():
        return load_pulseox_mat(str(mat2))

    raise FileNotFoundError(
        f"No PulseOx data found in: {d}\n"
        f"Run check_structure.py to see what is inside."
    )


# --- Signal processing --------------------------------------------------------

def get_actual_fps(timestamps: np.ndarray) -> float:
    return len(timestamps) / (timestamps[-1] - timestamps[0])

def bandpass_filter(signal: np.ndarray, fps: float) -> np.ndarray:
    nyq = fps / 2.0
    lo  = HR_MIN_HZ / nyq
    hi  = min(HR_MAX_HZ / nyq, 0.99)
    if lo >= hi:
        return signal
    b, a = scipy.signal.butter(FILTER_ORDER, [lo, hi], btype='bandpass')
    return scipy.signal.filtfilt(b, a, signal)

def extract_bpm_fft(ppg_segment: np.ndarray, fps: float) -> float:
    """Extract dominant BPM from PPG segment via FFT."""
    min_samples = fps * MIN_SECONDS
    if len(ppg_segment) < min_samples:
        raise ValueError(
            f"Segment {len(ppg_segment)} samples ({len(ppg_segment)/fps:.1f}s) "
            f"< {MIN_SECONDS}s minimum. Need clip_len >= {int(min_samples)+1}"
        )
    seg = scipy.signal.detrend(ppg_segment)
    seg = bandpass_filter(seg, fps)
    seg = seg * np.hanning(len(seg))
    fft_mag = np.abs(np.fft.rfft(seg))
    freqs   = np.fft.rfftfreq(len(seg), d=1.0/fps)
    valid   = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    if not np.any(valid):
        raise ValueError("No valid cardiac frequencies in FFT")
    return float(freqs[valid][np.argmax(fft_mag[valid])] * 60.0)

def get_gt_bpm_for_window(pulseox_data: dict, start: int, end: int) -> float:
    ppg = pulseox_data['ppg']
    ts  = pulseox_data['timestamps']
    s, e = max(0, start), min(len(ppg), end)
    fps  = get_actual_fps(ts[s:e]) if (e - s) > 1 else 50.0
    return extract_bpm_fft(ppg[s:e], fps)


# --- Session ------------------------------------------------------------------

class SessionData:
    """
    One condition folder. Finds frames recursively -- works with any
    extraction subfolder layout including NIR/NIR/ and PulseOx/PulseOX/.
    """

    def __init__(self, condition_dir: str):
        self.dir = Path(condition_dir)

        # Load PulseOx (recursive search handles capital-X subfolder)
        self.pulseox = load_pulseox_auto(str(self.dir))
        self.fps     = get_actual_fps(self.pulseox['timestamps'])

        # Discover NIR frames
        self._use_zip     = False
        self._nir_zip     = None
        self._frame_paths = []

        nir_folder = self.dir / 'NIR'
        nir_zip    = self.dir / 'NIR.zip'

        if nir_folder.exists():
            # Recursive search inside NIR/ -- finds NIR/NIR/Frame*.pgm too
            found = find_pgm_files(nir_folder)
            if found:
                self._frame_paths = found
                self._use_zip     = False
            elif nir_zip.exists():
                logger.warning(f"NIR folder empty -- falling back to NIR.zip")
                self._nir_zip     = nir_zip
                self._frame_paths = _list_pgm_from_zip(nir_zip)
                self._use_zip     = True
        elif nir_zip.exists():
            self._nir_zip     = nir_zip
            self._frame_paths = _list_pgm_from_zip(nir_zip)
            self._use_zip     = True
        else:
            raise FileNotFoundError(f"No NIR data in {self.dir}")

        if len(self._frame_paths) == 0:
            raise FileNotFoundError(
                f"No Frame*.pgm files found under {self.dir / 'NIR'}\n"
                f"Run check_structure.py to inspect the folder."
            )

        # Align with PulseOx sample count
        n_pox = self.pulseox['n_samples']
        n_fr  = len(self._frame_paths)
        if n_fr != n_pox:
            logger.warning(
                f"{self.dir.name}: frames={n_fr} vs PulseOx={n_pox}, "
                f"using min={min(n_fr, n_pox)}"
            )
        usable = min(n_fr, n_pox)
        self._frame_paths = self._frame_paths[:usable]
        self.n_frames     = usable
        self.min_clip_len = int(self.fps * MIN_SECONDS) + 1

        src = "zip" if self._use_zip else "folder"
        logger.info(
            f"Session: {self.dir.name} | {self.n_frames} frames [{src}] | "
            f"{self.fps:.2f} fps | {self.n_frames/self.fps:.1f}s | "
            f"min_clip_len={self.min_clip_len}"
        )

    def load_frame(self, idx: int) -> np.ndarray:
        name = self._frame_paths[idx]
        if self._use_zip:
            return _load_pgm_from_zip(self._nir_zip, str(name))
        else:
            return load_pgm_frame(name)

    def load_frames(self, start: int, end: int) -> list:
        return [self.load_frame(i) for i in range(start, min(end, self.n_frames))]

    def get_session_bpm(self) -> float:
        return extract_bpm_fft(self.pulseox['ppg'], self.fps)

    def get_window_bpm(self, start: int, end: int) -> float:
        return get_gt_bpm_for_window(self.pulseox, start, end)

    def get_clips(self, clip_len: int, stride: int) -> list:
        if clip_len < self.min_clip_len:
            logger.warning(
                f"{self.dir.name}: clip_len={clip_len} too short "
                f"(need >= {self.min_clip_len}). Auto-adjusting."
            )
            clip_len = self.min_clip_len
        clips = []
        for start in range(0, self.n_frames - clip_len, stride):
            end = start + clip_len
            try:
                bpm = self.get_window_bpm(start, end)
                if HR_MIN_BPM <= bpm <= HR_MAX_BPM:
                    clips.append((start, end, bpm))
            except ValueError as e:
                logger.debug(f"Clip [{start}:{end}] skipped: {e}")
        return clips


# --- Self-test ----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    candidates = [
        Path(r'F:\UST\project_2\rppg_project\data\MR-NIRP-Car'),
        Path(__file__).parent.parent / 'data' / 'MR-NIRP-Car',
    ]
    root = next((c for c in candidates if c.exists()), None)

    if root is None:
        print("No dataset -- running synthetic test")
        fps, true_bpm = 58.0, 72.0
        t   = np.linspace(0, 30, int(fps * 30))
        ppg = np.sin(2 * np.pi * (true_bpm/60) * t) + 0.3*np.random.randn(len(t))
        est = extract_bpm_fft(ppg, fps)
        print(f"True: {true_bpm} | Estimated: {est:.1f} | Error: {abs(est-true_bpm):.2f}")
        assert abs(est - true_bpm) < 3.0
        print("PASSED (synthetic)")
        sys.exit(0)

    # Real data test
    test_dir = None
    for subj in sorted(root.iterdir()):
        if not subj.is_dir(): continue
        for cond in sorted(subj.iterdir()):
            if not cond.is_dir() or 'large_motion' in cond.name: continue
            has_nir = (cond/'NIR').exists() or (cond/'NIR.zip').exists()
            has_pox = (cond/'PulseOx').exists() or (cond/'PulseOx.zip').exists()
            if has_nir and has_pox:
                test_dir = cond; break
        if test_dir: break

    if not test_dir:
        print("No sessions found."); sys.exit(0)

    print(f"Testing: {test_dir.name}")
    sess  = SessionData(str(test_dir))
    clips = sess.get_clips(clip_len=256, stride=128)
    print(f"FPS        : {sess.fps:.2f}")
    print(f"Frames     : {sess.n_frames}")
    print(f"Clips      : {len(clips)}")
    if clips:
        bpms = [b for _,_,b in clips]
        print(f"BPM range  : {min(bpms):.1f} - {max(bpms):.1f}")
        print(f"BPM mean   : {np.mean(bpms):.1f}")
    print("PASSED (real data)")