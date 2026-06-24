"""
face_roi.py
============
Face Detection & ROI Extraction for NIR Frames

Handles:
  - Direct .pgm file paths (NIR\ folder, already extracted)
  - Raw bytes (when called from zip reader in gt_preprocessing.py)

NIR frame format (verified on real data):
  - PGM P5, 640x640, 16-bit (maxval=65535)
  - Normalise to uint8 for detection and model input
  - Cache face bbox from first 5 frames, reuse across entire clip
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_HAAR = None

def _get_haar():
    global _HAAR
    if _HAAR is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _HAAR = cv2.CascadeClassifier(cascade_path)
    return _HAAR


def load_pgm_frame(path: str) -> np.ndarray:
    """
    Load 16-bit PGM (P5, maxval=65535) from disk -> uint8 array.
    Works on Windows paths.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frame not found: {path}")

    with open(str(path), 'rb') as f:
        data = f.read()
    return _parse_pgm_bytes(data)


def _parse_pgm_bytes(data: bytes) -> np.ndarray:
    """Parse raw PGM bytes -> uint8 (H, W) array."""
    lines = []
    i = 0
    while len(lines) < 3:
        end = data.index(b'\n', i)
        line = data[i:end].decode('ascii').strip()
        i = end + 1
        if not line.startswith('#'):
            lines.append(line)

    magic, dims, maxval_str = lines
    w, h     = map(int, dims.split())
    maxval   = int(maxval_str)
    pixel_data = data[i:]

    if maxval > 255:
        arr = np.frombuffer(pixel_data, dtype=np.dtype('>u2'))
        img = arr.reshape(h, w).astype(np.float32)
        img = (img / maxval * 255).astype(np.uint8)
    else:
        arr = np.frombuffer(pixel_data, dtype=np.uint8)
        img = arr.reshape(h, w)

    return img


def detect_face_bbox(frame_u8: np.ndarray):
    """
    Detect face using Haar cascade.
    Returns (x, y, w, h) or None.
    """
    faces = _get_haar().detectMultiScale(
        frame_u8, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
    )
    if len(faces) == 0:
        return None
    areas = [fw * fh for _, _, fw, fh in faces]
    return tuple(int(v) for v in faces[int(np.argmax(areas))])


def _centre_crop(h, w, fraction=0.6):
    ch = int(h * fraction)
    cw = int(w * fraction)
    return ((w - cw) // 2, (h - ch) // 2, cw, ch)


class FaceROIExtractor:
    """
    Detects face in first N frames, caches the bbox, applies it to all frames.
    Call reset() between clips.
    """

    def __init__(self, target_size: int = 64, detect_frames: int = 5, padding: float = 0.15):
        self.target_size   = target_size
        self.detect_frames = detect_frames
        self.padding       = padding
        self._bbox         = None
        self._seen         = 0
        self._detections   = []

    def reset(self):
        self._bbox       = None
        self._seen       = 0
        self._detections = []

    def _finalize_bbox(self, h, w):
        if self._detections:
            areas = [bw * bh for _, _, bw, bh in self._detections]
            x, y, bw, bh = self._detections[int(np.argmax(areas))]
        else:
            x, y, bw, bh = _centre_crop(h, w)
            logger.debug("Face not detected — using centre crop fallback")

        px = int(bw * self.padding)
        py = int(bh * self.padding)
        x  = max(0, x - px)
        y  = max(0, y - py)
        bw = min(w - x, bw + 2 * px)
        bh = min(h - y, bh + 2 * py)
        return (x, y, bw, bh)

    def extract(self, frame_u8: np.ndarray) -> np.ndarray:
        """
        Extract face ROI from uint8 frame.
        Returns float32 (target_size, target_size) in [0, 1].
        """
        h, w = frame_u8.shape[:2]

        if self._seen < self.detect_frames:
            bbox = detect_face_bbox(frame_u8)
            if bbox:
                self._detections.append(bbox)
            self._seen += 1
            if self._seen == self.detect_frames:
                self._bbox = self._finalize_bbox(h, w)
            roi = cv2.resize(frame_u8, (self.target_size, self.target_size))
        else:
            x, y, bw, bh = self._bbox
            crop = frame_u8[y:y+bh, x:x+bw]
            roi  = cv2.resize(crop if crop.size > 0 else frame_u8,
                              (self.target_size, self.target_size))

        return roi.astype(np.float32) / 255.0


def load_clip_frames(frame_paths: list, extractor: FaceROIExtractor) -> np.ndarray:
    """
    Load list of .pgm file paths -> (T, H, W) float32 array.
    """
    extractor.reset()
    frames = []
    for fp in frame_paths:
        try:
            img = load_pgm_frame(str(fp))
        except Exception as e:
            logger.warning(f"Failed to load {fp}: {e} — using zeros")
            img = np.zeros((640, 640), dtype=np.uint8)
        frames.append(extractor.extract(img))
    return np.stack(frames, axis=0)


def load_clip_from_arrays(frame_arrays: list, extractor: FaceROIExtractor) -> np.ndarray:
    """
    Load from list of uint8 numpy arrays (used when reading from zip).
    """
    extractor.reset()
    frames = []
    for img in frame_arrays:
        if img is None:
            img = np.zeros((640, 640), dtype=np.uint8)
        frames.append(extractor.extract(img))
    return np.stack(frames, axis=0)


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    # Look for any .pgm file in the dataset
    search_roots = [
        Path(r'F:\UST\project_2\rppg_project\data\MR-NIRP-Car'),
        Path(__file__).parent.parent / 'data' / 'MR-NIRP-Car',
    ]

    pgm_file = None
    for root in search_roots:
        if not root.exists(): continue
        hits = list(root.rglob('Frame00000.pgm'))
        if hits:
            pgm_file = hits[0]
            break

    if pgm_file is None:
        print("No .pgm frames found yet — running synthetic test")
        # Synthetic: just test ROI extraction on a fake frame
        fake = (np.random.rand(640, 640) * 200).astype(np.uint8)
        ext  = FaceROIExtractor(target_size=64)
        for _ in range(6): roi = ext.extract(fake)
        assert roi.shape == (64, 64)
        assert roi.dtype == np.float32
        print(f"Synthetic ROI: shape={roi.shape}, range=[{roi.min():.3f},{roi.max():.3f}]")
        print("face_roi.py self-test PASSED (synthetic)")
        sys.exit(0)

    print(f"Testing with: {pgm_file}")
    img = load_pgm_frame(str(pgm_file))
    print(f"Frame: shape={img.shape}, dtype={img.dtype}, range=[{img.min()},{img.max()}]")

    ext = FaceROIExtractor(target_size=64)
    for _ in range(6):
        roi = ext.extract(img)
    assert roi.shape == (64, 64)
    assert roi.dtype == np.float32
    print(f"ROI: shape={roi.shape}, range=[{roi.min():.3f},{roi.max():.3f}]")
    print("face_roi.py self-test PASSED (real data)")