"""Image preprocessing for YOLO-family detectors."""

from __future__ import annotations

import numpy as np


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_value: int = 114,
) -> np.ndarray:
    """Resize an HWC RGB uint8 image to ``target_size`` preserving aspect ratio.

    Pads with ``pad_value`` (default 114, matching ultralytics convention).
    Returns an HWC RGB uint8 image of shape (target_h, target_w, 3).
    """
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # cv2 is imported lazily so the package is importable without it
    # (e.g. for type-only use in the benchmark harness).
    import cv2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


def to_chw_float(image: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 [0, 255] to CHW float32 [0, 1]."""
    image = image.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))
