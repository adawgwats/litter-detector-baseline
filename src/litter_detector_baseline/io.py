"""Image I/O helpers — lazy cv2 import keeps the package importable
in headless / CI contexts that haven't installed opencv yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image from disk as HWC RGB uint8.

    Uses OpenCV under the hood (BGR-native) and converts to RGB.
    """
    import cv2

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to decode image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(image: np.ndarray, path: Path) -> None:
    """Save an HWC RGB uint8 image to disk (BGR-encoded for cv2)."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
