"""Public types for the litter detector baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Detection:
    """One bounding-box detection.

    Box coordinates are in pixel space of the input image (not the model
    input size). The format is xyxy (top-left x, top-left y, bottom-right
    x, bottom-right y), matching COCO conventions.
    """

    class_id: int
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "bbox_xyxy": [self.x1, self.y1, self.x2, self.y2],
        }


class LitterDetector(Protocol):
    """Protocol any detector backend must implement.

    Backends should accept HWC RGB uint8 numpy arrays as input and return
    a list of ``Detection``. The exact resize / normalization is the
    backend's concern.
    """

    def predict(
        self,
        image,  # numpy.ndarray, HWC RGB uint8 — left untyped to avoid forcing numpy at protocol import time
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Sequence[Detection]:
        """Run inference on a single image."""
        ...

    @property
    def class_names(self) -> Sequence[str]:
        """Ordered list of class names; index = class_id."""
        ...

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input (height, width)."""
        ...
