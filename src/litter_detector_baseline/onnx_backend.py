"""ONNX Runtime backend for YOLO-family detectors.

Targets ONNX Runtime CPU as the deployment baseline (Raspberry Pi 5).
Falls back gracefully when the package is imported without
``onnxruntime`` installed (the type stays importable; instantiation
fails clearly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from litter_detector_baseline.config import DetectorConfig
from litter_detector_baseline.postprocess import yolov8_decode
from litter_detector_baseline.preprocess import letterbox, to_chw_float
from litter_detector_baseline.types import Detection


class OnnxLitterDetector:
    """YOLO-family detector running on ONNX Runtime.

    Compatible with ultralytics-exported YOLOv8 ONNX (and shape-compatible
    successors). Input must be HWC RGB uint8; the wrapper handles
    letterbox + scale + transpose internally.
    """

    def __init__(
        self,
        weights: Path,
        class_names: Sequence[str],
        input_size: tuple[int, int] = (640, 640),
        providers: Sequence[str] = ("CPUExecutionProvider",),
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for OnnxLitterDetector. "
                "Install with: pip install onnxruntime"
            ) from e

        self._weights = Path(weights)
        self._class_names = tuple(class_names)
        self._input_size = tuple(input_size)
        self._score_threshold = float(score_threshold)
        self._iou_threshold = float(iou_threshold)

        if not self._weights.exists():
            raise FileNotFoundError(f"ONNX weights not found: {self._weights}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self._weights),
            sess_options=session_options,
            providers=list(providers),
        )
        self._input_name = self._session.get_inputs()[0].name

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "OnnxLitterDetector":
        return cls(
            weights=config.weights,
            class_names=config.class_names,
            input_size=config.input_size,
            providers=config.providers,
            score_threshold=config.score_threshold,
            iou_threshold=config.iou_threshold,
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def predict(
        self,
        image: np.ndarray,
        score_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[Detection]:
        """Run inference on a single HWC RGB uint8 image."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        score_thr = self._score_threshold if score_threshold is None else float(score_threshold)
        iou_thr = self._iou_threshold if iou_threshold is None else float(iou_threshold)

        original_size = image.shape[:2]  # (h, w)
        letterboxed = letterbox(image, self._input_size)
        tensor = to_chw_float(letterboxed)[np.newaxis, ...]  # NCHW

        outputs = self._session.run(None, {self._input_name: tensor})
        output = outputs[0]

        return yolov8_decode(
            output=output,
            class_names=self._class_names,
            score_threshold=score_thr,
            iou_threshold=iou_thr,
            input_size=self._input_size,
            original_size=original_size,
        )

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
        score_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[list[Detection]]:
        """Convenience wrapper for multiple independent images."""
        return [self.predict(img, score_threshold, iou_threshold) for img in images]
