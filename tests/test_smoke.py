"""Smoke tests — verify public API surface is importable and pure-Python
helpers behave correctly. Backend tests that require ONNX Runtime live
elsewhere and run in CI with the heavy deps installed.
"""

from __future__ import annotations

import numpy as np

from litter_detector_baseline import Detection, DetectorConfig
from litter_detector_baseline.postprocess import yolov8_decode


def test_import_smoke() -> None:
    """Top-level imports work."""
    from litter_detector_baseline import (  # noqa: F401
        Detection,
        LitterDetector,
        OnnxLitterDetector,
        DetectorConfig,
        load_detector_from_config,
    )


def test_detection_dataclass_basics() -> None:
    det = Detection(
        class_id=39,
        class_name="bottle",
        score=0.82,
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=50.0,
    )
    assert det.width == 20.0
    assert det.height == 30.0
    assert det.area == 600.0
    serialized = det.as_dict()
    assert serialized["class_name"] == "bottle"
    assert serialized["bbox_xyxy"] == [10.0, 20.0, 30.0, 50.0]


def test_detection_zero_area_box() -> None:
    """Degenerate boxes don't crash the area accessor."""
    det = Detection(
        class_id=0,
        class_name="x",
        score=0.5,
        x1=0.0,
        y1=0.0,
        x2=-5.0,
        y2=-5.0,
    )
    assert det.area == 0.0


def test_detector_config_from_dict_minimum() -> None:
    cfg = DetectorConfig.from_dict({
        "backend": "onnx",
        "weights": "model.onnx",
        "class_names": ["a", "b", "c"],
    })
    assert cfg.backend == "onnx"
    assert cfg.class_names == ("a", "b", "c")
    assert cfg.input_size == (640, 640)
    assert cfg.score_threshold == 0.25


def test_detector_config_from_dict_missing_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="missing required key"):
        DetectorConfig.from_dict({"backend": "onnx"})


def test_yolov8_decode_below_threshold_returns_empty() -> None:
    """When no anchor scores above threshold, decode returns []."""
    # 2 classes, 5 anchors, all scores zero
    output = np.zeros((1, 4 + 2, 5), dtype=np.float32)
    detections = yolov8_decode(
        output=output,
        class_names=["bottle", "can"],
        score_threshold=0.25,
        iou_threshold=0.45,
        input_size=(640, 640),
        original_size=(640, 640),
    )
    assert detections == []


def test_yolov8_decode_single_high_confidence() -> None:
    """One anchor with score=1 → one detection."""
    output = np.zeros((1, 4 + 2, 3), dtype=np.float32)
    # Anchor 0: cx=320, cy=320, w=100, h=100, class=bottle (idx 0) score=0.9
    output[0, :4, 0] = [320, 320, 100, 100]
    output[0, 4, 0] = 0.9

    detections = yolov8_decode(
        output=output,
        class_names=["bottle", "can"],
        score_threshold=0.25,
        iou_threshold=0.45,
        input_size=(640, 640),
        original_size=(640, 640),
    )
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "bottle"
    assert det.score == 0.9
    # Box should be 270, 270, 370, 370
    assert det.x1 == 270.0
    assert det.y1 == 270.0
    assert det.x2 == 370.0
    assert det.y2 == 370.0


def test_yolov8_decode_nms_suppresses_duplicate() -> None:
    """Two overlapping high-score boxes of same class → NMS keeps the best."""
    output = np.zeros((1, 4 + 1, 3), dtype=np.float32)
    output[0, :4, 0] = [100, 100, 50, 50]
    output[0, 4, 0] = 0.9
    output[0, :4, 1] = [105, 105, 50, 50]  # near-duplicate
    output[0, 4, 1] = 0.7
    output[0, :4, 2] = [400, 400, 30, 30]  # distinct
    output[0, 4, 2] = 0.5

    detections = yolov8_decode(
        output=output,
        class_names=["bottle"],
        score_threshold=0.25,
        iou_threshold=0.45,
        input_size=(640, 640),
        original_size=(640, 640),
    )
    assert len(detections) == 2
    assert detections[0].score == 0.9
    assert detections[1].score == 0.5
