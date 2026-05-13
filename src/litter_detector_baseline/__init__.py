"""Public litter detector baseline.

Public API:

- ``Detection`` — typed result of a single bounding-box detection
- ``LitterDetector`` — protocol any backend must satisfy
- ``OnnxLitterDetector`` — ONNX Runtime backend for YOLO-family models
- ``load_detector_from_config`` — config-driven factory

Designed to be embeddable in larger systems (the benchmark harness in
``litter-benchmark-harness`` consumes the ``LitterDetector`` protocol).
"""

from litter_detector_baseline.types import Detection, LitterDetector
from litter_detector_baseline.config import DetectorConfig, load_detector_from_config
from litter_detector_baseline.onnx_backend import OnnxLitterDetector

__all__ = [
    "Detection",
    "LitterDetector",
    "DetectorConfig",
    "OnnxLitterDetector",
    "load_detector_from_config",
]
