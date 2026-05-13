"""Config-driven detector loading.

Config schema (YAML):

    backend: onnx
    weights: path/to/model.onnx
    class_names:
      - bottle
      - can
      - ...
    input_size: [640, 640]
    providers:
      - CPUExecutionProvider
    score_threshold: 0.25
    iou_threshold: 0.45

Only the ``backend``, ``weights``, and ``class_names`` keys are required.
Other keys default sensibly. Future backends (e.g. tflite) plug in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml  # PyYAML — listed as a dependency


@dataclass(frozen=True)
class DetectorConfig:
    backend: str
    weights: Path
    class_names: tuple[str, ...]
    input_size: tuple[int, int] = (640, 640)
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    score_threshold: float = 0.25
    iou_threshold: float = 0.45

    @classmethod
    def from_yaml(cls, path: Path) -> "DetectorConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data, base_dir=Path(path).parent)

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path | None = None) -> "DetectorConfig":
        try:
            backend = str(data["backend"]).lower()
            weights = Path(data["weights"])
            class_names = tuple(data["class_names"])
        except KeyError as e:
            raise ValueError(f"DetectorConfig missing required key: {e}") from e

        if not weights.is_absolute() and base_dir is not None:
            weights = (base_dir / weights).resolve()

        input_size = data.get("input_size", [640, 640])
        if isinstance(input_size, list):
            input_size = tuple(input_size)

        providers = tuple(data.get("providers", ["CPUExecutionProvider"]))
        score_threshold = float(data.get("score_threshold", 0.25))
        iou_threshold = float(data.get("iou_threshold", 0.45))

        return cls(
            backend=backend,
            weights=weights,
            class_names=class_names,
            input_size=input_size,
            providers=providers,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )


def load_detector_from_config(path: Path):
    """Load a detector instance from a YAML config file.

    Returns a concrete ``LitterDetector`` implementation matching the
    config's ``backend`` field.
    """
    config = DetectorConfig.from_yaml(path)
    if config.backend == "onnx":
        from litter_detector_baseline.onnx_backend import OnnxLitterDetector
        return OnnxLitterDetector.from_config(config)
    raise ValueError(f"Unsupported backend: {config.backend!r}")
