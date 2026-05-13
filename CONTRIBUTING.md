# Contributing to litter-detector-baseline

Status: design review draft, 2026-05-13. Detailed contribution flows will
be finalized after the v0.1 architecture is approved and code lands.

## Project goals at a glance

This is a **public, hardware-portable, taxonomy-agnostic** detector
library. We accept contributions that expand the supported set of:

- Inference backends (ONNX Runtime, TFLite, TensorRT, Hailo, Coral, RKNN, etc.)
- Detector architectures (YOLOv8 today; YOLOv5, RT-DETR, EfficientDet welcomed)
- Postprocessing modes (detection today; segmentation as a future extension)
- Configs for application domains (litter today; pest, debris, marine, agricultural welcome)

We do **not** accept changes that:

- Hardcode a specific class taxonomy in code (taxonomies belong in configs)
- Add a hard dependency on a heavyweight framework (PyTorch, TensorFlow) to the main runtime path
- Couple to a specific hosted service for inference or registry

## Adding a new backend

A backend is one Python module under `src/litter_detector_baseline/backends/`
that implements the `LitterDetector` protocol for a specific inference
framework.

### Template

```python
# src/litter_detector_baseline/backends/my_backend.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from litter_detector_baseline.config import DetectorConfig
from litter_detector_baseline.preprocess import letterbox, to_chw_float
from litter_detector_baseline.registry import register_backend, get_postprocessor
from litter_detector_baseline.types import Detection


@register_backend("my-framework")
class MyFrameworkDetector:
    """Detector backed by [framework name]."""

    def __init__(
        self,
        weights: Path,
        class_names: Sequence[str],
        architecture: str,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **framework_specific_kwargs,
    ) -> None:
        # Import the framework lazily so the package stays importable
        # without it.
        try:
            import my_framework
        except ImportError as e:
            raise ImportError(
                "my-framework is required for MyFrameworkDetector. "
                "Install with: pip install '.[my-framework]'"
            ) from e

        self._class_names = tuple(class_names)
        self._input_size = tuple(input_size)
        self._score_threshold = float(score_threshold)
        self._iou_threshold = float(iou_threshold)
        self._postprocessor = get_postprocessor(architecture)
        # ... framework-specific session initialization

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "MyFrameworkDetector":
        return cls(
            weights=config.weights,
            class_names=config.class_names,
            architecture=config.architecture,
            input_size=config.input_size,
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
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        original_size = image.shape[:2]
        letterboxed = letterbox(image, self._input_size)
        tensor = to_chw_float(letterboxed)[np.newaxis, ...]

        # ... framework-specific inference; produce raw output tensor

        return self._postprocessor(
            output=raw_output,
            class_names=self._class_names,
            score_threshold=score_threshold or self._score_threshold,
            iou_threshold=iou_threshold or self._iou_threshold,
            input_size=self._input_size,
            original_size=original_size,
        )
```

### Required tests

Add `tests/test_my_backend.py`:

1. **Import smoke test**: backend imports without crashing, even when its underlying framework isn't installed (graceful ImportError).
2. **Construction smoke test**: backend instantiates from a config when the framework IS installed.
3. **Inference smoke test**: backend produces a valid `List[Detection]` on a synthetic input.
4. **Protocol conformance**: backend satisfies `LitterDetector` (verify via runtime check or static type check).

CI skips backend-specific tests when the framework isn't installed in
the job's environment.

### Required pyproject.toml updates

Add an optional extra:

```toml
[project.optional-dependencies]
my-framework = [
    "my-framework-runtime>=1.0",
]
```

### Required imports update

In `src/litter_detector_baseline/__init__.py`, import the backend module
to trigger registration:

```python
from litter_detector_baseline.backends import my_backend  # noqa: F401
```

### PR checklist

- [ ] Backend file added under `backends/`
- [ ] `@register_backend("name")` decorator applied
- [ ] Lazy framework import with clear ImportError
- [ ] `from_config` classmethod accepts `DetectorConfig`
- [ ] `predict` accepts HWC RGB uint8 numpy input
- [ ] Tests added under `tests/`
- [ ] Optional extra added to `pyproject.toml`
- [ ] Import added to `__init__.py` to trigger registration
- [ ] README updated to list the new backend as supported
- [ ] Sample config under `configs/` demonstrating the backend

## Adding a new postprocessor (detector architecture)

A postprocessor is one function that decodes raw model output to
`List[Detection]`. Add one when bringing a model architecture that
doesn't match an existing decoder (e.g. RT-DETR's output differs from
YOLOv8's).

### Template

```python
# src/litter_detector_baseline/postprocessors/my_arch.py

from typing import Sequence
import numpy as np
from litter_detector_baseline.registry import register_postprocessor
from litter_detector_baseline.types import Detection


@register_postprocessor("my-arch")
def decode_my_arch(
    output: np.ndarray,
    class_names: Sequence[str],
    score_threshold: float,
    iou_threshold: float,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list[Detection]:
    """Decode raw [my-arch] output to a list of Detection."""
    # 1. Parse the output tensor's specific layout
    # 2. Filter by score_threshold
    # 3. Convert boxes to xyxy in original-image pixel space (undo letterbox)
    # 4. Run per-class NMS
    # 5. Return sorted by score descending
    ...
```

### PR checklist

- [ ] Postprocessor file under `postprocessors/`
- [ ] `@register_postprocessor("name")` decorator
- [ ] Boxes returned in original-image pixel space (xyxy)
- [ ] Per-class NMS (not global)
- [ ] Sorted by score descending
- [ ] Tests under `tests/` covering edge cases (empty output, single detection, NMS overlap)
- [ ] Import added to `__init__.py` to trigger registration

## Adding a new application config

If you're using this library for an application domain other than
litter (pest detection, agricultural debris, marine cleanup, etc.),
contribute a sample config:

```yaml
# configs/agricultural_pests_v1.yaml
backend: onnxruntime
architecture: yolov8
weights: weights/pests_v1.onnx
input_size: [640, 640]
score_threshold: 0.30
iou_threshold: 0.50
class_names:
  - aphid
  - whitefly
  - cabbage_worm
  - ...
```

These configs document that the library is application-portable. They
don't have to ship the weights — a `weights_url` comment is fine.

## Filing issues

Two issue types:

- **Backend gap**: "I have hardware X and the library doesn't have a backend for it." Tag `backend-request`.
- **Architecture gap**: "I have a model trained on architecture Y and the library can't decode its output." Tag `architecture-request`.

For both, ideally attach a small reproducible example.

## License

Apache 2.0 (see `LICENSE`). Contributions are accepted under the same.
