# litter-detector-baseline

Public baseline detector package for litter detection. Designed to be a
reusable building block: the package exposes a small, typed API that
larger systems (training pipelines, benchmark harnesses, robot
inference loops) can compose against without taking a hard dependency
on a specific model architecture.

## Scope

- Configurable detector frontend with a clean `LitterDetector` protocol
- ONNX Runtime backend for deployment-friendly CPU inference (Raspberry Pi 5 baseline)
- YOLO-family post-processing (anchor-free YOLOv8 output decode, per-class NMS, letterbox-aware box scaling)
- CLI for one-off inference (`python -m litter_detector_baseline infer ...`)
- Export helper for converting ultralytics YOLO `.pt` checkpoints to ONNX

## Install

```bash
pip install -e .          # runtime + CLI
pip install -e ".[dev]"   # + pytest
pip install -e ".[export]"  # + ultralytics for the export helper
```

`opencv-python-headless` is the chosen OpenCV variant because the
deployment target is headless (Raspberry Pi OS Lite, no X server).

## Quick start

Get a baseline ONNX model. Easiest path on a development laptop:

```bash
pip install ".[export]"
python export/export_yolov8.py --weights yolov8n.pt --imgsz 640
# produces yolov8n.onnx in the current directory
```

Then run inference using the supplied config:

```bash
mv yolov8n.onnx configs/
python -m litter_detector_baseline infer \
  --config configs/yolov8n_coco_baseline.yaml \
  --image path/to/image.jpg
```

Output is a JSON list of `{image, image_size, detections[]}` records.

## Programmatic use

```python
from litter_detector_baseline import load_detector_from_config
from litter_detector_baseline.io import load_image_rgb

detector = load_detector_from_config("configs/yolov8n_coco_baseline.yaml")
image = load_image_rgb("path/to/image.jpg")
for det in detector.predict(image):
    print(det.class_name, det.score, det.x1, det.y1, det.x2, det.y2)
```

## Class taxonomy

The package is taxonomy-agnostic — `class_names` is supplied per
config. For deployment on a CrustBot-style litter robot, plug in a
14-class edge taxonomy via a project-specific config. For an
out-of-the-box COCO baseline (useful for sanity checks before custom
data is collected), use `configs/yolov8n_coco_baseline.yaml`.

## Hardware target

Designed for Raspberry Pi 5 CPU inference via ONNX Runtime. Memory
budget is tight on the 1GB Pi 5 variant — prefer int8-quantized models.
A static-quantization recipe lives in the export helper roadmap.

## Carried Over

This repo is seeded with a DregsBane workspace source note:

- `docs/source_distillation_strategy_for_trash_models.md`
