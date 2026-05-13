"""Export a YOLOv8 PyTorch checkpoint to ONNX.

Usage:

    python export/export_yolov8.py --weights yolov8n.pt --imgsz 640

Run on the LAPTOP, not the Pi — pulls in ultralytics + torch which
are heavy. The resulting ``yolov8n.onnx`` is what the Pi-side
``OnnxLitterDetector`` consumes.

Skipped on Pi-side ``pip install`` by keeping the import lazy and
gating with a ``__main__`` guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export YOLOv8 .pt to ONNX.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolov8n.pt"),
        help="Path to ultralytics .pt weights (default: yolov8n.pt, auto-downloads).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic axes.")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify graph.")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for the ONNX output (default: current directory).",
    )
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is required to export YOLOv8. Install with:\n"
            "    pip install ultralytics",
            file=sys.stderr,
        )
        return 1

    model = YOLO(str(args.weights))
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )
    src = Path(export_path)
    dst = args.output_dir / src.name
    if src != dst:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    print(f"Exported ONNX: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
