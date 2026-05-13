"""CLI entry: ``python -m litter_detector_baseline <command>``.

Commands:

- ``infer`` — run inference on one image or a directory of images, print
  detections as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from litter_detector_baseline.config import load_detector_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="litter-detector-baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_infer = sub.add_parser("infer", help="Run inference on image(s).")
    p_infer.add_argument("--config", required=True, type=Path, help="Detector YAML config.")
    p_infer.add_argument("--image", required=True, type=Path, help="Image file or directory.")
    p_infer.add_argument("--score-threshold", type=float, default=None)
    p_infer.add_argument("--iou-threshold", type=float, default=None)
    p_infer.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results. Defaults to stdout.",
    )

    args = parser.parse_args(argv)

    if args.command == "infer":
        return _run_infer(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_infer(args: argparse.Namespace) -> int:
    from litter_detector_baseline.io import load_image_rgb

    detector = load_detector_from_config(args.config)

    image_paths: list[Path]
    if args.image.is_dir():
        image_paths = sorted(
            p for p in args.image.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        image_paths = [args.image]

    results = []
    for path in image_paths:
        image = load_image_rgb(path)
        detections = detector.predict(
            image,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
        results.append({
            "image": str(path),
            "image_size": list(image.shape[:2]),
            "detections": [d.as_dict() for d in detections],
        })

    output_json = json.dumps(results, indent=2)
    if args.output is None:
        print(output_json)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        print(f"Wrote {len(results)} result(s) to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
