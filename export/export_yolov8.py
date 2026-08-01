"""Export a trained YOLOv8/v11 detector to ONNX + sidecar meta.json.

Produces the artifacts the trail PWA's on-device inference layer
consumes. See `dregsbane-web-trail/docs/on-device-inference-spec.md`
sections 11.1 + 12 for the producer contract.

Usage:
    python -m export.export_yolov8 \\
        --weights C:\\tmp\\runs\\v1-take6\\weights\\best.pt \\
        --data-yaml C:\\tmp\\v1_dataset\\data.yaml \\
        --output-dir ./dist/models/v1 \\
        --model-name yolo11n-litter \\
        --version v1.0.0-fp16 \\

Side effects (in --output-dir):
    <model-name>.onnx                 (the exported model)
    <model-name>.meta.json            (sidecar metadata)

Invariants enforced:
    - .onnx artifact MUST be <= 6 MB (spec §10 perf budget)
    - meta.json schema matches the consumer's expectation in spec §11.1
    - version string format: v<MAJOR>.<MINOR>.<PATCH>-<precision>

Supersedes the May-2026 v0 stub that only wrapped ultralytics.export()
without producing a meta.json or enforcing the size budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger("export.export_yolov8")

MAX_ONNX_BYTES = 6 * 1024 * 1024  # 6 MB ceiling per spec §10 perf budget
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+-(fp32|fp16|int8)$")


def _read_class_names(data_yaml: Path) -> list[str]:
    """Read the ordered class list from a YOLO data.yaml.

    YOLO emits classes as a dict keyed by integer index. We sort by
    index so the returned list is in the same order the model emits.
    """
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = raw.get("names")
    if not isinstance(names, dict):
        raise ValueError(
            f"{data_yaml} 'names' is not a dict (got {type(names).__name__}). "
            "Was this YAML produced by training/data/prepare_dataset.py?"
        )
    return [names[i] for i in sorted(names.keys())]


def _read_training_metrics(eval_receipt: Path | None) -> dict[str, Any]:
    """Pull mAP + hard-neg FPR + hazards recall out of an eval receipt.

    Returns {} if no receipt provided / readable, so meta.json still
    writes (just without trainingMetrics). Caller logs a warning in
    that case.
    """
    if eval_receipt is None or not eval_receipt.exists():
        return {}
    try:
        data = json.loads(eval_receipt.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("could not parse eval receipt %s: %s", eval_receipt, exc)
        return {}
    out: dict[str, Any] = {}
    for key in ("map50", "map5095", "hazards_mean_recall", "hard_negative_fpr"):
        if key in data and data[key] is not None:
            out[_camel(key)] = data[key]
    return out


def _read_trained_at(energy_receipt: Path | None) -> str | None:
    """Pull the training-finish timestamp from an energy receipt."""
    if energy_receipt is None or not energy_receipt.exists():
        return None
    try:
        data = json.loads(energy_receipt.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    val = data.get("finished_at_utc") or data.get("started_at_utc")
    return val if isinstance(val, str) else None


def _camel(snake: str) -> str:
    parts = snake.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _validate_version(version: str) -> None:
    """Spec §12 requires v<MAJOR>.<MINOR>.<PATCH>-<precision>."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(
            f"version {version!r} does not match "
            f"v<MAJOR>.<MINOR>.<PATCH>-<precision> "
            f"where precision in (fp32, fp16, int8)"
        )


def _precision_to_export_kwargs(precision: str) -> dict[str, Any]:
    """Map the version's precision suffix to ultralytics export() kwargs."""
    if precision == "fp32":
        return {"half": False}
    if precision == "fp16":
        return {"half": True}
    if precision == "int8":
        raise NotImplementedError(
            "INT8 export requires a calibration dataset; not wired for V1. "
            "See spec §3 non-goals. Use fp16 instead."
        )
    raise ValueError(f"unknown precision {precision!r}")


def export(
    weights: Path,
    data_yaml: Path,
    output_dir: Path,
    model_name: str,
    version: str,
    imgsz: int = DEFAULT_IMGSZ,
    eval_receipt: Path | None = None,
    energy_receipt: Path | None = None,
    conf_recommended: float = DEFAULT_CONF,
    iou_recommended: float = DEFAULT_IOU,
) -> tuple[Path, Path]:
    """Run the export. Returns (onnx_path, meta_path).

    Raises if the resulting .onnx exceeds the 6 MB ceiling.
    """
    from ultralytics import YOLO

    _validate_version(version)
    precision = version.rsplit("-", 1)[1]
    export_kwargs = _precision_to_export_kwargs(precision)

    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _read_class_names(data_yaml)

    LOG.info("loading weights from %s", weights)
    model = YOLO(str(weights))

    LOG.info(
        "exporting to ONNX (imgsz=%d, %s, simplify=True, dynamic=False)",
        imgsz, precision,
    )
    # ultralytics writes the .onnx alongside the .pt by default; we
    # accept that and move it into output_dir so the producer side
    # doesn't accumulate cruft in the training output dir.
    produced = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=True,
            dynamic=False,
            opset=17,
            **export_kwargs,
        )
    )
    if not produced.exists():
        raise RuntimeError(f"ultralytics reported success but {produced} is missing")

    onnx_dst = output_dir / f"{model_name}.onnx"
    shutil.move(str(produced), str(onnx_dst))

    size_bytes = onnx_dst.stat().st_size
    LOG.info("artifact: %s (%.2f MB)", onnx_dst, size_bytes / 1024 / 1024)
    if size_bytes > MAX_ONNX_BYTES:
        raise RuntimeError(
            f"{onnx_dst} is {size_bytes / 1024 / 1024:.2f} MB; "
            f"spec §10 caps at {MAX_ONNX_BYTES / 1024 / 1024:.0f} MB. "
            f"Use a smaller precision or revisit imgsz."
        )

    meta: dict[str, Any] = {
        "version": version,
        "modelName": model_name,
        "trainedAt": _read_trained_at(energy_receipt),
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputShape": [1, 3, imgsz, imgsz],
        "classes": classes,
        "classCount": len(classes),
        "confThresholdRecommended": conf_recommended,
        "iouThresholdRecommended": iou_recommended,
        "trainingMetrics": _read_training_metrics(eval_receipt),
        "artifactSizeBytes": size_bytes,
    }

    meta_dst = output_dir / f"{model_name}.meta.json"
    meta_dst.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    LOG.info("sidecar: %s", meta_dst)

    return onnx_dst, meta_dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained YOLO detector to ONNX + meta.json for the "
            "dregsbane-web-trail PWA. See spec sections 11.1, 11.3, 12."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"C:\tmp\runs\v1-take6\weights\best.pt"),
        help="Path to the .pt weights file (default: V1 take6 local path).",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path(r"C:\tmp\v1_dataset\data.yaml"),
        help="Path to the data.yaml used during training (for class names).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dist/models/v1"),
        help="Where to write the artifacts. Created if missing.",
    )
    parser.add_argument(
        "--model-name",
        default="yolo11n-litter",
        help="Stem of the artifact filenames.",
    )
    parser.add_argument(
        "--version",
        default="v1.0.0-fp16",
        help="Version string. Format: v<MAJOR>.<MINOR>.<PATCH>-(fp32|fp16|int8).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Square input edge length (default 640).",
    )
    parser.add_argument(
        "--eval-receipt",
        type=Path,
        default=Path(r"C:\tmp\eval-runs\v1-take6-eval\eval_receipt.json"),
        help="eval_receipt.json to pull training metrics from. Optional.",
    )
    parser.add_argument(
        "--energy-receipt",
        type=Path,
        default=Path(r"C:\tmp\runs\v1-take6\energy_receipt.json"),
        help="energy_receipt.json to pull trainedAt timestamp from. Optional.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_CONF,
        help="Recommended confidence threshold for downstream consumers.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU,
        help="Recommended IoU NMS threshold for downstream consumers.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.weights.exists():
        LOG.error("weights not found: %s", args.weights)
        return 2
    if not args.data_yaml.exists():
        LOG.error("data.yaml not found: %s", args.data_yaml)
        return 2

    eval_rcpt = args.eval_receipt if args.eval_receipt.exists() else None
    energy_rcpt = args.energy_receipt if args.energy_receipt.exists() else None
    if eval_rcpt is None:
        LOG.warning(
            "eval receipt %s missing; meta.json will lack trainingMetrics",
            args.eval_receipt,
        )
    if energy_rcpt is None:
        LOG.warning(
            "energy receipt %s missing; meta.json will lack trainedAt",
            args.energy_receipt,
        )

    try:
        onnx_path, meta_path = export(
            weights=args.weights,
            data_yaml=args.data_yaml,
            output_dir=args.output_dir,
            model_name=args.model_name,
            version=args.version,
            imgsz=args.imgsz,
            eval_receipt=eval_rcpt,
            energy_receipt=energy_rcpt,
            conf_recommended=args.conf_threshold,
            iou_recommended=args.iou_threshold,
        )
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        LOG.error("export failed: %s", exc)
        return 1

    LOG.info("export complete: %s, %s", onnx_path, meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
