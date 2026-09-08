"""Export a fine-tuned RF-DETR checkpoint to ONNX + sidecar meta.json.

Produces the artifacts the dregsbane-web-backend Lambda inference layer
consumes. The sidecar carries the same fields as the YOLO exporter's
(see export/export_yolov8.py) plus the two fields the backend's
architecture-aware branch keys on:

  - ``architecture: "rfdetr"`` — routes decode to the DETR path
    (dual-output, sigmoid over logits, top-k, NO NMS)
  - ``normalization: {mean, std}`` — ImageNet stats; RF-DETR's ONNX
    graph does NOT bake normalization in, the consumer must apply it
    after /255 (verified against rfdetr==1.9.0,
    https://rfdetr.roboflow.com/latest/learn/export/)

rfdetr export facts pinned against rfdetr==1.9.0 (verified 2026-08-01):
  - ``RFDETRSmall(pretrain_weights=<checkpoint.pth>).export(format="onnx",
    opset_version=17, output_dir=..., shape=(H, W), batch_size=1)``
    writes ``inference_model.onnx`` (stem overridable via output_name)
    into output_dir. https://rfdetr.roboflow.com/latest/learn/export/
  - Graph outputs are named ``dets`` (pred_boxes: [1, 300 queries, 4]
    cxcywh normalized to [0,1] of the model input) and ``labels``
    (pred_logits: [1, 300, C'] RAW — consumer applies sigmoid). The
    checkpoint's class head is built with one extra background column
    (rf-detr detr.py infers num_classes as class_embed.shape[0] - 1),
    so C' may be classCount + 1; canonical classes occupy columns
    0..classCount-1 in checkpoint class_names order and decode must
    ignore any trailing column.
  - Shape must be divisible by patch_size * num_windows (16 * 2 = 32
    for RFDETRSmall; native resolution 512).

Class-order safety: the sidecar's ``classes`` are read from the training
data.yaml (canonical 43-leaf alphabetical order). When rfdetr's
``training_config.json`` is found next to the checkpoint, its
class_names must match exactly (same order) or the export refuses — a
mismatch means logit column i is NOT canonical class i and the served
model would emit wrong labels.

Unlike the YOLO exporter there is no 6 MB ceiling: RF-DETR Small serves
from Lambda (server-side), not the browser. Per the energy policy
("compression at rest") a ``.onnx.gz`` is written alongside; upload both.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from export.export_yolov8 import (
    _read_class_names,
    _read_trained_at,
    _read_training_metrics,
    _validate_version,
)
from export.precision import convert, plan, target_for

LOG = logging.getLogger("export.export_rfdetr")

DEFAULT_RESOLUTION = 512  # RFDETRSmall native; must stay divisible by 32
# DETR confidence behaves differently from YOLO's: scores are per-query
# sigmoid probabilities with no NMS score-shaping, and RF-DETR's own
# predict() default is 0.5. We recommend 0.4 to bias toward recall — the
# contributor-assist flow surfaces suggestions for a human to confirm, so
# a missed detection costs more than an extra suggestion.
DEFAULT_CONF = 0.4
# Unused by the DETR decode path (no NMS); kept at the YOLO default so
# the sidecar satisfies the backend's ModelMeta schema.
DEFAULT_IOU = 0.45
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _check_class_alignment(checkpoint: Path, classes: list[str]) -> None:
    """Refuse to export when the checkpoint's class order disagrees with
    the canonical list the sidecar will advertise."""
    tc_path = checkpoint.parent / "training_config.json"
    if not tc_path.exists():
        LOG.warning(
            "no training_config.json next to %s — cannot verify the "
            "checkpoint's class order matches data.yaml; proceeding on trust",
            checkpoint,
        )
        return
    try:
        trained = json.loads(tc_path.read_text(encoding="utf-8")).get("class_names")
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("could not parse %s: %s", tc_path, exc)
        return
    if trained and list(trained) != list(classes):
        raise RuntimeError(
            f"class-order mismatch: {tc_path} has {len(trained)} classes "
            f"that differ from the canonical data.yaml order "
            f"({len(classes)} classes). Logit column i would not be "
            f"canonical class i. Re-train from a canonical-order dataset, "
            f"or pass --skip-class-check if you are CERTAIN this is safe."
        )


def _to_fp16(onnx_path: Path) -> None:
    """Convert the exported fp32 graph to fp16 in place.

    The conversion itself now lives in :mod:`export.precision`, which is
    the single place any precision conversion happens and where the IO
    boundary is an explicit field rather than a hard-coded keyword. This
    function is the shim that keeps that call site's *behaviour*
    identical: ``rfdetr-s-512-fp16`` declares ``io_precision=fp32``
    (i.e. ``keep_io_types=True``, so the backend's ORT session keeps
    feeding and reading float32) with ``repair``/``validate`` off,
    which is exactly what this exporter did before.

    That the shipping exporter neither repairs nor validates is a known
    defect, not a design choice — reports/CONVERSION-REPORT.md §2 shows
    the resulting artifact does not load in ONNX Runtime. Fixing it
    changes an artifact's bytes and so is a release decision; the
    descriptor that does fix it is ``rfdetr-s-512-fp16-repaired``, and
    it is what ``scripts/make_fp16.py`` uses.
    """
    from export.precision import convert, get_target, plan

    convert(plan(get_target("rfdetr-s-512-fp16")), onnx_path)


def _gzip_alongside(onnx_path: Path) -> Path:
    """Write <name>.onnx.gz next to the .onnx (energy policy: compression
    at rest). mtime pinned to 0 so re-exports of identical bytes are
    byte-identical archives."""
    gz_path = onnx_path.with_suffix(onnx_path.suffix + ".gz")
    with onnx_path.open("rb") as src, gz_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as dst:
            shutil.copyfileobj(src, dst)
    return gz_path


def export(
    checkpoint: Path,
    data_yaml: Path,
    output_dir: Path,
    model_name: str,
    version: str,
    resolution: int = DEFAULT_RESOLUTION,
    eval_receipt: Path | None = None,
    energy_receipt: Path | None = None,
    conf_recommended: float = DEFAULT_CONF,
    iou_recommended: float = DEFAULT_IOU,
    opset: int = 17,
    skip_class_check: bool = False,
) -> tuple[Path, Path, Path]:
    """Run the export. Returns (onnx_path, gz_path, meta_path)."""
    from rfdetr import RFDETRSmall

    _validate_version(version)
    precision = version.rsplit("-", 1)[1]
    # One descriptor carries precision, IO boundary, opset and input
    # shape. It also owns the two refusals that used to be inline here:
    # INT8 without a calibration pass, and a resolution that is not
    # divisible by RFDETRSmall's patch_size * num_windows.
    target = target_for(
        "rfdetr",
        precision,
        input_shape=(1, 3, resolution, resolution),
        opset=opset,
    )
    conversion = plan(target)

    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _read_class_names(data_yaml)
    if len(classes) != 43:
        LOG.warning(
            "expected the canonical 43-leaf space, data.yaml has %d classes",
            len(classes),
        )
    if not skip_class_check:
        _check_class_alignment(checkpoint, classes)

    LOG.info("loading checkpoint %s", checkpoint)
    model = RFDETRSmall(pretrain_weights=str(checkpoint))

    LOG.info("exporting to ONNX (shape=%dx%d, opset=%d) — %s",
             resolution, resolution, target.opset, conversion.describe())
    with tempfile.TemporaryDirectory(prefix="rfdetr-export-") as tmp:
        produced = Path(
            model.export(
                format="onnx",
                output_dir=tmp,
                opset_version=target.opset,
                shape=(resolution, resolution),
                batch_size=1,
            )
        )
        # export() returns the artifact path; older releases returned the
        # directory — resolve either way.
        onnx_src = produced if produced.suffix == ".onnx" \
            else produced / "inference_model.onnx"
        if not onnx_src.exists():
            raise RuntimeError(
                f"rfdetr export reported success but {onnx_src} is missing"
            )
        onnx_dst = output_dir / f"{model_name}.onnx"
        shutil.move(str(onnx_src), str(onnx_dst))

    # The shared precision path — the single place a conversion runs.
    # For fp32 it is a no-op; for fp16 it applies the graph conversion
    # with the descriptor's explicit IO boundary.
    convert(conversion, onnx_dst)

    size_bytes = target.check_artifact_size(onnx_dst)
    LOG.info("artifact: %s (%.2f MB)", onnx_dst, size_bytes / 1024 / 1024)

    gz_dst = _gzip_alongside(onnx_dst)
    LOG.info("compressed: %s (%.2f MB)", gz_dst, gz_dst.stat().st_size / 1024 / 1024)

    meta: dict[str, Any] = {
        "version": version,
        "modelName": model_name,
        "trainedAt": _read_trained_at(energy_receipt),
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputShape": [1, 3, resolution, resolution],
        "classes": classes,
        "classCount": len(classes),
        "confThresholdRecommended": conf_recommended,
        "iouThresholdRecommended": iou_recommended,
        "architecture": "rfdetr",
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "trainingMetrics": _read_training_metrics(eval_receipt),
        "artifactSizeBytes": size_bytes,
    }
    meta_dst = output_dir / f"{model_name}.meta.json"
    meta_dst.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    LOG.info("sidecar: %s", meta_dst)

    return onnx_dst, gz_dst, meta_dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a fine-tuned RF-DETR checkpoint to ONNX + meta.json "
            "for the dregsbane-web-backend Lambda inference layer."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="RF-DETR .pth checkpoint (use checkpoint_best_total.pth).",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="data.yaml used during training (canonical class order).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dist/models/v2"),
        help="Where to write the artifacts. Created if missing.",
    )
    parser.add_argument(
        "--model-name",
        default="rfdetr-s-litter",
        help="Stem of the artifact filenames.",
    )
    parser.add_argument(
        "--version",
        default="v2.0.0-fp16",
        help="Version string. Format: v<MAJOR>.<MINOR>.<PATCH>-(fp32|fp16|int8).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Square input edge (default 512; must be divisible by 32 and "
             "match the training resolution).",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--eval-receipt",
        type=Path,
        default=None,
        help="eval_receipt.json to pull training metrics from. Optional.",
    )
    parser.add_argument(
        "--energy-receipt",
        type=Path,
        default=None,
        help="energy_receipt.json to pull trainedAt timestamp from. Optional.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_CONF,
        help="Recommended confidence threshold (DETR-typical 0.4; see "
             "module docstring for why it differs from YOLO's 0.25).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU,
        help="Recommended IoU threshold. Unused by the DETR decode path "
             "(no NMS); kept for ModelMeta schema compatibility.",
    )
    parser.add_argument(
        "--skip-class-check",
        action="store_true",
        help="Skip the checkpoint-vs-data.yaml class-order verification. "
             "Only when you are CERTAIN the orders match.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.checkpoint.exists():
        LOG.error("checkpoint not found: %s", args.checkpoint)
        return 2
    if not args.data_yaml.exists():
        LOG.error("data.yaml not found: %s", args.data_yaml)
        return 2

    eval_rcpt = args.eval_receipt if args.eval_receipt and args.eval_receipt.exists() else None
    energy_rcpt = args.energy_receipt if args.energy_receipt and args.energy_receipt.exists() else None
    if eval_rcpt is None:
        LOG.warning("no eval receipt; meta.json will lack trainingMetrics")
    if energy_rcpt is None:
        LOG.warning("no energy receipt; meta.json will lack trainedAt")

    try:
        onnx_path, gz_path, meta_path = export(
            checkpoint=args.checkpoint,
            data_yaml=args.data_yaml,
            output_dir=args.output_dir,
            model_name=args.model_name,
            version=args.version,
            resolution=args.resolution,
            eval_receipt=eval_rcpt,
            energy_receipt=energy_rcpt,
            conf_recommended=args.conf_threshold,
            iou_recommended=args.iou_threshold,
            opset=args.opset,
            skip_class_check=args.skip_class_check,
        )
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        LOG.error("export failed: %s", exc)
        return 1

    LOG.info("export complete: %s, %s, %s", onnx_path, gz_path, meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
