"""V1 eval: runs the metrics gating success per docs/contributor_assist_goal.md.

  - mAP@0.5 (Ultralytics built-in)
  - Per-class recall, with explicit Hazards-class breakdown
  - Hard-negative FPR (any detection on a no-litter image counts)

Eval set is the held-out 20% split written by prepare_dataset.py. The
``hazards`` class set is hard-coded to the OLM leaves that we consider
operationally-risky false negatives.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Hazards: false-negatives here are operationally costly (sharps, biohazard).
# False-positives on a bottle are cheap; false-negatives on a syringe are not.
# Per docs/contributor_assist_goal.md the target is recall >= 0.85 on these.
HAZARDS_LEAVES: frozenset[str] = frozenset({
    "medical.syringe",
    "medical.gloves",
    "medical.face_mask",
    "medical.pill_pack",
    "medical.medicine_bottle",
    "sanitary.tampon",
    "sanitary.sanitary_pad",
    "sanitary.nappies",
    "alcohol.broken_glass",
    "softdrinks.broken_glass",
})


@dataclass
class EvalReceipt:
    run_name: str
    model_path: str
    data_yaml: str
    map50: Optional[float]
    map5095: Optional[float]
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    hazards_classes_present: list[str] = field(default_factory=list)
    hazards_mean_recall: Optional[float] = None
    hard_negative_fpr: Optional[float] = None
    n_val_images: int = 0
    notes: list[str] = field(default_factory=list)


def evaluate(
    *,
    weights: Path,
    data_yaml: Path,
    output_dir: Path,
    run_name: str,
    imgsz: int = 640,
    conf: float = 0.25,
) -> EvalReceipt:
    """Run Ultralytics validation + compute additional gating metrics."""
    from ultralytics import YOLO  # noqa: PLC0415

    yolo = YOLO(str(weights))
    metrics = yolo.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=conf,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
    )

    # Ultralytics returns a DetMetrics object; extract per-class numbers
    names = metrics.names  # dict[int, str]
    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    if metrics.box is not None:
        try:
            per_class_recall = {
                names[i]: float(metrics.box.r[i]) for i in range(len(metrics.box.r))
            }
            per_class_precision = {
                names[i]: float(metrics.box.p[i]) for i in range(len(metrics.box.p))
            }
        except Exception as exc:
            log.warning("per-class metrics extraction failed: %s", exc)

    hazards_present = [n for n in per_class_recall if n in HAZARDS_LEAVES]
    hazards_mean_recall = (
        sum(per_class_recall[n] for n in hazards_present) / len(hazards_present)
        if hazards_present else None
    )

    # Hard-negative FPR is computed below via a separate prediction pass on
    # the val images that have empty labels. Done here to keep this script
    # standalone — the Ultralytics val pass doesn't break out hard-neg FPR
    # natively.
    hard_neg_fpr = _compute_hard_negative_fpr(
        weights=weights, data_yaml=data_yaml, imgsz=imgsz, conf=conf,
    )

    receipt = EvalReceipt(
        run_name=run_name,
        model_path=str(weights),
        data_yaml=str(data_yaml),
        map50=float(metrics.box.map50) if metrics.box is not None else None,
        map5095=float(metrics.box.map) if metrics.box is not None else None,
        per_class_recall=per_class_recall,
        per_class_precision=per_class_precision,
        hazards_classes_present=hazards_present,
        hazards_mean_recall=hazards_mean_recall,
        hard_negative_fpr=hard_neg_fpr,
        n_val_images=0,  # Ultralytics 8.4 doesn't surface this directly on metrics.box
        notes=[],
    )
    out_path = output_dir / run_name / "eval_receipt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(receipt), indent=2), encoding="utf-8")
    log.info("wrote eval_receipt.json to %s", out_path)
    return receipt


def _compute_hard_negative_fpr(
    *, weights: Path, data_yaml: Path, imgsz: int, conf: float
) -> Optional[float]:
    """Run predictions on val images whose label.txt is empty (= hard
    negatives) and compute FPR as ``(images with any detection) / total``."""
    import yaml  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    val_dir = Path(cfg["path"]) / cfg["val"]
    labels_dir = val_dir.parent / "labels"

    hn_images = []
    for img_path in val_dir.glob("*.jpg"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists() or label_path.stat().st_size == 0:
            hn_images.append(img_path)
    if not hn_images:
        log.info("no hard-negative images in val split — skipping FPR")
        return None

    yolo = YOLO(str(weights))
    n_with_detection = 0
    for img in hn_images:
        result = yolo.predict(source=str(img), imgsz=imgsz, conf=conf, verbose=False)
        if result and result[0].boxes is not None and len(result[0].boxes) > 0:
            n_with_detection += 1
    return n_with_detection / len(hn_images)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to trained YOLO weights (e.g. runs/<name>/weights/best.pt)")
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval-runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from datetime import datetime, timezone
    run_name = args.run_name or f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    evaluate(
        weights=args.weights,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        run_name=run_name,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
