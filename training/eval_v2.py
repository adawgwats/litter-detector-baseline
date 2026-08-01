"""V2 eval: YOLO11n ONNX vs RF-DETR checkpoint on the SAME val set.

Both models are scored by the same self-contained metric code in this
module — NOT by each framework's own val loop — so the numbers are
apples-to-apples:

  - mAP@0.5 and mAP@[0.5:0.95] (greedy score-ordered matching,
    101-point interpolated AP; a faithful approximation of COCO mAP
    without area ranges / maxDets buckets)
  - Per-class precision + recall at the model's operating confidence
  - Hazards mean recall (leaf space only; see eval_v1.HAZARDS_LEAVES)
  - Hard-negative FPR (any detection on a zero-label image counts)

Each model is reported at TWO granularities:
  - ``leaf43``  — the canonical 43-leaf space from data.yaml
  - ``coarse10`` — the rollup defined by training/rollups/coarse10.yaml

Ground truth comes from the YOLO val split in --data-yaml by default, or
from an external COCO val set via --coco-val (e.g. the RF100-VL TACO
split). External COCO category names are mapped into the canonical space
by exact leaf name first, then via the TACO crosswalk
(configs/label_crosswalk.csv); unmappable boxes are dropped and counted.

Model inference paths:
  - YOLO: the existing baseline path (OnnxLitterDetector: letterbox 640,
    /255, single-output decode, class-wise NMS).
  - RF-DETR: the rfdetr package (``RFDETRSmall(pretrain_weights=...)``
    ``.predict(path, threshold=...)`` -> supervision Detections with
    pixel-space xyxy; for fine-tuned checkpoints class_id is a 0-based
    index into the checkpoint's class_names — pinned against
    rfdetr==1.9.0, https://rfdetr.roboflow.com/latest/learn/train/).

Output layout (one eval_receipt.json per model per granularity, in the
exact eval_v1.EvalReceipt shape, plus a comparison summary):

    <output-dir>/<run-name>/
      summary.json
      yolo11n-onnx/leaf43/eval_receipt.json
      yolo11n-onnx/coarse10/eval_receipt.json
      rfdetr-s/leaf43/eval_receipt.json
      rfdetr-s/coarse10/eval_receipt.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from training.eval_v1 import HAZARDS_LEAVES, EvalReceipt

log = logging.getLogger(__name__)

DEFAULT_ROLLUP = Path(__file__).parent / "rollups" / "coarse10.yaml"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# AP thresholds: COCO's 0.50:0.95:0.05 ladder
_AP_IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


# ─── Ground truth ─────────────────────────────────────────────────────────

@dataclass
class GtImage:
    path: Path
    # rows of (class_idx, x1, y1, x2, y2) in original-image pixels
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)


def read_canonical_classes(data_yaml: Path) -> list[str]:
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = raw.get("names")
    if not isinstance(names, dict):
        raise ValueError(
            f"{data_yaml} 'names' is not a dict — was it produced by "
            "training/data/prepare_dataset.py?"
        )
    return [names[i] for i in sorted(names.keys())]


def load_gt_yolo(data_yaml: Path) -> list[GtImage]:
    """Val split of a prepare_dataset.py YOLO-layout dataset."""
    from PIL import Image  # noqa: PLC0415

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    images_dir = Path(cfg["path"]) / cfg["val"]
    labels_dir = images_dir.parent / "labels"
    out: list[GtImage] = []
    for img_path in sorted(p for p in images_dir.iterdir()
                           if p.suffix.lower() in _IMAGE_EXTS):
        gt = GtImage(path=img_path)
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists() and label_path.stat().st_size > 0:
            with Image.open(img_path) as im:
                width, height = im.size
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc, yc, nw, nh = (float(v) for v in parts[1:])
                w, h = nw * width, nh * height
                x1, y1 = xc * width - w / 2.0, yc * height - h / 2.0
                gt.boxes.append((cls, x1, y1, x1 + w, y1 + h))
        out.append(gt)
    return out


def _taco_crosswalk() -> dict[str, str]:
    """TACO label -> OLM leaf, read straight from the crosswalk CSV.

    Deliberately NOT via litter_detector_baseline.ingest.crosswalk: the
    ingest package __init__ imports boto3/storage, which the eval env
    (rfdetr + onnxruntime, no ingest extras) does not install.
    """
    import csv  # noqa: PLC0415

    csv_path = Path(__file__).resolve().parent.parent / "configs" / "label_crosswalk.csv"
    rows = [
        ln for ln in csv_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    return {
        row["source_label"]: row["olm_leaf_label"]
        for row in csv.DictReader(rows)
        if row["source_dataset"] == "taco"
    }


def load_gt_coco(coco_val: Path, class_names: list[str]) -> list[GtImage]:
    """External COCO val set (json file or a Roboflow split dir containing
    _annotations.coco.json). Category names map into the canonical space
    by exact leaf name, else via the TACO crosswalk; iscrowd boxes and
    unmappable categories are dropped (dropped counts are logged)."""
    taco_to_leaf = _taco_crosswalk()

    ann_path = coco_val / "_annotations.coco.json" if coco_val.is_dir() else coco_val
    images_dir = ann_path.parent
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    name_to_idx = {n: i for i, n in enumerate(class_names)}
    cat_to_idx: dict[int, Optional[int]] = {}
    unmapped: set[str] = set()
    for cat in data.get("categories", []):
        name = cat["name"]
        idx = name_to_idx.get(name)
        if idx is None:
            leaf = taco_to_leaf.get(name)
            idx = name_to_idx.get(leaf) if leaf else None
        if idx is None:
            unmapped.add(name)
        cat_to_idx[cat["id"]] = idx
    if unmapped:
        log.warning("dropping GT boxes for %d unmappable categories: %s",
                    len(unmapped), sorted(unmapped))

    by_image: dict[int, GtImage] = {}
    for im in data.get("images", []):
        by_image[im["id"]] = GtImage(path=images_dir / im["file_name"])
    dropped = 0
    for ann in data.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        idx = cat_to_idx.get(ann["category_id"])
        if idx is None:
            dropped += 1
            continue
        x, y, w, h = ann["bbox"]
        gt = by_image.get(ann["image_id"])
        if gt is not None:
            gt.boxes.append((idx, x, y, x + w, y + h))
    if dropped:
        log.warning("dropped %d GT boxes with unmappable categories", dropped)
    return [by_image[k] for k in sorted(by_image)]


# ─── Predictors ───────────────────────────────────────────────────────────
#
# Both return, per image, rows of (class_idx, score, x1, y1, x2, y2) in
# original-image pixels with class_idx in the CANONICAL space. Scores are
# collected down to --ap-floor so mAP sees the full PR curve; the
# operating-threshold metrics filter later.

class YoloOnnxPredictor:
    def __init__(self, onnx_path: Path, class_names: list[str],
                 imgsz: int, iou_threshold: float, ap_floor: float) -> None:
        from litter_detector_baseline.onnx_backend import OnnxLitterDetector  # noqa: PLC0415

        self._detector = OnnxLitterDetector(
            weights=onnx_path,
            class_names=class_names,
            input_size=(imgsz, imgsz),
            score_threshold=ap_floor,
            iou_threshold=iou_threshold,
        )

    def __call__(self, image_path: Path) -> list[tuple[int, float, float, float, float, float]]:
        from litter_detector_baseline.io import load_image_rgb  # noqa: PLC0415

        dets = self._detector.predict(load_image_rgb(image_path))
        return [(d.class_id, d.score, d.x1, d.y1, d.x2, d.y2) for d in dets]


class RfdetrPredictor:
    def __init__(self, checkpoint: Path, class_names: list[str],
                 ap_floor: float) -> None:
        from rfdetr import RFDETRSmall  # noqa: PLC0415

        self._model = RFDETRSmall(pretrain_weights=str(checkpoint))
        self._ap_floor = ap_floor

        model_names = self._checkpoint_class_names(checkpoint)
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        self._remap = [name_to_idx.get(n, -1) for n in model_names]
        unknown = [n for n in model_names if n not in name_to_idx]
        if unknown:
            log.warning("rfdetr checkpoint has %d class names outside the "
                        "canonical space (their detections are dropped): %s",
                        len(unknown), unknown)

    def _checkpoint_class_names(self, checkpoint: Path) -> list[str]:
        """Checkpoint's own class order: the in-memory model first, then
        the training_config.json rfdetr writes next to checkpoints."""
        names = getattr(self._model, "class_names", None) \
            or getattr(self._model.model, "class_names", None)
        if names:
            return list(names)
        tc = checkpoint.parent / "training_config.json"
        if tc.exists():
            data = json.loads(tc.read_text(encoding="utf-8"))
            if data.get("class_names"):
                return list(data["class_names"])
        raise RuntimeError(
            f"cannot recover class names for {checkpoint}; expected them on "
            "the model or in training_config.json next to the checkpoint"
        )

    def __call__(self, image_path: Path) -> list[tuple[int, float, float, float, float, float]]:
        dets = self._model.predict(str(image_path), threshold=self._ap_floor)
        out = []
        for (x1, y1, x2, y2), score, cls in zip(
            dets.xyxy, dets.confidence, dets.class_id
        ):
            idx = self._remap[int(cls)] if 0 <= int(cls) < len(self._remap) else -1
            if idx >= 0:
                out.append((idx, float(score), float(x1), float(y1), float(x2), float(y2)))
        return out


# ─── Metrics ──────────────────────────────────────────────────────────────

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two (N,4)/(M,4) xyxy arrays."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _match_class(
    preds: list[tuple[int, float, np.ndarray]],   # (img_idx, score, box)
    gts: dict[int, np.ndarray],                    # img_idx -> (M,4)
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy score-ordered matching for one class. Returns (tp, fp, npos)
    with tp/fp aligned to preds sorted by descending score."""
    npos = sum(len(b) for b in gts.values())
    order = sorted(range(len(preds)), key=lambda k: -preds[k][1])
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched: dict[int, np.ndarray] = {i: np.zeros(len(b), dtype=bool)
                                      for i, b in gts.items()}
    for rank, k in enumerate(order):
        img_idx, _, box = preds[k]
        gt_boxes = gts.get(img_idx)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp[rank] = 1
            continue
        ious = _iou_matrix(box[None, :], gt_boxes)[0]
        ious[matched[img_idx]] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[rank] = 1
            matched[img_idx][j] = True
        else:
            fp[rank] = 1
    return tp, fp, npos


def _ap_101(tp: np.ndarray, fp: np.ndarray, npos: int) -> Optional[float]:
    """101-point interpolated AP. None when the class has no GT."""
    if npos == 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / npos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        mask = recall >= r
        ap += float(precision[mask].max()) if mask.any() else 0.0
    return ap / 101.0


@dataclass
class MetricResult:
    map50: Optional[float]
    map5095: Optional[float]
    per_class_recall: dict[str, float]
    per_class_precision: dict[str, float]
    hard_negative_fpr: Optional[float]
    n_val_images: int


def compute_metrics(
    gt_images: list[GtImage],
    preds_per_image: list[list[tuple[int, float, float, float, float, float]]],
    class_names: list[str],
    conf: float,
) -> MetricResult:
    n_classes = len(class_names)

    # Index GT + predictions by class
    gts_by_class: list[dict[int, np.ndarray]] = [dict() for _ in range(n_classes)]
    for img_idx, gt in enumerate(gt_images):
        for c in range(n_classes):
            boxes = [b[1:] for b in gt.boxes if b[0] == c]
            if boxes:
                gts_by_class[c][img_idx] = np.asarray(boxes, dtype=np.float64)
    preds_by_class: list[list[tuple[int, float, np.ndarray]]] = [[] for _ in range(n_classes)]
    for img_idx, preds in enumerate(preds_per_image):
        for cls, score, x1, y1, x2, y2 in preds:
            preds_by_class[cls].append(
                (img_idx, score, np.asarray([x1, y1, x2, y2], dtype=np.float64))
            )

    # mAP over the IoU ladder (all predictions, down to the AP floor)
    ap_per_thr: list[list[float]] = []
    map50 = None
    for thr in _AP_IOU_THRESHOLDS:
        aps = []
        for c in range(n_classes):
            tp, fp, npos = _match_class(preds_by_class[c], gts_by_class[c], thr)
            ap = _ap_101(tp, fp, npos)
            if ap is not None:
                aps.append(ap)
        ap_per_thr.append(aps)
        if thr == 0.5 and aps:
            map50 = float(np.mean(aps))
    thr_means = [float(np.mean(aps)) for aps in ap_per_thr if aps]
    map5095 = float(np.mean(thr_means)) if thr_means else None

    # Operating-point precision / recall at IoU 0.5, score >= conf
    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    for c in range(n_classes):
        preds_c = [p for p in preds_by_class[c] if p[1] >= conf]
        tp, fp, npos = _match_class(preds_c, gts_by_class[c], 0.5)
        if npos == 0:
            continue  # match eval_v1: only classes present in val are reported
        n_tp, n_fp = float(tp.sum()), float(fp.sum())
        per_class_recall[class_names[c]] = n_tp / npos
        per_class_precision[class_names[c]] = (
            n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        )

    # Hard-negative FPR: zero-GT images with any detection at >= conf
    hn_total = 0
    hn_hits = 0
    for img_idx, gt in enumerate(gt_images):
        if gt.boxes:
            continue
        hn_total += 1
        if any(p[1] >= conf for p in preds_per_image[img_idx]):
            hn_hits += 1
    hard_negative_fpr = (hn_hits / hn_total) if hn_total else None

    return MetricResult(
        map50=map50,
        map5095=map5095,
        per_class_recall=per_class_recall,
        per_class_precision=per_class_precision,
        hard_negative_fpr=hard_negative_fpr,
        n_val_images=len(gt_images),
    )


# ─── Rollup ───────────────────────────────────────────────────────────────

def load_rollup(rollup_yaml: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    """Load a rollup and return (group_names, leaf_idx -> group_idx).

    Every canonical class must be assigned to exactly one group; anything
    missing or duplicated is a config error, not a runtime condition.
    """
    raw = yaml.safe_load(rollup_yaml.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = raw["groups"]
    group_names = list(groups.keys())
    leaf_to_group: dict[str, int] = {}
    for gi, (gname, leaves) in enumerate(groups.items()):
        for leaf in leaves:
            if leaf in leaf_to_group:
                raise ValueError(f"{rollup_yaml}: {leaf!r} assigned to two groups")
            leaf_to_group[leaf] = gi
    missing = [n for n in class_names if n not in leaf_to_group]
    if missing:
        raise ValueError(f"{rollup_yaml}: classes not covered: {missing}")
    extra = [n for n in leaf_to_group if n not in set(class_names)]
    if extra:
        raise ValueError(f"{rollup_yaml}: unknown classes: {extra}")
    return group_names, [leaf_to_group[n] for n in class_names]


def _roll_gt(gt_images: list[GtImage], remap: list[int]) -> list[GtImage]:
    return [
        GtImage(path=g.path, boxes=[(remap[b[0]], *b[1:]) for b in g.boxes])
        for g in gt_images
    ]


def _roll_preds(
    preds_per_image: list[list[tuple[int, float, float, float, float, float]]],
    remap: list[int],
) -> list[list[tuple[int, float, float, float, float, float]]]:
    return [[(remap[p[0]], *p[1:]) for p in preds] for preds in preds_per_image]


# ─── Receipts ─────────────────────────────────────────────────────────────

def _write_receipt(
    *,
    out_dir: Path,
    run_name: str,
    model_path: str,
    data_yaml: Path,
    result: MetricResult,
    hazards: bool,
    notes: list[str],
) -> EvalReceipt:
    hazards_present = [n for n in result.per_class_recall if n in HAZARDS_LEAVES] \
        if hazards else []
    hazards_mean_recall = (
        sum(result.per_class_recall[n] for n in hazards_present) / len(hazards_present)
        if hazards_present else None
    )
    receipt = EvalReceipt(
        run_name=run_name,
        model_path=model_path,
        data_yaml=str(data_yaml),
        map50=result.map50,
        map5095=result.map5095,
        per_class_recall=result.per_class_recall,
        per_class_precision=result.per_class_precision,
        hazards_classes_present=hazards_present,
        hazards_mean_recall=hazards_mean_recall,
        hard_negative_fpr=result.hard_negative_fpr,
        n_val_images=result.n_val_images,
        notes=notes if hazards else notes + [
            "hazards metrics are leaf-space concepts; not computed at rollup granularity"
        ],
    )
    out_path = out_dir / "eval_receipt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(receipt), indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    return receipt


def evaluate_model(
    *,
    model_tag: str,
    model_path: str,
    predictor,
    gt_images: list[GtImage],
    class_names: list[str],
    group_names: list[str],
    leaf_to_group: list[int],
    conf: float,
    run_dir: Path,
    run_name: str,
    data_yaml: Path,
    base_notes: list[str],
) -> dict[str, EvalReceipt]:
    """Run one model over the val set once, score at both granularities."""
    preds_per_image = []
    for i, gt in enumerate(gt_images, start=1):
        preds_per_image.append(predictor(gt.path))
        if i % 100 == 0:
            log.info("[%s] %d/%d images", model_tag, i, len(gt_images))

    receipts: dict[str, EvalReceipt] = {}
    leaf_result = compute_metrics(gt_images, preds_per_image, class_names, conf)
    receipts["leaf43"] = _write_receipt(
        out_dir=run_dir / model_tag / "leaf43",
        run_name=f"{run_name}/{model_tag}/leaf43",
        model_path=model_path,
        data_yaml=data_yaml,
        result=leaf_result,
        hazards=True,
        notes=base_notes + [f"granularity=leaf43 ({len(class_names)} classes)"],
    )
    rolled_result = compute_metrics(
        _roll_gt(gt_images, leaf_to_group),
        _roll_preds(preds_per_image, leaf_to_group),
        group_names,
        conf,
    )
    receipts["coarse10"] = _write_receipt(
        out_dir=run_dir / model_tag / "coarse10",
        run_name=f"{run_name}/{model_tag}/coarse10",
        model_path=model_path,
        data_yaml=data_yaml,
        result=rolled_result,
        hazards=False,
        notes=base_notes + [f"granularity=coarse10 ({len(group_names)} groups)"],
    )
    return receipts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-yaml", type=Path, required=True,
                        help="data.yaml from prepare_dataset.py — defines the "
                             "canonical class order, and the default val split")
    parser.add_argument("--yolo-onnx", type=Path, default=None,
                        help="YOLO11n ONNX (skipped if omitted)")
    parser.add_argument("--detr-checkpoint", type=Path, default=None,
                        help="RF-DETR .pth checkpoint, e.g. "
                             "runs/<name>/checkpoint_best_total.pth "
                             "(skipped if omitted)")
    parser.add_argument("--coco-val", type=Path, default=None,
                        help="External COCO val set (json or Roboflow split "
                             "dir) to score against instead of the data.yaml "
                             "val split, e.g. the RF100-VL TACO split")
    parser.add_argument("--rollup", type=Path, default=DEFAULT_ROLLUP,
                        help="Rollup YAML (default: training/rollups/coarse10.yaml)")
    parser.add_argument("--output-dir", type=Path, default=Path("eval-runs"))
    parser.add_argument("--run-name", default=None,
                        help="Default: eval-v2-<UTC-timestamp>")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO operating confidence (v1 convention)")
    parser.add_argument("--detr-conf", type=float, default=0.4,
                        help="RF-DETR operating confidence (DETR-typical; "
                             "matches the exported meta.json recommendation)")
    parser.add_argument("--yolo-iou", type=float, default=0.45,
                        help="YOLO NMS IoU (the DETR path has no NMS)")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--ap-floor", type=float, default=0.01,
                        help="Score floor for prediction collection; mAP is "
                             "computed over everything above it")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.yolo_onnx is None and args.detr_checkpoint is None:
        parser.error("nothing to evaluate: pass --yolo-onnx and/or --detr-checkpoint")

    run_name = args.run_name or (
        f"eval-v2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    run_dir = args.output_dir / run_name

    class_names = read_canonical_classes(args.data_yaml)
    group_names, leaf_to_group = load_rollup(args.rollup, class_names)

    if args.coco_val is not None:
        gt_images = load_gt_coco(args.coco_val, class_names)
        gt_note = f"gt_source=coco:{args.coco_val}"
    else:
        gt_images = load_gt_yolo(args.data_yaml)
        gt_note = f"gt_source=yolo-val:{args.data_yaml}"
    log.info("val set: %d images, %d GT boxes",
             len(gt_images), sum(len(g.boxes) for g in gt_images))

    summary: dict = {
        "run_name": run_name,
        "gt_source": gt_note,
        "n_val_images": len(gt_images),
        "rollup": str(args.rollup),
        "models": {},
    }
    common_notes = [gt_note, f"ap_floor={args.ap_floor}",
                    "metrics=self-contained greedy-match AP101 (see eval_v2 docstring)"]

    if args.yolo_onnx is not None:
        predictor = YoloOnnxPredictor(
            args.yolo_onnx, class_names, args.yolo_imgsz,
            args.yolo_iou, args.ap_floor,
        )
        receipts = evaluate_model(
            model_tag="yolo11n-onnx",
            model_path=str(args.yolo_onnx),
            predictor=predictor,
            gt_images=gt_images,
            class_names=class_names,
            group_names=group_names,
            leaf_to_group=leaf_to_group,
            conf=args.yolo_conf,
            run_dir=run_dir,
            run_name=run_name,
            data_yaml=args.data_yaml,
            base_notes=common_notes + [f"conf={args.yolo_conf}",
                                       f"nms_iou={args.yolo_iou}"],
        )
        summary["models"]["yolo11n-onnx"] = {
            gran: {"map50": r.map50, "map5095": r.map5095,
                   "hazards_mean_recall": r.hazards_mean_recall,
                   "hard_negative_fpr": r.hard_negative_fpr}
            for gran, r in receipts.items()
        }

    if args.detr_checkpoint is not None:
        predictor = RfdetrPredictor(args.detr_checkpoint, class_names, args.ap_floor)
        receipts = evaluate_model(
            model_tag="rfdetr-s",
            model_path=str(args.detr_checkpoint),
            predictor=predictor,
            gt_images=gt_images,
            class_names=class_names,
            group_names=group_names,
            leaf_to_group=leaf_to_group,
            conf=args.detr_conf,
            run_dir=run_dir,
            run_name=run_name,
            data_yaml=args.data_yaml,
            base_notes=common_notes + [f"conf={args.detr_conf}", "nms=none"],
        )
        summary["models"]["rfdetr-s"] = {
            gran: {"map50": r.map50, "map5095": r.map5095,
                   "hazards_mean_recall": r.hazards_mean_recall,
                   "hard_negative_fpr": r.hard_negative_fpr}
            for gran, r in receipts.items()
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("wrote %s", run_dir / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
