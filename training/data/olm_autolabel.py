"""Auto-label OLM photos via autodistill-grounded-sam, emit YOLO-format
labels using the V1 class index.

Workflow:

  1. Load the OLM-leaf -> caption ontology from
     ``configs/olm_autolabel_ontology.json``.
  2. List ``openlittermap/photos/_metadata/*.json`` keys in R2 (each
     represents one OLM photo). Random-sample N of them with a fixed seed.
  3. For each sampled photo: download bytes, run Grounding-DINO + SAM
     (positive captions + negative-prompt suppression), translate the
     resulting detections into YOLO ``<cls> <xc> <yc> <w> <h>`` lines, and
     write the image + label into the output dataset directory.

Output layout matches the v1_dataset shape so the result can either be
trained standalone OR concatenated into an existing v1_dataset/train/
split for V1.1.

Class index: we re-use the V1 alphabetical-sort-of-43-OLM-leaves index
that ``prepare_dataset.py`` writes to ``data.yaml``. Leaves not present
in the ontology (food.other, other.other, etc. — too-generic prompts
that Grounding-DINO can't disambiguate) get a 0-detection budget for
their class but still occupy their original index, so the cls_idx values
in this output are identical to V1's. This means an auto-labeled OLM
image can be dropped into a V1-trained train/ directory without any
relabeling step.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from litter_detector_baseline.ingest.crosswalk import map_label
from litter_detector_baseline.ingest.storage import Storage

log = logging.getLogger(__name__)

CF_API_BASE = "https://api.cloudflare.com/client/v4"
OLM_METADATA_PREFIX = "openlittermap/photos/_metadata/"
ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "olm_autolabel_ontology.json"
TACO_ANNOTATIONS = Path(__file__).resolve().parent.parent.parent / "data" / "taco" / "annotations.json"


@dataclass
class AutolabelStats:
    n_photos_attempted: int = 0
    n_photos_labeled: int = 0
    n_photos_skipped_no_detections: int = 0
    n_photos_failed_download: int = 0
    n_photos_failed_predict: int = 0
    total_detections: int = 0
    detections_per_class: dict[str, int] = field(default_factory=dict)


def _load_ontology(path: Path) -> tuple[dict[str, str], list[str], dict]:
    """Returns (prompt_to_leaf, negative_prompts, thresholds)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["prompts"], data["negative_prompts"], data["thresholds"]


def _build_class_index() -> tuple[list[str], dict[str, int]]:
    """V1 class index: sorted union of OLM leaves TACO maps to (43 classes).
    Matches what prepare_dataset.py wrote to v1_dataset/data.yaml."""
    annotations = json.loads(TACO_ANNOTATIONS.read_text(encoding="utf-8"))
    leaves: set[str] = set()
    for cat in annotations["categories"]:
        leaf = map_label("taco", cat["name"])
        if leaf is None:
            raise RuntimeError(
                f"TACO category {cat['name']!r} not in crosswalk; refresh "
                "configs/label_crosswalk.csv before re-running"
            )
        leaves.add(leaf)
    class_names = sorted(leaves)
    return class_names, {leaf: idx for idx, leaf in enumerate(class_names)}


def _list_olm_metadata_keys(storage: Storage) -> list[dict]:
    """List every OLM metadata key in R2 with its image_key + photo_id."""
    assert storage.is_cf_mgmt_backend
    headers = {"Authorization": f"Bearer {storage.cf_api_token}"}
    base = f"{CF_API_BASE}/accounts/{storage.cf_account_id}/r2/buckets/{storage.bucket}/objects"
    out: list[dict] = []
    cursor: Optional[str] = None
    while True:
        params: dict = {"prefix": OLM_METADATA_PREFIX, "per_page": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(base, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for item in data.get("result") or []:
            out.append({"key": item["key"], "size": int(item.get("size", "0"))})
        cursor = (data.get("result_info") or {}).get("cursor")
        if not cursor:
            break
    return out


def _detections_to_yolo_lines(
    detections,
    prompt_to_leaf: dict[str, str],
    leaf_to_idx: dict[str, int],
    image_w: int,
    image_h: int,
    prompt_order: list[str],
) -> list[str]:
    """Convert supervision.Detections to YOLO ``<cls> <xc> <yc> <w> <h>`` lines."""
    lines: list[str] = []
    if detections.xyxy is None or len(detections.xyxy) == 0:
        return lines
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i]
        cls_id_in_ontology = int(detections.class_id[i])
        if cls_id_in_ontology < 0 or cls_id_in_ontology >= len(prompt_order):
            continue
        prompt = prompt_order[cls_id_in_ontology]
        leaf = prompt_to_leaf.get(prompt)
        if not leaf or leaf not in leaf_to_idx:
            continue
        w_box = x2 - x1
        h_box = y2 - y1
        if w_box <= 0 or h_box <= 0 or image_w <= 0 or image_h <= 0:
            continue
        xc = (x1 + w_box / 2.0) / image_w
        yc = (y1 + h_box / 2.0) / image_h
        nw = w_box / image_w
        nh = h_box / image_h
        # Clamp
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        lines.append(
            f"{leaf_to_idx[leaf]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
        )
    return lines


def autolabel(
    *,
    storage: Storage,
    out_dir: Path,
    max_photos: Optional[int] = None,
    seed: int = 1337,
    ontology_path: Path = ONTOLOGY_PATH,
    skip_sam: bool = True,
) -> AutolabelStats:
    """End-to-end auto-label pipeline. Heavy GPU import deferred until here."""
    # Heavy imports: defer so unit-tests of the module's helpers don't need GPU
    from autodistill.detection import CaptionOntology  # noqa: PLC0415
    from autodistill_grounded_sam import GroundedSAM  # noqa: PLC0415
    from autodistill_grounded_sam.helpers import (  # noqa: PLC0415
        combine_detections,
        load_grounding_dino,
        suppress_by_negative_boxes,
    )
    from autodistill.helpers import load_image  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    prompt_to_leaf, negative_prompts, thresholds = _load_ontology(ontology_path)
    prompt_order = list(prompt_to_leaf.keys())
    class_names, leaf_to_idx = _build_class_index()
    log.info("ontology: %d prompts -> %d unique OLM leaves; %d negative prompts",
             len(prompt_to_leaf), len({*prompt_to_leaf.values()}), len(negative_prompts))
    log.info("V1 class index: %d classes", len(class_names))

    if skip_sam:
        # GroundingDINO-only path: SAM masks are unused downstream (YOLO
        # detection labels only need bboxes), and SAM is ~half the wall time.
        # Bypass the SAM load and inline the Grounding-DINO + negative-prompt
        # logic from the fork's predict() — minus the SAM segmentation tail.
        grounding_dino = load_grounding_dino()
        log.info("GroundingDINO loaded (SAM skipped)")
        box_t = thresholds["box_threshold"]
        text_t = thresholds["text_threshold"]
        neg_iou = thresholds["negative_iou_threshold"]

        class _BoxesOnlyPredictor:
            ontology = CaptionOntology(prompt_to_leaf)

            def predict(self, image):
                detections_list = []
                for description in self.ontology.prompts():
                    d = grounding_dino.predict_with_classes(
                        image=image, classes=[description],
                        box_threshold=box_t, text_threshold=text_t,
                    )
                    detections_list.append(d)
                detections = combine_detections(
                    detections_list, overwrite_class_ids=range(len(detections_list))
                )
                if negative_prompts:
                    negative_boxes = []
                    for prompt in negative_prompts:
                        neg = grounding_dino.predict_with_classes(
                            image=image, classes=[prompt],
                            box_threshold=box_t, text_threshold=text_t,
                        )
                        if len(neg.xyxy):
                            negative_boxes.append(neg.xyxy)
                    negative_xyxy = (
                        np.concatenate(negative_boxes, axis=0)
                        if negative_boxes
                        else np.empty((0, 4))
                    )
                    detections = suppress_by_negative_boxes(
                        detections, negative_xyxy, neg_iou,
                    )
                return detections

        base_model = _BoxesOnlyPredictor()
    else:
        base_model = GroundedSAM(
            ontology=CaptionOntology(prompt_to_leaf),
            box_threshold=thresholds["box_threshold"],
            text_threshold=thresholds["text_threshold"],
            negative_prompts=negative_prompts,
            negative_iou_threshold=thresholds["negative_iou_threshold"],
        )
        log.info("GroundedSAM loaded")

    # Discover OLM photos in R2
    keys = _list_olm_metadata_keys(storage)
    log.info("found %d OLM photos in R2", len(keys))
    rng = random.Random(seed)
    rng.shuffle(keys)
    if max_photos is not None:
        keys = keys[:max_photos]
    log.info("processing %d photos", len(keys))

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = AutolabelStats(n_photos_attempted=len(keys))
    stats.detections_per_class = {n: 0 for n in class_names}

    for idx, item in enumerate(keys, start=1):
        try:
            md_bytes = storage.get_bytes(item["key"])
            md = json.loads(md_bytes)
        except Exception as exc:
            log.warning("metadata read failed %s: %s", item["key"], exc)
            stats.n_photos_failed_download += 1
            continue
        image_key = md.get("image_key")
        photo_id = md.get("photo_id")
        if not image_key or photo_id is None:
            stats.n_photos_failed_download += 1
            continue

        try:
            image_bytes = storage.get_bytes(image_key)
        except Exception as exc:
            log.warning("image read failed photo_id=%s: %s", photo_id, exc)
            stats.n_photos_failed_download += 1
            continue

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            stats.n_photos_failed_download += 1
            continue
        image_h, image_w = image.shape[:2]

        try:
            detections = base_model.predict(image)
        except Exception as exc:
            log.warning("predict failed photo_id=%s: %s", photo_id, exc)
            stats.n_photos_failed_predict += 1
            continue

        # Class-agnostic NMS — Grounding-DINO runs ONE forward pass per
        # caption with no cross-prompt dedup. A single bottle gets tagged
        # with caption 0 (beer bottle), caption 1 (beer bottle cap),
        # caption 2 (broken glass), ... etc., producing 30+ detections at
        # nearly-identical bboxes. Drop overlapping low-confidence dupes.
        if len(detections.xyxy) > 0:
            nms_iou = thresholds.get("class_agnostic_nms_iou", 0.5)
            detections = detections.with_nms(threshold=nms_iou, class_agnostic=True)

        yolo_lines = _detections_to_yolo_lines(
            detections, prompt_to_leaf, leaf_to_idx, image_w, image_h, prompt_order,
        )
        if not yolo_lines:
            stats.n_photos_skipped_no_detections += 1
            continue

        stem = f"olm_{photo_id}"
        (out_images / f"{stem}.jpg").write_bytes(image_bytes)
        (out_labels / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        stats.n_photos_labeled += 1
        stats.total_detections += len(yolo_lines)

        # Track per-class detections
        for line in yolo_lines:
            cls_idx = int(line.split()[0])
            cls_name = class_names[cls_idx]
            stats.detections_per_class[cls_name] = stats.detections_per_class.get(cls_name, 0) + 1

        if idx % 25 == 0 or idx == len(keys):
            log.info("[%d/%d] labeled=%d skipped_no_det=%d failed_download=%d failed_predict=%d total_det=%d",
                     idx, len(keys), stats.n_photos_labeled,
                     stats.n_photos_skipped_no_detections,
                     stats.n_photos_failed_download,
                     stats.n_photos_failed_predict,
                     stats.total_detections)

    # Emit stats receipt + class-names file so a downstream trainer can
    # reuse this dir directly as a YOLO data dir.
    out_stats = {
        "n_photos_attempted": stats.n_photos_attempted,
        "n_photos_labeled": stats.n_photos_labeled,
        "n_photos_skipped_no_detections": stats.n_photos_skipped_no_detections,
        "n_photos_failed_download": stats.n_photos_failed_download,
        "n_photos_failed_predict": stats.n_photos_failed_predict,
        "total_detections": stats.total_detections,
        "detections_per_class": stats.detections_per_class,
        "ontology_path": str(ontology_path),
        "seed": seed,
        "class_names": class_names,
    }
    (out_dir / "autolabel_stats.json").write_text(
        json.dumps(out_stats, indent=2), encoding="utf-8"
    )
    log.info("wrote stats to %s", out_dir / "autolabel_stats.json")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory; receives images/ + labels/ + autolabel_stats.json")
    parser.add_argument("--max-photos", type=int, default=None,
                        help="Cap to first N photos (after random shuffle with --seed)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ontology",
                        type=Path,
                        default=ONTOLOGY_PATH,
                        help="Path to OLM auto-label ontology JSON")
    parser.add_argument("--with-sam", action="store_true",
                        help="Run full GroundedSAM (slower) instead of GroundingDINO-only. "
                             "Default is GroundingDINO-only since we only need bboxes.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    storage = Storage.from_env()
    autolabel(
        storage=storage,
        out_dir=args.out_dir,
        max_photos=args.max_photos,
        seed=args.seed,
        ontology_path=args.ontology,
        skip_sam=not args.with_sam,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
