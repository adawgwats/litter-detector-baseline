"""Pull TACO + hard-negatives from R2 + convert to YOLO format on local disk.

Output directory layout (Ultralytics-standard):

    <out_dir>/
      train/
        images/<sha>.jpg
        labels/<sha>.txt    # YOLO format: <cls> <xc> <yc> <w> <h> per line
      val/
        images/
        labels/
      data.yaml             # generated config consumed by `ultralytics yolo train`

Class indexing
    The class set is the union of OLM leaves that TACO maps to via the
    crosswalk (43 leaves as of 2026-05-22). Hard-negative images get an
    empty .txt label file — YOLO treats those as pure background.

Coordinates
    TACO annotations are COCO-format bboxes ``[x, y, w, h]`` in pixel
    coords on the ORIGINAL image. YOLO needs normalized center-x, center-y,
    width, height as fractions of image dimensions. We convert per-image.

Determinism
    Train/val split uses a fixed random seed so the same images land in
    the same split across runs. Sort order is sha256(image_bytes) so the
    seeding is content-deterministic, not depend on R2 listing order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from litter_detector_baseline.ingest.crosswalk import map_label, valid_olm_leaves
from litter_detector_baseline.ingest.storage import Storage

log = logging.getLogger(__name__)

CF_API_BASE = "https://api.cloudflare.com/client/v4"

# Object-key prefixes set by the ingest modules
TACO_PREFIX = "taco/photos"
HN_PREFIX = "hard_negatives/photos"


@dataclass
class DatasetStats:
    """Aggregate statistics emitted alongside the prepared dataset."""

    class_names: list[str] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    n_train_bboxes: int = 0
    n_val_bboxes: int = 0
    bboxes_per_class: dict[str, int] = field(default_factory=dict)
    skipped: int = 0
    failed: int = 0
    total_bytes_pulled: int = 0


def _list_keys(storage: Storage, prefix: str) -> list[dict]:
    """List object keys + sizes under a prefix via the CF management API.

    Storage's S3 backend doesn't expose a list method, but we know we're on
    the CF-management-API backend (locked decision) — call the management
    list endpoint directly. Each entry: {key, size}.
    """
    assert storage.is_cf_mgmt_backend, "prepare_dataset only supports CF-mgmt backend currently"
    import requests  # noqa: PLC0415 — local to keep top imports tight

    headers = {"Authorization": f"Bearer {storage.cf_api_token}"}
    base = f"{CF_API_BASE}/accounts/{storage.cf_account_id}/r2/buckets/{storage.bucket}/objects"
    out: list[dict] = []
    cursor: Optional[str] = None
    while True:
        params: dict = {"prefix": prefix, "per_page": 1000}
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


def _coco_to_yolo(
    bbox_xywh: list[float], image_w: int, image_h: int
) -> Optional[tuple[float, float, float, float]]:
    """Convert COCO ``[x, y, w, h]`` (pixels) to YOLO ``(xc, yc, w, h)``
    normalized to [0, 1]. Returns None for degenerate bboxes."""
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
        return None
    xc = (x + w / 2.0) / image_w
    yc = (y + h / 2.0) / image_h
    nw = w / image_w
    nh = h / image_h
    # Clamp to [0, 1] for the rare TACO annotation that slightly oversteps
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return (xc, yc, nw, nh)


def _build_class_index(taco_annotations: dict) -> tuple[list[str], dict[str, int]]:
    """Build the YOLO class list as the union of OLM leaves TACO maps to.

    Order is deterministic (alphabetical by leaf name) so two runs produce
    the same class indices. Returns ``(class_names, leaf_to_idx)``.
    """
    leaves: set[str] = set()
    for cat in taco_annotations["categories"]:
        leaf = map_label("taco", cat["name"])
        if leaf is None:
            raise RuntimeError(
                f"TACO category {cat['name']!r} not in crosswalk; refresh "
                "configs/label_crosswalk.csv before re-running prepare_dataset"
            )
        leaves.add(leaf)
    class_names = sorted(leaves)
    return class_names, {leaf: idx for idx, leaf in enumerate(class_names)}


def _write_image_and_label(
    out_dir: Path,
    split: str,
    name_stem: str,
    image_bytes: bytes,
    yolo_lines: list[str],
) -> None:
    (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / split / "images" / f"{name_stem}.jpg").write_bytes(image_bytes)
    label_path = out_dir / split / "labels" / f"{name_stem}.txt"
    label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")


def prepare(
    *,
    storage: Storage,
    out_dir: Path,
    annotations_path: Path,
    val_ratio: float = 0.2,
    seed: int = 1337,
    max_taco: Optional[int] = None,
    max_hn: Optional[int] = None,
) -> DatasetStats:
    """Pull TACO + hard-negatives from R2, convert to YOLO format under
    ``out_dir``, write a data.yaml. Returns aggregated stats."""
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    class_names, leaf_to_idx = _build_class_index(annotations)
    log.info("class index: %d classes", len(class_names))

    stats = DatasetStats(class_names=class_names)
    stats.bboxes_per_class = {n: 0 for n in class_names}

    # ─── TACO ─────────────────────────────────────────────────────────────
    # Build photo_id -> (image_metadata, annotation list) lookup from the
    # snapshot annotations.json (which we already have locally).
    cats_by_id = {c["id"]: c for c in annotations["categories"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Pull list of TACO metadata JSONs from R2 — gives us (sha) -> photo_id
    # mapping that the ingest module wrote.
    taco_metadata_keys = [
        x for x in _list_keys(storage, f"{TACO_PREFIX}/_metadata/")
        if x["key"].endswith(".json")
    ]
    log.info("found %d TACO metadata records in R2", len(taco_metadata_keys))
    if max_taco is not None:
        taco_metadata_keys = taco_metadata_keys[:max_taco]

    rng = random.Random(seed)

    # First pass: load each TACO metadata to get image_key + annotations.
    # We need the TACO image (width, height) for normalization — that comes
    # from the annotations.json snapshot (NOT from the per-image metadata
    # we stored in R2, which has the file_name and dims).
    images_by_id = {im["id"]: im for im in annotations["images"]}

    for idx, meta in enumerate(taco_metadata_keys, start=1):
        try:
            md = json.loads(storage.get_bytes(meta["key"]))
        except Exception as exc:
            log.warning("failed to read TACO metadata %s: %s", meta["key"], exc)
            stats.failed += 1
            continue

        image_id = md.get("image_id")
        image_key = md.get("image_key")
        if not image_key or image_id is None:
            stats.skipped += 1
            continue

        img_info = images_by_id.get(image_id)
        if not img_info:
            stats.skipped += 1
            continue
        image_w = int(img_info["width"])
        image_h = int(img_info["height"])

        # Resolve YOLO bboxes from the snapshot annotations for this image
        yolo_lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            cat = cats_by_id.get(ann["category_id"])
            if not cat:
                continue
            leaf = map_label("taco", cat["name"])
            if leaf is None:
                continue
            yolo_xywh = _coco_to_yolo(ann["bbox"], image_w, image_h)
            if yolo_xywh is None:
                continue
            cls_idx = leaf_to_idx[leaf]
            xc, yc, nw, nh = yolo_xywh
            yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            stats.bboxes_per_class[leaf] += 1

        if not yolo_lines:
            stats.skipped += 1
            continue

        try:
            image_bytes = storage.get_bytes(image_key)
        except Exception as exc:
            log.warning("failed to read TACO image %s: %s", image_key, exc)
            stats.failed += 1
            continue

        stats.total_bytes_pulled += len(image_bytes)
        split = "val" if rng.random() < val_ratio else "train"
        # Stable filename: sha256 of original bytes
        sha = hashlib.sha256(image_bytes).hexdigest()[:16]
        _write_image_and_label(out_dir, split, f"taco_{sha}", image_bytes, yolo_lines)
        if split == "train":
            stats.n_train += 1
            stats.n_train_bboxes += len(yolo_lines)
        else:
            stats.n_val += 1
            stats.n_val_bboxes += len(yolo_lines)
        if idx % 100 == 0:
            log.info("[TACO %d/%d] train=%d val=%d", idx, len(taco_metadata_keys),
                     stats.n_train, stats.n_val)

    # ─── Hard negatives ───────────────────────────────────────────────────
    hn_metadata_keys = [
        x for x in _list_keys(storage, f"{HN_PREFIX}/_metadata/")
        if x["key"].endswith(".json")
    ]
    log.info("found %d hard-negative metadata records in R2", len(hn_metadata_keys))
    if max_hn is not None:
        hn_metadata_keys = hn_metadata_keys[:max_hn]

    for idx, meta in enumerate(hn_metadata_keys, start=1):
        try:
            md = json.loads(storage.get_bytes(meta["key"]))
        except Exception as exc:
            log.warning("failed to read HN metadata %s: %s", meta["key"], exc)
            stats.failed += 1
            continue
        image_key = md.get("image_key")
        if not image_key:
            stats.skipped += 1
            continue
        try:
            image_bytes = storage.get_bytes(image_key)
        except Exception as exc:
            log.warning("failed to read HN image %s: %s", image_key, exc)
            stats.failed += 1
            continue
        stats.total_bytes_pulled += len(image_bytes)
        split = "val" if rng.random() < val_ratio else "train"
        sha = hashlib.sha256(image_bytes).hexdigest()[:16]
        # Empty label file = YOLO treats as pure background (negative example)
        _write_image_and_label(out_dir, split, f"hn_{sha}", image_bytes, [])
        if split == "train":
            stats.n_train += 1
        else:
            stats.n_val += 1
        if idx % 200 == 0:
            log.info("[HN %d/%d] train=%d val=%d", idx, len(hn_metadata_keys),
                     stats.n_train, stats.n_val)

    # ─── data.yaml ────────────────────────────────────────────────────────
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    # ─── stats receipt ────────────────────────────────────────────────────
    receipt = {
        "n_classes": len(class_names),
        "class_names": class_names,
        "n_train": stats.n_train,
        "n_val": stats.n_val,
        "n_train_bboxes": stats.n_train_bboxes,
        "n_val_bboxes": stats.n_val_bboxes,
        "bboxes_per_class": stats.bboxes_per_class,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "total_bytes_pulled": stats.total_bytes_pulled,
        "val_ratio": val_ratio,
        "seed": seed,
    }
    (out_dir / "prepare_stats.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    log.info("wrote %d train + %d val images to %s", stats.n_train, stats.n_val, out_dir)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output dataset directory (will be created)")
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/taco/annotations.json"),
                        help="TACO annotations.json (for bbox + dims)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-taco", type=int, default=None,
                        help="Cap TACO image count (for smoke tests)")
    parser.add_argument("--max-hn", type=int, default=None,
                        help="Cap hard-negative count")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    storage = Storage.from_env()
    prepare(
        storage=storage,
        out_dir=args.out_dir,
        annotations_path=args.annotations,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_taco=args.max_taco,
        max_hn=args.max_hn,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
