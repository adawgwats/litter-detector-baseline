"""V2 training entry point: RF-DETR Small fine-tune on the 43-leaf dataset.

Model choice per the V2 plan:
  - RF-DETR Small (Apache-2.0 tier — the rfdetr package and its
    Apache-designated weights; the XL/2XL "Plus" models are PML-licensed
    and out of bounds). https://github.com/roboflow/rf-detr#license
  - Resolution 512 (RFDETRSmall native; must stay divisible by
    patch_size * num_windows = 16 * 2 = 32).
  - Pretrained fine-tune only (locked decision — no training from scratch).

rfdetr API facts pinned against rfdetr==1.9.0 (verified 2026-08-01):
  - Train: ``RFDETRSmall().train(dataset_dir=..., epochs=..., batch_size=...,
    grad_accum_steps=..., lr=..., output_dir=..., resolution=...)``.
    Docs recommend total batch (batch_size * grad_accum_steps) of 16;
    quick-start uses lr=1e-4, epochs=100.
    https://rfdetr.roboflow.com/latest/learn/train/
  - Dataset: COCO layout — ``<dir>/{train,valid,test}/_annotations.coco.json``
    plus images in the same split folder (exactly what a Roboflow project
    download in "coco" format produces).
    https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/
  - Checkpoints land in output_dir as ``checkpoint.pth`` /
    ``checkpoint_best_ema.pth`` / ``checkpoint_best_regular.pth`` /
    ``checkpoint_best_total.pth`` alongside ``training_config.json``
    (which records class_names — the export step cross-checks it).

Per docs/ai-energy-policy.md (dregsbane-web-trail) every run emits an
``energy_receipt.json``; the EnergyReceipt shape is shared with V1. In
addition, the run REFUSES to start unless ``--policy-ack`` is passed,
acknowledging that the five decision-framework questions from the energy
policy have been answered in the accompanying PR description.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from training.train_yolo11n_v1 import (
    DEFAULT_REGION,
    DEFAULT_GRID_G_CO2_PER_KWH,
    GPU_TDP_W,
    REGION_GRID_G_CO2_PER_KWH,
    EnergyReceipt,
    _detect_gpu,
    _peak_ram_mb,
)

log = logging.getLogger(__name__)

# ─── Energy-policy gate ───────────────────────────────────────────────────
#
# Verbatim from docs/ai-energy-policy.md (dregsbane-web-trail) § "Decision
# framework for new ML proposals". A training run is a new-model proposal;
# the answers belong in the PR description, and --policy-ack asserts they
# exist. "A proposal that does not answer these questions does not get
# built."
POLICY_DOC = "dregsbane-web-trail/docs/ai-energy-policy.md"
POLICY_QUESTIONS: tuple[str, ...] = (
    "1. Can this be solved without ML at all?",
    "2. Can this be solved with a tiny on-device model?",
    "3. If a server model is needed, can it scale to zero?",
    "4. Have we measured the actual energy / carbon impact?",
    "5. Have we considered a foundation-model API call (Claude/GPT) as "
    "the alternative?",
)

# Splits: YOLO layout name -> COCO layout name. rfdetr expects Roboflow's
# train/valid/test naming; our prepare_dataset.py writes train/val.
_SPLIT_MAP = (("train", "train"), ("val", "valid"))
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _read_data_yaml_classes(data_yaml: Path) -> list[str]:
    """Ordered class list from a prepare_dataset.py data.yaml (names is a
    dict keyed by integer index; order = canonical 43-leaf alphabetical)."""
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = raw.get("names")
    if not isinstance(names, dict):
        raise ValueError(
            f"{data_yaml} 'names' is not a dict — was it produced by "
            "training/data/prepare_dataset.py?"
        )
    return [names[i] for i in sorted(names.keys())]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when the filesystem allows it (NTFS does), copy otherwise.
    Keeps the valid->test mirror nearly free on the RTX box."""
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def yolo_to_coco(data_yaml: Path, out_dir: Path) -> Path:
    """Convert a prepare_dataset.py YOLO-layout dataset to the COCO layout
    rfdetr trains on.

    Writes ``out_dir/{train,valid,test}/_annotations.coco.json`` + images.
    ``test/`` mirrors ``valid/`` via hardlinks — rfdetr expects all three
    splits (Roboflow convention) and we have no third split to give it.

    Category ids are 1-based in data.yaml index order (= canonical
    alphabetical), so rfdetr's dataset-derived class_names come out in
    canonical order and a fine-tuned checkpoint's 0-based class_id i maps
    to canonical class i. Images with empty label files (hard negatives)
    are kept, with zero annotations.
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    classes = _read_data_yaml_classes(data_yaml)
    categories = [
        {"id": i + 1, "name": name, "supercategory": name.split(".")[0]}
        for i, name in enumerate(classes)
    ]

    from PIL import Image  # noqa: PLC0415 — train extra, lazy like v1

    for yolo_split, coco_split in _SPLIT_MAP:
        images_dir = root / cfg[yolo_split] if not Path(cfg[yolo_split]).is_absolute() \
            else Path(cfg[yolo_split])
        labels_dir = images_dir.parent / "labels"
        split_dir = out_dir / coco_split
        split_dir.mkdir(parents=True, exist_ok=True)

        images: list[dict] = []
        annotations: list[dict] = []
        ann_id = 1
        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )
        for img_id, img_path in enumerate(image_paths, start=1):
            with Image.open(img_path) as im:
                width, height = im.size
            _link_or_copy(img_path, split_dir / img_path.name)
            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            })
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                xc, yc, nw, nh = (float(v) for v in parts[1:])
                w = nw * width
                h = nh * height
                x = max(0.0, xc * width - w / 2.0)
                y = max(0.0, yc * height - h / 2.0)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_idx + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = {
            "info": {"description": f"converted from {data_yaml}"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8"
        )
        log.info("[%s] %d images, %d bboxes", coco_split, len(images), len(annotations))

    # rfdetr expects a test/ split; mirror valid/ (hardlinks, ~free).
    valid_dir = out_dir / "valid"
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in valid_dir.iterdir():
        _link_or_copy(p, test_dir / p.name)
    return out_dir


def _coco_split_stats(dataset_dir: Path, split: str) -> tuple[int, int]:
    """(n_images, total_bytes) for one COCO split. Bytes are summed over
    the image files the annotations actually reference."""
    ann_path = dataset_dir / split / "_annotations.coco.json"
    if not ann_path.exists():
        return 0, 0
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    total = 0
    for im in data.get("images", []):
        p = dataset_dir / split / im["file_name"]
        if p.exists():
            total += p.stat().st_size
    return len(data.get("images", [])), total


def train(
    *,
    dataset_dir: Path,
    output_dir: Path,
    run_name: str,
    epochs: int = 100,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    resolution: int = 512,
    grid_region: str = DEFAULT_REGION,
    early_stopping: bool = False,
    notes: Optional[list[str]] = None,
) -> EnergyReceipt:
    """Drive an RF-DETR Small training run + emit an energy receipt next
    to the resulting checkpoints (same EnergyReceipt shape as V1)."""
    from rfdetr import RFDETRSmall  # noqa: PLC0415 — heavy import, lazy

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_mono = time.monotonic()
    gpu_type, gpu_count = _detect_gpu()
    log.info("starting %s on %d x %s", run_name, gpu_count, gpu_type or "(no GPU)")

    n_train, train_bytes = _coco_split_stats(dataset_dir, "train")
    n_val, val_bytes = _coco_split_stats(dataset_dir, "valid")

    model = RFDETRSmall()  # downloads rf-detr-small.pth pretrained weights
    train_kwargs: dict = {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "resolution": resolution,
        "output_dir": str(run_dir),
    }
    if early_stopping:
        train_kwargs["early_stopping"] = True
    model.train(**train_kwargs)

    wall_seconds = time.monotonic() - started_mono
    finished_at = datetime.now(timezone.utc)

    tdp = GPU_TDP_W.get(gpu_type) if gpu_type else None
    estimated_gpu_hours = (wall_seconds / 3600.0) * max(1, gpu_count)
    estimated_kwh = (estimated_gpu_hours * (tdp or 0)) / 1000.0
    grid_factor = REGION_GRID_G_CO2_PER_KWH.get(grid_region, DEFAULT_GRID_G_CO2_PER_KWH)
    estimated_g_co2eq = estimated_kwh * grid_factor

    receipt = EnergyReceipt(
        run_name=run_name,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        wall_seconds=wall_seconds,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_tdp_w_per_unit=tdp,
        estimated_gpu_hours=estimated_gpu_hours,
        estimated_kwh=estimated_kwh,
        grid_region=grid_region,
        grid_g_co2_per_kwh=grid_factor,
        estimated_g_co2eq=estimated_g_co2eq,
        peak_ram_mb=_peak_ram_mb(),
        n_train_images=n_train,
        n_val_images=n_val,
        total_train_bytes=train_bytes + val_bytes,
        epochs=epochs,
        imgsz=resolution,
        batch=batch_size,
        model="rf-detr-small",
        optimizer="AdamW",  # rfdetr's fixed internal optimizer
        lr0=lr,
        notes=(notes or []) + [
            f"grad_accum_steps={grad_accum_steps}",
            f"effective_batch={batch_size * grad_accum_steps}",
            "arch=RF-DETR Small (Apache-2.0 tier), rfdetr>=1.9",
            f"policy_ack=true ({POLICY_DOC} decision framework answered in PR)",
        ],
    )
    (run_dir / "energy_receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2), encoding="utf-8"
    )
    log.info("wrote energy_receipt.json to %s", run_dir)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path, default=None,
                        help="COCO-layout dataset (train/valid/test with "
                             "_annotations.coco.json each), e.g. a Roboflow "
                             "project downloaded in 'coco' format")
    source.add_argument("--data-yaml", type=Path, default=None,
                        help="YOLO-layout data.yaml from prepare_dataset.py; "
                             "converted to COCO layout under --coco-out first")
    parser.add_argument("--coco-out", type=Path, default=None,
                        help="Where the YOLO->COCO conversion is written "
                             "(default: <output-dir>/<run-name>-dataset). "
                             "Only used with --data-yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"),
                        help="Where rfdetr writes the run + checkpoints")
    parser.add_argument("--run-name", default=None,
                        help="Subdirectory name. Default: v2-rfdetr-s-<UTC-timestamp>")
    parser.add_argument("--epochs", type=int, default=100,
                        help="rfdetr quick-start default (100)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="rfdetr docs recommend batch_size*grad_accum_steps=16; "
                             "4x4 fits a 24 GB RTX 4090 comfortably")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512,
                        help="Square input edge; RFDETRSmall native 512. Must "
                             "be divisible by 32 (patch_size*num_windows)")
    parser.add_argument("--grid-region", default=DEFAULT_REGION)
    parser.add_argument("--early-stopping", action="store_true",
                        help="Pass early_stopping=True through to rfdetr")
    parser.add_argument(
        "--policy-ack", action="store_true",
        help="REQUIRED. Asserts the five decision-framework questions from "
             f"{POLICY_DOC} have been answered in the PR description: "
             + " ".join(POLICY_QUESTIONS),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.policy_ack:
        print(
            "refusing to start: --policy-ack missing.\n"
            f"Per {POLICY_DOC}, a training run is a new-model proposal and "
            "must answer the decision framework (answers go in the PR "
            "description):\n  " + "\n  ".join(POLICY_QUESTIONS),
            file=sys.stderr,
        )
        return 2

    run_name = args.run_name or (
        f"v2-rfdetr-s-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )

    if args.data_yaml is not None:
        coco_out = args.coco_out or (args.output_dir / f"{run_name}-dataset")
        log.info("converting YOLO layout %s -> COCO layout %s", args.data_yaml, coco_out)
        dataset_dir = yolo_to_coco(args.data_yaml, coco_out)
    else:
        dataset_dir = args.dataset_dir
        if not (dataset_dir / "train" / "_annotations.coco.json").exists():
            log.error("%s does not look like a COCO-layout dataset "
                      "(missing train/_annotations.coco.json)", dataset_dir)
            return 2

    train(
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        run_name=run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution,
        grid_region=args.grid_region,
        early_stopping=args.early_stopping,
        notes=[f"host={platform.node()}", f"py={sys.version.split()[0]}"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
