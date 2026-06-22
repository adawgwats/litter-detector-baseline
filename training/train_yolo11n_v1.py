"""V1 training entry point: YOLO11n fine-tune on prepared TACO + HN dataset.

Per docs/contributor_assist_goal.md:
  - Model: YOLO11n (smallest viable detection model per energy policy)
  - imgsz: 640 (matches OLM ingest resize default; will revisit at V1.1)
  - Pretrained: yes (locked decision — no training from scratch)
  - Optimizer: AdamW (Ultralytics-default), lr0=0.001

Per docs/ai-energy-policy.md every run emits an energy receipt:
  - GPU type, total GPU-hours, estimated kWh, estimated CO2eq (grid factor)
  - Peak RAM, total bytes of training data ingested
  - Stored alongside the checkpoint as ``energy_receipt.json``.

The receipt's numbers are *estimates*. GPU TDP is the published nominal,
not measured power draw; kWh is GPU-hours * TDP; CO2 uses a per-region
grid factor (us-east-1 fallback). Pin "estimate" in any external claim.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ─── GPU TDP table (W) for kWh estimation ─────────────────────────────────
#
# Values are vendor-published nominal TDP. Used only for energy-receipt
# estimation; not load-bearing for training itself.
GPU_TDP_W: dict[str, int] = {
    "NVIDIA A10G":         150,   # g5.xlarge
    "NVIDIA A10":          150,
    "NVIDIA A100-SXM4-40GB": 400,
    "NVIDIA A100-SXM4-80GB": 400,
    "NVIDIA H100":         700,
    "NVIDIA L4":            72,
    "NVIDIA L40S":         350,
    "NVIDIA T4":            70,
    # Consumer cards (for local dev)
    "NVIDIA GeForce RTX 4070":      200,
    "NVIDIA GeForce RTX 4070 Ti":   285,
    "NVIDIA GeForce RTX 4080":      320,
    "NVIDIA GeForce RTX 4090":      450,
    "NVIDIA GeForce RTX 3060":      170,
    "NVIDIA GeForce RTX 3070":      220,
    "NVIDIA GeForce RTX 3080":      320,
    "NVIDIA GeForce RTX 3090":      350,
    "NVIDIA RTX 4090":              450,
    "NVIDIA RTX 3090":              350,
}

# ─── Per-region grid CO2 intensity (g CO2eq / kWh) ────────────────────────
#
# Approximations from public sources (AWS sustainability + EIA). Updated
# 2026-05. Not load-bearing for energy policy compliance — just for receipt.
REGION_GRID_G_CO2_PER_KWH: dict[str, int] = {
    "us-east-1":  370,   # Virginia mix
    "us-east-2":  500,   # Ohio (coal-heavier)
    "us-west-2":  130,   # Oregon (hydro-heavy)
    "eu-west-1":  300,   # Ireland
    "eu-north-1":  40,   # Sweden (very clean)
}
DEFAULT_REGION = "us-east-1"
DEFAULT_GRID_G_CO2_PER_KWH = 370


@dataclass
class EnergyReceipt:
    run_name: str
    started_at_utc: str
    finished_at_utc: str
    wall_seconds: float
    gpu_type: Optional[str]
    gpu_count: int
    gpu_tdp_w_per_unit: Optional[int]
    estimated_gpu_hours: float
    estimated_kwh: float
    grid_region: str
    grid_g_co2_per_kwh: int
    estimated_g_co2eq: float
    peak_ram_mb: Optional[float]
    n_train_images: int
    n_val_images: int
    total_train_bytes: int
    epochs: int
    imgsz: int
    batch: int
    model: str
    optimizer: str
    lr0: float
    notes: list[str] = field(default_factory=list)


def _detect_gpu() -> tuple[Optional[str], int]:
    """Return (gpu_type_name, gpu_count) by parsing nvidia-smi. Returns
    (None, 0) on systems without an NVIDIA GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8").strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, 0
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if not lines:
        return None, 0
    return lines[0], len(lines)


def _peak_ram_mb() -> Optional[float]:
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process(os.getpid()).memory_info().rss / 1_000_000
    except ImportError:
        return None


def _read_prepare_stats(dataset_dir: Path) -> dict:
    p = dataset_dir / "prepare_stats.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def train(
    *,
    data_yaml: Path,
    output_dir: Path,
    run_name: str,
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 32,
    model: str = "yolo11n.pt",
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    grid_region: str = DEFAULT_REGION,
    workers: int = 4,
    notes: Optional[list[str]] = None,
) -> EnergyReceipt:
    """Drive a YOLO training run + emit an energy receipt next to the
    resulting checkpoint."""
    from ultralytics import YOLO  # noqa: PLC0415 — heavy import, lazy

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_mono = time.monotonic()
    gpu_type, gpu_count = _detect_gpu()
    log.info("starting %s on %d x %s", run_name, gpu_count, gpu_type or "(no GPU)")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
        workers=workers,
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
    )

    wall_seconds = time.monotonic() - started_mono
    finished_at = datetime.now(timezone.utc)

    tdp = GPU_TDP_W.get(gpu_type) if gpu_type else None
    estimated_gpu_hours = (wall_seconds / 3600.0) * max(1, gpu_count)
    estimated_kwh = (estimated_gpu_hours * (tdp or 0)) / 1000.0
    grid_factor = REGION_GRID_G_CO2_PER_KWH.get(grid_region, DEFAULT_GRID_G_CO2_PER_KWH)
    estimated_g_co2eq = estimated_kwh * grid_factor

    stats = _read_prepare_stats(data_yaml.parent)

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
        n_train_images=stats.get("n_train", 0),
        n_val_images=stats.get("n_val", 0),
        total_train_bytes=stats.get("total_bytes_pulled", 0),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        model=model,
        optimizer=optimizer,
        lr0=lr0,
        notes=notes or [],
    )

    # Ultralytics writes to output_dir/run_name/. Drop the receipt there.
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "energy_receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2), encoding="utf-8"
    )
    log.info("wrote energy_receipt.json to %s", run_dir)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True,
                        help="Path to the data.yaml written by prepare_dataset.py")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"),
                        help="Where Ultralytics writes the run + checkpoint")
    parser.add_argument("--run-name", default=None,
                        help="Subdirectory name. Default: v1-yolo11n-<UTC-timestamp>")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--grid-region", default=DEFAULT_REGION)
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default 4; Ultralytics' "
                             "default 8 spawns enough torch processes to "
                             "exhaust Windows pagefile)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_name = args.run_name or f"v1-yolo11n-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    train(
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        run_name=run_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model=args.model,
        optimizer=args.optimizer,
        lr0=args.lr0,
        grid_region=args.grid_region,
        workers=args.workers,
        notes=[f"host={platform.node()}", f"py={sys.version.split()[0]}"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
