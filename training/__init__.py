"""V1 training pipeline.

Reads training data from R2 (TACO + hard-negatives ingested by the
litter_detector_baseline.ingest package), converts to YOLO format on
local disk, runs YOLO11n fine-tuning, emits an energy receipt alongside
the checkpoint per docs/ai-energy-policy.md.

Entry points:
  - prepare_dataset.py: pull from R2 + crosswalk + YOLO-format the data
  - train_yolo11n_v1.py: drive the YOLO training loop
  - eval_v1.py: per-class recall, hazards-class recall, hard-negative FPR
"""
