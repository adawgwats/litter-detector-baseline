# Claude Code kickoff — RF-DETR V2 training run (RTX 4090 box)

Paste this file's contents as the first prompt to Claude Code on the Windows
box (or just say: "Read docs/CLAUDE_PC_KICKOFF.md and execute it").

---

You are on Andrew's Windows machine with an RTX 4090. Your job is to execute
`docs/rfdetr_runbook.md` in this repo top to bottom: environment → dataset →
train RF-DETR Small → eval → export → upload. The runbook is the authority
for exact commands; this brief adds the constraints and judgment calls.

## Hard constraints

- `rfdetr==1.9.0` is pinned. The train/export API facts in the V2 scripts
  were verified against 1.9.0 docs only — if pip resolves anything else,
  stop and re-verify `train()`/`export()` signatures before proceeding.
- Install CUDA torch (cu124 index) BEFORE rfdetr, per the runbook, or you
  get CPU wheels.
- `train_rfdetr_v2.py` refuses to run without `--policy-ack`. That is
  deliberate (energy policy). Acknowledge it and fill the five-question
  template at the bottom of the runbook into the run notes.
- The canonical 43-class alphabetical order is load-bearing end to end:
  `export_rfdetr.py` will refuse a checkpoint whose class order deviates.
  The `--data-yaml` dataset route is canonical by construction — prefer it.
- Windows: keep dataloader `workers=4` (Ultralytics' default 8 exhausts the
  pagefile on this box — learned during the V1 run).

## Dataset route decision

If there is no Roboflow API key / project on this machine yet, do NOT block:
use the local route — `training/data/prepare_dataset.py` (TACO fetch +
crosswalk) and pass the result via `--data-yaml`. If R2 credentials for the
hard-negatives corpus are absent, a TACO-only first run is acceptable and
even defensible: the 2.6k hard-negative corpus was measured at ~17x over
Ultralytics' 0-10% background-image guidance, so run 1 without them is fine.
Note whichever route you took in the run notes.

## Execution discipline

- Launch training as a background task and monitor it; report early signal
  (GPU name, VRAM in use, first-epoch loss trending down) within the first
  few minutes so a bad config dies fast, then check in periodically rather
  than blocking.
- Expect multi-hour wall clock. Do not kill a healthy run to "try settings";
  the smallest-viable-first rule is part of the energy policy.
- After training: run `eval_v2.py` for BOTH models (the deployed YOLO11n
  ONNX baseline and the new checkpoint) at BOTH granularities (43-way and
  `training/rollups/coarse10.yaml`). If a Roboflow key is available, also
  eval on the RF100-VL TACO split per the runbook; otherwise note it as
  pending.
- Then `export_rfdetr.py` (ONNX + `.meta.json` sidecar + gzip). The sidecar
  must carry `architecture: rfdetr` — the backend decode branch keys on it.
- Keep `energy_receipt.json` and both `eval_receipt.json` files in the run
  directory — they are the source of truth for the writeup's eval table;
  nothing gets quoted from terminal memory.

## Upload + handback

- If AWS credentials are configured here, upload per the runbook to the
  models bucket under `litter-detector/v2.0.0-fp16/`. If not, skip upload,
  say so, and leave the artifacts in the run directory for transfer.
- Do NOT push any git changes, do NOT call any Roboflow write APIs beyond
  dataset download, do NOT deploy anything. Training artifacts + receipts
  only.
- Finish by printing a summary block: dataset route used, image/instance
  counts, epochs, wall clock, the headline eval numbers (both models, both
  granularities), artifact paths, and anything that deviated from the
  runbook. Andrew relays that back to the main planning session on his Mac.
