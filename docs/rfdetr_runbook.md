# RF-DETR V2 runbook (RTX Windows box)

Exact ordered commands for the V2 RF-DETR Small run: env setup, dataset,
train, eval (both models, both granularities, plus the RF100-VL TACO
split), export, upload. PowerShell throughout. Paths follow the V1
convention (`C:\tmp\...`).

Every training run must pass `--policy-ack` and answer the five
energy-policy questions in the PR description — the template is at the
bottom of this file. Policy: `dregsbane-web-trail/docs/ai-energy-policy.md`.

---

## 1. Environment

Python 3.12 x64. One venv for everything in this runbook.

```powershell
cd C:\dev\litter-detector-baseline
git fetch origin
git checkout feat/rfdetr-v2

python -m venv C:\tmp\venv-rfdetr
C:\tmp\venv-rfdetr\Scripts\Activate.ps1

# CUDA torch FIRST so rfdetr doesn't resolve the CPU wheel.
# cu124 wheels cover the RTX 4070 (sm_89, Ada).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Pinned versions (API facts in the V2 scripts were verified against
# rfdetr 1.9.0 — do not bump casually).
pip install "rfdetr[train,loggers]==1.9.0"
pip install roboflow onnx onnxconverter-common onnxruntime

# The repo package itself (YOLO ONNX inference path used by eval_v2)
pip install -e .[train]

# Sanity: GPU visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 2. Dataset

Two supported sources. Either way the canonical class order is the
43-leaf alphabetical order — the eval and export steps read it from
`data.yaml`, so keep a copy of the V1 `data.yaml` around even when
training from a Roboflow download.

### 2a. Roboflow project download (COCO format)

```powershell
python -c @"
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_ROBOFLOW_API_KEY')            # placeholder
project = rf.workspace('YOUR_WORKSPACE').project('YOUR_PROJECT_SLUG')  # placeholder
dataset = project.version(1).download('coco', location=r'C:\tmp\v2_dataset_coco')
print(dataset.location)
"@
```

Then VERIFY the categories are the canonical 43 in alphabetical order
(Roboflow can reorder / inject a supercategory row). If they are not,
prefer route 2b — a checkpoint trained on a different class order will
be refused by the exporter's class-order guard.

```powershell
python -c @"
import json
cats = json.load(open(r'C:\tmp\v2_dataset_coco\train\_annotations.coco.json'))['categories']
names = [c['name'] for c in cats]
print(len(names), names == sorted(names))
"@
```

### 2b. Convert the V1 YOLO-layout dataset (canonical by construction)

The trainer converts inline when given `--data-yaml` (writes the COCO
copy next to the run), so no separate step is needed. Use this route
when in doubt — it guarantees canonical category order.

## 3. Train

```powershell
python -m training.train_rfdetr_v2 `
  --data-yaml C:\tmp\v1_dataset\data.yaml `
  --coco-out C:\tmp\v2_dataset_coco `
  --output-dir C:\tmp\runs `
  --epochs 100 --batch-size 4 --grad-accum-steps 4 --lr 1e-4 `
  --resolution 512 `
  --grid-region us-east-1 `
  --policy-ack
```

(For a Roboflow-downloaded dataset, replace the first two lines with
`--dataset-dir C:\tmp\v2_dataset_coco`.)

Outputs in `C:\tmp\runs\v2-rfdetr-s-<UTC>\`:
`checkpoint_best_total.pth` (use this one), `training_config.json`,
`energy_receipt.json`. Defaults follow the rfdetr docs: total batch
16 (4 x 4 accumulation), lr 1e-4, 100 epochs; RF-DETR Small is the
Apache-2.0 tier. Budget roughly a day of RTX 4070 wall time at TACO scale;
the energy receipt records what it actually cost.

## 4. Eval — both models, both granularities

Set the run dir once:

```powershell
$RUN = "C:\tmp\runs\v2-rfdetr-s-<UTC>"   # fill in the real timestamp
```

### 4a. Internal val split (leaf43 + coarse10, YOLO + RF-DETR)

```powershell
python -m training.eval_v2 `
  --data-yaml C:\tmp\v1_dataset\data.yaml `
  --yolo-onnx C:\tmp\dist\models\v1\yolo11n-litter.onnx `
  --detr-checkpoint $RUN\checkpoint_best_total.pth `
  --output-dir C:\tmp\eval-runs `
  --run-name v2-internal
```

Writes `C:\tmp\eval-runs\v2-internal\{yolo11n-onnx,rfdetr-s}\{leaf43,coarse10}\eval_receipt.json`
plus `summary.json` (the side-by-side numbers for the PR).

### 4b. RF100-VL TACO split (external COCO val)

```powershell
python -c @"
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_ROBOFLOW_API_KEY')            # placeholder
project = rf.workspace('rf100-vl').project('RF100VL_TACO_PROJECT_SLUG')  # placeholder
dataset = project.version(1).download('coco', location=r'C:\tmp\rf100vl_taco')
"@

python -m training.eval_v2 `
  --data-yaml C:\tmp\v1_dataset\data.yaml `
  --yolo-onnx C:\tmp\dist\models\v1\yolo11n-litter.onnx `
  --detr-checkpoint $RUN\checkpoint_best_total.pth `
  --coco-val C:\tmp\rf100vl_taco\valid `
  --output-dir C:\tmp\eval-runs `
  --run-name v2-rf100vl-taco
```

TACO category names map into the 43-leaf space through
`configs/label_crosswalk.csv`; the log reports any dropped/unmappable
categories — paste that line into the PR too.

## 5. Export (ONNX + meta.json + gzip)

```powershell
python -m export.export_rfdetr `
  --checkpoint $RUN\checkpoint_best_total.pth `
  --data-yaml C:\tmp\v1_dataset\data.yaml `
  --output-dir C:\tmp\dist\models\v2 `
  --model-name rfdetr-s-litter `
  --version v2.0.0-fp16 `
  --resolution 512 `
  --eval-receipt C:\tmp\eval-runs\v2-internal\rfdetr-s\leaf43\eval_receipt.json `
  --energy-receipt $RUN\energy_receipt.json
```

Produces in `C:\tmp\dist\models\v2\`:

- `rfdetr-s-litter.onnx` — fp16 weights, fp32 IO
- `rfdetr-s-litter.onnx.gz` — compression at rest, per energy policy
- `rfdetr-s-litter.meta.json` — sidecar with `architecture: "rfdetr"` +
  ImageNet `normalization`; the backend's DETR decode branch keys on these

The exporter refuses on a class-order mismatch between the checkpoint's
`training_config.json` and `data.yaml` — that refusal is a real defect
in the dataset, not a nuisance; do not reach for `--skip-class-check`.

## 6. Upload

```powershell
aws s3 cp C:\tmp\dist\models\v2\rfdetr-s-litter.onnx      s3://<models-bucket>/litter-detector/v2.0.0-fp16/
aws s3 cp C:\tmp\dist\models\v2\rfdetr-s-litter.onnx.gz   s3://<models-bucket>/litter-detector/v2.0.0-fp16/
aws s3 cp C:\tmp\dist\models\v2\rfdetr-s-litter.meta.json s3://<models-bucket>/litter-detector/v2.0.0-fp16/
```

`<models-bucket>` = the bucket the backend's `MODEL_S3_URI` points at.
Point `MODEL_S3_URI` at
`s3://<models-bucket>/litter-detector/v2.0.0-fp16/rfdetr-s-litter.onnx`
(the sidecar is fetched by convention next to it). Do NOT flip the env
var until the backend's architecture-aware decode branch is deployed.

## 7. Energy-policy answers (paste into the PR description)

Required by `--policy-ack`. Framework from
`dregsbane-web-trail/docs/ai-energy-policy.md`:

```markdown
### AI energy policy — decision framework

1. **Can this be solved without ML at all?**
   <answer>

2. **Can this be solved with a tiny on-device model?**
   <answer — V1 YOLO11n exists; state what the eval receipts show it
   cannot do that justifies the larger model>

3. **If a server model is needed, can it scale to zero?**
   <answer — Lambda serving path, no always-warm endpoint>

4. **Have we measured the actual energy / carbon impact?**
   <answer — paste estimated_kwh / estimated_g_co2eq from
   energy_receipt.json>

5. **Have we considered a foundation-model API call (Claude/GPT) as the
   alternative?**
   <answer>
```
