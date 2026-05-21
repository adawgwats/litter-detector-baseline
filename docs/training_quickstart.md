# Training quickstart

**Status**: scaffolding. The infra decisions are locked but several
ingest modules + the training loop itself still need to be written.
This doc is the entry point for the "start training V1" conversation.

**Goal**: see `docs/contributor_assist_goal.md` for what we're training
toward and why. Read that first.

---

## Prerequisites (one-time setup)

### 1. AWS credentials

You need IAM permissions for:
- `secretsmanager:GetSecretValue` on the OLM secret
  (`arn:aws:secretsmanager:us-east-1:910757112705:secret:dregsbane/olm/ingest-bot-*`)
- `s3:*` on the training-data bucket (TBD — provisioned by the
  `feat/cloudflare-pulumi` branch of `dregsbane-web-cdk`)
- `ec2:*` on the spot launch templates (also from the Pulumi stack)

For local dev, `aws sso login` against the DregsBane account is
sufficient.

### 2. Pulumi-up the R2 + EC2 stacks

The decision is locked (`reference_ml_training_infra.md`, 2026-05-18)
but `pulumi up` hasn't been run on `dregsbane-web-cdk/feat/cloudflare-pulumi`
yet. First step is:

```bash
cd ../dregsbane-web-cdk
pulumi stack select beta
pulumi up
```

Expected resources:
- R2 bucket: `dregsbane-ml-data` (Cloudflare, egress-free for training)
- EC2 spot launch template: `dregsbane-ml-g5xlarge-spot`
- IAM roles for the training job
- Secrets Manager grants on the OLM secret

### 3. Python env

```bash
cd litter-detector-baseline
python -m venv .venv
.venv/Scripts/activate    # Windows; or `source .venv/bin/activate` on macOS/Linux
pip install -e ".[ingest,dev]"
```

### 4. Smoke-test OLM auth

```bash
python -m litter_detector_baseline.ingest.olm_auth
# Should print: OLM auth OK as adawgwats@gmail.com (clusters endpoint returned N features)
```

If it fails, see `docs/olm-auth-howto.md` § Failure modes.

---

## V1 training corpus — what to ingest first

Per `docs/contributor_assist_goal.md`, the V1 corpus is non-OLM-primary
(OLM bulk is gated on partnership outreach). Land these in order — each
unblocks the next:

1. **TACO** (`taco.py`, ⏳ todo) — 1,500 images, CC-BY-4.0, COCO bboxes
   - Source: <https://github.com/pedropro/TACO>
   - The highest-quality labeled set we can use without auth
   - ~60 classes; needs mapping to OLM leaf taxonomy (see `litter-taxonomy/`
     repo for the crosswalk schema)

2. **Drinking Waste Classification** (Kaggle, CC0, ⏳ todo) — 9K images,
   bottles/cans/glass
   - Source: <https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification>
   - Clean labels for 3 of the V1 buckets
   - Add a `drinking_waste.py` ingest module mirroring `openlittermap.py`

3. **ShitSpotter** (`shitspotter.py`, ⏳ todo) — 9K images, CC-BY-4.0,
   BAN protocol (before/after/negative)
   - Source: <https://erotemic.github.io/shitspotter/>
   - Adds the structural prior that proves cleanup happened
   - Only one class but the BAN protocol is the load-bearing piece

4. **UAVVaste** (⏳ todo) — 772 aerial images, CC-BY-4.0, hard-negatives
   included
   - Source: <https://github.com/PUTvision/UAVVaste>
   - Aerial perspective; useful even though contributor photos are
     ground-level — the hard-negatives are universally applicable

5. **Hard-negatives** (`hard_negatives.py`, ⏳ todo) — ~500-1000 curated
   images of leaves, mulch, shadows, painted asphalt
   - No off-the-shelf source; scrape + hand-curate from CC0 stock photo
     sites + Google Open Images
   - Per `docs/contributor_assist_goal.md` § Success metrics, target FPR
     ≤ 5% on this corpus before V1 ships

6. **OpenLitterMap** (auth available, deferred to V1.1+) — see
   `docs/olm-auth-howto.md` for the auth flow. Adds 525K images at
   inference-time-matching label distribution. Run only AFTER V1 ships
   on the 13-15K corpus so we have a clean baseline to compare against.

---

## Label crosswalk

We have THREE relevant taxonomies:

1. Each source dataset's own labels (TACO has 60 classes, Drinking
   Waste has 4, UAVVaste has a binary, etc.)
2. OLM leaf taxonomy (~200 classes) — our inference-time output format
3. The 10-bucket display schema from `litter-taxonomy/`

The training model outputs (2). The Trash Trail map consumes (3) via a
mechanical rollup. The label crosswalk maps (1) → (2). It does NOT need
to map (1) → (3) — that's a downstream concern.

Crosswalk file format: CSV with three columns —
`source_dataset,source_label,olm_leaf_label`. Lives at
`configs/label_crosswalk.csv` (not yet committed). Each ingest module
loads this file and re-maps its labels on the way to S3.

---

## Training loop

The `training/` directory is currently empty. The first training PR
should add:

```
training/
├── train_yolo11n_v1.py       # entry point
├── configs/
│   └── contributor_assist_v1.yaml
└── data/
    └── manifest_loader.py     # loads from R2 / S3 + applies label crosswalk
```

Recommended baseline:

```bash
# In an activated venv on EC2 g5.xlarge spot
ultralytics yolo train \
  model=yolo11n.pt \
  data=training/configs/contributor_assist_v1.yaml \
  epochs=80 \
  imgsz=640 \
  batch=32 \
  optimizer=AdamW \
  lr0=0.001 \
  device=0 \
  project=/mnt/r2/dregsbane-ml-data/runs \
  name=v1-yolo11n-$(date +%Y%m%d-%H%M)
```

Expected runtime on g5.xlarge: ~30-90 minutes for 80 epochs over ~13K
images. Spot interruptions handled by Ultralytics' built-in checkpoint
resume.

**Energy receipt** (per `dregsbane-web-trail/docs/ai-energy-policy.md`):
emit a JSON receipt at the end of every training run capturing GPU type,
GPU-hours, estimated kWh, CO₂eq with grid factor, peak RAM, total bytes
of training data ingested. Stored alongside the checkpoint at
`/mnt/r2/dregsbane-ml-data/runs/<run_name>/energy_receipt.json`. The
receipt schema lives at... well, also still ⏳ — a future small PR.

---

## Eval

Per `docs/contributor_assist_goal.md` § Success metrics:

| Metric | Target | Measured by |
|---|---|---|
| Top-3 acceptance rate | ≥ 60% (V1.0) → ≥ 75% (V1.2) | A/B against held-out user submissions |
| Item edit distance | ≤ 1.5 items / submission | Same |
| mAP@0.5 on TACO test | ≥ 0.55 | Ultralytics built-in eval |
| Hazards-class recall | ≥ 0.85 | Custom eval against TACO + hand-curated hazard set |
| Hard-negative FPR | ≤ 5% (leaves, mulch, shadows) | Run model on hard-negative corpus, count any detection |
| Hallucination rate | ≤ 5% | Run model on photos with known empty regions, count detections in empty regions |

The `litter-benchmark-harness` sibling repo holds the eval framework.
Custom evaluators for the metrics above are ⏳ todo per that repo's
README.

---

## Deployment to Trash Trail

The trained `.pt` → ONNX conversion already works via
`export/export_yolov8.py`. The output ONNX gets uploaded to the Trash
Trail S3 bucket as `data/models/contributor-assist-v{N}.onnx`. The
PWA's service worker fetches it on first load.

Frontend integration (not in this repo): adds `runInference(blob)` to
`src/components/PhotoCapture.tsx` in the `dregsbane-web-trail` repo +
new `<ItemEditList>` component. The trail repo's `docs/pile-rendering-spec.md`
will need a v1.7 update once the contributor-assist UI lands.

---

## First commits — concrete sequencing

1. **`scripts/init_db.py`** (~half day) — provisions the SQLite/Postgres
   table that holds the label crosswalk. Reads `configs/label_crosswalk.csv`.
2. **`src/litter_detector_baseline/ingest/taco.py`** (~1 day) — implement
   TACO ingestion mirroring `openlittermap.py`. Loads bboxes, applies
   crosswalk, writes to S3/R2.
3. **`src/litter_detector_baseline/ingest/drinking_waste.py`** (~half day) —
   simpler; just classification labels.
4. **`src/litter_detector_baseline/ingest/hard_negatives.py`** (~1 day) —
   curates the hard-negative corpus (manual seed list + auto-augmentation).
5. **`training/train_yolo11n_v1.py`** (~1 day) — Ultralytics 5-liner +
   energy receipt emission + Wandb/local logging.
6. **`training/eval_v1.py`** (~half day) — runs the 6 eval metrics above,
   emits a `eval_receipt.json` next to the model checkpoint.
7. **Frontend wire-up in trail repo** (~2 days) — separate PR in
   `dregsbane-web-trail`. ONNX runtime web integration, `<ItemEditList>`
   component, hook to existing PhotoCapture flow.

Total to first end-to-end V1: ~6-7 days from now, assuming no
infrastructure surprises and TACO downloads cleanly.

---

## Things explicitly out of scope for V1

- Federated learning (deferred per `ai-energy-policy.md`)
- VLM/captioning models (Grounding DINO, PaliGemma) — kept on the
  research backlog as the v2 escalation if YOLO can't reach the
  edit-distance target
- CrustBot operator-mode model — separate codebase, separate goal
  (context-aware "pickup/skip" decisions). Do NOT confuse with V1
  contributor-assist work.
- Real-time training-data flywheel (i.e., submissions trigger immediate
  re-training). V1 retrains in batch on (suggested, edited) pairs once
  a corpus has accumulated; the cadence is corrections-driven, not
  calendar-driven, per the AI energy policy.
