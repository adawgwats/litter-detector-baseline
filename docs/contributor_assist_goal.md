# V1 model goal — contributor-assist for OLM submissions

**Status**: definition locked 2026-05-20. Operationalizes the goal-reframe
we converged on the day before. Required reading before reviewing any
training PR.

---

## What the V1 model does

Given a photo a contributor just took, output an **itemized list** of
trash visible in the photo, in OLM's leaf taxonomy:

```
photo_in:  <bytes>
photo_out: [
  { "label": "cigarette_butt",   "count": 3,  "bbox": [...] },
  { "label": "plastic_bottle",   "count": 1,  "bbox": [...] },
  { "label": "food_wrapper",     "count": 2,  "bbox": [...] }
]
```

The user sees the list, edits it (adds missing items, removes false
positives, adjusts counts), and submits. The (suggested, user-edited)
pair is captured for the next training run.

## What V1 explicitly does NOT do

- **Does not detect leaves, mulch, shadows, or other non-litter objects.**
  Leaves are explicitly hard-negatives in V1. If a user wants to record
  leaves clogging a storm drain, they add it manually post-suggestion.
  Context-aware "leaves are sometimes cleanup-worthy" classification is
  V2+ operator-mode work, NOT V1 contributor-assist work.
- **Does not produce action recommendations.** V1 says "here is what
  is in the photo." It does not say "pick up" or "skip." That's
  CrustBot operator-mode (V2+).
- **Does not validate or auto-submit.** The user is the final authority
  on what they submit. The model's role is to reduce typing.
- **Does not detect novel items outside OLM's taxonomy.** If something
  is not in OLM's ~200-class leaf taxonomy, the user adds it as a
  custom tag through OLM's existing flow.

## Why this goal, not a map-feature classifier

Earlier scoping treated the V1 model as a classifier feeding the Trash
Trail map (the 5-bucket display schema, the painterly mountain renderer).
That goal is reachable BUT it's the wrong V1 because:

1. **The user-facing value of model accuracy is highest on the
   contribution path.** Map display is read-mostly — users see a tinted
   mountain regardless of whether the underlying classification is
   85% or 92% accurate. Submission flow is write-once — every
   model-suggested item the user accepts directly saves them ~5
   seconds of tag-selection time. Aggregate user time saved is the
   product metric, and contributor-assist optimizes it directly.

2. **The contribution loop generates supervised training data.**
   Map-display classification is a one-way pipeline (raw photos →
   classes → display). Contributor-assist is a closed loop: every
   user correction is a hand-validated label for the next training
   round. After 1000 submissions we have 1000 free hand-corrected
   labels. After 10K we have a proprietary dataset competitive with
   the gold-standard hand-labeled corpora.

3. **OLM contributors are the audience that has explicitly stated
   pain with manual tagging.** Per OLM's own community feedback
   (verified 2026-05-18), the 200-class tag picker is the single
   biggest friction in the OLM contribution flow. A model that even
   imperfectly populates the suggestion eliminates the dominant
   pain point.

4. **The map can still consume the same model outputs.** Once V1 ships
   we can re-run the model over the OLM corpus (once we have auth) and
   feed the per-photo itemizations into the Trash Trail map's pile-
   composition computation. The map gets the V1 model's output for
   free; the inverse is not true.

## Success metrics

**Primary (product):**
- **Top-3 acceptance rate**: % of user submissions where the user
  accepted the top-3 suggested items unchanged. Target ≥ 60% by
  V1.0; ≥ 75% by V1.2 (after first feedback loop).
- **Item edit distance**: average per-submission Levenshtein-like
  distance between (suggested_items) and (final_user_items).
  Target ≤ 1.5 items per submission by V1.0.
- **Median time-to-submit**: from photo capture to submit button.
  Target ≤ 25s by V1.0 (vs. 50-90s for OLM's manual flow today,
  per their own UX research).

**Secondary (technical):**
- **mAP@0.5** on TACO test split. Target ≥ 0.55 (YOLO11n COCO
  baseline is around this for general-purpose detection; litter
  classes are a subset).
- **Per-class recall on Hazards** items (sharps, sanitary,
  biohazard). Target ≥ 0.85. False-negative on a sharp is operationally
  worse than false-positive on a bottle, so we weight Hazards higher.
- **Hard-negative precision** on leaves / mulch / shadows / painted
  asphalt corpus. Target ≥ 0.95 (i.e., FPR ≤ 5%). The literature flagged
  958 false-positives-per-image from leaves; we will not ship a
  V1 that regresses on this baseline.
- **Hallucination rate**: items in model output not actually visible
  in image. Target ≤ 5%. Specific failure mode for VLM-style outputs;
  YOLO-style detection should be at-or-near zero naturally.

## What the model is trained on

See `docs/ingestion.md` for the per-source ingestion status. V1
training corpus (when ready):

| Source | Status | Adds |
|---|---|---|
| TACO | ⏳ todo (`taco.py`) | 1,500 images, COCO-format bboxes, 60 classes |
| Drinking Waste Classification | ⏳ todo (Kaggle CC0) | 9K images, bottles/cans/glass labels |
| ShitSpotter | ⏳ todo (`shitspotter.py`) | 9K images, BAN protocol |
| UAVVaste | ⏳ todo | 772 aerial images with hard-negatives |
| Hard-negatives | ⏳ todo (`hard_negatives.py`) | ~500-1000 curated leaves/mulch/shadow images |
| **OpenLitterMap** | auth available | 525K images. Blocked on bulk-enumeration access; see `docs/olm-auth-howto.md`. |

V1 training starts on the 13-15K non-OLM corpus when the first four
ingest modules land. OLM joins V1.1+ once bulk access is sorted (either
via partnership outreach or by iteratively pulling /api/points bbox
windows under the bot session).

## Deployment target

Per the AI energy policy and the contributor-assist UX, V1 runs
**client-side in the Trash Trail PWA**:

- ONNX YOLO11n (~6 MB), exported via `export/export_yolov8.py`
- Loaded by the PWA service worker, cached after first download
- Inference via ONNX Runtime Web (WASM backend, WebGPU when available)
- Output flows into `<ItemEditList>` (Trash Trail frontend, not in
  this repo) which renders the suggestion list with edit affordances
- Final submission goes through the existing `submission-create`
  Lambda; (suggested, edited) pair lands in the intake-clean S3
  bucket for next-round training data

Server-side inference is the fallback if the model grows past
browser-feasible size. Per the AI energy policy that fallback requires
explicit justification — the default is client-side.

## What's locked vs. what's open

**Locked (do not re-debate without strong new evidence):**
- V1 = contributor-assist, not map classifier
- Output format = OLM leaf taxonomy + per-item counts
- Deployment = client-side ONNX
- Training infra = R2 + EC2 spot (`reference_ml_training_infra.md`)
- Hard-negatives mandatory; leaves explicitly rejected in V1
- Hazards weighted higher than other classes in eval
- Contribution loop generates next round's training data

**Open (decide during the first training PR):**
- Exact YOLO11n vs. RT-DETR-S vs. autodistill-grounded-sam baseline
- Quantization timing (INT8 from day 1 or post-launch?)
- Validation split — TACO held-out, or a separate hand-curated set?
- Per-class confidence threshold defaults (per-class recall vs. precision
  tradeoff)
- How (suggested, edited) pairs feed back into training — daily batch
  job? On-demand? Manual review of accumulated diffs?
