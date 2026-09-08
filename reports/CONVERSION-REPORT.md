# Model conversion: what was measured

Measured 2026-08-31 against the RF-DETR-Small litter detector this repo
exports. Everything below is additive — no shipped artifact and no
serving path was modified. The deployed model is unchanged.

## 1. What was measured

**Environment.** macOS 26.5.1, arm64 (Apple silicon), Python 3.12.13,
ONNX Runtime 1.29.0 `CPUExecutionProvider`, onnx 1.22.0,
onnxconverter-common 1.16.0. No GPU, no CUDA, no TensorRT — see §4.

**Artifacts.**

| | path | bytes | sha256 (first 12) |
|---|---|---|---|
| A (reference) | `roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.onnx` | 120,039,167 | `a59fc4174a77` |
| B (candidate) | `dist/parity/rfdetr-s-litter.fp16.onnx` | 63,068,681 | `ccdddc438a3f` |

B is produced from A by `scripts/make_fp16.py`, which calls the
*shipping* exporter's own `export_rfdetr._to_fp16`
(`onnxconverter_common.float16`, `keep_io_types=True`) — plus the repair
described in §2, without which B does not exist as a loadable file.
fp16 is 47.5% smaller.

**Fixture set.** 200 real photographs: 100 cleanup pairs sampled with
seed 0 from the `cleanup-pairs-corpus` gallery-pair manifests
(DeTrashed ×80, TrashLove ×20), each contributing **both** frames — the
`before` frame (litter-bearing, dense, many queries near the operating
point) and the `after` frame (post-cleanup, sparse). Sampling pairs
rather than loose images is deliberate: a one-sided sample would measure
only one of the two failure modes. Manifest:
`reports/parity-fixtures-v1.txt`, regenerable byte-for-byte via
`scripts/build_parity_fixtures.py --seed 0 --pairs 100`.

No random tensors were used. Random inputs put activations in ranges the
network never sees, which both hides real fp16 overflow and manufactures
divergence no user would encounter.

**Reproduce.**

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,parity]"
.venv/bin/python scripts/build_parity_fixtures.py \
    --corpus ../cleanup-pairs-corpus --pairs 100 --seed 0 \
    --out reports/parity-fixtures-v1.txt
.venv/bin/python scripts/make_fp16.py \
    --in ../roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.onnx \
    --out dist/parity/rfdetr-s-litter.fp16.onnx
.venv/bin/python -m export.parity \
    --a ../roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.onnx \
    --b dist/parity/rfdetr-s-litter.fp16.onnx \
    --meta ../roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.meta.json \
    --fixtures reports/parity-fixtures-v1.txt \
    --report-json reports/parity-rfdetr-v2.0.1-fp32-vs-fp16.json
```

---

## 2. Result: the fp16 conversion did not produce a loadable model

Before any parity number existed, the first attempt to load B failed.
The exporter's fp16 path had never been executed end to end — and
`--version v2.0.0-fp16` is the RF-DETR exporter's **default**, so the
default invocation of that script produces an artifact ONNX Runtime
refuses.

Two independent defects in `onnxconverter_common.float16` 1.16.0 on this
graph:

1. **Colliding, degenerate cast nodes.** The converter names inserted
   casts `<node>_input_cast<i>` / `<node>_output_cast<i>`. RF-DETR's own
   torch export had *already* emitted casts around its TopK using that
   exact convention, so the converter regenerated two names that already
   existed. The nodes it emitted were also self-loops — single output
   name equal to single input name — giving those tensors two producers.
   ORT: `two nodes with same node name (/transformer/TopK_input_cast0)`.
   The fp32 source graph has 1,573 nodes, all uniquely named; the
   converted graph has 1,578 with 2 duplicated names.

2. **Orphaned `Cast(to=FLOAT)`.** The source graph carries 1,076 FLOAT
   tensor annotations; the converter flipped all but 2 of them to
   FLOAT16 (the 2 survivors sit either side of the TopK) — but left 35
   explicit `Cast` nodes with `to=FLOAT`, which then produce float into
   positions the graph now declares float16. ORT: `Type (tensor(float16)) ... does not match
   expected type (tensor(float))`.

`export/fp16_repair.py` repairs both, conservatively: it removes a
degenerate cast only when another node already produces that tensor
(so no consumer's value can change), and retypes a `Cast(to=FLOAT)` only
when the output is *not* a graph output — leaving the `keep_io_types`
fp32 boundary exactly where the exporter put it. On this graph: 2 casts
removed, 35 retyped, after which ORT builds a session and both outputs
are finite.

**A third defect, found while investigating.** The standard mitigation
for fp16 divergence is to keep numerically sensitive ops in fp32 via
`op_block_list`. Passing one containing `Div` crashes the converter:
`AttributeError: 'list' object has no attribute 'input'` in
`float16.py:787 remove_unnecessary_cast_node`. Bisected: the implicit
default, the explicit default, and the default plus each of `Sqrt`,
`Softmax`, `Exp`, `Pow`, `ReduceMean`, `ReduceSum`, `Erf`,
`InstanceNormalization`, `LayerNormalization` all convert fine. Only
`Div` triggers it.

### What this licenses

That the fp16 export path in `export/export_rfdetr.py`, as written,
emits an artifact that ONNX Runtime 1.29.0 cannot load — on this graph,
with onnxconverter-common 1.16.0. It does not license a claim about
other architectures, other converter versions, or the YOLO exporter's
fp16 path, which uses a different mechanism (ultralytics `half=True`)
and was not tested here.

It also does not license "the repair is correct." The repair is
*conservative* and the repaired model loads and produces finite outputs,
which is a much weaker statement. §3 is where its behaviour is measured.

---

## 3. Result: parity, fp32 vs repaired fp16

### Raw graph outputs, before any decode

200 images. `dets` is `[1, 300, 4]` box coordinates normalized to
[0,1]; `labels` is `[1, 300, 44]` raw logits (43 classes + 1 trailing
background column).

| output | elements | max abs Δ | mean abs Δ | median abs Δ | p99 abs Δ | max rel Δ |
|---|---|---|---|---|---|---|
| `dets` | 240,000 | 2.1608 | 0.1466 | 0.0505 | 0.8236 | 35,878 |
| `labels` | 2,640,000 | 12.1735 | 0.5003 | 0.3239 | 2.3984 | 79.5 |

Fraction of elements failing each tolerance rule (the rules are *not*
interchangeable, which is why all are reported):

| output | atol/rtol | \|Δ\|>atol | rel>rtol | `isclose` | AND | OR |
|---|---|---|---|---|---|---|
| `dets` | 1e-5 | 99.47% | 99.89% | 99.36% | 99.47% | 99.89% |
| `dets` | 1e-2 | 69.76% | 83.16% | 68.31% | 69.76% | 83.16% |
| `labels` | 1e-5 | 99.99% | 99.95% | 99.94% | 99.95% | 99.99% |
| `labels` | 1e-2 | 93.51% | 74.75% | 73.15% | 74.75% | 93.51% |

`isclose` is `|a-b| > atol + rtol*|a|` — numpy/`allclose` semantics, one
blended criterion, **not** a conjunction. `AND` requires an element to
be bad on both scales; `OR` requires either. The three differ by up to
20 percentage points on the same data at the same numbers, which is why
"N elements exceeded 1e-5" is not a reportable sentence on its own.

### Decoded detections, at the operating point

Decode mirrors the backend's `postprocess-detr.ts` exactly: argmax over
the 43 named columns only, sigmoid of the winner, threshold, top-k at
`maxDetections=100`, no NMS. Because both artifacts are the same graph,
query slot *q* means the same thing in each, so the comparison is
query-aligned and exact — no IoU matching, which would blur a threshold
crossing into a "missing detection".

Max |Δconfidence| on any single query: **0.807**. Max |Δbox coordinate|:
**1.0** (the full normalized range). Mean per-image max |Δconfidence|:
0.060.

| threshold | detections A | detections B | threshold flips | class flips | images affected |
|---|---|---|---|---|---|
| 0.25 | 325 | 321 | 36 | 14 | 16 |
| 0.30 | 281 | 276 | 29 | 14 | 16 |
| **0.40** ← sidecar | **202** | **203** | **19** | **6** | **10** |
| 0.50 | 160 | 159 | 15 | 3 | 9 |
| 0.60 | 129 | 130 | 11 | 2 | 8 |

"Threshold flips" = queries surviving in exactly one artifact. "Class
flips" = queries surviving in *both* but whose argmax class differs —
those are mislabels, not count changes, and they are invisible to a
detection-count comparison. Top-k truncation never bound at any
threshold, so `maxDetections` is not participating in these numbers.

**The headline sentence.** Across 200 real images, the max absolute
delta on the raw logits was **12.17** and the max relative delta
**79.5**; at the sidecar's recommended 0.4 threshold, **19** query-level
threshold crossings differed between fp32 and fp16 — against 202
detections in fp32 and 203 in fp16 — with **6** further queries
surviving in both but changing predicted class, affecting **10 of 200**
images.

Note what the near-identical totals conceal: 202 vs 203 looks like
parity. It is 19 disagreements that happen to nearly cancel. A gate that
compared detection counts would have passed this.

### Latency

`session.run()` wall time only, batch 1, 200 runs each, after 3 warmups.

| artifact | p50 | p95 |
|---|---|---|
| fp32 | 311.7 ms | 391.0 ms |
| fp16 | 589.7 ms | 692.4 ms |

fp16 is **1.89× slower**. ONNX Runtime's CPU provider has limited fp16
kernel coverage — it emitted `Could not find a CPU kernel and hence
can't constant fold` for `Sqrt`, `Exp` and `Add` nodes throughout the
backbone and decoder — so the fp16 graph pays conversion overhead
without gaining any fp16 arithmetic.

### 3a. A second recipe, to test whether the recipe is the problem

The obvious suspicion about §3 is that the divergence is an artifact of
*this particular* fp16 recipe — specifically of the repair in §2, which
forced 35 casts to fp16 that the converter had left targeting fp32. So a
second artifact was built with a different precision boundary:
`op_block_list` extended with `Sqrt`, `Softmax` and `Exp`, keeping those
numerically sensitive ops in fp32. (`Div` could not be added — it
crashes the converter, §2.) Same repair applied, same fixture set, same
runtime.

| | `labels` max abs Δ | `labels` mean abs Δ | flips @0.4 | class flips @0.4 | max \|Δconf\| | p50 latency |
|---|---|---|---|---|---|---|
| default recipe | 12.17 | 0.5003 | 19 | 6 | 0.807 | 589.7 ms |
| Sqrt/Softmax/Exp in fp32 | 10.77 | 0.5026 | 19 | 4 | 0.729 | 684.4 ms |

Moving those three op families back to fp32 did **not** meaningfully
reduce divergence: identical threshold-flip count, mean absolute delta
unchanged to three decimal places, max delta down 11% on a statistic
that is a single worst element out of 2.64 million. It cost a further
16% of latency.

Measurement noise, for scale: artifact A is the same file in both runs
and its own p50 moved 311.7 → 330.2 ms (6%) between them. The 1.89×
fp32→fp16 slowdown is far outside that; the 11% max-delta difference
above is not obviously outside it.

**What this rules out, and what it does not.** It rules out the
hypothesis that fp16 overflow in the attention's `Sqrt`/`Softmax`/`Exp`
is driving the divergence — two recipes with genuinely different
precision boundaries land in the same place. It does **not** clear the
§2 repair: both artifacts received the same 35 retyped casts, so that
remains a common factor neither run isolates. Separating it would mean
building a variant that keeps those 35 casts in fp32 by construction
rather than by repair, which the converter's `op_block_list` crash makes
awkward and which was not attempted.

### What this licenses

*On this fixture set, this runtime, and this execution provider*: the
fp16 conversion changes served output. It is not a rounding-level
difference — a max confidence shift of 0.81 on a single query is a
detection appearing or vanishing, and 6 class flips at the operating
point are wrong labels.

It licenses "on this input distribution, divergence was this large."
It does **not** license "fp16 is unsafe for this model." Specifically:

- Two recipes were measured (§3a) and agree, which rules out the
  attention's `Sqrt`/`Softmax`/`Exp` as the cause but leaves the §2
  repair's 35 retyped casts common to both and therefore untested as a
  cause.
- 200 images from three subreddits is not the production distribution.
- Nothing here separates "fp16 arithmetic is lossy for this
  architecture" from "onnxconverter-common converts this graph badly."
  Those have different fixes.

*On latency*: the 1.89× figure describes ONNX Runtime CPU on Apple
silicon. It says nothing about Lambda's arm64 Graviton CPU (where the
model actually serves), and nothing whatsoever about a GPU or an
embedded target, where fp16 normally *is* a speedup because the hardware
has fp16 units and the runtime has kernels for them.

---

## 4. What was attempted and could not be done

**TensorRT engine build, Polygraphy comparison, INT8 PTQ, build
non-determinism measurement — not attempted. No GPU was available.**

This work ran on macOS/arm64. `tensorrt` and `polygraphy` are not
installable here and there is no CUDA device to build an engine for. The
questions those tasks would answer — does the same ONNX produce
byte-identical plans across builds, does a timing cache change that
answer, does the engine agree with the ONNX and within what bound, what
is an engine's full identity tuple — remain **unanswered**. Nothing in
this report should be read as bearing on them.

That work needs the RTX box. It is worth noting that the parity harness
built here is the part that would be reused there: comparing an engine
against its source ONNX is the same measurement with a different
`InferenceSession`, and `export/parity.py` takes its execution provider
as a parameter.

**Also not done:** unifying the two exporters' precision paths. They
still use two different fp16 mechanisms with two different IO
boundaries — `export_rfdetr.py` via `onnxconverter_common` with
`keep_io_types=True` (an explicit decision), `export_yolov8.py` via
ultralytics `{'half': True}` (where the boundary is not a decision at
all). §2 is an argument for doing that work, since the RF-DETR path's
defects were invisible partly because nothing shared code with a path
that gets exercised.

**Pre-existing, untouched:** `tests/test_smoke.py` has two failures
unrelated to this work — `assert det.score == 0.9` against a float32
round-trip that yields 0.8999999761581421. They were failing before any
change here and are left alone.

---

## 5. What would be needed to make any of this a release gate

None of this is a gate today. `export/parity.py` reports and exits 0.
To make it gate, in rough order of what each step buys:

1. **A fixture set that represents production.** 200 images from three
   subreddits is a convenience sample of cleanup photography. The
   deployed model sees user uploads from the Trail app. Until the
   fixture set is drawn from that distribution, a parity number
   describes Reddit, not production.

2. **A threshold placed by a human, on this evidence.** The harness
   deliberately does not pick one. Someone has to decide whether 19
   threshold flips and 6 class flips per 200 images is acceptable for a
   contributor-assist flow where a human confirms every suggestion. That
   is a product judgement about a review queue, not a numerical fact,
   and the argument for 0.4 in the sidecar ("a missed detection costs
   more than an extra suggestion") is the right frame for making it.

3. **A stored baseline per artifact pair, versioned with the artifact.**
   `--baseline` already gates on regression. What is missing is a home
   for the baseline: today it is a file in `reports/`, related to the
   artifact only by filename. It belongs on the release record next to
   the provenance tuple, as a required field rather than a report filed
   elsewhere.

4. **The gate in CI, on a machine with the model.** The 200-image run
   takes ~3.5 minutes per artifact pair on CPU here. The artifacts are
   60–120 MB and live in S3, so this is a nightly or release-time job,
   not a per-commit one.

5. **`validate_loadable` promoted into `_to_fp16` itself.** §2's first
   defect would have been caught the day the fp16 path was written by
   one ORT session construction. `onnx.checker` alone would not have —
   it accepts the graph ORT rejects. This is the cheapest change in this
   document and the one with the clearest payoff.

## 6. What shipped

| path | what it is |
|---|---|
| `export/parity.py` | The harness. Raw-tensor + operating-point comparison, JSON + human report, regression gate. |
| `export/fp16_repair.py` | Repairs the two converter defects that make the fp16 artifact unloadable; `validate_loadable` is the real check. |
| `scripts/make_fp16.py` | Produces an fp16 artifact via the shipping `_to_fp16`, then repairs and validates it. |
| `scripts/build_parity_fixtures.py` | Deterministic fixture manifest from the cleanup-pairs corpus. |
| `tests/test_export_rfdetr.py` | 18 tests. `_check_class_alignment` now demonstrably raises on reordering, warns on a missing config, and is bypassable only via the explicit flag. |
| `tests/test_parity.py` | 21 tests over the decode mirror, the manifest reader, the regression gate, and the repair. |
| `reports/parity-rfdetr-v2.0.1-fp32-vs-fp16.json` | The machine-readable result behind §3. |
| `reports/parity-rfdetr-v2.0.1-fp32-vs-fp16-blocked.json` | The second recipe, §3a. |
| `reports/parity-fixtures-v1.txt` | The fixture manifest, seed 0. |

`pyproject.toml` gains a `parity` optional extra. Nothing was added to
the required dependency set. No serving path, no shipped artifact, and
no exporter behaviour was changed.
