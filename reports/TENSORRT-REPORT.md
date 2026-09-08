# TensorRT conversion: what was measured

Measured 2026-09-08 against the RF-DETR-Small litter detector this repo
exports. Everything is additive — no shipped artifact and no serving path
was modified. The deployed model is unchanged.

This is the GPU continuation of `reports/CONVERSION-REPORT.md`, which
measured precision conversion on CPU. **The two reports do not share a
baseline artifact.** §1.1 explains why, and that turned out to be the
most load-bearing result here.

---

## 1. What was measured

**Builder / runtime host.** Windows 11 (10.0.26200), AMD64,
**NVIDIA GeForce RTX 4070** — compute capability **8.9** (Ada), 46 SMs,
12,878,086,144 B VRAM — driver **591.86**, CUDA runtime **13.1**.
Python 3.12.7, **TensorRT 11.2.1.2**, onnx 1.22.0, polygraphy 0.53.4,
numpy 2.5.3. Parity runs used ONNX Runtime **1.29.0**
`CPUExecutionProvider`; the CUDA-EP latency comparison in §5 required a
separate environment (ONNX Runtime **1.20.2** + CUDA 12.4 libraries) and
is labelled accordingly.

This is a **desktop GPU**. Nothing here describes an embedded target, and
nothing here describes the arm64 Lambda host where this model serves
today.

**Source artifact.** Re-exported on this machine from
`checkpoint_best_total.pth` of run `v2-rfdetr-s-20260802T0348`, via
`export.export_rfdetr --version v2.0.1-fp32 --resolution 512 --opset 17`.

| | bytes | sha256 (first 12) |
|---|---|---|
| source ONNX | 120,111,594 | `c055e87cd710` |

Input `input` `[1,3,512,512]`; outputs `dets` `[1,300,4]`,
`labels` `[1,300,44]` (43 classes + 1 trailing background column).
Sidecar: ImageNet normalization, `confThresholdRecommended` 0.4.

**Fixture set.** 200 real photographs from the model's **held-out val
split** (`v2_dataset_coco/valid`), seed 0: 100 `taco_*` frames
(litter-bearing, dense, many queries near the operating point) and 100
`hn_*` hard negatives (clean/sparse, where a conversion can invent a
detection). Manifest: `reports/parity-fixtures-win-v1.txt`. The
`cleanup-pairs-corpus` used by the CPU report is not present on this
machine, so flip counts here are **not** comparable in absolute terms to
that report's. No random tensors were used anywhere, including in the
Polygraphy runs (§4).

### 1.1 The same checkpoint produced two different ONNX files — on the same machine

The CPU report's baseline (`a59fc4174a77`, 120,039,167 B, 2 Aug) and this
one (`c055e87cd710`, 7 Sep) come from the **same checkpoint**, the **same
exporter**, the **same opset**, the **same producer `pytorch 2.6.0`** —
and, contrary to an earlier reading of this evidence, **the same
Windows host and the same virtualenv**. They are not the same file.

Both artifacts embed the Python stack recorded at trace time in their
node `doc_string` fields, and those frames are absolute local paths.
Counting them: this export carries **60,832** frames rooted at
`C:\tmp\venv-rfdetr` and **4,677** at its repository checkout, and
**zero** POSIX/macOS paths; the 2 Aug artifact carries 60,795 and 4,674
respectively, likewise with none. The CPU report's environment block
describes where its *parity run* executed, not where the export did.

| | `a59fc417` (2 Aug) | `c055e87c` (7 Sep) |
|---|---|---|
| bytes | 120,039,167 | 120,111,594 |
| ir_version / opset | 8 / 17 | 8 / 17 |
| producer | pytorch 2.6.0 | pytorch 2.6.0 |
| nodes | 1,573 | **1,570** |
| initializers | 354 | 354 |
| initializer bytes | 113,931,968 | **113,931,968** |
| distinct op types | 40 | 40 |
| `Cast` nodes | 39 | **36** |
| all other op counts | — | identical |
| node `doc_string` bytes | 5,553,677 | 5,767,539 |
| `value_info` entries | 1,575 | **0** |

So this is not platform variance, which is expected and easy to dismiss.
It is **the same export failing to reproduce on one machine across five
weeks** — the harder problem, and the reason a registry should
content-address the artifact rather than trust the recipe.

The two artifacts differ in **three** unrelated ways, and conflating them
hides the important one.

**(1) Debug metadata churn — cosmetic, and it dominates the byte delta.**
Node `doc_strings` are **4.80%** of this file (5,767,539 B over 1,559 of
1,570 nodes). They are pure tracing debris: absolute source paths and
line numbers. The 213,862 B `doc_string` difference between the two
artifacts is **98.4% accounted for by the checkout having moved** — the
recorded root is 76 characters here against 31 in the older export, and
45 B × 4,677 frames = **210,465 B**. Moving a checkout changes the
artifact's hash.

**(2) Shape-inference metadata — ~138 KB, and only one artifact has it.**
With `doc_strings` stripped from both, the older graph is 114,480,614 B
against this one's 114,342,294 B. The older carries **1,575 `value_info`
entries** — inferred shapes for every intermediate tensor — and this one
carries **zero**. One export ran shape inference and the other did not.
The full 138,320 B gap reconciles to **23 bytes**:

| term | bytes |
|---|---|
| older artifact's `value_info` | +138,054 |
| protobuf framing for those entries | +3,154 |
| node section (this graph's nodes are collectively *larger*) | −2,911 |
| **unaccounted** | **+23** |

Note the node term runs the *opposite* way: this graph has 3 fewer nodes
yet ~3.2 KB more node bytes across the 1,570 they share, so the delta is
not "three extra `Cast` nodes" in any simple sense. `value_info` is not
semantic for execution — TensorRT and ONNX Runtime both infer shapes
themselves — but it is not cosmetic either: any consumer that reads
shapes off the graph sees a different graph.

**(3) Three `Cast` nodes — ~250 B, semantic, and load-bearing.** Weights
are bit-identical and every op type except `Cast` matches exactly. §6
shows these three nodes decide whether the repo's fp16 conversion
produces a model that will load at all.

**Byte size and behavioural risk are anti-correlated here.** ~214 KB of
changed path strings and ~138 KB of shape annotations changed nothing
that executes. Roughly 250 bytes of `Cast` nodes made the artifact
unloadable in two independent runtimes. Any registry policy that ranks
artifact diffs by size will rank these exactly backwards.

---

## 2. Engines built (T1)

TensorRT 11 **removed the `FP16` and `INT8` builder flags**. The only
type-related `NetworkDefinitionCreationFlag` is `STRONGLY_TYPED`, and
precision is carried by the ONNX graph itself — so "build an fp16 engine"
is now "convert the ONNX, then build", and precision becomes an
artifact-level decision rather than a builder-level one. The two engines
below therefore differ on **TF32**, which is still a builder flag and is
**on by default**.

| engine | TF32 | plan bytes | plan sha256 (12) |
|---|---|---|---|
| `fp32_tc_b1.plan` | on (default) | 118,940,300 | `3b1e7817f7bd` |
| `fp32_notf32.plan` | off | 118,556,796 | `9eb7a83f26fc` |

**Full identity tuple**, recorded per engine by `export/trt_build.py`
into a `.provenance.json` beside every plan:
source ONNX sha256 `c055e87cd710` · opset 17 · TF32 on/off ·
`STRONGLY_TYPED` false · workspace 4096 MB · builder optimization level 3
· `avgTimingIterations` 1 · resolved flag set · TensorRT 11.2.1.2 ·
driver 591.86 · CUDA runtime 13.1 · RTX 4070 sm_89 (46 SMs) ·
Windows-11-10.0.26200 · Python 3.12.7 · plan bytes · plan sha256.

The ONNX parsed with **no unsupported layers** — RF-DETR's TopK and
attention blocks converted without complaint into a 3,863-layer network.

---

## 3. Build determinism (T2)

Same ONNX, same builder config, repeated builds.

**(a) Without a timing cache — not deterministic, and not equivalent.**

| build | plan bytes | plan sha256 (12) |
|---|---|---|
| a1 | 119,729,836 | `b8d9b3adfdf8` |
| a2 | 119,665,620 | `4d18bb17105b` |
| a3 | 119,657,980 | `49007232e64a` |

Three different hashes **and three different sizes** (spread 71,856 B),
so the builder selected genuinely different tactics, not merely
serialized differently. Running all three over 20 real fixtures: **0/20
images bit-identical** between any pair; max abs delta on `labels` up to
**3.487**, on `dets` up to **0.908**. Threshold flips at 0.4 across those
20 images: **0**.

**(b) With a timing cache supplied — behaviourally deterministic, still
not byte-identical.**

| build | plan bytes | plan sha256 (12) |
|---|---|---|
| b1 | 118,940,300 | `3b1e7817f7bd` |
| b2 | 118,940,300 | `72528a216d77` |
| b3 | 118,940,300 | `51b4d84d53a6` |

Identical sizes, different hashes. Build time fell **15.5 s → 4.4 s**.
Over the same 20 fixtures the three plans were **bit-identical on 20/20
images** — max abs delta exactly **0.0** on both outputs. The differing
bytes are **0.005%** of the file (4,658–6,476 B of 118.9 MB), confined to
the region between offsets 5,402 and 1,922,153; the ~117 MB weight region
is byte-identical.

**What this licenses.** On this host and this TensorRT, a plan file's
hash is not a usable identity for this model: two builds from an
identical recipe never produced identical bytes, with or without a timing
cache. A timing cache did something stronger and narrower than
byte-reproducibility — it pinned *tactic selection*, which made the
engines numerically identical. Without one, two builds of the same ONNX
are not numerically interchangeable at all: a max logit delta of 3.487
between two "identical" builds is the same order as the fp32→fp16
conversion delta the CPU report measured (12.17). A registry for these
artifacts therefore cannot content-address the plan. It must
content-address the **recipe plus builder environment plus timing
cache**, and verify equivalence behaviourally rather than by hash. This
is a statement about TensorRT 11.2.1.2 on sm_89 with this graph; it is
not a general claim about TensorRT.

---

## 4. Polygraphy: ONNX vs engine (T3)

25 real fixtures from the same manifest, fed to both runners as
**identical saved input tensors** (`--load-inputs`), so the two sides
differ only in runtime.

Polygraphy's failure rule, read from `comparator/compare.py:1023-1025`,
is **AND**: an element is a mismatch when `absdiff > atol` **and**
`reldiff > rtol` (defaults 1e-5). That is the permissive rule — the same
column reported as `AND` by `export/parity.py`. An iteration fails if any
element mismatches.

| atol = rtol | TF32 on | TF32 off |
|---|---|---|
| 1e-5 | 0/25 (0%) | 5/25 (20%) |
| 1e-3 | 0/25 (0%) | 22/25 (88%) |
| 1e-2 | **0/25 (0%)** | 22/25 (88%) |

The TF32 engine fails every iteration at every tolerance tried, including
1e-2. The non-TF32 engine plateaus at 88% — three images never pass, so
that residual is specific images, not a tolerance choice.

---

## 5. Parity at the operating point, and latency (T4, T5)

200 fixtures. Decode is imported from `export/parity.py`, not
reimplemented: argmax over the 43 named columns only (background column
dropped), sigmoid of the winning logit, threshold, top-k 100, no NMS.
Because engine and ONNX are the same graph, query slot *q* means the same
thing in both, so the comparison is **query-aligned and exact** — no IoU
matching. Top-k truncation never bound.

**Raw tensors**

| engine | output | max abs Δ | mean abs Δ | p99 abs Δ | AND@1e-3 |
|---|---|---|---|---|---|
| TF32 on | `dets` | 1.7347 | 0.06203 | 0.7346 | 48.71% |
| TF32 on | `labels` | 9.3222 | 0.18207 | 1.6019 | 69.98% |
| TF32 off | `dets` | 0.8681 | **0.000271** | 0.000357 | 0.43% |
| TF32 off | `labels` | 2.7136 | **0.000932** | 0.004267 | 0.54% |

**Decoded detections**

| engine | thr | det ONNX | det engine | flips | A-only | B-only | class flips | images |
|---|---|---|---|---|---|---|---|---|
| TF32 on | 0.25 | 242 | 242 | 8 | 4 | 4 | 0 | 5 |
| TF32 on | 0.30 | 207 | 207 | 2 | 1 | 1 | 0 | 1 |
| TF32 on | **0.40** | **183** | **183** | **2** | **1** | **1** | **0** | **1** |
| TF32 on | 0.50 | 149 | 149 | 2 | 1 | 1 | 0 | 1 |
| TF32 on | 0.60 | 121 | 121 | 2 | 1 | 1 | 0 | 1 |
| TF32 off | all | — | — | **0** | 0 | 0 | **0** | 0 |

Max |Δconfidence| on any single query: 0.7696 (TF32 on), 0.0317 (off).

**Note what the totals conceal.** At 0.4 the TF32 engine returns 183
detections against the ONNX's 183 — a perfect match on count — while 2
queries disagree, one in each direction. A count-based gate passes this.
This is the same failure mode the CPU report documented for fp16 (202 vs
203, concealing 19 flips), reproduced here at smaller magnitude.

**Latency** — batch 1, after warmup. Engine timing is device
`execute_async_v3` + synchronize only, H2D/D2H excluded; ONNX Runtime
timing is `session.run` wall time. These are **not the same
measurement** and are not interchangeable.

| runtime | p50 | p95 | noise (p50 spread, repeat runs) |
|---|---|---|---|
| TensorRT, TF32 on | **5.11 ms** | 5.57 ms | **0.7%** (3 runs) |
| TensorRT, TF32 off | 6.60 ms | 6.94 ms | — |
| ORT CUDA EP fp32 (ORT 1.20.2) | 7.93–8.74 ms | 8.57–12.12 ms | **10.2%** (2 runs) |
| ORT CPU EP fp32 | 139–154 ms | — | — |

Against ORT's *best* CUDA-EP p50 (7.93 ms), TensorRT is **1.55×** with
TF32 and **1.20×** without. Against its slower run, 1.71× and 1.32×.
ORT's own run-to-run noise is 10.2%, so the TF32-off advantage (1.20×) is
close enough to that floor to be reported as a range rather than a
number; the TF32-on advantage is outside it. Clearing TF32 costs **29%**
of engine latency (5.11 → 6.60 ms), far outside the engine's own 0.7%
noise floor.

### What these results license

**On conversion fidelity.** On this fixture set, with this engine built
from `c055e87c` on this GPU, converting to TensorRT changed **2 of 183**
served detections at the recommended 0.4 threshold and changed **no**
predicted classes. It does not license "the conversion is safe" — it
describes 200 images from one held-out split, one GPU, one TensorRT
build.

**On TF32.** Essentially all of the divergence is attributable to TF32,
which is on by default: clearing it drops mean `labels` delta by ~195×
(0.182 → 0.00093) and every flip count to zero. A nominally "fp32" engine
is doing 10-bit-mantissa matmuls unless someone clears that flag. That is
a precision decision hiding inside a default, and it is worth 29% of
latency. This says nothing about whether 2 flips matter for this product
— that is a judgement about acceptable risk, not a measurement.

**On the tensor-vs-decision gap.** Polygraphy called the TF32 engine a
total failure — 0/25 at every tolerance including 1e-2 — while the same
engine changed 2 of 183 decisions. Both numbers are correct; they answer
different questions. A tensor-level tolerance gate on this graph would
block a conversion that barely moves served output, which is why the
operating-point comparison is the one worth gating on.

**On latency.** These are RTX 4070 numbers for a desktop part under WDDM.
They license nothing about any embedded target and nothing about the
arm64 Lambda host in production, whose CPU-provider numbers are ~140-154
ms here.

---

## 6. Attempted and blocked

**fp16 engine — blocked, and the cause is §1.1.** `scripts/make_fp16.py`
produced an fp16 ONNX from `c055e87c` that **neither runtime will load**:

- ONNX Runtime: `Type parameter (T) of Optype (Conv) bound to different
  types (tensor(float) and tensor(float16)) in node
  /backbone/.../patch_embeddings/projection/Conv`
- TensorRT, more precisely: `IConvolutionLayer 'input' and 'kernel' must
  be of same type. 'input' type is Float but 'kernel' is of type Half.`

Mechanism, confirmed by inspecting both graphs: in the fp32 source that
`Conv` is fed by a `Cast` node; in the converted fp16 graph the `Cast` is
gone and the raw fp32 graph `input` feeds a `Conv` whose kernel is now
fp16. This is a **third** fp16 defect, distinct from the two the CPU
report documents and not covered by `export/fp16_repair.py`. The CPU
report's fp16 artifact, built from the 39-`Cast` export, loaded fine
after repair. The same pipeline on the 36-`Cast` export of the same
checkpoint, from the same machine, does not. No repair was attempted —
inventing one here would be a new claim needing its own validation.

**`trtexec` — unavailable.** Not shipped in the `tensorrt` pip wheel (it
is a C++ binary from the tarball/zip distribution). The scripted builder
in `export/trt_build.py` was used instead, which is what T1 wanted
version-controlled anyway; conversion is proven by the engines in §2.

**ORT CUDA EP — required a downgrade.** `onnxruntime-gpu` 1.29.0 needs
CUDA 13 runtime libraries not present on this box and silently fell back
to `CPUExecutionProvider`. The first latency run therefore produced a
"29× vs ORT-CUDA" figure that was really 29× vs ORT-CPU; it is discarded.
§5 uses ORT 1.20.2 with CUDA 12.4 libraries, provider resolution asserted
at session creation rather than assumed.

**INT8 — not attempted.** TensorRT 11 removed the `INT8` builder flag, so
int8 requires a QDQ-annotated ONNX produced with a real calibration set,
not a builder toggle. Given that the fp16 precision-converted variant of
this exact graph does not load (above), that path needs the §6 defect
resolved first. The exporter's existing refusal to emit int8 without
calibration remains the right behaviour.

---

## 7. What would be needed to make this a release gate

It is not one, and nothing above should be read as one.

1. **A fixture set that represents production.** These are held-out val
   images, which is closer to production than the CPU report's Reddit
   cleanup photography but still not the deployed input distribution.
2. **A decision about acceptable flips.** 2 in 183 is a measurement.
   Whether it passes is a product judgement nobody has made. Until
   someone does, `export/parity.py --baseline` (no-regression against a
   stored result) is the only gate the data supports.
3. **Pinned builder identity.** Because plans are not content-
   addressable (§3), a gate must record the recipe + environment +
   timing cache and re-verify behaviour, not compare hashes.
4. **Strip `doc_strings` before hashing, or export without them.** This
   is the cheapest reproducibility win available and it is worth taking
   independently of everything else. They are **4.80%** of artifact size
   (5.77 MB here), they embed absolute local paths — `C:\tmp\venv-rfdetr`
   and the full checkout path — into a file that gets published to S3,
   and they destroy hash stability for no semantic gain: simply moving
   the checkout accounted for 210,465 B of drift between two exports of
   one checkpoint (§1.1). Stripping them is **not** a fix for the 3
   `Cast` nodes or the fp16 defect in §6, which are semantic and
   survive it.
5. **The fp16 path fixed** (§6), or fp16 explicitly declared out of scope
   for GPU serving.
6. **Measurement on the target device.** Every number here is RTX 4070.
   For any other accelerator this report is a method, not a result.

---

### Reproduce

```powershell
# engines + provenance
python -m export.trt_build --onnx <onnx> --out fp32_tf32.plan --workspace-mb 4096
python -m export.trt_build --onnx <onnx> --out fp32_notf32.plan --workspace-mb 4096 --no-tf32
python -m export.trt_build --onnx <onnx> --out gen.plan --timing-cache-out timing.cache
python -m export.trt_build --onnx <onnx> --out b1.plan  --timing-cache-in  timing.cache

# parity: ONNX vs engine, raw tensors + operating point + latency
python -m export.trt_parity --onnx <onnx> --engine b1.plan --meta <meta.json> \
    --fixtures reports/parity-fixtures-win-v1.txt \
    --report-json reports/trt-parity-fp32-vs-engine-tf32on.json

# polygraphy input file — REAL fixtures, not polygraphy's random default
python -c @"
import json; from pathlib import Path
from polygraphy.json import save_json
from export.parity import preprocess_detr, read_fixture_manifest
m = json.loads(Path('<meta.json>').read_text())
mean, std = tuple(m['normalization']['mean']), tuple(m['normalization']['std'])
imgs = read_fixture_manifest(Path('reports/parity-fixtures-win-v1.txt'))[:25]
save_json([{'input': preprocess_detr(p, 512, 512, mean, std)} for p in imgs],
          'pg_inputs.json', description='25 real litter fixtures')
"@

# polygraphy, real inputs, AND semantics at 1e-3
polygraphy run <onnx>   --onnxrt --load-inputs pg_inputs.json --save-outputs pg_onnx_out.json
polygraphy run b1.plan  --model-type engine --trt --load-inputs pg_inputs.json \
    --load-outputs pg_onnx_out.json --atol 1e-3 --rtol 1e-3
```

TensorRT and Polygraphy are an **optional** extra (`trt`); they are not
required dependencies of this package.

JSON results: `reports/trt-parity-fp32-vs-engine-tf32on.json`,
`reports/trt-parity-fp32-vs-engine-tf32off.json`.

`export/trt_build.py` writes a `<plan>.provenance.json` beside every plan
it produces. Those records are committed under
`reports/trt-provenance/` — all nine, including the six determinism
builds — so every plan hash quoted in §2 and §3 is checkable without
rebuilding. The plan files themselves (~119 MB each) are deliberately not
committed; `dist/` is gitignored for that reason, and §3 is the argument
that storing the blob would not establish identity anyway.
