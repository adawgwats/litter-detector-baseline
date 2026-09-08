# Static post-training quantization: faster, and far less faithful

Measured 2026-09-08. Static PTQ of the fp32 RF-DETR-Small litter ONNX
(`c055e87c`) with a real, coverage-chosen calibration set, compared against fp32
and against the dynamic int8 the repo already builds. Additive; x86 CPU
throughout. The one-line answer the peer flagged as acceptable is the one that
holds: **static PTQ costs too much accuracy on this model** — it is meaningfully
faster than both fp32 and dynamic int8, and it loses about a third of the
detections doing it.

## Environment & method

x86 AMD64, ONNX Runtime 1.29.0 `CPUExecutionProvider`. `quantize_static`,
**QDQ** format, **per-channel int8 weights**, **per-tensor uint8 activations**
(ORT's static activation quant is per-tensor; only weights are per-channel —
that asymmetry is standard and is itself part of why activations are where the
error enters). Two calibrators: **MinMax** and **Percentile** (99.999). Eval:
the same 100 held-out images used elsewhere (50 `taco_*` + 50 `hn_*`),
query-aligned decode at 0.4 via `export.parity`.

**Calibration set — chosen for coverage, not count.** Drawn from the *other*
half of the val split (disjoint from the 100 eval images), balanced on two axes:
density (litter-bearing `taco` = dense scenes vs hard-negative `hn` = clean /
sparse) and lighting (below/above each class's median measured luminance).
Target was 48 images — 12 per (density × brightness) cell. **MinMax calibrated
on all 48. Percentile could not:** its calibrator augments the graph to emit
every intermediate tensor and keeps a full 2048-bin histogram for each of 958
tensors, and it exhausted memory (`bad allocation`) at 48 images. It completed
only after dropping to **8** calibration images with chunked collection. That
memory cost is a real operational property of the Percentile/Entropy calibrators
on a graph this size, not a footnote — so the two calibrators below are compared
both at matched size (8 each) and as-built.

---

## 1. Latency and size

| model | p50 (100 imgs) | size | vs fp32 |
|---|---|---|---|
| fp32 | 131.6 ms | 120,111,594 B | — |
| int8 **dynamic** | 167.7 ms | 36,170,250 B | **1.27× slower** |
| int8 **static MinMax** | 95.4 ms | 36,475,892 B | **1.38× faster** |
| int8 **static Percentile** | 95.9 ms | 36,475,967 B | 1.37× faster |

The speed story inverts the naive expectation. **Dynamic int8 is *slower* than
fp32** here — it computes activation quantization parameters at run time, every
inference, and on this CPU that overhead outweighs the int8 arithmetic saving.
**Static int8 is faster than both** because the activation scales are baked in at
calibration and paid once. So if latency were the only axis, static wins
outright. It is not the only axis.

**This whole table is an x86 statement.** On both ARM hosts measured in
`PI-EDGE-REPORT.md §2a`, dynamic int8 is *faster* than fp32 (1.53× on the Pi,
1.85× on Apple silicon), where here it is slower — the int8 speedup changes sign
with the host. Static PTQ was not run on ARM, so the static-vs-dynamic ordering
above may not hold there: the premise that makes static the winner (dynamic being
slow) is itself x86-specific. Read the latency ranking as "on this AMD Zen 4
box", not as a property of the conversions.

## 2. Fidelity to fp32 (query-aligned at 0.4)

fp32 produced **90** detections over the 100 images.

| variant | det fp32 | det int8 | threshold flips | class flips | images affected |
|---|---|---|---|---|---|
| dynamic | 90 | 84 | 14 | 13 | 14 / 100 |
| static MinMax (48-img calib) | 90 | **58** | **40** | **25** | **37 / 100** |
| static Percentile (8-img calib) | 90 | 61 | 43 | 19 | 32 / 100 |

Static PTQ **loses about a third of the detections** (58–61 vs 90) and disagrees
with fp32 on roughly a third of images — against dynamic's 14%. Raw logit
divergence tells the same story: max |Δ labels| vs fp32 is 10.33 (dynamic), 12.50
(static MinMax), 13.30 (static Percentile); mean |Δ| 0.538 vs 0.908 vs 0.686.
Static is the larger perturbation on every measure that matters at the operating
point.

**Where it degrades:** the loss is concentrated in dropped detections
(survivors in fp32 that fall below 0.4 under static), plus a substantial class-
flip count — 25 queries that survive in both static-MinMax and fp32 but are
assigned a different class. It is not a uniform confidence shift; it is decisions
changing.

**Why.** DETR/transformer activations — LayerNorm outputs, attention scores —
have wide, input-dependent dynamic range. Per-tensor static activation scales,
fixed from a handful of calibration images, clip or coarsely quantize that range
the same way for every input. Dynamic quantization recomputes the activation
range per inference and tracks it; that adaptivity is exactly what static gives
up, and on a transformer it is expensive to give up. This is the well-known
reason transformers are hard to quantize statically and usually need QAT to
recover the accuracy — which is the path this result points to, not a tolerance
to loosen.

## 3. The calibrator matters, and so does the calibration set

At matched calibration size (8 images each), MinMax and Percentile still make
different decisions:

| comparison | det A | det B | flips | class flips | images |
|---|---|---|---|---|---|
| MinMax-8 vs Percentile-8 | 65 | 61 | 34 | 13 | 25 / 100 |
| MinMax-48 vs dynamic | 58 | 84 | 34 | 24 | 35 / 100 |

The two calibrators disagree on **25 of 100 images** on the same calibration
data — Percentile's outlier-clipping (99.999) and MinMax's raw extremes land on
different activation scales, and the operating point feels it. Note also that
MinMax at 8 images (65 detections) recovered slightly over MinMax at 48 (58):
the static result is sensitive to calibration-set composition as well as
calibrator, which is a caution, not a tuning knob to exploit — chasing a
calibration set that happens to restore detections would be fitting the metric.

---

## What this licenses

On this model and this fixture set: static PTQ is a **1.3–1.4× latency win over
fp32 and ~1.75× over dynamic int8**, bought at the cost of **~35% of the
detections and disagreement on ~1/3 of images** — an order more decision churn
than dynamic int8, which the repo already ships and which is the more faithful
(if slower) int8 here. It does not license "static int8 is unusable": it
licenses "static int8, plain PTQ, is not accurate enough to substitute for fp32
on this model without recovery training." The calibrator and calibration set
move the result by tens of images, so no single static number is the number.

The honest recommendation is the unglamorous one: if int8 is wanted for fidelity,
dynamic is the better PTQ here despite being slower; if static's speed is wanted,
the accuracy has to be bought back with QAT rather than with a larger calibration
set or a looser threshold. "Static costs too much accuracy here" is the result.

### Reproduce
`quantize_static` (QDQ, per-channel int8 weights, per-tensor uint8 activations)
with a `CalibrationDataReader` over the coverage-selected real fixtures, for
`CalibrationMethod.MinMax` and `.Percentile`; then run fp32, dynamic int8, and
both static models over the 100 eval images and compare decoded survivors
query-aligned at 0.4 with `export.parity.decode_detr_queries`. Percentile needs
`CalibMaxIntermediateOutputs` set and a small calibration set to fit in memory.
Scripts kept out of the repo.
