# The litter model on the Pi 5, and the same bytes across two architectures

Measured 2026-09-08. The real RF-DETR-Small litter detector run on a Raspberry
Pi 5 — closing the gap in `/benchmarks`, which until now had only ever
benchmarked a stock COCO YOLOv8n on the Pi, never this model. Additive: nothing
shipped or serving was touched.

Everything here is **Pi 5 / Cortex-A76 / CPU execution provider**. It is an
embedded-class board, not an Orin: no GPU, no DLA, no automotive runtime. None
of these numbers transfer to one.

## Environment

| | Pi | x86 (reference host) |
|---|---|---|
| machine | aarch64, Raspberry Pi 5 Model B Rev 1.1 | AMD64 |
| CPU | Cortex-A76 ×4 | desktop x86 |
| RAM | 987 MiB (+986 MiB swap) | — |
| ONNX Runtime | 1.29.0, `CPUExecutionProvider` | 1.29.0, `CPUExecutionProvider` |
| numpy / Pillow | 2.5.3 / 12.3.0 | 2.5.3 / 12.3.0 (identical) |
| Python | 3.13.5 | 3.12 |

Artifacts (both built on the x86 box, from source ONNX `c055e87c`):
- **int8-dyn** — `scripts/make_int8.py` (ORT dynamic quantization), 36,170,250 B, sha `080ab637`.
- **fp32** — 120,111,594 B, sha `c055e87c`.

Fixtures: 100 real photographs (50 `taco_*` litter-bearing + 50 `hn_*` hard
negatives) from the held-out val split, the same set on both machines.

---

## 1. The model runs on the Pi — and fp32 fits, against expectation

The prior assumption was that a 120 MB fp32 graph would not fit in 1 GB and that
int8 was what made the model *runnable* at all. **That is not what happened.**
fp32 loaded and ran with **380 MiB peak RSS and zero swap used.** int8's memory
advantage over fp32 on this model is real but modest — it is a latency
optimization here, not a fit-enabler.

| artifact | p50 | p95 | peak RSS | fits in 1 GB? |
|---|---|---|---|---|
| int8-dyn | **745.3 ms** | 782.3 ms | **303.9 MiB** | yes |
| fp32 | **1142.6 ms** | 1184.1 ms | **379.7 MiB** | yes, no swap |

int8 is **1.53× faster** than fp32 on the Pi and uses **~20% less RAM** (76 MiB).
Both are far from the board's memory ceiling. The hypothesized ~500 MB fp32
working set did not materialize: RF-DETR-Small's activation footprint at
512×512 batch 1 is small enough that the graph size, not the working set,
dominates, and 120 MB of weights plus ORT overhead peaks under 400 MiB.

(fp32 was measured over 5 images rather than 100 — peak RSS is set by the
session plus a single inference, so it is representative; the latency percentiles
are over those 5 and are steadier than they look, min 1100 / max 1193 ms.)

### Anchor against the published Pi YOLOv8n

`/benchmarks` reports the stock COCO YOLOv8n on the Pi at 140 MiB RSS with a
73.6 °C peak over a 1-hour soak. This litter model is a **substantially heavier
edge load**: 2.2× (int8) to 2.7× (fp32) the memory, and — being an RF-DETR with
a DINOv2 backbone rather than a YOLO — much slower per frame. The comparison is
memory-and-architecture only; **no soak was run here**, so the thermal number is
not comparable — see §3.

## 2. Latency vs the x86 reference (same int8 bytes)

| | Pi (aarch64) | x86 (AMD64) | Pi ÷ x86 |
|---|---|---|---|
| int8-dyn p50 | 745.3 ms | 172.8 ms | **4.31×** |
| int8-dyn p95 | 782.3 ms | 184.0 ms | 4.25× |
| peak RSS | 303.9 MiB | 277.2 MiB | — |

Same ONNX (`080ab637`), same ORT version, same execution provider. The Pi is
~4.3× slower per frame. Neither is a GPU number and neither is the arm64 Lambda
serving host.

### 2a. The int8 speedup changes *sign* with the host

Comparing dynamic int8 against its own fp32 baseline *within each machine* —
same conversion, same mechanism — the direction of the speedup is not stable
across architectures:

| host | fp32 p50 | int8-dyn p50 | int8 vs fp32 |
|---|---|---|---|
| Pi 5, Cortex-A76, ORT 1.29.0 | 1142.6 ms | 745.3 ms | **1.53× faster** |
| macOS arm64 (Apple silicon), ORT 1.29.0 | 284.3 ms | 154.1 ms | **1.85× faster** |
| this x86_64 box, ORT 1.29.0 | 131.6 ms | 167.7 ms | **1.27× *slower*** |

The two ARM hosts agree in direction; the x86 host inverts it. "Is int8 faster?"
has no answer without naming the host — the cross-platform problem again, this
time with the **sign** changing, not just the magnitude.

**State the confound plainly, because it is large.** These are three different
machines, three operating systems, and there is no guarantee the ORT build or
the CPU feature set is matched across them. Two ARM points agreeing is
*suggestive* of a kernel-availability effect, not proof of an architecture law.
It is worth noting that the inversion is **not** simply "x86 lacks int8
acceleration": this x86 box is an **AMD Ryzen 7 7700 (Zen 4) whose CPU flags
include `avx512vnni`** — the int8 dot-product instructions are present. int8 is
slower here anyway, which points at dynamic quantization's per-inference
activation-scaling overhead outweighing the int8 arithmetic saving on this
graph, rather than at missing hardware. (x86 side pinned:
`onnxruntime.get_available_providers()` includes CPU/CUDA/TensorRT EPs; the run
used `CPUExecutionProvider`; CPU int8-relevant flags: avx, avx2, avx512f,
avx512vnni, f16c, fma, sse4_2.) The clean version of this experiment — one
artifact, one ORT build, ARM and x86 hosts otherwise matched — has not been run;
these three machines cannot support a claim stronger than "the sign is not
host-invariant, and here is one case of each."

## 3. Thermals

Read once after the runs (short bursts, not a soak): **41.1 °C**, ARM clock
1.6 GHz, no *current* throttling. `vcgencmd get_throttled` returned `0x50000` —
bits for *under-voltage occurred* and *throttling occurred* **since boot** are
set (the live throttle bits are clear). That is the power-delivery condition the
repo's own `robot/pi/bringup/` sequence exists to fix (`usb_max_current_enable`,
`verify_power.sh`). A real thermal characterization would need a sustained soak
like the YOLOv8n reference's; this was 100 inferences, not an hour, so no peak
temperature is claimed.

---

## 4. The measurement that matters: the same bytes decide differently on ARM and x86

The same int8 ONNX was run on both architectures and its decoded detections
compared **query-aligned at the recommended 0.4 threshold** (`export.parity`'s
decode, not a reimplementation). To isolate the architecture, the first images'
preprocessed input tensors were saved on both machines and compared:

```
input parity:  max|Δinput| = 0.000e+00  on every sampled image
```

Both platforms fed the model **byte-identical** tensors. So everything below is
the runtime and the architecture, not preprocessing.

**Raw outputs** already diverge:

| output | max abs Δ | mean abs Δ | p99 abs Δ |
|---|---|---|---|
| `dets` | 1.698 | 0.170 | 0.855 |
| `labels` | 9.063 | 0.527 | 2.152 |

**Decoded at 0.4, query-aligned:**

| | value |
|---|---|
| detections, Pi | 85 |
| detections, x86 | 84 |
| threshold flips | **9** (Pi-only 5, x86-only 4) |
| class flips (survive both, different argmax) | **9** |
| images with any disagreement | **9 / 100** |

**Read the totals, then read past them.** 85 vs 84 detections looks like the two
architectures agree. They do not: 9 queries cross the 0.4 threshold on exactly
one of the two, and 9 more survive on both but are assigned a *different class*.
A cross-platform check that compared detection counts — or even mAP against a
shared ground truth, which would blur these into near-noise — would report
"identical" and ship. The disagreement is 18 query-level decisions across 9
images, hidden behind a count difference of one.

**Why.** Dynamic int8 quantization computes with int8 matmuls and fp32
accumulation; the kernels that do this differ between x86 (AVX-512/VNNI) and
Cortex-A76 (NEON/dot-product), and so do their rounding and accumulation order.
Different arithmetic on the same weights yields logits that differ by up to 9.06
here — decisive on a query sitting near p=0.4, irrelevant on one at p=0.02.

### What this licenses, and what it does not

This is **decision instability across architectures, not accuracy loss.**
Nothing here says which platform is more nearly correct — that would require
scoring both against ground truth, which this measurement does not have. What it
establishes is that *"we validated the int8 model" is not a claim about the
model; it is a claim about the model on one architecture.* An int8 artifact
signed off on x86 makes 18 different query-level decisions on the Pi over 100
images, and the count of detections nearly hides it. For anything that ships one
artifact to more than one CPU architecture, the operating-point comparison — not
a count, not an aggregate score — is the check that catches this.

The flip and class-flip counts here are non-zero and query-aligned; if they are
worth scoring against blind human ground truth to establish whether the
instability is also a quality difference, that ground truth exists off this box
and the per-image outputs are saved (`pi_int8_out.npz`, `x86_int8_out.npz`).

---

### Reproduce
Ship the int8 ONNX + fixture images + `export/parity.py` to the Pi; run the same
runner on both machines (loads the ONNX, preprocesses via `parity.preprocess_detr`,
saves `dets`/`labels` per image + latency + peak RSS); pull the Pi's outputs
back; decode both with `export.parity.decode_detr_queries` and compare survivors
query-aligned at 0.4. Scripts kept out of the repo.
