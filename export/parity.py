"""Quality-parity harness: compare two ONNX artifacts of the same checkpoint.

Answers the question an export pipeline cannot answer from artifact
metadata alone: *when we convert precision, does the served model still
behave the same?* The exporters in this package stamp a precision suffix
into the version string and a ``confThresholdRecommended`` into the
sidecar (see export/export_rfdetr.py), but nothing has ever measured
whether that recommendation survives the conversion it is stamped
alongside.

Two comparisons, in this order, because they license different claims:

1. **Raw graph outputs, before any decode.** Feed both artifacts the
   identical input tensor and diff every output tensor elementwise.
   This measures the conversion, and only the conversion. It is the
   number that generalizes across thresholds and decode changes.

2. **Decoded detections, at the operating point.** Run the real decode
   for the architecture and count how many detections cross
   ``confThresholdRecommended`` differently between the two artifacts.
   This is the number a consumer feels. Tensor deltas alone do not
   license a claim about served behaviour: a 1e-3 logit delta is
   irrelevant on a query at p=0.02 and decisive on one at p=0.400.

Deliberate non-features, both load-bearing:

*   **No baked pass/fail tolerance.** The harness reports the empirical
    distribution of divergence and stops. A tolerance is a judgement
    about acceptable risk on a particular input distribution; it is not
    a property of the conversion, and hard-coding one here would launder
    a guess into a gate. When you do want a gate, pass ``--baseline`` —
    that asserts *no regression against a previously measured result*,
    which is a claim the harness can actually support.

*   **Real images only.** Random tensors put activations in ranges the
    network never sees in service. They simultaneously hide real fp16
    overflow (because random inputs rarely excite the largest
    activations) and manufacture divergence that no user would ever
    encounter. Build a fixture manifest with
    ``scripts/build_parity_fixtures.py``.

Scope of any result this produces: it describes the two artifacts named
on the command line, executed by the ONNX Runtime build and execution
provider recorded in the report's ``environment`` block, over the
fixture set recorded in its ``fixtures`` block. It is not a statement
about a different runtime, a different execution provider, a different
accelerator, or an input distribution the fixture set does not cover.

Usage::

    python -m export.parity \\
        --a dist/parity/rfdetr-s-litter.fp32.onnx \\
        --b dist/parity/rfdetr-s-litter.fp16.onnx \\
        --meta roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.meta.json \\
        --fixtures reports/parity-fixtures-v1.txt \\
        --report-json reports/parity-rfdetr-v2.0.1-fp32-vs-fp16.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

LOG = logging.getLogger("export.parity")

SCHEMA_VERSION = 2

# Tolerance pairs reported by default. These are REPORTING buckets, not
# gates — the harness never fails on them. 1e-5 is numpy/Polygraphy's
# familiar default and is included so a reader can locate this result
# against tooling they already know; the larger pairs exist because fp16
# has ~3 decimal digits of mantissa and 1e-5 is not a meaningful ask of
# it.
DEFAULT_TOLERANCES: tuple[tuple[float, float], ...] = (
    (1e-5, 1e-5),
    (1e-3, 1e-3),
    (1e-2, 1e-2),
)

# Thresholds swept in the decode comparison. The sidecar's recommended
# value is always added to this set.
DEFAULT_SWEEP: tuple[float, ...] = (0.25, 0.3, 0.4, 0.5, 0.6)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# preprocessing — mirrors dregsbane-web-backend src/lib/inference/preprocess-detr.ts
# --------------------------------------------------------------------------

def preprocess_detr(
    image_path: Path,
    height: int,
    width: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """Image file -> [1, 3, H, W] float32 NCHW RGB, (x/255 - mean) / std.

    Mirrors the backend's preprocess-detr.ts: a PLAIN squash resize to
    the model input dims (``fit: 'fill'`` — no letterbox, no padding),
    alpha removed, then per-channel ImageNet normalization after /255.

    One deliberate difference: the backend resizes with sharp (libvips,
    lanczos3 by default) and this resizes with Pillow. Pillow's LANCZOS
    is also a 3-lobe Lanczos, but the two implementations do not produce
    bit-identical pixels. That does not affect a parity measurement —
    both artifacts are fed the SAME tensor, so the resize kernel cancels
    out of every delta reported here. It does mean absolute detection
    counts are close to, but not identical to, what the deployed
    pipeline would produce on the same file.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")  # also drops alpha, matching .removeAlpha()
        im = im.resize((width, height), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32)  # HWC, 0..255

    arr /= 255.0
    arr -= np.asarray(mean, dtype=np.float32)
    arr /= np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)


# --------------------------------------------------------------------------
# decode — mirrors dregsbane-web-backend src/lib/inference/postprocess-detr.ts
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DetrQueryScores:
    """Per-query decode intermediates, kept query-aligned.

    Both artifacts are the same graph, so query slot *q* means the same
    thing in each. Comparing query-to-query is therefore exact and needs
    no bipartite matching — a considerably sharper instrument than
    matching two detection *sets* by IoU, which would blur a threshold
    crossing into a "missing detection".
    """

    best_class: np.ndarray  # [Q] int64, argmax over NAMED classes only
    confidence: np.ndarray  # [Q] float64, sigmoid of the winning logit
    boxes_xyxy: np.ndarray  # [Q, 4] float64, normalized, clamped to [0,1]


def decode_detr_queries(
    boxes: np.ndarray,
    logits: np.ndarray,
    num_named_classes: int,
) -> DetrQueryScores:
    """Raw ``dets``/``labels`` -> per-query class, confidence, box.

    Reproduces postprocess-detr.ts exactly, including the trailing
    background column: rf-detr checkpoints export ``classCount + 1``
    logit columns and canonical classes occupy 0..C-1, so column C is
    ignored rather than rejected. Sigmoid is applied only to the winning
    logit — it is monotonic, so argmax over logits is argmax over
    scores.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    lg = np.asarray(logits, dtype=np.float64).reshape(b.shape[0], -1)
    if lg.shape[1] not in (num_named_classes, num_named_classes + 1):
        raise ValueError(
            f"decode_detr_queries: tensor has {lg.shape[1]} class channels "
            f"but classNames has {num_named_classes} (expected "
            f"{num_named_classes} or {num_named_classes + 1})"
        )
    named = lg[:, :num_named_classes]
    best_class = named.argmax(axis=1)
    best_logit = named[np.arange(named.shape[0]), best_class]
    conf = 1.0 / (1.0 + np.exp(-best_logit))

    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
    ).clip(0.0, 1.0)
    return DetrQueryScores(best_class=best_class, confidence=conf, boxes_xyxy=xyxy)


# --------------------------------------------------------------------------
# tensor divergence
# --------------------------------------------------------------------------

@dataclass
class OutputAccumulator:
    """Elementwise divergence for one output tensor, accumulated over images.

    Keeps ``deltas`` and ``refs`` aligned (same length, same order) so
    the tolerance rules below can be evaluated per element. The
    reference magnitudes are what make a *relative* criterion definable
    at all.
    """

    name: str
    deltas: list[np.ndarray] = field(default_factory=list)
    refs: list[np.ndarray] = field(default_factory=list)
    per_image_max_abs: list[float] = field(default_factory=list)
    n_elements: int = 0

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        af = np.asarray(a, dtype=np.float64).ravel()
        bf = np.asarray(b, dtype=np.float64).ravel()
        if af.shape != bf.shape:
            raise ValueError(
                f"output {self.name!r}: shape mismatch {af.shape} vs {bf.shape} — "
                "the two artifacts are not the same graph"
            )
        d = np.abs(af - bf)
        self.deltas.append(d)
        self.refs.append(np.abs(af))
        self.per_image_max_abs.append(float(d.max()) if d.size else 0.0)
        self.n_elements += int(d.size)

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.deltas:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(self.deltas), np.concatenate(self.refs)

    @staticmethod
    def _relative(d: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """|a-b| / |a|, with 0/0 -> 0 and x/0 -> inf.

        An exactly-zero reference with a non-zero delta really is an
        infinite relative error; representing it as inf keeps it visible
        in the tolerance rules instead of silently dropping it.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.divide(d, ref)
        rel[(ref == 0) & (d == 0)] = 0.0
        rel[(ref == 0) & (d > 0)] = np.inf
        return rel

    def summarize(
        self,
        tolerances: Sequence[tuple[float, float]],
        rel_floor: float = 1e-6,
    ) -> dict[str, Any]:
        d, ref = self._arrays()
        rel = self._relative(d, ref)
        # Summary relative statistics exclude elements whose reference
        # magnitude is below rel_floor: a single denormal reference makes
        # the max relative error meaningless without telling you anything
        # about the conversion. The exclusion is counted, not hidden, and
        # it applies ONLY to these summary stats — the tolerance rules
        # below are evaluated on every element.
        usable = ref > rel_floor
        rel_u = rel[usable]
        out: dict[str, Any] = {
            "name": self.name,
            "elementsCompared": self.n_elements,
            "maxAbsDelta": float(d.max()) if d.size else 0.0,
            "meanAbsDelta": float(d.mean()) if d.size else 0.0,
            "medianAbsDelta": float(np.median(d)) if d.size else 0.0,
            "p95AbsDelta": float(np.percentile(d, 95)) if d.size else 0.0,
            "p99AbsDelta": float(np.percentile(d, 99)) if d.size else 0.0,
            "p999AbsDelta": float(np.percentile(d, 99.9)) if d.size else 0.0,
            "maxRelDelta": float(rel_u.max()) if rel_u.size else 0.0,
            "meanRelDelta": float(rel_u.mean()) if rel_u.size else 0.0,
            "p99RelDelta": float(np.percentile(rel_u, 99)) if rel_u.size else 0.0,
            "relFloor": rel_floor,
            "relElementsCompared": int(rel_u.size),
            "relElementsSkippedBelowFloor": int(self.n_elements - rel_u.size),
            "perImageMaxAbsDelta": {
                "max": max(self.per_image_max_abs) if self.per_image_max_abs else 0.0,
                "mean": (
                    statistics.fmean(self.per_image_max_abs)
                    if self.per_image_max_abs
                    else 0.0
                ),
                "min": min(self.per_image_max_abs) if self.per_image_max_abs else 0.0,
            },
            "exceedance": [
                self._exceedance(d, ref, rel, atol, rtol) for atol, rtol in tolerances
            ],
        }
        return out

    @staticmethod
    def _exceedance(
        d: np.ndarray,
        ref: np.ndarray,
        rel: np.ndarray,
        atol: float,
        rtol: float,
    ) -> dict[str, Any]:
        """Element counts failing each of three DIFFERENT tolerance rules.

        The three are not interchangeable, and the difference between
        them is exactly what gets lost when a report says "N elements
        exceeded 1e-5" without saying which rule produced N:

        ``combined``  fail when ``|a-b| > atol + rtol*|a|``. This is
                      ``numpy.isclose`` / ``numpy.allclose`` semantics —
                      one blended criterion, NOT a conjunction of two.
        ``and``       fail when ``|a-b| > atol`` AND ``rel > rtol``.
                      The most permissive: an element must be bad on
                      both scales to count.
        ``or``        fail when ``|a-b| > atol`` OR ``rel > rtol``.
                      The strictest: bad on either scale counts.

        All three are reported so the reader picks the rule rather than
        inheriting one by accident. ``absOnly`` and ``relOnly`` are the
        two arms in isolation, which is what lets you see which arm is
        driving a given rule.
        """
        n = int(d.size) or 1
        abs_fail = d > atol
        rel_fail = rel > rtol
        combined = d > (atol + rtol * ref)

        def tally(mask: np.ndarray) -> dict[str, Any]:
            c = int(mask.sum())
            return {"count": c, "fraction": c / n}

        return {
            "atol": atol,
            "rtol": rtol,
            "absOnly": tally(abs_fail),
            "relOnly": tally(rel_fail),
            "combinedIsclose": tally(combined),
            "and": tally(np.logical_and(abs_fail, rel_fail)),
            "or": tally(np.logical_or(abs_fail, rel_fail)),
        }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_fixture_manifest(path: Path) -> list[Path]:
    """One image path per line; ``#`` comments and blanks ignored."""
    images: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        images.append(Path(line))
    if not images:
        raise ValueError(f"fixture manifest {path} lists no images")
    return images


def _make_session(model_path: Path, providers: Sequence[str] | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=list(providers) if providers else ["CPUExecutionProvider"],
    )


def compare(
    model_a: Path,
    model_b: Path,
    meta: dict[str, Any],
    images: Sequence[Path],
    tolerances: Sequence[tuple[float, float]] = DEFAULT_TOLERANCES,
    sweep: Sequence[float] = DEFAULT_SWEEP,
    rel_floor: float = 1e-6,
    max_detections: int = 100,
    warmup: int = 3,
    providers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run both artifacts over ``images`` and return the report dict."""
    import onnxruntime as ort

    sess_a = _make_session(model_a, providers)
    sess_b = _make_session(model_b, providers)

    names_a = [o.name for o in sess_a.get_outputs()]
    names_b = [o.name for o in sess_b.get_outputs()]
    if names_a != names_b:
        raise ValueError(
            f"output names differ: {names_a} vs {names_b} — not the same graph"
        )
    in_a = sess_a.get_inputs()[0].name
    in_b = sess_b.get_inputs()[0].name

    _, _, height, width = meta["inputShape"]
    classes: list[str] = meta["classes"]
    norm = meta.get("normalization") or {}
    mean = tuple(norm.get("mean", IMAGENET_MEAN))
    std = tuple(norm.get("std", IMAGENET_STD))
    recommended = float(meta["confThresholdRecommended"])

    thresholds = sorted({*(float(t) for t in sweep), recommended})

    accs = {n: OutputAccumulator(n) for n in names_a}
    lat_a: list[float] = []
    lat_b: list[float] = []

    # Per-threshold decode tallies.
    flips = {t: 0 for t in thresholds}          # survived in exactly one artifact
    flips_a_only = {t: 0 for t in thresholds}   # survived in A, not B
    flips_b_only = {t: 0 for t in thresholds}   # survived in B, not A
    det_a = {t: 0 for t in thresholds}
    det_b = {t: 0 for t in thresholds}
    class_flips = {t: 0 for t in thresholds}    # both survived, different class
    images_with_flip = {t: 0 for t in thresholds}
    truncation_hits = {t: 0 for t in thresholds}

    conf_deltas: list[float] = []
    box_deltas: list[float] = []
    per_image: list[dict[str, Any]] = []

    dummy = np.zeros((1, 3, height, width), dtype=np.float32)
    for _ in range(warmup):
        sess_a.run(None, {in_a: dummy})
        sess_b.run(None, {in_b: dummy})

    for idx, img in enumerate(images):
        tensor = preprocess_detr(img, height, width, mean, std)

        t0 = time.perf_counter()
        out_a = sess_a.run(None, {in_a: tensor})
        lat_a.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        out_b = sess_b.run(None, {in_b: tensor})
        lat_b.append((time.perf_counter() - t0) * 1000.0)

        for name, a_arr, b_arr in zip(names_a, out_a, out_b):
            accs[name].update(a_arr, b_arr)

        boxes_i = names_a.index("dets") if "dets" in names_a else 0
        logits_i = names_a.index("labels") if "labels" in names_a else 1
        qa = decode_detr_queries(out_a[boxes_i], out_a[logits_i], len(classes))
        qb = decode_detr_queries(out_b[boxes_i], out_b[logits_i], len(classes))

        conf_deltas.append(float(np.abs(qa.confidence - qb.confidence).max()))
        box_deltas.append(float(np.abs(qa.boxes_xyxy - qb.boxes_xyxy).max()))

        row: dict[str, Any] = {"image": str(img), "flips": {}}
        for t in thresholds:
            sa = qa.confidence >= t
            sb = qb.confidence >= t
            xor = np.logical_xor(sa, sb)
            n_flip = int(xor.sum())
            flips[t] += n_flip
            flips_a_only[t] += int(np.logical_and(sa, ~sb).sum())
            flips_b_only[t] += int(np.logical_and(sb, ~sa).sum())
            det_a[t] += int(sa.sum())
            det_b[t] += int(sb.sum())
            both = np.logical_and(sa, sb)
            class_flips[t] += int(
                (qa.best_class[both] != qb.best_class[both]).sum()
            )
            if n_flip:
                images_with_flip[t] += 1
            # Does top-k truncation bind? If it never does, the decode's
            # maxDetections cap is not participating in any of this and
            # the flip counts are unaffected by it.
            if int(sa.sum()) > max_detections or int(sb.sum()) > max_detections:
                truncation_hits[t] += 1
            row["flips"][str(t)] = n_flip
        per_image.append(row)

        if (idx + 1) % 25 == 0:
            LOG.info("  %d/%d images", idx + 1, len(images))

    def pct(xs: list[float], p: float) -> float:
        return float(np.percentile(xs, p)) if xs else 0.0

    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "executionProviders": sess_a.get_providers(),
            "note": (
                "CPU execution provider. Latency figures here describe this "
                "machine only and license nothing about any other host, "
                "runtime, or accelerator."
            ),
        },
        "artifacts": {
            "a": {
                "path": str(model_a),
                "bytes": model_a.stat().st_size,
                "sha256": sha256_file(model_a),
            },
            "b": {
                "path": str(model_b),
                "bytes": model_b.stat().st_size,
                "sha256": sha256_file(model_b),
            },
        },
        "fixtures": {
            "count": len(images),
            "inputShape": [1, 3, height, width],
            "normalization": {"mean": list(mean), "std": list(std)},
            "resizeNote": (
                "Pillow LANCZOS squash resize; the deployed pipeline uses "
                "sharp/libvips. Both artifacts receive the identical tensor, "
                "so the kernel cancels out of every delta below."
            ),
        },
        "rawOutputs": [accs[n].summarize(tolerances, rel_floor) for n in names_a],
        "decode": {
            "confThresholdRecommended": recommended,
            "maxDetections": max_detections,
            "classCount": len(classes),
            "queriesPerImage": int(qa.confidence.shape[0]),
            "maxConfidenceDeltaAnyQuery": max(conf_deltas) if conf_deltas else 0.0,
            "meanPerImageMaxConfidenceDelta": (
                statistics.fmean(conf_deltas) if conf_deltas else 0.0
            ),
            "maxBoxCoordDeltaAnyQuery": max(box_deltas) if box_deltas else 0.0,
            "byThreshold": [
                {
                    "threshold": t,
                    "isRecommended": t == recommended,
                    "detectionsA": det_a[t],
                    "detectionsB": det_b[t],
                    "thresholdFlips": flips[t],
                    "flipsAOnly": flips_a_only[t],
                    "flipsBOnly": flips_b_only[t],
                    "classArgmaxFlips": class_flips[t],
                    "imagesWithAnyFlip": images_with_flip[t],
                    "imagesWhereTopKTruncationBinds": truncation_hits[t],
                }
                for t in thresholds
            ],
        },
        "latencyMs": {
            "note": (
                "session.run() wall time only — excludes preprocess and "
                "decode. Desktop CPU. NOT an embedded or accelerator number."
            ),
            "a": {"p50": pct(lat_a, 50), "p95": pct(lat_a, 95), "n": len(lat_a)},
            "b": {"p50": pct(lat_b, 50), "p95": pct(lat_b, 95), "n": len(lat_b)},
        },
        "perImage": per_image,
    }


# --------------------------------------------------------------------------
# regression gate
# --------------------------------------------------------------------------

def check_regression(
    report: dict[str, Any],
    baseline: dict[str, Any],
    margin: float = 0.10,
) -> list[str]:
    """Compare a fresh report against a stored one. Returns failures.

    Gates on *regression against a measured baseline*, never against an
    absolute tolerance someone guessed. A metric fails when it is worse
    than the baseline by more than ``margin`` (relative), with a small
    absolute floor so that a baseline of 0 or near-0 does not make every
    comparison fail on noise.
    """
    failures: list[str] = []

    base_raw = {o["name"]: o for o in baseline.get("rawOutputs", [])}
    for out in report.get("rawOutputs", []):
        b = base_raw.get(out["name"])
        if b is None:
            failures.append(f"raw output {out['name']!r} absent from baseline")
            continue
        for key in ("maxAbsDelta", "meanAbsDelta", "p99AbsDelta"):
            new, old = float(out[key]), float(b[key])
            allowed = old * (1.0 + margin) + 1e-12
            if new > allowed:
                failures.append(
                    f"{out['name']}.{key} regressed: {new:.6g} > "
                    f"{allowed:.6g} (baseline {old:.6g} +{margin:.0%})"
                )

    base_thr = {
        t["threshold"]: t for t in baseline.get("decode", {}).get("byThreshold", [])
    }
    for t in report.get("decode", {}).get("byThreshold", []):
        b = base_thr.get(t["threshold"])
        if b is None:
            continue
        for key in ("thresholdFlips", "classArgmaxFlips"):
            new, old = int(t[key]), int(b[key])
            allowed = int(old * (1.0 + margin)) + 1
            if new > allowed:
                failures.append(
                    f"decode@{t['threshold']}.{key} regressed: {new} > "
                    f"{allowed} (baseline {old} +{margin:.0%}, +1 slack)"
                )
    return failures


# --------------------------------------------------------------------------
# human summary
# --------------------------------------------------------------------------

def format_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    a, b = report["artifacts"]["a"], report["artifacts"]["b"]
    lines.append("PARITY REPORT")
    lines.append(f"  A: {Path(a['path']).name}  {a['bytes'] / 1e6:.1f} MB  {a['sha256'][:12]}")
    lines.append(f"  B: {Path(b['path']).name}  {b['bytes'] / 1e6:.1f} MB  {b['sha256'][:12]}")
    lines.append(f"  {report['fixtures']['count']} real images, "
                 f"{report['environment']['onnxruntime']} / "
                 f"{','.join(report['environment']['executionProviders'])}")
    lines.append("")
    lines.append("RAW GRAPH OUTPUTS (before decode)")
    for out in report["rawOutputs"]:
        lines.append(
            f"  {out['name']:<8} n={out['elementsCompared']:,}  "
            f"max|Δ|={out['maxAbsDelta']:.6g}  mean|Δ|={out['meanAbsDelta']:.6g}  "
            f"p99|Δ|={out['p99AbsDelta']:.6g}  max rel={out['maxRelDelta']:.6g}"
        )
        lines.append(
            f"           (rel stats exclude {out['relElementsSkippedBelowFloor']:,} "
            f"elements with |ref| <= {out['relFloor']:g})"
        )
        lines.append(
            f"           {'atol/rtol':>12} {'|Δ|>atol':>12} {'rel>rtol':>12} "
            f"{'isclose':>12} {'AND':>12} {'OR':>12}"
        )
        for ex in out["exceedance"]:
            lines.append(
                f"           {ex['atol']:g}/{ex['rtol']:g}".ljust(24)
                + f"{ex['absOnly']['fraction']:>11.4%} "
                + f"{ex['relOnly']['fraction']:>12.4%} "
                + f"{ex['combinedIsclose']['fraction']:>12.4%} "
                + f"{ex['and']['fraction']:>12.4%} "
                + f"{ex['or']['fraction']:>12.4%}"
            )
    lines.append("")
    d = report["decode"]
    lines.append(
        f"DECODED DETECTIONS (real decode path, {d['queriesPerImage']} queries/image)"
    )
    lines.append(
        f"  max |Δconfidence| on any query: {d['maxConfidenceDeltaAnyQuery']:.6g}   "
        f"max |Δbox coord|: {d['maxBoxCoordDeltaAnyQuery']:.6g}"
    )
    lines.append(
        f"  {'thr':>6} {'det A':>7} {'det B':>7} {'flips':>6} {'A-only':>7} "
        f"{'B-only':>7} {'clsflip':>8} {'imgs':>5}"
    )
    for t in d["byThreshold"]:
        star = " *" if t["isRecommended"] else "  "
        lines.append(
            f"{star}{t['threshold']:>4} {t['detectionsA']:>7} {t['detectionsB']:>7} "
            f"{t['thresholdFlips']:>6} {t['flipsAOnly']:>7} {t['flipsBOnly']:>7} "
            f"{t['classArgmaxFlips']:>8} {t['imagesWithAnyFlip']:>5}"
        )
    lines.append("  * = confThresholdRecommended from the sidecar")
    lines.append("")
    lat = report["latencyMs"]
    lines.append("LATENCY (session.run only; desktop CPU — not an embedded number)")
    lines.append(f"  A p50={lat['a']['p50']:.1f}ms p95={lat['a']['p95']:.1f}ms")
    lines.append(f"  B p50={lat['b']['p50']:.1f}ms p95={lat['b']['p95']:.1f}ms")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compare two ONNX artifacts of the same checkpoint on real "
            "images: raw tensor divergence, then decoded detections at "
            "the operating point."
        )
    )
    p.add_argument("--a", type=Path, required=True, help="Reference artifact (e.g. fp32).")
    p.add_argument("--b", type=Path, required=True, help="Candidate artifact (e.g. fp16).")
    p.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="meta.json sidecar — supplies inputShape, classes, "
             "normalization and confThresholdRecommended.",
    )
    p.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="Manifest of real image paths, one per line "
             "(scripts/build_parity_fixtures.py writes one).",
    )
    p.add_argument("--limit", type=int, default=None, help="Use only the first N fixtures.")
    p.add_argument("--report-json", type=Path, default=None)
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="A previous report JSON. When given, the run gates on "
             "REGRESSION against it and exits non-zero on a regression. "
             "Without it the harness only reports.",
    )
    p.add_argument("--regression-margin", type=float, default=0.10)
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--rel-floor", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for path in (args.a, args.b, args.meta, args.fixtures):
        if not path.exists():
            LOG.error("not found: %s", path)
            return 2

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    images = read_fixture_manifest(args.fixtures)
    if args.limit:
        images = images[: args.limit]
    missing = [i for i in images if not i.exists()]
    if missing:
        LOG.error("%d fixture images are missing, first: %s", len(missing), missing[0])
        return 2

    LOG.info("comparing %s vs %s over %d images", args.a.name, args.b.name, len(images))
    report = compare(
        model_a=args.a,
        model_b=args.b,
        meta=meta,
        images=images,
        max_detections=args.max_detections,
        rel_floor=args.rel_floor,
        warmup=args.warmup,
    )
    report["fixtures"]["manifest"] = str(args.fixtures)
    report["fixtures"]["manifestSha256"] = sha256_file(args.fixtures)

    print()
    print(format_summary(report))
    print()

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        LOG.info("wrote %s", args.report_json)

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        failures = check_regression(report, baseline, args.regression_margin)
        if failures:
            for f in failures:
                LOG.error("REGRESSION: %s", f)
            return 1
        LOG.info("no regression against %s", args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
