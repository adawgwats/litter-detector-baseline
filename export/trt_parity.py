"""Compare an ONNX artifact against a TensorRT engine built from it.

This is ``export/parity.py`` extended across a runtime boundary. That
harness compares two ONNX graphs through two ONNX Runtime sessions; a
serialized TensorRT plan is not an ONNX graph and ORT will not load one,
so the comparison needs a second runtime on the B side.

Everything that decides *what the numbers mean* is imported from
``export.parity`` rather than reimplemented — ``preprocess_detr``,
``decode_detr_queries``, ``OutputAccumulator`` and the tolerance rules.
That is deliberate: a reimplemented decode would produce numbers that
are merely similar to the CPU ones, and the whole point of a parity
result is that the two sides differ in exactly one variable. Here that
variable is the runtime, not the arithmetic around it.

What this measures, and what it does not:

*   It compares ONE ONNX file against ONE engine built from that exact
    file, with both hashes recorded. It says nothing about an engine
    built from a different ONNX, on a different GPU, or by a different
    TensorRT.
*   Latency here is device execution time — ``execute_async_v3`` plus a
    stream synchronize, host-to-device and device-to-host copies
    excluded. That is the number that describes the engine. It is not
    the number that describes a service, which pays the copies, the
    preprocess, and the dispatch.
*   A desktop GPU result licenses nothing about an embedded target.
    Different memory bandwidth, different power envelope, different
    TensorRT build, frequently a different precision story.

Usage::

    python -m export.trt_parity \\
        --onnx model.onnx --engine model.plan \\
        --meta model.meta.json --fixtures reports/fixtures.txt \\
        --report-json reports/trt-parity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from export.parity import (
    DEFAULT_SWEEP,
    DEFAULT_TOLERANCES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OutputAccumulator,
    decode_detr_queries,
    preprocess_detr,
    read_fixture_manifest,
    sha256_file,
)

LOG = logging.getLogger("export.trt_parity")

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# TensorRT runtime wrapper
# --------------------------------------------------------------------------

class TrtRunner:
    """Load a serialized plan and run it on fixed [1,3,H,W] input.

    Buffers are allocated once and reused, which is what a serving path
    would do and what makes the latency figure meaningful. Device memory
    is freed in ``close()``.
    """

    def __init__(self, plan_path: Path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._trt = trt
        self._cudart = cudart
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {plan_path}")
        self.context = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")

        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, np.dtype] = {}
        self.device: dict[str, int] = {}
        self.host: dict[str, np.ndarray] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            nbytes = int(np.prod(shape)) * dtype.itemsize
            err, ptr = cudart.cudaMalloc(nbytes)
            self._check(err, f"cudaMalloc({name})")
            self.device[name] = int(ptr)
            self.host[name] = np.empty(shape, dtype=dtype)
            self.context.set_tensor_address(name, int(ptr))
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

    def _check(self, err, what: str) -> None:
        if int(err) != 0:
            raise RuntimeError(f"{what} failed: CUDA error {int(err)}")

    def infer(self, tensor: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        """Run once. Returns (outputs, device_execute_ms).

        The timer brackets only ``execute_async_v3`` + synchronize. The
        H2D and D2H copies are outside it on purpose — including them
        would measure PCIe, not the engine.
        """
        cudart = self._cudart
        name_in = self.inputs[0]
        arr = np.ascontiguousarray(tensor, dtype=self.dtypes[name_in])
        err, = cudart.cudaMemcpyAsync(
            self.device[name_in], arr.ctypes.data, arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        )[:1]
        self._check(err, "H2D")
        err, = cudart.cudaStreamSynchronize(self.stream)[:1]
        self._check(err, "sync after H2D")

        t0 = time.perf_counter()
        ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        err, = cudart.cudaStreamSynchronize(self.stream)[:1]
        self._check(err, "sync after execute")
        exec_ms = (time.perf_counter() - t0) * 1000.0

        out: dict[str, np.ndarray] = {}
        for name in self.outputs:
            host = self.host[name]
            err, = cudart.cudaMemcpyAsync(
                host.ctypes.data, self.device[name], host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            )[:1]
            self._check(err, f"D2H({name})")
            err, = cudart.cudaStreamSynchronize(self.stream)[:1]
            self._check(err, f"sync after D2H({name})")
            out[name] = host.copy()
        return out, exec_ms

    def close(self) -> None:
        cudart = self._cudart
        for ptr in self.device.values():
            cudart.cudaFree(ptr)
        self.device.clear()
        try:
            cudart.cudaStreamDestroy(self.stream)
        except Exception:
            pass

    def __enter__(self) -> "TrtRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _ort_session(model_path: Path, providers: Sequence[str]):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=list(providers))


def _pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def compare_onnx_to_engine(
    onnx_path: Path,
    engine_path: Path,
    meta: dict[str, Any],
    images: list[Path],
    *,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    tolerances: Sequence[tuple[float, float]] = DEFAULT_TOLERANCES,
    sweep: Sequence[float] = DEFAULT_SWEEP,
    max_detections: int = 100,
    rel_floor: float = 1e-6,
    warmup: int = 10,
) -> dict[str, Any]:
    import onnxruntime as ort
    import tensorrt as trt

    sess = _ort_session(onnx_path, providers)
    in_name = sess.get_inputs()[0].name
    ort_names = [o.name for o in sess.get_outputs()]

    _, _, height, width = meta["inputShape"]
    classes: list[str] = meta["classes"]
    norm = meta.get("normalization") or {}
    mean = tuple(norm.get("mean", IMAGENET_MEAN))
    std = tuple(norm.get("std", IMAGENET_STD))
    recommended = float(meta["confThresholdRecommended"])
    thresholds = sorted({*(float(t) for t in sweep), recommended})

    runner = TrtRunner(engine_path)
    trt_names = list(runner.outputs)
    common = [n for n in ort_names if n in trt_names]
    if len(common) != len(ort_names):
        raise ValueError(
            f"output names differ: ONNX {ort_names} vs engine {trt_names}"
        )

    accs = {n: OutputAccumulator(n) for n in ort_names}
    lat_onnx: list[float] = []
    lat_trt: list[float] = []

    flips = {t: 0 for t in thresholds}
    flips_a_only = {t: 0 for t in thresholds}
    flips_b_only = {t: 0 for t in thresholds}
    det_a = {t: 0 for t in thresholds}
    det_b = {t: 0 for t in thresholds}
    class_flips = {t: 0 for t in thresholds}
    images_with_flip = {t: 0 for t in thresholds}
    truncation_hits = {t: 0 for t in thresholds}
    conf_deltas: list[float] = []
    box_deltas: list[float] = []
    per_image: list[dict[str, Any]] = []

    dummy = np.zeros((1, 3, height, width), dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {in_name: dummy})
        runner.infer(dummy)

    try:
        for idx, img in enumerate(images):
            tensor = preprocess_detr(img, height, width, mean, std)

            t0 = time.perf_counter()
            out_a_list = sess.run(None, {in_name: tensor})
            lat_onnx.append((time.perf_counter() - t0) * 1000.0)
            out_a = dict(zip(ort_names, out_a_list))

            out_b, exec_ms = runner.infer(tensor)
            lat_trt.append(exec_ms)

            for name in ort_names:
                accs[name].update(out_a[name], out_b[name])

            qa = decode_detr_queries(out_a["dets"], out_a["labels"], len(classes))
            qb = decode_detr_queries(out_b["dets"], out_b["labels"], len(classes))
            conf_deltas.append(float(np.abs(qa.confidence - qb.confidence).max()))
            box_deltas.append(float(np.abs(qa.boxes_xyxy - qb.boxes_xyxy).max()))

            row: dict[str, Any] = {"image": str(img), "flips": {}}
            for t in thresholds:
                sa = qa.confidence >= t
                sb = qb.confidence >= t
                n_flip = int(np.logical_xor(sa, sb).sum())
                flips[t] += n_flip
                flips_a_only[t] += int(np.logical_and(sa, ~sb).sum())
                flips_b_only[t] += int(np.logical_and(sb, ~sa).sum())
                det_a[t] += int(sa.sum())
                det_b[t] += int(sb.sum())
                both = np.logical_and(sa, sb)
                class_flips[t] += int((qa.best_class[both] != qb.best_class[both]).sum())
                if n_flip:
                    images_with_flip[t] += 1
                if int(sa.sum()) > max_detections or int(sb.sum()) > max_detections:
                    truncation_hits[t] += 1
                row["flips"][str(t)] = n_flip
            per_image.append(row)

            if (idx + 1) % 25 == 0:
                LOG.info("  %d/%d images", idx + 1, len(images))
    finally:
        runner.close()

    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison": "onnx (ONNX Runtime) vs tensorrt engine",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "onnxruntimeProviders": sess.get_providers(),
            "tensorrt": trt.__version__,
            "note": (
                "Engine latency is device execute + synchronize only (H2D/D2H "
                "excluded). ONNX latency is session.run wall time and includes "
                "whatever that provider does internally. The two are NOT the "
                "same measurement and are reported separately for that reason."
            ),
        },
        "artifacts": {
            "a": {
                "role": "onnx",
                "path": str(onnx_path),
                "sha256": sha256_file(onnx_path),
                "bytes": onnx_path.stat().st_size,
            },
            "b": {
                "role": "tensorrt-engine",
                "path": str(engine_path),
                "sha256": sha256_file(engine_path),
                "bytes": engine_path.stat().st_size,
            },
        },
        "fixtures": {"count": len(images), "first": str(images[0])},
        "rawOutputs": [
            accs[n].summarize(tolerances, rel_floor=rel_floor) for n in ort_names
        ],
        "decode": {
            "confThresholdRecommended": recommended,
            "maxDetections": max_detections,
            "maxAbsConfidenceDelta": max(conf_deltas) if conf_deltas else 0.0,
            "meanPerImageMaxAbsConfidenceDelta": (
                statistics.fmean(conf_deltas) if conf_deltas else 0.0
            ),
            "maxAbsBoxDelta": max(box_deltas) if box_deltas else 0.0,
            "thresholds": [
                {
                    "threshold": t,
                    "detectionsA": det_a[t],
                    "detectionsB": det_b[t],
                    "thresholdFlips": flips[t],
                    "flipsAOnly": flips_a_only[t],
                    "flipsBOnly": flips_b_only[t],
                    "classFlips": class_flips[t],
                    "imagesAffected": images_with_flip[t],
                    "topKTruncationHits": truncation_hits[t],
                }
                for t in thresholds
            ],
        },
        "latencyMs": {
            "onnxRuntime": {
                "p50": _pct(lat_onnx, 50), "p95": _pct(lat_onnx, 95),
                "min": min(lat_onnx) if lat_onnx else 0.0, "n": len(lat_onnx),
            },
            "tensorrtExecute": {
                "p50": _pct(lat_trt, 50), "p95": _pct(lat_trt, 95),
                "min": min(lat_trt) if lat_trt else 0.0, "n": len(lat_trt),
            },
        },
        "perImage": per_image,
    }


def format_summary(rep: dict[str, Any]) -> str:
    lines: list[str] = []
    a, b = rep["artifacts"]["a"], rep["artifacts"]["b"]
    lines.append(f"A onnx   {a['bytes']:>12,} B  {a['sha256'][:12]}")
    lines.append(f"B engine {b['bytes']:>12,} B  {b['sha256'][:12]}")
    lines.append(f"fixtures {rep['fixtures']['count']}")
    lines.append("")
    lines.append("raw outputs")
    for o in rep["rawOutputs"]:
        lines.append(
            f"  {o['name']:<8} n={o['elementsCompared']:>9,}  "
            f"maxAbs={o['maxAbsDelta']:.6g}  meanAbs={o['meanAbsDelta']:.6g}  "
            f"p99={o['p99AbsDelta']:.6g}  maxRel={o['maxRelDelta']:.6g}"
        )
        for e in o["exceedance"]:
            lines.append(
                f"    atol/rtol {e['atol']:.0e}: "
                f"AND {e['and']['fraction']*100:6.2f}%  "
                f"OR {e['or']['fraction']*100:6.2f}%  "
                f"isclose {e['combinedIsclose']['fraction']*100:6.2f}%"
            )
    lines.append("")
    d = rep["decode"]
    lines.append(
        f"decode  maxAbsConfDelta={d['maxAbsConfidenceDelta']:.6g}  "
        f"maxAbsBoxDelta={d['maxAbsBoxDelta']:.6g}"
    )
    lines.append("  thr    detA   detB   flips  A-only  B-only  classFlips  images")
    for t in d["thresholds"]:
        mark = " <-" if t["threshold"] == d["confThresholdRecommended"] else ""
        lines.append(
            f"  {t['threshold']:<5} {t['detectionsA']:>6} {t['detectionsB']:>6} "
            f"{t['thresholdFlips']:>7} {t['flipsAOnly']:>7} {t['flipsBOnly']:>7} "
            f"{t['classFlips']:>11} {t['imagesAffected']:>7}{mark}"
        )
    lines.append("")
    lat = rep["latencyMs"]
    lines.append(
        f"latency  onnx p50={lat['onnxRuntime']['p50']:.2f}ms "
        f"p95={lat['onnxRuntime']['p95']:.2f}ms | "
        f"trt-exec p50={lat['tensorrtExecute']['p50']:.2f}ms "
        f"p95={lat['tensorrtExecute']['p95']:.2f}ms"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--report-json", type=Path, default=None)
    p.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime execution providers for side A.",
    )
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--rel-floor", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    images = read_fixture_manifest(args.fixtures)
    if args.limit:
        images = images[: args.limit]

    rep = compare_onnx_to_engine(
        args.onnx,
        args.engine,
        meta,
        images,
        providers=[s.strip() for s in args.providers.split(",") if s.strip()],
        max_detections=args.max_detections,
        rel_floor=args.rel_floor,
        warmup=args.warmup,
    )
    print(format_summary(rep))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        LOG.info("report: %s", args.report_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
