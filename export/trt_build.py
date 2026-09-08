"""Build a TensorRT engine from an ONNX artifact, and record its identity.

An engine is not identified by its filename. It is identified by the
tuple of everything that could have changed its bytes: the source graph,
the builder configuration, the toolchain versions, and the GPU the
builder tuned against. This module emits that tuple alongside every plan
it writes, because a model release registry that stores only
``rfdetr-s-fp16.plan`` cannot answer the question "is the artifact on
this machine the one we validated?".

Two things worth knowing before reading the flags below.

**TensorRT 11 removed the FP16 and INT8 builder flags.** In TRT 8/10 you
built one network and asked the builder for a precision. In TRT 11 the
only ``NetworkDefinitionCreationFlag`` concerning types is
``STRONGLY_TYPED``, and precision is carried by the ONNX graph itself.
So "build an fp16 engine" becomes "convert the ONNX to fp16, then
build", which makes precision an artifact-level decision rather than a
builder-level one — and means the fp16 conversion defects already
documented for this graph sit upstream of TensorRT rather than being
bypassed by it.

**TF32 is still a builder flag and it is on by default.** On Ada that
means a nominally "fp32" engine performs 10-bit-mantissa matmuls unless
the flag is cleared. That is a precision decision hiding inside a
default, so this module always records the resolved flag set rather than
assuming one.

Usage::

    python -m export.trt_build --onnx model.onnx --out model.plan
    python -m export.trt_build --onnx model.onnx --out m.plan --no-tf32
    python -m export.trt_build --onnx model.onnx --out m.plan \\
        --timing-cache-out cache.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("export.trt_build")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def gpu_identity() -> dict[str, Any]:
    """The half of an engine's identity that lives in the builder host."""
    import tensorrt as trt

    info: dict[str, Any] = {
        "tensorrt": trt.__version__,
        "driver": _driver_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        from cuda.bindings import runtime as cudart

        err, props = cudart.cudaGetDeviceProperties(0)
        if int(err) == 0:
            name = props.name
            info["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
            info["computeCapability"] = f"{props.major}.{props.minor}"
            info["multiProcessorCount"] = int(props.multiProcessorCount)
            info["totalGlobalMemBytes"] = int(props.totalGlobalMem)
        err, rt = cudart.cudaRuntimeGetVersion()
        if int(err) == 0:
            info["cudaRuntime"] = f"{rt // 1000}.{(rt % 1000) // 10}"
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["gpuProbeError"] = repr(exc)
    return info


def _flags_set(config, trt) -> list[str]:
    names = [
        n
        for n in dir(trt.BuilderFlag)
        if not n.startswith("_") and n not in ("name", "value")
    ]
    out = []
    for n in names:
        try:
            if config.get_flag(getattr(trt.BuilderFlag, n)):
                out.append(n)
        except Exception:
            continue
    return sorted(out)


def build_engine(
    onnx_path: Path,
    *,
    tf32: bool = True,
    workspace_bytes: int | None = None,
    timing_cache_in: Path | None = None,
    timing_cache_out: Path | None = None,
    strongly_typed: bool = False,
    optimization_level: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Parse ONNX, build a serialized plan, return ``(plan_bytes, provenance)``."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)

    flags = 0
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(
            "ONNX parse failed. TensorRT reported:\n  " + "\n  ".join(errs)
        )

    config = builder.create_builder_config()
    if workspace_bytes is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if optimization_level is not None:
        config.builder_optimization_level = optimization_level
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)

    if timing_cache_in is not None:
        raw = timing_cache_in.read_bytes() if timing_cache_in.exists() else b""
        cache = config.create_timing_cache(raw)
        config.set_timing_cache(cache, ignore_mismatch=False)
        LOG.info("timing cache supplied: %d bytes", len(raw))
    elif timing_cache_out is not None:
        cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)

    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if plan is None:
        raise RuntimeError("build_serialized_network returned None")
    plan_bytes = bytes(plan)

    if timing_cache_out is not None:
        cache_obj = config.get_timing_cache()
        if cache_obj is not None:
            timing_cache_out.parent.mkdir(parents=True, exist_ok=True)
            timing_cache_out.write_bytes(bytes(cache_obj.serialize()))
            LOG.info("timing cache written: %s", timing_cache_out)

    provenance: dict[str, Any] = {
        "builtAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "onnx": str(onnx_path),
            "onnxSha256": sha256_file(onnx_path),
            "onnxBytes": onnx_path.stat().st_size,
        },
        "builder": {
            "tf32": tf32,
            "stronglyTyped": strongly_typed,
            "workspaceBytes": workspace_bytes,
            "builderOptimizationLevel": int(config.builder_optimization_level),
            "avgTimingIterations": int(config.avg_timing_iterations),
            "flagsSet": _flags_set(config, trt),
            "timingCacheIn": str(timing_cache_in) if timing_cache_in else None,
            "timingCacheInBytes": (
                timing_cache_in.stat().st_size
                if timing_cache_in and timing_cache_in.exists()
                else None
            ),
        },
        "environment": gpu_identity(),
        "plan": {"bytes": len(plan_bytes), "sha256": sha256_bytes(plan_bytes)},
        "buildSeconds": round(build_s, 2),
        "network": {
            "numLayers": network.num_layers,
            "inputs": [
                {
                    "name": network.get_input(i).name,
                    "shape": list(network.get_input(i).shape),
                    "dtype": str(network.get_input(i).dtype),
                }
                for i in range(network.num_inputs)
            ],
            "outputs": [
                {
                    "name": network.get_output(i).name,
                    "shape": list(network.get_output(i).shape),
                    "dtype": str(network.get_output(i).dtype),
                }
                for i in range(network.num_outputs)
            ],
        },
    }
    return plan_bytes, provenance


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-tf32", action="store_true", help="Clear the TF32 flag.")
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--timing-cache-in", type=Path, default=None)
    p.add_argument("--timing-cache-out", type=Path, default=None)
    p.add_argument("--strongly-typed", action="store_true")
    p.add_argument("--optimization-level", type=int, default=None)
    p.add_argument("--provenance-json", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.onnx.exists():
        LOG.error("onnx not found: %s", args.onnx)
        return 2

    plan, prov = build_engine(
        args.onnx,
        tf32=not args.no_tf32,
        workspace_bytes=args.workspace_mb * 1024 * 1024,
        timing_cache_in=args.timing_cache_in,
        timing_cache_out=args.timing_cache_out,
        strongly_typed=args.strongly_typed,
        optimization_level=args.optimization_level,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(plan)
    prov["plan"]["path"] = str(args.out)

    dest = args.provenance_json or args.out.with_suffix(
        args.out.suffix + ".provenance.json"
    )
    dest.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")

    LOG.info(
        "plan: %s (%d bytes) sha256=%s",
        args.out,
        prov["plan"]["bytes"],
        prov["plan"]["sha256"][:16],
    )
    LOG.info(
        "built in %.1fs on %s", prov["buildSeconds"], prov["environment"].get("gpu", "?")
    )
    LOG.info("provenance: %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
