#!/usr/bin/env python3
"""Produce an INT8 counterpart of an exported fp32 ONNX artifact.

Uses ONNX Runtime's *dynamic* quantization, which quantizes weights
ahead of time and picks activation scales at run time. That distinction
matters here: ``export_rfdetr`` refuses ``-int8`` versions with "INT8
export requires a calibration pass", which is true of *static* PTQ and
not of this. No calibration set is needed, so this path was reachable
the whole time.

Like scripts/make_fp16.py, nothing here touches a deployed artifact: it
writes a new file to ``--out`` and never modifies the source.

Usage::

    python scripts/make_int8.py \\
        --in  ../roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.onnx \\
        --out dist/parity/rfdetr-s-litter.int8-dyn.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export.precision import convert, get_target, plan  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, required=True)
    p.add_argument(
        "--weight-type",
        choices=("int8", "uint8"),
        default="int8",
        help="Weight quantization type. Default int8 (QInt8).",
    )
    args = p.parse_args(argv)

    if not args.src.exists():
        print(f"not found: {args.src}", file=sys.stderr)
        return 2
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if args.dst.resolve() == args.src.resolve():
        print("refusing to convert in place — --out must differ", file=sys.stderr)
        return 2

    src_bytes = args.src.stat().st_size
    print(f"source {args.src}  {src_bytes:,} bytes")
    print(f"  sha256 {sha256(args.src)}")

    # Same shared conversion path the exporters use; the only thing that
    # differs is the mechanism named on the descriptor.
    target = get_target("rfdetr-s-512-int8-dynamic").derive(
        quant_weight_type=args.weight_type
    )
    conversion = plan(target)
    print(f"  {conversion.describe()}")

    t0 = time.perf_counter()
    # The gate the fp16 work argued for is on the descriptor
    # (validate=True), applied here on day one rather than a month
    # later. onnx.checker is not a substitute: it accepts graphs ONNX
    # Runtime refuses to build a session from.
    result = convert(conversion, args.src, args.dst)
    elapsed = time.perf_counter() - t0

    dst_bytes = result.output_bytes
    print(f"  quantized ({args.weight_type}) in {elapsed:.1f}s")
    if result.validated:
        print("  validated: ONNX Runtime builds a session from this artifact")

    print(f"wrote  {args.dst}  {dst_bytes:,} bytes "
          f"({dst_bytes / src_bytes:.1%} of source)")
    print(f"  sha256 {sha256(args.dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
