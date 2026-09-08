#!/usr/bin/env python3
"""Produce the fp16 counterpart of an exported fp32 ONNX artifact.

Deliberately thin: it runs the ``rfdetr-s-512-fp16-repaired`` descriptor
through ``export.precision.convert`` — the same shared conversion path
both exporters use — so what the parity harness measures is the
conversion this repository actually performs, not a re-implementation of
it that might differ. If the descriptor or the shared path changes, the
measured artifact changes with it, which is the point.

The descriptor this uses differs from the one ``export_rfdetr.export()``
ships (``rfdetr-s-512-fp16``) in exactly two fields: ``repair`` and
``validate``. That difference is why the artifact below loads in ONNX
Runtime and the shipped one does not — see reports/CONVERSION-REPORT.md
§2. Keeping it visible as two descriptors is deliberate.

Nothing here touches a deployed artifact: it writes a new file to
``--out`` and never modifies the source.

Usage::

    python scripts/make_fp16.py \\
        --in  roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.onnx \\
        --out dist/parity/rfdetr-s-litter.fp16.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
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
        "--no-repair",
        action="store_true",
        help="Skip export.fp16_repair. The raw converter output does not "
             "load in ONNX Runtime for this graph — use this only to "
             "reproduce that failure.",
    )
    args = p.parse_args(argv)

    if not args.src.exists():
        print(f"not found: {args.src}", file=sys.stderr)
        return 2
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if args.dst.resolve() == args.src.resolve():
        print("refusing to convert in place — --out must differ", file=sys.stderr)
        return 2

    print(f"source {args.src}  {args.src.stat().st_size:,} bytes")
    print(f"  sha256 {sha256(args.src)}")

    target = get_target("rfdetr-s-512-fp16-repaired")
    conversion = plan(target)
    print(f"  {conversion.describe()}")

    # `validate` stays on even with --no-repair: the point of that flag
    # is to reproduce the load failure, and the gate is what shows it.
    result = convert(
        conversion, args.src, args.dst, repair=not args.no_repair
    )

    if result.duplicate_node_names:
        print(
            f"  duplicate node names after conversion: "
            f"{result.duplicate_node_names}"
        )
    if args.no_repair:
        print("  --no-repair: leaving converter output as-is")
    elif result.repaired:
        print(
            f"  repaired: removed {len(result.removed_degenerate_casts)} "
            f"degenerate cast(s), retyped "
            f"{len(result.retyped_float_casts)} Cast(to=FLOAT) node(s)"
        )
        for n in result.removed_degenerate_casts:
            print(f"    removed {n}")
    else:
        print("  repair was a no-op — converter output needed nothing")

    # The gate. A model onnx.checker accepts can still fail here, which
    # is precisely how both defects reached this repo unnoticed.
    if result.validated:
        print("  validated: ONNX Runtime builds a session from this artifact")

    print(f"wrote  {args.dst}  {result.output_bytes:,} bytes")
    print(f"  sha256 {sha256(args.dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
