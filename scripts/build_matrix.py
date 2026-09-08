#!/usr/bin/env python3
"""Walk the declared target matrix for one source artifact.

The point this exists to demonstrate: a target is DATA, not a code path.
Every row below is produced by the same three calls — get_target, plan,
convert — differing only in a dict entry in export.precision.TARGETS. Adding
a fourth precision, or a third architecture, is a new descriptor and not a
new script.

That is the difference between a conversion framework and a drawer full of
one-off scripts, and it is the whole reason precision policy was lifted out
of the two exporters into one place.

Deliberately NOT included: TensorRT engine targets. Those cannot be built on
a machine with no GPU, and a matrix that silently omits the targets it could
not build would be lying by construction. Rows it cannot build are reported
as BLOCKED, with the reason.
"""
from __future__ import annotations

import argparse, json, sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export.precision import TARGETS, convert, get_target, plan, UnsupportedTargetError  # noqa: E402
from export.registry import sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--architecture", default="rfdetr")
    ap.add_argument("--outdir", type=Path, default=Path("dist/matrix"))
    ap.add_argument("--report", type=Path, default=Path("reports/target-matrix.json"))
    a = ap.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    src_sha = sha256_file(a.source)
    src_bytes = a.source.stat().st_size
    print(f"source {a.source.name}  {src_bytes:,} B  sha {src_sha[:12]}\n")

    rows = []
    names = [n for n, t in TARGETS.items() if t.architecture == a.architecture]
    for name in names:
        t = get_target(name)
        row: dict = {
            "target": name, "architecture": t.architecture,
            "precision": str(t.precision.value), "io": str(t.io_precision.value),
            "mechanism": t.mechanism.value if t.mechanism else None,
            "opset": t.opset, "inputShape": list(t.input_shape),
            "maxArtifactBytes": t.max_artifact_bytes,
            "repair": t.repair, "validate": t.validate,
        }
        try:
            p = plan(t)
            if not p.post_export and p.backend_kwargs.get("half") in (None, False) \
                    and t.precision.value == "fp32":
                # fp32 IS the source; no conversion to perform.
                dst, out_sha, out_bytes = a.source, src_sha, src_bytes
                row["action"] = "source (no conversion)"
            else:
                dst = a.outdir / f"{name}.onnx"
                shutil.copyfile(a.source, dst)
                convert(p, dst, dst)
                out_sha, out_bytes = sha256_file(dst), dst.stat().st_size
                row["action"] = "converted"
            row.update(status="ok", bytes=out_bytes, sha256=out_sha,
                       pctOfSource=round(out_bytes / src_bytes * 100, 1),
                       withinBudget=(t.max_artifact_bytes is None
                                     or out_bytes <= t.max_artifact_bytes))
        except UnsupportedTargetError as exc:
            row.update(status="refused", reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            row.update(status="blocked", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        st = row["status"]
        size = f"{row.get('bytes', 0):>12,}" if st == "ok" else " " * 12
        pct = f"{row.get('pctOfSource', 0):5.1f}%" if st == "ok" else "      "
        print(f"{name:28s} {row['precision']:5s} io={row['io']:14s} "
              f"{size} {pct}  {st}"
              + ("" if st == "ok" else f"  <- {row.get('reason','')[:60]}"))

    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(
        {"source": {"path": str(a.source), "bytes": src_bytes, "sha256": src_sha},
         "architecture": a.architecture, "targets": rows}, indent=2))
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} targets built from one source, one code path.")
    print(f"wrote {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
