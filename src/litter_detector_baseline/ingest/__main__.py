"""CLI entry point for corpus ingestion.

Usage:
    python -m litter_detector_baseline.ingest verify-olm-tags
    python -m litter_detector_baseline.ingest olm-bbox \\
        --bbox -77.65,38.13,-77.38,38.40 \\
        --year 2024 \\
        --max-photos 100 \\
        --manifest out/olm-manifest.json

Required env vars (see ingest/storage.py for details):
    S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL, S3_BUCKET
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import openlittermap as olm
from .storage import Storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # verify-olm-tags
    sub.add_parser("verify-olm-tags", help="Sample OLM result_string format from a dense bbox")

    # olm-bbox
    p = sub.add_parser("olm-bbox", help="Ingest OpenLitterMap photos in a bbox + year")
    p.add_argument("--bbox", required=True, help="left,bottom,right,top in lon/lat (e.g. -77.65,38.13,-77.38,38.40)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--max-photos", type=int, default=None)
    p.add_argument("--manifest", type=Path, default=None, help="path to write JSON manifest")
    p.add_argument("--key-prefix", default="openlittermap/photos")
    p.add_argument("--user-agent", default=olm.DEFAULT_USER_AGENT)
    p.add_argument("-v", "--verbose", action="store_true")

    # olm-tags
    p2 = sub.add_parser("olm-tags", help="Download OLM's full 200+ class tag taxonomy")
    p2.add_argument("--out", type=Path, default=Path("olm-tags.json"))
    p2.add_argument("--user-agent", default=olm.DEFAULT_USER_AGENT)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "verify-olm-tags":
        samples = olm.verify_result_string_format()
        print(json.dumps(samples, indent=2))
        return 0

    if args.cmd == "olm-tags":
        tax = olm.fetch_tag_taxonomy(user_agent=args.user_agent)
        args.out.write_text(json.dumps(tax, indent=2))
        print(f"wrote {len(json.dumps(tax))} bytes to {args.out}")
        return 0

    if args.cmd == "olm-bbox":
        bbox_parts = [float(x) for x in args.bbox.split(",")]
        if len(bbox_parts) != 4:
            print("--bbox must be 4 comma-separated floats: left,bottom,right,top", file=sys.stderr)
            return 2
        storage = Storage.from_env()
        manifest = olm.ingest_bbox(
            bbox=tuple(bbox_parts),
            year=args.year,
            storage=storage,
            user_agent=args.user_agent,
            key_prefix=args.key_prefix,
            manifest_path=args.manifest,
            max_photos=args.max_photos,
        )
        print(f"ingested {len(manifest.photos)} photos; {len(manifest.contributors)} unique contributors")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
