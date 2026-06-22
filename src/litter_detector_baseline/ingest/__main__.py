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

from . import hard_negatives as hn_mod
from . import olm_csv as olm_csv_mod
from . import openlittermap as olm
from . import taco as taco_mod
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

    # taco
    p3 = sub.add_parser("taco", help="Ingest TACO (CC-BY-4.0) images + annotations")
    p3.add_argument("--dry-run", action="store_true",
                    help="Enumerate + crosswalk only; do not download images or write to S3")
    p3.add_argument("--annotations", type=Path, default=None,
                    help="Path to a local annotations.json; if omitted, fetched + cached under data/taco/")
    p3.add_argument("--max-photos", type=int, default=None)
    p3.add_argument("--manifest", type=Path, default=None, help="Path to write JSON manifest")
    p3.add_argument("--key-prefix", default="taco/photos")
    p3.add_argument("--image-url-field", default="flickr_640_url",
                    choices=("flickr_640_url", "flickr_url"))
    p3.add_argument("--user-agent", default=taco_mod.DEFAULT_USER_AGENT)
    p3.add_argument("--workers", type=int, default=8,
                    help="Concurrent download+upload workers (default 8)")
    p3.add_argument("-v", "--verbose", action="store_true")

    # hard-negatives
    p4 = sub.add_parser(
        "hard-negatives",
        help="Download CC0/CC-BY hard-negatives seeded by scripts/build_hard_negatives_seed.py",
    )
    p4.add_argument("--dry-run", action="store_true")
    p4.add_argument("--seed", type=Path, default=hn_mod.DEFAULT_SEED_PATH)
    p4.add_argument("--licenses", type=Path, default=hn_mod.DEFAULT_LICENSES_PATH)
    p4.add_argument("--max-photos", type=int, default=None)
    p4.add_argument("--manifest", type=Path, default=None)
    p4.add_argument("--key-prefix", default="hard_negatives/photos")
    p4.add_argument("--user-agent", default=hn_mod.DEFAULT_USER_AGENT)
    p4.add_argument("--workers", type=int, default=8,
                    help="Concurrent download+upload workers (default 8)")
    p4.add_argument("-v", "--verbose", action="store_true")

    # olm-csv
    p5 = sub.add_parser(
        "olm-csv",
        help="Ingest an OpenLitterMap CSV export (from POST /api/download)",
    )
    p5.add_argument("--csv", type=Path, required=True,
                    help="Path to the OLM 'number-based' CSV export")
    p5.add_argument("--dry-run", action="store_true",
                    help="Parse + crosswalk only; do not fetch images or write to R2")
    p5.add_argument("--max-photos", type=int, default=None)
    p5.add_argument("--manifest", type=Path, default=None)
    p5.add_argument("--key-prefix", default="openlittermap/photos")
    p5.add_argument("--user-agent", default=olm_csv_mod.DEFAULT_USER_AGENT)
    p5.add_argument("--workers", type=int, default=4,
                    help="Concurrent workers (default 4; OLM caps at ~3 concurrent image downloads)")
    p5.add_argument("-v", "--verbose", action="store_true")

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

    if args.cmd == "taco":
        storage = None if args.dry_run else Storage.from_env()
        manifest = taco_mod.ingest_taco(
            storage=storage,
            annotations_path=args.annotations,
            user_agent=args.user_agent,
            key_prefix=args.key_prefix,
            manifest_path=args.manifest,
            max_photos=args.max_photos,
            dry_run=args.dry_run,
            image_url_field=args.image_url_field,
            workers=args.workers,
        )
        # One line per image with its crosswalked labels — satisfies the
        # Step 3 gate "lists the 1500 images and their crosswalked labels".
        for img in manifest.images:
            leaves = [a["olm_leaf"] for a in img["annotations"]]
            print(f"{img['image_id']:5d}  {img['file_name']:30s}  {leaves}")
        print(
            f"\n{manifest.image_count} images, {manifest.annotation_count} annotations, "
            f"{len(manifest.olm_leaf_histogram)} distinct OLM leaves"
        )
        if args.dry_run:
            print("(dry-run: no images downloaded, no storage writes)")
        return 0

    if args.cmd == "olm-csv":
        storage = None if args.dry_run else Storage.from_env()
        if args.dry_run:
            # Pure parser test — no storage required
            from .olm_csv import parse_csv  # noqa: PLC0415
            recs = list(parse_csv(args.csv))
            print(f"parsed {len(recs)} photos")
            from collections import Counter
            leaf_hist: Counter[str] = Counter()
            for r in recs:
                for leaf, cnt in r.leaves:
                    leaf_hist[leaf] += cnt
            print(f"distinct OLM leaves: {len(leaf_hist)}")
            print("top 20 by count:")
            for leaf, cnt in leaf_hist.most_common(20):
                print(f"  {leaf:35s} {cnt}")
            return 0
        manifest = olm_csv_mod.ingest_olm_csv(
            csv_path=args.csv,
            storage=storage,
            user_agent=args.user_agent,
            key_prefix=args.key_prefix,
            manifest_path=args.manifest,
            max_photos=args.max_photos,
            workers=args.workers,
        )
        print(
            f"{manifest.image_count} ingested, {manifest.skipped_count} skipped, "
            f"{manifest.failed_count} failed; "
            f"{len(manifest.olm_leaf_histogram)} distinct OLM leaves"
        )
        return 0

    if args.cmd == "hard-negatives":
        storage = None if args.dry_run else Storage.from_env()
        manifest = hn_mod.ingest_hard_negatives(
            storage=storage,
            seed_path=args.seed,
            licenses_path=args.licenses,
            user_agent=args.user_agent,
            key_prefix=args.key_prefix,
            manifest_path=args.manifest,
            max_photos=args.max_photos,
            dry_run=args.dry_run,
            workers=args.workers,
        )
        print(
            f"{manifest.image_count} hard-negative images; "
            f"licenses: {dict(sorted(manifest.license_histogram.items()))}; "
            f"failed={manifest.failed_count} skipped={manifest.skipped_count}"
        )
        if args.dry_run:
            print("(dry-run: no images downloaded, no storage writes)")
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
