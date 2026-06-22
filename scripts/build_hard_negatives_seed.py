"""Build the hard-negatives seed manifest from Wikimedia Commons.

Hard negatives are images of visually-confusing non-litter objects (leaves,
mulch, shadows, painted asphalt) that the V1 model must NOT classify as
litter. Per docs/contributor_assist_goal.md, the corpus must contain at
least 500 such images, all with verified open licenses, and a license
manifest must live next to the corpus.

This script queries Wikimedia Commons' MediaWiki API for files under a
curated list of categories, filters to images with acceptable open
licenses (CC0, Public Domain, CC-BY, CC-BY-SA — any variant), and writes
two artifacts to configs/:

  * configs/hard_negatives_seed.txt        — one image URL per line
  * configs/hard_negatives_licenses.json   — per-URL attribution + license

Re-run periodically to refresh the corpus (the source pool grows over time).

Politeness:
  * 1 req/sec to the MediaWiki API per Wikimedia's published guidance
  * User-Agent identifies this tool + a contact path
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests

log = logging.getLogger(__name__)

WIKI_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "litter-detector-baseline/0.1 hard-negatives-seed "
    "(+https://github.com/adawgwats/litter-detector-baseline; "
    "contact: adawgwats@gmail.com)"
)

REQUEST_INTERVAL_SEC = 1.0
TIMEOUT_SEC = 60

# Licenses we'll accept. Match the LicenseShortName from Wikimedia's
# extmetadata, normalised to lower-case for comparison. The strings here
# come from Wikimedia's published license tags; if you see a new variant
# show up in the manifest's licenses['unmatched'] list, add it here.
ACCEPTED_LICENSES_LOWER: frozenset[str] = frozenset(
    {
        "cc0",
        "public domain",
        "pd",
        "no restrictions",
        "cc by 2.0",
        "cc by 3.0",
        "cc by 4.0",
        "cc by-sa 2.0",
        "cc by-sa 2.5",
        "cc by-sa 3.0",
        "cc by-sa 4.0",
        "attribution",
    }
)

# Curated categories. Add or remove to rebalance the corpus. Each category
# is queried up to ``max_per_category`` files; results are deduplicated by
# URL so categories with overlap don't inflate the count.
DEFAULT_CATEGORIES: list[str] = [
    "Autumn leaves on streets",
    "Fallen leaves",
    "Leaves on the ground",
    "Mulch",
    "Shadows",
    "Shadows on pavement",
    "Asphalt pavement",
    "Cracked asphalt",
    "Wood chips",
    "Storm drains with leaves",
]


# ─── HTTP helper ─────────────────────────────────────────────────────────


_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


def _api_get(params: dict) -> dict:
    """GET the MediaWiki API. Sleeps 1s after each call to stay polite."""
    params = {**params, "format": "json"}
    resp = _session.get(WIKI_API, params=params, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    time.sleep(REQUEST_INTERVAL_SEC)
    return resp.json()


# ─── Category enumeration ────────────────────────────────────────────────


def list_category_files(category: str, *, limit: int) -> Iterator[str]:
    """Yield File: titles under a Commons category. Paginates internally.

    ``limit`` is the soft cap on per-category yields; the API max per page
    is 500 so this almost always finishes in one request.
    """
    yielded = 0
    cmcontinue: str | None = None
    while yielded < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
            "cmlimit": min(500, limit - yielded),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _api_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]
            yielded += 1
            if yielded >= limit:
                return
        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            return


# ─── Metadata + license fetch ────────────────────────────────────────────


def fetch_image_metadata(titles: list[str]) -> dict[str, dict]:
    """Batch-fetch URL + license metadata for up to 50 File: titles.

    Returns a mapping title -> {url, license, attribution}. Files lacking
    a usable imageinfo entry are silently omitted.
    """
    out: dict[str, dict] = {}
    if not titles:
        return out
    params = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "titles": "|".join(titles),
    }
    data = _api_get(params)
    pages = data.get("query", {}).get("pages", {}) or {}
    for page in pages.values():
        title = page.get("title")
        info_list = page.get("imageinfo") or []
        if not title or not info_list:
            continue
        info = info_list[0]
        url = info.get("url")
        mime = info.get("mime") or ""
        if not url or not mime.startswith("image/"):
            continue
        ext = info.get("extmetadata") or {}
        license_short = (ext.get("LicenseShortName") or {}).get("value", "")
        license_url = (ext.get("LicenseUrl") or {}).get("value", "")
        artist_html = (ext.get("Artist") or {}).get("value", "")
        out[title] = {
            "title": title,
            "url": url,
            "mime": mime,
            "width": info.get("width"),
            "height": info.get("height"),
            "license_short": license_short,
            "license_url": license_url,
            "attribution_html": artist_html,
            "source": "wikimedia_commons",
        }
    return out


def _batched(items: Iterable[str], n: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for x in items:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# ─── End-to-end build ────────────────────────────────────────────────────


def build_seed(
    *,
    categories: list[str],
    max_per_category: int,
    target_total: int,
    out_seed: Path,
    out_licenses: Path,
) -> dict:
    """Drive the full seed-build. Returns a stats summary."""
    seen_urls: set[str] = set()
    accepted: list[dict] = []
    rejected_license: list[dict] = []
    unmatched_license_tags: dict[str, int] = {}

    for cat in categories:
        if len(accepted) >= target_total:
            break
        log.info("category=%s (currently %d accepted)", cat, len(accepted))
        titles = list(list_category_files(cat, limit=max_per_category))
        for batch in _batched(titles, 50):
            if len(accepted) >= target_total:
                break
            meta = fetch_image_metadata(batch)
            for record in meta.values():
                if record["url"] in seen_urls:
                    continue
                license_lower = record["license_short"].strip().lower()
                if license_lower not in ACCEPTED_LICENSES_LOWER:
                    rejected_license.append(record)
                    unmatched_license_tags[record["license_short"]] = (
                        unmatched_license_tags.get(record["license_short"], 0) + 1
                    )
                    continue
                seen_urls.add(record["url"])
                accepted.append({**record, "category_hint": cat})
                if len(accepted) >= target_total:
                    break

    out_seed.parent.mkdir(parents=True, exist_ok=True)
    out_seed.write_text("\n".join(r["url"] for r in accepted) + "\n", encoding="utf-8")

    licenses_manifest = {
        "source": "Wikimedia Commons (commons.wikimedia.org)",
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accepted_license_tags": sorted(ACCEPTED_LICENSES_LOWER),
        "category_seeds": categories,
        "accepted_count": len(accepted),
        "accepted": accepted,
        "rejected_for_license_count": len(rejected_license),
        "unmatched_license_tags": unmatched_license_tags,
    }
    out_licenses.write_text(json.dumps(licenses_manifest, indent=2), encoding="utf-8")

    return {
        "accepted": len(accepted),
        "rejected_license": len(rejected_license),
        "categories_used": categories,
        "out_seed": str(out_seed),
        "out_licenses": str(out_licenses),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=600,
                        help="Stop once this many accepted URLs collected (>=500 for gate)")
    parser.add_argument("--max-per-category", type=int, default=200,
                        help="Soft cap on files yielded per category")
    parser.add_argument("--out-seed", type=Path,
                        default=Path("configs/hard_negatives_seed.txt"))
    parser.add_argument("--out-licenses", type=Path,
                        default=Path("configs/hard_negatives_licenses.json"))
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Override default category list")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = build_seed(
        categories=args.categories or DEFAULT_CATEGORIES,
        max_per_category=args.max_per_category,
        target_total=args.target_total,
        out_seed=args.out_seed,
        out_licenses=args.out_licenses,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["accepted"] >= 500 else 1


if __name__ == "__main__":
    sys.exit(main())
