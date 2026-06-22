"""Object storage interface for corpus ingest + retrieval.

Two backends, selected at :meth:`Storage.from_env` time:

1. **S3 protocol (boto3, SigV4)** — for AWS S3, Backblaze B2, MinIO, and
   for R2 *when an R2-specific access key pair is available*. This is the
   path EC2 spot training instances will use (per the locked decision in
   `reference_ml_training_infra.md` — instance profile reads R2 access
   key from AWS Secrets Manager and passes via env vars).

2. **Cloudflare R2 management API (Bearer auth)** — for local-dev ingest
   using the existing `Workers R2 Storage:Edit` CF API token. The token
   can read/write objects via Cloudflare's REST API without an S3
   credential pair. This bypasses the need to mint a separate R2 S3
   access key for local work.

Backend selection: set R2_API_TOKEN + R2_ACCOUNT_ID + R2_BUCKET to use the
CF mgmt-API path. Else fall back to S3_BUCKET + S3_ACCESS_KEY_ID +
S3_SECRET_ACCESS_KEY + S3_ENDPOINT_URL for the S3 path.

Both backends present the same interface — :meth:`put_bytes`,
:meth:`put_file`, :meth:`exists`, :meth:`get_bytes` — so ingest modules
don't care which is in use.

Required env vars (S3 backend):
    S3_ACCESS_KEY_ID         — access key
    S3_SECRET_ACCESS_KEY     — secret key
    S3_ENDPOINT_URL          — e.g. https://<account>.r2.cloudflarestorage.com
                               (omit for AWS S3 default)
    S3_BUCKET                — bucket name
    S3_REGION                — optional, defaults to 'auto' (works for R2)

Required env vars (CF R2 management-API backend):
    R2_API_TOKEN             — CF API token with Workers R2 Storage:Edit
    R2_ACCOUNT_ID            — 32-hex Cloudflare account ID
    R2_BUCKET                — bucket name (e.g. dregsbane-beta-ml-corpus)
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import boto3
import requests
from botocore.client import Config
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

CF_API_BASE = "https://api.cloudflare.com/client/v4"

# CF management API rate limits hit ~1200 req/min/account in practice; with
# 8 concurrent workers we burned through that on big ingests. Retry policy:
# exponential backoff with jitter, honor server's Retry-After when present.
_CF_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_CF_MAX_RETRIES = 10
_CF_BASE_BACKOFF_SEC = 1.5
_CF_MAX_BACKOFF_SEC = 60.0  # Cap Retry-After honoring; CF sometimes asks for
                            # 300s but a 60s cooldown is enough in practice.


def _cf_request_with_backoff(method: str, url: str, **kwargs) -> requests.Response:
    """Call requests.<method>(url, **kwargs) with exponential backoff on
    retryable status codes (429, 5xx). Honors Retry-After when present
    (capped at _CF_MAX_BACKOFF_SEC). Raises after _CF_MAX_RETRIES."""
    backoff = _CF_BASE_BACKOFF_SEC
    for attempt in range(_CF_MAX_RETRIES):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code not in _CF_RETRY_STATUSES:
            return resp
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            sleep_for = min(float(retry_after), _CF_MAX_BACKOFF_SEC)
        else:
            sleep_for = backoff + random.uniform(0, backoff * 0.3)
        log.warning(
            "CF mgmt API %d on %s %s (attempt %d/%d) — sleeping %.1fs",
            resp.status_code, method, url, attempt + 1, _CF_MAX_RETRIES, sleep_for,
        )
        time.sleep(sleep_for)
        backoff = min(backoff * 2, _CF_MAX_BACKOFF_SEC)
    return resp  # last response (will likely fail at the raise_for_status caller)


@dataclass
class Storage:
    """Object storage wrapper supporting S3-protocol AND CF R2 management API.

    See module docstring for backend selection logic.
    """

    bucket: str

    # S3-backend fields (populated for SigV4 path)
    endpoint_url: Optional[str] = None
    region: Optional[str] = None

    # Cloudflare management-API-backend fields (populated for Bearer path)
    cf_account_id: Optional[str] = None
    cf_api_token: Optional[str] = field(default=None, repr=False)  # never log

    @property
    def is_cf_mgmt_backend(self) -> bool:
        """True when configured to use the CF management API instead of S3."""
        return bool(self.cf_api_token and self.cf_account_id)

    @classmethod
    def from_env(cls) -> "Storage":
        """Construct from environment variables. Prefers CF mgmt backend when
        the R2_* trio is set; otherwise falls back to S3 protocol."""
        cf_token = os.environ.get("R2_API_TOKEN")
        cf_account = os.environ.get("R2_ACCOUNT_ID")
        cf_bucket = os.environ.get("R2_BUCKET")
        if cf_token and cf_account and cf_bucket:
            return cls(
                bucket=cf_bucket,
                cf_account_id=cf_account,
                cf_api_token=cf_token,
            )
        try:
            bucket = os.environ["S3_BUCKET"]
        except KeyError as exc:
            raise RuntimeError(
                "Storage requires either the CF-mgmt-backend env vars "
                "(R2_API_TOKEN + R2_ACCOUNT_ID + R2_BUCKET) or the S3 trio "
                "(S3_BUCKET + S3_ACCESS_KEY_ID + S3_SECRET_ACCESS_KEY). "
                "See docs/ingestion.md."
            ) from exc
        endpoint = os.environ.get("S3_ENDPOINT_URL") or None
        region = os.environ.get("S3_REGION", "auto")
        return cls(bucket=bucket, endpoint_url=endpoint, region=region)

    # ─── S3 backend helpers ─────────────────────────────────────────────

    def _s3(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region,
            config=Config(signature_version="s3v4"),
        )

    # ─── CF management-API backend helpers ──────────────────────────────

    def _cf_object_url(self, key: str) -> str:
        # URL-quote the key so '/' shards land as literal path components and
        # other safe-special characters travel through the gateway unmodified.
        from urllib.parse import quote
        safe_key = quote(key, safe="/")
        return (
            f"{CF_API_BASE}/accounts/{self.cf_account_id}"
            f"/r2/buckets/{self.bucket}/objects/{safe_key}"
        )

    def _cf_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cf_api_token}"}

    # ─── public interface ──────────────────────────────────────────────

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        if self.is_cf_mgmt_backend:
            r = _cf_request_with_backoff(
                "PUT",
                self._cf_object_url(key),
                headers={**self._cf_headers(), "Content-Type": content_type},
                data=data,
                timeout=180,
            )
            r.raise_for_status()
            return
        self._s3().put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None:
        if self.is_cf_mgmt_backend:
            self.put_bytes(key, path.read_bytes(), content_type)
            return
        with path.open("rb") as f:
            self._s3().put_object(
                Bucket=self.bucket, Key=key, Body=f, ContentType=content_type
            )

    def exists(self, key: str) -> bool:
        if self.is_cf_mgmt_backend:
            # CF management API doesn't allow HEAD on /objects/{key} (returns
            # 405). Use a GET with a one-byte Range header — fetches at most
            # 1 byte to confirm existence, with negligible cost vs a HEAD.
            r = _cf_request_with_backoff(
                "GET",
                self._cf_object_url(key),
                headers={**self._cf_headers(), "Range": "bytes=0-0"},
                timeout=30,
            )
            if r.status_code in (200, 206):
                return True
            if r.status_code in (404, 410):
                return False
            r.raise_for_status()
            return False  # unreachable; raise_for_status fires on 4xx/5xx
        try:
            self._s3().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get_bytes(self, key: str) -> bytes:
        if self.is_cf_mgmt_backend:
            r = _cf_request_with_backoff(
                "GET",
                self._cf_object_url(key),
                headers=self._cf_headers(),
                timeout=180,
            )
            r.raise_for_status()
            return r.content
        resp = self._s3().get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()


def content_addressed_key(prefix: str, data: bytes, ext: str) -> str:
    """Build a content-addressable object key.

    Uses sha256 of the file bytes so re-uploading the same image is a no-op
    (idempotent ingest). Prefix-then-hash structure means downstream tools
    can list a prefix to enumerate everything from one source.

    Example: ``content_addressed_key("openlittermap/photos", b"...", "jpg")``
    yields ``"openlittermap/photos/ab/cd/abcd1234...jpg"`` (sha256 hex,
    sharded by first 2 + next 2 chars for filesystem-friendly listing).
    """
    h = hashlib.sha256(data).hexdigest()
    return f"{prefix.rstrip('/')}/{h[:2]}/{h[2:4]}/{h}.{ext.lstrip('.')}"
