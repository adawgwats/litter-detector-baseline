"""S3-compatible object storage interface for corpus ingest + retrieval.

Works with any S3-API backend: AWS S3, Cloudflare R2, Backblaze B2,
MinIO, etc. Configured via env vars so the same code runs in any
deployment.

Required env vars:
    S3_ACCESS_KEY_ID         — access key
    S3_SECRET_ACCESS_KEY     — secret key
    S3_ENDPOINT_URL          — e.g. https://<account>.r2.cloudflarestorage.com
                               (omit for AWS S3 default)
    S3_BUCKET                — bucket name
    S3_REGION                — optional, defaults to 'auto' (works for R2)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


@dataclass
class Storage:
    """Thin wrapper around an S3-compatible bucket client."""

    bucket: str
    endpoint_url: Optional[str]
    region: str

    @classmethod
    def from_env(cls) -> "Storage":
        try:
            bucket = os.environ["S3_BUCKET"]
        except KeyError as exc:
            raise RuntimeError(
                "S3_BUCKET env var required. See docs/ingestion.md for setup."
            ) from exc
        endpoint = os.environ.get("S3_ENDPOINT_URL") or None
        region = os.environ.get("S3_REGION", "auto")
        return cls(bucket=bucket, endpoint_url=endpoint, region=region)

    def _client(self):
        # signature_version=s3v4 + region=auto is what Cloudflare R2 wants
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region,
            config=Config(signature_version="s3v4"),
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None:
        with path.open("rb") as f:
            self._client().put_object(
                Bucket=self.bucket,
                Key=key,
                Body=f,
                ContentType=content_type,
            )

    def exists(self, key: str) -> bool:
        try:
            self._client().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get_bytes(self, key: str) -> bytes:
        resp = self._client().get_object(Bucket=self.bucket, Key=key)
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
