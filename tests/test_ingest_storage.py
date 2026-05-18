"""Tests for the S3-compatible storage wrapper used by ingest modules.

Uses moto to mock the S3 API — no real network. Validates:
    - content-addressable key generation is deterministic
    - put_bytes / exists / get_bytes round-trip
    - put_file works with a real file on disk
    - from_env reads the right env vars
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from litter_detector_baseline.ingest.storage import Storage, content_addressed_key


@pytest.fixture
def s3_env(monkeypatch):
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("S3_BUCKET", "test-corpus")
    monkeypatch.setenv("S3_REGION", "us-east-1")
    # do not set S3_ENDPOINT_URL — let it default to AWS S3 (moto intercepts)


@pytest.fixture
def storage_with_bucket(s3_env):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-corpus")
        yield Storage.from_env()


def test_content_addressed_key_is_deterministic():
    data = b"some-image-bytes"
    k1 = content_addressed_key("foo/photos", data, "jpg")
    k2 = content_addressed_key("foo/photos", data, "jpg")
    assert k1 == k2
    # Sharded structure: prefix/aa/bb/<full-sha>.ext
    parts = k1.split("/")
    assert parts[0] == "foo"
    assert parts[1] == "photos"
    assert len(parts[2]) == 2
    assert len(parts[3]) == 2
    assert parts[4].endswith(".jpg")


def test_content_addressed_key_handles_prefix_trailing_slash():
    data = b"x"
    assert content_addressed_key("foo/", data, ".jpg") == content_addressed_key("foo", data, "jpg")


def test_storage_put_exists_get_roundtrip(storage_with_bucket):
    storage = storage_with_bucket
    key = "photos/aa/bb/cafebabe.jpg"
    payload = b"\xff\xd8\xff"  # JPEG magic
    assert not storage.exists(key)
    storage.put_bytes(key, payload, content_type="image/jpeg")
    assert storage.exists(key)
    assert storage.get_bytes(key) == payload


def test_storage_put_file(storage_with_bucket, tmp_path):
    storage = storage_with_bucket
    p: Path = tmp_path / "img.jpg"
    p.write_bytes(b"file-bytes")
    storage.put_file("foo/img.jpg", p, content_type="image/jpeg")
    assert storage.get_bytes("foo/img.jpg") == b"file-bytes"


def test_from_env_missing_bucket_raises(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="S3_BUCKET"):
        Storage.from_env()
