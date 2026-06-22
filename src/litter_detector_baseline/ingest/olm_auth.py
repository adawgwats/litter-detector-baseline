"""OpenLitterMap session-cookie authentication.

OLM (a Laravel/Sanctum app) does NOT expose a personal-access-token
issuance endpoint publicly. Authentication is session-cookie based.
For automated ingestion the flow is:

    1. GET  /                       -> seeds XSRF-TOKEN + session cookies
    2. GET  /sanctum/csrf-cookie    -> canonical CSRF endpoint (204 No Content)
    3. POST /api/auth/login         -> with X-XSRF-TOKEN header, body
                                       {"identifier","password","device_name"}
                                       -> 200 + sets authenticated session cookie
    4. All subsequent calls         -> include the cookie + X-XSRF-TOKEN
                                       header until session expires
                                       (default Laravel session lifetime
                                       is ~2 hours of inactivity)

The login endpoint accepts EITHER email OR username under the
``identifier`` field (verified 2026-05-20).

This module loads credentials from AWS Secrets Manager (default) or
environment variables (dev fallback), executes the CSRF + login dance,
and returns a configured ``requests.Session`` that downstream code can
use exactly like a normal ``requests`` session. Auto-re-login on 401/419
is built in via a thin retrying adapter.

Secret format expected in Secrets Manager (key: `dregsbane/olm/ingest-bot`):

    {
        "identifier":  "<email-or-username>",
        "password":    "<password>",
        "username":    "<olm-display-username>",
        "olm_user_id": <int>,
        "device_name": "<descriptive label>"
    }

Environment variable fallback (only ``identifier`` + ``password`` required):

    OLM_IDENTIFIER, OLM_PASSWORD, OLM_DEVICE_NAME

See docs/olm-auth-howto.md in this repo for the operational guide.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import requests

log = logging.getLogger(__name__)

OLM_BASE = "https://openlittermap.com"
SECRET_ID_DEFAULT = "dregsbane/olm/ingest-bot"

# Laravel default session lifetime is 120 minutes idle. Re-login proactively
# after this to avoid an in-flight 419/401.
SESSION_REFRESH_SEC = 90 * 60  # 90 minutes — conservative

# Persist session cookies across script runs. OLM throttles /api/auth/login
# aggressively (we got a 429 after ~3 logins in one hour). Caching cookies
# keeps us at ≤ 1 login per ~90 minutes, well inside any reasonable throttle.
DEFAULT_COOKIE_CACHE = Path.home() / ".cache" / "litter-detector-baseline" / "olm-session.pickle"


@dataclass(frozen=True)
class OlmCredentials:
    """Username/email + password pair used to mint a session cookie."""

    identifier: str
    password: str
    device_name: str = "dregsbane-trash-trail-ingest"

    def __repr__(self) -> str:  # never print the password
        return (
            f"OlmCredentials(identifier={self.identifier!r}, "
            f"password='[REDACTED]', device_name={self.device_name!r})"
        )


def load_credentials_from_secrets_manager(
    secret_id: str = SECRET_ID_DEFAULT,
    region_name: str = "us-east-1",
) -> OlmCredentials:
    """Read credentials from AWS Secrets Manager.

    Lazy-imports boto3 so callers who only use env-var fallback don't
    pay the import cost.
    """
    import boto3  # noqa: PLC0415 — lazy by design

    client = boto3.client("secretsmanager", region_name=region_name)
    resp = client.get_secret_value(SecretId=secret_id)
    secret = json.loads(resp["SecretString"])
    return OlmCredentials(
        identifier=secret["identifier"],
        password=secret["password"],
        device_name=secret.get("device_name", OlmCredentials.__dataclass_fields__["device_name"].default),
    )


def load_credentials_from_env() -> OlmCredentials:
    """Read credentials from OLM_IDENTIFIER / OLM_PASSWORD / OLM_DEVICE_NAME.

    Intended for local development where AWS auth isn't set up. Production
    callers should use load_credentials_from_secrets_manager().
    """
    try:
        identifier = os.environ["OLM_IDENTIFIER"]
        password = os.environ["OLM_PASSWORD"]
    except KeyError as exc:
        raise RuntimeError(
            "OLM credentials not in env. Set OLM_IDENTIFIER and OLM_PASSWORD, "
            "or use load_credentials_from_secrets_manager()."
        ) from exc
    return OlmCredentials(
        identifier=identifier,
        password=password,
        device_name=os.environ.get(
            "OLM_DEVICE_NAME",
            OlmCredentials.__dataclass_fields__["device_name"].default,
        ),
    )


def load_credentials(*, source: str = "secrets") -> OlmCredentials:
    """Load credentials from the configured source.

    ``source`` is one of ``"secrets"`` (Secrets Manager, default) or
    ``"env"`` (environment variables).
    """
    if source == "secrets":
        return load_credentials_from_secrets_manager()
    if source == "env":
        return load_credentials_from_env()
    raise ValueError(f"unknown credentials source: {source!r}")


class OlmSession:
    """A ``requests.Session`` wrapper that logs into OLM and stays logged in.

    Usage::

        creds = load_credentials()
        with OlmSession(creds) as session:
            r = session.get(f"{OLM_BASE}/api/clusters", params={"zoom": 16})
            r.raise_for_status()

    The session auto-re-logs-in if it goes stale (419 page-expired or
    401 unauthenticated). Use as a normal ``requests.Session`` for any
    OLM call — public endpoints work too, the auth is just additive.
    """

    def __init__(
        self,
        credentials: OlmCredentials,
        *,
        user_agent: str = "litter-detector-baseline/0.1 (+olm-ingest-bot)",
        base_url: str = OLM_BASE,
        cookie_cache_path: Optional[Path] = DEFAULT_COOKIE_CACHE,
    ) -> None:
        self._creds = credentials
        self._base = base_url.rstrip("/")
        self._session: Optional[requests.Session] = None
        # RLock so request() can hold the lock while transitively calling
        # _login() (same thread re-acquires; non-reentrant Lock would
        # deadlock here — observed 2026-05-23 after the 90-min refresh
        # boundary stalled all workers).
        self._login_lock = threading.RLock()
        self._user_agent = user_agent
        self._login_ts: float = 0.0
        # Set cookie_cache_path=None to disable caching (e.g. in tests).
        self._cookie_cache_path = cookie_cache_path

    # ─── context manager ────────────────────────────────────────────────

    def __enter__(self) -> "OlmSession":
        if not self._try_load_cached():
            self._login()
        return self

    # ─── cookie cache (avoid hammering /api/auth/login) ─────────────────

    def _try_load_cached(self) -> bool:
        """Try to restore a session from the on-disk cookie cache.

        Returns True on success (session restored AND fresh enough). On any
        failure — file missing, corrupt, expired — returns False and the
        caller falls back to _login().
        """
        if self._cookie_cache_path is None or not self._cookie_cache_path.exists():
            return False
        try:
            with self._cookie_cache_path.open("rb") as fh:
                blob = pickle.load(fh)
            login_ts = float(blob["login_ts"])
            identifier = blob.get("identifier")
            cookies = blob["cookies"]
        except Exception as exc:
            log.warning("OLM cookie cache unreadable (%s); will re-login", exc)
            return False
        # Treat as expired if the cached session belongs to a different
        # identifier than the current creds, or if it's past our refresh window.
        if identifier and identifier != self._creds.identifier:
            log.info("OLM cookie cache identifier mismatch; re-login")
            return False
        if (time.time() - login_ts) > SESSION_REFRESH_SEC:
            log.info("OLM cookie cache is older than %ds; re-login", SESSION_REFRESH_SEC)
            return False
        s = requests.Session()
        s.headers["User-Agent"] = self._user_agent
        s.cookies.update(cookies)
        self._session = s
        self._login_ts = login_ts
        log.info("OLM session restored from cache (age=%.0fs)", time.time() - login_ts)
        return True

    def _persist_cookies(self) -> None:
        if self._cookie_cache_path is None or self._session is None:
            return
        try:
            self._cookie_cache_path.parent.mkdir(parents=True, exist_ok=True)
            blob = {
                "login_ts": self._login_ts,
                "identifier": self._creds.identifier,
                "cookies": dict(self._session.cookies),
            }
            # Write atomically: write to a temp sibling, then rename.
            tmp = self._cookie_cache_path.with_suffix(".pickle.tmp")
            with tmp.open("wb") as fh:
                pickle.dump(blob, fh)
            tmp.replace(self._cookie_cache_path)
        except Exception as exc:
            log.warning("OLM cookie cache write failed: %s", exc)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    # ─── login ──────────────────────────────────────────────────────────

    def _login(self) -> None:
        """Do the CSRF + login dance, populate ``self._session``."""
        with self._login_lock:
            s = requests.Session()
            s.headers["User-Agent"] = self._user_agent
            # 1. Seed XSRF + session cookies via homepage GET
            r = s.get(self._base + "/", timeout=15)
            r.raise_for_status()
            # 2. Hit the canonical Sanctum CSRF endpoint (204 No Content)
            r = s.get(self._base + "/sanctum/csrf-cookie", timeout=15)
            r.raise_for_status()
            xsrf = self._extract_xsrf(s)
            if not xsrf:
                raise RuntimeError("OLM did not return XSRF-TOKEN cookie after csrf-cookie GET")
            # 3. POST login with XSRF header
            r = s.post(
                self._base + "/api/auth/login",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-XSRF-TOKEN": xsrf,
                    "Referer": self._base + "/",
                    "Origin": self._base,
                },
                json={
                    "identifier": self._creds.identifier,
                    "password": self._creds.password,
                    "device_name": self._creds.device_name,
                },
                timeout=15,
            )
            if r.status_code == 422:
                # Wrong field name OR wrong credentials — surface OLM's error JSON
                try:
                    err = r.json()
                except ValueError:
                    err = {"raw": r.text[:300]}
                raise RuntimeError(f"OLM login rejected (422): {err}")
            r.raise_for_status()
            self._session = s
            self._login_ts = time.time()
            log.info("OLM login OK as %s", self._creds.identifier)
            self._persist_cookies()

    def _extract_xsrf(self, s: requests.Session) -> Optional[str]:
        # Under concurrent use, OLM occasionally Set-Cookies the same name
        # twice (different path/domain), and `s.cookies.get('XSRF-TOKEN')`
        # raises CookieConflictError. Iterate explicitly and pick the most
        # recent matching cookie so concurrent workers don't crash.
        matches = [c for c in s.cookies if c.name == "XSRF-TOKEN"]
        if not matches:
            return None
        # The cookie value is URL-encoded by Laravel; the header expects the
        # decoded form.
        return unquote(matches[-1].value)

    def _maybe_refresh(self) -> None:
        """Re-login if the session is stale or doesn't exist yet.

        Tries the on-disk cookie cache first so we don't hit /api/auth/login
        on every script run (which is throttled aggressively — 429 after a
        few attempts per hour).
        """
        if self._session is None:
            if self._try_load_cached():
                return
            self._login()
            return
        if (time.time() - self._login_ts) > SESSION_REFRESH_SEC:
            self._login()

    # ─── request proxy with auto-relogin on 401/419 ─────────────────────

    def request(self, method: str, url: str, **kw) -> requests.Response:
        # The underlying `requests.Session` cookies jar is NOT thread-safe.
        # Concurrent calls produce CookieConflictError when both threads see
        # Set-Cookie responses simultaneously. Serialize the entire HTTP
        # exchange here. The signed-url call is fast (~200-500ms) so this
        # isn't a real throughput hit — and the caller's rate limiter
        # already pins us to 1 OLM API call per second anyway. The lock is
        # an RLock so this call can transitively invoke _login() (which
        # also acquires the lock) without deadlocking.
        with self._login_lock:
            self._maybe_refresh()
            assert self._session is not None  # narrows for type-checkers
            xsrf = self._extract_xsrf(self._session)
            headers = kw.pop("headers", {}) or {}
            if xsrf and "X-XSRF-TOKEN" not in {h.title() for h in headers}:
                headers["X-XSRF-TOKEN"] = xsrf
            headers.setdefault("Accept", "application/json")
            headers.setdefault("Referer", self._base + "/")
            r = self._session.request(method, url, headers=headers, **kw)
            if r.status_code in (401, 419):
                log.warning("OLM returned %s — re-logging in once", r.status_code)
                self._login()
                xsrf = self._extract_xsrf(self._session)
                if xsrf:
                    headers["X-XSRF-TOKEN"] = xsrf
                r = self._session.request(method, url, headers=headers, **kw)
            return r

    def get(self, url: str, **kw) -> requests.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> requests.Response:
        return self.request("POST", url, **kw)


# ─── convenience: quick token-style "smoke test" entry point ─────────────


def _smoke_test() -> None:
    """Print "OLM auth OK as <identifier>" if credentials + login work.

    Run with: ``python -m litter_detector_baseline.ingest.olm_auth``
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    creds = load_credentials()
    with OlmSession(creds) as session:
        r = session.get(f"{OLM_BASE}/api/clusters", params={"zoom": 16})
        r.raise_for_status()
        # /api/clusters works unauth too, so the body shape doesn't prove auth.
        # The fact that login succeeded above is the proof.
        n_features = len(r.json().get("features", []))
        print(f"OLM auth OK as {creds.identifier} (clusters endpoint returned {n_features} features)")


if __name__ == "__main__":
    _smoke_test()
