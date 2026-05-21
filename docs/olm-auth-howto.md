# OpenLitterMap auth — operational guide

**TL;DR**: OLM does not issue API tokens. Auth is session-cookie based.
Credentials live in AWS Secrets Manager. The Python ingest module
`litter_detector_baseline.ingest.olm_auth` handles the login dance.

---

## What OLM gives us

OLM is a Laravel/Sanctum app. Authentication uses **session cookies +
CSRF tokens**, not stateless bearer tokens. There is **no personal-access-
token issuance endpoint exposed publicly** (verified 2026-05-20 by
probing `/api/tokens`, `/api/user/tokens`, `/api/auth/tokens`,
`/user/api-tokens`, all return the SPA shell instead of JSON).

Every ingest run must:

1. Hit the homepage → server seeds `XSRF-TOKEN` + `openlittermap_session`
   cookies.
2. Hit `/sanctum/csrf-cookie` (204 No Content) to refresh the CSRF token.
3. POST `/api/auth/login` with:
   - body: `{"identifier","password","device_name"}` (`identifier` is
     the field name — accepts email OR username; `email` is rejected)
   - header: `X-XSRF-TOKEN: <decoded cookie value>`
   - header: `Referer: https://openlittermap.com/`
4. Carry the cookies (+ `X-XSRF-TOKEN` header for non-GET calls) on
   every subsequent request until the session expires.

Session lifetime is Laravel's default — ~2 hours of inactivity.

## Credentials storage

Credentials live in AWS Secrets Manager at secret ID
`dregsbane/olm/ingest-bot` (us-east-1). Secret payload:

```json
{
    "identifier":  "adawgwats@gmail.com",
    "password":    "...",
    "username":    "morally-exhausted-civic-disruptor-1135",
    "olm_user_id": 9773,
    "device_name": "dregsbane-trash-trail-ingest"
}
```

The account was created on 2026-05-19 for ingestion purposes. It is not
shared with any personal use. Compromise consequences are bounded: the
worst an attacker can do is upload bad photos under our bot name; they
cannot escalate to anything else.

To rotate the password:

1. Log into OLM as the bot account via the web UI → change password.
2. Update the secret:
   ```bash
   aws secretsmanager update-secret \
     --secret-id dregsbane/olm/ingest-bot \
     --secret-string '{"identifier":"adawgwats@gmail.com","password":"<new>","username":"morally-exhausted-civic-disruptor-1135","olm_user_id":9773,"device_name":"dregsbane-trash-trail-ingest"}'
   ```
3. Secrets Manager versions automatically; the previous version stays
   recoverable for 30 days.

## Using the auth module

```python
from litter_detector_baseline.ingest.olm_auth import (
    OlmSession,
    load_credentials,
    OLM_BASE,
)

creds = load_credentials()          # Secrets Manager (default)
# creds = load_credentials(source="env")  # for local dev (OLM_IDENTIFIER + OLM_PASSWORD)

with OlmSession(creds) as session:
    r = session.get(f"{OLM_BASE}/api/clusters", params={"zoom": 16})
    r.raise_for_status()
    data = r.json()
```

`OlmSession.request()` auto-re-logins once on 401/419. For long-running
jobs the session is proactively refreshed every 90 minutes (well inside
the 120-minute Laravel default).

## Smoke test

```bash
python -m litter_detector_baseline.ingest.olm_auth
```

Should print `OLM auth OK as <identifier> (clusters endpoint returned N features)`.
If you get an `AccessDenied` from boto3, your AWS credentials are missing
or the IAM policy doesn't grant `secretsmanager:GetSecretValue` on the
secret's ARN. Required policy:

```json
{
  "Effect": "Allow",
  "Action": "secretsmanager:GetSecretValue",
  "Resource": "arn:aws:secretsmanager:us-east-1:910757112705:secret:dregsbane/olm/ingest-bot-*"
}
```

For local dev you can sidestep AWS by exporting `OLM_IDENTIFIER` and
`OLM_PASSWORD` and calling `load_credentials(source="env")`.

## What auth UNLOCKS

The unauth endpoints (`/api/clusters`, `/api/points`, `/api/tags/all`,
`/global/stats-data`) already work without auth — those don't need this
module. What auth gets you:

- `/user/profile/photos` — bot account's own uploads (empty for us
  since we don't upload from this account, but available)
- Per-photo endpoints that require user context
- Bulk-export endpoints (if/when OLM team grants them)
- Any future endpoint OLM gates behind login

The **session cookie does not unlock arbitrary bulk download of all
525K OLM photos**. OLM does not expose that endpoint to any unauth or
account-only user. Bulk access requires partnership outreach (see
`dregsbane-ops#18` issue) — that's a separate workstream from this
auth module.

## What auth does NOT unlock — be honest

- We still rate-limit at ~1 req/sec on metadata, 3 concurrent on images.
  This is policy, not a technical floor. Going faster than OLM expects
  is how the bot account gets banned.
- We do not get write-access to other contributors' photos.
- The session cookie ages out after ~2hrs idle. Long-running jobs need
  the auto-refresh built into `OlmSession`.

## Failure modes + how to triage

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: OLM login rejected (422)` | Wrong identifier or password in secret | Re-test creds via PS/curl, then update secret |
| `RuntimeError: OLM did not return XSRF-TOKEN cookie` | OLM changed their CSRF flow OR a CDN is stripping cookies | Compare current cookie set to `XSRF-TOKEN, openlittermap_session` |
| 419 on every call despite re-login | Stale XSRF — we're reading the cookie wrong (URL-decoded vs encoded) | Module already handles this; if regressing, check `_extract_xsrf` |
| 401 on every call | Account was banned OR password rotated outside Secrets Manager | Log into OLM web UI, check account status, update secret if needed |
| Connection-level errors (ConnectionError, Timeout) | OLM is down OR our IP is blocked | Wait 5min, retry. If sustained, contact Seán Lynch directly |

## Future work

- **Federated/scoped tokens via partnership outreach** — Seán Lynch
  may grant a scoped API token if asked. That would simplify this
  module to a static header, no session dance. See `dregsbane-ops#18`.
- **Per-job credential isolation** — currently one bot account
  authenticates for all training data ingestion. If we add operator-
  mode model training (CrustBot fleet), they should get their own
  credentials so a per-account ban doesn't take down both pipelines.
- **Cost cap on Secrets Manager calls** — at $0.05 per 10K
  `GetSecretValue` calls, this is trivial today. If we ever do
  per-request credential fetching at high QPS, cache the credentials
  in-process.
