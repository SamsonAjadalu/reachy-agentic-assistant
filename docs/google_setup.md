# Google setup

Google Calendar, Gmail, Contacts and Drive share one OAuth grant stored in an encrypted file under APP_DATA_DIR.

## Prerequisites

- A Google Cloud project with OAuth consent configured
- The Fernet encryption key configured (`PA_SECRET_KEY_FILE` or systemd `LoadCredential`) — see [Operations](operations.md)
- A browser on the workstation for the one-time consent flow

## Cloud Console

1. [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → **OAuth consent screen** — configure for internal or external test users as appropriate.
2. **Credentials** → Create **OAuth client ID** → Application type **Desktop app** (installed application flow).
3. Note **Client ID** and **Client secret**.

Enable APIs you need: Gmail, Calendar, People (Contacts), Drive.

## Configuration

In `.env`:

```env
GOOGLE_ENABLED=true
GOOGLE_CLIENT_ID=<client id>
GOOGLE_CLIENT_SECRET=<client secret>
GOOGLE_REDIRECT_URI=http://localhost:8765/oauth2callback
```

Token store default: `${APP_DATA_DIR}/secrets/google_tokens.enc`

The redirect URI must match the Console entry. Setup uses a loopback listener on port 8765 (`scripts/setup_google_oauth.py`).

## Authorisation (one-time)

```bash
python scripts/check_configuration.py
python scripts/setup_google_oauth.py
```

The script opens a browser, catches the redirect, and writes the refresh token to the encrypted store. Nothing is printed to the terminal.

Verify via API (with bearer token):

```bash
curl -s -H "Authorization: Bearer $PA_API_TOKEN" \
  http://127.0.0.1:8080/api/v1/integrations/google
```

## Scopes

Read scopes are always requested. Write scopes (`gmail.compose`, `gmail.send`, `gmail.modify`, `calendar.events`) are included at setup time for headless operation.

- Draft creation uses `gmail.compose`; sending follows the approval flow.
- Outbound sends use `gmail.send` against an existing draft and remain **approval-gated** (exact draft revision hash).
- Archive and label changes use `gmail.modify` synchronously (reversible); trash/spam are refused on this path.
- Calendar create/update/delete are **approval-gated**; `POST /calendar/events/propose` is a dry-run only.

If an existing grant predates `gmail.modify`, re-run `setup_google_oauth.py`.

See `integrations/google/oauth.py` for the exact scope list.

## Mock mode

`MOCK_MODE=true` forces mock Google providers regardless of `GOOGLE_ENABLED`. Refused when `APP_ENV=production`.

## Revocation

Re-run `setup_google_oauth.py` to replace tokens, or use the Google Account security page to revoke access and clear the local store manually.

## Related

- [Operations](operations.md) — key storage outside `APP_DATA_DIR`
- [API overview](api.md) — Gmail, Calendar, Contacts, Drive routes
