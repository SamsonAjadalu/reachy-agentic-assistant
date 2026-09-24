# Telegram setup

Telegram is the approval and notification channel. Destructive or externally visible actions create a pending approval; the owner taps **Approve** or **Reject** on an inline keyboard.

## Architecture

The bot uses **long polling**, not webhooks — the workstation typically has no inbound route from the internet.

The listener runs **inside the scheduler-owning API process** only (`app/lifespan.py`). If another instance holds `${APP_DATA_DIR}/scheduler.lock`, Telegram is not started there. There is no separate `reachy-personal-assistant-telegram.service`.

## Bot creation

1. Message [@BotFather](https://t.me/BotFather) on Telegram.
2. `/newbot` — choose a name and username.
3. Copy the **bot token**.

## Configuration

In `.env`:

```env
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=<token from BotFather>
TELEGRAM_ALLOWED_CHAT_IDS=<your numeric chat id>
```

`TELEGRAM_ALLOWED_CHAT_IDS` is a comma-separated allowlist. Updates from any other chat are dropped.

### Finding your chat id

1. Start a chat with your bot (send any message).
2. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser (or curl locally).
3. Read `message.chat.id` from the JSON.

Only that id (and any others you explicitly allow) may approve actions or receive notifications.

## Production checks

With `APP_ENV=production`, the service refuses to start if:

- `TELEGRAM_BOT_TOKEN` is missing or a placeholder
- `TELEGRAM_ALLOWED_CHAT_IDS` is empty

Run `python scripts/check_configuration.py` before enabling the unit.

## Restart

After changing Telegram settings:

```bash
systemctl --user restart reachy-personal-assistant-api.service
```

Confirm the scheduler owns the lock (`/ready` → `checks.scheduler.owned_by_this_process: true`) and that `mock_mode` is false when using the real bot.

## Security notes

- Message text is matched against a fixed command table; when `REACHY_TEXT_TURN_ENABLED=true`, other plain text is forwarded to the Reachy Pi `/api/v1/text-turn` endpoint.
- Approval callbacks carry opaque tokens; the listener resolves them through the approvals state machine.
- See `integrations/telegram/listener.py` for allowlist and callback handling.

## Related

- [Operations](operations.md) — process layout and lingering
- [API overview](api.md) — `/api/v1/approvals`
