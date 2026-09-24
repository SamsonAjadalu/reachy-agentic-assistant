# Reachy Pi tool adapters

The verified official-conversation-app bridge is `reachy_app_tools.py`. It is a
thin async wrapper around `reachy_client` and contains no backend business logic
or provider credentials.

## What lives here

- `reachy_app_tools.py` — 29 official `Tool` subclasses, explicit JSON schemas,
  bounded async HTTP calls, sanitized errors, and write/idempotency safeguards.
- `proposed/adapters.py` — earlier plain Python functions that validate arguments,
  call the main-PC API through `PersonalAssistantClient`, and return short
  structured dicts. They are retained for non-conversation-app experiments.
- `templates/tool_stub.py.template` — historical adapter template.

## What this is not

- Not part of the upstream Reachy repository.
- Not a backend service, credential store, scheduler, or database.
- Not live-tested against the main PC or external providers without credentials.

## Usage sketch

```python
from reachy_client import PersonalAssistantClient
from reachy_tools.proposed.adapters import assistant_ping

client = PersonalAssistantClient(base_url, token)
result = assistant_ping(client)
print(result["spoken"])
```

The official app loads `reachy_app_tools.py` as a profile-local tool module. Set
`PA_API_URL` and `PA_API_TOKEN`; optionally set `PA_API_CONNECT_TIMEOUT` and
`PA_API_TIMEOUT`. See `docs/reachy_pi_handoff.md` for Pi deployment details.
