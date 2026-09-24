# `data/` contains repository placeholders. Runtime data uses APP_DATA_DIR.

Nothing the running service writes belongs here. All mutable state - the SQLite
databases, the encrypted OAuth token store, wardrobe images, the document index,
background task outputs, caches and backups - lives under `APP_DATA_DIR`, which
is validated at startup as a separate runtime-data directory.

This directory exists only so tooling that expects a `data/` path finds one, and
so the layout in the README matches what is on disk. It is gitignored apart from
this file.

To see where runtime data actually goes:

```bash
python scripts/check_configuration.py | grep "data directory"
```
