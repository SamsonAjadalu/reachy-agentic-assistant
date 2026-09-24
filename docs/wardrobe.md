# Wardrobe

Local wardrobe cataloguing, outfit recommendations, and laundry tracking. Images and metadata live under `APP_DATA_DIR`.

## Storage

Default image root: `${APP_DATA_DIR}/wardrobe`

```
wardrobe/
  items/        # full-size photos
  thumbnails/   # generated derivatives
  outfits/      # composite outfit images (if used)
```

Override with `WARDROBE_IMAGE_ROOT` if you want to use a different local folder.

## Adding items

**HTTP API** — see `/api/v1/wardrobe` in [API overview](api.md) and `/openapi.json`.

**Batch import** from CSV:

```bash
python scripts/import_wardrobe.py items.csv --images ./photos --dry-run
python scripts/import_wardrobe.py items.csv --images ./photos
```

Required CSV columns: `name`, `category`, `primary_color`.

## Recommendations

`GET /api/v1/wardrobe/recommend` scores saved outfits against the weather forecast. Returns reasoning and exclusions. Works in mock mode with mock weather.

Briefing section `outfit` pulls from the same data (`proactive/sections.py`).

## Demo data

```bash
python scripts/seed_demo_data.py
```

Adds sample items tagged `demo-seed` in the `notes` field.

## Backup

Wardrobe images are files on disk under `APP_DATA_DIR`. Include the whole directory in filesystem backups; Filesystem backups include image binaries; SQLite backups cover database records.

## Related

- [Operations](operations.md) — `APP_DATA_DIR` layout
- [API overview](api.md)
