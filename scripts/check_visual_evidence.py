#!/usr/bin/env python3
"""Check visual evidence files against assistant.db.

    python scripts/check_visual_evidence.py
    python scripts/check_visual_evidence.py --repair

Rows without files are marked ``missing``. Files without rows are quarantined
Uses the configured workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from database.session import create_engine, set_engine  # noqa: E402
from vision.retention import refresh_ref_count_hints, sweep_orphans  # noqa: E402


async def _run(*, repair: bool) -> int:
    settings = Settings()
    engine = create_engine(settings)
    set_engine(engine)
    from database.session import get_sessionmaker

    factory = get_sessionmaker()
    async with factory() as session:
        report = await sweep_orphans(session, settings)
        await refresh_ref_count_hints(session)
        if repair:
            await session.commit()
        else:
            await session.rollback()
    await engine.dispose()
    print(f"Evidence root:        {settings.visual_evidence_path}")
    print(f"Missing rows marked:  {report.missing_marked}")
    print(f"Orphans quarantined:  {report.orphans_quarantined}")
    print("Repair" if repair else "Dry run (pass --repair to write)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Apply missing-row marks and quarantine orphans.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(repair=args.repair))


if __name__ == "__main__":
    raise SystemExit(main())
