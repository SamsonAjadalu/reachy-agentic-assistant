"""Child process used by the scheduler persistence tests.

A genuine process restart is the only way to prove the job store survives one, so
these tests spawn this module rather than simulating a restart in-process.

    python scheduler_child.py add <data_dir> <job_id> <run_at_iso>
    python scheduler_child.py list <data_dir>
    python scheduler_child.py hold <data_dir> <seconds>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.config import AppEnv, Settings  # noqa: E402
from scheduler import jobs as job_targets  # noqa: E402
from scheduler.service import SchedulerService  # noqa: E402
from shared.timeutils import isoformat_utc, parse_iso8601  # noqa: E402


def build_settings(data_dir: str) -> Settings:
    return Settings(
        app_env=AppEnv.TEST,
        app_data_dir=Path(data_dir),
        pa_api_token="x" * 40,
        database_url=f"sqlite+aiosqlite:///{Path(data_dir) / 'test.db'}",
        scheduler_enabled=True,
        worker_enabled=False,
        mock_mode=True,
    )


async def run(command: str, data_dir: str, args: list[str]) -> int:
    settings = build_settings(data_dir)
    settings.ensure_directories()
    service = SchedulerService(settings)

    started = await service.start()
    if not started:
        print(json.dumps({"started": False, "reason": "lock_held"}))
        return 3

    try:
        if command == "add":
            job_id, run_at = args[0], parse_iso8601(args[1])
            service.add_date_job(
                job_targets.REMINDER_JOB,
                run_at,
                job_id=job_id,
                kwargs={"reminder_id": job_id.removeprefix("reminder:")},
            )
            print(json.dumps({"started": True, "added": job_id}))
        elif command == "list":
            print(
                json.dumps(
                    {
                        "started": True,
                        "jobs": [
                            {
                                "id": job["id"],
                                "next_run_at": isoformat_utc(job["next_run_at"])
                                if job["next_run_at"]
                                else None,
                            }
                            for job in service.list_jobs()
                        ],
                    }
                )
            )
        elif command == "hold":
            print(json.dumps({"started": True, "holding": True}), flush=True)
            await asyncio.sleep(float(args[0]))
        else:
            print(json.dumps({"error": f"unknown command {command}"}))
            return 2
    finally:
        await service.stop()
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: scheduler_child.py <command> <data_dir> [args...]"}))
        return 2
    return asyncio.run(run(sys.argv[1], sys.argv[2], sys.argv[3:]))


if __name__ == "__main__":
    raise SystemExit(main())
