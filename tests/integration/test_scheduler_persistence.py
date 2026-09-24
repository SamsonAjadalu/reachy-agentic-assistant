"""Scheduler durability.

The claim being tested is that a reminder scheduled before a crash still fires
after the machine comes back. Simulating a restart in-process would not prove
that, so these tests spawn and kill real subprocesses.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from app.config import Settings
from scheduler.lock import SchedulerLock
from scheduler.service import SchedulerService
from shared.timeutils import isoformat_utc, utcnow

CHILD = Path(__file__).parent / "scheduler_child.py"
REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.slow


def run_child(command: str, data_dir: Path, *args: str, timeout: float = 60) -> dict:
    """Run the helper in a genuinely separate interpreter."""
    env = {
        **os.environ,
        "PYTHONPATH": "",
        "PA_ENV_FILE": str(REPO_ROOT / "tests" / ".env.absent"),
        "APP_ENV": "test",
    }
    result = subprocess.run(
        [sys.executable, str(CHILD), command, str(data_dir), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    if not lines:
        pytest.fail(f"child produced no JSON.\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return json.loads(lines[-1])


class TestJobStorePersistence:
    def test_a_job_added_before_a_restart_is_still_scheduled_after_it(self, data_dir: Path) -> None:
        run_at = utcnow() + timedelta(hours=6)

        added = run_child("add", data_dir, "reminder:persist-1", isoformat_utc(run_at))
        assert added["added"] == "reminder:persist-1"

        # The first process has fully exited by now; this is a cold start.
        listed = run_child("list", data_dir)
        job_ids = {job["id"] for job in listed["jobs"]}
        assert "reminder:persist-1" in job_ids

    def test_the_restored_job_keeps_its_original_run_time(self, data_dir: Path) -> None:
        run_at = (utcnow() + timedelta(hours=6)).replace(microsecond=0)
        run_child("add", data_dir, "reminder:persist-2", isoformat_utc(run_at))

        listed = run_child("list", data_dir)
        restored = next(job for job in listed["jobs"] if job["id"] == "reminder:persist-2")
        assert restored["next_run_at"].startswith(isoformat_utc(run_at)[:16])

    def test_the_job_store_file_lives_under_the_data_directory(self, data_dir: Path) -> None:
        run_child(
            "add", data_dir, "reminder:persist-3", isoformat_utc(utcnow() + timedelta(hours=1))
        )
        assert (data_dir / "scheduler.db").exists()
        assert REPO_ROOT not in (data_dir / "scheduler.db").resolve().parents

    def test_multiple_jobs_all_survive(self, data_dir: Path) -> None:
        for index in range(3):
            run_child(
                "add",
                data_dir,
                f"reminder:multi-{index}",
                isoformat_utc(utcnow() + timedelta(hours=index + 1)),
            )
        job_ids = {job["id"] for job in run_child("list", data_dir)["jobs"]}
        assert {f"reminder:multi-{i}" for i in range(3)} <= job_ids

    def test_a_job_survives_a_sigkill_rather_than_a_clean_shutdown(self, data_dir: Path) -> None:
        """SIGKILL gives no chance to flush, so this proves the store is written eagerly."""
        run_child("add", data_dir, "reminder:killed", isoformat_utc(utcnow() + timedelta(hours=4)))

        env = {**os.environ, "PYTHONPATH": "", "APP_ENV": "test"}
        holder = subprocess.Popen(
            [sys.executable, str(CHILD), "hold", str(data_dir), "30"],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            assert holder.stdout is not None
            json.loads(holder.stdout.readline())
            holder.send_signal(signal.SIGKILL)
            holder.wait(timeout=10)
        finally:
            if holder.poll() is None:  # pragma: no cover - defensive
                holder.kill()

        job_ids = {job["id"] for job in run_child("list", data_dir)["jobs"]}
        assert "reminder:killed" in job_ids


class TestSingleSchedulerLock:
    def test_a_second_process_cannot_start_a_scheduler(self, data_dir: Path) -> None:
        env = {**os.environ, "PYTHONPATH": "", "APP_ENV": "test"}
        holder = subprocess.Popen(
            [sys.executable, str(CHILD), "hold", str(data_dir), "20"],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            assert holder.stdout is not None
            first = json.loads(holder.stdout.readline())
            assert first["started"] is True

            second = run_child("list", data_dir, timeout=30)
            assert second["started"] is False
            assert second["reason"] == "lock_held"
        finally:
            holder.terminate()
            holder.wait(timeout=10)

    def test_the_lock_is_released_when_the_holder_is_killed(self, data_dir: Path) -> None:
        """flock is released by the kernel, so a SIGKILLed holder does not wedge the system."""
        env = {**os.environ, "PYTHONPATH": "", "APP_ENV": "test"}
        holder = subprocess.Popen(
            [sys.executable, str(CHILD), "hold", str(data_dir), "30"],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert holder.stdout is not None
        json.loads(holder.stdout.readline())
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=10)
        time.sleep(0.2)

        assert run_child("list", data_dir)["started"] is True

    def test_lock_acquisition_is_reentrant_within_one_process(self, tmp_path: Path) -> None:
        lock = SchedulerLock(tmp_path / "scheduler.lock")
        assert lock.acquire() is True
        assert lock.acquire() is True
        lock.release()

    def test_the_lock_records_the_holder_pid(self, tmp_path: Path) -> None:
        lock = SchedulerLock(tmp_path / "scheduler.lock")
        lock.acquire()
        try:
            assert lock.holder_pid() == os.getpid()
        finally:
            lock.release()


class TestSchedulerServiceLifecycle:
    async def test_start_and_stop_installs_the_periodic_jobs(self, settings: Settings) -> None:
        settings.scheduler_enabled = True
        service = SchedulerService(settings)
        assert await service.start() is True
        try:
            job_ids = {job["id"] for job in service.list_jobs()}
            assert "system:reconcile" in job_ids
            assert "system:maintenance" in job_ids
        finally:
            await service.stop()

    async def test_removing_a_missing_job_is_not_an_error(self, settings: Settings) -> None:
        service = SchedulerService(settings)
        await service.start()
        try:
            assert service.remove_job("does-not-exist") is False
        finally:
            await service.stop()

    async def test_stopping_releases_the_lock_for_the_next_start(self, settings: Settings) -> None:
        first = SchedulerService(settings)
        assert await first.start() is True
        await first.stop()

        second = SchedulerService(settings)
        assert await second.start() is True
        await second.stop()
