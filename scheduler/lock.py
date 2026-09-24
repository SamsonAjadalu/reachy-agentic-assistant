"""Single-instance lock for the scheduler.

Running two schedulers against one persistent job store fires every reminder
twice. A database lease alone is not enough, because a process killed with
SIGKILL leaves a lease that looks live until it expires. An advisory ``flock``
is released by the kernel the moment the holding process dies, so combining the
two gives both immediate release and cross-host visibility.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType

from app.logging_config import get_logger

logger = get_logger(__name__)


class SchedulerLock:
    """Non-blocking exclusive lock on a file under ``APP_DATA_DIR``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        """Try to take the lock. Returns ``False`` if another process holds it."""
        if self._fd is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            logger.info(
                "Scheduler lock is held by another process", extra={"path": str(self._path)}
            )
            return False

        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def holder_pid(self) -> int | None:
        """Best-effort read of the pid recorded by the current holder."""
        try:
            content = self._path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return int(content) if content.isdigit() else None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
