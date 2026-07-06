"""
Portable file locking for Petals.

On POSIX systems this uses ``fcntl.flock`` (shared or exclusive).
On Windows it uses ``msvcrt.locking`` (always exclusive — Windows does not
expose shared/reader locks through the CRT interface, so we conservatively
lock exclusively for both modes; this is correct but slightly reduces
read concurrency on Windows).
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

if sys.platform != "win32":
    import fcntl

    _LOCK_SH = fcntl.LOCK_SH
    _LOCK_EX = fcntl.LOCK_EX

    def _lock(fd: int, mode: int) -> None:
        fcntl.flock(fd, mode)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

else:
    import msvcrt

    # msvcrt.locking modes: LK_NBLCK (non-blocking), LK_LOCK (blocking retry)
    # We always use LK_LOCK (blocking), locking the entire file (up to 2**31-1 bytes).
    _LOCK_SH = 0  # sentinel — unused on Windows
    _LOCK_EX = 1  # sentinel — unused on Windows
    _MSVCRT_LOCK_SIZE = 2**31 - 1

    def _lock(fd: int, mode: int) -> None:  # noqa: ARG001  (mode ignored on win32)
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, _MSVCRT_LOCK_SIZE)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, _MSVCRT_LOCK_SIZE)


@contextmanager
def file_lock(path: Path, *, exclusive: bool):
    """
    Acquire a file lock on *path*, creating parent directories as needed.

    :param path: path to the lock file (created if absent).
    :param exclusive: if True, acquire an exclusive (write) lock; if False, a shared
        (read) lock.  On Windows both modes are mapped to exclusive.
    """
    os.makedirs(path.parent, exist_ok=True)
    mode = _LOCK_EX if (exclusive or sys.platform == "win32") else _LOCK_SH
    with open(path, "wb+") as lock_fh:
        _lock(lock_fh.fileno(), mode)
        try:
            yield
        finally:
            _unlock(lock_fh.fileno())
