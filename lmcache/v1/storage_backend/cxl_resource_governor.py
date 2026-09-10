# SPDX-License-Identifier: Apache-2.0
"""Serving-priority arbitration for shared CXL bandwidth.

The CXL backend is synchronous and is used by both the serving data path and
the best-effort CPU->CXL offload path.  An asyncio task or a background thread
does not, by itself, give the two paths different resource priority.  This
small, process-local governor provides that missing admission control:

* serving operations may wait for one already-running background chunk, then
  take priority over future background work;
* background work is admitted only when there is no serving operation active
  or waiting, and never waits for the resource;
* background callers can therefore defer a chunk without blocking the serving
  event loop or consuming an unbounded queue of pinned buffers.

The governor deliberately limits only one background operation at a time per
backend process.  The native CXL shared-memory lock still provides the
cross-process correctness guarantee; this class is the local latency and
bandwidth-priority layer.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import hashlib
import logging
import os
import time
from threading import Condition, Lock
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - CXL deployments are Linux-based.
    fcntl = None


logger = logging.getLogger(__name__)


def _resource_trace_enabled() -> bool:
    return os.getenv("LMCACHE_CXL_RESOURCE_TRACE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True, slots=True)
class CxlResourceSnapshot:
    """A bounded diagnostic snapshot of the local CXL arbiter."""

    serving_active: int
    serving_waiting: int
    background_active: bool
    background_admitted: int
    background_deferred: int
    serving_waited: int
    serving_wait_ms_total: float
    serving_wait_ms_max: float
    serving_wait_by_operation: dict[str, int] = field(default_factory=dict)
    serving_wait_ms_by_operation: dict[str, float] = field(default_factory=dict)
    background_active_ms_total: float = 0.0
    background_active_ms_max: float = 0.0


@dataclass(frozen=True, slots=True)
class CxlSharedResourceSnapshot:
    """Telemetry for the process-shared POSIX CXL access lock."""

    serving_lock_waited: int
    serving_lock_wait_ms_total: float
    serving_lock_wait_ms_max: float
    background_admitted: int
    background_deferred: int
    background_active_ms_total: float
    background_active_ms_max: float
    serving_lock_wait_by_operation: dict[str, int] = field(default_factory=dict)
    serving_lock_wait_ms_by_operation: dict[str, float] = field(default_factory=dict)


class CxlSharedResourceLock:
    """Coordinate CXL payload access between local worker processes.

    The Python governor is process-local.  vLLM/LMCache normally starts one
    EngineCore process per GPU, so a process-local lock alone would still let
    two idle-looking workers write the shared CXL mapping at the same time.
    A short-lived POSIX advisory lock extends the same policy to processes on
    this host:

    * serving uses a shared lock and can proceed concurrently with other
      serving operations;
    * background offload uses a non-blocking exclusive lock and is deferred if
      any serving process currently owns the shared lock.

    The lock coordinates payload copies only.  CXL metadata correctness
    remains the responsibility of the native CXL shared-memory locks.
    """

    def __init__(
        self,
        resource_id: str,
        *,
        lock_dir: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._stats_lock = Lock()
        self._serving_lock_waited = 0
        self._serving_lock_wait_ms_total = 0.0
        self._serving_lock_wait_ms_max = 0.0
        self._serving_lock_wait_by_operation: dict[str, int] = {}
        self._serving_lock_wait_ms_by_operation: dict[str, float] = {}
        self._background_admitted = 0
        self._background_deferred = 0
        self._background_active_ms_total = 0.0
        self._background_active_ms_max = 0.0

        self.enabled = bool(enabled and fcntl is not None)
        if not self.enabled:
            self.path = None
            return

        directory = lock_dir or os.getenv(
            "LMCACHE_CXL_RESOURCE_LOCK_DIR", "/tmp"
        )
        if not os.path.isdir(directory):
            self.enabled = False
            self.path = None
            return
        digest = hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:24]
        self.path = os.path.join(directory, f"lmcache-cxl-resource-{digest}.lock")

    def _open(self) -> int | None:
        if not self.enabled or self.path is None:
            return None
        return os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)

    @staticmethod
    def _close(fd: int | None) -> None:
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError:
            pass

    @contextmanager
    def serving(self, operation: str = "unknown") -> Iterator[None]:
        """Hold a process-shared read lock for one serving CXL operation."""

        fd = self._open()
        blocked = False
        wait_started = 0.0
        try:
            if fd is not None:
                assert fcntl is not None
                # Probe first so telemetry distinguishes an uncontended lock
                # acquisition from a serving operation that really waited for
                # an exclusive background offload in another process.
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    blocked = True
                    wait_started = time.perf_counter()
                    fcntl.flock(fd, fcntl.LOCK_SH)
                if blocked:
                    wait_ms = (time.perf_counter() - wait_started) * 1000.0
                    with self._stats_lock:
                        self._serving_lock_waited += 1
                        self._serving_lock_wait_ms_total += wait_ms
                        self._serving_lock_wait_ms_max = max(
                            self._serving_lock_wait_ms_max, wait_ms
                        )
                        self._serving_lock_wait_by_operation[operation] = (
                            self._serving_lock_wait_by_operation.get(operation, 0) + 1
                        )
                        self._serving_lock_wait_ms_by_operation[operation] = (
                            self._serving_lock_wait_ms_by_operation.get(operation, 0.0)
                            + wait_ms
                        )
                    if _resource_trace_enabled():
                        logger.info(
                            "CXL_RESOURCE_TRACE kind=serving_wait source=shared_lock "
                            "operation=%s wait_ms=%.3f pid=%s",
                            operation,
                            wait_ms,
                            os.getpid(),
                        )
            yield
        finally:
            if fd is not None:
                assert fcntl is not None
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    self._close(fd)

    @contextmanager
    def try_background(self, operation: str = "unknown") -> Iterator[bool]:
        """Try to hold a process-shared exclusive lock without waiting."""

        fd = self._open()
        if fd is None:
            with self._stats_lock:
                self._background_admitted += 1
            started = time.perf_counter()
            yield True
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._stats_lock:
                self._background_active_ms_total += elapsed_ms
                self._background_active_ms_max = max(
                    self._background_active_ms_max, elapsed_ms
                )
            return

        admitted = False
        started = 0.0
        try:
            assert fcntl is not None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                admitted = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
            with self._stats_lock:
                if admitted:
                    self._background_admitted += 1
                else:
                    self._background_deferred += 1
            if admitted:
                started = time.perf_counter()
            yield admitted
        finally:
            if admitted:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with self._stats_lock:
                    self._background_active_ms_total += elapsed_ms
                    self._background_active_ms_max = max(
                        self._background_active_ms_max, elapsed_ms
                    )
                assert fcntl is not None
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    self._close(fd)
            else:
                self._close(fd)

    def snapshot(self) -> CxlSharedResourceSnapshot:
        with self._stats_lock:
            return CxlSharedResourceSnapshot(
                serving_lock_waited=self._serving_lock_waited,
                serving_lock_wait_ms_total=self._serving_lock_wait_ms_total,
                serving_lock_wait_ms_max=self._serving_lock_wait_ms_max,
                serving_lock_wait_by_operation=dict(
                    self._serving_lock_wait_by_operation
                ),
                serving_lock_wait_ms_by_operation=dict(
                    self._serving_lock_wait_ms_by_operation
                ),
                background_admitted=self._background_admitted,
                background_deferred=self._background_deferred,
                background_active_ms_total=self._background_active_ms_total,
                background_active_ms_max=self._background_active_ms_max,
            )


class CxlResourceGovernor:
    """Give serving CXL operations priority over background copies.

    ``serving()`` is a normal blocking context manager because a demand read
    or write must eventually complete.  ``try_background()`` is intentionally
    non-blocking: an offload that loses the race is deferred to the next
    planner round instead of waiting behind serving work.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._serving_active = 0
        self._serving_waiting = 0
        self._background_active = False
        self._background_admitted = 0
        self._background_deferred = 0
        self._serving_waited = 0
        self._serving_wait_ms_total = 0.0
        self._serving_wait_ms_max = 0.0
        self._serving_wait_by_operation: dict[str, int] = {}
        self._serving_wait_ms_by_operation: dict[str, float] = {}
        self._background_active_ms_total = 0.0
        self._background_active_ms_max = 0.0

    @contextmanager
    def serving(self, operation: str = "unknown") -> Iterator[None]:
        """Enter a high-priority serving CXL operation."""

        waited = False
        wait_started = 0.0
        with self._condition:
            self._serving_waiting += 1
            try:
                while self._background_active:
                    waited = True
                    if wait_started == 0.0:
                        wait_started = time.perf_counter()
                    self._condition.wait()
                self._serving_active += 1
            finally:
                self._serving_waiting -= 1
            if waited:
                wait_ms = (time.perf_counter() - wait_started) * 1000.0
                self._serving_waited += 1
                self._serving_wait_ms_total += wait_ms
                self._serving_wait_ms_max = max(self._serving_wait_ms_max, wait_ms)
                self._serving_wait_by_operation[operation] = (
                    self._serving_wait_by_operation.get(operation, 0) + 1
                )
                self._serving_wait_ms_by_operation[operation] = (
                    self._serving_wait_ms_by_operation.get(operation, 0.0) + wait_ms
                )
                if _resource_trace_enabled():
                    logger.info(
                        "CXL_RESOURCE_TRACE kind=serving_wait source=local_governor "
                        "operation=%s wait_ms=%.3f pid=%s",
                        operation,
                        wait_ms,
                        os.getpid(),
                    )

        try:
            yield
        finally:
            with self._condition:
                self._serving_active -= 1
                self._condition.notify_all()

    @contextmanager
    def try_background(self, operation: str = "unknown") -> Iterator[bool]:
        """Try to enter one low-priority background CXL operation.

        This method never waits.  In particular, a waiting serving caller
        prevents a new background operation from starting, which avoids
        starving demand traffic between two background chunks.
        """

        admitted = False
        with self._condition:
            if (
                not self._background_active
                and self._serving_active == 0
                and self._serving_waiting == 0
            ):
                self._background_active = True
                self._background_admitted += 1
                admitted = True
            else:
                self._background_deferred += 1

        started = time.perf_counter() if admitted else 0.0
        try:
            yield admitted
        finally:
            if admitted:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._background_active_ms_total += elapsed_ms
                self._background_active_ms_max = max(
                    self._background_active_ms_max, elapsed_ms
                )
                with self._condition:
                    self._background_active = False
                    self._condition.notify_all()

    def snapshot(self) -> CxlResourceSnapshot:
        """Return counters suitable for logs or benchmark telemetry."""

        with self._condition:
            return CxlResourceSnapshot(
                serving_active=self._serving_active,
                serving_waiting=self._serving_waiting,
                background_active=self._background_active,
                background_admitted=self._background_admitted,
                background_deferred=self._background_deferred,
                serving_waited=self._serving_waited,
                serving_wait_ms_total=self._serving_wait_ms_total,
                serving_wait_ms_max=self._serving_wait_ms_max,
                serving_wait_by_operation=dict(self._serving_wait_by_operation),
                serving_wait_ms_by_operation=dict(
                    self._serving_wait_ms_by_operation
                ),
                background_active_ms_total=self._background_active_ms_total,
                background_active_ms_max=self._background_active_ms_max,
            )
