# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the bounded, demand-priority CXL prefetch queue."""

from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from lmcache.v1.storage_backend.cxl_resource_governor import CxlResourceGovernor
from lmcache.v1.storage_backend.storage_manager import StorageManager


class _FakeLocalCPU:
    def __init__(self):
        self.keys = set()

    def contains(self, key):
        return key in self.keys


class _FakeCXL:
    def __init__(self, local_cpu, governor):
        self.local_cpu = local_cpu
        self.governor = governor
        self.reason = None
        self.attempted = threading.Event()
        self.attempts = 0

    def contains(self, key):
        return key in self.local_cpu.keys

    def get_prefetch_failure_reason(self):
        return self.reason

    def submit_prefetch_task(self, key):
        self.attempted.set()
        self.attempts += 1
        with self.governor.try_background(operation="submit_prefetch_task") as admitted:
            if not admitted:
                self.reason = "cxl_resource_busy_local"
                return False
            self.reason = None
            self.local_cpu.keys.add(key)
            return True

    def batched_get_blocking(self, keys):
        # Return a stand-in staging object for the serving-side CXL read.  The
        # test only needs to exercise the demand/prefetch arbitration.
        return [object() for _ in keys]


def _make_manager(*, max_pending=4):
    local_cpu = _FakeLocalCPU()
    governor = CxlResourceGovernor()
    cxl = _FakeCXL(local_cpu, governor)

    manager = StorageManager.__new__(StorageManager)
    manager.storage_backends = {
        "CxlBackend": cxl,
        "LocalCPUBackend": local_cpu,
    }
    manager.worker_id = 0
    manager._freeze = False
    manager._freeze_lock = threading.Lock()
    manager.cxl_prefetch_enabled = True
    manager.cxl_prefetch_max_inflight = 1
    manager.cxl_prefetch_max_chunks = 8
    manager.cxl_prefetch_max_pending = max_pending
    manager.cxl_prefetch_queue_ttl_ms = 1000
    manager.cxl_prefetch_retry_interval_ms = 2
    manager._cxl_prefetch_executor = ThreadPoolExecutor(max_workers=1)
    manager._cxl_prefetch_futures = {}
    manager._cxl_prefetch_lock = threading.Lock()
    manager._cxl_prefetch_condition = threading.Condition(manager._cxl_prefetch_lock)
    manager._cxl_prefetch_queue = deque()
    manager._cxl_prefetch_pending = {}
    manager._cxl_prefetch_running = set()
    manager._cxl_prefetch_demand_won = set()
    manager._cxl_prefetch_active_attempts = 0
    manager._cxl_prefetch_stopping = False
    manager._cxl_prefetch_dispatcher = threading.Thread(
        target=manager._cxl_prefetch_dispatch_loop,
        name="test-cxl-prefetch-dispatcher",
        daemon=True,
    )
    manager._cxl_prefetch_stats = defaultdict(int)
    manager._cxl_prefetch_trace_capacity = 32
    manager._cxl_prefetch_admissions = {}
    manager._cxl_prefetch_task_outcomes = {}
    manager._cxl_prefetch_completed_trace = OrderedDict()
    manager._cxl_prefetch_dispatcher.start()
    return manager, cxl, local_cpu, governor


def _stop_manager(manager):
    with manager._cxl_prefetch_condition:
        manager._cxl_prefetch_stopping = True
        queued = list(manager._cxl_prefetch_queue)
        manager._cxl_prefetch_queue.clear()
        manager._cxl_prefetch_pending.clear()
        manager._cxl_prefetch_condition.notify_all()
    manager._cxl_prefetch_dispatcher.join(timeout=2)
    for work in queued:
        manager._set_cxl_prefetch_terminal_result(work, False, "shutdown")
    manager._cxl_prefetch_executor.shutdown(wait=True, cancel_futures=True)


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_resource_busy_is_retried_after_demand_drains():
    manager, cxl, local_cpu, governor = _make_manager()
    key = "key-1"
    try:
        with governor.serving(operation="demand"):
            result = manager.submit_cxl_prefetch([key], max_chunks=1)
            assert result["scheduled"] == 1
            future = manager._get_prefetch_future(key)
            assert future is not None
            assert cxl.attempted.wait(timeout=1)
            assert not future.done()
            assert not local_cpu.contains(key)

        assert _wait_until(future.done)
        assert future.result() is True
        assert local_cpu.contains(key)
        stats = manager.get_cxl_prefetch_stats()
        assert stats["resource_deferred"] >= 1
        assert stats.get("failed", 0) == 0
    finally:
        _stop_manager(manager)


def test_pending_queue_is_bounded():
    manager, _, _, governor = _make_manager(max_pending=1)
    try:
        with governor.serving(operation="demand"):
            result = manager.submit_cxl_prefetch(
                ["key-1", "key-2"], max_chunks=2
            )
            assert result["scheduled"] == 1
            assert result["queue_rejected"] == 1
    finally:
        # Release the demand gate so the accepted item can finish before the
        # test-owned executor is shut down.
        _stop_manager(manager)


def test_demand_does_not_wait_for_queued_retry():
    manager, cxl, _, governor = _make_manager(max_pending=1)
    try:
        with governor.serving(operation="demand"):
            result = manager.submit_cxl_prefetch(["key-demand"], max_chunks=1)
            assert result["scheduled"] == 1
            future = manager._get_prefetch_future("key-demand")
            assert future is not None
            assert cxl.attempted.wait(timeout=1)
            assert _wait_until(
                lambda: "key-demand" not in manager._cxl_prefetch_running
            )

            started = time.monotonic()
            assert manager._wait_for_cxl_prefetch("key-demand") is False
            assert time.monotonic() - started < 0.1
            assert not future.done()
    finally:
        _stop_manager(manager)


def test_resource_busy_eventually_expires_from_queue():
    manager, _, _, governor = _make_manager(max_pending=1)
    manager.cxl_prefetch_queue_ttl_ms = 20
    try:
        with governor.serving(operation="demand"):
            result = manager.submit_cxl_prefetch(["key-expire"], max_chunks=1)
            assert result["scheduled"] == 1
            future = manager._get_prefetch_future("key-expire")
            assert future is not None
            assert _wait_until(future.done)
        assert future.result() is False
        assert manager.get_cxl_prefetch_stats()["queue_expired"] >= 1
    finally:
        _stop_manager(manager)


def test_cxl_demand_cancels_queued_prefetch():
    manager, cxl, _, governor = _make_manager()
    key = "key-demand-cxl"
    try:
        # Keep the speculative attempt out of the CXL resource while the
        # serving read wins.  The failed background attempt is requeued.
        with governor.serving(operation="demand"):
            result = manager.submit_cxl_prefetch([key], max_chunks=1)
            assert result["scheduled"] == 1
            future = manager._get_prefetch_future(key)
            assert future is not None
            assert cxl.attempted.wait(timeout=1)
            assert _wait_until(
                lambda: key not in manager._cxl_prefetch_running
                and key in manager._cxl_prefetch_pending
            )

            # This is the point at which the real demand path has received a
            # CXL staging object.
            memory_objs = manager.batched_get(
                [key], location="CxlBackend", request_id="request-1"
            )
            assert memory_objs is not None
            assert memory_objs[0] is not None

        assert _wait_until(future.done)
        assert future.result() is False
        assert cxl.attempts == 1
        stats = manager.get_cxl_prefetch_stats()
        assert stats["demand_won"] == 1
        assert stats["queue_cancelled"] == 1
        assert stats.get("failed", 0) == 0
        assert stats.get("succeeded", 0) == 0
    finally:
        _stop_manager(manager)


def test_demand_won_marker_stops_late_background_attempt():
    manager, cxl, local_cpu, _ = _make_manager()
    key = "key-late-prefetch"
    try:
        manager._cxl_prefetch_demand_won.add(key)
        assert (
            manager._run_cxl_prefetch_task(
                key, cxl, local_cpu, request_id="request-2"
            )
            is False
        )
        assert cxl.attempts == 0
        outcome = manager._get_cxl_prefetch_task_outcome(key)
        assert outcome["promotion_outcome"] == "demand_won"
    finally:
        _stop_manager(manager)
