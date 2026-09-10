# SPDX-License-Identifier: Apache-2.0
"""Regression tests for direct worker-to-controller KV operation messages."""

import asyncio

from lmcache.v1.cache_controller.controller_manager import LMCacheControllerManager
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg


class _FakeKVController:
    def __init__(self):
        self.sequence_checks = []
        self.admits = []
        self.evicts = []

    def check_sequence_number(self, msg):
        self.sequence_checks.append(msg)

    async def admit(self, msg):
        self.admits.append(msg)

    async def evict(self, msg):
        self.evicts.append(msg)


def test_direct_kv_messages_are_dispatched_to_kv_controller():
    manager = LMCacheControllerManager.__new__(LMCacheControllerManager)
    manager.kv_controller = _FakeKVController()
    admit = KVAdmitMsg("instance", 0, 11, "CxlBackend", 3)
    evict = KVEvictMsg("instance", 0, 12, "CxlBackend", 4)

    async def run():
        await manager.handle_worker_message(admit)
        await manager.handle_worker_message(evict)

    asyncio.run(run())

    assert manager.kv_controller.sequence_checks == [admit, evict]
    assert manager.kv_controller.admits == [admit]
    assert manager.kv_controller.evicts == [evict]
