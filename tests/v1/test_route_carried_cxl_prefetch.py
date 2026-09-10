# SPDX-License-Identifier: Apache-2.0

"""Unit tests for request-carried CXL promotion registration."""

import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine


class _StorageManager:
    def __init__(self):
        self.calls = []

    def submit_cxl_prefetch(self, keys, *, max_chunks, request_id):
        self.calls.append((list(keys), max_chunks, request_id))
        return {"scheduled": len(keys), "deduplicated": 0}


def _key(chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey(
        fmt="vllm",
        model_name="test",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


def test_route_hint_registers_before_lookup_with_sparse_candidates():
    engine = LMCacheEngine.__new__(LMCacheEngine)
    storage_manager = _StorageManager()
    engine.storage_manager = storage_manager
    keys = [_key(index) for index in range(4)]

    engine._submit_cxl_prefetch_hint(
        "request-1",
        keys,
        {
            "lmcache.cxl_prefetch_hint": {
                "candidate_block_indices": [2, 0],
                "max_chunks": 2,
            }
        },
    )

    assert storage_manager.calls == [([keys[0], keys[2]], 2, "request-1")]


def test_route_hint_submits_all_candidates_in_bounded_batches():
    engine = LMCacheEngine.__new__(LMCacheEngine)
    storage_manager = _StorageManager()
    engine.storage_manager = storage_manager
    keys = [_key(index) for index in range(5)]

    engine._submit_cxl_prefetch_hint(
        "request-batched",
        keys,
        {
            "lmcache.cxl_prefetch_hint": {
                "candidate_block_indices": [0, 1, 2, 3, 4],
                "max_chunks": 2,
            }
        },
    )

    assert storage_manager.calls == [
        ([keys[0], keys[1]], 2, "request-batched"),
        ([keys[2], keys[3]], 2, "request-batched"),
        ([keys[4]], 1, "request-batched"),
    ]


def test_route_hint_respects_worker_batch_cap():
    engine = LMCacheEngine.__new__(LMCacheEngine)
    storage_manager = _StorageManager()
    storage_manager.cxl_prefetch_max_chunks = 1
    engine.storage_manager = storage_manager
    keys = [_key(index) for index in range(3)]

    engine._submit_cxl_prefetch_hint(
        "request-worker-cap",
        keys,
        {
            "lmcache.cxl_prefetch_hint": {
                "candidate_block_indices": [0, 1, 2],
                "max_chunks": 3,
            }
        },
    )

    assert storage_manager.calls == [
        ([keys[0]], 1, "request-worker-cap"),
        ([keys[1]], 1, "request-worker-cap"),
        ([keys[2]], 1, "request-worker-cap"),
    ]


def test_malformed_or_out_of_range_route_hint_is_fail_open():
    engine = LMCacheEngine.__new__(LMCacheEngine)
    storage_manager = _StorageManager()
    engine.storage_manager = storage_manager
    keys = [_key(1)]

    engine._submit_cxl_prefetch_hint(
        "request-2",
        keys,
        {"lmcache.cxl_prefetch_hint": {"candidate_block_indices": [9], "max_chunks": 1}},
    )
    engine._submit_cxl_prefetch_hint(
        "request-3",
        keys,
        {"lmcache.cxl_prefetch_hint": {"candidate_block_indices": ["bad"], "max_chunks": 1}},
    )

    assert storage_manager.calls == []


def test_route_hint_uses_absolute_block_indices_after_computed_prefix():
    engine = LMCacheEngine.__new__(LMCacheEngine)
    storage_manager = _StorageManager()
    engine.storage_manager = storage_manager
    keys = [_key(index) for index in range(2)]

    engine._submit_cxl_prefetch_hint(
        "request-4",
        keys,
        {
            "lmcache.cxl_prefetch_hint": {
                "candidate_block_indices": [4, 5],
                "max_chunks": 2,
                "key_offset": 4,
            }
        },
    )

    assert storage_manager.calls == [([keys[0], keys[1]], 2, "request-4")]
