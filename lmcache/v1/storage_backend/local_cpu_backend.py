# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union
import hashlib
import os
import threading
import time
from pathlib import Path

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import (
    CacheEngineKey,
    CacheRemoveEvent,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    parse_cache_key,
)
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lazy_memory_allocator import LazyMixedMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.system_detection import NUMADetector, SystemMemoryDetector

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalCPUBackend(AllocatorBackendInterface):
    """
    Even if local_cpu is False (the hot_cache is not used), contains(),
    insert_key(), remove(), get_blocking(), get_keys(), and clear()
    are still callable by the storage manager.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        memory_allocator: Optional[MemoryAllocatorInterface] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()
        # Prefetch entries are tracked separately so a later demand access can
        # promote them from probationary to protected accounting. Pure
        # prefetch allocation uses eviction=False in the engine, so it cannot
        # evict an existing demand entry under pressure.
        self.prefetch_keys: set[CacheEngineKey] = set()
        self.prefetch_metadata: dict[CacheEngineKey, dict[str, int | float]] = {}
        extra_config = config.extra_config if isinstance(config.extra_config, dict) else {}
        self.prefetch_ttl_ns = int(
            extra_config.get("prefetch_ttl_ms", 500) * 1_000_000
        )

        self.use_hot = config.local_cpu
        # NOTE: we keep the memory allocator argument for temporary
        # test compatibility
        # TODO: fix the tests to get rid the memory allocator
        assert metadata is not None or memory_allocator is not None
        self.memory_allocator = (
            self.initialize_allocator(config, metadata)  # type: ignore
            if memory_allocator is None
            else memory_allocator
        )
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.cpu_lock = threading.Lock()

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending
        self._kv_event_sink: Optional[
            Callable[[CacheStoreEvent | CacheRemoveEvent], None]
        ] = None

        # Store config and metadata for chunk budget calculation
        self.config = config
        self.metadata = metadata

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        # Batched message sender for controller communication
        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        # Initialize batched message sender
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),  # Backend location
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.local_cpu_hot_cache_count.set_function(
                lambda: len(self.hot_cache)
            )
            prometheus_logger.local_cpu_keys_in_request_count.set_function(
                lambda: len(self.keys_in_request)
            )

    def _get_evict_candidates_locked(self, num_candidates: int = 1):
        """Prefer probationary prefetch entries before demand-resident data.

        ``cpu_lock`` must be held by the caller.  Iterating ``hot_cache`` keeps
        the cache policy's LRU order while the membership check gives
        speculative entries an admission-protection priority.
        """
        now_ns = time.time_ns()
        probationary = [
            key
            for key, memory_obj in self.hot_cache.items()
            if key in self.prefetch_keys and memory_obj.can_evict
        ]
        if probationary:
            probationary.sort(
                key=lambda key: (
                    0
                    if self.prefetch_metadata.get(key, {}).get("expire_ns", 0)
                    and int(self.prefetch_metadata[key]["expire_ns"]) <= now_ns
                    else 1,
                    float(self.prefetch_metadata.get(key, {}).get("score", 0.0)),
                )
            )
            return probationary[:num_candidates]
        return self.cache_policy.get_evict_candidates(
            self.hot_cache, num_candidates=num_candidates
        )

    def __str__(self):
        return self.__class__.__name__

    def set_kv_event_sink(
        self, sink: Callable[[CacheStoreEvent | CacheRemoveEvent], None]
    ) -> None:
        self._kv_event_sink = sink

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.cpu_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.hot_cache)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        contains() and exists_in_put_tasks() should be checked together
        """
        return False

    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        """
        Synchronously put the MemoryObj into the local cpu backend.
        """

        with self.cpu_lock:
            if key in self.hot_cache:
                return None

            memory_obj.ref_count_up()
            self.hot_cache[key] = memory_obj

            self.cache_policy.update_on_put(key)

            # Push kv admit msg with batching
            if self.batched_msg_sender is not None:
                self.batched_msg_sender.add_kv_op(
                    op_type=OpType.ADMIT,
                    key=key.chunk_hash,
                )

        return None

    def submit_prefetch_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        *,
        expire_ns: Optional[int] = None,
        score: float = 0.0,
        trigger_key: Optional[CacheEngineKey] = None,
    ) -> Optional[Future]:
        """Admit a prefetch object without evicting demand-resident entries."""

        with self.cpu_lock:
            if key in self.hot_cache:
                return None
            memory_obj.ref_count_up()
            self.hot_cache[key] = memory_obj
            self.prefetch_keys.add(key)
            self.prefetch_metadata[key] = {
                "inserted_ns": time.time_ns(),
                "expire_ns": expire_ns or time.time_ns() + self.prefetch_ttl_ns,
                "score": score,
                "demand_access_count": 0,
            }
            if trigger_key is not None:
                self.prefetch_metadata[key]["trigger_key_hash"] = int(
                    trigger_key.chunk_hash
                )
            self.cache_policy.update_on_put(key)
            if self.batched_msg_sender is not None:
                self.batched_msg_sender.add_kv_op(
                    op_type=OpType.ADMIT,
                    key=key.chunk_hash,
                )
        return None

    def mark_demand_access(self, key: CacheEngineKey) -> bool:
        """Mark a prefetched entry as demand-protected."""

        with self.cpu_lock:
            if key not in self.prefetch_keys:
                return False
            self.prefetch_keys.discard(key)
            metadata = self.prefetch_metadata.get(key)
            if metadata is not None:
                metadata["demand_access_count"] = int(
                    metadata.get("demand_access_count", 0)
                ) + 1
                self.prefetch_metadata.pop(key, None)
            return True

    def expire_prefetch_entries(self, now_ns: Optional[int] = None) -> int:
        """Remove expired probationary entries and return the count removed."""
        now = now_ns or time.time_ns()
        with self.cpu_lock:
            expired = [
                key
                for key in self.prefetch_keys
                if self.prefetch_metadata.get(key, {}).get("expire_ns", 0)
                and int(self.prefetch_metadata[key]["expire_ns"]) <= now
            ]
        return self.batched_remove(expired, force=False) if expired else 0

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """
        Synchronously put the MemoryObjs into the local cpu backend.
        """
        if not self.use_hot:
            return

        # TODO(Jiayi): optimize this with batching
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            # ref count up for caller to avoid situation where the memory_obj
            # is evicted from the local cpu backend before the caller calls
            # ref count up themselves
            memory_obj.ref_count_up()
            return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs = []
        with self.cpu_lock:
            for key in keys:
                mem_obj = self.hot_cache[key]
                mem_obj.ref_count_up()
                mem_objs.append(mem_obj)
        return mem_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        # NOTE(Jiayi): Only prefix chunks are counted.
        num_hit_chunks = 0
        with self.cpu_lock:
            for key in keys:
                if key not in self.hot_cache:
                    return num_hit_chunks
                if pin:
                    self.hot_cache[key].pin()
                    # vllm lookup sets pin to True
                    self.keys_in_request.append(key)
                num_hit_chunks += 1
        return num_hit_chunks

    def pin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        # NOTE: Historically `force=False` skipped locking and policy update.
        # That is risky under concurrency and leaks policy state (e.g., LRU access
        # counts). LocalCPUBackend is in-process and should be thread-safe, so we
        # always take the lock and always update policy state when we remove.
        evicted_items: list[tuple[CacheEngineKey, MemoryObj]] = []
        with self.cpu_lock:
            memory_obj = self.hot_cache.pop(key, None)
            if memory_obj is None:
                return False

            self.prefetch_keys.discard(key)
            self.prefetch_metadata.pop(key, None)

            self.cache_policy.update_on_force_evict(key)
            evicted_items.append((key, memory_obj))

            if self.batched_msg_sender is not None:
                self.batched_msg_sender.add_kv_op(
                    op_type=OpType.EVICT,
                    key=key.chunk_hash,
                )

        # Notify outside the lock to avoid deadlocks / long critical sections.
        #
        # Important: keep the backend's ownership ref alive during the callback
        # so listeners can safely inspect/copy from the MemoryObj. Listeners that
        # need to retain the object asynchronously must take their own ref via
        # ref_count_up()/ref_count_down().
        self._notify_evict(evicted_items)
        # Now release the backend's ownership ref (added in submit_put_task()).
        for _, memory_obj in evicted_items:
            memory_obj.ref_count_down()
        if self._kv_event_sink is not None:
            self._kv_event_sink(
                CacheRemoveEvent(block_hashes=[key.chunk_hash], medium="CPU")
            )
        return True

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        # Override base implementation so we can:
        # 1) atomically remove multiple keys
        # 2) notify eviction listeners once with (key, MemoryObj) pairs
        evicted_items: list[tuple[CacheEngineKey, MemoryObj]] = []
        num_removed = 0

        with self.cpu_lock:
            for key in keys:
                memory_obj = self.hot_cache.pop(key, None)
                if memory_obj is None:
                    continue

                self.prefetch_keys.discard(key)
                self.prefetch_metadata.pop(key, None)

                self.cache_policy.update_on_force_evict(key)
                evicted_items.append((key, memory_obj))
                num_removed += 1

                if self.batched_msg_sender is not None:
                    self.batched_msg_sender.add_kv_op(
                        op_type=OpType.EVICT,
                        key=key.chunk_hash,
                    )

        # Notify first (listeners may take their own refs), then drop the
        # backend ownership refs.
        self._notify_evict(evicted_items)
        for _, memory_obj in evicted_items:
            memory_obj.ref_count_down()
        if self._kv_event_sink is not None and evicted_items:
            self._kv_event_sink(
                CacheRemoveEvent(
                    block_hashes=[k.chunk_hash for k, _ in evicted_items], medium="CPU"
                )
            )
        return num_removed

    def _calculate_effective_cpu_size(
        self,
        configured_cpu_size: float,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ) -> float:
        """
        Calculate the effective CPU memory size based on system available memory
        and reserve memory configuration.

        Args:
            configured_cpu_size: The configured CPU memory size in GB
            config: The LMCache engine configuration
            metadata: Optional metadata for first rank handling

        Returns:
            The effective CPU memory size in GB
        """

        save_only_first_rank = (
            metadata is not None
            and config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if not save_only_first_rank:
            # Do not adjust cpu_size if save_only_first_rank is False for now
            return configured_cpu_size

        # Get the system available memory and calculate effective cpu_size
        system_available_memory_gb = SystemMemoryDetector.get_available_memory_gb()
        # Get reserve memory size from config
        reserve_cpu_size = config.reserve_local_cpu_size

        # TODO(baoloongmao): For disable save_only_first_rank case,
        #  we need to avoid multi-rank race condition in future.
        #  But for enable save_only_first_rank case,
        #  we can handle reserve memory simply since non-first ranks
        #  do not allocate memory.
        # Effective memory: min(configured_size, available_memory - reserve_size)
        if system_available_memory_gb > 0:
            max_usable_memory = max(0, system_available_memory_gb - reserve_cpu_size)
            effective_cpu_size = min(configured_cpu_size, max_usable_memory)
            logger.info(
                f"Adjusted CPU memory size from {configured_cpu_size:.2f} GB "
                f"to {effective_cpu_size:.2f} GB "
                f"(system available: {system_available_memory_gb:.2f} GB, "
                f"reserve: {reserve_cpu_size:.2f} GB)"
            )
            assert effective_cpu_size > 0
            return effective_cpu_size
        else:
            logger.warning(
                "Could not determine system available memory, using configured cpu_size"
            )
            return configured_cpu_size

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ) -> MemoryAllocatorInterface:
        cpu_size = config.max_local_cpu_size

        if metadata is not None:
            # save_only_first_rank only works when use mla
            save_only_first_rank = (
                config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
                and metadata.use_mla
            )

            if save_only_first_rank and metadata.is_first_rank():
                # Only the first rank will save the cache,
                # so we need to set it larger than other ranks
                cpu_size = config.get_extra_config_value(
                    "first_rank_max_local_cpu_size", cpu_size
                )

        # Detect the numa mapping
        numa_mapping = NUMADetector.get_numa_mapping(config)
        logger.info(f"NUMA mapping {numa_mapping}")

        # Calculate effective CPU memory size
        cpu_size = self._calculate_effective_cpu_size(cpu_size, config, metadata)

        if config.enable_p2p:
            # TODO(baoloongmao): Add lazy memory allocator support for P2P mode
            # For now, keep the original P2P implementation
            assert metadata is not None
            meta_shape = torch.Size(metadata.kv_shape)
            # TODO(Jiayi): remove this hardcode
            new_shape = torch.Size(
                [
                    meta_shape[1],
                    meta_shape[0],
                    meta_shape[2],
                    meta_shape[3] * meta_shape[4],
                ]
            )
            paged_mem_allocator = PagedCpuGpuMemoryAllocator()
            chunk_size_bytes = get_size_bytes([new_shape], [metadata.kv_dtype])
            origin_cpu_size_bytes = int(cpu_size * 1024**3)
            align_cpu_size_bytes = (
                origin_cpu_size_bytes // chunk_size_bytes * chunk_size_bytes
            )
            logger.info(
                f"Auto align cpu size bytes, origin: {origin_cpu_size_bytes}, "
                f"aligned: {align_cpu_size_bytes}, chunk size: {chunk_size_bytes}"
            )
            paged_mem_allocator.init_cpu_memory_allocator(
                align_cpu_size_bytes,
                shapes=[new_shape],
                dtypes=[metadata.kv_dtype],
                fmt=MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
                numa_mapping=numa_mapping,
            )
            return paged_mem_allocator
        else:
            # Check if lazy memory allocator should be enabled
            use_lazy = (
                config.enable_lazy_memory_allocator
                and cpu_size > config.lazy_memory_safe_size
            )

            if use_lazy:
                logger.info(
                    f"Using LazyMixedMemoryAllocator with "
                    f"initial_ratio={config.lazy_memory_initial_ratio}, "
                    f"expand_trigger_ratio="
                    f"{config.lazy_memory_expand_trigger_ratio}, "
                    f"step_ratio={config.lazy_memory_step_ratio}"
                )
                return LazyMixedMemoryAllocator(
                    int(cpu_size * 1024**3),
                    config=config,
                    numa_mapping=numa_mapping,
                    memory_limit_callback=lambda: int(
                        self._calculate_effective_cpu_size(cpu_size, config, metadata)
                        * 1024**3
                    ),
                )
            else:
                if config.enable_lazy_memory_allocator:
                    logger.info(
                        f"LazyMixedMemoryAllocator is disabled because "
                        f"cpu_size ({cpu_size:.2f} GB) does not exceed "
                        f"lazy_memory_safe_size "
                        f"({config.lazy_memory_safe_size:.2f} GB). "
                        f"Using MixedMemoryAllocator instead."
                    )
                return MixedMemoryAllocator(
                    int(cpu_size * 1024**3),
                    numa_mapping=numa_mapping,
                )

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False

        busy_loop should only be used for retrieve
        the reasoning is that:

        1. synchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - one retrieve at a time (okay to busy loop because stores will clear)

        2. asynchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - many retrieves happen concurrently
        (we use the async serializer to handle this)
        """
        logger.debug(
            f"Allocating memory in local cpu backend with busy loop: {busy_loop}"
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if memory_obj is not None or not eviction:
            return memory_obj

        evict_keys_count = 0
        num_attempts = 0
        while True:
            # whether or not this request needs to wait or other requests
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = 1
                # NOTE: Pick eviction candidates under lock, but perform the
                # actual removal outside the lock. LocalCPUBackend.remove/batched_remove
                # will take the lock and may notify listeners; calling them while
                # holding cpu_lock would deadlock.
                evict_keys: list[CacheEngineKey] = []
                with self.cpu_lock:
                    evict_keys = self._get_evict_candidates_locked(
                        num_candidates
                    )

                if evict_keys:
                    # we can continue trying to evict from the hot_cache
                    # and don't need to wait for other requests yet
                    wait_other_requests = False
                    logger.debug(f"Evicting {len(evict_keys)} chunks from cpu memory")
                    self.batched_remove(evict_keys, force=False)
                    evict_keys_count += len(evict_keys)
                else:
                    self.stats_monitor.update_local_cpu_evict_failed_count(num_candidates)

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local cpu backend allocate()"
            )

        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False

        busy_loop should only be used for retrieve
        the reasoning is that:

        1. synchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - one retrieve at a time (okay to busy loop because stores will clear)

        2. asynchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - many retrieves happen concurrently
        (we use the async serializer to handle this)
        """
        logger.debug(
            f"Batched allocating memory in local cpu backend"
            f" with busy loop: {busy_loop}"
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_objs = self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt
        )

        if memory_objs is not None or not eviction:
            return memory_objs

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)

        evict_keys_count = 0
        num_attempts = 0
        while True:
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = 1
                evict_keys = None
                with self.cpu_lock:
                    evict_keys = self._get_evict_candidates_locked(
                        num_candidates
                    )

                    # HACK: We assume batch_size=num_layers here.
                    # FIXME: We also assume if the one layer's ref_count > 1 or pinned,
                    # then the other layers are also ref_count > 1 or
                    # pinned in the cpu memory. This might not be true.
                    if evict_keys:
                        evict_keys_count += len(evict_keys)
                        wait_other_requests = False
                        for evict_key in evict_keys:
                            evict_key_all_layer = evict_key.split_layers(batch_size)

                            # TODO(Jiayi): batched allocate is not supported through
                            # `batched_remove`. Therefore, features like usage tracking
                            # is not supported.
                            old_mem_objs = []
                            for key in evict_key_all_layer:
                                old_mem_objs.append(self.hot_cache[key])
                                self.cache_policy.update_on_force_evict(key)
                                self.hot_cache.pop(key, None)
                                self.prefetch_keys.discard(key)
                                self.prefetch_metadata.pop(key, None)

                            self.memory_allocator.batched_free(old_mem_objs)

                            logger.debug(
                                f"Evicting {len(old_mem_objs)} chunks from cpu memory"
                            )
                    else:
                        self.stats_monitor.update_local_cpu_evict_failed_count(
                            num_candidates
                        )

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            memory_objs = self.memory_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
            if memory_objs:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local cpu backend batched_allocate()"
            )
        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_objs

    def get_full_chunk_size(self) -> int:
        logger.info("Calculating the size of a single LMCache chunk")
        assert self.metadata is not None, (
            "metadata required for chunk budget calculation"
        )

        chunk_tokens = self.config.chunk_size
        # already accounted for parallelism
        kv_shape = (
            self.metadata.kv_shape
        )  # [num_layers, kv_size, chunk_size, num_heads, head_size]
        num_layers = kv_shape[0]
        kv_size = kv_shape[1]  # 1 for MLA, 2 for regular
        # per gpu
        num_heads = kv_shape[3]
        head_size = kv_shape[4]
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize

        if self.layerwise:
            # layerwise: [chunk_tokens, kv_size, hidden_dim]
            chunk_bytes = chunk_tokens * kv_size * hidden_dim * dtype_size
        else:
            # full: [kv_size, num_layers, chunk_tokens, hidden_dim]
            chunk_bytes = kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size
        logger.debug(
            f"Stats received: num_layers={num_layers}, kv_size={kv_size}, "
            f"chunk_tokens={chunk_tokens}, head_dim={head_size}, "
            f"dtype_size={dtype_size}, "
            f"hidden_dim={hidden_dim}"
        )
        logger.debug(f"Calculated bytes per chunk per rank: {chunk_bytes}")
        return chunk_bytes

    def calculate_chunk_budget(self) -> int:
        """
        Calculate the maximum number of chunks that can be allocated concurrently
        without causing memory deadlocks in the async loading system.

        Returns:
            int: The estimated chunk budget for concurrent allocations
        """
        total_memory = int(self.config.max_local_cpu_size * 1024**3)
        chunk_bytes = self.get_full_chunk_size()
        # add alignment overhead
        # (MixedMemoryAllocator uses TensorMemoryAllocator with 4KB alignment)
        assert hasattr(self.memory_allocator, "align_bytes")
        alignment = self.memory_allocator.align_bytes
        aligned_chunk_bytes = ((chunk_bytes + alignment - 1) // alignment) * alignment

        # calculate budget with safety margin
        max_chunks = total_memory // aligned_chunk_bytes

        return max_chunks

    def get_keys(self) -> List[CacheEngineKey]:
        """
        array ordering of keys from LRU to MRU
        """
        with self.cpu_lock:
            return list(self.hot_cache.keys())

    @staticmethod
    def _snapshot_digest_update(
        digest: "hashlib._Hash",
        key: CacheEngineKey,
        tensor: torch.Tensor,
        fmt: MemoryFormat,
        pin_count: int,
        is_prefetch: bool,
        access_count: int,
    ) -> None:
        """Add stable cache identity and contents to a LocalCPU digest."""
        digest.update(
            (
                f"{key.to_string()}|{fmt.value}|{pin_count}|{int(is_prefetch)}|"
                f"{access_count}|{tuple(tensor.shape)}|{tensor.dtype}\n"
            ).encode("utf-8")
        )
        raw = tensor.detach().contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())

    def snapshot(
        self,
        path: str,
        event_metadata: Optional[
            dict[int, tuple[Optional[int], list[int], Optional[int], Optional[int]]]
        ] = None,
    ) -> dict[str, Any]:
        """Persist the allocator-owned LocalCPU cache without evicting it.

        The file contains logical cache objects rather than allocator addresses;
        restore allocates fresh objects from the destination worker's allocator
        and copies the tensor bytes into them.  Objects are temporarily
        reference-counted while their tensors are copied, so taking a snapshot
        is safe against concurrent eviction.
        """
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        held: list[tuple[CacheEngineKey, MemoryObj, dict[str, Any], int]] = []
        policy_access = getattr(self.cache_policy, "get_access_count", None)

        with self.cpu_lock:
            for key, memory_obj in self.hot_cache.items():
                memory_obj.ref_count_up()
                access_count = (
                    int(policy_access(key)) if callable(policy_access) else 0
                )
                held.append(
                    (
                        key,
                        memory_obj,
                        {
                            "pin_count": int(memory_obj.meta.pin_count),
                            "is_prefetch": key in self.prefetch_keys,
                            "prefetch_metadata": dict(
                                self.prefetch_metadata.get(key, {})
                            ),
                        },
                        access_count,
                    )
                )

        entries: list[dict[str, Any]] = []
        skipped: list[str] = []
        digest = hashlib.sha256()
        total_tokens = 0
        total_bytes = 0
        try:
            for key, memory_obj, state, access_count in held:
                tensor = memory_obj.tensor
                if tensor is None:
                    skipped.append(key.to_string())
                    continue
                tensor_copy = tensor.detach().contiguous().clone().cpu()
                event = None
                if event_metadata is not None:
                    metadata = event_metadata.get(int(key.chunk_hash))
                    if metadata is not None:
                        parent, token_ids, block_size, lora_id = metadata
                        event = {
                            "parent_block_hash": parent,
                            "token_ids": list(token_ids),
                            "block_size": int(block_size or 0),
                            "lora_id": lora_id,
                        }
                fmt = memory_obj.meta.fmt
                self._snapshot_digest_update(
                    digest,
                    key,
                    tensor_copy,
                    fmt,
                    state["pin_count"],
                    state["is_prefetch"],
                    access_count,
                )
                entries.append(
                    {
                        "key": key.to_string(),
                        "tensor": tensor_copy,
                        "fmt": int(fmt.value),
                        "pin_count": state["pin_count"],
                        "is_prefetch": state["is_prefetch"],
                        "prefetch_metadata": state["prefetch_metadata"],
                        "access_count": access_count,
                        "event": event,
                    }
                )
                total_tokens += int(memory_obj.get_num_tokens())
                total_bytes += int(tensor_copy.numel() * tensor_copy.element_size())
        finally:
            for _, memory_obj, _, _ in held:
                memory_obj.ref_count_down()

        payload = {
            "version": 1,
            "backend": "LocalCPUBackend",
            "entries": entries,
            "skipped": skipped,
            "digest": digest.hexdigest(),
        }
        temporary = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, snapshot_path)
        if skipped:
            logger.warning(
                "LocalCPU snapshot skipped %d objects without tensor data: %s",
                len(skipped),
                skipped[:4],
            )
        return {
            "path": str(snapshot_path),
            "entries": len(entries),
            "skipped": len(skipped),
            "tokens": total_tokens,
            "bytes": total_bytes,
            "digest": digest.hexdigest(),
        }

    def restore(self, path: str, clear_existing: bool = True) -> dict[str, Any]:
        """Restore allocator-owned LocalCPU objects from :meth:`snapshot`."""
        if clear_existing:
            self.clear()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch versions lack ``weights_only``
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"Unsupported LocalCPU snapshot: {path}")

        entries = payload.get("entries", [])
        digest = hashlib.sha256()
        restored: list[dict[str, Any]] = []
        total_tokens = 0
        total_bytes = 0
        missing_event_metadata = 0
        policy_access = getattr(self.cache_policy, "key_to_access_count", None)

        for entry in entries:
            key = parse_cache_key(entry["key"])
            tensor = entry["tensor"]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"LocalCPU snapshot entry has no tensor: {entry['key']}")
            fmt = MemoryFormat(int(entry["fmt"]))
            memory_obj = self.allocate(
                tensor.shape,
                tensor.dtype,
                fmt=fmt,
                eviction=False,
                busy_loop=False,
            )
            if memory_obj is None or memory_obj.tensor is None:
                raise RuntimeError(
                    "LocalCPU restore ran out of destination capacity at "
                    f"{entry['key']}"
                )
            memory_obj.tensor.copy_(tensor, non_blocking=False)
            inserted = key not in self.hot_cache
            if inserted:
                self.submit_put_task(key, memory_obj)
            memory_obj.ref_count_down()
            if not inserted:
                continue

            with self.cpu_lock:
                if entry.get("is_prefetch", False):
                    self.prefetch_keys.add(key)
                    self.prefetch_metadata[key] = dict(
                        entry.get("prefetch_metadata") or {}
                    )
                if isinstance(policy_access, dict):
                    policy_access[key] = int(entry.get("access_count", 0))
                restored_obj = self.hot_cache[key]
                pin_count = int(entry.get("pin_count", 0))
                for _ in range(max(0, pin_count)):
                    restored_obj.pin()

            self._snapshot_digest_update(
                digest,
                key,
                tensor,
                fmt,
                int(entry.get("pin_count", 0)),
                bool(entry.get("is_prefetch", False)),
                int(entry.get("access_count", 0)),
            )
            event = entry.get("event")
            if event is None:
                missing_event_metadata += 1
            restored.append(
                {
                    "key": entry["key"],
                    "chunk_hash": int(key.chunk_hash),
                    "event": event,
                }
            )
            total_tokens += int(tensor.shape[fmt.token_dim()])
            total_bytes += int(tensor.numel() * tensor.element_size())

        actual_digest = digest.hexdigest()
        expected_digest = str(payload.get("digest", ""))
        if expected_digest and actual_digest != expected_digest:
            raise RuntimeError(
                f"LocalCPU restore digest mismatch: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        return {
            "path": str(path),
            "entries": len(restored),
            "tokens": total_tokens,
            "bytes": total_bytes,
            "digest": actual_digest,
            "missing_event_metadata": missing_event_metadata,
            "restored": restored,
        }

    def clear(self) -> int:
        """
        counts the number of memory objects removed
        """
        if not self.use_hot:
            return 0
        clear_keys = []
        num_cleared_tokens = 0
        with self.cpu_lock:
            for key in self.hot_cache:
                memory_obj = self.hot_cache[key]
                if not memory_obj.can_evict:
                    continue
                clear_keys.append(key)
                num_cleared_tokens += memory_obj.get_num_tokens()

        # TODO(Jiayi): might not be accurate if we don't calculate
        # `num_cleared_token` and remove the keys in an atomic way.
        self.batched_remove(clear_keys)

        return num_cleared_tokens

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self.memory_allocator.close()
        self.clear()
