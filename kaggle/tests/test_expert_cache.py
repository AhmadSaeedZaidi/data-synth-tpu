"""Exercise actual sharded bytes through cache hits, misses and evictions."""

import jax
import numpy as np
import pytest
from glm53_offload.expert_cache import ExpertCache
from glm53_offload.weights import PackedExperts
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def test_startup_gate_compares_direct_and_compact_math_on_all_cpu_shards():
    from glm53_offload.run import expert_cache_gate

    result = expert_cache_gate()
    assert result["expert_cache"] == "PASS"
    assert result["expert_cache_max_error"] == 0


def test_large_working_set_with_finer_hit_and_miss_buckets_is_lossless():
    from glm53.engine import AXIS

    raw = np.arange(8 * 288 * 2 * 4 * 4, dtype=np.uint8).reshape(8, 288, 2, 4, 4)
    store = PackedExperts({0: (raw,) * 3}, {0: ("F32",) * 3}, 8)
    mesh = Mesh(np.array(jax.devices()), (AXIS,))

    def transport(arrays):
        return tuple(
            jax.device_put(a.reshape(*a.shape[:3], -1), NamedSharding(mesh, P(AXIS)))
            for a in arrays
        )

    per_expert = sum(a[:, :1].nbytes for a in store.tables[0])
    cache = ExpertCache(store, mesh, transport, 48 * per_expert)
    for selected in (np.arange(65), np.arange(32), np.arange(32), np.arange(32, 97)):
        routes = selected.astype(np.int32)[None]
        packed, local, ep = cache.gather(0, routes)
        assert ep == (80 if len(selected) == 65 else 32)
        for array in packed:
            actual = np.asarray(array).reshape(8, ep, 2, 4, 4)
            np.testing.assert_array_equal(actual[:, local], raw[:, routes])
            np.testing.assert_array_equal(actual[:, len(selected) :], 0)
    assert cache.stats["hits"] > 0
    assert cache.stats["evictions"] > 0


@pytest.mark.parametrize("capacity", [0, 1, 3, 8])
@pytest.mark.parametrize("direct", [False, True])
def test_lossless_cache_with_working_sets_larger_than_capacity(capacity, direct):
    n = len(jax.devices())
    rng = np.random.default_rng(752)
    raw = rng.integers(0, 256, (n, 8, 2, 3, 4), np.uint8)
    tables = (raw, raw ^ np.uint8(75), raw ^ np.uint8(189))
    store = PackedExperts({0: tables}, {0: ("F32",) * 3}, n)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    sent = []

    def transport(arrays):
        sent.append(sum(a.nbytes for a in arrays))
        return tuple(jax.device_put(a.reshape(*a.shape[:3], -1), sharding) for a in arrays)

    per_expert = sum(a[:, :1].nbytes for a in tables)
    cache = ExpertCache(store, mesh, transport, capacity * per_expert)
    for ids in ([0, 1], [0, 1], [1, 4, 6, 7], [6, 7], [2, 3], [0, 1, 2, 3, 4, 5, 6, 7]):
        routes = np.array([ids, ids[::-1]], np.int32)
        packed, inverse, ep = cache.gather(0, routes, direct=direct)
        if len(packed) == 7:
            selectors = packed[6]
            restored = []
            for resident, fresh in zip(packed[:3], packed[3:6], strict=True):
                resident, fresh = np.asarray(resident), np.asarray(fresh)
                values = np.where(
                    (selectors[:, 0] >= 0)[None, :, None, None],
                    resident[:, selectors[:, 0].clip(0)],
                    fresh[:, selectors[:, 1].clip(0)],
                )
                valid = np.any(selectors >= 0, axis=1)
                restored.append(np.where(valid[None, :, None, None], values, 0))
            packed = restored
        for got, expected in zip(packed, tables, strict=True):
            got = np.asarray(got).reshape(n, ep, *expected.shape[2:])
            for chip in range(n):
                np.testing.assert_array_equal(got[chip][inverse], expected[chip][routes])
            np.testing.assert_array_equal(got[:, len(set(ids)) : ep], 0)
        assert cache.stats["allocated_bytes"] <= capacity * per_expert
    assert sum(sent) == store.stats["bytes_transferred"]
    assert cache.stats["hits"] > 0 if capacity else cache.stats["hits"] == 0
    assert (
        cache.stats["compact_table_bytes_avoided"] > 0
        if direct and capacity
        else (cache.stats["compact_table_bytes_avoided"] == 0)
    )


def test_direct_hits_return_original_resident_buffers_without_a_compact_gather():
    n = len(jax.devices())
    raw = np.arange(n * 8 * 4, dtype=np.uint8).reshape(n, 8, 1, 1, 4)
    store = PackedExperts({0: (raw,) * 3}, {0: ("F32",) * 3}, n)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    cache = ExpertCache(
        store,
        mesh,
        lambda arrays: tuple(
            jax.device_put(a.reshape(*a.shape[:3], -1), NamedSharding(mesh, P("x"))) for a in arrays
        ),
        store.nbytes,
    )
    routes = np.array([[0, 3, 0]], np.int32)
    cache.gather(0, routes, direct=True)
    programs, transferred = len(cache.programs), store.stats["bytes_transferred"]
    packed, _, _ = cache.gather(0, routes, direct=True)
    assert len(cache.programs) == programs
    assert store.stats["bytes_transferred"] == transferred
    assert all(a is b for a, b in zip(packed[:3], cache.buffers[0], strict=True))


def test_cache_hit_transfers_no_expert_payload_and_layers_are_isolated():
    n = len(jax.devices())
    raw = np.arange(n * 8 * 4, dtype=np.uint8).reshape(n, 8, 1, 1, 4)
    store = PackedExperts({0: (raw,) * 3, 1: (raw + 3,) * 3}, {0: ("F32",) * 3, 1: ("F32",) * 3}, n)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    cache = ExpertCache(
        store,
        mesh,
        lambda arrays: tuple(
            jax.device_put(a.reshape(*a.shape[:3], -1), NamedSharding(mesh, P("x"))) for a in arrays
        ),
        store.nbytes,
    )
    routes = np.array([[2, 2, 3]], np.int32)
    for layer in (0, 1):
        first = cache.gather(layer, routes)
        jax.block_until_ready(first)
        before = store.stats["bytes_transferred"]
        second = cache.gather(layer, routes)
        jax.block_until_ready(second)
        assert store.stats["bytes_transferred"] == before
        for a, b in zip(first[0], second[0], strict=True):
            np.testing.assert_array_equal(a, b)
    assert cache.stats["hits"] == 4
