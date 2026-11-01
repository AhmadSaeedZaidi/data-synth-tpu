import numpy as np
import pytest
from glm53_offload.routing import expert_bucket, expert_tile_capacities, expert_token_rows
from glm53_offload.weights import PackedExperts


def test_transfer_buckets_reduce_payload_and_keep_routes_and_padding_exact():
    raw = np.arange(8 * 288 * 2 * 4 * 4, dtype=np.uint8).reshape(8, 288, 2, 4, 4)
    store = PackedExperts({0: (raw,) * 3}, {0: ("F32",) * 3}, 8)
    routes = np.arange(65, dtype=np.int32)[None]
    packed, ids, ep = store.gather(0, routes)
    assert ep == 80  # Old power-of-two payload had 128 expert rows.
    for array in packed:
        np.testing.assert_array_equal(array[:, ids], raw[:, routes])
        np.testing.assert_array_equal(array[:, 65:], 0)
    stats = store.stats
    assert stats["padding_bytes_transferred"] == sum(a[:, 65:].nbytes for a in packed)
    assert (
        stats["useful_bytes_transferred"] + stats["padding_bytes_transferred"]
        == stats["bytes_transferred"]
    )
    for tile in (4, 8):
        shapes = {expert_bucket(n, 288, tile) for n in range(1, 289)}
        assert len(shapes) <= 20
        for n in range(1, 289):
            ep = expert_bucket(n, 288, tile)
            old = ((min(288, max(tile, 1 << (n - 1).bit_length())) + tile - 1) // tile) * tile
            assert n <= ep <= old
            assert ep % tile == 0


def test_routing_plan_deduplicates_token_rows_and_excludes_padded_tokens():
    ids = np.array([[0, 0, 1], [1, 2, 3], [0, 3, 4], [-1, -1, -1]], np.int32)
    rows = expert_token_rows(ids, 8)
    assert rows.shape == (8, 2)
    np.testing.assert_array_equal(rows[0], [0, 2])
    assert not np.any(rows == 3)
    np.testing.assert_array_equal(rows[5:], -1)
    assert np.count_nonzero(rows >= 0) == 8


@pytest.mark.parametrize("ids,ep", [([], 4), ([[4]], 4), ([[-2]], 4), ([[0.0]], 4), ([[-1]], 4)])
def test_invalid_routing_plan_rejected(ids, ep):
    with pytest.raises(ValueError):
        expert_token_rows(ids, ep)


def test_hot_expert_only_expands_its_own_tile_and_empty_tiles_do_no_math():
    ids = np.stack((np.zeros(32, np.int32), 1 + np.arange(32) % 15), axis=1)
    rows = expert_token_rows(ids, 20)
    capacities = expert_tile_capacities(rows, 4)
    np.testing.assert_array_equal(capacities, [32, 2, 2, 2, 0])
    assert int(capacities.sum()) * 4 == 152
    assert rows.size == 640
    assert np.count_nonzero(rows >= 0) == 64


def test_tile_capacity_preserves_non_power_of_two_token_limits():
    rows = expert_token_rows(np.array([[0], [0], [0]], np.int32), 4)
    np.testing.assert_array_equal(expert_tile_capacities(rows, 4), [3])
