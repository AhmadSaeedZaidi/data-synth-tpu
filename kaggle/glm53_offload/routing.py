"""Small host routing plans and bounded packed-expert shape buckets."""

import numpy as np


def expert_bucket(count, total, tile):
    if count < 1 or count > total or tile < 1:
        raise ValueError("expert count must be in [1, total] and tile must be positive")
    # Keep small decode shapes, then use 16-expert steps instead of doubling
    # large transfers (e.g. 65 misses now send 80 experts, rather than 128).
    bucket = 1 << (int(count) - 1).bit_length() if count <= 16 else ((count + 15) // 16) * 16
    return ((min(total, max(tile, bucket)) + tile - 1) // tile) * tile


def expert_token_rows(local_ids, ep):
    """Return [Ep, capacity] token rows, with -1 marking inactive slots.

    Each expert sees each routed token once, even if a token's top-k list
    repeats that expert. Device code sums those route weights. Capacity uses
    a power-of-two occupancy bucket, bounded by the full token count. Building
    this plan reuses router IDs already fetched for host expert staging.
    """
    ids = np.asarray(local_ids)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer) or not ids.size:
        raise ValueError("local expert IDs must be a nonempty integer [tokens, topk] array")
    if ep < 1 or np.any(ids < -1) or np.any(ids >= ep):
        raise ValueError("local expert IDs must be -1 or within Ep")
    assigned = np.zeros((ep, ids.shape[0]), dtype=np.bool_)
    rows, routes = np.nonzero(ids >= 0)
    assigned[ids[rows, routes], rows] = True
    occupancy = assigned.sum(axis=1)
    maximum = int(occupancy.max())
    if not maximum:
        raise ValueError("a sparse layer produced no routed experts")
    capacity = min(ids.shape[0], 1 << (maximum - 1).bit_length())
    result = np.full((ep, capacity), -1, dtype=np.int32)
    for expert in np.flatnonzero(occupancy):
        selected = np.flatnonzero(assigned[expert])
        result[expert, : len(selected)] = selected
    return result


def expert_tile_capacities(rows, tile):
    """Limit each expert tile to its own busiest expert, rather than the layer's.

    Zero means the tile has no routes and can skip decoding/matmul entirely.
    Shapes stay unchanged across calls: device code selects a bounded set of
    static matmul branches using these small integer capacities.
    """
    rows = np.asarray(rows)
    if (
        rows.ndim != 2
        or not np.issubdtype(rows.dtype, np.integer)
        or not rows.size
        or tile < 1
        or rows.shape[0] % tile
        or np.any(rows < -1)
    ):
        raise ValueError("token rows must contain whole nonempty expert tiles")
    maxima = np.count_nonzero(rows >= 0, axis=1).reshape(-1, tile).max(axis=1)
    return np.array(
        [min(rows.shape[1], 1 << (int(n) - 1).bit_length()) if n else 0 for n in maxima],
        dtype=np.int32,
    )
