"""RAM lookup correctness, cache ownership, padding and bounded decoding."""

import numpy as np
import pytest
from qwen38_tpu.compute import pack_q8
from qwen38_tpu.ngram import HostNGramTable


def table(rows=32, width=160, cache_rows=16):
    rng = np.random.default_rng(318)
    raw = pack_q8(rng.normal(size=(rows, width)).astype(np.float32))
    # Independent Q8_0 oracle, including the stored FP16 scales.
    scale = np.ascontiguousarray(raw[..., :2]).view("<f2").reshape(rows, -1)
    codes = np.ascontiguousarray(raw[..., 2:]).view(np.int8)
    expected = (codes.astype(np.float32) * scale[..., None]).reshape(rows, width)
    return HostNGramTable(raw, "Q8_0", width, cache_rows), expected


def test_duplicate_misses_are_decoded_once_with_exact_lru_order():
    lookup, expected = table(cache_rows=2)
    ids = np.array([[3, 1, 3, 2, 3, 1]])
    actual = lookup.gather(ids)
    np.testing.assert_array_equal(actual, expected[ids].reshape(1, -1))
    assert list(lookup.cache) == [3, 1]
    assert (lookup.hits, lookup.misses, lookup.decoded_rows) == (2, 4, 3)
    assert lookup.looked_up_bytes == 3 * lookup.packed[0].nbytes
    assert all(row.base is None for row in lookup.cache.values())
    actual[:] = 0
    np.testing.assert_array_equal(
        lookup.gather(np.array([[1, 3]])), expected[[1, 3]].reshape(1, -1)
    )
    assert lookup.decoded_rows == 3


def test_batched_prefill_and_inactive_rows_never_pollute_cache():
    lookup, expected = table()
    ids = np.array([[[0, 1], [3, 4], [7, 8]], [[3, 5], [8, 9], [10, 11]]])
    active = np.array([[True, True, False], [True, False, False]])
    actual = lookup.gather(ids, active=active)
    oracle = expected[ids].reshape(2, 3, -1)
    oracle[~active] = 0
    np.testing.assert_array_equal(actual, oracle)
    assert lookup.decoded_rows == 5
    assert list(lookup.cache) == [0, 1, 4, 3, 5]
    assert lookup.hits == 1 and lookup.misses == 5
    before = lookup.looked_up_bytes
    np.testing.assert_array_equal(lookup.gather(ids, active=np.zeros((2, 3), bool)), 0)
    assert lookup.looked_up_bytes == before


def test_no_cache_empty_shapes_and_invalid_inputs():
    lookup, expected = table(cache_rows=0)
    np.testing.assert_array_equal(lookup.gather([1, 1, 3]), expected[[1, 1, 3]].reshape(-1))
    assert not lookup.cache and lookup.misses == 3 and lookup.decoded_rows == 2
    assert lookup.gather(np.empty((0, 16), int)).shape == (0, 2560)
    assert lookup.gather(np.empty((2, 0), int)).shape == (2, 0)
    with pytest.raises(ValueError, match="head axis"):
        lookup.gather(1)
    with pytest.raises(ValueError, match="mask"):
        lookup.gather([[1, 2]], active=[True, False])
    with pytest.raises(ValueError, match="outside"):
        lookup.gather([[32]])
    with pytest.raises(ValueError, match="capacity"):
        table(cache_rows=-1)


def test_decoder_temporary_size_is_bounded(monkeypatch):
    from qwen38_tpu import ngram

    lookup, expected = table(rows=514, width=2560, cache_rows=0)
    original, calls = ngram.dequantize_rows, []

    def decode(raw, *args, **kwargs):
        calls.append(len(raw))
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(ngram, "dequantize_rows", decode)
    actual = lookup.gather(np.arange(514)[:, None])
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 2 and max(calls) * 2560 * 4 <= 4 << 20
