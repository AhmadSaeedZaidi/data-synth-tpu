"""Exact host-side hashing and bounded row gathers. Never dequantize the full table."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .quant import block_layout, dequantize_rows

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


def splitmix(value):
    value = (value + GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def is_prime(n):
    if n < 2 or (n % 2 == 0 and n != 2):
        return False
    return all(n % d for d in range(3, int(n**0.5) + 1, 2))


@dataclass
class NGramHash:
    config: object

    def __post_init__(self):
        c = self.config
        bound = max(1, (((1 << 63) - 1) // c.vocab) // 2)
        self.multipliers = [
            2 * (splitmix((c.ngram_seed + GAMMA * (i + 1)) & MASK64) % bound) + 1
            for i in range(c.ngram_size)
        ]
        primes, candidate = [], c.ngram_base
        for _ in range((c.ngram_size - 1) * c.heads_per_ngram):
            while not is_prime(candidate):
                candidate += 1
            primes.append(candidate)
            candidate += 1
        self.sizes = np.asarray(primes, dtype=np.int64)
        self.offsets = np.cumsum([0] + primes[:-1], dtype=np.int64)
        self.rows = int(sum(primes) + 127) // 128 * 128

    def initial_history(self, batch):
        return np.full((batch, self.config.ngram_size - 1), self.config.eos, dtype=np.int64)

    def step(self, ids, history, active=None):
        c = self.config
        ids, history = np.asarray(ids, np.int64), np.asarray(history, np.int64)
        if ids.ndim != 1 or history.shape != (ids.size, c.ngram_size - 1):
            raise ValueError("N-gram history shape differs from batch")
        if np.any(ids < 0) or np.any(ids >= c.vocab):
            raise ValueError("Token outside vocabulary")
        active = np.ones(ids.size, bool) if active is None else np.asarray(active, bool)
        if active.shape != ids.shape:
            raise ValueError("N-gram active mask shape differs from batch")
        shifted = [ids]
        for shift in range(1, c.ngram_size):
            value = history[:, -shift].copy()
            # A prior EOS resets all positions before it, but the current EOS
            # itself can still read the preceding segment (HF semantics).
            crossed = np.any(history[:, -shift:] == c.eos, axis=1)
            shifted.append(np.where(crossed, c.eos, value))
        blocks = []
        mixed = shifted[0] * self.multipliers[0]
        for n in range(2, c.ngram_size + 1):
            mixed = mixed ^ (shifted[n - 1] * self.multipliers[n - 1])
            sl = slice((n - 2) * c.heads_per_ngram, (n - 1) * c.heads_per_ngram)
            blocks.append(mixed[:, None] % self.sizes[sl] + self.offsets[sl])
        updated = np.concatenate((history[:, 1:], ids[:, None]), axis=1)
        return np.concatenate(blocks, axis=1), np.where(active[:, None], updated, history)


class HostNGramTable:
    """Packed RAM array or explicitly selected SSD memmap with an LRU row cache."""

    def __init__(self, packed, qtype, row_width, cache_rows=16384):
        if type(cache_rows) is not int or cache_rows < 0:
            raise ValueError("Row cache capacity must be a nonnegative integer")
        if type(row_width) is not int or row_width < 1:
            raise ValueError("Row width must be a positive integer")
        self.packed = packed
        self.qtype, self.row_width = qtype, row_width
        self.cache_rows = cache_rows
        self.cache = OrderedDict()
        values, size = block_layout(qtype)
        if packed.dtype != np.uint8 or packed.shape[1:] != (
            (row_width + values - 1) // values,
            size,
        ):
            raise ValueError("Invalid n-gram table storage")
        self.hits = self.misses = self.looked_up_bytes = 0
        self.decoded_rows = 0

    def gather(self, indices, active=None):
        """Coalesce misses and decode in bounded NumPy batches, preserving LRU order.

        An inactive request returns zeros without touching the RAM table/cache.
        Cache entries own just one row: retaining a view of a decoded batch
        would otherwise keep entire batches alive after most rows are evicted.
        """
        indices = np.asarray(indices, np.int64)
        if indices.ndim < 1:
            raise ValueError("Row indices require a head axis")
        shape = indices.shape[:-1] + (indices.shape[-1] * self.row_width,)
        active = np.ones(indices.shape[:-1], bool) if active is None else np.asarray(active, bool)
        if active.shape != indices.shape[:-1]:
            raise ValueError("Row lookup mask shape differs from requests")
        if np.any(indices < 0) or np.any(indices >= len(self.packed)):
            raise ValueError("N-gram row outside table")
        flat = indices[active].reshape(-1)
        if not flat.size:
            return np.zeros(shape, np.float32)
        unique, inverse = np.unique(flat, return_inverse=True)
        decoded = np.empty((len(unique), self.row_width), np.float32)
        missing = []
        for index, row_id in enumerate(unique):
            row = self.cache.get(int(row_id))
            if row is None:
                missing.append(index)
            else:
                decoded[index] = row
        # At most 4 MiB of expanded rows per decoder call, independent of the
        # host table's 50+ GiB size or a caller's prompt length.
        chunk_rows = max(1, (4 << 20) // (4 * self.row_width))
        for first in range(0, len(missing), chunk_rows):
            slots = missing[first : first + chunk_rows]
            raw = self.packed[unique[slots]]
            decoded[slots] = dequantize_rows(raw, self.qtype, self.row_width)
            self.looked_up_bytes += raw.nbytes
            self.decoded_rows += len(slots)
        # Count logical cache hits/misses in request order, even when a miss
        # was coalesced with another lookup in this call.
        for key, index in zip(flat.tolist(), inverse.tolist(), strict=True):
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
            else:
                self.misses += 1
                if self.cache_rows:
                    self.cache[key] = decoded[index].copy()
                    if len(self.cache) > self.cache_rows:
                        self.cache.popitem(last=False)
        gathered = decoded[inverse]
        if np.all(active):
            return gathered.reshape(shape)
        output = np.zeros(indices.shape + (self.row_width,), np.float32)
        output[active] = gathered.reshape(*indices[active].shape, self.row_width)
        return output.reshape(shape)
