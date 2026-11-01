"""Lossless host repacking of Q6_K blocks into contiguous TPU input tiles.

GGUF input is [output, input_blocks, 210] uint8. A 64-output tile stores
the original low bits, high bits, signed scales and FP16 scale bytes in
separate contiguous regions. No quantization or floating arithmetic occurs.
Flattening tiles avoids exposing a tiny block/scale axis to HBM padding.
"""

from dataclasses import dataclass

import numpy as np

QK = 256
BLOCK_BYTES = 210
OUTPUT_TILE = 64
TILE_BYTES = OUTPUT_TILE * 208
SCALE_TILE_BYTES = OUTPUT_TILE * 2
SCALE_FETCH_BYTES = 512


@dataclass(frozen=True)
class PackedQ6:
    data: np.ndarray
    scales: np.ndarray
    input_features: int
    output_features: int

    @property
    def input_blocks(self):
        return (self.input_features + QK - 1) // QK

    @property
    def output_tiles(self):
        return (self.output_features + OUTPUT_TILE - 1) // OUTPUT_TILE

    def storage(self):
        original = self.output_features * self.input_blocks * BLOCK_BYTES
        size = self.data.nbytes + self.scales.nbytes
        return dict(
            original_bytes=original,
            packed_bytes=size,
            padding_bytes=size - original,
            padding_ratio=size / original - 1,
            layout="q6_k_soa_n64_k256_v1",
            physical_hbm_bytes="requires target allocator measurement",
        )


def pack_q6(raw, input_features):
    """Repack one projection without expanding to integer or BF16 weights.

    Output dimensions divisible by 64 (including 320 and 640) require no
    weight padding. The separate FP16 scale-byte plane pads by at most 384
    bytes to legal 512-byte DMA windows. A partial K block remains the
    original GGUF block; its unused values are masked by the projection's
    zero-padded input.
    """
    if type(input_features) is not int or input_features < 1:
        raise ValueError("input_features must be a positive integer")
    raw = np.asarray(raw)
    if raw.dtype != np.uint8:
        raise TypeError("GGUF Q6_K input must have dtype uint8")
    blocks = (input_features + QK - 1) // QK
    if raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[1:] != (blocks, BLOCK_BYTES):
        raise ValueError("Expected [output_features, ceil(input_features / 256), 210]")
    # Reject corrupted FP16 block scales, but preserve every finite bit pattern.
    scales = np.ascontiguousarray(raw[..., 208:210]).view("<f2")
    if not np.isfinite(scales).all():
        raise ValueError("Q6_K block scales must be finite")
    outputs = raw.shape[0]
    tiles = (outputs + OUTPUT_TILE - 1) // OUTPUT_TILE
    data = np.empty((tiles, blocks, TILE_BYTES), np.uint8)
    scale_bytes = tiles * blocks * SCALE_TILE_BYTES
    scales = np.zeros(
        ((scale_bytes + SCALE_FETCH_BYTES - 1) // SCALE_FETCH_BYTES * SCALE_FETCH_BYTES,), np.uint8
    )
    scale_view = scales[:scale_bytes].reshape(tiles, blocks, SCALE_TILE_BYTES)
    # Bound temporary packing memory to one 64-row tile for this projection.
    for tile in range(tiles):
        rows = raw[tile * OUTPUT_TILE : (tile + 1) * OUTPUT_TILE]
        if len(rows) != OUTPUT_TILE:
            rows = np.pad(rows, ((0, OUTPUT_TILE - len(rows)), (0, 0), (0, 0)))
        rows = rows.transpose(1, 0, 2)
        offset = 0
        for first, last in ((0, 128), (128, 192), (192, 208)):
            size = OUTPUT_TILE * (last - first)
            data[tile, :, offset : offset + size] = rows[:, :, first:last].reshape(blocks, size)
            offset += size
        scale_view[tile] = rows[:, :, 208:210].reshape(blocks, SCALE_TILE_BYTES)
    return PackedQ6(data.reshape(-1), scales, input_features, outputs)


def unpack_q6(packed):
    """Inverse byte transform, used for round-trip validation and portability."""
    tiles = np.asarray(packed.data).reshape(packed.output_tiles, packed.input_blocks, TILE_BYTES)
    raw = np.empty((packed.output_tiles, packed.input_blocks, OUTPUT_TILE, BLOCK_BYTES), np.uint8)
    offset = 0
    for first, last in ((0, 128), (128, 192), (192, 208)):
        size = OUTPUT_TILE * (last - first)
        raw[..., first:last] = tiles[..., offset : offset + size].reshape(
            packed.output_tiles, packed.input_blocks, OUTPUT_TILE, last - first
        )
        offset += size
    raw[..., 208:210] = packed.scales[
        : packed.output_tiles * packed.input_blocks * SCALE_TILE_BYTES
    ].reshape(packed.output_tiles, packed.input_blocks, OUTPUT_TILE, 2)
    return raw.transpose(0, 2, 1, 3).reshape(-1, packed.input_blocks, BLOCK_BYTES)[
        : packed.output_features
    ]
