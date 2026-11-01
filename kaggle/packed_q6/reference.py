"""Independent scalar GGML Q6_K semantics and bounded synthetic fixtures.

This oracle directly indexes the original byte stream, without the Pallas
tile layout or the application's NumPy/JAX decoder. The field convention
is documented by ggml-org/llama.cpp, ggml/src/ggml-quants.c.
"""

import numpy as np


def decode_reference(raw, input_features):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 210:
        raise ValueError("Expected uint8 [N, K_blocks, 210]")
    if input_features < 1 or raw.shape[1] != (input_features + 255) // 256:
        raise ValueError("Input width does not match Q6 block count")
    out = np.empty((raw.shape[0], raw.shape[1] * 256), np.float32)
    # Vectorize rows/blocks, but index all 256 code positions independently.
    scale = np.ascontiguousarray(raw[:, :, 208:210]).view("<f2").reshape(raw.shape[:2])
    scale = scale.astype(np.float32)
    for position in range(256):
        half, lane = position // 128, position % 32
        quarter = (position % 128) // 32
        lo_byte = raw[:, :, half * 64 + (quarter % 2) * 32 + lane].astype(np.int32)
        lo = (lo_byte >> (4 * (quarter // 2))) & 15
        hi = (raw[:, :, 128 + half * 32 + lane].astype(np.int32) >> (2 * quarter)) & 3
        code = (lo | (hi << 4)) - 32
        sub_scale = raw[:, :, 192 + position // 16].astype(np.int32)
        sub_scale = np.where(sub_scale >= 128, sub_scale - 256, sub_scale)
        value = (scale * sub_scale.astype(np.float32)) * code.astype(np.float32)
        out[:, position::256] = value
    return out[:, :input_features]


def bf16_round(x):
    """Independent NumPy round-to-nearest-even boundary, returned as FP32."""
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    return np.where(nan, bits | np.uint32(0x00400000), rounded).view(np.float32)


def projection_reference(x, raw):
    x = bf16_round(x)
    weights = bf16_round(decode_reference(raw, x.shape[1]))
    # FP64 accumulation isolates operand/decoder mistakes from reduction order.
    return (x.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float32)


def fixture(outputs, inputs, seed=3826):
    """Legal Q6 bytes, signed/extreme subgroup scales and finite block scales."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (outputs, (inputs + 255) // 256, 210), dtype=np.uint8)
    scales = rng.choice(np.array([-0.003, -0.001, 0, 0.001, 0.003], dtype="<f2"), raw.shape[:2])
    raw[:, :, 208:210] = scales.view(np.uint8).reshape(*raw.shape[:2], 2)
    return raw
