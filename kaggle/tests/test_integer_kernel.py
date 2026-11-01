"""Check GGUF affine contraction independently and preserve INT4 lowering."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from glm53_offload.integer_kernel import (
    dot_probe,
    grouped_q4_matmul,
    q4_groups,
    quantize_groups,
    split_activation_nibbles,
)


def fixture():
    path = Path(__file__).parents[1] / "glm53_offload/data/decoder_reference.npz"
    with np.load(path, allow_pickle=False) as reference:
        # Frozen gguf-py outputs, rather than another use of our decoder.
        return reference["Q4_K_raw"].copy(), reference["Q4_K_expected"].copy()


def test_affine_recentring_preserves_unsigned_gguf_codes_and_min_offsets():
    raw, expected = fixture()
    codes, scale, offset = jax.jit(q4_groups)(raw)
    restored = (codes.astype(jnp.float32) * scale[..., None] + offset[..., None]).reshape(
        expected.shape
    )
    assert np.min(codes) == -8 and np.max(codes) == 7
    np.testing.assert_allclose(restored, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("bits", [4, 8])
def test_integer_matmul_matches_independent_weight_times_quantized_activation(bits):
    raw, expected = fixture()
    rng = np.random.default_rng(59)
    x = rng.normal(size=(raw.shape[0], 5, expected.shape[-1])).astype(np.float32)
    x[0, 0] = 0
    x[1, 2, 33] = 70  # outlier calibration is local to its 32-value group
    codes, scales = quantize_groups(jnp.asarray(x), bits)
    reconstructed = np.asarray(codes, dtype=np.float32) * np.asarray(scales)[..., None]
    reconstructed = reconstructed.reshape(x.shape)
    want = np.einsum("erk,eok->ero", reconstructed, expected)
    got = jax.jit(lambda a, b: grouped_q4_matmul(a, b, activation_bits=bits))(x, raw)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-3)


def test_two_signed_nibbles_preserve_all_int8_values_without_activation_rounding():
    codes = jnp.arange(-128, 128, dtype=jnp.int16).astype(jnp.int8)
    high, low = jax.jit(split_activation_nibbles)(codes)
    assert int(high.min()) == int(low.min()) == -8
    assert int(high.max()) == int(low.max()) == 7
    restored = 16 * np.asarray(high, dtype=np.int16) + np.asarray(low, dtype=np.int16) + 8
    np.testing.assert_array_equal(restored, np.arange(-128, 128))
    w = np.random.default_rng(84).integers(-8, 8, (256, 17), dtype=np.int32)
    split_dot = 16 * np.asarray(high, np.int32) @ w + np.asarray(low, np.int32) @ w + 8 * w.sum(0)
    np.testing.assert_array_equal(split_dot, np.arange(-128, 128, dtype=np.int32) @ w)


@pytest.mark.parametrize(("bits", "dots"), [(4, 1), (8, 2)])
def test_native_int4_hlo_has_two_int4_operands_and_int32_accumulation(bits, dots):
    raw, _ = fixture()
    x = jnp.zeros((raw.shape[0], 4, raw.shape[-2] * 256), jnp.float32)
    lowered = jax.jit(
        lambda a, b: grouped_q4_matmul(a, b, activation_bits=bits, native_int4=True)
    ).lower(x, raw)
    lines = [
        line for line in str(lowered.compiler_ir()).splitlines() if "stablehlo.dot_general" in line
    ]
    assert len(lines) == dots
    assert all(line.count("xi4>") == 2 and "xi32>" in line for line in lines)


def test_small_cpu_probe_is_honest_about_native_int4_support():
    result = dot_probe(rows=(1,), repeats=1)
    assert result["status"] == "COMPLETE"
    assert {item["dtype"]: item["status"] for item in result["cases"]} == {
        "bf16": "PASS",
        "int8": "PASS",
        "int4": "SKIPPED",
    }
    assert "xi4>" in result["cases"][-1]["stablehlo_dot"][0]


def test_real_format_projection_profile_does_not_claim_cpu_int4_or_model_quality():
    from glm53_offload.profiling import integer_projection_profile
    from glm53_offload.weights import PackedExperts
    from jax.sharding import Mesh

    raw, _ = fixture()
    chip_major = np.broadcast_to(raw[None], (8, *raw.shape)).copy()
    store = PackedExperts({0: (chip_major,) * 3}, {0: ("Q4_K",) * 3}, 8)
    result = integer_projection_profile(store, Mesh(np.array(jax.devices()), ("x",)), lambda: None)
    assert result["status"] == "COMPLETE" and len(result["cases"]) == 8
    for item in result["cases"]:
        assert item["status"] == ("SKIPPED" if "int4" in item["mode"] else "MEASURED")
        if item["status"] == "MEASURED":
            assert item["quality_gate"] == "NOT_RUN"


def test_rejects_incompatible_block_and_activation_shapes():
    with pytest.raises(ValueError, match="144 bytes"):
        q4_groups(jnp.zeros((2, 143), jnp.uint8))
    with pytest.raises(ValueError, match="complete 256"):
        grouped_q4_matmul(jnp.zeros((1, 2, 32)), jnp.zeros((1, 4, 1, 144), jnp.uint8))
