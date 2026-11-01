"""Independent byte/value oracles, ragged boundaries and TPU-target lowering."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax import export
from packed_q6 import pack_q6, unpack_q6
from packed_q6.kernel import decode_fp16_scale, decode_tile, q6_matmul
from packed_q6.layout import SCALE_FETCH_BYTES, SCALE_TILE_BYTES, TILE_BYTES
from packed_q6.reference import bf16_round, decode_reference, fixture, projection_reference


@pytest.mark.parametrize("outputs,inputs", [(1, 1), (65, 257), (320, 2560), (640, 2560)])
def test_lossless_repack_preserves_every_byte_and_accounts_for_padding(outputs, inputs):
    raw = fixture(outputs, inputs)
    packed = pack_q6(raw, inputs)
    np.testing.assert_array_equal(unpack_q6(packed), raw)
    assert packed.data.flags.c_contiguous and packed.data.ndim == 1
    assert packed.storage()["original_bytes"] == raw.nbytes
    if outputs in (320, 640):
        assert packed.storage()["padding_bytes"] <= 384
        assert packed.storage()["padding_ratio"] < 0.0004


def test_decoder_handles_all_code_and_signed_scale_values_at_bf16_boundary():
    raw = fixture(64, 256)
    # This adds FP16 extrema and subnormals to random byte coverage.
    d = np.array([0, -0.0, 65504, -65504, 2**-24, -(2**-24), 2**-14, 1], dtype="<f2")
    raw[:8, 0, 208:] = d.view(np.uint8).reshape(8, 2)
    raw[..., 192:208] = np.arange(1024, dtype=np.uint16).astype(np.uint8).reshape(64, 1, 16)
    packed = pack_q6(raw, 256)
    expected = decode_reference(raw, 256).astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_array_equal(bf16_round(decode_reference(raw, 256)), expected)
    actual = np.asarray(
        jax.jit(decode_tile)(jnp.asarray(packed.data), jnp.asarray(packed.scales[:128])), np.float32
    )
    np.testing.assert_array_equal(actual, expected)


def test_every_finite_fp16_scale_is_expanded_exactly_without_fp16_arithmetic():
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & 0x7C00) != 0x7C00]
    expected = bits.view(np.float16).astype(np.float32)
    actual = np.asarray(jax.jit(decode_fp16_scale)(jnp.asarray(bits, jnp.uint32)))
    # Compare bits to check signed zero as well as all finite values.
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize(
    "batch,outputs,inputs",
    [
        (1, 1, 1),
        (2, 67, 255),
        (4, 128, 257),
        (8, 64, 512),
        (9, 65, 263),
        (1, 320, 2560),
        (8, 640, 2560),
    ],
)
def test_interpreted_projection_matches_independent_fp64_dot(batch, outputs, inputs):
    raw = fixture(outputs, inputs)
    packed = pack_q6(raw, inputs)
    x = np.random.default_rng(721).normal(size=(batch, inputs)).astype(np.float32)
    if batch > 1:
        x[0] = 0
    actual = np.asarray(
        q6_matmul(
            jnp.asarray(x, jnp.bfloat16),
            jnp.asarray(packed.data),
            jnp.asarray(packed.scales),
            output_features=outputs,
            interpret=True,
        )
    )
    expected = projection_reference(x, raw)
    assert actual.dtype == np.float32 and actual.shape == (batch, outputs)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)


def test_accumulator_is_reset_between_m_and_n_tiles():
    raw = fixture(128, 512)
    raw[64:] = raw[:64]
    packed = pack_q6(raw, 512)
    x = np.ones((16, 512), np.float32)
    x[8:] = -1
    result = np.asarray(
        q6_matmul(
            jnp.asarray(x, jnp.bfloat16),
            jnp.asarray(packed.data),
            jnp.asarray(packed.scales),
            output_features=128,
            interpret=True,
        )
    )
    np.testing.assert_array_equal(result[:, :64], result[:, 64:])
    np.testing.assert_array_equal(result[:8], -result[8:])


@pytest.mark.parametrize("outputs,batch", [(320, 1), (320, 8), (640, 2), (640, 8), (1280, 4)])
def test_exports_to_v5e_mosaic_without_a_tpu_device(outputs, batch):
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("core",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU v5 lite",
            num_cores=1,
            platform="tpu",
        ),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        function = jax.jit(lambda x, p, s: q6_matmul(x, p, s, output_features=outputs))
        scale_size = (
            (outputs // 64 * 10 * SCALE_TILE_BYTES + SCALE_FETCH_BYTES - 1) // SCALE_FETCH_BYTES
        ) * SCALE_FETCH_BYTES
        lowered = export.export(function, platforms=("tpu",))(
            jax.ShapeDtypeStruct((batch, 2560), jnp.bfloat16),
            jax.ShapeDtypeStruct((outputs // 64 * 10 * TILE_BYTES,), jnp.uint8),
            jax.ShapeDtypeStruct((scale_size,), jnp.uint8),
        )
    text = lowered.mlir_module()
    assert "tpu_custom_call" in text
    # Export verifies the Pallas -> Mosaic path, not hardware compilation or speed.
    assert lowered.platforms == ("tpu",)


def test_oracle_agrees_with_optional_independent_gguf_library():
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize

    raw = fixture(64, 512)
    expected = dequantize(raw.reshape(64, -1), gguf.GGMLQuantizationType.Q6_K)
    np.testing.assert_array_equal(decode_reference(raw, 512), expected)


def test_xla_baseline_matches_independent_oracle():
    from packed_q6.baselines import decode_rows, xla_q6_matmul

    raw = fixture(65, 263)
    decoded = np.asarray(jax.jit(lambda r: decode_rows(r, 263))(jnp.asarray(raw)))
    np.testing.assert_array_equal(decoded, decode_reference(raw, 263))
    x = np.random.default_rng(26).normal(size=(4, 263)).astype(np.float32)
    result = np.asarray(
        xla_q6_matmul(
            jnp.asarray(x, jnp.bfloat16), jnp.asarray(raw.reshape(-1)), output_features=65
        )
    )
    np.testing.assert_allclose(result, projection_reference(x, raw), rtol=2e-5, atol=2e-4)


def test_xla_baseline_matches_application_decoder():
    from packed_q6.baselines import decode_rows

    # Optional in the standalone bundle; required/present in this repository.
    runtime = pytest.importorskip("glm53_offload.gguf_quant")
    raw = fixture(65, 263)
    decoded = np.asarray(jax.jit(lambda r: decode_rows(r, 263))(jnp.asarray(raw)))
    np.testing.assert_array_equal(decoded, runtime.dequantize_rows(raw, "Q6_K", 263))


@pytest.mark.parametrize("scale", [np.inf, -np.inf, np.nan])
def test_invalid_block_scale_rejected_before_device_upload(scale):
    raw = fixture(1, 256)
    raw[0, 0, 208:] = np.array([scale], dtype="<f2").view(np.uint8)
    with pytest.raises(ValueError, match="finite"):
        pack_q6(raw, 256)


def test_invalid_layout_or_projection_contract_rejected():
    with pytest.raises(TypeError, match="uint8"):
        pack_q6(np.zeros((64, 1, 210)), 256)
    with pytest.raises(ValueError, match="Expected"):
        pack_q6(np.zeros((64, 1, 211), np.uint8), 256)
    raw = jnp.zeros((TILE_BYTES,), jnp.uint8)
    scales = jnp.zeros((SCALE_FETCH_BYTES,), jnp.uint8)
    with pytest.raises(ValueError, match="BF16"):
        q6_matmul(jnp.ones((1, 256), jnp.float32), raw, scales, output_features=64, interpret=True)
    with pytest.raises(ValueError, match="multiple of 8"):
        q6_matmul(
            jnp.ones((1, 256), jnp.bfloat16),
            raw,
            scales,
            output_features=64,
            batch_tile=1,
            interpret=True,
        )
    with pytest.raises(ValueError, match="dtype/shape"):
        q6_matmul(jnp.ones((1, 257), jnp.bfloat16), raw, scales, output_features=64, interpret=True)
