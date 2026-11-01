"""Numerical tower, sharding, padding isolation and image preprocessing checks."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from glm53 import vision
from glm53_offload import run as worker
from jax.sharding import Mesh
from PIL import Image


@pytest.fixture
def reference():
    path = Path(worker.__file__).parent / "data/vision_reference.npz"
    with np.load(path, allow_pickle=False) as data:
        yield data


def parameters(data):
    p = vision.from_state_dict(lambda k: data["weight/" + k], 2, 32)
    p["n_heads"] = 8
    return p


def test_unsharded_vision_agrees_with_independent_transformers(reference):
    p = parameters(reference)
    grids = tuple(map(tuple, reference["grids"].tolist()))
    result = vision.forward(p, reference["patches"], grids)
    np.testing.assert_allclose(result, reference["expected"], rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sharded_vision_preserves_images_and_ignores_padding(reference, dtype):
    mesh = Mesh(np.array(jax.devices()), ("x",))
    p = vision.to_sharded(parameters(reference), mesh, dtype=dtype)
    forward = vision.make_forward_sharded(p, mesh, dtype=dtype, q_chunk=16)
    grids = tuple(map(tuple, reference["grids"].tolist()))
    result = np.asarray(forward(reference["patches"], grids))
    expected = reference["expected" if dtype == jnp.float32 else "expected_bf16"]
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-4
    np.testing.assert_allclose(result, expected, rtol=0.04, atol=tolerance)
    # Another image or dummy padding must not affect the first image's tokens.
    padded = reference["patches"].copy()
    padded[24:] = 999.0
    changed = np.asarray(forward(padded, grids))
    np.testing.assert_allclose(changed[:6], result[:6], rtol=2e-4, atol=2e-6)
    alone = forward(reference["patches"][:24], (grids[0],))
    np.testing.assert_allclose(alone, result[:6], rtol=2e-4, atol=2e-6)


def test_preprocess_patch_order_temporal_duplication_and_black_padding():
    pixels = np.zeros((28, 55, 3), np.uint8)
    pixels[:, :28] = (255, 0, 0)
    pixels[:, 28:] = (0, 0, 255)
    patches, grid = vision.preprocess(Image.fromarray(pixels), min_tokens=1, max_tokens=16)
    assert grid == (1, 2, 4)
    patches = patches.reshape(8, 3, 2, 14, 14)
    np.testing.assert_array_equal(patches[:, :, 0], patches[:, :, 1])
    red = (np.array([1, 0, 0]) - vision.MEAN) / vision.STD
    blue = (np.array([0, 0, 1]) - vision.MEAN) / vision.STD
    black = -vision.MEAN / vision.STD
    np.testing.assert_allclose(patches[:4, :, 0, 0, 0], np.broadcast_to(red, (4, 3)))
    np.testing.assert_allclose(patches[4:, :, 0, 0, 0], np.broadcast_to(blue, (4, 3)))
    np.testing.assert_allclose(patches[[5, 7], :, 0, 0, -1], np.broadcast_to(black, (2, 3)))
    assert vision.n_tokens(grid) == 2


def test_image_tower_shape_is_checked_before_embedding_injection():
    from io import BytesIO

    from test_run import Tokenizer

    image = BytesIO()
    Image.new("RGB", (56, 56), "red").save(image, format="PNG")
    chat = worker.VisionChat(Tokenizer(), lambda patches, grids: np.zeros((1, 4)), hidden_size=4)
    with pytest.raises(ValueError, match="vision encoder shape"):
        chat.image(image.getvalue())
