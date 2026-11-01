"""Native image ViT and Qwen patch/merge/three-axis position protocol."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .config import VisionConfig
from .precision import compute_cast


def patch_order(height, width, merge):
    if height % merge or width % merge:
        raise ValueError("Image grid must divide into merge groups")
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    def reorder(x):
        return (
            x.reshape(height // merge, merge, width // merge, merge)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )

    return np.stack((reorder(rows), reorder(cols)), axis=-1)


def preprocess(image, c=VisionConfig(), *, min_pixels=65536, max_pixels=589824):
    image = image.convert("RGB")
    width, height = image.size
    factor = c.patch * c.merge
    if max(width, height) / min(width, height) > 200:
        raise ValueError("Image aspect ratio exceeds Qwen processor limit")
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h, w = (
            max(factor, math.floor(height / scale / factor) * factor),
            max(factor, math.floor(width / scale / factor) * factor),
        )
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h, w = (
            math.ceil(height * scale / factor) * factor,
            math.ceil(width * scale / factor) * factor,
        )
    if (w, h) != image.size:
        image = image.resize((w, h), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, np.float32).transpose(2, 0, 1) / 127.5 - 1
    gh, gw = h // c.patch, w // c.patch
    patches = pixels.reshape(3, gh // c.merge, c.merge, c.patch, gw // c.merge, c.merge, c.patch)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    patches = np.repeat(patches[..., None, :, :], c.temporal, axis=-3)
    return patches.reshape(gh * gw, -1), (1, gh, gw)


def layer_norm(x, w):
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, -1, keepdims=True)
    norm = (xf - mean) * jax.lax.rsqrt(jnp.mean((xf - mean) ** 2, -1, keepdims=True) + 1e-6)
    return compute_cast(norm * w["weight"] + w["bias"], x.dtype)


def affine(x, w):
    # F.linear/Conv includes bias before the BF16 output rounding.
    return compute_cast(
        jnp.matmul(x, w["weight"].T, preferred_element_type=jnp.float32)
        + w["bias"].astype(jnp.float32),
        x.dtype,
    )


def vision_rope(x, coordinates, c):
    head_dim = c.hidden // c.heads
    spatial = head_dim // 2
    inv = c.theta ** (-jnp.arange(0, spatial, 2, dtype=jnp.float32) / spatial)
    angle = (coordinates[..., None] * inv).reshape(x.shape[0], head_dim // 2)
    a, b = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    cs, sn = jnp.cos(angle)[:, None, :], jnp.sin(angle)[:, None, :]
    return compute_cast(jnp.concatenate((a * cs - b * sn, b * cs + a * sn), -1), x.dtype)


def block(x, w, coordinates, c):
    normalized = layer_norm(x, w["ln1"])
    qkv = affine(normalized, w["qkv"]).reshape(x.shape[0], 3, c.heads, c.hidden // c.heads)
    q, k, v = (qkv[:, i] for i in range(3))
    q, k = vision_rope(q, coordinates, c), vision_rope(k, coordinates, c)
    raw_score = compute_cast(
        jnp.einsum("nhd,mhd->hnm", q, k, preferred_element_type=jnp.float32), x.dtype
    )
    scores = compute_cast(
        raw_score.astype(jnp.float32) / (c.hidden // c.heads) ** 0.5, x.dtype
    ).astype(jnp.float32)
    attention = compute_cast(
        jnp.einsum(
            "hnm,mhd->nhd",
            compute_cast(jax.nn.softmax(scores, -1), x.dtype),
            v,
            preferred_element_type=jnp.float32,
        ).reshape(x.shape),
        x.dtype,
    )
    x = compute_cast(x + affine(attention, w["out"]), x.dtype)
    expanded = affine(layer_norm(x, w["ln2"]), w["up"])
    ffn = compute_cast(jax.nn.gelu(expanded.astype(jnp.float32), approximate=True), x.dtype)
    return compute_cast(x + affine(ffn, w["down"]), x.dtype)


def interpolate_positions(table, gh, gw, c):
    coordinates = patch_order(gh, gw, c.merge)
    # align_corners=True, as used by Qwen4ExpVisionModel.
    h = coordinates[:, 0] * (c.grid_side - 1) / max(gh - 1, 1)
    w = coordinates[:, 1] * (c.grid_side - 1) / max(gw - 1, 1)
    h0, w0 = h.astype(np.int32), w.astype(np.int32)
    h1, w1 = np.minimum(h0 + 1, c.grid_side - 1), np.minimum(w0 + 1, c.grid_side - 1)
    dh, dw = jnp.asarray(h - h0), jnp.asarray(w - w0)
    return (1 - dh)[:, None] * (
        (1 - dw)[:, None] * table[h0 * c.grid_side + w0]
        + dw[:, None] * table[h0 * c.grid_side + w1]
    ) + dh[:, None] * (
        (1 - dw)[:, None] * table[h1 * c.grid_side + w0]
        + dw[:, None] * table[h1 * c.grid_side + w1]
    )


class VisionEncoder:
    def __init__(self, weights, config=VisionConfig()):
        self.weights, self.config = weights, config
        self.step = jax.jit(lambda x, w, pos: block(x, w, pos, config))
        self.merge = jax.jit(self._merge)

    def _merge(self, x):
        c, w = self.config, self.weights
        x = layer_norm(x, w["post_norm"]).reshape(-1, c.hidden * c.merge**2)
        expanded = affine(x, w["merge_up"])
        activated = compute_cast(
            jax.nn.gelu(expanded.astype(jnp.float32), approximate=False), x.dtype
        )
        return affine(activated, w["merge_down"])

    def encode_patches(self, patches, grid):
        c, w = self.config, self.weights
        temporal, gh, gw = grid
        if temporal != 1:
            raise ValueError(
                "Initial trial supports still images; video requires temporal segmentation"
            )
        coordinates = jnp.asarray(patch_order(gh, gw, c.merge))
        x = affine(jnp.asarray(patches, w["patch"]["weight"].dtype), w["patch"])
        x = x + interpolate_positions(w["position"], gh, gw, c).astype(x.dtype)
        for weights in w["blocks"]:
            x = self.step(x, weights, coordinates)
        return self.merge(x)


def text_positions(ids, grids, c):
    """Positions for text plus image spans; image placeholder IDs also feed PLE."""
    ids = np.asarray(ids, np.int64)
    out = np.empty((len(ids), 3), np.int32)
    cursor, pos, image = 0, 0, 0
    while cursor < len(ids):
        if ids[cursor] != c.image_token:
            out[cursor] = pos
            cursor, pos = cursor + 1, pos + 1
            continue
        if image >= len(grids):
            raise ValueError("Image placeholder without corresponding image")
        t, gh, gw = grids[image]
        if t != 1:
            raise ValueError("Only still images supported")
        mh, mw = gh // 2, gw // 2
        count = mh * mw
        if (
            not np.all(ids[cursor : cursor + count] == c.image_token)
            or len(ids[cursor : cursor + count]) != count
        ):
            raise ValueError("Image placeholder count differs from merged patch count")
        rows, cols = np.meshgrid(np.arange(mh), np.arange(mw), indexing="ij")
        out[cursor : cursor + count] = (
            np.stack((np.zeros(count), rows.flatten(), cols.flatten()), -1) + pos
        )
        cursor, pos, image = cursor + count, pos + max(mh, mw), image + 1
    if image != len(grids):
        raise ValueError("Unused image grid")
    return out, pos


def load_vision(checkpoint, mesh, c=VisionConfig()):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    replica = NamedSharding(mesh, P())

    def pair(prefix, *, norm=False):
        matrix = checkpoint.dense(prefix + ".weight")
        bias = checkpoint.dense(prefix + ".bias")
        layout = replica if matrix.ndim != 2 or norm else NamedSharding(mesh, P("expert", None))
        dtype = jnp.float32 if norm else jnp.bfloat16
        return dict(
            weight=jax.device_put(matrix, layout).astype(dtype),
            bias=jax.device_put(bias, replica).astype(dtype),
        )

    first = checkpoint.dense("v.patch_embd.weight")
    second = checkpoint.dense("v.patch_embd.weight.1")
    patch = np.stack((first, second), axis=2).reshape(c.hidden, -1)
    return dict(
        patch=dict(
            weight=jax.device_put(patch, NamedSharding(mesh, P("expert", None))).astype(
                jnp.bfloat16
            ),
            bias=jax.device_put(checkpoint.dense("v.patch_embd.bias"), replica).astype(
                jnp.bfloat16
            ),
        ),
        position=jax.device_put(checkpoint.dense("v.position_embd.weight"), replica),
        blocks=[
            {
                target: pair(f"v.blk.{i}.{source}", norm=target.startswith("ln"))
                for target, source in (
                    ("ln1", "ln1"),
                    ("ln2", "ln2"),
                    ("qkv", "attn_qkv"),
                    ("out", "attn_out"),
                    ("up", "ffn_up"),
                    ("down", "ffn_down"),
                )
            }
            for i in range(c.layers)
        ],
        post_norm=pair("v.post_ln", norm=True),
        merge_up=pair("mm.0"),
        merge_down=pair("mm.2"),
    )
