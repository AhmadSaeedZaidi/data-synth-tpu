"""Independent checks for the smaller EP4/TP2 candidates and calibration oracle."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from qwen38_tpu import expert_calibration as calibration
from qwen38_tpu.compute import pack_q8
from qwen38_tpu.experts import expert_parallel
from qwen38_tpu.parallel import expert_mesh, put_expert_bytes


def test_tp2_single_expert_cannot_overwrite_serial_calibration_reference(monkeypatch):
    """Exercise the actual calibration loop with a deliberately wrong TP2 result."""
    mesh = SimpleNamespace(size=8, devices=np.array([SimpleNamespace(platform="tpu")] * 8))
    c = SimpleNamespace(hidden=256, intermediate=64, experts=40, top_k=2)
    monkeypatch.setattr(calibration, "NamedSharding", lambda *a: None)
    monkeypatch.setattr(calibration, "expert_mesh", lambda *a: mesh)
    monkeypatch.setattr(calibration.jax, "device_put", lambda value, *a: value)
    monkeypatch.setattr(calibration.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        calibration,
        "_place",
        lambda *a: {name: SimpleNamespace(delete=lambda: None) for name in ("gate", "up", "down")},
    )

    def expert(*a, group_size, tensor_parallel, **kw):
        class Compiled:
            def __call__(self, gate, up, down, x, ids, weights):
                return np.ones_like(x, dtype=np.float32) * (2 if tensor_parallel == 2 else 1)

            def memory_analysis(self):
                return SimpleNamespace(temp_size_in_bytes=1024)

        return SimpleNamespace(lower=lambda *a: SimpleNamespace(compile=Compiled))

    monkeypatch.setattr(calibration, "expert_parallel", expert)
    result = calibration.calibrate_experts(mesh, c, rows=[1, 4], decode_rows=[1])
    rows = result["measurements"]
    assert all(r["accuracy_pass"] for r in rows if r["candidate"] == "serial_ep8")
    for candidate in ("sliced_ep4_tp2", "paired_ep4_tp2", "grouped_ep4_tp2"):
        values = [r for r in rows if r["candidate"] == candidate]
        assert len(values) == 8
        assert all(not r["accuracy_pass"] and r["max_abs_error"] == 1 for r in values)
    assert result["selected"]["tensor_parallel"] == 1


@pytest.mark.parametrize("group", [1, 2, 4])
@pytest.mark.parametrize("policy", ["bf16", "int8_q8"])
def test_idle_owners_and_padded_slots_do_not_leak_nan(group, policy):
    mesh = Mesh(np.asarray(jax.devices()), ("expert",))
    moe_mesh = expert_mesh(mesh, 2)
    hidden, intermediate, count = 32, 64, 16
    packed = []
    for name in ("gate", "up", "down"):
        width, rows = (intermediate, hidden) if name == "down" else (hidden, intermediate)
        raw = pack_q8(np.full((count * rows, width), 0.02, np.float32))
        raw = raw.reshape(count, rows, width // 32, 34)
        raw[..., :2] = np.array([np.nan], "<f2").view(np.uint8)
        raw[3, ..., :2] = np.array([0.02 / 127], "<f2").view(np.uint8)
        packed.append(put_expert_bytes(raw, name, "Q8_0", hidden, intermediate, moe_mesh))
    replica = NamedSharding(mesh, P())
    x, ids, weights = (
        jax.device_put(v, replica)
        for v in (
            np.full((2, hidden), 0.2, np.dtype(jnp.bfloat16)),
            np.full((2, 2), 3, np.int32),  # duplicate routes sum; one finished row
            np.array([[0.25, 0.75], [0, 0]], np.dtype(jnp.bfloat16)),
        )
    )
    fn = expert_parallel(
        mesh,
        gate_type="Q8_0",
        up_type="Q8_0",
        down_type="Q8_0",
        hidden=hidden,
        intermediate=intermediate,
        compute=policy,
        group_size=group,
        tensor_parallel=2,
    )
    result = np.asarray(fn(*packed, x, ids, weights), np.float32)
    assert np.isfinite(result).all() and np.any(result[0] != 0)
    np.testing.assert_array_equal(result[1], 0)
    np.testing.assert_array_equal(np.asarray(fn(*packed, x, ids, weights * 0), np.float32), 0)
