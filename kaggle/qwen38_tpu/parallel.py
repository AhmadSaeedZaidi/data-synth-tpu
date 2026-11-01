"""Packed, non-replicating expert/tensor placement on the same TPU slice.

Each EP group owns a disjoint set of experts. Its TP chips own disjoint
intermediate channels: rows of gate/up, matching input blocks of down.
The physical [rank-major expert, bytes] layout avoids tiny TPU tile axes.
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .quant import block_layout


def ordered_devices(devices):
    """Place the frequently communicating TP pair on the inner topology axis."""
    devices = list(devices)
    if devices and all(d.platform == "tpu" for d in devices):
        from jax.experimental import mesh_utils

        return mesh_utils.create_device_mesh(
            (len(devices) // 2, 2), devices, allow_split_physical_axes=True
        ).reshape(-1)
    return np.asarray(devices)


def expert_mesh(mesh, tensor_parallel=1):
    if type(tensor_parallel) is not int or tensor_parallel < 1 or mesh.size % tensor_parallel:
        raise ValueError("Tensor parallel size must divide the device count")
    if tensor_parallel == 1:
        return Mesh(np.asarray(mesh.devices).reshape(-1), ("expert",))
    return Mesh(np.asarray(mesh.devices).reshape(-1, tensor_parallel), ("expert", "tensor"))


def packed_spec(mesh):
    return P(("expert", "tensor")) if "tensor" in mesh.axis_names else P("expert")


def split_expert_bytes(raw, kind, qtype, hidden, intermediate, tensor_parallel):
    """Split one EP group's raw GGUF bytes; no requantization or replication.

    Returns [tensor_rank, local_expert, bytes]. Down splits on complete
    quantization blocks, never in the middle of a scale or a packed block.
    """
    if kind not in {"gate", "up", "down"}:
        raise ValueError("Unknown expert projection")
    block, size = block_layout(qtype)
    if intermediate % tensor_parallel:
        raise ValueError("Intermediate channels must divide tensor parallel size")
    local = intermediate // tensor_parallel
    count = len(raw)
    if kind == "down":
        if local % block:
            raise ValueError("Tensor slices must contain whole down quantization blocks")
        rows = np.asarray(raw).reshape(count, hidden, tensor_parallel, local // block, size)
        return rows.transpose(2, 0, 1, 3, 4).reshape(tensor_parallel, count, -1).copy()
    if hidden % block:
        raise ValueError("Hidden channels must contain whole quantization blocks")
    rows = np.asarray(raw).reshape(count, tensor_parallel, local, hidden // block, size)
    return rows.transpose(1, 0, 2, 3, 4).reshape(tensor_parallel, count, -1).copy()


def put_expert_bytes(raw, kind, qtype, hidden, intermediate, mesh):
    """Small fixture/calibration uploader; production streams one EP group."""
    ep, tp = mesh.shape["expert"], mesh.shape.get("tensor", 1)
    if len(raw) % ep:
        raise ValueError("Expert count must divide expert parallel size")
    per_group = len(raw) // ep
    chunks = []
    for group in range(ep):
        chunks.extend(
            split_expert_bytes(
                raw[group * per_group : (group + 1) * per_group],
                kind,
                qtype,
                hidden,
                intermediate,
                tp,
            )
        )
    arrays = [
        jax.device_put(part, device) for part, device in zip(chunks, mesh.devices.flat, strict=True)
    ]
    return jax.make_array_from_single_device_arrays(
        (len(raw) * tp, arrays[0].shape[1]), NamedSharding(mesh, packed_spec(mesh)), arrays
    )
