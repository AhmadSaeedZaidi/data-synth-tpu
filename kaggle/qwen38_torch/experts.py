"""Static grouped experts with explicit EP/TP ownership and reduction order."""

from dataclasses import dataclass

import numpy as np
import torch

from .ops import cast, decode_rows, matmul_f32, silu


@dataclass(frozen=True)
class Placement:
    experts: int
    hidden: int
    intermediate: int
    ep: int = 4
    tp: int = 2
    group_size: int = 4
    gate_type: str = "Q6_K"
    up_type: str = "Q6_K"
    down_type: str = "Q8_0"

    def __post_init__(self):
        if any(
            type(x) is not int or x < 1
            for x in (
                self.experts,
                self.hidden,
                self.intermediate,
                self.ep,
                self.tp,
                self.group_size,
            )
        ):
            raise ValueError("Placement sizes must be positive integers")
        if self.experts % self.ep or self.intermediate % self.tp:
            raise ValueError("Expert and intermediate dimensions must divide the mesh")
        for kind, qtype in (
            ("gate", self.gate_type),
            ("up", self.up_type),
            ("down", self.down_type),
        ):
            if qtype not in {"Q6_K", "Q8_0"}:
                raise ValueError("Unsupported per-layer expert quantization")
            block = 256 if qtype == "Q6_K" else 32
            width = self.intermediate // self.tp if kind == "down" else self.hidden
            if width % block:
                raise ValueError("Expert slices must split on complete quantization blocks")

    @property
    def local_experts(self):
        return self.experts // self.ep

    @property
    def local_width(self):
        return self.intermediate // self.tp


def shard_packed(gate, up, down, plan, ep_rank, tp_rank):
    """Host loader for an EP-owned group. Original bytes are never requantized.

    Production loading must range-read only this owner's experts. This helper
    accepts a small complete fixture for comparison, not a second HBM copy.
    """
    if not 0 <= ep_rank < plan.ep or not 0 <= tp_rank < plan.tp:
        raise ValueError("Rank outside placement")
    count, local = plan.local_experts, plan.local_width
    first = ep_rank * count
    result = []
    for kind, source in (("gate", gate), ("up", up), ("down", down)):
        a = np.asarray(source)
        qtype = getattr(plan, kind + "_type")
        block, size = (256, 210) if qtype == "Q6_K" else (32, 34)
        expected = (
            (plan.experts, plan.hidden, plan.intermediate // block, size)
            if kind == "down"
            else (plan.experts, plan.intermediate, plan.hidden // block, size)
        )
        if a.dtype != np.uint8 or a.shape != expected:
            raise ValueError("Packed expert table does not match placement")
        owned = a[first : first + count]
        if kind == "down":
            owned = owned[:, :, tp_rank * (local // block) : (tp_rank + 1) * (local // block)]
        else:
            owned = owned[:, tp_rank * local : (tp_rank + 1) * local]
        result.append(np.ascontiguousarray(owned))
    return tuple(result)


def active_expert_slots(ids, weights, first, count):
    """Same sorted slots on every TP partner, even if no local expert is used.

    Capacity is an upper bound on all nonzero experts, never an overflow cap.
    No tensor scalar reads, dynamic nonzero shapes or per-request host routing.
    """
    owned = first + torch.arange(count, device=ids.device, dtype=ids.dtype)
    routes = torch.where(ids[None] == owned[:, None, None], weights[None], 0).float().sum(-1)
    used = torch.any(routes != 0, dim=-1)
    capacity = min(ids.numel(), count)
    # Distinct keys retain ascending expert order; dummy slots sort last.
    key = torch.where(used, owned, first + count + owned)
    selected = torch.argsort(key, stable=True)[:capacity]
    return selected, routes[selected]


def expert_partials(gate, up, down, x, ids, weights, plan, ep_rank, sum_tensor):
    """Local packed selection + dots; TP sums happen BEFORE BF16 rounding."""
    selected, routed = active_expert_slots(ids, weights, ep_rank * plan.local_experts, len(gate))
    output = torch.zeros_like(x, dtype=torch.float32)
    for start in range(0, len(selected), plan.group_size):
        index = selected[start : start + plan.group_size]
        route = routed[start : start + plan.group_size]
        gw = decode_rows(gate[index], plan.gate_type, plan.hidden, dtype=x.dtype)
        uw = decode_rows(up[index], plan.up_type, plan.hidden, dtype=x.dtype)
        dw = decode_rows(down[index], plan.down_type, plan.local_width, dtype=x.dtype)
        tokens = x[None].expand(len(index), -1, -1)
        joined = cast(matmul_f32(tokens, torch.cat((gw, uw), dim=1)), x.dtype)
        g, u = joined.chunk(2, dim=-1)
        activation = cast(silu(g) * u, x.dtype)
        values = sum_tensor(matmul_f32(activation, dw))
        values = cast(values, x.dtype).float()
        for slot in range(len(index)):
            # Inactive routes contribute zero even if dummy weights are NaN.
            output = output + torch.where(
                route[slot, :, None] != 0, values[slot] * route[slot, :, None], 0
            )
    return output


def distributed_experts(plan, ep_rank, tensor_group, expert_group):
    """Construct an SPMD callable; every rank must call groups in the same order.

    Tensor groups join intermediate slices. Expert groups contain a fixed tensor
    coordinate, so EP reduction does not count already replicated TP results twice.
    Replicated activations/routes are an input contract; only local weight shards
    are passed. No collective gathers expert weights.
    """
    from torch.distributed import _functional_collectives as collectives

    def run(gate, up, down, x, ids, weights):
        def tp_sum(value):
            if plan.tp == 1:
                return value
            return collectives.all_reduce(value, "sum", tensor_group)

        output = expert_partials(gate, up, down, x, ids, weights, plan, ep_rank, tp_sum)
        if plan.ep > 1:
            output = collectives.all_reduce(output, "sum", expert_group)
        return cast(output, x.dtype)

    return run
