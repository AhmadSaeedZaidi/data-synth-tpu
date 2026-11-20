"""Explicit TorchTPU initialization; never falls back to CPU or legacy XLA."""

import importlib.metadata


def configure_tpu():
    import torch
    import torch_tpu  # noqa: F401 - registers the native backend

    if not hasattr(torch, "tpu"):
        raise RuntimeError("Native TorchTPU registration is unavailable")
    # Must precede the first compile. Retain rounding at every explicit cast.
    torch.tpu.allow_excess_precision = False
    # FP32 router and recurrent math must not silently become single-pass BF16.
    # Native BF16 expert dots still use their BF16 operands and FP32 accumulator.
    torch.set_float32_matmul_precision("highest")
    return dict(
        torch=torch.__version__,
        torch_tpu=importlib.metadata.version("torch-tpu"),
        libtpu=importlib.metadata.version("libtpu"),
        precision=torch.get_float32_matmul_precision(),
        allow_excess_precision=torch.tpu.allow_excess_precision,
        device="tpu",
    )


def make_groups(plan):
    """Call on every rank, in identical order, after tpu_dist initialization."""
    import torch.distributed as dist

    if dist.get_world_size() != plan.ep * plan.tp:
        raise ValueError("Distributed world size differs from EP/TP placement")
    rank = dist.get_rank()
    tensor_group = expert_group = None
    for ep in range(plan.ep):
        ranks = [ep * plan.tp + tp for tp in range(plan.tp)]
        group = dist.new_group(ranks)
        if rank in ranks:
            tensor_group = group
    for tp in range(plan.tp):
        ranks = [ep * plan.tp + tp for ep in range(plan.ep)]
        group = dist.new_group(ranks)
        if rank in ranks:
            expert_group = group
    return rank // plan.tp, tensor_group, expert_group
