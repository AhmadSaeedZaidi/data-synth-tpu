"""Small four-process Gloo test of actual EP/TP collectives; not TPU timing."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .experts import Placement, distributed_experts, expert_partials, shard_packed
from .runtime import make_groups


def data():
    # Local synthetic inputs; no model downloads or external processes.
    from packed_q6.reference import fixture

    rng = np.random.default_rng(914)
    gate = np.stack([fixture(64, 256, seed=30 + i) for i in range(8)])
    up = np.stack([fixture(64, 256, seed=60 + i) for i in range(8)])
    down = rng.integers(0, 256, (8, 256, 2, 34), dtype=np.uint8)
    down[..., :2] = np.asarray([0.003], dtype="<f2").view(np.uint8)
    x = torch.from_numpy(rng.normal(0, 0.01, (4, 256)).astype(np.float32)).to(torch.bfloat16)
    ids = torch.tensor([[0, 4], [7, 1], [2, 2], [5, 6]])
    weights = torch.tensor([[0.4, 0.6], [0.7, 0.3], [0, 0], [0.5, 0.5]], dtype=torch.bfloat16)
    return gate, up, down, x, ids, weights


def worker(rank, rendezvous, output, device="cpu", tp=2):
    torch.set_num_threads(1)
    if device == "tpu":
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="8")
        from .runtime import configure_tpu

        configure_tpu()
    world = 8 if device == "tpu" else 4
    dist.init_process_group(
        "tpu_dist" if device == "tpu" else "gloo",
        init_method=rendezvous,
        world_size=world,
        rank=rank,
        timeout=timedelta(seconds=75),
    )
    try:
        plan = Placement(8, 256, 64, ep=world // tp, tp=tp, group_size=2)
        ep, tg, eg = make_groups(plan)
        gate, up, down, x, ids, weights = data()
        local = [
            torch.from_numpy(a).to(device)
            for a in shard_packed(gate, up, down, plan, ep, rank % tp)
        ]
        run = distributed_experts(plan, ep, tg, eg)
        # Compare against an unpartitioned execution: TP rounding too early or
        # a second tensor-axis reduction changes this result significantly.
        serial = Placement(8, 256, 64, ep=1, tp=1, group_size=2)
        reference = expert_partials(
            *(torch.from_numpy(a) for a in (gate, up, down)),
            x,
            ids,
            weights,
            serial,
            0,
            lambda v: v,
        ).to(x.dtype)
        actual = (
            run(*local, x.to(device), ids.to(device), weights.to(device)).float().cpu().to(x.dtype)
        )
        torch.testing.assert_close(actual, reference, rtol=0.012, atol=0.00008)
        assert torch.count_nonzero(actual[2]) == 0
        if rank == 0:
            Path(output).write_text(
                json.dumps(
                    dict(
                        status="passed" if device == "tpu" else "PASS",
                        world_size=world,
                        ep=plan.ep,
                        tp=tp,
                        device=device,
                        scope="small_eight_chip_correctness_not_throughput",
                        dtype="bfloat16",
                        max_error=float((actual.float() - reference.float()).abs().max()),
                    ),
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "tpu"], default="cpu")
    parser.add_argument("--tp", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendezvous = "file://" + str((args.output.parent / "gloo-rendezvous").resolve())
    if args.device == "tpu":
        from .runtime import configure_tpu

        configure_tpu()
        torch.tpu.distributed.set_tpu_launch_env()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29671")
        rendezvous = "env://"
    mp.spawn(
        worker,
        args=(rendezvous, str(args.output), args.device, args.tp),
        nprocs=8 if args.device == "tpu" else 4,
        join=True,
    )


if __name__ == "__main__":
    main()
